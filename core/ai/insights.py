"""Produce validated insights for one run: AI if it works and passes the guardrails, else the template.

    results.json -> facts sheet -> prompt -> provider -> parse -> validate
                                      |                               |
                                      +---- retry once with the errors +
                                                          |
                                            still bad / unavailable
                                                          v
                                          template provider -> validate

The package returned (and saved as insights.json) says which path produced the
text ("mode": "ai" or "fallback"), why, what was rejected on the way, and which
facts the text cites. The page shows all of that, so nobody mistakes a template
for AI output or the reverse.

Failures here never fail a pipeline run: the worst case is mode "none" with a reason.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone

from pydantic import ValidationError

from core.ai.facts import Fact, build_facts, facts_by_id
from core.ai.fallback import TemplateProvider
from core.ai.guardrails import validate
from core.ai.prompt import build_prompt
from core.ai.schema import InsightDoc, json_schema
from core.ai.snwolley_adapter import Provider, ProviderUnavailable
from core.use_case import UseCase

logger = logging.getLogger(__name__)
MAX_ERRORS_KEPT = 8
NOT_CONFIGURED = "No AI provider is configured (set SNWOLLEY_API_KEY and SNWOLLEY_BASE_URL in .env)."


def parse_reply(text: str) -> InsightDoc:
    """Accept a bare JSON object, or one wrapped in a code fence or a sentence. Raises ValueError with readable problems."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty reply")
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("reply is not a JSON object")
    try:
        data = json.loads(body[start: end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"reply is not valid JSON ({exc.msg})") from exc
    try:
        return InsightDoc.model_validate(data)
    except ValidationError as exc:
        raise ValueError("; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:MAX_ERRORS_KEPT])) from exc


def _try(provider: Provider, uc: UseCase, facts: list[Fact], by_id: dict[str, Fact], feedback: list[str] | None):
    """One attempt. Returns (doc, validation) or raises ValueError (bad reply) / ProviderUnavailable."""
    system, user = build_prompt(uc, facts, feedback)
    reply = provider.generate(system, user, json_schema())
    doc = parse_reply(reply)
    return doc, validate(doc, by_id, uc.ai.allowed_recommendation_types)


def generate_insights(results: dict, uc: UseCase, provider: Provider | None = None, max_attempts: int = 2) -> dict:
    facts = build_facts(results, uc)
    by_id = facts_by_id(facts)
    attempts: list[dict] = []
    reason: str | None = NOT_CONFIGURED if provider is None else None
    doc = validation = None
    mode, used = "none", "none"

    if provider is not None:
        feedback: list[str] | None = None
        for n in range(1, max_attempts + 1):
            try:
                doc, validation = _try(provider, uc, facts, by_id, feedback)
            except ProviderUnavailable as exc:
                reason = f"AI provider unavailable: {str(exc)[:200]}"
                attempts.append({"n": n, "outcome": "unavailable", "errors": [str(exc)[:200]]})
                break
            except ValueError as exc:  # unreadable reply: worth one retry with the problem stated
                feedback = [str(exc)[:300]]
                attempts.append({"n": n, "outcome": "unreadable", "errors": feedback})
                doc = validation = None
                reason = "The AI reply could not be read as the required JSON."
                continue
            except Exception as exc:  # network error, timeout, anything the adapter did not map
                logger.warning("AI provider error: %s", type(exc).__name__)
                reason = f"AI provider error ({type(exc).__name__})."
                attempts.append({"n": n, "outcome": "error", "errors": [type(exc).__name__]})
                break
            if validation["ok"]:
                mode, used, reason = "ai", provider.name, None
                attempts.append({"n": n, "outcome": "accepted", "errors": []})
                break
            feedback = validation["errors"][:MAX_ERRORS_KEPT]
            attempts.append({"n": n, "outcome": "rejected", "errors": feedback})
            reason = "The AI reply broke the guardrails, so the template summary is shown."
            doc = validation = None

    if mode != "ai":
        try:
            doc, validation = _try(TemplateProvider(uc, facts), uc, facts, by_id, None)
            if validation["ok"]:
                mode, used = "fallback", "template"
            else:  # a bug in a template: never show it
                logger.error("Template summary failed its own guardrails: %s", validation["errors"])
                reason = "The template summary failed validation; nothing is shown. " + (reason or "")
                doc = None
        except Exception:
            logger.exception("Template summary failed")
            doc = validation = None
            reason = "The template summary could not be built. " + (reason or "")

    cited = (validation or {}).get("facts_cited", [])
    return {
        "mode": mode,
        "provider": used,
        "reason": reason,
        "attempts": attempts,
        "validation": None if validation is None else {"ok": validation["ok"], "numbers_checked": validation["numbers_checked"], "facts_cited": len(cited)},
        "document": doc.model_dump() if doc is not None else None,
        "facts": [{"id": i, "text": by_id[i].text} for i in cited],
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
