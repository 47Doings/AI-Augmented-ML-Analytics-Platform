"""Guardrails: decide whether an AI answer may be shown.

Each check exists because of a specific failure:

  1. citations   AI cites a fact id that does not exist -> invented evidence.
  2. numbers     AI writes "churn fell 12%" -> every number in a sentence must
                 equal (after rounding) a number in the facts THAT SENTENCE CITES.
                 A real number attached to the wrong fact is rejected too.
  3. causation   data like this shows association, not cause -> words that
                 claim cause are banned ("because", "caused", "due to", ...).
  4. types       recommendations must be one of the use case's allowed types.
  5. markup      no HTML or links in text (it is shown in a web page).

Limits (said honestly, also in the README): number WORDS ("two thirds") are not
checked; small whole numbers 0-10 are allowed without a source because
sentences like "the top 3 inputs" are normal; the checks do not understand
meaning, so a true number can still be used in a misleading sentence. They
make invented numbers and invented evidence hard, not impossible to miss.
"""
from __future__ import annotations

import re

from core.ai.facts import Fact
from core.ai.schema import Bullet, InsightDoc

# number: digits with optional thousands commas and decimals. Not part of a word (r2, 90d, f1).
_NUMBER = re.compile(r"(?<![A-Za-z_\d.,])\d[\d,]*(?:\.\d+)?")
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b(?:19|20)\d{2}\b")
_CAUSAL = re.compile(
    r"\b(because|caus\w*|due to|as a result|result(?:s|ed|ing)? in|lead(?:s|ing)? to|led to|therefore|thanks to|"
    r"the reason (?:for|why)|is responsible for)\b",
    re.IGNORECASE,
)
_MARKUP = re.compile(r"<[^>]*>|https?://|www\.|javascript:|\bon\w+\s*=", re.IGNORECASE)
SMALL_INT_MAX = 10


def _tokens(text: str) -> list[tuple[float, int, bool, bool]]:
    """(value, decimals, is_percent, is_plain_small_int) for each number in the text."""
    text = _DATE.sub(" ", text)
    out = []
    for m in _NUMBER.finditer(text):
        raw = m.group().rstrip(",").replace(",", "")
        if not raw:
            continue
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        after = text[m.end(): m.end() + 9].lower()
        is_pct = after.startswith("%") or after.lstrip().startswith(("percent", "per cent", "pct"))
        value = float(raw)
        out.append((value, decimals, is_pct, decimals == 0 and not is_pct and value <= SMALL_INT_MAX))
    return out


def _allowed_numbers(facts: list[Fact]) -> list[tuple[float, bool]]:
    """(number, only_if_written_as_percent) for every number the cited facts stand behind."""
    allowed = []
    for f in facts:
        for k, v in f.values.items():
            allowed.append((abs(v), False))
            if k in f.fractions:
                allowed.append((abs(v) * 100, True))
        for label in f.labels.values():  # e.g. the "5" in plan "Starter 5GB", the "90" in failed_payments_90d
            for value, *_ in _tokens(label):
                allowed.append((value, False))
    return allowed


def _matches(value: float, decimals: int, is_pct: bool, allowed: list[tuple[float, bool]]) -> bool:
    tol = 0.5 * 10 ** (-decimals) + 1e-9
    for a, pct_only in allowed:
        if pct_only and not is_pct:
            continue
        if abs(a - value) <= tol:
            return True
    return False


def check_numbers(text: str, cited: list[Fact]) -> tuple[list[str], int]:
    allowed = _allowed_numbers(cited)
    bad, checked = [], 0
    for value, decimals, is_pct, small in _tokens(text):
        if small:
            continue
        checked += 1
        if not _matches(value, decimals, is_pct, allowed):
            shown = f"{value:,.{decimals}f}" + ("%" if is_pct else "")
            bad.append(shown)
    return bad, checked


def _check_bullet(where: str, b: Bullet, by_id: dict[str, Fact]) -> tuple[list[str], int]:
    errors: list[str] = []
    unknown = [i for i in b.facts if i not in by_id]
    if unknown:
        errors.append(f"{where}: cites unknown fact ids {unknown}. Use only ids from the facts sheet.")
    cited = [by_id[i] for i in b.facts if i in by_id]
    bad, checked = check_numbers(b.text, cited)
    if bad and cited:
        errors.append(
            f"{where}: the number(s) {bad} do not appear in the cited facts {[f.id for f in cited]}. "
            f"Use only numbers written in the facts you cite, and cite the fact each number comes from."
        )
    if m := _CAUSAL.search(b.text):
        errors.append(f"{where}: causal wording '{m.group(0)}' is not allowed. Describe associations only (\"is associated with\", \"goes with\").")
    if _MARKUP.search(b.text):
        errors.append(f"{where}: contains markup or a link, which is not allowed. Plain text only.")
    return errors, checked


def validate(doc: InsightDoc, facts: dict[str, Fact], allowed_types: list[str]) -> dict:
    """Returns {"ok": bool, "errors": [...], "numbers_checked": int, "facts_cited": [ids]}."""
    errors: list[str] = []
    checked = 0
    cited_ids: set[str] = set()
    sections = [("executive_summary", doc.executive_summary), ("key_findings", doc.key_findings),
                ("recommendations", doc.recommendations), ("caveats", doc.caveats)]
    for name, items in sections:
        for i, b in enumerate(items, 1):
            e, n = _check_bullet(f"{name}[{i}]", b, facts)
            errors += e
            checked += n
            cited_ids.update(x for x in b.facts if x in facts)
    for i, r in enumerate(doc.recommendations, 1):
        if r.type not in allowed_types:
            errors.append(f"recommendations[{i}]: type '{r.type}' is not allowed. Allowed: {allowed_types}.")
    return {"ok": not errors, "errors": errors, "numbers_checked": checked, "facts_cited": sorted(cited_ids)}
