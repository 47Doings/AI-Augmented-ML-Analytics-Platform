"""The no-AI summary: plain templates filled from the same facts.

WHY IT EXISTS
    The AI provider can be missing (no key), down, or can fail the guardrails
    twice. The report must still say something useful and must say HONESTLY
    where it came from. This provider writes the same JSON shape the AI is
    asked for, so it goes through the SAME validator: if a template ever wrote
    a number that is not in its facts, that would be caught too (tests do this
    for both use cases).

WHAT IS GENERIC AND WHAT IS NOT
    The summary, findings and caveats below talk only about "the model",
    "inputs", "entities" and so on, never about churn or revenue.
    Business wording (the recommendations) lives in the use-case YAML under
    ai.fallback_recommendations, so a new use case brings its own.
"""
from __future__ import annotations

import json
import re

from core.ai.facts import Fact, facts_by_id
from core.use_case import UseCase

FORMATS = {
    "pct0": lambda v: f"{v * 100:.0f}%",
    "pct1": lambda v: f"{v * 100:.1f}%",
    "n0": lambda v: f"{v:,.0f}",
    "n1": lambda v: f"{v:,.1f}",
    "n2": lambda v: f"{v:,.2f}",
    "n3": lambda v: f"{v:,.3f}",
    "x1": lambda v: f"{v:.1f}x",
    "ghs0": lambda v: f"GHS {v:,.0f}",
}
_PLACEHOLDER = re.compile(r"\{([a-z0-9_.]+)\.([a-z0-9_]+)(?:\|([a-z0-9]+))?\}")
MAX_RECOMMENDATIONS = 5


class _Missing(Exception):
    pass


def render(template: str, by_id: dict[str, Fact]) -> str:
    """Fill {fact.id.field|format}. Raises _Missing when a fact or field is not there."""

    def sub(m: re.Match) -> str:
        fid, field, fmt = m.groups()
        fact = by_id.get(fid)
        if fact is None:
            raise _Missing(fid)
        if field in fact.values:
            if fmt not in FORMATS:
                raise ValueError(f"unknown format '{fmt}' in template (known: {sorted(FORMATS)})")
            return FORMATS[fmt](fact.values[field])
        if field in fact.labels:
            return fact.labels[field]
        raise _Missing(f"{fid}.{field}")

    return " ".join(_PLACEHOLDER.sub(sub, template).split())


def _direction(d: str) -> str:
    d = d.lower()
    if d.startswith("higher value") and "lower score" in d:
        return "higher values go with lower scores"
    if d.startswith("higher value") and "higher score" in d:
        return "higher values go with higher scores"
    return d


class TemplateProvider:
    name = "template"

    def __init__(self, uc: UseCase, facts: list[Fact]):
        self.uc, self.facts, self.by_id = uc, facts, facts_by_id(facts)

    # same interface as a real provider; the prompt is ignored
    def generate(self, system: str, user: str, json_schema: dict) -> str:
        return json.dumps(self._document())

    def _document(self) -> dict:
        by, uc = self.by_id, self.uc
        entity = uc.entity.name
        target = by["scope"].labels["target"]
        kpis = [f for f in self.facts if f.id.startswith("kpi.")]

        summary = []
        if kpis:
            summary.append({"text": "Headline numbers: " + "; ".join(f"{f.labels['label']} is {f.labels['shown']}" for f in kpis) + ".",
                            "facts": [f.id for f in kpis]})
        else:
            summary.append({"text": f"This run answers the business question for '{by['scope'].labels['use_case']}' using the platform's standard analysis.",
                            "facts": ["scope"]})
        if "model.compare" in by:
            c = by["model.compare"]
            summary.append({
                "text": (f"The selected model ({c.labels['model']}) scores {c.labels['metric']} {FORMATS['n3'](c.values['selected'])} on data it had not seen, "
                         f"against {FORMATS['n3'](c.values['baseline'])} for a baseline that ignores the inputs."),
                "facts": ["model.compare"]})
        else:
            summary.append({"text": "The model was trained and tested on held-out data; see the model comparison table for its scores.", "facts": ["model.test"]})

        findings = []
        drivers = [by[f"driver.{i}"] for i in (1, 2, 3) if f"driver.{i}" in by]
        if drivers:
            parts = "; ".join(f"{d.labels['name']} ({_direction(d.labels['direction'])})" for d in drivers)
            findings.append({"text": f"The inputs most strongly associated with {target} in this model are: {parts}.", "facts": [d.id for d in drivers]})
        if "decile.top" in by and "decile.bottom" in by:
            t, b = by["decile.top"], by["decile.bottom"]
            findings.append({
                "text": (f"Scores rank {entity}s usefully: in the riskiest decile {FORMATS['pct1'](t.values['rate'])} had the outcome, against "
                         f"{FORMATS['pct1'](t.values['base_rate'])} overall ({FORMATS['x1'](t.values['lift'])} the average) and "
                         f"{FORMATS['pct1'](b.values['rate'])} in the safest decile."),
                "facts": ["decile.top", "decile.bottom"]})
        if "drift.summary" in by:
            d = by["drift.summary"]
            findings.append({
                "text": (f"Drift check at {d.labels['later_cutoff']}: {FORMATS['n0'](d.values['stable'])} of {FORMATS['n0'](d.values['variables'])} input "
                         f"variables are stable, {FORMATS['n0'](d.values['watch'])} on watch and {FORMATS['n0'](d.values['alert'])} in alert."),
                "facts": ["drift.summary"]})
        if "dq.summary" in by:
            q = by["dq.summary"]
            text = (f"{FORMATS['n0'](q.values['checks_with_issues'])} of {FORMATS['n0'](q.values['checks'])} data-quality checks found problems, "
                    f"{FORMATS['n0'](q.values['high_severity_checks'])} of them high severity.")
            ids = ["dq.summary"]
            if "dq.1" in by:
                w = by["dq.1"]
                text += f" The most serious is {w.labels['check']}, affecting {FORMATS['n2'](w.values['pct_affected'])}% of its rows."
                ids.append("dq.1")
            findings.append({"text": text, "facts": ids})
        if not findings:
            findings.append({"text": "The model compared with its baseline is the main result of this run.", "facts": ["model.test"]})

        caveats = []
        if "model.split" in by:
            s = by["model.split"]
            text = f"Scores are measured on {FORMATS['n0'](s.values['test_rows'])} held-out {entity}s, chosen at random rather than by date."
            if "test_positives" in s.values:
                text += f" Only {FORMATS['n0'](s.values['test_positives'])} of them had the outcome, so the scores would move on another sample."
            caveats.append({"text": text, "facts": ["model.split"]})
        if by["scope"].labels.get("data_source") == "uploaded by the business":
            caveats.append({"text": "All figures come from the file the business uploaded. It was not independently verified, and the model can only reflect patterns in that file.",
                            "facts": ["scope"]})
        else:
            caveats.append({"text": "All figures come from synthetic demo data, not from a real business.", "facts": ["scope"]})
        caveats.append({"text": "The inputs listed describe patterns in past data. They show association only and do not explain why the outcome happens.",
                        "facts": [drivers[0].id] if drivers else ["model.test"]})

        recs = []
        for fr in self.uc.ai.fallback_recommendations:
            try:
                text = render(fr.text, by)
            except _Missing:
                continue  # the facts this recommendation needs are not in this run
            if all(i in by for i in fr.facts):
                recs.append({"type": fr.type, "text": text, "facts": fr.facts})
        return {"executive_summary": summary, "key_findings": findings, "recommendations": recs[:MAX_RECOMMENDATIONS], "caveats": caveats[:5]}
