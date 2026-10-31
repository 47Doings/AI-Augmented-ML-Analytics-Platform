"""The facts sheet: the ONLY thing the AI ever sees.

THE PROBLEM THIS SOLVES
    If you hand a language model a raw dataset (or just ask "why are customers
    leaving?") it will happily write fluent text with numbers that were never
    computed anywhere: "churn fell 12% last quarter". So the AI never gets
    data. It gets a numbered list of facts, every number in it computed by SQL
    or the ML pipeline and read from results.json. Every sentence it writes
    must cite fact ids, and guardrails.py checks the sentence's numbers against
    the cited facts.

WHAT IS LEFT OUT ON PURPOSE
    - raw rows and customer identifiers (model.top_entities is never read)
    - arithmetic: differences, lifts and shares are computed HERE, so the AI
      never has to calculate
    Text that originally came from the data (plan names, regions) is cleaned
    (control characters removed, length capped) because the workspace owner
    controls that data and it will end up inside a prompt.

FACT IDS
    kpi.<id>                 headline numbers
    scope                    what question this run answers, and where the data came from (demo or uploaded)
    model.compare            selected model vs baseline on the held-out test set
    model.test               every test metric of the selected model
    model.baseline           every test metric of the baseline
    model.cv                 cross-validation scores
    model.split              train/test sizes
    model.flag               the "flag the riskiest X%" operating point (classification)
    driver.1 .. driver.5     most important input variables, with direction
    decile.top/bottom/top2_share   how well risk scores rank real outcomes (classification)
    drift.summary, drift.score     population stability after a later date
    dq.summary, dq.1 .. dq.3       data-quality checks
    bq.<question>.last/.prev/.r1..r3   rows of the business-question tables
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from core.use_case import UseCase

_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
MAX_LABEL = 60
N_DRIVERS = 5
N_DQ = 3
SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


@dataclass
class Fact:
    id: str
    text: str
    values: dict[str, float] = field(default_factory=dict)  # numbers this fact stands behind
    labels: dict[str, str] = field(default_factory=dict)  # words/dates this fact stands behind
    fractions: frozenset[str] = frozenset()  # value fields stored as 0..1 that may be written as a percentage


def clean_text(value, limit: int = MAX_LABEL) -> str:
    """Make text that came from data safe to place in a prompt and on a page."""
    s = _CONTROL.sub(" ", str(value))
    s = s.replace("<", "").replace(">", "")  # no markup, no arrows
    s = re.sub(r"\s+", " ", s).strip()
    return s[:limit]


def _num(v) -> float | None:
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _rows(payload: dict | None) -> list[dict]:
    if not payload:
        return []
    cols = payload["columns"]
    return [dict(zip(cols, r)) for r in payload["rows"]]


def _fmt(v: float) -> str:
    if float(v).is_integer():
        return f"{int(v):,}"
    return f"{v:,.4g}" if abs(v) < 1 else f"{v:,.2f}"


def _is_date(s) -> bool:
    return isinstance(s, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", s) is not None


def _row_fact(fid: str, what: str, row: dict) -> Fact:
    values, labels = {}, {}
    for k, v in row.items():
        n = _num(v)
        if n is not None:
            values[str(k)] = n
        elif isinstance(v, str):
            labels[str(k)] = clean_text(v)
    parts = [f"{k}={v}" for k, v in labels.items()] + [f"{k}={_fmt(v)}" for k, v in values.items()]
    return Fact(fid, f"{what}: " + ", ".join(parts), values, labels)


# ------------------------------------------------------------------ builders
def build_facts(results: dict, uc: UseCase) -> list[Fact]:
    facts: list[Fact] = []
    add = facts.append
    ucinfo = results["use_case"]

    uploaded = ucinfo.get("data_source") == "upload"
    source_text = ("The data is a file uploaded by the business; it was not independently verified."
                   if uploaded else "The data is synthetic demo data, not a real business.")
    add(Fact(
        "scope",
        f"Use case '{clean_text(ucinfo['name'])}'. Question: {clean_text(ucinfo['business_question'], 300)} "
        f"Target: {clean_text(ucinfo['target_description'], 300)} {source_text}",
        values={k: float(v) for k, v in ucinfo.get("parameters", {}).items() if k.endswith("_days") and _num(v) is not None},
        labels={"use_case": clean_text(ucinfo["name"]), "data_source": "uploaded by the business" if uploaded else "synthetic demo data",
                "target": clean_text(ucinfo["target"])},
    ))

    for k in results.get("kpis", []):
        add(Fact(f"kpi.{k['id']}", f"{clean_text(k['label'])}: {clean_text(k['shown'])}",
                 values={"value": float(k["value"])}, labels={"label": clean_text(k["label"]), "shown": clean_text(k["shown"])},
                 fractions=frozenset({"value"}) if k["format"] == "percent" else frozenset()))

    facts += _model_facts(results["model"])
    facts += _decile_facts(results["model"])
    facts += _drift_facts(results.get("drift"))
    facts += _dq_facts(results.get("data_quality"))
    facts += _bq_facts(results.get("business_questions", {}))

    ids = [f.id for f in facts]
    assert len(ids) == len(set(ids)), "duplicate fact ids"
    return facts


def _model_facts(model: dict) -> list[Fact]:
    out: list[Fact] = []
    card = model["card"]
    primary = card["primary_metric"]

    comp = _rows(model["comparison"])
    base_row = next((r for r in comp if r["role"] == "baseline"), None)
    sel_row = next((r for r in comp if r["selected"]), None)

    test = {k: float(v) for k, v in card["test_metrics"].items() if _num(v) is not None}
    out.append(Fact("model.test",
                    f"Selected model '{card['selected_model']}' on held-out test data: " + ", ".join(f"{k}={_fmt(v)}" for k, v in test.items()),
                    test, {"model": card["selected_model"]}, frozenset(k for k in test if k in {"precision", "recall", "f1", "accuracy"})))

    if base_row:
        base = {k[5:]: float(v) for k, v in base_row.items() if k.startswith("test_") and _num(v) is not None and k != "test_flagged"}
        out.append(Fact("model.baseline",
                        f"Baseline '{base_row['model']}' (ignores the inputs) on the same test data: " + ", ".join(f"{k}={_fmt(v)}" for k, v in base.items()),
                        base, {"model": base_row["model"]}))
        if sel_row and _num(base_row.get(f"test_{primary}")) is not None and _num(sel_row.get(f"test_{primary}")) is not None:
            s, b = float(sel_row[f"test_{primary}"]), float(base_row[f"test_{primary}"])
            out.append(Fact("model.compare",
                            f"Test {primary}: selected {_fmt(s)} vs baseline {_fmt(b)} (difference {_fmt(s - b)}).",
                            {"selected": s, "baseline": b, "delta": s - b},
                            {"metric": primary, "model": card["selected_model"], "baseline_model": base_row["model"]}))

    cv = card["cv_score"]
    out.append(Fact("model.cv",
                    f"Cross-validation {primary} on the training data: selected {_fmt(cv['selected'])} vs baseline {_fmt(cv['baseline'])}. "
                    f"Beats baseline: {'yes' if card['beats_baseline_in_cv'] else 'no'}.",
                    {"selected": float(cv["selected"]), "baseline": float(cv["baseline"]), "delta": float(cv["selected"] - cv["baseline"])},
                    {"metric": primary, "beats_baseline": "yes" if card["beats_baseline_in_cv"] else "no"}))

    rows = card["rows"]
    split_vals = {"total_rows": float(rows["total"]), "train_rows": float(rows["train"]), "test_rows": float(rows["test"])}
    text = f"{rows['total']:,} rows in total: {rows['train']:,} to train, {rows['test']:,} held out for testing. The split is random, not by time."
    pr = card.get("positive_rate")
    if pr:
        split_vals["test_positives"] = float(round(pr["test"] * rows["test"]))
        split_vals["test_positive_rate"] = float(pr["test"])
        text += f" The test set holds {int(split_vals['test_positives'])} positive cases ({pr['test']:.1%})."
    out.append(Fact("model.split", text, split_vals, {}, frozenset({"test_positive_rate"})))

    op = card.get("operating_point")
    if op:
        out.append(Fact("model.flag",
                        f"Operating point: flag the riskiest {op['flag_top_fraction']:.0%} of entities.",
                        {"fraction": float(op["flag_top_fraction"])}, {}, frozenset({"fraction"})))

    imp = _rows(model["importance"])[:N_DRIVERS]
    for i, r in enumerate(imp, 1):
        name = clean_text(r["feature"])
        out.append(Fact(f"driver.{i}",
                        f"Input #{i} by importance: {name} (permutation importance {_fmt(float(r['importance']))}; "
                        f"{clean_text(r['direction'])}; clearly matters: {'yes' if r['clearly_matters'] else 'no'}).",
                        {"rank": float(i), "importance": float(r["importance"])},
                        {"name": name, "direction": clean_text(r["direction"]), "clearly_matters": "yes" if r["clearly_matters"] else "no"}))
    return out


def _decile_facts(model: dict) -> list[Fact]:
    dec = _rows(model.get("deciles"))
    if not dec:
        return []
    total_n = sum(r["entities"] for r in dec)
    total_pos = sum(r["positives"] for r in dec)
    base = total_pos / total_n if total_n else 0.0
    by = sorted(dec, key=lambda r: r["risk_decile"], reverse=True)
    top, bottom = by[0], by[-1]
    out = [
        Fact("decile.top",
             f"Riskiest decile (highest scores): {_fmt(top['entities'])} entities, {top['rate']:.1%} actually positive, "
             f"versus {base:.1%} overall ({top['rate'] / base:.1f}x the average)." if base else "Riskiest decile.",
             {"entities": float(top["entities"]), "positives": float(top["positives"]), "rate": float(top["rate"]),
              "base_rate": float(base), "lift": float(top["rate"] / base) if base else 0.0},
             {}, frozenset({"rate", "base_rate"})),
        Fact("decile.bottom",
             f"Safest decile (lowest scores): {bottom['rate']:.1%} actually positive.",
             {"rate": float(bottom["rate"]), "entities": float(bottom["entities"])}, {}, frozenset({"rate"})),
    ]
    if total_pos and len(by) >= 2:
        share = (by[0]["positives"] + by[1]["positives"]) / total_pos
        out.append(Fact("decile.top2_share",
                        f"The two riskiest deciles (20% of entities) contain {share:.1%} of all positive cases.",
                        {"share": float(share)}, {}, frozenset({"share"})))
    return out


def _drift_facts(drift: dict | None) -> list[Fact]:
    rows = _rows(drift)
    if not rows:
        return []
    feats = [r for r in rows if r["kind"] != "score"]
    score = next((r for r in rows if r["kind"] == "score"), None)
    count = lambda s: float(sum(1 for r in feats if r["status"] == s))  # noqa: E731
    worst = max(feats, key=lambda r: r["psi"]) if feats else None
    out = [Fact(
        "drift.summary",
        f"Drift check against {drift['later_cutoff']} ({drift['entities']:,} entities): {int(count('stable'))} of {len(feats)} input variables stable, "
        f"{int(count('watch'))} on watch, {int(count('alert'))} in alert." + (f" Largest PSI: {worst['variable']} at {worst['psi']:.3f}." if worst else ""),
        {"variables": float(len(feats)), "stable": count("stable"), "watch": count("watch"), "alert": count("alert"),
         "max_psi": float(worst["psi"]) if worst else 0.0, "entities": float(drift["entities"])},
        {"later_cutoff": drift["later_cutoff"], "worst_variable": clean_text(worst["variable"]) if worst else ""},
    )]
    if score:
        out.append(Fact("drift.score", f"Model score distribution drift (PSI): {score['psi']:.3f}, status {score['status']}.",
                        {"psi": float(score["psi"])}, {"status": clean_text(score["status"])}))
    return out


def _dq_facts(dq: dict | None) -> list[Fact]:
    rows = _rows(dq)
    if not rows:
        return []
    bad = [r for r in rows if (r["issues"] or 0) > 0]
    high = [r for r in bad if r["severity"] == "high"]
    out = [Fact("dq.summary",
                f"{len(bad)} of {len(rows)} data-quality checks found problems; {len(high)} of them are high severity. "
                f"Problems are handled in the clean layer (views) and the raw data is not modified.",
                {"checks": float(len(rows)), "checks_with_issues": float(len(bad)), "high_severity_checks": float(len(high))})]
    worst = sorted(bad, key=lambda r: (SEVERITY_ORDER.get(r["severity"], 3), -(r["pct_affected"] or 0)))[:N_DQ]
    for i, r in enumerate(worst, 1):
        out.append(Fact(f"dq.{i}",
                        f"Data-quality issue #{i}: {clean_text(r['check_id'])} ({r['severity']} severity) affects {_fmt(float(r['issues']))} of "
                        f"{_fmt(float(r['total']))} rows ({r['pct_affected']}%). Handling: {clean_text(r['handling'], 120)}.",
                        {"issues": float(r["issues"]), "total": float(r["total"]), "pct_affected": float(r["pct_affected"])},
                        {"check": clean_text(r["check_id"]), "severity": clean_text(r["severity"]), "handling": clean_text(r["handling"], 120)}))
    return out


def _bq_facts(questions: dict[str, dict]) -> list[Fact]:
    out: list[Fact] = []
    for name, payload in questions.items():
        rows = _rows(payload)
        if not rows:
            continue
        nm = clean_text(name)
        first_col = payload["columns"][0]
        is_series = _is_date(rows[0][first_col])
        if is_series:
            out.append(_row_fact(f"bq.{nm}.last", f"{nm}, latest period", rows[-1]))
            if len(rows) > 1:
                out.append(_row_fact(f"bq.{nm}.prev", f"{nm}, previous period", rows[-2]))
        else:
            for i, r in enumerate(rows[:3], 1):
                out.append(_row_fact(f"bq.{nm}.r{i}", f"{nm}, row {i}", r))
            if len(rows) > 3:
                out.append(_row_fact(f"bq.{nm}.last", f"{nm}, last row", rows[-1]))
    return out


def facts_by_id(facts: list[Fact]) -> dict[str, Fact]:
    return {f.id: f for f in facts}
