"""The whole analysis flow, for any use case, in one function.

    clean layer -> data quality -> training table -> business questions
        -> KPIs -> model -> drift check -> results.json

Nothing in here mentions churn or revenue. Everything use-case specific comes
from the use-case config (YAML) and the SQL files it points to. The API calls
run_pipeline() in the background; scripts and notebooks can call it too.

results.json is the single record of a run. The report (Step 9) and the AI
insights (Step 8) read ONLY this file, so every number they show or explain can
be traced back to a computed value.
"""
from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from core.ai.insights import generate_insights
from core.ai.snwolley_adapter import get_provider
from core.config import PROJECT_ROOT, get_settings
from core.drift import drift_report
from core.loader import apply_sql_file, load_csv_dir
from core.ml_pipeline import prepare_frame, train
from core.sql_runner import build_training_set, features_as_of, run_named
from core.use_case import UseCase
from datagen.telecom import GenConfig, generate, write_outputs

logger = logging.getLogger(__name__)

VIEWS_SQL = PROJECT_ROOT / "db" / "clean_views.sql"
MAX_ROWS_PER_TABLE = 100

STEPS = [
    ("clean_layer", "Clean layer (views)"),
    ("data_quality", "Data-quality checks"),
    ("training_table", "Training table"),
    ("business_questions", "Business questions"),
    ("kpis", "KPIs"),
    ("model", "Model training and explanation"),
    ("monitoring", "Drift check"),
    ("insights", "AI insights (checked against the numbers)"),
]


def initial_steps() -> list[dict]:
    return [{"id": i, "label": label, "status": "pending", "seconds": None, "detail": None} for i, label in STEPS]


# ------------------------------------------------------------------ helpers
def _clean(v):
    """Make any value JSON-safe: numpy -> python, NaN/NaT -> None, dates -> 'YYYY-MM-DD'."""
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, (float, np.floating)):
        return None if math.isnan(v) or math.isinf(v) else float(v)
    if isinstance(v, (pd.Timestamp, datetime, date)):
        return None if pd.isna(v) else v.isoformat()[:10]
    if v is pd.NA or v is pd.NaT:
        return None
    return v


def table_payload(df: pd.DataFrame, limit: int = MAX_ROWS_PER_TABLE) -> dict:
    head = df.head(limit)
    rows = [[_clean(v) for v in rec] for rec in head.itertuples(index=False, name=None)]
    return {"columns": [str(c) for c in head.columns], "rows": rows, "total_rows": int(len(df))}


def _shown(value: float, fmt: str) -> str:
    return {"percent": f"{value:.1%}", "currency": f"GHS {value:,.0f}"}.get(fmt, f"{value:,.2f}")


class _Tracker:
    """Keeps the step list up to date and tells the caller (the API) after each change."""

    def __init__(self, on_progress: Callable[[list[dict]], None] | None):
        self.steps = initial_steps()
        self.on_progress = on_progress

    def run(self, step_id: str, fn: Callable[[], object], detail: Callable[[object], str] | None = None):
        step = next(s for s in self.steps if s["id"] == step_id)
        step["status"] = "running"
        self._emit()
        t0 = time.perf_counter()
        try:
            out = fn()
        except Exception:
            step["status"], step["seconds"] = "failed", round(time.perf_counter() - t0, 1)
            self._emit()
            raise
        step["status"], step["seconds"] = "done", round(time.perf_counter() - t0, 1)
        step["detail"] = detail(out) if detail else None
        self._emit()
        return out

    def skip(self, step_id: str, why: str) -> None:
        step = next(s for s in self.steps if s["id"] == step_id)
        step["status"], step["detail"] = "skipped", why
        self._emit()

    def _emit(self) -> None:
        if self.on_progress:
            self.on_progress([dict(s) for s in self.steps])


# ------------------------------------------------------------------ data
def load_synthetic_data(schema: str, uc: UseCase, customers: int, seed: int, data_dir: Path) -> dict[str, int]:
    """Generate the demo business's data and load it into the workspace (replacing what was there).

    The generator needs the use case's cutoff and label window only to write its answer key.
    """
    cutoff = date.fromisoformat(str(uc.parameters["cutoff_date"]))
    window = int(uc.parameters["label_window_days"])
    cfg = GenConfig(n_customers=customers, seed=seed, cutoff_date=cutoff, label_window_days=window)
    tables, manifest, answer_key, dup_pairs = generate(cfg)
    write_outputs(tables, manifest, answer_key, dup_pairs, data_dir)
    return load_csv_dir(schema, data_dir)


def later_cutoff(uc: UseCase) -> date | None:
    """Last day of the month before the data export date: a later 'today' to check drift against."""
    end = uc.parameters.get("data_end_date")
    if not end:
        return None
    first_of_month = date.fromisoformat(str(end)).replace(day=1)
    cutoff = first_of_month - timedelta(days=1)
    return cutoff if cutoff > date.fromisoformat(str(uc.parameters["cutoff_date"])) else None


# ------------------------------------------------------------------ the flow
def run_pipeline(schema: str, uc: UseCase, out_dir: Path, on_progress: Callable[[list[dict]], None] | None = None) -> dict:
    """Run every stage for one use case in one workspace; write and return results.json's content."""
    out_dir.mkdir(parents=True, exist_ok=True)
    t = _Tracker(on_progress)
    params = uc.parameters

    t.run("clean_layer", lambda: apply_sql_file(schema, VIEWS_SQL))

    dq = t.run(
        "data_quality",
        lambda: run_named(schema, uc.sql.data_quality, params)["data_quality"] if uc.sql.data_quality else pd.DataFrame(),
        lambda df: f"{int((df['issues'] > 0).sum())} of {len(df)} checks found problems" if len(df) else "none configured",
    )

    summary = t.run(
        "training_table",
        lambda: build_training_set(schema, uc),
        lambda s: f"{s['rows']:,} rows" + (f", {s['positives']:,} positives ({s['positive_rate']:.1%})" if "positives" in s else ""),
    )

    questions: dict[str, pd.DataFrame] = {}
    if uc.sql.business_questions:
        questions = t.run("business_questions", lambda: run_named(schema, uc.sql.business_questions, params),
                          lambda q: f"{len(q)} questions")
    else:
        t.skip("business_questions", "none configured")

    def kpis():
        out = []
        for k in uc.kpis:
            value = float(next(iter(run_named(schema, k.sql, params).values())).iloc[0, 0])
            out.append({"id": k.id, "label": k.label, "value": value, "format": k.format, "shown": _shown(value, k.format)})
        return out

    kpi_values = t.run("kpis", kpis, lambda k: f"{len(k)} KPIs")

    ml = t.run("model", lambda: train(schema, uc, out_dir / "ml"),
               lambda r: f"selected {r['best_name']}")

    drift = None
    cutoff2 = later_cutoff(uc)
    if cutoff2 is None:
        t.skip("monitoring", "no later date available")
    else:
        def monitor():
            bundle = joblib.load(out_dir / "ml" / "model.joblib")
            new = prepare_frame(features_as_of(schema, uc, cutoff2.isoformat()), uc)
            X = new[bundle["features"]]
            pipe = bundle["pipeline"]
            scores = pipe.predict_proba(X)[:, 1] if bundle["problem_type"] == "classification" else pipe.predict(X)
            return drift_report(bundle, X, scores), len(new)

        drift_df, n_new = t.run("monitoring", monitor, lambda r: f"{int((r[0]['status'] != 'stable').sum())} of {len(r[0])} variables moved")
        drift = {"later_cutoff": cutoff2.isoformat(), "entities": n_new, **table_payload(drift_df.round({"psi": 4}))}

    comparison = ml["comparison"].copy()
    model = {
        "selected": ml["best_name"],
        "comparison": table_payload(comparison),
        "importance": table_payload(ml["importance"].round({"importance": 5, "std": 5})),
        "card": ml["card"],
    }
    if uc.problem_type == "classification":
        s = ml["scored"]
        dec = (s.groupby("risk_decile").agg(entities=("actual", "size"), positives=("actual", "sum"), rate=("actual", "mean"))
               .sort_index(ascending=False).reset_index())
        model["deciles"] = table_payload(dec.round({"rate": 4}))
        top = s.sort_values("score", ascending=False).head(15)[[uc.entity.key, "score"]].round({"score": 4})
        model["top_entities"] = table_payload(top)

    results = {
        "use_case": {
            "id": uc.id, "name": uc.name, "problem_type": uc.problem_type, "business_question": uc.business_question.strip(),
            "target": uc.target.name, "target_description": uc.target.description.strip(),
            "parameters": {k: v for k, v in uc.parameters.items()},
        },
        "training_table": {k: v for k, v in summary.items() if k != "columns"},
        "kpis": kpi_values,
        "data_quality": table_payload(dq) if len(dq) else None,
        "business_questions": {name: table_payload(df) for name, df in questions.items()},
        "model": model,
        "drift": drift,
        "steps": t.steps,
    }
    # AI insights read ONLY `results` (never raw data). generate_insights never raises: the worst case is
    # mode "none" with a reason, so a missing or broken AI provider cannot fail the analysis run.
    insights = t.run(
        "insights",
        lambda: generate_insights(results, uc, get_provider(get_settings())),
        lambda r: {"ai": "written by the AI provider", "fallback": "template summary (no AI provider answered)"}.get(r["mode"], "not available"),
    )
    results["steps"] = t.steps
    (out_dir / "insights.json").write_text(json.dumps(insights, indent=2, allow_nan=False, default=_clean), encoding="utf-8")
    # allow_nan=False: if a NaN slipped through, fail loudly here instead of writing invalid JSON
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False, default=_clean), encoding="utf-8")
    return results
