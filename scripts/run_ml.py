"""Train the model for a use case, and monitor it for drift.

Run from the project root (the SQL layer must already be set up: see Step 4).

    python -m scripts.run_ml train   --schema ws_demo
    python -m scripts.run_ml monitor --schema ws_demo --cutoff 2026-08-31

train    builds the training table, trains baseline + candidates, explains the winner,
         and writes everything to outputs/ml/<use_case>/.
monitor  loads the saved model, rebuilds the features as they look at a LATER cutoff
         with the same SQL, and reports PSI drift vs the training data.
"""
import argparse
import sys

import joblib
import pandas as pd
import psycopg
from pydantic import ValidationError

from core.config import get_settings
from core.drift import drift_report
from core.loader import LoaderError
from core.logging_setup import configure_logging
from core.ml_pipeline import MlError, default_out_dir, prepare_frame, train
from core.sql_runner import SqlError, build_training_set, features_as_of
from core.use_case import UseCaseError, load_use_case

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)


def do_train(schema, uc) -> None:
    summary = build_training_set(schema, uc)
    print(f"Training table: {summary['rows']:,} rows" +
          (f", {summary['positives']:,} positives ({summary['positive_rate']:.1%})" if "positives" in summary else ""))
    r = train(schema, uc)
    out = default_out_dir(uc)
    comp = r["comparison"]
    cols = ["model", "role", "selected", "cv_score"] + [c for c in comp.columns if c.startswith("test_") and c != "test_flagged"]
    print("\n=== Model comparison (cv_score = cross-validation on train; test_* = held-out test set) ===")
    shown = comp[cols].dropna(axis=1, how="all")  # e.g. MAPE is not defined when some targets are 0
    print(shown.round(3).to_string(index=False))
    card = r["card"]
    print(f"\nSelected: {r['best_name']}  (chosen by cross-validation {uc.metrics.primary}, not by the test set)")
    print(f"Beats the baseline in cross-validation: {card['beats_baseline_in_cv']}")
    if uc.problem_type == "classification":
        print(f"Operating point: flag the riskiest {uc.evaluation.flag_top_fraction:.0%} -> "
              f"precision {card['test_metrics']['precision']:.1%}, recall {card['test_metrics']['recall']:.1%}, "
              f"lift {card['test_metrics']['lift']:.1f}x")
    print("\n=== What drives the prediction (permutation importance on the test set) ===")
    print(r["importance"].head(10).round(4).to_string(index=False))
    print(f"\nSaved to {out}:\n  model.joblib, model_card.json, model_comparison.csv, feature_importance.csv, scores.csv")


def do_monitor(schema, uc, cutoff) -> None:
    path = default_out_dir(uc) / "model.joblib"
    if not path.is_file():
        raise MlError(f"No saved model at {path}. Run 'train' first.")
    bundle = joblib.load(path)
    new = prepare_frame(features_as_of(schema, uc, cutoff), uc)
    X = new[bundle["features"]]
    scores = bundle["pipeline"].predict_proba(X)[:, 1] if bundle["problem_type"] == "classification" else bundle["pipeline"].predict(X)
    rep = drift_report(bundle, X, scores)
    out = default_out_dir(uc)
    rep.round({"psi": 4}).to_csv(out / f"drift_{cutoff}.csv", index=False)
    print(f"Drift: training data (cutoff {uc.parameters['cutoff_date']}) vs the same SQL at cutoff {cutoff} "
          f"({len(new):,} entities)\n")
    print(rep.round(4).to_string(index=False))
    worst = rep["status"].value_counts().to_dict()
    print(f"\nSummary: {worst}.  <0.10 stable | 0.10-0.25 watch | >0.25 alert")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["train", "monitor"])
    p.add_argument("--schema", required=True)
    p.add_argument("--use-case", default="churn")
    p.add_argument("--cutoff", help="with 'monitor': the later date to compare against, YYYY-MM-DD")
    args = p.parse_args()
    if args.step == "monitor" and not args.cutoff:
        p.error("monitor needs --cutoff YYYY-MM-DD")
    try:
        configure_logging(get_settings().log_level)
        uc = load_use_case(args.use_case)
        do_train(args.schema, uc) if args.step == "train" else do_monitor(args.schema, uc, args.cutoff)
    except ValidationError as exc:
        print("Settings problem (check your .env):", "; ".join(f"{e['loc'][0]}: {e['msg']}" for e in exc.errors()))
        return 1
    except (UseCaseError, SqlError, LoaderError, MlError) as exc:
        print(f"Failed: {exc}")
        return 1
    except psycopg.Error as exc:
        print(f"Database error: {str(exc).strip().splitlines()[0]}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
