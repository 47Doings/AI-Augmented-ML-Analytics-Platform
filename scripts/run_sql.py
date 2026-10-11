"""Run the SQL layer for a use case against a workspace.

Run from the project root. Steps, in the order the pipeline uses them:

    python -m scripts.run_sql clean-layer --schema ws_demo   # (re)create the clean_* views
    python -m scripts.run_sql dq          --schema ws_demo   # data-quality report
    python -m scripts.run_sql training    --schema ws_demo   # build the training_set table
    python -m scripts.run_sql questions   --schema ws_demo   # business questions (needs training)
    python -m scripts.run_sql kpis        --schema ws_demo   # headline KPIs      (needs training)
    python -m scripts.run_sql all         --schema ws_demo   # everything, in order

Add --query <name> with 'questions' to run just one. Results are also saved
as CSV files under outputs/sql/<use_case>/ (used later by the report).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import psycopg
from pydantic import ValidationError

from core.config import PROJECT_ROOT, get_settings
from core.loader import LoaderError, apply_sql_file
from core.logging_setup import configure_logging
from core.sql_runner import SqlError, build_training_set, run_named
from core.use_case import UseCase, UseCaseError, load_use_case

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 70)


def show(title: str, df: pd.DataFrame, out_dir: Path, filename: str, max_rows: int = 60) -> None:
    print(f"\n=== {title} ===")
    print(df.head(max_rows).to_string(index=False) if len(df) else "(no rows)")
    if len(df) > max_rows:
        print(f"... {len(df) - max_rows} more rows (see the CSV)")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / filename, index=False)


def step_dq(schema: str, uc: UseCase, out_dir: Path) -> None:
    df = run_named(schema, uc.sql.data_quality, uc.parameters)["data_quality"]
    df = df.astype({"issues": "int64", "total": "int64"})
    found = df[df["issues"] > 0]
    show(f"Data-quality report: {len(found)} of {len(df)} checks found problems", found, out_dir, "data_quality.csv")


def step_training(schema: str, uc: UseCase, out_dir: Path) -> None:
    summary = build_training_set(schema, uc)
    print("\n=== Training table built: training_set ===")
    print(f"rows: {summary['rows']:,}")
    if "positives" in summary:
        print(f"{uc.target.name} = 1: {summary['positives']:,} ({summary['positive_rate']:.1%})")
    print("columns:", ", ".join([uc.entity.key, *uc.feature_columns, uc.target.name]))


def step_questions(schema: str, uc: UseCase, out_dir: Path, only: str | None) -> None:
    for name, df in run_named(schema, uc.sql.business_questions, uc.parameters, only).items():
        show(name, df, out_dir, f"{name}.csv", max_rows=45)


def step_kpis(schema: str, uc: UseCase, out_dir: Path) -> None:
    rows = []
    for kpi in uc.kpis:
        value = float(next(iter(run_named(schema, kpi.sql, uc.parameters).values())).iloc[0, 0])
        shown = {"percent": f"{value:.1%}", "currency": f"GHS {value:,.0f}", "number": f"{value:,.2f}"}[kpi.format]
        rows.append({"kpi": kpi.id, "label": kpi.label, "value": value, "shown": shown})
    show("KPIs", pd.DataFrame(rows), out_dir, "kpis.csv")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["clean-layer", "dq", "training", "questions", "kpis", "all"])
    p.add_argument("--schema", required=True, help="workspace schema, e.g. ws_demo")
    p.add_argument("--use-case", default="churn")
    p.add_argument("--query", help="with 'questions': run only this query")
    p.add_argument("--views-file", type=Path, default=PROJECT_ROOT / "db" / "clean_views.sql")
    args = p.parse_args()

    try:
        settings = get_settings()
        configure_logging(settings.log_level)
        uc = load_use_case(args.use_case)
        out_dir = PROJECT_ROOT / "outputs" / "sql" / uc.id

        if args.step in ("clean-layer", "all"):
            apply_sql_file(args.schema, args.views_file)
            print("Clean layer applied: clean_customers, clean_subscriptions, clean_payments, clean_usage, clean_tickets")
        if args.step in ("dq", "all"):
            step_dq(args.schema, uc, out_dir)
        if args.step in ("training", "all"):
            step_training(args.schema, uc, out_dir)
        if args.step in ("questions", "all"):
            step_questions(args.schema, uc, out_dir, args.query)
        if args.step in ("kpis", "all"):
            step_kpis(args.schema, uc, out_dir)
    except ValidationError as exc:
        print("Settings problem (check your .env):")
        for err in exc.errors():
            print(f"  {'.'.join(str(x) for x in err['loc'])}: {err['msg']}")
        return 1
    except (UseCaseError, SqlError, LoaderError, ValueError) as exc:
        print(f"Failed: {exc}")
        return 1
    except psycopg.Error as exc:
        print(f"Database error: {str(exc).strip().splitlines()[0]}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
