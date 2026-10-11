"""Generate the synthetic dataset, and optionally load it into a workspace schema.

Run from the project root, for example:

    # just write the CSV files
    python -m scripts.generate_data

    # write CSVs, create the workspace schema if needed, and load everything
    python -m scripts.generate_data --schema ws_demo --create-schema --load

Same --seed and same library versions give byte-identical files.
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

import psycopg
from pydantic import ValidationError

from core.config import PROJECT_ROOT, get_settings
from core.loader import LoaderError, ensure_workspace_schema, load_csv_dir
from core.logging_setup import configure_logging
from core.use_case import UseCaseError, load_use_case
from datagen.telecom import GenConfig, generate, write_outputs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--customers", type=int, default=5000, help="number of customers before duplicates (default 5000)")
    p.add_argument("--seed", type=int, default=42, help="random seed (default 42)")
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "generated", help="output folder for CSVs")
    p.add_argument("--use-case", default="churn", help="use case whose cutoff/label window define the answer key")
    p.add_argument("--schema", help="workspace schema to load into, e.g. ws_demo")
    p.add_argument("--create-schema", action="store_true", help="create the schema and tables if they do not exist")
    p.add_argument("--schema-file", type=Path, default=PROJECT_ROOT / "db" / "schema.sql")
    p.add_argument("--views-file", type=Path, default=PROJECT_ROOT / "db" / "clean_views.sql")
    p.add_argument("--load", action="store_true", help="load the CSVs into --schema")
    args = p.parse_args()

    if args.load and not args.schema:
        p.error("--load needs --schema")

    # The answer key uses the SAME cutoff and window as the use case config (single source of truth).
    try:
        uc = load_use_case(args.use_case)
        cutoff = date.fromisoformat(str(uc.parameters["cutoff_date"]))
        window = int(uc.parameters["label_window_days"])
    except (UseCaseError, KeyError) as exc:
        print(f"Could not read cutoff_date / label_window_days from use case '{args.use_case}': {exc}")
        return 1

    cfg = GenConfig(n_customers=args.customers, seed=args.seed, cutoff_date=cutoff, label_window_days=window)
    tables, manifest, answer_key, dup_pairs = generate(cfg)
    write_outputs(tables, manifest, answer_key, dup_pairs, args.out)

    print(f"Wrote {len(tables)} CSV files + manifest.json to {args.out}")
    for name, n in manifest["row_counts"].items():
        print(f"  {name:<16}{n:>9,} rows")
    gt = manifest["ground_truth"]
    print(
        f"Answer key at cutoff {gt['cutoff_date']}: {gt['active_at_cutoff']:,} active customers, "
        f"{gt['churned_in_window']:,} cancel within {gt['label_window_days']} days ({gt['churn_rate_in_window']:.1%})"
    )

    if not args.load:
        return 0

    try:
        settings = get_settings()
        configure_logging(settings.log_level)
        if args.create_schema:
            created = ensure_workspace_schema(args.schema, args.schema_file, args.views_file)
            print(f"Schema {args.schema}: {'created tables' if created else 'tables already exist'}")
        counts = load_csv_dir(args.schema, args.out)
    except ValidationError as exc:
        print("Settings problem (check your .env):")
        for err in exc.errors():
            print(f"  {'.'.join(str(x) for x in err['loc'])}: {err['msg']}")
        return 1
    except (LoaderError, ValueError) as exc:
        print(f"Load failed: {exc}")
        return 1
    except psycopg.Error as exc:
        print(f"Database error: {str(exc).strip().splitlines()[0]}")
        return 1

    expected = manifest["row_counts"]
    ok = all(counts[t] == expected[t] for t in expected)
    print(f"Loaded into {args.schema}:")
    for t, n in counts.items():
        print(f"  {t:<16}{n:>9,} rows  {'OK' if n == expected[t] else 'MISMATCH (expected ' + str(expected[t]) + ')'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
