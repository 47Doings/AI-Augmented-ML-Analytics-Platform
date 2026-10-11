"""Prove the SQL layer is correct, using the answer key from the data generator.

Run from the project root (needs the database and data/generated/):
    python -m scripts.verify_sql --schema ws_demo

What it checks
  1. Data-quality checks vs the injected-problem counts in data/generated/manifest.json
       exact   : the SQL must find exactly what was injected
       at least: later steps (duplicating rows) can add more of the same problem, so
                 the SQL must find at least what was injected
  2. The training table's TARGET against an independent source of truth
       churn   : labels vs the generator's answer key (answer_key/labels_at_cutoff.csv);
                 every difference must be a customer whose cancellation date is missing
       revenue : revenue_next_30d recomputed in pandas straight from the raw CSV files
                 (own re-implementation of the cleaning rules), must match to the pesewa
       other   : generic sanity check only (no missing / non-numeric target)
  3. No leakage: delete ALL data after the cutoff, rebuild the training table, and
       confirm no feature changed (done inside a transaction that is rolled back)
  4. The use-case config matches the SQL (every promised column exists)
  5. Every business question and KPI runs and returns rows
"""
import argparse
import json
import sys

import pandas as pd
import psycopg
from pydantic import ValidationError

from core.config import PROJECT_ROOT, get_settings
from core.db import connect
from core.loader import LoaderError, apply_sql_file
from core.sql_runner import SqlError, build_training_set, load_named_queries, run_named, run_query
from core.use_case import UseCaseError, load_use_case

DATA_DIR = PROJECT_ROOT / "data" / "generated"

# check_id in the SQL -> key in manifest.json. Same text unless listed here.
EXACT = {
    "customers.region_missing", "customers.region_inconsistent_spelling", "customers.phone_missing",
    "customers.phone_format_inconsistent", "customers.name_missing", "customers.customer_type_inconsistent_case",
    "customers.acquisition_channel_missing", "customers.birth_year_missing_individuals",
    "customers.birth_year_implausible", "subscriptions.cancelled_but_end_date_missing",
    "subscriptions.end_date_before_start_date", "subscriptions.status_inconsistent_case",
    "subscriptions.cancel_reason_missing", "payments.exact_duplicate_rows",
    "usage_monthly.conflicting_duplicate_month_rows", "support_tickets.exact_duplicate_rows",
}
AT_LEAST = {
    "customers.email_missing", "customers.name_case_inconsistent", "payments.amount_missing",
    "payments.amount_negative", "payments.method_inconsistent", "payments.status_inconsistent_case",
    "payments.subscription_id_missing", "payments.paid_at_in_future", "usage_monthly.data_used_missing",
    "usage_monthly.data_used_negative", "usage_monthly.active_days_over_31", "usage_monthly.month_not_first_of_month",
    "support_tickets.resolved_before_opened", "support_tickets.satisfaction_out_of_range",
    "support_tickets.category_inconsistent", "support_tickets.priority_inconsistent_case",
}

failures = 0


def report(ok: bool, label: str, detail: str = "") -> None:
    global failures
    failures += 0 if ok else 1
    print(f"[{'OK' if ok else 'FAIL'}] {label}" + (f" - {detail}" if detail else ""))


def check_data_quality(schema, uc, manifest) -> None:
    print("\n-- 1. Data-quality checks vs the answer key --")
    dq = run_named(schema, uc.sql.data_quality, uc.parameters)["data_quality"].set_index("check_id")["issues"]
    injected = manifest["injected_issues"]
    bad = []
    for key in sorted(EXACT):
        if int(dq[key]) != injected[key]:
            bad.append(f"{key}: SQL {int(dq[key])} vs injected {injected[key]}")
    report(not bad, f"{len(EXACT)} checks match the injected count exactly", "; ".join(bad))
    bad = [f"{k}: SQL {int(dq[k])} < injected {injected[k]}" for k in sorted(AT_LEAST) if int(dq[k]) < injected[k]]
    report(not bad, f"{len(AT_LEAST)} checks find at least the injected count", "; ".join(bad))
    d = injected["customers.duplicate_person_rows"]
    by_phone = int(dq["customers.duplicate_accounts_by_phone"])
    no_sub = int(dq["customers.accounts_without_subscription"])
    report(no_sub == d, "Accounts without a subscription = injected duplicate accounts", f"{no_sub} vs {d}")
    report(0.8 * d <= by_phone <= d, "Duplicates found by phone match (some have no phone to match)", f"{by_phone} of {d}")


def check_labels(schema, uc) -> None:
    print("\n-- 2. Churn labels vs the generator answer key --")
    key = pd.read_csv(DATA_DIR / "answer_key" / "labels_at_cutoff.csv")
    truth = key[key["active_at_cutoff"] == 1].set_index("customer_id")["churned_in_window"]
    ts = run_query(schema, "SELECT customer_id, churned FROM training_set").set_index("customer_id")["churned"]
    unknown_date = set(run_query(
        schema, "SELECT customer_id FROM clean_subscriptions WHERE is_latest AND status = 'cancelled' AND end_date IS NULL"
    )["customer_id"])
    only_sql = set(ts.index) - set(truth.index)
    only_truth = set(truth.index) - set(ts.index)
    common = list(set(ts.index) & set(truth.index))
    wrong = int((ts.loc[common] != truth.loc[common]).sum())
    report(not only_sql, "No customer in the SQL table that should not be there", f"{len(only_sql)} extra")
    report(only_truth <= unknown_date, "Missing customers are only those with an unknown cancellation date",
           f"{len(only_truth)} missing, {len(only_truth - unknown_date)} unexplained")
    report(wrong == 0, "Labels match the truth for every customer in both", f"{wrong} wrong out of {len(common):,}")
    print(f"       truth: {len(truth):,} active, {int(truth.sum())} cancel in window | "
          f"SQL: {len(ts):,} rows, {int(ts.sum())} cancel in window | excluded for unknown date: {len(only_truth)}")


def check_revenue_target(schema, uc) -> None:
    """Recompute revenue_next_30d in pandas from the RAW CSVs, without using any SQL view."""
    print("\n-- 2. Revenue target vs an independent pandas recomputation from the raw CSVs --")
    cutoff = pd.Timestamp(str(uc.parameters["cutoff_date"]))
    start = cutoff + pd.Timedelta(days=1)
    end = start + pd.Timedelta(days=int(uc.parameters["label_window_days"]))
    pay = pd.read_csv(DATA_DIR / "payments.csv", parse_dates=["paid_at"])
    subs = pd.read_csv(DATA_DIR / "subscriptions.csv")[["subscription_id", "plan_id"]]
    plans = pd.read_csv(DATA_DIR / "plans.csv")[["plan_id", "monthly_price_ghs"]]
    pay["status"] = pay["status"].str.strip().str.lower()
    pay["method"] = pay["method"].str.strip().str.lower()
    # same cleaning rules as clean_payments: drop exact duplicates, invalid amount -> plan price
    pay = pay.sort_values("payment_id")
    key = ["customer_id", "subscription_id", "paid_at", "amount_ghs", "method", "status"]
    pay = pay.drop_duplicates(subset=key, keep="first")
    pay = pay.merge(subs, on="subscription_id", how="left").merge(plans, on="plan_id", how="left")
    valid = pay["amount_ghs"].where(pay["amount_ghs"] > 0)
    pay["amount"] = valid.fillna(pay["monthly_price_ghs"])
    window = pay[(pay["status"] == "success") & (pay["paid_at"] >= start) & (pay["paid_at"] < end)]
    truth = window.groupby("customer_id")["amount"].sum()

    ts = run_query(schema, f"SELECT customer_id, {uc.target.name} AS y FROM training_set").set_index("customer_id")["y"]
    expected = truth.reindex(ts.index).fillna(0.0)
    diff = (ts - expected).abs()
    wrong = int((diff > 0.005).sum())
    report(wrong == 0, "Target matches the independent recomputation for every customer",
           f"{wrong} differ out of {len(ts):,}; max difference {diff.max():.2f}")
    print(f"       total next-30-day revenue: SQL {ts.sum():,.2f} | pandas {expected.sum():,.2f} | "
          f"customers paying nothing: {int((ts == 0).sum())}")


def check_generic_target(schema, uc) -> None:
    print("\n-- 2. Target sanity (no use-case specific answer key) --")
    ts = run_query(schema, f"SELECT {uc.target.name} AS y FROM training_set")["y"]
    report(ts.notna().all() and pd.api.types.is_numeric_dtype(ts), "Target is numeric and never missing")


TARGET_CHECKS = {"churn": check_labels, "revenue": check_revenue_target}


def check_no_leakage(schema, uc) -> None:
    print("\n-- 3. Leakage test: remove everything after the cutoff, features must not change --")
    cutoff = str(uc.parameters["cutoff_date"])
    body = next(iter(load_named_queries(uc.sql.training_table).values()))
    features = uc.feature_columns
    with connect(schema) as conn, psycopg.ClientCursor(conn) as cur:
        try:
            # Run the SAME query twice: once on all the data, once with the future removed.
            cur.execute(f"CREATE TEMP TABLE before_cut AS {body}", uc.parameters)
            p = {"c": cutoff}
            cur.execute("DELETE FROM payments WHERE paid_at >= %(c)s::date + 1", p)
            # compare on the MONTH (some raw rows are stored as the 15th of the month)
            cur.execute("DELETE FROM usage_monthly WHERE date_trunc('month', usage_month) > "
                        "(date_trunc('month', %(c)s::date + 1) - interval '1 month')", p)
            cur.execute("DELETE FROM support_tickets WHERE opened_at >= %(c)s::date + 1", p)
            cur.execute("UPDATE support_tickets SET resolved_at = NULL, satisfaction_score = NULL "
                        "WHERE resolved_at >= %(c)s::date + 1", p)
            cur.execute(f"CREATE TEMP TABLE after_cut AS {body}", uc.parameters)
            cur.execute("SELECT count(*) FROM before_cut a JOIN after_cut b USING (customer_id)")
            joined = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM before_cut")
            total = cur.fetchone()[0]
            diffs = {}
            for col in features:
                cur.execute(f'SELECT count(*) FROM before_cut a JOIN after_cut b USING (customer_id) '
                            f'WHERE a."{col}" IS DISTINCT FROM b."{col}"')
                n = cur.fetchone()[0]
                if n:
                    diffs[col] = n
        finally:
            conn.rollback()  # nothing is kept: the deletes are undone
    report(joined == total, "Same customers with and without post-cutoff data", f"{joined} of {total}")
    report(not diffs, f"All {len(features)} features identical without post-cutoff data",
           ", ".join(f"{c}: {n} rows differ" for c, n in diffs.items()))


def check_questions_and_kpis(schema, uc) -> None:
    print("\n-- 4/5. Config matches the SQL; every question and KPI runs --")
    summary = build_training_set(schema, uc)  # raises if the config promises a column the SQL lacks
    report(True, "Use-case config matches the training table columns", f"{summary['rows']:,} rows")
    results = run_named(schema, uc.sql.business_questions, uc.parameters)
    empty = [n for n, df in results.items() if df.empty]
    report(not empty, f"All {len(results)} business questions return rows", f"empty: {empty}" if empty else "")
    bad = []
    for kpi in uc.kpis:
        df = next(iter(run_named(schema, kpi.sql, uc.parameters).values()))
        if df.shape != (1, 1) or df.columns[0] != "value" or pd.isna(df.iloc[0, 0]):
            bad.append(kpi.id)
    report(not bad, f"All {len(uc.kpis)} KPI queries return a single 'value'", f"bad: {bad}" if bad else "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--schema", required=True)
    ap.add_argument("--use-case", default="churn")
    args = ap.parse_args()
    try:
        get_settings()
        uc = load_use_case(args.use_case)
        manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
        apply_sql_file(args.schema, PROJECT_ROOT / "db" / "clean_views.sql")
        build_training_set(args.schema, uc)
        check_data_quality(args.schema, uc, manifest)
        TARGET_CHECKS.get(uc.id, check_generic_target)(args.schema, uc)
        check_no_leakage(args.schema, uc)
        check_questions_and_kpis(args.schema, uc)
    except ValidationError as exc:
        print("Settings problem (check your .env):", "; ".join(f"{e['loc'][0]}: {e['msg']}" for e in exc.errors()))
        return 1
    except FileNotFoundError as exc:
        print(f"Missing file: {exc.filename}. Run: python -m scripts.generate_data --schema {args.schema} --create-schema --load")
        return 1
    except (UseCaseError, SqlError, LoaderError) as exc:
        print(f"Failed: {exc}")
        return 1
    except psycopg.Error as exc:
        print(f"Database error: {str(exc).strip().splitlines()[0]}")
        return 1
    print()
    print("All checks passed." if failures == 0 else f"{failures} check(s) failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
