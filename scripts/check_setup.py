"""Sanity check for the Step 2 setup.

Run from the project root:
    python -m scripts.check_setup

Prints OK / FAIL for each check and exits with a non-zero code if any fail.
"""
import sys

import psycopg
from pydantic import ValidationError

from core.config import get_settings
from core.db import connect, make_schema_name, validate_schema_name
from core.logging_setup import configure_logging
from core.use_case import UseCase, UseCaseError, list_use_cases

failures = 0


def report(ok: bool, label: str, detail: str = "") -> None:
    global failures
    if not ok:
        failures += 1
    print(f"[{'OK' if ok else 'FAIL'}] {label}" + (f" - {detail}" if detail else ""))


def first_line(exc: Exception) -> str:
    text = str(exc).strip()
    return text.splitlines()[0] if text else type(exc).__name__


# 1. Settings -------------------------------------------------------------
try:
    settings = get_settings()
except ValidationError as exc:
    # Print only which setting is wrong and why. Never print the values.
    report(False, "Settings load from .env")
    for err in exc.errors():
        print(f"       {'.'.join(str(p) for p in err['loc'])}: {err['msg']}")
    sys.exit(1)

configure_logging(settings.log_level)
report(True, "Settings load from .env")
for key, value in settings.safe_summary().items():
    print(f"       {key}: {value}")

# 2. Database connection --------------------------------------------------
try:
    with connect() as conn:
        version = conn.execute("SELECT version()").fetchone()[0]
    report(True, "Connect to PostgreSQL", version.split(",")[0])
except psycopg.Error as exc:
    report(False, "Connect to PostgreSQL", first_line(exc))
    print("       Is the pg-npontu container running?  (docker ps)")

# 3. The Step 1 workspace schema -----------------------------------------
RAW_TABLES = ["plans", "customers", "subscriptions", "payments", "usage_monthly", "support_tickets"]
try:
    with connect("ws_demo") as conn:
        count = conn.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_schema = 'ws_demo' AND table_type = 'BASE TABLE' AND table_name = ANY(%s)",
            (RAW_TABLES,),
        ).fetchone()[0]
    report(count == 6, "Workspace schema ws_demo has the 6 raw tables", f"found {count} of 6")
except psycopg.Error as exc:
    report(False, "Workspace schema ws_demo has the 6 raw tables", first_line(exc))

# 4. Use cases load and validate -----------------------------------------
use_cases = []
try:
    use_cases = list_use_cases()
    names = ", ".join(f"{u.id} ({u.problem_type})" for u in use_cases)
    report(len(use_cases) > 0, "Use cases load and validate", names)
except UseCaseError as exc:
    report(False, "Use cases load and validate", str(exc))

# 5. Validation really rejects bad input ---------------------------------
if use_cases:
    bad = use_cases[0].model_dump()
    bad["models"]["baseline"] = "dummy_regressor"  # regression model on a classification problem
    try:
        UseCase.model_validate(bad)
        report(False, "Bad use-case config is rejected", "it was accepted!")
    except ValidationError:
        report(True, "Bad use-case config is rejected")

    leaky = use_cases[0].model_dump()
    leaky["features"]["numeric"].append(leaky["target"]["name"])  # target used as a feature
    try:
        UseCase.model_validate(leaky)
        report(False, "Target-as-feature (leakage) is rejected", "it was accepted!")
    except ValidationError:
        report(True, "Target-as-feature (leakage) is rejected")

# 6. Schema-name safety ---------------------------------------------------
try:
    validate_schema_name("ws_demo; DROP TABLE customers")
    report(False, "Injection-style schema name is rejected", "it was accepted!")
except ValueError:
    report(True, "Injection-style schema name is rejected")

try:
    validate_schema_name(make_schema_name())
    report(True, "Generated workspace schema names pass validation")
except ValueError as exc:
    report(False, "Generated workspace schema names pass validation", str(exc))

# 7. Snwolley (informational only; no call is made here, the API is rate limited) -----
if settings.snwolley_configured:
    print("[INFO] Snwolley API key is set; AI insights will be requested during each run")
else:
    print("[INFO] SNWOLLEY_API_KEY not set - runs will show the labelled template summary instead of AI insights")

print()
print("All checks passed." if failures == 0 else f"{failures} check(s) failed.")
sys.exit(1 if failures else 0)
