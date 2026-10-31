"""Run a use case's SQL against a workspace.

The platform never contains business SQL itself. It only knows how to:
  * find the SQL files a use case points to (and refuse paths outside sql/),
  * split a file into named queries,
  * run a query inside a workspace schema, read-only, with a time limit,
  * materialise the training table and check it matches the use case config.

A .sql file can hold several queries, each introduced by a line like
    -- name: monthly_churn_trend
Text before the first marker (file header comments) is ignored.

Parameters from the use case config are passed as named values, e.g.
%(cutoff_date)s. The driver fills them in safely (they are escaped, never
pasted in as raw text). Because of that, a literal percent sign must not
appear inside a parametrised query.
"""
from __future__ import annotations

import logging
import re
from datetime import date
from decimal import Decimal
from pathlib import Path

import pandas as pd
import psycopg

from core.config import PROJECT_ROOT
from core.db import connect, validate_schema_name
from core.use_case import UseCase

logger = logging.getLogger(__name__)

SQL_ROOT = (PROJECT_ROOT / "sql").resolve()
TRAINING_TABLE = "training_set"  # fixed name, never taken from user input
_NAME_RE = re.compile(r"^--\s*name:\s*([a-z][a-z0-9_]*)\s*$", re.MULTILINE)


class SqlError(Exception):
    """A SQL file is missing/invalid, or a query failed."""


def resolve_sql_path(relative: str) -> Path:
    """Turn 'sql/churn/x.sql' into a real path, but only if it stays inside sql/."""
    path = (PROJECT_ROOT / relative).resolve()
    if not path.is_relative_to(SQL_ROOT):
        raise SqlError(f"SQL path must be inside the sql/ folder: {relative!r}")
    if not path.is_file():
        raise SqlError(f"SQL file not found: {relative}")
    return path


def load_named_queries(relative: str) -> dict[str, str]:
    """Split a .sql file into {query_name: sql_text} (trailing semicolon removed)."""
    text = resolve_sql_path(relative).read_text(encoding="utf-8")
    markers = list(_NAME_RE.finditer(text))
    if not markers:
        return {Path(relative).stem: text.strip().rstrip(";").strip()}
    queries: dict[str, str] = {}
    for i, marker in enumerate(markers):
        end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
        name = marker.group(1)
        if name in queries:
            raise SqlError(f"{relative}: query name '{name}' appears twice")
        queries[name] = text[marker.end():end].strip().rstrip(";").strip()
    return queries


def _decimals_to_float(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == object:
            first = next((v for v in df[col] if v is not None), None)
            if isinstance(first, Decimal):
                df[col] = pd.to_numeric(df[col].map(lambda v: None if v is None else float(v)))
    return df


def run_query(schema: str, query: str, params: dict | None = None, *, read_only: bool = True) -> pd.DataFrame:
    """Run one query inside a workspace schema and return a DataFrame."""
    validate_schema_name(schema)
    try:
        with connect(schema) as conn:
            conn.read_only = read_only
            # ClientCursor fills parameters in on the client (escaped), which also
            # lets us use them inside CREATE TABLE AS.
            with psycopg.ClientCursor(conn) as cur:
                cur.execute("SET statement_timeout = '60s'")
                cur.execute(query, params or None)
                if cur.description is None:
                    raise SqlError("The query returned no result set")
                columns = [d.name for d in cur.description]
                rows = cur.fetchall()
    except psycopg.errors.UndefinedTable as exc:
        raise SqlError(
            f"{str(exc).strip().splitlines()[0]}  -> run the clean layer (and the training-table step) first"
        ) from exc
    return _decimals_to_float(pd.DataFrame(rows, columns=columns))


def run_named(schema: str, relative_path: str, params: dict | None = None, name: str | None = None) -> dict[str, pd.DataFrame]:
    """Run every query in a file (or just `name`). Returns {query_name: DataFrame}."""
    queries = load_named_queries(relative_path)
    if name is not None:
        if name not in queries:
            raise SqlError(f"No query '{name}' in {relative_path}. Available: {', '.join(queries)}")
        queries = {name: queries[name]}
    return {n: run_query(schema, q, params) for n, q in queries.items()}


# ---- where a use case's queries come from: files (demo use cases) or generated SQL (uploaded data)
def training_body(use_case: UseCase) -> str:
    """The one SELECT that builds the training table."""
    if use_case.inline_sql is not None:
        return use_case.inline_sql.training_table.strip().rstrip(";")
    queries = load_named_queries(use_case.sql.training_table)
    if len(queries) != 1:
        raise SqlError("The training-table file must contain exactly one query")
    return next(iter(queries.values()))


def named_queries(use_case: UseCase, kind: str) -> dict[str, str]:
    """{name: SELECT} for 'data_quality' or 'business_questions' (empty when the use case has none)."""
    if kind not in ("data_quality", "business_questions"):
        raise ValueError(f"unknown query kind {kind!r}")
    if use_case.inline_sql is not None:
        inline = use_case.inline_sql
        if kind == "data_quality":
            return {"data_quality": inline.data_quality.strip().rstrip(";")} if inline.data_quality else {}
        return dict(inline.business_questions)
    path = getattr(use_case.sql, kind)
    return load_named_queries(path) if path else {}


def kpi_query(use_case: UseCase, kpi) -> str:
    if use_case.inline_sql is not None:
        return use_case.inline_sql.kpis[kpi.id].strip().rstrip(";")
    queries = load_named_queries(kpi.sql)
    return next(iter(queries.values()))


def run_queries(schema: str, queries: dict[str, str], params: dict | None = None) -> dict[str, pd.DataFrame]:
    """Run already-loaded queries (see named_queries) in a workspace. Returns {name: DataFrame}."""
    return {name: run_query(schema, query, params) for name, query in queries.items()}


def build_training_set(schema: str, use_case: UseCase) -> dict:
    """Materialise the use case's training query as the table `training_set` in the workspace.

    Checks the result has the columns the use case config promises
    (entity key, every feature, the target) and that a classification
    target contains only 0 and 1.
    """
    validate_schema_name(schema)
    body = training_body(use_case)

    with connect(schema) as conn, psycopg.ClientCursor(conn) as cur:
        cur.execute(f"DROP TABLE IF EXISTS {TRAINING_TABLE}")
        cur.execute(f"CREATE TABLE {TRAINING_TABLE} AS {body}", use_case.parameters or None)
        cur.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_schema = %s AND table_name = %s",
            (schema, TRAINING_TABLE),
        )
        have = {r[0] for r in cur.fetchall()}
        wanted = [use_case.entity.key, *use_case.feature_columns, use_case.target.name]
        missing = [c for c in wanted if c not in have]
        if missing:
            conn.rollback()
            raise SqlError(
                f"The training query does not return columns the use case config expects: {missing}. "
                f"Fix the SQL or the 'features'/'target' of use case '{use_case.id}'."
            )
        cur.execute(f"SELECT count(*) FROM {TRAINING_TABLE}")
        rows = cur.fetchone()[0]
        summary = {"rows": rows, "columns": sorted(have)}
        if use_case.problem_type == "classification":
            cur.execute(f"SELECT count(*) FILTER (WHERE {use_case.target.name} = 1), "
                        f"count(*) FILTER (WHERE {use_case.target.name} NOT IN (0, 1) OR {use_case.target.name} IS NULL) "
                        f"FROM {TRAINING_TABLE}")
            positives, bad = cur.fetchone()
            if bad:
                conn.rollback()
                raise SqlError(f"Target '{use_case.target.name}' must be 0 or 1 for classification; found {bad} other values")
            summary.update(positives=positives, positive_rate=round(positives / rows, 4) if rows else 0.0)
    logger.info("Built %s: %s", TRAINING_TABLE, {k: v for k, v in summary.items() if k != "columns"})
    return summary


def features_as_of(schema: str, use_case: UseCase, cutoff_date: str) -> pd.DataFrame:
    """Run the SAME training SQL with a different cutoff (used for drift monitoring).

    Nothing is written; the result is returned as a DataFrame.
    """
    try:
        cutoff = date.fromisoformat(str(cutoff_date))
    except ValueError as exc:
        raise SqlError(f"cutoff_date must be YYYY-MM-DD, got {cutoff_date!r}") from exc
    body = training_body(use_case)
    params = {**use_case.parameters, "cutoff_date": cutoff.isoformat()}
    return run_query(schema, body, params)


def read_training_set(schema: str) -> pd.DataFrame:
    """The materialised training table, ready for the ML pipeline (Step 5)."""
    return run_query(schema, f"SELECT * FROM {TRAINING_TABLE} ORDER BY 1")
