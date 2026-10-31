"""Load generated CSV files into a workspace schema.

Used by the data generator now, and by the workspace flow in Step 7 (when a
new workspace is created and given its data).

Two guarantees:
- Re-running replaces the data (TRUNCATE then load), so loading is repeatable.
- Everything happens in ONE transaction: either all six tables load, or none do.
"""
from __future__ import annotations

import logging
from pathlib import Path

import psycopg
from psycopg import sql

from core.db import connect, validate_schema_name

logger = logging.getLogger(__name__)

# Parents before children, because of the foreign keys.
TABLES = ["plans", "customers", "subscriptions", "payments", "usage_monthly", "support_tickets"]


class LoaderError(Exception):
    pass


def apply_sql_file(schema: str, path: Path) -> None:
    """Run a whole .sql file (e.g. the clean-layer views) inside a workspace schema."""
    validate_schema_name(schema)
    if not path.is_file():
        raise LoaderError(f"SQL file not found: {path}")
    with connect(schema) as conn:
        conn.execute(path.read_text(encoding="utf-8"))


def apply_sql_text(schema: str, text: str) -> None:
    """Run SQL text the platform generated itself (never text that came from a user) inside a workspace schema."""
    validate_schema_name(schema)
    with connect(schema) as conn:
        conn.execute(text)


def ensure_workspace_schema(schema: str, schema_sql: Path, clean_views_sql: Path | None = None) -> bool:
    """Create the schema and its tables if they do not exist yet, then (re)apply the clean-layer views.

    Returns True if it created the tables. The views use CREATE OR REPLACE, so applying them again is safe.
    """
    validate_schema_name(schema)
    if not schema_sql.is_file():
        raise LoaderError(f"Schema file not found: {schema_sql}")
    with connect() as conn:
        conn.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema)))
    created = False
    with connect(schema) as conn:
        existing = conn.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_schema = %s", (schema,)
        ).fetchone()[0]
        if existing == 0:
            conn.execute(schema_sql.read_text(encoding="utf-8"))
            logger.info("Created tables in schema %s", schema)
            created = True
    if clean_views_sql is not None:
        apply_sql_file(schema, clean_views_sql)
        logger.info("Applied clean-layer views in schema %s", schema)
    return created


def load_csv_dir(schema: str, directory: Path) -> dict[str, int]:
    """Replace the contents of the workspace's six tables with the CSVs in `directory`.

    Returns {table: row count in the database after loading}.
    """
    validate_schema_name(schema)
    for table in TABLES:
        if not (directory / f"{table}.csv").is_file():
            raise LoaderError(f"Missing {table}.csv in {directory}")

    counts: dict[str, int] = {}
    try:
        with connect(schema) as conn:
            conn.execute(
                sql.SQL("TRUNCATE {} RESTART IDENTITY CASCADE").format(
                    sql.SQL(", ").join(sql.Identifier(t) for t in TABLES)
                )
            )
            for table in TABLES:
                path = directory / f"{table}.csv"
                with path.open(encoding="utf-8") as f:
                    columns = f.readline().strip().split(",")
                copy_sql = sql.SQL("COPY {} ({}) FROM STDIN WITH (FORMAT csv, HEADER true)").format(
                    sql.Identifier(table), sql.SQL(", ").join(sql.Identifier(c) for c in columns)
                )
                with conn.cursor() as cur, cur.copy(copy_sql) as copy, path.open("rb") as f:
                    while chunk := f.read(1 << 16):
                        copy.write(chunk)
                counts[table] = conn.execute(sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(table))).fetchone()[0]
                logger.info("Loaded %s: %d rows", table, counts[table])
    except psycopg.errors.UndefinedTable as exc:
        raise LoaderError(
            f"Tables are missing in schema {schema}. Run with --create-schema, or load db/schema.sql into it first."
        ) from exc
    return counts
