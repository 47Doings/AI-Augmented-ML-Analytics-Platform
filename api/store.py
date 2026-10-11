"""Reads and writes the platform's own tables (users, workspaces, runs).

All SQL here uses %s placeholders: user-supplied values are never pasted into
a query string. The only identifier built dynamically is a workspace schema
name, and that is validated and quoted (see delete_account).
"""
from __future__ import annotations

import logging
import shutil

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from core.config import PROJECT_ROOT
from core.db import connect, make_schema_name, validate_schema_name
from core.loader import ensure_workspace_schema

logger = logging.getLogger(__name__)

PLATFORM_SQL = PROJECT_ROOT / "db" / "platform.sql"
SCHEMA_SQL = PROJECT_ROOT / "db" / "schema.sql"
VIEWS_SQL = PROJECT_ROOT / "db" / "clean_views.sql"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "workspaces"
DATA_ROOT = PROJECT_ROOT / "data" / "workspaces"


class EmailTaken(Exception):
    pass


class RunInProgress(Exception):
    pass


def init_platform() -> None:
    """Create the platform tables if needed, and fail any run left 'running' by a crashed server."""
    with connect() as conn:
        conn.execute(PLATFORM_SQL.read_text(encoding="utf-8"))
        n = conn.execute(
            "UPDATE platform.runs SET status = 'failed', finished_at = now(), "
            "error = 'The server restarted while this run was in progress.' WHERE status = 'running'"
        ).rowcount
    if n:
        logger.warning("Marked %d interrupted run(s) as failed", n)


# ---------------------------------------------------------------- accounts
def create_account(email: str, password_hash: str, business_name: str) -> int:
    """Create the user, their workspace row and the workspace's database schema. Returns user_id."""
    schema = make_schema_name()
    with connect() as conn:
        try:
            user_id = conn.execute(
                "INSERT INTO platform.users (email, password_hash) VALUES (%s, %s) RETURNING user_id",
                (email, password_hash),
            ).fetchone()[0]
        except psycopg.errors.UniqueViolation as exc:
            raise EmailTaken() from exc
        conn.execute(
            "INSERT INTO platform.workspaces (user_id, business_name, schema_name) VALUES (%s, %s, %s)",
            (user_id, business_name, schema),
        )
    try:
        ensure_workspace_schema(schema, SCHEMA_SQL, VIEWS_SQL)
    except Exception:
        # do not leave a user with a workspace that has no tables behind it
        with connect() as conn:
            conn.execute("DELETE FROM platform.users WHERE user_id = %s", (user_id,))
        raise
    logger.info("Created account user_id=%s workspace=%s", user_id, schema)
    return user_id


def delete_account(user_id: int) -> None:
    """Remove a user, their workspace schema (all their data) and their files."""
    with connect() as conn:
        row = conn.execute("SELECT schema_name FROM platform.workspaces WHERE user_id = %s", (user_id,)).fetchone()
        if row:
            validate_schema_name(row[0])
            conn.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(row[0])))
        conn.execute("DELETE FROM platform.users WHERE user_id = %s", (user_id,))
    if row:
        for root in (OUTPUT_ROOT, DATA_ROOT):
            shutil.rmtree(root / row[0], ignore_errors=True)


def get_user_by_email(email: str) -> dict | None:
    with connect() as conn, conn.cursor(row_factory=dict_row) as cur:
        return cur.execute("SELECT user_id, email, password_hash FROM platform.users WHERE email = %s", (email,)).fetchone()


def get_context(user_id: int) -> tuple[dict, dict] | None:
    """(user, workspace) for a user id, or None if the user no longer exists."""
    with connect() as conn, conn.cursor(row_factory=dict_row) as cur:
        row = cur.execute(
            "SELECT u.user_id, u.email, w.workspace_id, w.business_name, w.schema_name, w.use_case_id, "
            "       w.data_rows, w.data_loaded_at "
            "FROM platform.users u JOIN platform.workspaces w ON w.user_id = u.user_id WHERE u.user_id = %s",
            (user_id,),
        ).fetchone()
    if not row:
        return None
    user = {"user_id": row["user_id"], "email": row["email"]}
    ws = {k: row[k] for k in ("workspace_id", "business_name", "schema_name", "use_case_id", "data_rows", "data_loaded_at")}
    return user, ws


# ---------------------------------------------------------------- workspace state
def set_use_case(workspace_id: int, use_case_id: str) -> None:
    with connect() as conn:
        conn.execute("UPDATE platform.workspaces SET use_case_id = %s WHERE workspace_id = %s", (use_case_id, workspace_id))


def set_data_loaded(workspace_id: int, counts: dict[str, int]) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE platform.workspaces SET data_rows = %s, data_loaded_at = now() WHERE workspace_id = %s",
            (Jsonb(counts), workspace_id),
        )


# ---------------------------------------------------------------- runs
def start_run(workspace_id: int, use_case_id: str, steps: list[dict]) -> int:
    try:
        with connect() as conn:
            return conn.execute(
                "INSERT INTO platform.runs (workspace_id, use_case_id, status, steps) "
                "VALUES (%s, %s, 'running', %s) RETURNING run_id",
                (workspace_id, use_case_id, Jsonb(steps)),
            ).fetchone()[0]
    except psycopg.errors.UniqueViolation as exc:  # the partial unique index: one running run per workspace
        raise RunInProgress() from exc


def update_steps(run_id: int, steps: list[dict]) -> None:
    with connect() as conn:
        conn.execute("UPDATE platform.runs SET steps = %s WHERE run_id = %s", (Jsonb(steps), run_id))


def finish_run(run_id: int, status: str, error: str | None = None) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE platform.runs SET status = %s, error = %s, finished_at = now() WHERE run_id = %s",
            (status, error, run_id),
        )


def latest_run(workspace_id: int) -> dict | None:
    with connect() as conn, conn.cursor(row_factory=dict_row) as cur:
        return cur.execute(
            "SELECT run_id, use_case_id, status, steps, error, started_at, finished_at "
            "FROM platform.runs WHERE workspace_id = %s ORDER BY run_id DESC LIMIT 1",
            (workspace_id,),
        ).fetchone()
