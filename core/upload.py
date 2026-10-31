"""A business's own data: read an uploaded CSV, describe it, and load it into the workspace.

THE PROBLEM THIS SOLVES
    The demo business has six hand-designed tables. A real business arrives with ONE file
    from its own system (an export of customers, orders, loans ...) whose columns the
    platform has never seen, whose names may contain quotes and spaces, and which may be
    huge, mis-encoded or hostile. Before any of that reaches SQL the platform must:
      1. refuse what it cannot safely handle (size, encoding, shape),
      2. give every column a safe, predictable name (so a column called  a"; DROP TABLE x;--
         can never become SQL),
      3. work out what each column is (number, date, category, free text) so the business
         can choose the target and the inputs from a list instead of writing SQL,
      4. load it, untouched, as text into the workspace's private schema. Cleaning and typing
         happen later, in SQL views (core/custom_usecase.py), so the raw upload is never modified.

WHAT IS STORED
    Only the table `upload_raw` in the workspace's own schema (every column as text, plus a
    row_id). The original file is not kept on disk.

LIMITS (on purpose, for a prototype)
    One flat file, one row per record (customer, order, loan ...). Up to 10 MB, 200,000 rows,
    80 columns. UTF-8 text. See the README for what a multi-table upload would need.
"""
from __future__ import annotations

import csv
import io
import re

import pandas as pd
from psycopg import sql

from core.ai.facts import clean_text
from core.db import connect, validate_schema_name

MAX_BYTES = 10 * 1024 * 1024
MAX_ROWS = 200_000
MIN_ROWS = 50
MAX_COLS = 80
MAX_NAME = 40
UPLOAD_TABLE = "upload_raw"

# Names the platform's own generated SQL uses; an uploaded column may not take them.
RESERVED = {"entity_id", "event_date", "row_id", "rn", "is_duplicate", "typed", "cleaned", "ranked"}

# Values that mean "nothing here" in exported spreadsheets. Matched after trim + lower-case.
MISSING_TOKENS = ("", "na", "n/a", "null", "none", "nan", "-", "?")

_NUMERIC = re.compile(r"^-?[0-9]+(\.[0-9]+)?$")
_DATE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}")
_NAME_OK = re.compile(r"^[a-z][a-z0-9_]{0,49}$")

CATEGORY_MAX_DISTINCT = 50
TOP_VALUES = 30


class UploadError(Exception):
    """The file cannot be used. The message is written for the person who uploaded it."""


# ---------------------------------------------------------------- column names
def sanitise_columns(names: list[str]) -> list[str]:
    """Plain lower-case names that are safe to use as SQL identifiers: [a-z][a-z0-9_]{0,39}, all different."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in names:
        name = re.sub(r"[^a-z0-9]+", "_", str(raw).strip().lower()).strip("_")
        if not name:
            name = "column"
        if not name[0].isalpha():
            name = "c_" + name
        name = name[:MAX_NAME].rstrip("_") or "column"
        if name in RESERVED:
            name += "_col"
        base, n = name, 1
        while name in seen:
            n += 1
            name = f"{base[: MAX_NAME - len(str(n)) - 1]}_{n}"
        seen.add(name)
        out.append(name)
    return out


def is_safe_identifier(name: str) -> bool:
    return isinstance(name, str) and _NAME_OK.fullmatch(name) is not None


# ---------------------------------------------------------------- reading the file
def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")  # utf-8-sig also drops the byte-order mark Excel adds
    except UnicodeDecodeError as exc:
        raise UploadError(
            "The file is not UTF-8 text. In Excel use 'Save as > CSV UTF-8', or export as UTF-8 from your system."
        ) from exc


def _delimiter(text: str) -> str:
    head = text[:8192]
    try:
        return csv.Sniffer().sniff(head, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def _check_rectangular(text: str, sep: str) -> None:
    """Every row must have as many values as the header.

    pandas quietly treats the extra leading values of a too-wide row as an index, which would shift the
    columns of the whole file. A shifted file is worse than a refused one, so refuse it and say where.
    """
    reader = csv.reader(io.StringIO(text), delimiter=sep)
    width = len(next(reader, []))
    for line_no, row in enumerate(reader, start=2):
        if row and len(row) != width:  # an empty list is a blank line, which is fine
            raise UploadError(
                f"The file could not be read as a table: row {line_no} has {len(row)} values but the header has {width}. "
                "Check for a stray comma or an unquoted comma inside a value."
            )


def parse_csv(data: bytes) -> tuple[pd.DataFrame, list[str]]:
    """Bytes -> (every value as text, original header names). Raises UploadError with a readable reason."""
    if not data or not data.strip():
        raise UploadError("The file is empty.")
    if len(data) > MAX_BYTES:
        raise UploadError(f"The file is larger than {MAX_BYTES // (1024 * 1024)} MB.")
    if b"\x00" in data[:4096]:
        raise UploadError("This does not look like a CSV text file (it contains binary data).")
    text = _decode(data)
    sep = _delimiter(text)
    _check_rectangular(text, sep)
    try:
        df = pd.read_csv(
            io.StringIO(text), sep=sep, dtype=str, keep_default_na=False, na_filter=False,
            on_bad_lines="error", skipinitialspace=True,
        )
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise UploadError(f"The file could not be read as a table: {str(exc).splitlines()[0][:200]}") from exc
    if df.shape[1] < 2:
        raise UploadError("The file needs at least two columns (a target to predict and something to predict it from).")
    if df.shape[1] > MAX_COLS:
        raise UploadError(f"The file has {df.shape[1]} columns; the limit is {MAX_COLS}.")
    if len(df) > MAX_ROWS:
        raise UploadError(f"The file has {len(df):,} rows; the limit is {MAX_ROWS:,}.")
    if len(df) < MIN_ROWS:
        raise UploadError(f"The file has only {len(df)} rows; at least {MIN_ROWS} are needed to train and test a model.")
    original = [str(c) for c in df.columns]
    df.columns = sanitise_columns(original)
    return df, original


# ---------------------------------------------------------------- profiling
def _is_missing(series: pd.Series) -> pd.Series:
    return series.str.strip().str.lower().isin(MISSING_TOKENS)


def _kind(values: pd.Series, n_rows: int) -> str:
    """'numeric', 'date', 'categorical', 'identifier', 'text' or 'empty', judged on the non-missing values."""
    n = len(values)
    if n == 0:
        return "empty"
    v = values.str.strip()
    if v.str.fullmatch(_NUMERIC.pattern).mean() >= 0.95:
        return "numeric"
    if v.str.match(_DATE.pattern).mean() >= 0.95:
        return "date"
    distinct = v.str.lower().nunique()
    if distinct <= CATEGORY_MAX_DISTINCT and distinct <= max(2, 0.5 * n):
        return "categorical"
    return "identifier" if distinct >= 0.9 * n else "text"


def profile_frame(df: pd.DataFrame, original_names: list[str]) -> dict:
    """What the business sees before choosing a target: per column its type, gaps and common values."""
    n_rows = len(df)
    columns = []
    for name, original in zip(df.columns, original_names):
        s = df[name]
        missing = _is_missing(s)
        present = s[~missing].str.strip()
        kind = _kind(present, n_rows)
        distinct = int(present.str.lower().nunique())
        col = {
            "name": name,
            "original": clean_text(original, 80),
            "kind": kind,
            "missing": int(missing.sum()),
            "missing_pct": round(100.0 * missing.mean(), 1),
            "distinct": distinct,
            "samples": [clean_text(v, 40) for v in present.drop_duplicates().head(4)],
            # features must carry information: a number or category with at least two values
            "usable_as_feature": kind in ("numeric", "categorical") and distinct >= 2,
        }
        if kind in ("categorical", "numeric") and distinct <= CATEGORY_MAX_DISTINCT:
            counts = present.str.lower().value_counts().head(TOP_VALUES)
            col["top_values"] = [{"value": clean_text(k, 40), "count": int(c)} for k, c in counts.items()]
        if kind == "numeric":
            nums = pd.to_numeric(present, errors="coerce")
            col["min"], col["max"], col["mean"] = (round(float(f(nums)), 4) for f in (pd.Series.min, pd.Series.max, pd.Series.mean))
        columns.append(col)
    return {"rows": n_rows, "columns": columns, "suggested": _suggest(columns, n_rows)}


_TARGET_HINTS = ("churn", "target", "label", "outcome", "default", "fraud", "converted", "cancel", "status")


def _suggest(columns: list[dict], n_rows: int) -> dict:
    """A starting point for the mapping form. The business can change every choice."""
    ident = next((c["name"] for c in columns
                  if re.search(r"(^|_)(id|uuid|key|number|no)$", c["name"]) and c["distinct"] >= 0.9 * (n_rows - c["missing"])), None)
    target = next((c["name"] for c in columns if any(h in c["name"] for h in _TARGET_HINTS) and c["name"] != ident), None)
    date = next((c["name"] for c in columns if c["kind"] == "date"), None)
    return {"id_column": ident, "target_column": target, "date_column": date}


# ---------------------------------------------------------------- loading
def load_upload(schema: str, df: pd.DataFrame) -> int:
    """Replace the workspace's uploaded table with this data (one transaction). Returns the row count."""
    validate_schema_name(schema)
    names = list(df.columns)
    if not all(is_safe_identifier(n) for n in names):
        raise UploadError("Column names were not made safe; refusing to load.")  # cannot happen via parse_csv
    cols_sql = sql.SQL(", ").join(sql.SQL("{} text").format(sql.Identifier(n)) for n in names)
    with connect(schema) as conn:
        conn.execute(sql.SQL("DROP VIEW IF EXISTS clean_upload CASCADE"))
        conn.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(UPLOAD_TABLE)))
        conn.execute(
            sql.SQL("CREATE TABLE {} (row_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY, {})").format(
                sql.Identifier(UPLOAD_TABLE), cols_sql
            )
        )
        copy_sql = sql.SQL("COPY {} ({}) FROM STDIN").format(
            sql.Identifier(UPLOAD_TABLE), sql.SQL(", ").join(sql.Identifier(n) for n in names)
        )
        with conn.cursor() as cur, cur.copy(copy_sql) as copy:
            for row in df.itertuples(index=False, name=None):
                copy.write_row(row)
        n = conn.execute(sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(UPLOAD_TABLE))).fetchone()[0]
    return int(n)
