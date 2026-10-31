"""From a business's column choices to a complete use case, with generated SQL.

THE PROBLEM THIS SOLVES
    For the demo business a developer wrote the SQL. A business that uploads its own file
    cannot (and should not have to). It says, in a form: "this column is the outcome, these
    columns are the inputs, this one identifies a row". The platform turns that into the same
    thing a developer would have written by hand: a use case (target, features, models,
    metrics, KPIs, AI context) plus SQL (clean layer, data-quality checks, training table,
    business questions, KPIs). After that, the SAME pipeline runs; nothing downstream knows
    the use case came from a form.

HOW A HOSTILE FORM CANNOT BECOME HOSTILE SQL
    The browser sends only column names and one category value. The server
      * accepts a column name only if it is one of the names it generated itself at upload time
        (sanitised to [a-z][a-z0-9_]*, checked again here), and quotes it;
      * accepts the "positive" outcome value only if it is one of the values the profile saw in
        that column, and writes it as an escaped string literal;
      * never accepts SQL from the browser, and rebuilds the SQL on every run.

WHAT THE GENERATED SQL SHOWS
    window functions (ROW_NUMBER for duplicates, RANK, NTILE quartiles, LAG for month-on-month
    change), CTEs, conditional aggregation (FILTER), percentile_cont for outliers, and
    UNION ALL to turn many checks into one result table.

WHAT IT CANNOT DO (said plainly)
    A flat file has no "as of" date, so the platform cannot prove features were known before the
    outcome the way the demo's SQL does. Instead it warns about a column that predicts the outcome
    almost perfectly (leakage_checks) and the README says the business must confirm each input
    is known before the outcome happens.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from core.ai.facts import clean_text
from core.upload import MISSING_TOKENS, UPLOAD_TABLE, is_safe_identifier
from core.use_case import (
    AiContext, Entity, Evaluation, FallbackRecommendation, InlineSql, Kpi, Metrics, Models, Target, UseCase,
    Features,
)

CUSTOM_ID = "custom"
MIN_PER_OUTCOME = 30  # positives and negatives needed for a classification target
MAX_FEATURES = 40
MAX_SEGMENT_QUESTIONS = 3
MAX_QUARTILE_QUESTIONS = 2
MAX_OUTLIER_CHECKS = 10
SEGMENT_MAX_DISTINCT = 30


class MappingError(Exception):
    """The column choices do not make a usable analysis. The message says what to change."""


# ---------------------------------------------------------------- what the form sends
class Mapping(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=2, max_length=80)
    business_question: str = Field(min_length=5, max_length=300)
    problem_type: Literal["classification", "regression"]
    target_column: str = Field(min_length=1, max_length=50)
    positive_value: str | None = Field(default=None, max_length=60)  # classification: the outcome that counts as "yes"
    id_column: str | None = Field(default=None, max_length=50)
    date_column: str | None = Field(default=None, max_length=50)
    features: list[str] = Field(min_length=1, max_length=MAX_FEATURES)

    @field_validator("name", "business_question")
    @classmethod
    def _plain_text(cls, v: str) -> str:
        v = clean_text(v, 300)
        if len(v) < 2:
            raise ValueError("too short")
        return v

    @field_validator("positive_value")
    @classmethod
    def _positive(cls, v: str | None) -> str | None:
        return None if v is None else v.strip() or None

    @field_validator("target_column", "id_column", "date_column")
    @classmethod
    def _identifier(cls, v: str | None) -> str | None:
        if v is not None and not is_safe_identifier(v):
            raise ValueError("not a valid column name")
        return v

    @field_validator("features")
    @classmethod
    def _features(cls, v: list[str]) -> list[str]:
        if any(not is_safe_identifier(c) for c in v):
            raise ValueError("a feature is not a valid column name")
        if len(set(v)) != len(v):
            raise ValueError("a feature is listed twice")
        return v


# ---------------------------------------------------------------- SQL building blocks
def q(name: str) -> str:
    """A column name as a quoted identifier. Refuses anything the upload step did not produce."""
    if not is_safe_identifier(name):
        raise MappingError(f"Unsafe column name: {name!r}")
    return f'"{name}"'


def lit(value: str) -> str:
    """A text value as an escaped SQL string literal."""
    if "\x00" in value:
        raise MappingError("Values may not contain NUL characters")
    return "'" + value.replace("'", "''") + "'"


_MISSING_SQL = ", ".join(lit(t) for t in MISSING_TOKENS)
_NUMBER_RE = r"'^-?[0-9]+(\.[0-9]+)?$'"  # a regular expression inside a SQL string; the backslash is literal in PostgreSQL


def _missing_cond(col: str) -> str:
    return f"lower(trim({q(col)})) IN ({_MISSING_SQL})"


def _clean_text(col: str) -> str:
    return f"CASE WHEN {_missing_cond(col)} THEN NULL ELSE trim({q(col)}) END"


def _numeric_of(expr: str) -> str:
    return f"CASE WHEN {expr} ~ {_NUMBER_RE} THEN ({expr})::numeric END"


def _raw_numeric(col: str) -> str:
    return _numeric_of(_clean_text(col))


_SAFE_DATE_FN = """
CREATE OR REPLACE FUNCTION safe_date(t text) RETURNS date LANGUAGE plpgsql IMMUTABLE AS $fn$
BEGIN
    IF t ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}' THEN
        RETURN left(t, 10)::date;
    END IF;
    RETURN NULL;
EXCEPTION WHEN others THEN
    RETURN NULL;   -- something like 2024-02-31
END
$fn$;
"""


# ---------------------------------------------------------------- validating the form against the profile
def _display(col: dict) -> str:
    return col.get("original") or col["name"]


def _split_features(m: Mapping, cols: dict[str, dict]) -> tuple[list[str], list[str]]:
    numeric, categorical = [], []
    for f in m.features:
        c = cols[f]
        (numeric if c["kind"] == "numeric" else categorical).append(f)
    return numeric, categorical


def validate_mapping(m: Mapping, profile: dict) -> dict[str, dict]:
    """Check the choices against what the upload contains. Returns {column name: profile entry}."""
    cols = {c["name"]: c for c in profile["columns"]}
    for label, name in [("target", m.target_column), ("id column", m.id_column), ("date column", m.date_column), *[("input", f) for f in m.features]]:
        if name is not None and name not in cols:
            raise MappingError(f"The {label} '{name}' is not a column of the uploaded file.")
    roles = [m.target_column, m.id_column, m.date_column]
    chosen = [r for r in roles if r is not None]
    if len(set(chosen)) != len(chosen):
        raise MappingError("The target, id and date must be three different columns.")
    for f in m.features:
        if f in chosen:
            raise MappingError(f"'{_display(cols[f])}' cannot be both an input and the target, id or date.")
        c = cols[f]
        if not c["usable_as_feature"]:
            raise MappingError(
                f"'{_display(c)}' cannot be used as an input: it looks like {c['kind']} "
                "(inputs must be numbers or categories with at least two different values, not free text or unique ids)."
            )

    target = cols[m.target_column]
    n_present = profile["rows"] - target["missing"]
    if n_present < 50:
        raise MappingError(f"The target '{_display(target)}' has a value in only {n_present} rows; at least 50 are needed.")
    if m.problem_type == "classification":
        if "top_values" not in target or target["kind"] not in ("categorical", "numeric"):
            raise MappingError(f"'{_display(target)}' has too many different values to be a yes/no outcome. Pick a column with a few values, or choose 'regression'.")
        if not m.positive_value:
            raise MappingError("Choose which value of the target counts as the outcome to predict (for example 'yes').")
        counts = {v["value"].lower(): v["count"] for v in target["top_values"]}
        pos = counts.get(m.positive_value.strip().lower())
        if pos is None:
            raise MappingError(f"'{m.positive_value}' is not a value that appears in '{_display(target)}'.")
        neg = n_present - pos
        if pos < MIN_PER_OUTCOME or neg < MIN_PER_OUTCOME:
            raise MappingError(
                f"Only {min(pos, neg)} rows have the rarer outcome; at least {MIN_PER_OUTCOME} of each are needed to train and test a model."
            )
    else:
        if target["kind"] != "numeric":
            raise MappingError(f"'{_display(target)}' is not a number, so it cannot be predicted as an amount. Choose 'classification' for categories.")
        if target["distinct"] < 10:
            raise MappingError(f"'{_display(target)}' has only {target['distinct']} different values; use 'classification' for outcomes with few values.")
    if m.date_column is not None and cols[m.date_column]["kind"] != "date":
        raise MappingError(f"'{_display(cols[m.date_column])}' does not look like a date (YYYY-MM-DD).")
    return cols


# ---------------------------------------------------------------- generated SQL
def clean_layer_sql(m: Mapping, cols: dict[str, dict]) -> str:
    """The clean view: text -> numbers/categories, missing tokens -> NULL, duplicates flagged (never deleted)."""
    numeric, categorical = _split_features(m, cols)
    used = [m.target_column, *m.features] + ([m.id_column] if m.id_column else []) + ([m.date_column] if m.date_column else [])
    cleaned = ",\n           ".join(f"{_clean_text(c)} AS {q(c)}" for c in used)

    entity = "'row_' || row_id::text" if not m.id_column else f"COALESCE({q(m.id_column)}, 'row_' || row_id::text)"
    if m.problem_type == "classification":
        target = (f"CASE WHEN {q(m.target_column)} IS NULL THEN NULL "
                  f"WHEN lower({q(m.target_column)}) = lower({lit(m.positive_value or '')}) THEN 1 ELSE 0 END AS {q(m.target_column)}")
    else:
        target = f"{_numeric_of(q(m.target_column))} AS {q(m.target_column)}"
    typed_cols = [f"{entity} AS entity_id", target]
    typed_cols += [f"{_numeric_of(q(c))} AS {q(c)}" for c in numeric]
    typed_cols += [f"lower({q(c)}) AS {q(c)}" for c in categorical]
    if m.date_column:
        typed_cols.append(f"safe_date({q(m.date_column)}) AS event_date")
    typed = ",\n           ".join(typed_cols)

    out_cols = ["entity_id", q(m.target_column), *[q(f) for f in m.features]] + (["event_date"] if m.date_column else [])
    return f"""{_SAFE_DATE_FN}
DROP VIEW IF EXISTS clean_upload CASCADE;
CREATE VIEW clean_upload AS
WITH cleaned AS (
    SELECT row_id,
           {cleaned}
    FROM {UPLOAD_TABLE}
),
typed AS (
    SELECT row_id,
           {typed}
    FROM cleaned
),
ranked AS (
    SELECT typed.*, ROW_NUMBER() OVER (PARTITION BY entity_id ORDER BY row_id) AS rn
    FROM typed
)
SELECT {", ".join(out_cols)}, (rn > 1) AS is_duplicate
FROM ranked;
"""


def training_sql(m: Mapping) -> str:
    cols = ["entity_id", *[q(f) for f in m.features], q(m.target_column)] + (["event_date"] if m.date_column else [])
    return (f"SELECT {', '.join(cols)}\nFROM clean_upload\n"
            f"WHERE NOT is_duplicate AND {q(m.target_column)} IS NOT NULL\nORDER BY entity_id")


def data_quality_sql(m: Mapping, cols: dict[str, dict]) -> str:
    """One row per check, in the same shape as the demo's data-quality query."""
    numeric, categorical = _split_features(m, cols)
    blocks: list[str] = []

    def block(check_id: str, dimension: str, issues: str, total: str, source: str, handling: str, hi: float, mid: float) -> None:
        blocks.append(
            f"SELECT {lit(UPLOAD_TABLE)} AS table_name, {lit(check_id)} AS check_id, {lit(dimension)} AS dimension, "
            f"{issues} AS issues, {total} AS total, {lit(handling)} AS handling, {hi}::numeric AS hi, {mid}::numeric AS mid "
            f"FROM {source}"
        )

    for c in m.features:
        block(f"missing:{c}", "completeness", f"count(*) FILTER (WHERE {_missing_cond(c)})", "count(*)", UPLOAD_TABLE,
              "Left empty; the model fills numbers with the median and categories with 'missing'", 30, 5)
    block(f"missing:{m.target_column}", "completeness", f"count(*) FILTER (WHERE {_missing_cond(m.target_column)})", "count(*)", UPLOAD_TABLE,
          "Rows without a target value are left out of training", 10, 1)
    if m.id_column:
        block(f"missing:{m.id_column}", "completeness", f"count(*) FILTER (WHERE {_missing_cond(m.id_column)})", "count(*)", UPLOAD_TABLE,
              "A row number is used as the id", 5, 1)
        block(f"duplicate_ids:{m.id_column}", "uniqueness", "count(*) FILTER (WHERE is_duplicate)", "count(*)", "clean_upload",
              "Only the first row for each id is used; the raw upload is not changed", 5, 0.1)
    for c in numeric:
        block(f"non_numeric:{c}", "validity", f"count(*) FILTER (WHERE NOT ({_missing_cond(c)}) AND NOT (trim({q(c)}) ~ {_NUMBER_RE}))",
              "count(*)", UPLOAD_TABLE, "Treated as missing (the value is not a plain number)", 10, 1)
    if m.problem_type == "regression":
        block(f"non_numeric:{m.target_column}", "validity",
              f"count(*) FILTER (WHERE NOT ({_missing_cond(m.target_column)}) AND NOT (trim({q(m.target_column)}) ~ {_NUMBER_RE}))",
              "count(*)", UPLOAD_TABLE, "Treated as missing; the row is left out of training", 5, 0.5)
    if m.date_column:
        block(f"bad_date:{m.date_column}", "validity",
              f"count(*) FILTER (WHERE NOT ({_missing_cond(m.date_column)}) AND safe_date(trim({q(m.date_column)})) IS NULL)",
              "count(*)", UPLOAD_TABLE, "Treated as missing", 10, 1)
    for c in categorical:
        # rows that use a minority spelling of a value ("Yes" / "yes " / "YES"): window function over the spelling counts
        blocks.append(
            f"SELECT {lit(UPLOAD_TABLE)}, {lit('spelling_variants:' + c)}, 'consistency', "
            f"COALESCE(sum(cnt) FILTER (WHERE rk > 1), 0), (SELECT count(*) FROM {UPLOAD_TABLE}), "
            f"{lit('Lower-cased and trimmed so spellings merge')}, 20::numeric, 5::numeric "
            f"FROM (SELECT cnt, ROW_NUMBER() OVER (PARTITION BY k ORDER BY cnt DESC) AS rk "
            f"      FROM (SELECT lower(trim({q(c)})) AS k, trim({q(c)}) AS v, count(*) AS cnt FROM {UPLOAD_TABLE} "
            f"            WHERE NOT ({_missing_cond(c)}) GROUP BY 1, 2) g) r"
        )
    num_for_outliers = numeric + ([m.target_column] if m.problem_type == "regression" else [])
    for c in num_for_outliers[:MAX_OUTLIER_CHECKS]:
        v = _raw_numeric(c)
        blocks.append(
            f"SELECT {lit(UPLOAD_TABLE)}, {lit('outliers:' + c)}, 'validity', "
            f"count(*) FILTER (WHERE x.v < b.q1 - 3 * (b.q3 - b.q1) OR x.v > b.q3 + 3 * (b.q3 - b.q1)), count(*), "
            f"{lit('Kept as is; extreme values can distort some models, so check them in the source system')}, 5::numeric, 1::numeric "
            f"FROM (SELECT {v} AS v FROM {UPLOAD_TABLE}) x "
            f"CROSS JOIN (SELECT percentile_cont(0.25) WITHIN GROUP (ORDER BY v) AS q1, percentile_cont(0.75) WITHIN GROUP (ORDER BY v) AS q3 "
            f"            FROM (SELECT {v} AS v FROM {UPLOAD_TABLE}) z WHERE v IS NOT NULL) b "
            f"WHERE x.v IS NOT NULL"
        )

    union = "\n    UNION ALL\n    ".join(blocks)
    return f"""WITH checks AS (
    {union}
),
scored AS (
    SELECT table_name, check_id, dimension,
           CASE WHEN issues = 0 THEN 'low'
                WHEN 100.0 * issues / NULLIF(total, 0) >= hi  THEN 'high'
                WHEN 100.0 * issues / NULLIF(total, 0) >= mid THEN 'medium'
                ELSE 'low' END AS severity,
           issues, total, round(100.0 * issues / NULLIF(total, 0), 2) AS pct_affected, handling
    FROM checks
)
SELECT table_name, check_id, dimension, severity, issues, total, pct_affected, handling
FROM scored
ORDER BY CASE severity WHEN 'high' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END, issues DESC, check_id"""


def _rate_columns(m: Mapping, y: str) -> str:
    """The outcome column(s) for a grouped query. Rates are written as percentages (28.1 means 28.1%)."""
    if m.problem_type == "classification":
        return f"round(100.0 * avg({y}::numeric), 1) AS rate_pct"
    return f"round(avg({y}::numeric), 2) AS avg_value"


def business_questions_sql(m: Mapping, cols: dict[str, dict]) -> dict[str, str]:
    numeric, categorical = _split_features(m, cols)
    y = q(m.target_column)
    out: dict[str, str] = {}

    segments = [c for c in categorical if 2 <= cols[c]["distinct"] <= SEGMENT_MAX_DISTINCT][:MAX_SEGMENT_QUESTIONS]
    for c in segments:
        f = q(c)
        extra = f"count(*) FILTER (WHERE {y} = 1) AS positives, " if m.problem_type == "classification" else ""
        sort = "rate_pct" if m.problem_type == "classification" else "avg_value"
        out[f"outcome_by_{c}"] = (
            f"SELECT {f} AS segment, count(*) AS records, {extra}{_rate_columns(m, y)}, "
            f"RANK() OVER (ORDER BY avg({y}::numeric) DESC) AS rank_highest\n"
            f"FROM training_set\nWHERE {f} IS NOT NULL\nGROUP BY {f}\nHAVING count(*) >= 10\nORDER BY {sort} DESC"
        )
    for c in [n for n in numeric if cols[n]["distinct"] >= 8][:MAX_QUARTILE_QUESTIONS]:
        f = q(c)
        out[f"outcome_by_{c}_quartile"] = (
            f"WITH banded AS (\n    SELECT {f} AS x, {y} AS y, NTILE(4) OVER (ORDER BY {f}) AS quartile\n"
            f"    FROM training_set WHERE {f} IS NOT NULL\n)\n"
            f"SELECT quartile, count(*) AS records, round(min(x)::numeric, 2) AS from_value, round(max(x)::numeric, 2) AS to_value, "
            f"{_rate_columns(m, 'y')}\nFROM banded\nGROUP BY quartile\nORDER BY quartile"
        )
    if m.date_column:
        out["monthly_trend"] = (
            f"WITH monthly AS (\n    SELECT date_trunc('month', event_date)::date AS month, count(*) AS records, avg({y}::numeric) AS r\n"
            f"    FROM training_set WHERE event_date IS NOT NULL GROUP BY 1\n)\n"
            f"SELECT month, records, round({'100.0 * ' if m.problem_type == 'classification' else ''}r, {1 if m.problem_type == 'classification' else 2}) AS "
            f"{'rate_pct' if m.problem_type == 'classification' else 'avg_value'},\n"
            f"       round({'100.0 * ' if m.problem_type == 'classification' else ''}(r - LAG(r) OVER (ORDER BY month)), {1 if m.problem_type == 'classification' else 2}) AS change_vs_prev_month\n"
            f"FROM monthly\nORDER BY month"
        )
    return out


def kpis_sql(m: Mapping) -> dict[str, str]:
    y = q(m.target_column)
    if m.problem_type == "classification":
        return {"records": "SELECT count(*)::numeric FROM training_set", "outcome_rate": f"SELECT avg({y}::numeric) FROM training_set"}
    return {"records": "SELECT count(*)::numeric FROM training_set", "average_target": f"SELECT avg({y}::numeric) FROM training_set"}


# ---------------------------------------------------------------- the use case
def build_custom_use_case(m: Mapping, profile: dict) -> UseCase:
    """Validate the mapping against the profile and return a use case the pipeline can run."""
    cols = validate_mapping(m, profile)
    numeric, categorical = _split_features(m, cols)
    target = cols[m.target_column]
    t_name = clean_text(_display(target), 40)

    if m.problem_type == "classification":
        pos = clean_text(m.positive_value or "", 40)
        description = f"1 if '{t_name}' is '{pos}', otherwise 0 (as uploaded by the business)."
        kpis = [
            Kpi(id="records", label="Records analysed", format="number"),
            Kpi(id="outcome_rate", label=clean_text(f"Share where {t_name} is {pos}", 60), format="percent"),
        ]
        models = Models(baseline="dummy_classifier", candidates=["logistic_regression", "gradient_boosting"])
        metrics = Metrics(primary="roc_auc", report=["roc_auc", "precision", "recall", "f1"])
        recs = [
            FallbackRecommendation(
                type="act_on_prediction", facts=["decile.top", "decile.top2_share"],
                text=("Start with the records the model scores in its riskiest decile: {decile.top.rate|pct0} of them had the outcome, against "
                      "{decile.top.base_rate|pct0} across all records ({decile.top.lift|x1} the average). The top two deciles hold "
                      "{decile.top2_share.share|pct0} of all records with the outcome.")),
            FallbackRecommendation(
                type="act_on_prediction", facts=["model.test", "model.flag"],
                text=("Acting on the riskiest {model.flag.fraction|pct0} of records catches {model.test.recall|pct0} of the records with the outcome, "
                      "and {model.test.precision|pct0} of the records on that list have it, so not everyone on the list will.")),
        ]
    else:
        description = f"The value of '{t_name}' in each record (as uploaded by the business)."
        kpis = [
            Kpi(id="records", label="Records analysed", format="number"),
            Kpi(id="average_target", label=clean_text(f"Average {t_name}", 60), format="number"),
        ]
        models = Models(baseline="dummy_regressor", candidates=["linear_regression", "gradient_boosting_regressor"])
        metrics = Metrics(primary="mae", report=["mae", "rmse", "r2"])
        recs = [
            FallbackRecommendation(
                type="act_on_prediction", facts=["model.test"],
                text=("On records the model had not seen, its typical error (MAE) is {model.test.mae|n2}. "
                      "Use the predictions as a guide for planning, not as a guarantee.")),
        ]
    recs += [
        FallbackRecommendation(
            type="focus_segment", facts=["driver.1", "driver.2"],
            text=("Look first at {driver.1.name} and {driver.2.name}: they are the two inputs most strongly associated with the outcome in this model. "
                  "The data shows association only; it does not explain why the outcome happens.")),
        FallbackRecommendation(
            type="improve_data", facts=["dq.summary"],
            text=("{dq.summary.checks_with_issues|n0} of {dq.summary.checks|n0} data-quality checks found problems. "
                  "Fix the most serious ones in the source file before relying on these results.")),
    ]
    return UseCase(
        id=CUSTOM_ID,
        name=clean_text(m.name, 80),
        description=f"Analysis of the business's own data: {clean_text(m.business_question, 200)}",
        business_question=clean_text(m.business_question, 300),
        problem_type=m.problem_type,
        entity=Entity(name="record", key="entity_id"),
        target=Target(name=m.target_column, description=description),
        parameters={},
        inline_sql=InlineSql(
            clean_layer=clean_layer_sql(m, cols), training_table=training_sql(m), data_quality=data_quality_sql(m, cols),
            business_questions=business_questions_sql(m, cols), kpis=kpis_sql(m),
        ),
        data_source="upload",
        features=Features(numeric=numeric, categorical=categorical),
        models=models,
        metrics=metrics,
        evaluation=Evaluation(),
        kpis=kpis,
        ai=AiContext(
            audience="The business owner or manager who supplied the data (non-technical)",
            goal=f"Decide how to act on the results for: {clean_text(m.business_question, 200)}",
            glossary={"record": "One row of the file the business uploaded.", t_name: description},
            allowed_recommendation_types=["act_on_prediction", "focus_segment", "improve_data"],
            fallback_recommendations=recs,
        ),
    )


# ---------------------------------------------------------------- leakage warning
LEAKAGE_THRESHOLD = 0.97


def _eta_squared(x: pd.Series, y: pd.Series) -> float:
    """Share of the target's variation explained by the category (0 to 1)."""
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df) < 50 or df["x"].nunique() < 2 or df["x"].nunique() > len(df) / 5:
        return 0.0
    total = ((df["y"] - df["y"].mean()) ** 2).sum()
    if total == 0:
        return 0.0
    g = df.groupby("x")["y"]
    between = (g.count() * (g.mean() - df["y"].mean()) ** 2).sum()
    return float(between / total)


def leakage_checks(df: pd.DataFrame, uc: UseCase) -> pd.DataFrame:
    """Inputs that predict the target almost perfectly, written as data-quality rows.

    A column that nearly equals the outcome is usually something recorded AFTER the outcome
    (a cancellation date, a 'status' that already says cancelled). The model would look
    brilliant and be useless on new records.
    """
    y = pd.to_numeric(df[uc.target.name], errors="coerce")
    rows = []
    for f in uc.features.numeric:
        x = pd.to_numeric(df[f], errors="coerce")
        pair = pd.DataFrame({"x": x, "y": y}).dropna()
        if len(pair) >= 50 and pair["x"].nunique() > 1 and pair["y"].nunique() > 1:
            r = abs(float(np.corrcoef(pair["x"], pair["y"])[0, 1]))
            score = r * r
        else:
            continue
        if score >= LEAKAGE_THRESHOLD:
            rows.append((f, score))
    for f in uc.features.categorical:
        score = _eta_squared(df[f], y)
        if score >= LEAKAGE_THRESHOLD:
            rows.append((f, score))
    n = int(len(df))
    return pd.DataFrame(
        [{
            "table_name": "training_set", "check_id": f"possible_leakage:{f}", "dimension": "validity", "severity": "high",
            "issues": n, "total": n, "pct_affected": 100.0,
            "handling": "Check this input is known BEFORE the outcome happens; if not, remove it and run again",
        } for f, _ in rows],
        columns=["table_name", "check_id", "dimension", "severity", "issues", "total", "pct_affected", "handling"],
    )
