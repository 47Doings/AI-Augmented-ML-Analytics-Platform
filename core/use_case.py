"""Use-case definitions: the part of the platform that changes per business problem.

The platform (signup, workspaces, running SQL, the ML pipeline, the AI layer,
reporting) never mentions "churn" or "sales". Everything specific to one
business problem lives in a YAML file under use_cases/, which is loaded and
validated here. Adding a new use case means adding a new YAML file (and its
SQL files), not changing the platform code.

Validation is strict on purpose: a typo or an impossible combination in a
config (for example a regression model on a classification problem) fails
at load time with a clear message, not halfway through a training run.
"""
from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
USE_CASES_DIR = PROJECT_ROOT / "use_cases"

_ID_RE = re.compile(r"^[a-z][a-z0-9_]{1,40}$")

# What the ML pipeline (Step 5) will know how to build for each problem type.
# A use case can only pick from these names.
ALLOWED_MODELS: dict[str, set[str]] = {
    "classification": {"dummy_classifier", "logistic_regression", "random_forest", "gradient_boosting"},
    "regression": {"dummy_regressor", "linear_regression", "random_forest_regressor", "gradient_boosting_regressor"},
}
ALLOWED_METRICS: dict[str, set[str]] = {
    "classification": {"roc_auc", "average_precision", "precision", "recall", "f1", "accuracy"},
    "regression": {"mae", "rmse", "r2", "mape"},
}


class UseCaseError(Exception):
    """A use-case file is missing or invalid."""


def _check_sql_path(value: str) -> str:
    """SQL files must live under sql/ and be relative paths. Blocks '../' tricks."""
    path = PurePosixPath(value)
    if (not path.parts) or path.is_absolute() or ".." in path.parts or path.parts[0] != "sql" or path.suffix != ".sql":
        raise ValueError(f"SQL path must be a relative path under sql/ ending in .sql, got: {value!r}")
    return value


class _Strict(BaseModel):
    # extra="forbid": a misspelt key in the YAML is an error, not silently ignored.
    model_config = ConfigDict(extra="forbid")


class Entity(_Strict):
    name: str  # what one row of the training table represents, e.g. "customer"
    key: str  # the column that identifies it, e.g. "customer_id"


class Target(_Strict):
    name: str  # column the model predicts
    description: str  # plain-English definition (also given to the AI later)


class InlineSql(_Strict):
    """SQL written by the platform itself (not read from sql/*.sql).

    Used for a business's own uploaded data: the platform generates the SQL from the column mapping
    the business chose. The browser never sends SQL; the server builds these strings from validated,
    quoted column names (core/custom_usecase.py).
    """

    clean_layer: str  # CREATE VIEW ... that cleans the uploaded table (the 'clean layer')
    training_table: str  # one SELECT, one row per entity
    data_quality: str | None = None  # one SELECT returning the standard data-quality columns
    business_questions: dict[str, str] = Field(default_factory=dict)  # {query name: SELECT}
    kpis: dict[str, str] = Field(default_factory=dict)  # {kpi id: SELECT returning one number}


class SqlPaths(_Strict):
    training_table: str
    data_quality: str | None = None
    business_questions: str | None = None

    @field_validator("training_table", "data_quality", "business_questions")
    @classmethod
    def _safe_paths(cls, value):
        return value if value is None else _check_sql_path(value)


class Features(_Strict):
    numeric: list[str] = Field(default_factory=list)
    categorical: list[str] = Field(default_factory=list)


class Models(_Strict):
    baseline: str  # the simple model every other model must beat
    candidates: list[str] = Field(min_length=1)


class Metrics(_Strict):
    primary: str  # the one number used to pick the best model
    report: list[str] = Field(min_length=1)  # everything shown in the report


class Evaluation(_Strict):
    test_size: float = Field(default=0.2, gt=0, lt=1)
    random_seed: int = 42
    # capacity-based operating point: "flag the riskiest X% of entities" (classification reporting)
    flag_top_fraction: float = Field(default=0.10, gt=0, le=0.5)


class Kpi(_Strict):
    id: str
    label: str
    sql: str | None = None  # path under sql/; left empty when the use case carries inline_sql
    format: Literal["number", "percent", "currency"] = "number"

    @field_validator("sql")
    @classmethod
    def _safe_sql(cls, value: str | None) -> str | None:
        return value if value is None else _check_sql_path(value)


class FallbackRecommendation(_Strict):
    """A recommendation the no-AI template may write, filled in from computed facts.

    text uses placeholders like {decile.top.rate|pct0}: {<fact id>.<field>|<format>}.
    If any referenced fact is missing in a run, the recommendation is skipped.
    """

    type: str
    text: str
    facts: list[str] = Field(min_length=1)  # fact ids this recommendation rests on


class AiContext(_Strict):
    """What the AI layer is told about this use case (Step 8)."""

    audience: str
    goal: str
    glossary: dict[str, str] = Field(default_factory=dict)
    allowed_recommendation_types: list[str] = Field(default_factory=list)
    fallback_recommendations: list[FallbackRecommendation] = Field(default_factory=list)

    @model_validator(mode="after")
    def _types_allowed(self) -> "AiContext":
        for rec in self.fallback_recommendations:
            if rec.type not in self.allowed_recommendation_types:
                raise ValueError(
                    f"fallback recommendation type '{rec.type}' is not in allowed_recommendation_types "
                    f"{self.allowed_recommendation_types}"
                )
        return self


class UseCase(_Strict):
    id: str
    name: str
    description: str
    business_question: str
    problem_type: Literal["classification", "regression"]
    entity: Entity
    target: Target
    # Named values passed into the SQL files, e.g. the cutoff date.
    parameters: dict[str, str | int | float] = Field(default_factory=dict)
    # Exactly one of these: SQL files written by a developer (the demo use cases) or SQL the platform
    # generated for an uploaded dataset.
    sql: SqlPaths | None = None
    inline_sql: InlineSql | None = None
    data_source: Literal["demo", "upload"] = "demo"  # where the rows came from; shown in reports and given to the AI
    features: Features
    models: Models
    metrics: Metrics
    evaluation: Evaluation = Field(default_factory=Evaluation)
    kpis: list[Kpi] = Field(default_factory=list)
    ai: AiContext

    @field_validator("id")
    @classmethod
    def _valid_id(cls, value: str) -> str:
        if not _ID_RE.fullmatch(value):
            raise ValueError("id must be lowercase letters, digits or underscores, starting with a letter")
        return value

    @model_validator(mode="after")
    def _check_consistency(self) -> "UseCase":
        if (self.sql is None) == (self.inline_sql is None):
            raise ValueError("exactly one of 'sql' (files) or 'inline_sql' (generated) is required")
        for kpi in self.kpis:
            if self.inline_sql is not None and kpi.id not in self.inline_sql.kpis:
                raise ValueError(f"kpi '{kpi.id}' has no query in inline_sql.kpis")
            if self.sql is not None and kpi.sql is None:
                raise ValueError(f"kpi '{kpi.id}' needs a 'sql' path")
        feature_cols = self.features.numeric + self.features.categorical
        if not feature_cols:
            raise ValueError("features: at least one numeric or categorical feature is required")
        if self.target.name in feature_cols:
            raise ValueError(f"target '{self.target.name}' must not also be a feature (that would be leakage)")
        if self.entity.key in feature_cols:
            raise ValueError(f"entity key '{self.entity.key}' is an identifier and must not be a feature")
        if set(self.features.numeric) & set(self.features.categorical):
            raise ValueError("a column cannot be both numeric and categorical")

        allowed_models = ALLOWED_MODELS[self.problem_type]
        for name in [self.models.baseline, *self.models.candidates]:
            if name not in allowed_models:
                raise ValueError(
                    f"model '{name}' is not valid for {self.problem_type}; choose from {sorted(allowed_models)}"
                )

        allowed_metrics = ALLOWED_METRICS[self.problem_type]
        for name in self.metrics.report:
            if name not in allowed_metrics:
                raise ValueError(
                    f"metric '{name}' is not valid for {self.problem_type}; choose from {sorted(allowed_metrics)}"
                )
        if self.metrics.primary not in self.metrics.report:
            raise ValueError("metrics.primary must also appear in metrics.report")
        return self

    @property
    def feature_columns(self) -> list[str]:
        return self.features.numeric + self.features.categorical


def load_use_case(use_case_id: str, directory: Path = USE_CASES_DIR) -> UseCase:
    """Load and validate one use case by id, e.g. load_use_case('churn')."""
    # The id may later come from an API request, so check it before it goes near a file path.
    if not isinstance(use_case_id, str) or not _ID_RE.fullmatch(use_case_id):
        raise UseCaseError(f"Invalid use case id: {use_case_id!r}")
    path = directory / f"{use_case_id}.yaml"
    if not path.is_file():
        raise UseCaseError(f"Use case '{use_case_id}' not found")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))  # safe_load: never runs code from the file
        use_case = UseCase.model_validate(raw)
    except yaml.YAMLError as exc:
        raise UseCaseError(f"{path.name}: invalid YAML: {exc}") from exc
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())
        raise UseCaseError(f"{path.name}: {problems}") from exc
    if use_case.id != use_case_id:
        raise UseCaseError(f"{path.name}: id '{use_case.id}' does not match the file name")
    return use_case


def list_use_cases(directory: Path = USE_CASES_DIR) -> list[UseCase]:
    """Every use case the platform offers, validated."""
    return [load_use_case(p.stem, directory) for p in sorted(directory.glob("*.yaml"))]
