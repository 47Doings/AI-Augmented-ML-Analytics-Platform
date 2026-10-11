"""Tests that prove the platform is generic: both use cases are plain config + SQL.

No database needed. Run from the project root:  python -m pytest -q
"""
import re

import pytest

from core.sql_runner import load_named_queries
from core.use_case import UseCaseError, list_use_cases, load_use_case


def test_both_use_cases_load_and_validate():
    ids = {u.id for u in list_use_cases()}
    assert {"churn", "revenue"} <= ids


def test_use_cases_differ_only_in_content_not_in_shape():
    churn, revenue = load_use_case("churn"), load_use_case("revenue")
    assert churn.problem_type == "classification" and revenue.problem_type == "regression"
    assert churn.target.name != revenue.target.name
    assert churn.sql.training_table != revenue.sql.training_table
    assert churn.sql.data_quality == revenue.sql.data_quality  # the dataset description is shared


@pytest.mark.parametrize("uc_id", ["churn", "revenue"])
def test_sql_files_exist_and_use_only_declared_parameters(uc_id):
    uc = load_use_case(uc_id)
    files = [uc.sql.training_table, uc.sql.business_questions, uc.sql.data_quality, *[k.sql for k in uc.kpis]]
    for rel in filter(None, files):
        for name, query in load_named_queries(rel).items():
            used = set(re.findall(r"%\((\w+)\)s", query))
            assert used <= set(uc.parameters), f"{rel}:{name} uses undeclared parameters {used - set(uc.parameters)}"
            stripped = re.sub(r"%\(\w+\)s", "", query)
            assert "%" not in stripped, f"{rel}:{name} contains a literal percent sign (driver reads it as a parameter)"


def test_revenue_training_sql_has_the_target_column_and_every_feature():
    uc = load_use_case("revenue")
    query = next(iter(load_named_queries(uc.sql.training_table).values()))
    for col in [uc.entity.key, uc.target.name, *uc.feature_columns]:
        assert re.search(rf"\bAS\s+{col}\b|\.{col}\b|\b{col},?\s*$", query, re.MULTILINE), f"{col} not found in training SQL"


def test_regression_config_rejects_classification_models():
    uc = load_use_case("revenue")
    bad = uc.model_dump()
    bad["models"]["candidates"] = ["logistic_regression"]
    with pytest.raises(Exception):
        type(uc).model_validate(bad)


def test_unknown_use_case_is_a_clean_error():
    with pytest.raises(UseCaseError):
        load_use_case("does_not_exist")
    with pytest.raises(UseCaseError):
        load_use_case("../churn")
