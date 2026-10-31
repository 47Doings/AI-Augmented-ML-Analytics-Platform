"""Tests for the business's own data: upload, profile, column mapping, generated SQL, and the API path.

The first groups need no database. The last two (pipeline on uploaded data, and the API flow) use the
real PostgreSQL from the Docker container and are skipped automatically if it is not reachable. They
create throw-away schemas and accounts and delete them afterwards.

Run from the project root:  python -m pytest -q tests/test_upload.py
"""
import io
import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from core import upload
from core.custom_usecase import (
    Mapping, MappingError, build_custom_use_case, leakage_checks, lit, q, validate_mapping,
)
from core.upload import UploadError, parse_csv, profile_frame, sanitise_columns
from core.use_case import UseCase
from scripts.make_sample_csv import make

FEATURES = ["branch", "employment_type", "monthly_income_ghs", "loan_amount_ghs", "term_months", "prior_loans",
            "late_payments_prior", "has_guarantor"]


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


@pytest.fixture(scope="module")
def sample() -> pd.DataFrame:
    return make(rows=600, seed=3)


@pytest.fixture(scope="module")
def profile(sample):
    df, original = parse_csv(csv_bytes(sample))
    return profile_frame(df, original)


def good_mapping(**over) -> Mapping:
    base = dict(name="Loan defaults", business_question="Which applicants are most likely to default?",
                problem_type="classification", target_column="defaulted", positive_value="yes",
                id_column="loan_id", date_column="applied_on", features=FEATURES)
    base.update(over)
    return Mapping(**base)


# ---------------------------------------------------------------- column names
def test_column_names_become_safe_unique_identifiers():
    names = sanitise_columns(['Monthly Income (GHS)', 'monthly_income_ghs', '2024 sales', 'a"; DROP TABLE x;--', '', 'Entity ID', 'x' * 100])
    assert names[0] == "monthly_income_ghs" and names[1] == "monthly_income_ghs_2"  # clashes get a number
    assert names[2] == "c_2024_sales"  # may not start with a digit
    assert names[3] == "a_drop_table_x"  # quotes, semicolons and dashes are gone
    assert names[4] == "column"
    assert names[5] == "entity_id_col"  # the platform's own names are reserved
    assert len(names[6]) <= upload.MAX_NAME
    assert all(upload.is_safe_identifier(n) for n in names) and len(set(names)) == len(names)


def test_quoting_helpers_refuse_unsafe_input():
    assert q("income") == '"income"'
    for bad in ['a"b', "a b", "1abc", "", "x;y", "Robert'); DROP TABLE students;--"]:
        with pytest.raises(MappingError):
            q(bad)
    assert lit("it's") == "'it''s'"
    assert lit("x' OR '1'='1") == "'x'' OR ''1''=''1'"  # the quote that would close the string is doubled
    with pytest.raises(MappingError):
        lit("a\x00b")


# ---------------------------------------------------------------- reading the file
def test_parse_reads_a_normal_file_with_bom_and_semicolons(sample):
    text = sample.to_csv(index=False, sep=";")
    df, original = parse_csv(b"\xef\xbb\xbf" + text.encode("utf-8"))
    assert df.shape == sample.shape and original[0] == "loan_id" and df.columns[0] == "loan_id"


@pytest.mark.parametrize("data,message", [
    (b"", "empty"),
    (b"   \n  ", "empty"),
    (b"a,b\n" + b"1,2\n" * 10, "only 10 rows"),
    (b"a,b\n" + "caf\xe9,x\n".encode("latin-1") * 100, "UTF-8"),  # an Excel 'CSV' saved in a Western encoding
    (b"\xff\xfe" + "a,b\n1,2\n".encode("utf-16-le"), "binary"),  # UTF-16 text is full of NUL bytes
    (b"id\n" + b"1\n" * 100, "at least two columns"),
    (b"a,b\n" + b"1,2,3,4\n" * 100, "could not be read"),
    (b"a,b\x00,c\n" + b"1,2,3\n" * 100, "binary"),
])
def test_parse_rejects_files_it_cannot_use_with_a_readable_reason(data, message):
    with pytest.raises(UploadError, match=message):
        parse_csv(data)


def test_parse_rejects_too_many_columns_and_too_large_files():
    wide = pd.DataFrame(np.zeros((60, upload.MAX_COLS + 1), dtype=int)).astype(str)
    with pytest.raises(UploadError, match="columns"):
        parse_csv(csv_bytes(wide))
    with pytest.raises(UploadError, match="MB"):
        parse_csv(b"a,b\n" + b"1,2\n" * (upload.MAX_BYTES // 4 + 10))


# ---------------------------------------------------------------- profile
def test_profile_recognises_column_types_gaps_and_suggestions(profile):
    cols = {c["name"]: c for c in profile["columns"]}
    assert profile["rows"] > 600
    assert cols["loan_id"]["kind"] == "identifier"
    assert cols["applied_on"]["kind"] == "date"
    assert cols["branch"]["kind"] == "categorical" and cols["monthly_income_ghs"]["kind"] == "numeric"
    assert cols["monthly_income_ghs"]["missing"] > 0  # blanks and "N/A" both count as missing
    assert not cols["loan_id"]["usable_as_feature"] and cols["branch"]["usable_as_feature"]
    assert {v["value"] for v in cols["has_guarantor"]["top_values"]} == {"yes", "no"}  # "Yes", "yes " and "YES" are one value
    assert profile["suggested"] == {"id_column": "loan_id", "target_column": "defaulted", "date_column": "applied_on"}


# ---------------------------------------------------------------- mapping
def test_a_good_mapping_builds_a_complete_use_case(profile):
    uc = build_custom_use_case(good_mapping(), profile)
    assert isinstance(uc, UseCase) and uc.id == "custom" and uc.data_source == "upload" and uc.sql is None
    assert uc.features.numeric == ["monthly_income_ghs", "loan_amount_ghs", "term_months", "prior_loans", "late_payments_prior"]
    assert uc.features.categorical == ["branch", "employment_type", "has_guarantor"]
    sql = uc.inline_sql
    assert "%" not in "".join([sql.clean_layer, sql.training_table, sql.data_quality, *sql.business_questions.values(), *sql.kpis.values()])
    assert {"outcome_by_branch", "monthly_trend"} <= set(sql.business_questions)


def test_regression_mapping_and_no_date_no_id(profile):
    m = good_mapping(problem_type="regression", target_column="loan_amount_ghs", positive_value=None, id_column=None, date_column=None,
                     features=[f for f in FEATURES if f != "loan_amount_ghs"])
    uc = build_custom_use_case(m, profile)
    assert uc.problem_type == "regression" and "monthly_trend" not in uc.inline_sql.business_questions
    assert "row_id::text" in uc.inline_sql.clean_layer  # no id column: the row number identifies the record


@pytest.mark.parametrize("change,message", [
    (dict(target_column="nope"), "not a column"),
    (dict(features=FEATURES + ["defaulted"]), "both an input"),
    (dict(features=FEATURES + ["loan_id"]), "both an input"),
    (dict(features=["loan_id", "branch"], id_column=None), "cannot be used as an input"),
    (dict(positive_value=None), "Choose which value"),
    (dict(positive_value="maybe"), "not a value that appears"),
    (dict(problem_type="regression"), "not a number"),
    (dict(date_column="branch", features=[f for f in FEATURES if f != "branch"]), "does not look like a date"),
    (dict(id_column="defaulted"), "three different columns"),
])
def test_bad_choices_are_refused_with_a_clear_message(profile, change, message):
    with pytest.raises(MappingError, match=message):
        build_custom_use_case(good_mapping(**change), profile)


def test_rare_outcome_is_refused(sample):
    df = sample.copy()
    df["defaulted"] = "no"
    df.loc[df.index[:5], "defaulted"] = "yes"
    d, original = parse_csv(csv_bytes(df))
    with pytest.raises(MappingError, match="rarer outcome"):
        validate_mapping(good_mapping(), profile_frame(d, original))


def test_hostile_form_values_cannot_reach_sql(profile):
    for field in ("target_column", "id_column", "date_column"):
        with pytest.raises(ValidationError):
            good_mapping(**{field: 'x"; DROP TABLE platform.users; --'})
    with pytest.raises(ValidationError):
        good_mapping(features=['branch"; DROP TABLE x;--'])
    with pytest.raises(ValidationError):
        Mapping(**{**good_mapping().model_dump(), "inline_sql": "SELECT 1"})  # SQL is never accepted from the browser
    # a hostile outcome value is not in the profile, so it is refused before any SQL is written
    with pytest.raises(MappingError, match="not a value that appears"):
        build_custom_use_case(good_mapping(positive_value="x' OR '1'='1"), profile)
    # a hostile free-text name ends up as plain text, without markup
    uc = build_custom_use_case(good_mapping(name="<script>alert(1)</script> Loans", business_question="Why <b>do</b> people default?"), profile)
    assert "<" not in uc.name and "<" not in uc.business_question


# ---------------------------------------------------------------- leakage warning
def test_leakage_check_flags_an_input_that_gives_the_answer_away(profile):
    uc = build_custom_use_case(good_mapping(), profile)
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    frame = pd.DataFrame({
        "entity_id": range(400), "defaulted": y,
        "monthly_income_ghs": rng.normal(1000, 200, 400), "loan_amount_ghs": y * 100.0 + rng.normal(0, 1, 400),  # nearly the target
        "term_months": rng.integers(3, 18, 400), "prior_loans": rng.integers(0, 5, 400), "late_payments_prior": rng.integers(0, 5, 400),
        "branch": rng.choice(list("abc"), 400), "employment_type": np.where(y == 1, "closed", "open"),  # equals the target in words
        "has_guarantor": rng.choice(["yes", "no"], 400),
    })
    flagged = set(leakage_checks(frame, uc)["check_id"])
    assert flagged == {"possible_leakage:loan_amount_ghs", "possible_leakage:employment_type"}


# ---------------------------------------------------------------- the pipeline on uploaded data (needs the database)
@pytest.fixture
def schema():
    from core.db import connect, make_schema_name
    from core.loader import ensure_workspace_schema
    from psycopg import sql

    name = make_schema_name()
    try:
        ensure_workspace_schema(name, Path(__file__).resolve().parent.parent / "db" / "schema.sql")
    except Exception as exc:
        pytest.skip(f"database not available: {exc}")
    yield name
    with connect() as conn:
        conn.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(name)))


def load(schema, df):
    d, original = parse_csv(csv_bytes(df))
    prof = profile_frame(d, original)
    assert upload.load_upload(schema, d) == len(df)
    return prof


@pytest.fixture
def no_ai(monkeypatch):
    """Never call the real Snwolley API from a test (it is rate limited and needs a key)."""
    monkeypatch.setattr("core.pipeline.get_provider", lambda settings: None)
    monkeypatch.setattr("api.main.get_provider", lambda settings: None)


def test_classification_pipeline_on_uploaded_data(schema, sample, tmp_path, no_ai):
    from core.pipeline import run_pipeline

    prof = load(schema, sample)
    uc = build_custom_use_case(good_mapping(), prof)
    res = run_pipeline(schema, uc, tmp_path)
    assert [s["status"] for s in res["steps"]] == ["done"] * 6 + ["skipped", "done"]  # drift needs a later date, so it is skipped
    n_dup = len(sample) - sample["loan_id"].nunique()
    assert res["training_table"]["rows"] == len(sample) - n_dup - int(sample["defaulted"].eq("").sum())  # duplicates left out, raw untouched
    assert res["kpis"][0]["shown"] == f"{res['training_table']['rows']:,}"
    assert res["model"]["card"]["beats_baseline_in_cv"] is True and "deciles" in res["model"]
    checks = {r[1]: r for r in res["data_quality"]["rows"]}
    assert checks["duplicate_ids:loan_id"][4] == n_dup  # the SQL found exactly the duplicates the sample generator injected
    assert any(k.startswith("missing:monthly_income") and checks[k][4] > 0 for k in checks)
    assert any(k.startswith("spelling_variants:has_guarantor") for k in checks)
    assert "monthly_trend" in res["business_questions"] and "outcome_by_branch" in res["business_questions"]
    seg = res["business_questions"]["outcome_by_branch"]
    assert seg["columns"][:2] == ["segment", "records"] and seg["total_rows"] == 5
    ins = json.loads((tmp_path / "insights.json").read_text())
    assert ins["mode"] == "fallback" and ins["validation"]["ok"]
    assert "synthetic" not in json.dumps(ins).lower() and "uploaded" in json.dumps(ins).lower()


def test_regression_pipeline_on_uploaded_data(schema, sample, tmp_path, no_ai):
    from core.pipeline import run_pipeline

    prof = load(schema, sample)
    m = good_mapping(problem_type="regression", target_column="loan_amount_ghs", positive_value=None, id_column=None, date_column=None,
                     features=[f for f in FEATURES if f != "loan_amount_ghs"] + ["defaulted"])
    res = run_pipeline(schema, build_custom_use_case(m, prof), tmp_path)
    assert res["use_case"]["problem_type"] == "regression" and res["model"]["card"]["test_metrics"]["mae"] > 0
    assert json.loads((tmp_path / "insights.json").read_text())["validation"]["ok"]


def test_leakage_is_reported_in_the_data_quality_table(schema, sample, tmp_path, no_ai):
    from core.pipeline import run_pipeline

    leaky = sample.copy()
    leaky["outcome_status"] = np.where(leaky["defaulted"] == "yes", "written off", "repaid")  # recorded after the outcome
    prof = load(schema, leaky)
    res = run_pipeline(schema, build_custom_use_case(good_mapping(features=FEATURES + ["outcome_status"]), prof), tmp_path)
    first = res["data_quality"]["rows"][0]
    assert first[1] == "possible_leakage:outcome_status" and first[3] == "high"  # worst first
    facts = json.loads((tmp_path / "insights.json").read_text())["facts"]
    assert any("possible_leakage" in f["text"] for f in facts)


def test_uploading_again_replaces_the_data_and_junk_columns_are_harmless(schema, sample):
    prof = load(schema, sample)
    assert prof["rows"] == len(sample)
    odd = sample.rename(columns={"branch": 'br"anch; DROP TABLE x;--'})
    prof2 = load(schema, odd.head(100))
    assert prof2["rows"] == 100 and "br_anch_drop_table_x" in {c["name"] for c in prof2["columns"]}


# ---------------------------------------------------------------- the API flow (needs the database)
@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from api.main import app

    try:
        with TestClient(app) as c:
            yield c
    except Exception as exc:
        if "connection" in str(exc).lower() or "OperationalError" in type(exc).__name__:
            pytest.skip(f"database not available: {exc}")
        raise


@pytest.fixture
def make_user(client):
    from fastapi.testclient import TestClient

    from api import store
    from api.main import app

    created = []

    def _make(business="Own Data Co"):
        email = f"t_{uuid.uuid4().hex[:10]}@example.com"
        c = TestClient(app)
        assert c.post("/api/signup", json={"email": email, "password": "a-long-test-password", "business_name": business}).status_code == 201
        created.append(email)
        return c

    yield _make
    for email in created:
        u = store.get_user_by_email(email)
        if u:
            store.delete_account(u["user_id"])


def put_csv(c, data: bytes, name="loans.csv"):
    return c.put(f"/api/workspace/upload?filename={name}", content=data, headers={"Content-Type": "text/csv"})


def test_upload_endpoints_need_login_and_validate_input(client, make_user, sample):
    from fastapi.testclient import TestClient

    from api.main import app

    anon = TestClient(app)
    assert anon.put("/api/workspace/upload", content=b"a,b").status_code == 401
    assert anon.get("/api/workspace/upload").status_code == 401
    assert anon.put("/api/workspace/custom-use-case", json={}).status_code == 401

    c = make_user()
    assert c.get("/api/workspace/upload").status_code == 404  # nothing uploaded yet
    assert c.put("/api/workspace/custom-use-case", json=good_mapping().model_dump()).status_code == 409  # upload first
    assert put_csv(c, b"").status_code == 422
    assert put_csv(c, b"just some text, not a table").status_code == 422
    assert put_csv(c, b"\xff\xfe\x00bad").status_code == 422
    too_big = put_csv(c, b"a,b\n" + b"1,2\n" * (upload.MAX_BYTES // 4 + 1000))
    assert too_big.status_code == 413 and "10 MB" in too_big.json()["detail"]
    assert c.get("/api/me").json()["upload"] is None  # every refusal left the workspace untouched


def test_own_data_flow_end_to_end_and_tenant_isolation(client, make_user, sample, no_ai):
    a, b = make_user("Business A"), make_user("Business B")

    r = put_csv(a, csv_bytes(sample), "my loans.csv")
    assert r.status_code == 200 and r.json()["rows"] == len(sample) and r.json()["filename"] == "my loans.csv"
    me = a.get("/api/me").json()
    assert me["upload"]["rows"] == len(sample) and me["use_case_id"] is None and me["data_loaded"] is False
    assert a.post("/api/workspace/run").status_code == 409  # no use case yet

    assert b.get("/api/workspace/upload").status_code == 404  # B cannot see A's file
    assert b.get("/api/me").json()["upload"] is None

    bad = a.put("/api/workspace/custom-use-case", json=good_mapping(positive_value="maybe").model_dump())
    assert bad.status_code == 422 and "not a value" in bad.json()["detail"]
    assert a.put("/api/workspace/custom-use-case", json={**good_mapping().model_dump(), "sql": "DROP TABLE x"}).status_code == 422
    assert a.put("/api/workspace/custom-use-case", json=good_mapping().model_dump()).status_code == 200
    me = a.get("/api/me").json()
    assert me["use_case_id"] == "custom" and me["data_loaded"] is True and me["data_rows"] == {"records": len(sample)}
    assert a.post("/api/workspace/data", json={}).status_code == 409  # demo data is not generated over the business's own data

    assert a.post("/api/workspace/run").status_code == 202
    run = a.get("/api/workspace/run").json()["run"]
    assert run["status"] == "succeeded", run
    res = a.get("/api/workspace/results").json()
    assert res["use_case"]["id"] == "custom" and res["use_case"]["data_source"] == "upload"
    assert "ws_" not in json.dumps(res)
    ins = a.get("/api/workspace/insights").json()
    assert ins["validation"]["ok"] and "synthetic" not in json.dumps(ins).lower()
    assert a.post("/api/workspace/insights").status_code == 200  # rewriting the text works for a custom use case too
    rep = a.get("/api/workspace/report")
    assert rep.status_code == 200 and "uploaded by the business" in rep.text and "synthetic" not in rep.text.lower()
    assert b.get("/api/workspace/results").status_code == 404

    # a new file clears the old choices and results, so nothing describes a file that is gone
    assert put_csv(a, csv_bytes(sample.head(200))).status_code == 200
    me = a.get("/api/me").json()
    assert me["custom_mapping"] is None and me["use_case_id"] is None and "custom" not in me["results_available"]
    assert a.get("/api/workspace/results?use_case=custom").status_code == 404

    # the demo path still works in the same workspace
    assert a.put("/api/workspace/use-case", json={"use_case_id": "churn"}).status_code == 200
    assert a.post("/api/workspace/run").status_code == 409  # demo data was never generated here
