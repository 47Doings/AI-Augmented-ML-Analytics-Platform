"""Tests for the API and its security.

The first group needs no database. The second group is an integration test against the
real PostgreSQL from the Docker container; it is skipped automatically if the database
is not reachable. The integration test creates throw-away accounts and deletes them
(and their schemas) afterwards.

Run from the project root:  python -m pytest -q tests/test_api.py
"""
import time
import uuid

import jwt
import pytest

from api import security, store
from core.config import get_settings


# ---------------------------------------------------------------- security (no database)
def test_password_hash_is_salted_and_verifies():
    a, b = security.hash_password("correct horse"), security.hash_password("correct horse")
    assert a != b  # different random salt each time
    assert "correct horse" not in a
    assert security.verify_password("correct horse", a)
    assert not security.verify_password("wrong horse", a)
    assert not security.verify_password("x", "not-a-valid-hash")


def test_token_roundtrip_and_tampering():
    token = security.create_token(42)
    assert security.read_token(token) == 42
    assert security.read_token(token[:-2] + ("AA" if not token.endswith("AA") else "BB")) is None  # signature broken
    assert security.read_token("garbage") is None


def test_token_with_wrong_key_or_algorithm_or_expired_is_rejected():
    key = get_settings().app_secret_key.get_secret_value()
    assert security.read_token(jwt.encode({"sub": "1", "exp": time.time() + 60}, "another-secret-key-another-secret-key", algorithm="HS256")) is None
    assert security.read_token(jwt.encode({"sub": "1", "exp": time.time() + 60}, key, algorithm="HS512")) is None  # algorithm is pinned
    assert security.read_token(jwt.encode({"sub": "1", "exp": time.time() - 10}, key, algorithm="HS256")) is None  # expired
    assert security.read_token(jwt.encode({"sub": "1"}, key, algorithm="HS256")) is None  # no expiry claim


def test_login_throttle_blocks_then_resets():
    t = security.LoginThrottle(max_failures=3, window_seconds=60)
    k = ("1.2.3.4", "a@b.co")
    for _ in range(3):
        assert not t.blocked(k)
        t.record_failure(k)
    assert t.blocked(k)
    assert not t.blocked(("1.2.3.4", "other@b.co"))  # other accounts are not affected
    t.reset(k)
    assert not t.blocked(k)


# ---------------------------------------------------------------- API (needs the database)
@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from api.main import app

    try:
        with TestClient(app) as c:
            yield c
    except Exception as exc:  # database not reachable
        if "connection" in str(exc).lower() or "OperationalError" in type(exc).__name__:
            pytest.skip(f"database not available: {exc}")
        raise


@pytest.fixture
def make_user(client):
    """Creates users through the real signup endpoint; deletes them (and their data) at the end."""
    from fastapi.testclient import TestClient

    from api.main import app

    created, clients = [], []

    def _make(business="Test Co"):
        email = f"t_{uuid.uuid4().hex[:10]}@example.com"
        c = TestClient(app)  # its own cookie jar = its own browser
        r = c.post("/api/signup", json={"email": email, "password": "a-long-test-password", "business_name": business})
        assert r.status_code == 201, r.text
        created.append(email)
        clients.append(c)
        return c, email

    yield _make
    for email in created:
        u = store.get_user_by_email(email)
        if u:
            store.delete_account(u["user_id"])


def test_signup_sets_http_only_cookie_and_creates_workspace(make_user):
    c, email = make_user("Kofi's Telecom")
    me = c.get("/api/me").json()
    assert me["email"] == email and me["business_name"] == "Kofi's Telecom"
    assert me["use_case_id"] is None and me["data_loaded"] is False
    set_cookie = c.post("/api/login", json={"email": email, "password": "a-long-test-password"}).headers["set-cookie"].lower()
    assert "httponly" in set_cookie and "samesite=lax" in set_cookie


def test_signup_validation_and_duplicates(client, make_user):
    _, email = make_user()
    body = {"email": email, "password": "a-long-test-password", "business_name": "X Co"}
    assert client.post("/api/signup", json=body).status_code == 409
    assert client.post("/api/signup", json={**body, "email": "new@example.com", "password": "short"}).status_code == 422
    assert client.post("/api/signup", json={**body, "email": "not-an-email"}).status_code == 422
    assert client.post("/api/signup", json={**body, "email": "z@example.com", "is_admin": True}).status_code == 422  # unknown field


def test_login_errors_are_generic_and_throttled(client, make_user):
    _, email = make_user()
    wrong = client.post("/api/login", json={"email": email, "password": "wrong-password"})
    nobody = client.post("/api/login", json={"email": f"nobody_{uuid.uuid4().hex[:6]}@example.com", "password": "wrong-password"})
    assert wrong.status_code == nobody.status_code == 401
    assert wrong.json() == nobody.json()  # does not reveal whether the email exists
    for _ in range(5):
        client.post("/api/login", json={"email": email, "password": "wrong-password"})
    assert client.post("/api/login", json={"email": email, "password": "a-long-test-password"}).status_code == 429


def test_protected_routes_need_login(client):
    from fastapi.testclient import TestClient

    from api.main import app

    anon = TestClient(app)
    for method, path in [("get", "/api/me"), ("get", "/api/use-cases"), ("post", "/api/workspace/run"),
                         ("get", "/api/workspace/results"), ("get", "/api/workspace/insights"), ("post", "/api/workspace/insights"), ("post", "/api/workspace/data"), ("put", "/api/workspace/use-case")]:
        assert getattr(anon, method)(path).status_code == 401, path
    assert anon.get("/api/me", headers={"Authorization": "Bearer nonsense"}).status_code == 401


def test_hostile_business_name_is_stored_as_plain_text(make_user):
    name = "x'); DROP SCHEMA platform; --<script>alert(1)</script>"
    c, _ = make_user(name)
    assert c.get("/api/me").json()["business_name"] == name  # stored literally; the page escapes it when showing it
    assert store.get_user_by_email("nobody@example.com") is None  # platform tables still there


def test_use_case_choice_rejects_bad_ids_and_order_of_steps(make_user):
    c, _ = make_user()
    assert c.post("/api/workspace/run").status_code == 409  # no use case yet
    assert c.put("/api/workspace/use-case", json={"use_case_id": "../churn"}).status_code == 404
    assert c.put("/api/workspace/use-case", json={"use_case_id": "nope"}).status_code == 404
    assert c.post("/api/workspace/data", json={}).status_code == 409  # choose a use case first
    assert c.put("/api/workspace/use-case", json={"use_case_id": "churn"}).status_code == 200
    assert c.post("/api/workspace/run").status_code == 409  # no data yet
    assert c.post("/api/workspace/data", json={"customers": 10}).status_code == 422  # out of range


def test_only_one_running_pipeline_per_workspace(make_user):
    c, email = make_user()
    ws = store.get_context(store.get_user_by_email(email)["user_id"])[1]
    run_id = store.start_run(ws["workspace_id"], "churn", [])
    with pytest.raises(store.RunInProgress):
        store.start_run(ws["workspace_id"], "churn", [])  # the database refuses a second one
    store.finish_run(run_id, "succeeded")
    store.finish_run(store.start_run(ws["workspace_id"], "churn", []), "succeeded")  # fine once the first finished


def test_full_flow_for_two_use_cases_and_tenant_isolation(make_user):
    a, _ = make_user("Business A")
    b, _ = make_user("Business B")

    # --- A: churn, end to end
    assert a.put("/api/workspace/use-case", json={"use_case_id": "churn"}).status_code == 200
    r = a.post("/api/workspace/data", json={"customers": 1000, "seed": 7})
    assert r.status_code == 200 and r.json()["rows"]["customers"] > 1000
    assert a.post("/api/workspace/run").status_code == 202  # the test client waits for the background task
    run = a.get("/api/workspace/run").json()["run"]
    assert run["status"] == "succeeded", run
    assert all(s["status"] in ("done", "skipped") for s in run["steps"])
    res = a.get("/api/workspace/results").json()
    assert res["use_case"]["id"] == "churn" and res["model"]["selected"] in ("logistic_regression", "gradient_boosting")
    assert res["model"]["card"]["beats_baseline_in_cv"] is True
    assert len(res["business_questions"]) == 7 and "deciles" in res["model"]
    assert "schema" not in str(res).lower() or "ws_" not in str(res)  # internal schema name is not exposed

    # --- insights: written for the run, labelled as template (no Snwolley key in the test env), no secrets inside
    ins = a.get("/api/workspace/insights").json()
    assert ins["mode"] in ("fallback", "ai") and ins["validation"]["ok"] and ins["document"]["executive_summary"]
    assert ins["facts"] and "ws_" not in str(ins)
    assert any(st["id"] == "insights" and st["status"] == "done" for st in run["steps"])
    again = a.post("/api/workspace/insights")  # rewrite the text from the saved results, no re-run
    assert again.status_code == 200 and again.json()["document"]["executive_summary"]

    # --- the SAME workspace switches use case: revenue (regression), no code change
    assert a.put("/api/workspace/use-case", json={"use_case_id": "revenue"}).status_code == 200
    assert a.post("/api/workspace/run").status_code == 202
    rev = a.get("/api/workspace/results").json()
    assert rev["use_case"]["problem_type"] == "regression" and "deciles" not in rev["model"]
    assert sorted(a.get("/api/me").json()["results_available"]) == ["churn", "revenue"]  # both results are kept

    # --- B cannot see A's anything
    me_b = b.get("/api/me").json()
    assert me_b["results_available"] == [] and me_b["data_loaded"] is False
    assert b.get("/api/workspace/results").status_code == 404
    assert b.get("/api/workspace/results", params={"use_case": "churn"}).status_code == 404
    assert b.get("/api/workspace/insights").status_code == 404
    assert b.get("/api/workspace/insights", params={"use_case": "churn"}).status_code == 404
    assert b.post("/api/workspace/insights", params={"use_case": "churn"}).status_code == 404
    assert b.get("/api/workspace/insights", params={"use_case": "../../etc"}).status_code == 404

    # --- regenerating data clears results that described the old data
    assert a.post("/api/workspace/data", json={"customers": 1000, "seed": 8}).status_code == 200
    assert a.get("/api/me").json()["results_available"] == []
