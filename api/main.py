"""The platform's web API and page.

Run from the project root:
    uvicorn api.main:app --reload
then open http://127.0.0.1:8000  (interactive API docs: /docs)

THE FLOW
    POST /api/signup            business signs up -> account + workspace (own DB schema)
    PUT  /api/workspace/use-case  choose which use case (YAML) the workspace works on
    POST /api/workspace/data    generate synthetic data and load it into the workspace
    POST /api/workspace/run     start the analysis pipeline (runs in the background)
    GET  /api/workspace/run     progress of the latest run
    GET  /api/workspace/results the finished results

TENANT ISOLATION (the rule that matters most)
    The browser never sends a schema name or a workspace id. The server reads the
    user id from the signed login token, looks up THAT user's workspace in the
    database, and uses its server-generated schema name. A user cannot ask for
    someone else's data because there is nowhere to put someone else's id.
"""
from __future__ import annotations

import json
import logging
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

import psycopg
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from api import security, store
from api.schemas import DataIn, LoginIn, SignupIn, UseCaseIn
from core.ai.insights import generate_insights
from core.ai.snwolley_adapter import get_provider
from core.config import get_settings
from core.report import build_report_html
from core.loader import LoaderError
from core.logging_setup import configure_logging
from core.ml_pipeline import MlError
from core.pipeline import initial_steps, load_synthetic_data, run_pipeline
from core.sql_runner import SqlError
from core.use_case import UseCaseError, list_use_cases, load_use_case

logger = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"
COOKIE = "session"


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(get_settings().log_level)  # fails fast with a clear message if .env is wrong
    store.init_platform()
    yield


app = FastAPI(title="AI-Augmented ML Analytics Platform", version="0.7", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "same-origin"
    if not request.url.path.startswith(("/docs", "/redoc", "/openapi")):
        # no inline scripts or styles, nothing loaded from other sites
        response.headers.setdefault("Content-Security-Policy", "default-src 'self'; frame-ancestors 'none'")
    return response


# ------------------------------------------------------------------ auth plumbing
def current_context(request: Request) -> tuple[dict, dict]:
    """The logged-in user and THEIR workspace. Token comes from the cookie or an Authorization: Bearer header."""
    token = request.cookies.get(COOKIE)
    auth = request.headers.get("authorization", "")
    if not token and auth.lower().startswith("bearer "):
        token = auth[7:].strip()
    user_id = security.read_token(token) if token else None
    ctx = store.get_context(user_id) if user_id else None
    if ctx is None:
        raise HTTPException(401, "Please log in")
    return ctx


def _set_session(response: Response, user_id: int) -> None:
    s = get_settings()
    response.set_cookie(
        COOKIE, security.create_token(user_id), max_age=s.session_hours * 3600, path="/",
        httponly=True,  # JavaScript cannot read it, so an injected script cannot steal it
        samesite="lax",  # not sent on cross-site POSTs (basic CSRF protection)
        secure=s.cookie_secure,
    )


def _results_path(schema: str, use_case_id: str) -> Path:
    return store.OUTPUT_ROOT / schema / use_case_id / "results.json"


def _jsonable_run(run: dict | None) -> dict | None:
    if not run:
        return None
    return {**run, "started_at": run["started_at"].isoformat(), "finished_at": run["finished_at"].isoformat() if run["finished_at"] else None}


# ------------------------------------------------------------------ pages
@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok"}


# ------------------------------------------------------------------ accounts
@app.post("/api/signup", status_code=201)
def signup(body: SignupIn, response: Response):
    try:
        user_id = store.create_account(body.email, security.hash_password(body.password), body.business_name)
    except store.EmailTaken:
        raise HTTPException(409, "That email is already registered")
    _set_session(response, user_id)
    return {"email": body.email, "business_name": body.business_name}


@app.post("/api/login")
def login(body: LoginIn, request: Request, response: Response):
    key = (request.client.host if request.client else "?", body.email)
    if security.throttle.blocked(key):
        raise HTTPException(429, "Too many failed attempts. Try again in a few minutes.")
    user = store.get_user_by_email(body.email)
    if user is None:
        security.burn_time()  # same response time whether or not the email exists
        ok = False
    else:
        ok = security.verify_password(body.password, user["password_hash"])
    if not ok:
        security.throttle.record_failure(key)
        raise HTTPException(401, "Invalid email or password")  # same message for both cases
    security.throttle.reset(key)
    _set_session(response, user["user_id"])
    return {"email": user["email"]}


@app.post("/api/logout")
def logout(response: Response):
    response.delete_cookie(COOKIE, path="/")
    return {"ok": True}


@app.get("/api/me")
def me(ctx=Depends(current_context)):
    user, ws = ctx
    results = [p.parent.name for p in (store.OUTPUT_ROOT / ws["schema_name"]).glob("*/results.json")]
    return {
        "email": user["email"],
        "business_name": ws["business_name"],
        "use_case_id": ws["use_case_id"],
        "data_loaded": ws["data_loaded_at"] is not None,
        "data_rows": ws["data_rows"],
        "latest_run": _jsonable_run(store.latest_run(ws["workspace_id"])),
        "results_available": sorted(results),
    }


# ------------------------------------------------------------------ the analysis flow
@app.get("/api/use-cases")
def use_cases(ctx=Depends(current_context)):
    return [
        {"id": u.id, "name": u.name, "description": u.description.strip(), "business_question": u.business_question.strip(),
         "problem_type": u.problem_type}
        for u in list_use_cases()
    ]


def _require_idle(ws: dict) -> None:
    run = store.latest_run(ws["workspace_id"])
    if run and run["status"] == "running":
        raise HTTPException(409, "An analysis is running. Wait for it to finish.")


@app.put("/api/workspace/use-case")
def choose_use_case(body: UseCaseIn, ctx=Depends(current_context)):
    _, ws = ctx
    try:
        uc = load_use_case(body.use_case_id)  # validates the id (no path tricks) and the whole config
    except UseCaseError:
        raise HTTPException(404, "Unknown use case")
    store.set_use_case(ws["workspace_id"], uc.id)
    return {"use_case_id": uc.id}


@app.post("/api/workspace/data")
def generate_data(body: DataIn, ctx=Depends(current_context)):
    _, ws = ctx
    if not ws["use_case_id"]:
        raise HTTPException(409, "Choose a use case first")
    _require_idle(ws)
    uc = load_use_case(ws["use_case_id"])
    try:
        counts = load_synthetic_data(ws["schema_name"], uc, body.customers, body.seed, store.DATA_ROOT / ws["schema_name"])
    except (LoaderError, psycopg.Error):
        logger.exception("Data load failed for workspace %s", ws["schema_name"])
        raise HTTPException(500, "Could not load the data")
    store.set_data_loaded(ws["workspace_id"], counts)
    shutil.rmtree(store.OUTPUT_ROOT / ws["schema_name"], ignore_errors=True)  # old results describe old data
    return {"rows": counts}


def _execute_run(run_id: int, schema: str, uc, out_dir: Path) -> None:
    """Runs in the background after the HTTP response has been sent."""
    try:
        run_pipeline(schema, uc, out_dir, on_progress=lambda steps: store.update_steps(run_id, steps))
        store.finish_run(run_id, "succeeded")
    except (SqlError, MlError, LoaderError, UseCaseError, ValueError) as exc:
        logger.warning("Run %s failed: %s", run_id, exc)
        store.finish_run(run_id, "failed", str(exc)[:500])
    except psycopg.Error as exc:
        logger.exception("Run %s: database error", run_id)
        store.finish_run(run_id, "failed", "Database error: " + (str(exc).strip().splitlines() or ["?"])[0][:200])
    except Exception:
        logger.exception("Run %s: unexpected error", run_id)
        store.finish_run(run_id, "failed", "Unexpected error. See the server log.")


@app.post("/api/workspace/run", status_code=202)
def start_run(background: BackgroundTasks, ctx=Depends(current_context)):
    _, ws = ctx
    if not ws["use_case_id"]:
        raise HTTPException(409, "Choose a use case first")
    if not ws["data_loaded_at"]:
        raise HTTPException(409, "Generate the data first")
    uc = load_use_case(ws["use_case_id"])
    try:
        run_id = store.start_run(ws["workspace_id"], uc.id, initial_steps())
    except store.RunInProgress:
        raise HTTPException(409, "An analysis is already running")
    background.add_task(_execute_run, run_id, ws["schema_name"], uc, store.OUTPUT_ROOT / ws["schema_name"] / uc.id)
    return {"run_id": run_id}


@app.get("/api/workspace/run")
def run_status(ctx=Depends(current_context)):
    _, ws = ctx
    return {"run": _jsonable_run(store.latest_run(ws["workspace_id"]))}


@app.get("/api/workspace/results")
def results(use_case: str | None = None, ctx=Depends(current_context)):
    _, ws = ctx
    uc_id = use_case or ws["use_case_id"]
    if not uc_id:
        raise HTTPException(404, "No results yet")
    try:
        uc_id = load_use_case(uc_id).id  # validated id; never used in a path before this check
    except UseCaseError:
        raise HTTPException(404, "Unknown use case")
    path = _results_path(ws["schema_name"], uc_id)
    if not path.is_file():
        raise HTTPException(404, "No results yet. Run the analysis first.")
    return json.loads(path.read_text(encoding="utf-8"))


def _validated_results_path(ctx, use_case: str | None) -> tuple[Path, str, str]:
    """Same checks as /results: validated use-case id first, then the workspace's own folder."""
    _, ws = ctx
    uc_id = use_case or ws["use_case_id"]
    if not uc_id:
        raise HTTPException(404, "No results yet")
    try:
        uc_id = load_use_case(uc_id).id
    except UseCaseError:
        raise HTTPException(404, "Unknown use case")
    return _results_path(ws["schema_name"], uc_id), ws["schema_name"], uc_id


@app.get("/api/workspace/insights")
def insights(use_case: str | None = None, ctx=Depends(current_context)):
    """The saved AI (or template) insights for the latest run, with the facts they cite."""
    path, _, _ = _validated_results_path(ctx, use_case)
    file = path.with_name("insights.json")
    if not file.is_file():
        raise HTTPException(404, "No insights yet. Run the analysis first.")
    return json.loads(file.read_text(encoding="utf-8"))


@app.post("/api/workspace/insights")
def regenerate_insights(use_case: str | None = None, ctx=Depends(current_context)):
    """Write the insights again from the saved results.json, without re-running the analysis.

    Useful after the AI provider is configured: the numbers stay exactly the same, only the text is redone.
    """
    path, _, uc_id = _validated_results_path(ctx, use_case)
    if not path.is_file():
        raise HTTPException(404, "No results yet. Run the analysis first.")
    result = generate_insights(json.loads(path.read_text(encoding="utf-8")), load_use_case(uc_id), get_provider(get_settings()))
    path.with_name("insights.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result


@app.get("/api/workspace/report", response_class=HTMLResponse)
def report(use_case: str | None = None, ctx=Depends(current_context)):
    """The printable business report for the latest run (use the browser's Print > Save as PDF)."""
    path, _, _ = _validated_results_path(ctx, use_case)
    if not path.is_file():
        raise HTTPException(404, "No results yet. Run the analysis first.")
    ins_file = path.with_name("insights.json")
    ins = json.loads(ins_file.read_text(encoding="utf-8")) if ins_file.is_file() else None
    html = build_report_html(json.loads(path.read_text(encoding="utf-8")), ins, ctx[1]["business_name"])
    # The report is one self-contained page: inline <style> and inline SVG, nothing loaded, no scripts.
    return HTMLResponse(html, headers={"Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; frame-ancestors 'none'"})
