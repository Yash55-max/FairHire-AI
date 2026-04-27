from __future__ import annotations

import csv
import json
from datetime import datetime
from io import StringIO
import os
from pathlib import Path
from uuid import uuid4
from typing import Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from .auth import AuthenticatedUser, decode_token, hash_password, issue_token, verify_password
from .schemas import AuthLoginRequest, AuthRegisterRequest, AuthResponse, UploadResponse, UserResponse
from .store import InMemoryStore

ROOT_DIR = Path(__file__).resolve().parents[2]
WEB_DIR = ROOT_DIR / "frontend" / "WEB"
MOBILE_DIR = ROOT_DIR / "MOBILE"

WEB_PAGES = {
    "landing": "landing_page",
    "login": "login",
    "dashboard": "dashboard_overview",
    "upload": "upload_dataset",
    "model-analysis": "model_analysis",
    "bias-report": "bias_detection",
    "explainability": "explainability_engine",
    "reports": "audit_reports",
    "settings": "settings",
}

MOBILE_PAGES = {
    "landing": "landing_page_mobile_v2",
    "login": "login_mobile_v2",
    "dashboard": "dashboard_mobile_v2",
    "upload": "upload_mobile_v2",
    "model-analysis": "model_analysis_mobile_v3",
    "bias-report": "bias_detection_mobile_v2",
    "explainability": "explainability_mobile_v2",
    "reports": "reports_mobile_v2",
}

app = FastAPI(
    title="FairHire AI Backend",
    description="Runtime backend service",
    version="1.0.0",
)

store = InMemoryStore()
DATA_DIR = ROOT_DIR / "backend" / "data"
DATASETS_DIR = DATA_DIR / "datasets"
STATE_FILE = DATA_DIR / "state.json"

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        o.strip()
        for o in os.getenv(
            "ALLOWED_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173,https://fairhire-67f38.web.app,https://fairhire-67f38.firebaseapp.com",
        ).split(",")
        if o.strip()
    ],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_page(platform_map: dict[str, str], base_dir: Path, page: str) -> FileResponse:
    folder = platform_map.get(page)
    if not folder:
        raise HTTPException(status_code=404, detail=f"Page '{page}' not found")

    file_path = base_dir / folder / "code.html"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Missing view file: {file_path}")

    return FileResponse(str(file_path), media_type="text/html")


def _generate_user_id() -> str:
    return f"usr_{uuid4().hex[:10]}"


def _generate_employee_id() -> str:
    return f"EMP-{uuid4().hex[:6].upper()}"


def _ensure_user_identity(user_record: dict[str, object]) -> bool:
    changed = False
    if not user_record.get("user_id"):
        user_record["user_id"] = _generate_user_id()
        changed = True
    if not user_record.get("employee_id"):
        user_record["employee_id"] = _generate_employee_id()
        changed = True
    return changed


def _record_to_user(record: dict[str, object]) -> AuthenticatedUser:
    return AuthenticatedUser(
        user_id=str(record["user_id"]),
        employee_id=str(record["employee_id"]),
        email=str(record["email"]),
        name=str(record["name"]),
        role=str(record.get("role", "analyst")),
        created_at=str(record["created_at"]),
    )


def _serialize_user(user: AuthenticatedUser) -> UserResponse:
    created_at = user.created_at.isoformat() if hasattr(user.created_at, "isoformat") else str(user.created_at)
    return UserResponse(
        user_id=user.user_id,
        employee_id=user.employee_id,
        email=user.email,
        name=user.name,
        role=user.role,
        created_at=created_at,
    )


def _current_user(authorization: str | None = Header(default=None)) -> AuthenticatedUser:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")

    token = authorization.removeprefix("Bearer ").strip()
    try:
        payload = decode_token(token)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    email = str(payload.get("sub", "")).lower()
    user_record = store.get_user(email)
    if not user_record:
        # Cloud runtimes are stateless. Rehydrate a minimal profile from token subject
        # so authenticated requests remain valid across instance restarts/scale-outs.
        now = datetime.utcnow().isoformat()
        user_record = {
            "user_id": _generate_user_id(),
            "employee_id": _generate_employee_id(),
            "email": email,
            "name": email.split("@", 1)[0] if email else "user",
            "role": "analyst",
            "created_at": now,
            "password_salt": "",
            "password_hash": "",
        }
        store.put_user(user_record)
    if _ensure_user_identity(user_record):
        store.put_user(user_record)
    return AuthenticatedUser(
        user_id=str(user_record["user_id"]),
        employee_id=str(user_record["employee_id"]),
        email=user_record["email"],
        name=user_record["name"],
        role=user_record.get("role", "analyst"),
        created_at=user_record["created_at"],
    )


def _fallback_target_suggestions(columns: list[str]) -> list[str]:
    preferred = ["hired", "target", "label", "outcome", "decision", "selected", "approved", "rejected"]
    lowered = {str(column).lower(): str(column) for column in columns}
    suggestions = [lowered[key] for key in preferred if key in lowered]
    if suggestions:
        return suggestions
    return [str(columns[-1])] if columns else []


def _infer_value_type(value: Any) -> str:
    if value is None:
        return "string"
    text = str(value).strip()
    if not text:
        return "string"
    try:
        int(text)
        return "integer"
    except ValueError:
        pass
    try:
        float(text)
        return "number"
    except ValueError:
        pass
    if text.lower() in {"true", "false"}:
        return "boolean"
    return "string"


def _ml_unavailable_response() -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={
            "detail": "ML stack unavailable in Python 3.14 runtime. Install Python 3.11/3.12 and reinstall requirements to enable train/bias/explain/report endpoints.",
        },
    )


def _serialize_dataset_to_state(dataset_id: str, filename: str, rows: list[dict[str, Any]], columns: list[str]) -> None:
    DATASETs_DIR = DATASETS_DIR
    DATASETs_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    dataset_path = DATASETs_DIR / f"{dataset_id}.csv"
    with dataset_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})

    record = {
        "dataset_id": dataset_id,
        "rows": int(len(rows)),
        "columns": columns,
        "metadata": {
            "filename": filename,
        },
        "path": str(dataset_path.relative_to(DATA_DIR)),
    }

    state = {"datasets": {}, "runs": {}, "users": {}}
    if STATE_FILE.exists():
        try:
            with STATE_FILE.open("r", encoding="utf-8") as file:
                state = json.load(file)
        except Exception:
            state = {"datasets": {}, "runs": {}, "users": {}}

    state.setdefault("datasets", {})[dataset_id] = record
    with STATE_FILE.open("w", encoding="utf-8") as file:
        json.dump(state, file, indent=2)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    web_links = "".join([f'<li><a href="/web/{slug}">{slug}</a></li>' for slug in WEB_PAGES])
    mobile_links = "".join([f'<li><a href="/mobile/{slug}">{slug}</a></li>' for slug in MOBILE_PAGES])
    return f"""
    <html>
      <head><title>FairHire AI Launcher</title></head>
      <body style=\"font-family: Inter, sans-serif; padding: 24px;\">
        <h1>FairHire AI Backend Running</h1>
        <p>API Docs: <a href=\"/docs\">/docs</a></p>
        <h2>Web Screens</h2>
        <ul>{web_links}</ul>
        <h2>Mobile Screens</h2>
        <ul>{mobile_links}</ul>
      </body>
    </html>
    """


@app.get("/web/{page}")
def web_page(page: str) -> FileResponse:
    return _load_page(WEB_PAGES, WEB_DIR, page)


@app.get("/mobile/{page}")
def mobile_page(page: str) -> FileResponse:
    return _load_page(MOBILE_PAGES, MOBILE_DIR, page)


@app.post("/upload", response_model=UploadResponse)
async def upload_dataset(file: UploadFile = File(...), target_column: str | None = Form(default=None)) -> UploadResponse:
    filename = (file.filename or "dataset.csv").strip() or "dataset.csv"
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    text = raw_bytes.decode("utf-8-sig", errors="replace")
    rows: list[dict[str, Any]] = []
    columns: list[str] = []

    if filename.lower().endswith(".json"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"Failed to parse JSON file: {exc}") from exc
        if isinstance(parsed, list):
            rows = [row for row in parsed if isinstance(row, dict)]
        elif isinstance(parsed, dict):
            rows = [parsed]
        else:
            raise HTTPException(status_code=400, detail="JSON upload must be an object or a list of objects")
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(str(key))
    else:
        reader = csv.DictReader(StringIO(text))
        columns = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
        if not columns and rows:
            columns = list(rows[0].keys())

    if not columns:
        raise HTTPException(status_code=400, detail="No columns found in uploaded file")

    normalized_rows = [{column: row.get(column, "") for column in columns} for row in rows]
    dataset_id = f"ds_{uuid4().hex[:10]}"
    preview = normalized_rows[:8]
    schema = {column: _infer_value_type(next((row.get(column) for row in normalized_rows if str(row.get(column, "")).strip()), "")) for column in columns}
    null_counts = {
        column: sum(1 for row in normalized_rows if str(row.get(column, "")).strip() == "")
        for column in columns
    }

    _serialize_dataset_to_state(dataset_id, filename, normalized_rows, columns)

    target_suggestions = _fallback_target_suggestions(columns)
    if target_column and target_column not in target_suggestions and target_column in columns:
        target_suggestions = [target_column, *[column for column in target_suggestions if column != target_column]]

    return UploadResponse(
        dataset_id=dataset_id,
        filename=filename,
        rows=len(normalized_rows),
        columns=columns,
        target_suggestions=target_suggestions,
        preview=preview,
        schema=schema,
        null_counts=null_counts,
    )


@app.post("/train")
def train_placeholder() -> JSONResponse:
    return _ml_unavailable_response()


@app.get("/bias")
def bias_placeholder() -> JSONResponse:
    return _ml_unavailable_response()


@app.get("/explain")
def explain_placeholder() -> JSONResponse:
    return _ml_unavailable_response()


@app.get("/report")
def report_placeholder() -> JSONResponse:
    return _ml_unavailable_response()


@app.post("/auth/register", response_model=AuthResponse)
def register_user(payload: AuthRegisterRequest) -> AuthResponse:
    email = payload.email.strip().lower()
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="A valid email address is required")
    if len(payload.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters long")
    if store.get_user(email):
        raise HTTPException(status_code=409, detail="User already exists")

    now = datetime.utcnow().isoformat()
    password_material = hash_password(payload.password)
    user_record = {
        "user_id": _generate_user_id(),
        "employee_id": payload.employee_id.strip() if payload.employee_id else _generate_employee_id(),
        "email": email,
        "name": payload.name.strip() if payload.name else email.split("@", 1)[0],
        "role": "analyst",
        "created_at": now,
        "password_salt": password_material["salt"],
        "password_hash": password_material["hash"],
    }
    store.put_user(user_record)
    return AuthResponse(token=issue_token(email), user=_serialize_user(_record_to_user(user_record)))


@app.get("/auth/exists")
def check_user_exists(email: str) -> dict[str, bool]:
    email = email.strip().lower()
    user_record = store.get_user(email)
    return {"exists": bool(user_record)}


@app.post("/auth/login", response_model=AuthResponse)
def login_user(payload: AuthLoginRequest) -> AuthResponse:
    email = payload.email.strip().lower()
    user_record = store.get_user(email)
    if not user_record:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if not verify_password(payload.password, str(user_record["password_salt"]), str(user_record["password_hash"])):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if _ensure_user_identity(user_record):
        store.put_user(user_record)
    return AuthResponse(token=issue_token(email), user=_serialize_user(_record_to_user(user_record)))


@app.get("/auth/me", response_model=UserResponse)
def get_current_user(user: AuthenticatedUser = Depends(_current_user)) -> UserResponse:
    return _serialize_user(user)
