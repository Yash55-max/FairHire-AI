from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from .auth import AuthenticatedUser, decode_token, hash_password, issue_token, verify_password
from .schemas import AuthLoginRequest, AuthRegisterRequest, AuthResponse, UserResponse
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
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
        raise HTTPException(status_code=401, detail="Unknown user")
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


@app.post("/upload")
def upload_placeholder() -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={
            "detail": "ML stack unavailable in Python 3.14 runtime. Install Python 3.11/3.12 and reinstall requirements to enable upload/train/bias/explain/report endpoints.",
        },
    )


@app.post("/train")
def train_placeholder() -> JSONResponse:
    return upload_placeholder()


@app.get("/bias")
def bias_placeholder() -> JSONResponse:
    return upload_placeholder()


@app.get("/explain")
def explain_placeholder() -> JSONResponse:
    return upload_placeholder()


@app.get("/report")
def report_placeholder() -> JSONResponse:
    return upload_placeholder()


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
