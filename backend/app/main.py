from __future__ import annotations

import json
import os
from datetime import datetime
from io import BytesIO
from pathlib import Path
from uuid import uuid4
from typing import Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

from .auth import AuthenticatedUser, decode_token, hash_password, issue_token, verify_password
from .jobs import JobManager
from .pdf_export import build_report_pdf
from .validation import validate_upload
from .debiasing import DebiasEngine
from .schemas import (
    AuthLoginRequest,
    AuthRegisterRequest,
    AuthResponse,
    BiasResponse,
    ExplainResponse,
    JobStatusResponse,
    JobSubmissionResponse,
    PdfReportResponse,
    ReportResponse,
    TrainRequest,
    TrainResponse,
    UploadResponse,
    UserResponse,
)
from .store import InMemoryStore, TrainingRun
try:
    import pandas as pd
except ImportError:  # pragma: no cover - runtime fallback for Python 3.14 environments
    pd = None

try:
    from .ml_pipeline import compute_bias, compute_explainability, suggest_target_columns, train_pipeline
    ML_AVAILABLE = True
except Exception as exc:  # noqa: BLE001
    compute_bias = None
    compute_explainability = None
    suggest_target_columns = None
    train_pipeline = None
    ML_AVAILABLE = False
    ML_IMPORT_ERROR = str(exc)


ML_UNAVAILABLE_DETAIL = (
    "ML audit features are unavailable in this runtime. Install the project dependencies on Python 3.11 or 3.12 "
    "to enable upload, train, bias, explain, and report endpoints."
)

ROOT_DIR = Path(__file__).resolve().parents[2]
WEB_DIR = ROOT_DIR / "frontend" / "WEB"
MOBILE_DIR = ROOT_DIR / "MOBILE"
DATA_DIR = ROOT_DIR / "backend" / "data"
RUNS_JSON = DATA_DIR / "runs.json"

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

store = InMemoryStore()
jobs = JobManager()

app = FastAPI(
    title="FairHire AI Backend",
    description="Responsible AI audit API implementing PRD/TRD requirements.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _persist_run_summary(run: TrainingRun, bias_payload: dict | None = None, explain_payload: dict | None = None) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    if RUNS_JSON.exists():
        with RUNS_JSON.open("r", encoding="utf-8") as f:
            records = json.load(f)

    record = {
        "run_id": run.run_id,
        "dataset_id": run.dataset_id,
        "model_type": run.model_type,
        "target_column": run.target_column,
        "created_at": run.created_at.isoformat(),
        "metrics": run.metrics,
        "bias": bias_payload,
        "explain": explain_payload,
    }
    records.append(record)

    with RUNS_JSON.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


def _read_uploaded_file(upload: UploadFile, content: bytes) -> Any:
    if pd is None:
        raise HTTPException(status_code=503, detail=ML_UNAVAILABLE_DETAIL)

    filename = (upload.filename or "dataset.csv").lower()
    try:
        if filename.endswith(".csv"):
            return pd.read_csv(BytesIO(content))
        if filename.endswith(".json"):
            return pd.read_json(BytesIO(content))
        if filename.endswith(".xlsx") or filename.endswith(".xls"):
            return pd.read_excel(BytesIO(content))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {exc}") from exc

    raise HTTPException(status_code=400, detail="Unsupported file format. Use CSV, JSON, or XLSX.")


def _load_page(platform_map: dict[str, str], base_dir: Path, page: str) -> FileResponse:
    folder = platform_map.get(page)
    if not folder:
        raise HTTPException(status_code=404, detail=f"Page '{page}' not found")

    file_path = base_dir / folder / "code.html"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Missing view file: {file_path}")

    return FileResponse(str(file_path), media_type="text/html")


def _fallback_target_suggestions(columns: list[str]) -> list[str]:
    preferred = ["hired", "target", "label", "outcome", "decision", "selected", "approved", "rejected"]
    lowered = {str(col).lower(): str(col) for col in columns}
    suggestions = [lowered[key] for key in preferred if key in lowered]
    if suggestions:
        return suggestions
    if columns:
        return [str(columns[-1])]
    return []


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


def _record_to_user(record: dict[str, object]) -> AuthenticatedUser:
    return AuthenticatedUser(
        user_id=str(record["user_id"]),
        employee_id=str(record["employee_id"]),
        email=str(record["email"]),
        name=str(record["name"]),
        role=str(record.get("role", "analyst")),
        created_at=str(record["created_at"]),
    )


def _train_job(payload: TrainRequest) -> dict[str, object]:
    if not ML_AVAILABLE or train_pipeline is None:
        raise HTTPException(status_code=503, detail=ML_UNAVAILABLE_DETAIL)

    try:
        frame = store.get_dataset(payload.dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        artifacts = train_pipeline(
            frame=frame,
            target_column=payload.target_column,
            model_type=payload.model_type,
            test_size=payload.test_size,
            random_state=payload.random_state,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    diagnostics = list(artifacts.metrics.get("diagnostics", []))
    fairness_summary: dict[str, object] | None = None

    sensitive_column = (payload.sensitive_column or "").strip() if payload.sensitive_column else None
    if payload.include_fairness_proof and sensitive_column:
        baseline_input = artifacts.test_frame.drop(columns=[payload.target_column], errors="ignore")
        baseline_input[payload.target_column] = artifacts.y_true
        if sensitive_column in baseline_input.columns and compute_bias is not None:
            try:
                baseline_bias = compute_bias(
                    test_frame=baseline_input,
                    y_true=artifacts.y_true,
                    y_pred=artifacts.y_pred,
                    sensitive_column=sensitive_column,
                )

                engine = DebiasEngine(frame, payload.target_column)
                mitigated_frame, masked_columns = engine.auto_mask()
                after_bias: dict[str, object] | None = None

                if mitigated_frame.shape[1] > 1 and sensitive_column in frame.columns:
                    mitigated_artifacts = train_pipeline(
                        frame=mitigated_frame,
                        target_column=payload.target_column,
                        model_type=payload.model_type,
                        test_size=payload.test_size,
                        random_state=payload.random_state,
                    )
                    mitigated_input = mitigated_artifacts.test_frame.drop(columns=[payload.target_column], errors="ignore")
                    mitigated_input[payload.target_column] = mitigated_artifacts.y_true

                    if sensitive_column in mitigated_input.columns:
                        after_bias = compute_bias(
                            test_frame=mitigated_input,
                            y_true=mitigated_artifacts.y_true,
                            y_pred=mitigated_artifacts.y_pred,
                            sensitive_column=sensitive_column,
                        )

                before_block = {
                    "fairness_index": float(baseline_bias["fairness_index"]),
                    "demographic_parity_difference": float(baseline_bias["demographic_parity_difference"]),
                    "equal_opportunity_difference": float(baseline_bias["equal_opportunity_difference"]),
                    "selection_rate_by_group": baseline_bias["selection_rate_by_group"],
                }
                after_block = None
                delta_fairness = None
                delta_dpd = None
                if after_bias is not None:
                    after_block = {
                        "fairness_index": float(after_bias["fairness_index"]),
                        "demographic_parity_difference": float(after_bias["demographic_parity_difference"]),
                        "equal_opportunity_difference": float(after_bias["equal_opportunity_difference"]),
                        "selection_rate_by_group": after_bias["selection_rate_by_group"],
                    }
                    delta_fairness = float(after_block["fairness_index"] - before_block["fairness_index"])
                    delta_dpd = float(before_block["demographic_parity_difference"] - after_block["demographic_parity_difference"])
                    if delta_fairness >= 0:
                        diagnostics.append("Fairness mitigation improved fairness index while preserving model utility.")
                    else:
                        diagnostics.append("Fairness mitigation did not improve fairness index; review masking strategy and sensitive attributes.")
                else:
                    diagnostics.append("Fairness baseline computed, but mitigation-after metrics were not available for the selected sensitive column.")

                fairness_summary = {
                    "sensitive_column": sensitive_column,
                    "before": before_block,
                    "after": after_block,
                    "delta_fairness_index": delta_fairness,
                    "delta_demographic_parity_difference": delta_dpd,
                    "mitigated_columns": masked_columns,
                }
            except Exception as exc:  # noqa: BLE001
                diagnostics.append(f"Fairness proof generation skipped: {exc}")
        else:
            diagnostics.append(f"Sensitive column '{sensitive_column}' not found in test frame; fairness proof not computed.")

    artifacts.metrics["diagnostics"] = diagnostics
    artifacts.metrics["fairness"] = fairness_summary

    run_id = f"run_{uuid4().hex[:10]}"
    run = TrainingRun(
        run_id=run_id,
        dataset_id=payload.dataset_id,
        model_type=payload.model_type,
        target_column=payload.target_column,
        model=artifacts.model,
        test_frame=artifacts.test_frame,
        y_true=artifacts.y_true,
        y_pred=artifacts.y_pred,
        metrics=artifacts.metrics,
        created_at=datetime.utcnow(),
    )
    store.put_run(run)

    preview_df = artifacts.test_frame.copy()
    preview_df["prediction"] = artifacts.y_pred
    preview = preview_df.head(10).fillna("").to_dict(orient="records")
    response = TrainResponse(
        run_id=run_id,
        dataset_id=payload.dataset_id,
        model_type=payload.model_type,
        target_column=payload.target_column,
        accuracy=artifacts.metrics["accuracy"],
        precision=artifacts.metrics["precision"],
        recall=artifacts.metrics["recall"],
        f1_score=artifacts.metrics["f1_score"],
        confusion_matrix=artifacts.metrics["confusion_matrix"],
        prediction_preview=preview,
        feature_count=int(artifacts.metrics.get("feature_count", 0)),
        train_rows=int(artifacts.metrics.get("train_rows", 0)),
        test_rows=int(artifacts.metrics.get("test_rows", 0)),
        cv_best_score=artifacts.metrics.get("cv_best_score"),
        cv_score_std=artifacts.metrics.get("cv_score_std"),
        cv_folds=artifacts.metrics.get("cv_folds"),
        leakage_dropped_columns=list(artifacts.metrics.get("leakage_dropped_columns", [])),
        validation_notes=list(artifacts.metrics.get("validation_notes", [])),
        diagnostics=diagnostics,
        fairness=fairness_summary,
    )
    _persist_run_summary(run)
    return response.model_dump()


def _explain_job(run_id: str, sample_size: int) -> dict[str, object]:
    if not ML_AVAILABLE or compute_explainability is None:
        raise HTTPException(status_code=503, detail=ML_UNAVAILABLE_DETAIL)

    try:
        run = store.get_run(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    feature_frame = run.test_frame.drop(columns=[run.target_column], errors="ignore")
    try:
        explain = compute_explainability(run.model, feature_frame, sample_size=sample_size)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    payload = ExplainResponse(run_id=run_id, **explain)
    _persist_run_summary(run, explain_payload=payload.model_dump())
    return payload.model_dump()


def _build_report(run_id: str, sensitive_column: str, sample_size: int) -> ReportResponse:
    train_run = train_metrics(run_id)
    bias = bias_metrics(run_id=run_id, sensitive_column=sensitive_column)
    explain_payload = ExplainResponse(**_explain_job(run_id, sample_size))
    return ReportResponse(run_id=run_id, train=train_run, bias=bias, explain=explain_payload)


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
    user = _record_to_user(user_record)
    return AuthResponse(token=issue_token(email), user=_serialize_user(user))


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


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str) -> JobStatusResponse:
    try:
        job = jobs.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return JobStatusResponse(
        job_id=job.job_id,
        kind=job.kind,
        status=job.status,
        message=job.message,
        result=job.result,
        error=job.error,
    )


@app.get("/report/pdf")
def download_report_pdf(run_id: str, sensitive_column: str = "gender", sample_size: int = 40) -> Response:
    report = _build_report(run_id, sensitive_column, sample_size)
    try:
        pdf_bytes = build_report_pdf(report.model_dump())
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    filename = f"fairhire-report-{run_id}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
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
        <h1>FairHire AI Full-Stack Application</h1>
        <p>Backend docs: <a href=\"/docs\">/docs</a></p>
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
    if pd is None:
        raise HTTPException(
            status_code=503,
            detail="Dataset upload requires pandas in this runtime. Install dependencies or use Python 3.11/3.12.",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    validate_upload(file, content)

    frame = _read_uploaded_file(file, content)
    if frame.empty:
        raise HTTPException(status_code=400, detail="Parsed dataset is empty")

    dataset_id = f"ds_{uuid4().hex[:10]}"
    frame.columns = [str(c).strip() for c in frame.columns]
    store.put_dataset(dataset_id, frame, metadata={"filename": file.filename or "dataset.csv"})

    suggestions = suggest_target_columns(frame) if suggest_target_columns is not None else _fallback_target_suggestions(list(frame.columns))
    if target_column and target_column in frame.columns and target_column not in suggestions:
        suggestions = [target_column] + suggestions

    preview = frame.head(8).fillna("").to_dict(orient="records")
    return UploadResponse(
        dataset_id=dataset_id,
        filename=file.filename or "dataset.csv",
        rows=int(frame.shape[0]),
        columns=list(frame.columns),
        target_suggestions=suggestions,
        preview=preview,
    )


@app.post("/train", response_model=JobSubmissionResponse)
def train_model(payload: TrainRequest) -> JobSubmissionResponse:
    if payload.async_job:
        job = jobs.submit("train", _train_job, payload)
        return JobSubmissionResponse(job_id=job.job_id, kind=job.kind, status=job.status, message="Training queued")

    response = _train_job(payload)
    return JobSubmissionResponse(job_id=f"train_{uuid4().hex[:10]}", kind="train", status="completed", message="Training completed", result=response)


@app.get("/bias", response_model=BiasResponse)
def bias_metrics(run_id: str, sensitive_column: str = "gender") -> BiasResponse:
    if not ML_AVAILABLE or compute_bias is None:
        raise HTTPException(status_code=503, detail=ML_UNAVAILABLE_DETAIL)

    try:
        run = store.get_run(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        bias = compute_bias(
            test_frame=run.test_frame,
            y_true=run.y_true,
            y_pred=run.y_pred,
            sensitive_column=sensitive_column,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    payload = BiasResponse(
        run_id=run_id,
        sensitive_column=bias["sensitive_column"],
        demographic_parity_difference=bias["demographic_parity_difference"],
        equal_opportunity_difference=bias["equal_opportunity_difference"],
        selection_rate_by_group=bias["selection_rate_by_group"],
        true_positive_rate_by_group=bias["true_positive_rate_by_group"],
        fairness_index=bias["fairness_index"],
    )
    _persist_run_summary(run, bias_payload=payload.model_dump())
    return payload


@app.get("/explain", response_model=JobSubmissionResponse)
def explain_metrics(run_id: str, sample_size: int = 40, async_job: bool = True) -> JobSubmissionResponse:
    if async_job:
        job = jobs.submit("explain", _explain_job, run_id, sample_size)
        return JobSubmissionResponse(job_id=job.job_id, kind=job.kind, status=job.status, message="Explainability queued")

    response = _explain_job(run_id, sample_size)
    return JobSubmissionResponse(job_id=f"explain_{uuid4().hex[:10]}", kind="explain", status="completed", message="Explainability completed", result=response)


@app.get("/report", response_model=ReportResponse)
def report(run_id: str, sensitive_column: str = "gender", sample_size: int = 40) -> ReportResponse:
    if not ML_AVAILABLE:
        raise HTTPException(status_code=503, detail=ML_UNAVAILABLE_DETAIL)

    return _build_report(run_id, sensitive_column, sample_size)


@app.get("/train/{run_id}", response_model=TrainResponse)
def train_metrics(run_id: str) -> TrainResponse:
    try:
        run = store.get_run(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    preview_df = run.test_frame.copy()
    preview_df["prediction"] = run.y_pred
    preview = preview_df.head(10).fillna("").to_dict(orient="records")

    return TrainResponse(
        run_id=run_id,
        dataset_id=run.dataset_id,
        model_type=run.model_type,
        target_column=run.target_column,
        accuracy=run.metrics["accuracy"],
        precision=run.metrics["precision"],
        recall=run.metrics["recall"],
        f1_score=run.metrics["f1_score"],
        confusion_matrix=run.metrics["confusion_matrix"],
        prediction_preview=preview,
        feature_count=int(run.metrics.get("feature_count", 0)),
        train_rows=int(run.metrics.get("train_rows", 0)),
        test_rows=int(run.metrics.get("test_rows", 0)),
        cv_best_score=run.metrics.get("cv_best_score"),
        cv_score_std=run.metrics.get("cv_score_std"),
        cv_folds=run.metrics.get("cv_folds"),
        leakage_dropped_columns=list(run.metrics.get("leakage_dropped_columns", [])),
        validation_notes=list(run.metrics.get("validation_notes", [])),
        diagnostics=list(run.metrics.get("diagnostics", run.metrics.get("validation_notes", []))),
        fairness=run.metrics.get("fairness"),
    )
