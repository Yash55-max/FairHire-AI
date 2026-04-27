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
from dotenv import load_dotenv

load_dotenv()

from .auth import AuthenticatedUser, decode_token, hash_password, issue_token, verify_password
from .jobs import JobManager
from .pdf_export import build_report_pdf
from .validation import validate_upload
from .debiasing import DebiasEngine
from .persistence import persistence
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
    from .assistant import chat
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
    allow_origins=[
        o.strip()
        for o in os.getenv(
            "ALLOWED_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173,https://fairhire-67f38.web.app,https://fairhire-67f38.firebaseapp.com",
        ).split(",")
        if o.strip()
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _persist_run_summary(run: TrainingRun, email: str = "anonymous", bias_payload: dict | None = None, explain_payload: dict | None = None) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    if RUNS_JSON.exists():
        with RUNS_JSON.open("r", encoding="utf-8") as f:
            records = json.load(f)

    record = {
        "run_id": run.run_id,
        "user_email": email,
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
    
    # Also save to Firestore for history tracking
    persistence.save_analysis(
        run_id=run.run_id,
        email=email,
        metrics=run.metrics,
        bias_data=bias_payload
    )


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


def _compute_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    truth = pd.Series(y_true).astype(str).reset_index(drop=True)
    pred = pd.Series(y_pred).astype(str).reset_index(drop=True)
    if truth.empty:
        return 0.0
    return float((truth == pred).mean())


def _build_binary_threshold_predictions(model: Any, feature_frame: pd.DataFrame, y_true: pd.Series, threshold: float) -> pd.Series | None:
    if not hasattr(model, "predict_proba"):
        return None

    labels = sorted(pd.Series(y_true).dropna().unique().tolist(), key=lambda x: str(x))
    if len(labels) != 2:
        return None

    negative_label, positive_label = labels[0], labels[-1]
    probabilities = model.predict_proba(feature_frame)

    classes = None
    classifier = getattr(model, "named_steps", {}).get("classifier") if hasattr(model, "named_steps") else None
    if classifier is not None and hasattr(classifier, "classes_"):
        classes = list(classifier.classes_)

    if classes and positive_label in classes:
        positive_index = classes.index(positive_label)
    else:
        positive_index = probabilities.shape[1] - 1

    positive_probs = probabilities[:, positive_index]
    predictions = [positive_label if p >= threshold else negative_label for p in positive_probs]
    return pd.Series(predictions, index=feature_frame.index)


def _build_fairness_block(bias_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "fairness_index": float(bias_payload["fairness_index"]),
        "demographic_parity_difference": float(bias_payload["demographic_parity_difference"]),
        "equal_opportunity_difference": float(bias_payload["equal_opportunity_difference"]),
        "selection_rate_by_group": bias_payload["selection_rate_by_group"],
    }


def _make_reweighted_frame(frame: pd.DataFrame, target_column: str, random_state: int) -> pd.DataFrame:
    value_counts = frame[target_column].value_counts(dropna=False)
    if value_counts.empty:
        return frame

    inverse_weights = frame[target_column].map(lambda val: 1.0 / float(value_counts.get(val, 1.0)))
    sampled_indices = frame.sample(
        n=len(frame),
        replace=True,
        weights=inverse_weights,
        random_state=random_state,
    ).index
    return frame.loc[sampled_indices].copy()


def _train_job(payload: TrainRequest) -> dict[str, object]:
    if not ML_AVAILABLE or train_pipeline is None:
        raise HTTPException(status_code=503, detail=ML_UNAVAILABLE_DETAIL)

    try:
        frame = store.get_dataset(payload.dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    pre_diagnostics: list[str] = []
    required_position = (payload.required_position or "").strip()
    if required_position:
        role_columns = [column for column in ("role_applied", "job_role", "position", "role") if column in frame.columns]
        if not role_columns:
            raise HTTPException(
                status_code=400,
                detail="Required position was provided, but no role column was found. Add one of: role_applied, job_role, position, role.",
            )

        role_column = role_columns[0]
        filtered_frame = frame[
            frame[role_column].astype(str).str.strip().str.casefold() == required_position.casefold()
        ].copy()
        if filtered_frame.empty:
            raise HTTPException(
                status_code=400,
                detail=f"No records found for required position '{required_position}' in column '{role_column}'.",
            )

        pre_diagnostics.append(
            f"Training scope filtered to required position '{required_position}' using column '{role_column}' ({len(filtered_frame)} rows)."
        )
        frame = filtered_frame

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

    diagnostics = [*pre_diagnostics, *list(artifacts.metrics.get("diagnostics", []))]
    fairness_summary: dict[str, object] | None = None

    sensitive_column = (payload.sensitive_column or "").strip() if payload.sensitive_column else None
    if payload.include_fairness_proof and sensitive_column:
        baseline_features = artifacts.test_frame.drop(columns=[payload.target_column], errors="ignore")
        baseline_eval_frame = baseline_features.copy()
        baseline_eval_frame[payload.target_column] = artifacts.y_true

        if sensitive_column in baseline_eval_frame.columns and compute_bias is not None:
            try:
                baseline_bias = compute_bias(
                    test_frame=baseline_eval_frame,
                    y_true=artifacts.y_true,
                    y_pred=artifacts.y_pred,
                    sensitive_column=sensitive_column,
                )

                baseline_accuracy = float(artifacts.metrics.get("accuracy", _compute_accuracy(artifacts.y_true, artifacts.y_pred)))
                max_accuracy_drop = 0.03
                candidates: list[dict[str, Any]] = []

                def register_candidate(strategy: str, parameter: Any, acc: float, bias_payload: dict[str, Any], details: dict[str, Any] | None = None) -> None:
                    acc_drop = baseline_accuracy - float(acc)
                    if acc_drop > max_accuracy_drop:
                        return
                    candidates.append(
                        {
                            "strategy": strategy,
                            "parameter": parameter,
                            "accuracy": float(acc),
                            "acc_drop": float(acc_drop),
                            "gap": float(bias_payload["demographic_parity_difference"]),
                            "bias": bias_payload,
                            "details": details or {},
                        }
                    )

                # Strategy 1: threshold tuning on baseline probabilities.
                try:
                    for threshold in (0.4, 0.5, 0.6):
                        threshold_pred = _build_binary_threshold_predictions(
                            model=artifacts.model,
                            feature_frame=baseline_features,
                            y_true=artifacts.y_true,
                            threshold=threshold,
                        )
                        if threshold_pred is None:
                            break

                        threshold_bias = compute_bias(
                            test_frame=baseline_eval_frame,
                            y_true=artifacts.y_true,
                            y_pred=threshold_pred,
                            sensitive_column=sensitive_column,
                        )
                        threshold_acc = _compute_accuracy(artifacts.y_true, threshold_pred)
                        register_candidate("threshold", threshold, threshold_acc, threshold_bias)
                except Exception as exc:  # noqa: BLE001
                    diagnostics.append(f"Threshold mitigation skipped: {exc}")

                # Strategy 2: class reweighting via balanced bootstrap.
                try:
                    reweighted_frame = _make_reweighted_frame(frame, payload.target_column, payload.random_state)
                    reweighted_artifacts = train_pipeline(
                        frame=reweighted_frame,
                        target_column=payload.target_column,
                        model_type=payload.model_type,
                        test_size=payload.test_size,
                        random_state=payload.random_state,
                    )
                    reweighted_eval = reweighted_artifacts.test_frame.drop(columns=[payload.target_column], errors="ignore")
                    reweighted_eval[payload.target_column] = reweighted_artifacts.y_true
                    if sensitive_column not in reweighted_eval.columns and sensitive_column in reweighted_frame.columns:
                        reweighted_eval[sensitive_column] = reweighted_frame.loc[reweighted_artifacts.y_true.index, sensitive_column]

                    if sensitive_column in reweighted_eval.columns:
                        reweighted_bias = compute_bias(
                            test_frame=reweighted_eval,
                            y_true=reweighted_artifacts.y_true,
                            y_pred=reweighted_artifacts.y_pred,
                            sensitive_column=sensitive_column,
                        )
                        reweighted_acc = float(reweighted_artifacts.metrics.get("accuracy", _compute_accuracy(reweighted_artifacts.y_true, reweighted_artifacts.y_pred)))
                        register_candidate("reweight", "balanced_bootstrap", reweighted_acc, reweighted_bias)
                except Exception as exc:  # noqa: BLE001
                    diagnostics.append(f"Reweight mitigation skipped: {exc}")

                # Strategy 3: feature masking through de-bias engine.
                try:
                    engine = DebiasEngine(frame, payload.target_column)
                    masked_frame, masked_columns = engine.auto_mask()
                    if masked_frame.shape[1] > 1 and sensitive_column in frame.columns:
                        masked_artifacts = train_pipeline(
                            frame=masked_frame,
                            target_column=payload.target_column,
                            model_type=payload.model_type,
                            test_size=payload.test_size,
                            random_state=payload.random_state,
                        )
                        masked_eval = masked_artifacts.test_frame.drop(columns=[payload.target_column], errors="ignore")
                        masked_eval[payload.target_column] = masked_artifacts.y_true
                        if sensitive_column not in masked_eval.columns:
                            masked_eval[sensitive_column] = frame.loc[masked_artifacts.y_true.index, sensitive_column]

                        masked_bias = compute_bias(
                            test_frame=masked_eval,
                            y_true=masked_artifacts.y_true,
                            y_pred=masked_artifacts.y_pred,
                            sensitive_column=sensitive_column,
                        )
                        masked_acc = float(masked_artifacts.metrics.get("accuracy", _compute_accuracy(masked_artifacts.y_true, masked_artifacts.y_pred)))
                        register_candidate("mask", ", ".join(masked_columns[:5]) if masked_columns else "auto_mask", masked_acc, masked_bias, {"masked_columns": masked_columns})
                except Exception as exc:  # noqa: BLE001
                    diagnostics.append(f"Mask mitigation skipped: {exc}")

                before_block = _build_fairness_block(baseline_bias)
                selected = None
                if candidates:
                    selected = min(candidates, key=lambda item: (item["gap"], item["acc_drop"], -item["accuracy"]))
                    after_block = _build_fairness_block(selected["bias"])
                    delta_fairness = float(after_block["fairness_index"] - before_block["fairness_index"])
                    delta_dpd = float(before_block["demographic_parity_difference"] - after_block["demographic_parity_difference"])
                    diagnostics.append(
                        f"Fairness optimization selected '{selected['strategy']}' with parameter '{selected['parameter']}' under accuracy-drop guardrail ({selected['acc_drop']:.3f} <= {max_accuracy_drop:.3f})."
                    )
                    if delta_fairness >= 0 and delta_dpd >= 0:
                        diagnostics.append("Fairness improvement accepted because disparity decreased while model performance stayed within limits.")
                    else:
                        diagnostics.append("No strategy improved both fairness and parity under the configured performance guardrail; baseline retained for deployment safety.")
                else:
                    after_block = before_block
                    delta_fairness = 0.0
                    delta_dpd = 0.0
                    diagnostics.append("No mitigation strategy met the accuracy-drop guardrail; fairness baseline retained.")

                fairness_summary = {
                    "sensitive_column": sensitive_column,
                    "before": before_block,
                    "after": after_block,
                    "delta_fairness_index": delta_fairness,
                    "delta_demographic_parity_difference": delta_dpd,
                    "method": selected["strategy"] if selected else "baseline",
                    "parameter": selected["parameter"] if selected else "none",
                    "accuracy_before": baseline_accuracy,
                    "accuracy_after": float(selected["accuracy"]) if selected else baseline_accuracy,
                    "accuracy_impact": float((selected["accuracy"] - baseline_accuracy)) if selected else 0.0,
                    "accuracy_guardrail": max_accuracy_drop,
                    "candidate_strategies": [
                        {
                            "strategy": c["strategy"],
                            "parameter": c["parameter"],
                            "gap": c["gap"],
                            "accuracy": c["accuracy"],
                            "acc_drop": c["acc_drop"],
                        }
                        for c in sorted(candidates, key=lambda item: (item["gap"], item["acc_drop"], -item["accuracy"]))
                    ],
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
    _persist_run_summary(run, email=payload.user_email)
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
def download_report_pdf(run_id: str, sensitive_column: str = "gender", sample_size: int = 40, user: AuthenticatedUser = Depends(_current_user)) -> Response:
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
async def upload_dataset(file: UploadFile = File(...), target_column: str | None = Form(default=None), user: AuthenticatedUser = Depends(_current_user)) -> UploadResponse:
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

    role_columns = [column for column in ("role_applied", "job_role", "position", "role") if column in frame.columns]
    role_suggestions = []
    if role_columns:
        role_column = role_columns[0]
        # Get up to 100 unique values, sorted, skipping nulls
        role_suggestions = sorted(frame[role_column].dropna().astype(str).unique().tolist())[:100]

    preview = frame.head(8).fillna("").to_dict(orient="records")
    return UploadResponse(
        dataset_id=dataset_id,
        filename=file.filename or "dataset.csv",
        rows=int(frame.shape[0]),
        columns=list(frame.columns),
        target_suggestions=suggestions,
        role_suggestions=role_suggestions,
        preview=preview,
    )


@app.post("/train", response_model=JobSubmissionResponse)
def train_model(payload: TrainRequest, user: AuthenticatedUser = Depends(_current_user)) -> JobSubmissionResponse:
    # Inject user email into payload for the background job
    payload.user_email = user.email
    if payload.async_job:
        job = jobs.submit("train", _train_job, payload)
        return JobSubmissionResponse(job_id=job.job_id, kind=job.kind, status=job.status, message="Training queued")

    response = _train_job(payload)
    return JobSubmissionResponse(job_id=f"train_{uuid4().hex[:10]}", kind="train", status="completed", message="Training completed", result=response)


@app.get("/bias", response_model=BiasResponse)
def bias_metrics(run_id: str, sensitive_column: str = "gender", user: AuthenticatedUser = Depends(_current_user)) -> BiasResponse:
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
    _persist_run_summary(run, email=user.email, bias_payload=payload.model_dump())
    return payload


@app.get("/explain", response_model=JobSubmissionResponse)
def explain_metrics(run_id: str, sample_size: int = 40, async_job: bool = False, user: AuthenticatedUser = Depends(_current_user)) -> JobSubmissionResponse:
    if async_job:
        job = jobs.submit("explain", _explain_job, run_id, sample_size)
        return JobSubmissionResponse(job_id=job.job_id, kind=job.kind, status=job.status, message="Explainability queued")

    response = _explain_job(run_id, sample_size)
    return JobSubmissionResponse(job_id=f"explain_{uuid4().hex[:10]}", kind="explain", status="completed", message="Explainability completed", result=response)


@app.get("/report", response_model=ReportResponse)
def report(run_id: str, sensitive_column: str = "gender", sample_size: int = 40, user: AuthenticatedUser = Depends(_current_user)) -> ReportResponse:
    if not ML_AVAILABLE:
        raise HTTPException(status_code=503, detail=ML_UNAVAILABLE_DETAIL)

    return _build_report(run_id, sensitive_column, sample_size)


@app.post("/assistant")
def assistant_chat(payload: dict[str, Any], user: AuthenticatedUser = Depends(_current_user)) -> dict[str, Any]:
    question = payload.get("question", "")
    context = payload.get("context", {})
    # Inject some system context if not provided
    if not context.get("user_name"):
        context["user_name"] = user.name
    return chat(question, context)


@app.get("/history")
def get_analysis_history(user: AuthenticatedUser = Depends(_current_user)) -> list[dict[str, Any]]:
    return persistence.get_history(email=user.email)


@app.get("/train/{run_id}", response_model=TrainResponse)
def train_metrics(run_id: str, user: AuthenticatedUser = Depends(_current_user)) -> TrainResponse:
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
