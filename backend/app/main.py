from __future__ import annotations

import json
import os
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .assistant import chat as assistant_chat
from .bias_stress_test import run_stress_test
from .candidate_scorer import candidate_whatif, score_candidates
from .debiasing import DebiasEngine
from .dual_eval import run_dual_evaluation
from .ethical_validator import validate_decisions
from .ml_pipeline import compute_bias, compute_explainability, suggest_target_columns, train_pipeline
from .schemas import (
    BiasResponse,
    CandidateWhatIfRequest,
    CandidateWhatIfResponse,
    ChatRequest,
    ChatResponse,
    DebiasRequest,
    DebiasResponse,
    DualEvalRequest,
    DualEvalResponse,
    ExplainResponse,
    ReportResponse,
    ScoringRequest,
    ScoringResponse,
    SimulateRequest,
    StressTestRequest,
    StressTestResponse,
    StressTestResultItem,
    TrainRequest,
    TrainResponse,
    UploadResponse,
    ValidationRequest,
    ValidationResponse,
    WhatIfRequest,
    WhatIfResponse,
)
from .store import InMemoryStore, TrainingRun

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "backend" / "data"
RUNS_JSON = DATA_DIR / "runs.json"

store = InMemoryStore()

# ── Application ────────────────────────────────────────────────────────────
app = FastAPI(
    title="FairHire AI",
    description=(
        "AI-powered Fairness Audit and Decision Accountability Platform. "
        "Dual-model evaluation, de-biasing engine, candidate scoring, ethical "
        "validation, what-if simulation, and a conversational fairness assistant."
    ),
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════

def _persist_run(run: TrainingRun, extra: dict | None = None) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    if RUNS_JSON.exists():
        try:
            with RUNS_JSON.open("r", encoding="utf-8") as f:
                records = json.load(f)
        except (json.JSONDecodeError, OSError):
            records = []

    idx = next((i for i, r in enumerate(records) if r.get("run_id") == run.run_id), None)
    record: dict[str, Any] = {
        "run_id": run.run_id,
        "dataset_id": run.dataset_id,
        "model_type": run.model_type,
        "target_column": run.target_column,
        "created_at": run.created_at.isoformat(),
        "metrics": run.metrics,
        **(extra or {}),
    }
    if idx is not None:
        records[idx] = {**records[idx], **record}
    else:
        records.append(record)

    with RUNS_JSON.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, default=str)


def _read_uploaded_file(upload: UploadFile, content: bytes) -> pd.DataFrame:
    filename = (upload.filename or "dataset.csv").lower()
    try:
        if filename.endswith(".csv"):
            return pd.read_csv(BytesIO(content))
        if filename.endswith(".json"):
            return pd.read_json(BytesIO(content))
        if filename.endswith((".xlsx", ".xls")):
            return pd.read_excel(BytesIO(content))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {exc}") from exc
    raise HTTPException(status_code=400, detail="Unsupported format. Use CSV, JSON, or XLSX.")


def _build_assistant_context(run_id: str | None) -> dict[str, Any]:
    """Pull relevant audit data from the store to enrich chatbot responses."""
    ctx: dict[str, Any] = {}
    if not run_id:
        return ctx
    try:
        run = store.get_run(run_id)
        ctx["run_id"] = run_id
        ctx["model_type"] = run.model_type
        ctx["target_column"] = run.target_column
        ctx.update(run.metrics)
        # Enrich with cached bias/scoring if present
        if hasattr(run, "bias_cache") and run.bias_cache:
            ctx.update(run.bias_cache)
        if hasattr(run, "scoring_cache") and run.scoring_cache:
            ctx.update(run.scoring_cache)
    except KeyError:
        pass
    return ctx


# ═══════════════════════════════════════════════════════════════════════════
# System
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["System"])
def health() -> dict[str, str]:
    return {"status": "ok", "version": "3.0.0", "timestamp": datetime.utcnow().isoformat()}


@app.get("/runs", tags=["System"])
def list_runs() -> list[dict]:
    if not RUNS_JSON.exists():
        return []
    try:
        with RUNS_JSON.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


# ═══════════════════════════════════════════════════════════════════════════
# Upload
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/upload", response_model=UploadResponse, tags=["Dataset"])
async def upload_dataset(
    file: UploadFile = File(...),
    target_column: str | None = Form(default=None),
) -> UploadResponse:
    """Upload a candidate dataset (CSV / JSON / XLSX)."""
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    frame = _read_uploaded_file(file, content)
    if frame.empty:
        raise HTTPException(status_code=400, detail="Parsed dataset is empty")

    dataset_id = f"ds_{uuid4().hex[:10]}"
    frame.columns = [str(c).strip() for c in frame.columns]
    store.put_dataset(dataset_id, frame)

    suggestions = suggest_target_columns(frame)
    if target_column and target_column in frame.columns and target_column not in suggestions:
        suggestions = [target_column] + suggestions

    preview = frame.head(8).fillna("").to_dict(orient="records")
    schema  = {col: str(frame[col].dtype) for col in frame.columns}

    return UploadResponse(
        dataset_id=dataset_id,
        filename=file.filename or "dataset.csv",
        rows=int(frame.shape[0]),
        columns=list(frame.columns),
        target_suggestions=suggestions,
        preview=preview,
        schema=schema,
        null_counts={col: int(frame[col].isna().sum()) for col in frame.columns},
    )


# ═══════════════════════════════════════════════════════════════════════════
# De-biasing Engine  ✨ NEW
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/debias", response_model=DebiasResponse, tags=["De-biasing"])
def debias_dataset(payload: DebiasRequest) -> DebiasResponse:
    """Audit a dataset for sensitive attributes, proxy variables, and high-correlation features."""
    try:
        frame = store.get_dataset(payload.dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    engine = DebiasEngine(frame, payload.target_column)
    audit  = engine.full_audit()

    masked_id: str | None = None
    masked_cols: list[str] = []

    if payload.auto_mask or payload.columns_to_remove:
        if payload.columns_to_remove:
            masked_frame = engine.mask_columns(payload.columns_to_remove)
            masked_cols  = payload.columns_to_remove
        else:
            masked_frame, masked_cols = engine.auto_mask()

        masked_id = f"ds_{uuid4().hex[:10]}_masked"
        store.put_dataset(masked_id, masked_frame)

    return DebiasResponse(
        dataset_id=payload.dataset_id,
        masked_dataset_id=masked_id,
        total_columns=audit["total_columns"],
        safe_columns=audit["safe_columns"],
        flagged_count=audit["flagged_count"],
        sensitive_columns=audit["sensitive_columns"],
        proxy_columns=audit["proxy_columns"],
        correlated_columns=audit["correlated_columns"],
        masked_columns=masked_cols,
        risk_summary=audit["risk_summary"],
    )


# ═══════════════════════════════════════════════════════════════════════════
# Dual Evaluation  ✨ NEW
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/dual-eval", response_model=DualEvalResponse, tags=["Dual Evaluation"])
def dual_eval(payload: DualEvalRequest) -> DualEvalResponse:
    """Train Model A (full data) vs Model B (bias-masked) and compare bias influence."""
    try:
        frame = store.get_dataset(payload.dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    run_id = f"dual_{uuid4().hex[:10]}"
    try:
        result = run_dual_evaluation(
            frame=frame,
            target_column=payload.target_column,
            model_type=payload.model_type,
            test_size=payload.test_size,
            random_state=payload.random_state,
            sensitive_columns_override=payload.sensitive_columns_override,
            dataset_id=payload.dataset_id,
            run_id=run_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return DualEvalResponse(
        run_id=result.run_id,
        dataset_id=result.dataset_id,
        model_type=result.model_type,
        target_column=result.target_column,
        masked_columns=result.masked_columns,
        model_a_accuracy=result.model_a_accuracy,
        model_a_f1=result.model_a_f1,
        model_a_features=result.model_a_features,
        model_b_accuracy=result.model_b_accuracy,
        model_b_f1=result.model_b_f1,
        model_b_features=result.model_b_features,
        accuracy_delta=result.accuracy_delta,
        f1_delta=result.f1_delta,
        ranking_divergence=result.ranking_divergence,
        bias_influenced_fraction=result.bias_influenced_fraction,
        verdict=result.verdict,
        verdict_detail=result.verdict_detail,
        per_candidate_comparison=result.per_candidate_comparison,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Model Training (existing, cleaned up)
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/train", response_model=TrainResponse, tags=["Model"])
def train_model(payload: TrainRequest) -> TrainResponse:
    """Train a classifier on an uploaded dataset."""
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
        feature_count=len(artifacts.test_frame.columns) - 1,
        train_rows=int(frame.shape[0] - len(artifacts.test_frame)),
        test_rows=int(len(artifacts.test_frame)),
    )
    _persist_run(run)
    return response


@app.get("/train/{run_id}", response_model=TrainResponse, tags=["Model"])
def get_train_metrics(run_id: str) -> TrainResponse:
    try:
        run = store.get_run(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    preview_df = run.test_frame.copy()
    preview_df["prediction"] = run.y_pred
    preview = preview_df.head(10).fillna("").to_dict(orient="records")
    origin = store.datasets.get(run.dataset_id)
    train_rows = int(origin.shape[0] - len(run.test_frame)) if origin is not None else 0

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
        feature_count=len(run.test_frame.columns) - 1,
        train_rows=train_rows,
        test_rows=int(len(run.test_frame)),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Candidate Scoring  ✨ NEW
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/candidates/score", response_model=ScoringResponse, tags=["Candidate Scoring"])
def score_candidates_endpoint(payload: ScoringRequest) -> ScoringResponse:
    """Score all test-set candidates with feature contribution breakdowns."""
    try:
        run = store.get_run(payload.run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        result = score_candidates(
            model=run.model,
            test_frame=run.test_frame,
            target_column=run.target_column,
            run_id=payload.run_id,
            threshold_recommend=payload.threshold_recommend,
            threshold_borderline=payload.threshold_borderline,
            fairness_penalty_columns=payload.fairness_penalty_columns or [],
            penalty_weight=payload.penalty_weight,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Cache scoring result on the run object for the assistant
    run.tags.append("scored")

    return ScoringResponse(
        run_id=result.run_id,
        total_candidates=result.total_candidates,
        recommended=result.recommended,
        borderline=result.borderline,
        not_recommended=result.not_recommended,
        candidate_scores=result.candidate_scores,
        ranking=result.ranking,
        fairness_adjusted_ranking=result.fairness_adjusted_ranking,
    )


# ── Candidate-level What-If  ✨ NEW ─────────────────────────────────────────

@app.post("/candidates/whatif", response_model=CandidateWhatIfResponse, tags=["Candidate Scoring"])
def candidate_whatif_endpoint(payload: CandidateWhatIfRequest) -> CandidateWhatIfResponse:
    """Re-score a single candidate after overriding feature values."""
    try:
        run = store.get_run(payload.run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        result = candidate_whatif(
            model=run.model,
            test_frame=run.test_frame,
            target_column=run.target_column,
            candidate_index=payload.candidate_index,
            feature_overrides=payload.feature_overrides,
            run_id=payload.run_id,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return CandidateWhatIfResponse(**result)


# ═══════════════════════════════════════════════════════════════════════════
# Bias / Fairness (existing, kept intact)
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/bias", response_model=BiasResponse, tags=["Fairness"])
def bias_metrics(run_id: str, sensitive_column: str = "gender") -> BiasResponse:
    """Compute fairness / bias metrics for a training run."""
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

    dpd = bias["demographic_parity_difference"]
    fi  = bias["fairness_index"]
    eo  = bias["equal_opportunity_difference"]

    if fi >= 0.85 and dpd <= 0.08:
        verdict, verdict_detail = "PASS", "Model meets EEOC-aligned fairness thresholds."
    elif fi >= 0.70:
        verdict, verdict_detail = "REVIEW", "Moderate disparity detected. Manual review recommended."
    else:
        verdict, verdict_detail = "FAIL", "Significant bias detected. Remediation required before deployment."

    recommendations: list[str] = []
    if dpd > 0.1:
        recommendations.append("Reweight training data to balance group representation.")
    if eo > 0.1:
        recommendations.append("Apply equalized odds post-processing to close the TPR gap.")
    sr_vals = list(bias.get("selection_rate_by_group", {}).values())
    if sr_vals and max(sr_vals) - min(sr_vals) > 0.12:
        recommendations.append("Review referral sources — possible proxy variable for protected class.")
    if not recommendations:
        recommendations.append("Continue monitoring fairness metrics after each retraining cycle.")

    payload = BiasResponse(
        run_id=run_id,
        sensitive_column=bias["sensitive_column"],
        demographic_parity_difference=bias["demographic_parity_difference"],
        equal_opportunity_difference=bias["equal_opportunity_difference"],
        selection_rate_by_group=bias["selection_rate_by_group"],
        true_positive_rate_by_group=bias["true_positive_rate_by_group"],
        fairness_index=bias["fairness_index"],
        verdict=verdict,
        verdict_detail=verdict_detail,
        recommendations=recommendations,
    )
    _persist_run(run, {"bias": payload.model_dump()})
    return payload


# ═══════════════════════════════════════════════════════════════════════════
# Explainability (existing)
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/explain", response_model=ExplainResponse, tags=["Explainability"])
def explain_metrics(run_id: str, sample_size: int = 40) -> ExplainResponse:
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
    _persist_run(run, {"explain": payload.model_dump()})
    return payload


# ═══════════════════════════════════════════════════════════════════════════
# Ethical Decision Validator  ✨ NEW
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/validate", response_model=ValidationResponse, tags=["Ethical Validator"])
def validate_decisions_endpoint(payload: ValidationRequest) -> ValidationResponse:
    """Classify each hiring decision as Fair, Needs Review, or Biased."""
    try:
        run = store.get_run(payload.run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    # First compute scores
    try:
        scoring = score_candidates(
            model=run.model,
            test_frame=run.test_frame,
            target_column=run.target_column,
            run_id=payload.run_id,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Scoring failed: {exc}") from exc

    # Pull bias data if available
    sr_by_group: dict[str, float] | None = None
    if payload.sensitive_column:
        try:
            bias_data = compute_bias(
                test_frame=run.test_frame,
                y_true=run.y_true,
                y_pred=run.y_pred,
                sensitive_column=payload.sensitive_column,
            )
            sr_by_group = bias_data.get("selection_rate_by_group")
        except Exception:  # noqa: BLE001
            pass

    report = validate_decisions(
        candidate_scores=scoring.candidate_scores,
        test_frame=run.test_frame,
        sensitive_column=payload.sensitive_column,
        selection_rate_by_group=sr_by_group,
        run_id=payload.run_id,
    )

    return ValidationResponse(
        run_id=report.run_id,
        total_decisions=report.total_decisions,
        fair_count=report.fair_count,
        needs_review_count=report.needs_review_count,
        biased_count=report.biased_count,
        bias_rate=report.bias_rate,
        validated_decisions=report.validated_decisions,
        group_disparities=report.group_disparities,
        statistically_significant_patterns=report.statistically_significant_patterns,
        overall_assessment=report.overall_assessment,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Full Report (existing, enhanced)
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/report", response_model=ReportResponse, tags=["Reports"])
def full_report(run_id: str, sensitive_column: str = "gender", sample_size: int = 40) -> ReportResponse:
    """Consolidated audit report combining training, bias, and explainability."""
    train   = get_train_metrics(run_id)
    bias    = bias_metrics(run_id=run_id, sensitive_column=sensitive_column)
    explain = explain_metrics(run_id=run_id, sample_size=sample_size)

    generated_at = datetime.utcnow().isoformat()
    top_feat = explain.top_global_features[0]["feature"] if explain.top_global_features else "N/A"
    executive_summary = (
        f"Model achieved {train.accuracy:.1%} accuracy. "
        f"Fairness verdict: {bias.verdict}. "
        f"Top influential feature: {top_feat}. "
        f"Report generated: {generated_at}."
    )
    return ReportResponse(
        run_id=run_id, train=train, bias=bias, explain=explain,
        generated_at=generated_at, executive_summary=executive_summary,
    )


# ═══════════════════════════════════════════════════════════════════════════
# What-If Simulation (aggregate level — existing)
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/simulate/whatif", response_model=WhatIfResponse, tags=["Simulation"])
def whatif_simulate(payload: WhatIfRequest) -> WhatIfResponse:
    fi  = payload.base_fairness_index
    pg  = payload.base_parity_gap
    thr = payload.threshold
    rw  = payload.reweight_strength

    thr_delta = (thr - 0.5) * 0.14
    rw_delta  = rw * 0.18 * (1 - fi)
    sim_fi    = min(0.97, max(0.40, fi + rw_delta - abs(thr_delta) * 0.3))
    sim_pg    = max(0.01, pg - thr_delta * 0.09 - rw * 0.12)
    improvement = sim_fi - fi
    verdict = "Improved" if improvement > 0.02 else "Marginal change" if improvement > -0.01 else "Degraded"

    return WhatIfResponse(
        threshold=thr, reweight_strength=rw,
        simulated_fairness_index=round(sim_fi, 4),
        simulated_parity_gap=round(sim_pg, 4),
        improvement=round(improvement, 4),
        verdict=verdict,
    )


@app.post("/simulate/batch", response_model=list[WhatIfResponse], tags=["Simulation"])
def batch_simulate(payload: SimulateRequest) -> list[WhatIfResponse]:
    results = []
    for thr in payload.thresholds:
        for rw in payload.reweight_values:
            results.append(whatif_simulate(WhatIfRequest(
                base_fairness_index=payload.base_fairness_index,
                base_parity_gap=payload.base_parity_gap,
                threshold=thr, reweight_strength=rw,
            )))
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Sample Dataset Download  ✨ NEW
# ═══════════════════════════════════════════════════════════════════════════

from fastapi.responses import FileResponse

@app.get("/sample-dataset", tags=["Dataset"])
def download_sample_dataset() -> FileResponse:
    """Download the built-in sample hiring dataset (200 rows)."""
    sample_path = DATA_DIR / "sample_dataset.csv"
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail="Sample dataset not found.")
    return FileResponse(
        path=str(sample_path),
        media_type="text/csv",
        filename="fairhire_sample_dataset.csv",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Bias Stress Test  ✨ NEW
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/stress-test", response_model=StressTestResponse, tags=["Bias Stress Test"])
def stress_test_endpoint(payload: StressTestRequest) -> StressTestResponse:
    """Inject controlled bias into the dataset and validate whether the detection pipeline catches it."""
    try:
        frame = store.get_dataset(payload.dataset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        results = run_stress_test(
            frame=frame,
            target_column=payload.target_column,
            sensitive_column=payload.sensitive_column,
            model_type=payload.model_type,
            strategies=payload.strategies,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Stress test failed: {exc}") from exc

    detected = sum(1 for r in results if r.bias_detected)
    missed   = len(results) - detected

    return StressTestResponse(
        dataset_id=payload.dataset_id,
        target_column=payload.target_column,
        sensitive_column=payload.sensitive_column,
        total_strategies=len(results),
        detected_count=detected,
        missed_count=missed,
        detection_rate=round(detected / len(results), 4) if results else 0.0,
        results=[
            StressTestResultItem(
                strategy=r.strategy,
                sensitive_column=r.sensitive_column,
                target_group=r.target_group,
                description=r.description,
                baseline_fairness_index=r.baseline_fairness_index,
                baseline_dpd=r.baseline_dpd,
                baseline_verdict=r.baseline_verdict,
                biased_fairness_index=r.biased_fairness_index,
                biased_dpd=r.biased_dpd,
                biased_verdict=r.biased_verdict,
                bias_detected=r.bias_detected,
                detection_confidence=r.detection_confidence,
                delta_fairness_index=r.delta_fairness_index,
                delta_dpd=r.delta_dpd,
                detection_summary=r.detection_summary,
                injected_params=r.injected_params,
            )
            for r in results
        ],
    )


# ═══════════════════════════════════════════════════════════════════════════
# Conversational Assistant  ✨ NEW
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/assistant/chat", response_model=ChatResponse, tags=["Assistant"])
def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    """Ask natural-language questions about bias, decisions, and fairness."""
    context = _build_assistant_context(payload.run_id)
    result  = assistant_chat(payload.question, context)
    return ChatResponse(**result)
