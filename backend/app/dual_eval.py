"""
Dual Evaluation System
======================
Trains Model A (full data) and Model B (bias-masked data) on the same
dataset and compares their prediction behaviour to surface bias influence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from .debiasing import DebiasEngine
from .ml_pipeline import SUPPORTED_MODELS, build_model, train_pipeline


@dataclass
class DualEvalResult:
    run_id: str
    dataset_id: str
    model_type: str
    target_column: str
    # Model A – full features
    model_a_accuracy: float
    model_a_f1: float
    model_a_features: list[str]
    # Model B – bias-masked
    model_b_accuracy: float
    model_b_f1: float
    model_b_features: list[str]
    masked_columns: list[str]
    # Comparison
    accuracy_delta: float          # model_a - model_b (positive = A was inflated by bias)
    f1_delta: float
    ranking_divergence: float      # mean |rank_A - rank_B| normalised 0-1
    bias_influenced_fraction: float  # fraction of candidates ranked differently
    per_candidate_comparison: list[dict[str, Any]]
    verdict: str                   # "BIAS DETECTED" | "MINIMAL BIAS" | "NO BIAS"
    verdict_detail: str


def run_dual_evaluation(
    frame: pd.DataFrame,
    target_column: str,
    model_type: str = "random_forest",
    test_size: float = 0.2,
    random_state: int = 42,
    sensitive_columns_override: list[str] | None = None,
    dataset_id: str = "",
    run_id: str = "",
) -> DualEvalResult:
    """Train Model A + B, compare predictions, and surface bias influence."""

    if model_type not in SUPPORTED_MODELS:
        model_type = "random_forest"

    # ── De-bias Engine to determine what to mask ──────────────────────────
    engine = DebiasEngine(frame, target_column)
    if sensitive_columns_override is not None:
        cols_to_mask = [c for c in sensitive_columns_override if c in frame.columns and c != target_column]
        masked_frame = engine.mask_columns(cols_to_mask)
    else:
        masked_frame, cols_to_mask = engine.auto_mask()

    # ── Train Model A (full data) ─────────────────────────────────────────
    arts_a = train_pipeline(
        frame=frame,
        target_column=target_column,
        model_type=model_type,
        test_size=test_size,
        random_state=random_state,
    )

    # ── Train Model B (masked data) ───────────────────────────────────────
    # Ensure target is still present
    if target_column not in masked_frame.columns:
        masked_frame[target_column] = frame[target_column]

    arts_b = train_pipeline(
        frame=masked_frame,
        target_column=target_column,
        model_type=model_type,
        test_size=test_size,
        random_state=random_state,
    )

    # ── Align on shared test indices ──────────────────────────────────────
    shared_idx = arts_a.test_frame.index.intersection(arts_b.test_frame.index)
    if len(shared_idx) == 0:
        shared_idx = arts_a.test_frame.index

    y_a = arts_a.y_pred.reindex(shared_idx).fillna(0)
    y_b = arts_b.y_pred.reindex(shared_idx).fillna(0)
    y_t = arts_a.y_true.reindex(shared_idx).fillna(0)

    # ── Per-candidate comparison ──────────────────────────────────────────
    per_candidate: list[dict[str, Any]] = []
    for idx in shared_idx[:50]:          # cap at 50 for response size
        pa = int(y_a.loc[idx])
        pb = int(y_b.loc[idx])
        per_candidate.append({
            "index": int(idx),
            "prediction_model_a": pa,
            "prediction_model_b": pb,
            "agreement": pa == pb,
            "bias_signal": pa != pb,
        })

    # ── Ranking divergence ────────────────────────────────────────────────
    agreed = sum(1 for c in per_candidate if c["agreement"])
    diverged = len(per_candidate) - agreed
    bias_frac = diverged / max(len(per_candidate), 1)

    # Rank by prediction probability if available, else use prediction value
    try:
        prob_a = arts_a.model.predict_proba(arts_a.test_frame.drop(columns=[target_column], errors="ignore"))[:, -1]
        prob_b = arts_b.model.predict_proba(arts_b.test_frame.drop(columns=[target_column], errors="ignore"))[:, -1]
        n = min(len(prob_a), len(prob_b))
        rank_a = pd.Series(prob_a[:n]).rank(ascending=False).values
        rank_b = pd.Series(prob_b[:n]).rank(ascending=False).values
        rank_div = float(np.mean(np.abs(rank_a - rank_b)) / max(n, 1))
    except Exception:  # noqa: BLE001
        rank_div = float(bias_frac)

    acc_a = float(accuracy_score(y_t, y_a))
    acc_b = float(accuracy_score(y_t, y_b))
    avg = "binary" if y_t.nunique() == 2 else "weighted"
    f1_a = float(f1_score(y_t, y_a, average=avg, zero_division=0))
    f1_b = float(f1_score(y_t, y_b, average=avg, zero_division=0))
    acc_delta = acc_a - acc_b
    f1_delta  = f1_a - f1_b

    # ── Verdict ───────────────────────────────────────────────────────────
    if bias_frac >= 0.20 or abs(acc_delta) >= 0.08:
        verdict = "BIAS DETECTED"
        verdict_detail = (
            f"{bias_frac:.0%} of candidates received different decisions when sensitive "
            f"attributes were removed. Accuracy delta = {acc_delta:+.3f}. "
            "The removed features are influencing hiring outcomes."
        )
    elif bias_frac >= 0.08 or abs(acc_delta) >= 0.03:
        verdict = "MINIMAL BIAS"
        verdict_detail = (
            f"{bias_frac:.0%} of candidates differ between models. "
            "Some influence from potentially biased features detected. Monitor closely."
        )
    else:
        verdict = "NO BIAS"
        verdict_detail = (
            f"Only {bias_frac:.0%} of candidates differ between models. "
            "Sensitive attributes appear to have minimal impact on decisions."
        )

    feat_a = [c for c in arts_a.test_frame.columns if c != target_column]
    feat_b = [c for c in arts_b.test_frame.columns if c != target_column]

    return DualEvalResult(
        run_id=run_id,
        dataset_id=dataset_id,
        model_type=model_type,
        target_column=target_column,
        model_a_accuracy=acc_a,
        model_a_f1=f1_a,
        model_a_features=feat_a,
        model_b_accuracy=acc_b,
        model_b_f1=f1_b,
        model_b_features=feat_b,
        masked_columns=cols_to_mask,
        accuracy_delta=acc_delta,
        f1_delta=f1_delta,
        ranking_divergence=rank_div,
        bias_influenced_fraction=bias_frac,
        per_candidate_comparison=per_candidate,
        verdict=verdict,
        verdict_detail=verdict_detail,
    )
