from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ════════════════════════════════════════════════════════════════════
# Upload
# ════════════════════════════════════════════════════════════════════

class UploadResponse(BaseModel):
    dataset_id: str
    filename: str
    rows: int
    columns: list[str]
    target_suggestions: list[str]
    preview: list[dict[str, Any]]
    schema: dict[str, str] = Field(default_factory=dict)
    null_counts: dict[str, int] = Field(default_factory=dict)


# ════════════════════════════════════════════════════════════════════
# Train
# ════════════════════════════════════════════════════════════════════

class TrainRequest(BaseModel):
    dataset_id: str
    target_column: str
    model_type: Literal[
        "logistic_regression", "random_forest",
        "gradient_boosting", "decision_tree",
    ] = "random_forest"
    test_size: float = Field(default=0.2, ge=0.1, le=0.4)
    random_state: int = 42


class TrainResponse(BaseModel):
    run_id: str
    dataset_id: str
    model_type: str
    target_column: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: dict[str, int]
    prediction_preview: list[dict[str, Any]]
    feature_count: int = 0
    train_rows: int = 0
    test_rows: int = 0


# ════════════════════════════════════════════════════════════════════
# Bias / Fairness
# ════════════════════════════════════════════════════════════════════

class BiasResponse(BaseModel):
    run_id: str
    sensitive_column: str
    demographic_parity_difference: float
    equal_opportunity_difference: float
    selection_rate_by_group: dict[str, float]
    true_positive_rate_by_group: dict[str, float]
    fairness_index: float
    verdict: Literal["PASS", "REVIEW", "FAIL"] = "REVIEW"
    verdict_detail: str = ""
    recommendations: list[str] = Field(default_factory=list)


# ════════════════════════════════════════════════════════════════════
# Explainability
# ════════════════════════════════════════════════════════════════════

class ExplainResponse(BaseModel):
    run_id: str
    sample_size: int
    top_global_features: list[dict[str, Any]]
    local_explanation: list[dict[str, Any]]


# ════════════════════════════════════════════════════════════════════
# Report
# ════════════════════════════════════════════════════════════════════

class ReportResponse(BaseModel):
    run_id: str
    train: TrainResponse
    bias: BiasResponse
    explain: ExplainResponse
    generated_at: str = ""
    executive_summary: str = ""


# ════════════════════════════════════════════════════════════════════
# What-If Simulation (aggregate / threshold level)
# ════════════════════════════════════════════════════════════════════

class WhatIfRequest(BaseModel):
    base_fairness_index: float = Field(ge=0.0, le=1.0)
    base_parity_gap: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(default=0.5, ge=0.3, le=0.8)
    reweight_strength: float = Field(default=0.0, ge=0.0, le=1.0)


class WhatIfResponse(BaseModel):
    threshold: float
    reweight_strength: float
    simulated_fairness_index: float
    simulated_parity_gap: float
    improvement: float
    verdict: str


class SimulateRequest(BaseModel):
    base_fairness_index: float
    base_parity_gap: float
    thresholds: list[float] = Field(default=[0.4, 0.45, 0.5, 0.55, 0.6])
    reweight_values: list[float] = Field(default=[0.0, 0.25, 0.5, 0.75, 1.0])


# ════════════════════════════════════════════════════════════════════
# NEW: De-biasing Engine
# ════════════════════════════════════════════════════════════════════

class DebiasRequest(BaseModel):
    dataset_id: str
    target_column: str | None = None
    auto_mask: bool = False
    columns_to_remove: list[str] = Field(default_factory=list)


class DebiasResponse(BaseModel):
    dataset_id: str
    masked_dataset_id: str | None = None
    total_columns: int
    safe_columns: list[str]
    flagged_count: int
    sensitive_columns: list[dict[str, Any]]
    proxy_columns: list[dict[str, Any]]
    correlated_columns: list[dict[str, Any]]
    masked_columns: list[str] = Field(default_factory=list)
    risk_summary: dict[str, int] = Field(default_factory=dict)


# ════════════════════════════════════════════════════════════════════
# NEW: Dual Evaluation
# ════════════════════════════════════════════════════════════════════

class DualEvalRequest(BaseModel):
    dataset_id: str
    target_column: str
    model_type: str = "random_forest"
    test_size: float = Field(default=0.2, ge=0.1, le=0.4)
    random_state: int = 42
    sensitive_columns_override: list[str] | None = None


class DualEvalResponse(BaseModel):
    run_id: str
    dataset_id: str
    model_type: str
    target_column: str
    masked_columns: list[str]
    # Model A
    model_a_accuracy: float
    model_a_f1: float
    model_a_features: list[str]
    # Model B
    model_b_accuracy: float
    model_b_f1: float
    model_b_features: list[str]
    # Delta
    accuracy_delta: float
    f1_delta: float
    ranking_divergence: float
    bias_influenced_fraction: float
    # Verdict
    verdict: str
    verdict_detail: str
    # Candidate-level
    per_candidate_comparison: list[dict[str, Any]]


# ════════════════════════════════════════════════════════════════════
# NEW: Candidate Scoring
# ════════════════════════════════════════════════════════════════════

class ScoringRequest(BaseModel):
    run_id: str
    threshold_recommend: float = Field(default=0.65, ge=0.0, le=1.0)
    threshold_borderline: float = Field(default=0.45, ge=0.0, le=1.0)
    fairness_penalty_columns: list[str] = Field(default_factory=list)
    penalty_weight: float = Field(default=0.10, ge=0.0, le=1.0)


class ScoringResponse(BaseModel):
    run_id: str
    total_candidates: int
    recommended: int
    borderline: int
    not_recommended: int
    candidate_scores: list[dict[str, Any]]
    ranking: list[dict[str, Any]]
    fairness_adjusted_ranking: list[dict[str, Any]]


# ════════════════════════════════════════════════════════════════════
# NEW: Candidate-level What-If
# ════════════════════════════════════════════════════════════════════

class CandidateWhatIfRequest(BaseModel):
    run_id: str
    candidate_index: str
    feature_overrides: dict[str, Any]


class CandidateWhatIfResponse(BaseModel):
    run_id: str
    candidate_index: str
    changed_features: list[dict[str, Any]]
    original_score: float
    modified_score: float
    original_decision: str
    modified_decision: str
    score_delta: float
    decision_changed: bool
    sensitivity: str
    interpretation: str


# ════════════════════════════════════════════════════════════════════
# NEW: Ethical Decision Validator
# ════════════════════════════════════════════════════════════════════

class ValidationRequest(BaseModel):
    run_id: str
    sensitive_column: str | None = None


class ValidationResponse(BaseModel):
    run_id: str
    total_decisions: int
    fair_count: int
    needs_review_count: int
    biased_count: int
    bias_rate: float
    validated_decisions: list[dict[str, Any]]
    group_disparities: list[dict[str, Any]]
    statistically_significant_patterns: list[dict[str, Any]]
    overall_assessment: str


# ════════════════════════════════════════════════════════════════════
# NEW: Conversational Assistant
# ════════════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    question: str
    run_id: str | None = None          # if provided, inject audit context automatically


class ChatResponse(BaseModel):
    question: str
    intent: str
    answer: str
    suggested_followups: list[str]
    data_used: list[str]


# ════════════════════════════════════════════════════════════════════
# NEW: Bias Stress Test
# ════════════════════════════════════════════════════════════════════

class StressTestRequest(BaseModel):
    dataset_id: str
    target_column: str
    sensitive_column: str
    model_type: str = "random_forest"
    strategies: list[str] | None = None


class StressTestResultItem(BaseModel):
    strategy: str
    sensitive_column: str
    target_group: str
    description: str
    baseline_fairness_index: float
    baseline_dpd: float
    baseline_verdict: str
    biased_fairness_index: float
    biased_dpd: float
    biased_verdict: str
    bias_detected: bool
    detection_confidence: float
    delta_fairness_index: float
    delta_dpd: float
    detection_summary: str
    injected_params: dict[str, Any] = Field(default_factory=dict)


class StressTestResponse(BaseModel):
    dataset_id: str
    target_column: str
    sensitive_column: str
    total_strategies: int
    detected_count: int
    missed_count: int
    detection_rate: float
    results: list[StressTestResultItem]
