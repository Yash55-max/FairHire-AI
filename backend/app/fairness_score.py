"""
Fairness Score Generator
========================
Computes a single 0–100 fairness score from multiple sub-dimensions:
  - Selection Parity    (40 pts)  — demographic parity and group SR gaps
  - Feature Balance     (30 pts)  — no single feature dominates decisions
  - Bias Detection      (20 pts)  — clean fairness index from bias module
  - Model Calibration   (10 pts)  — accuracy gap across groups (if computable)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class FairnessScoreResult:
    overall_score: float              # 0–100
    grade: str                        # A+, A, B, C, D, F
    label: str                        # Excellent / Good / Moderate / Poor / Critical
    breakdown: list[dict[str, Any]]   # per-dimension scores
    strengths: list[str]
    weaknesses: list[str]
    recommendations: list[str]
    raw_inputs: dict[str, Any]


# ── Grade thresholds ──────────────────────────────────────────────────────────
def _grade(score: float) -> tuple[str, str]:
    if score >= 90: return "A+", "Excellent"
    if score >= 80: return "A",  "Good"
    if score >= 70: return "B",  "Moderate"
    if score >= 55: return "C",  "Fair"
    if score >= 40: return "D",  "Poor"
    return "F", "Critical"


# ── Component calculators ─────────────────────────────────────────────────────

def _selection_parity_score(
    demographic_parity_difference: float,
    equal_opportunity_difference: float,
    selection_rate_by_group: dict[str, float],
) -> tuple[float, str]:
    """40 points max. Penalises gaps in selection rates."""
    max_gap = 0.0
    rates = list(selection_rate_by_group.values())
    if len(rates) >= 2:
        max_gap = max(rates) - min(rates)

    dpd_penalty  = min(40.0, demographic_parity_difference * 220)
    eod_penalty  = min(10.0, equal_opportunity_difference * 60)
    gap_penalty  = min(10.0, max_gap * 70)
    raw          = max(0.0, 40.0 - dpd_penalty - eod_penalty * 0.3 - gap_penalty * 0.3)
    note = (
        f"DPD={demographic_parity_difference:.3f}, EOD={equal_opportunity_difference:.3f}, "
        f"SR gap={max_gap:.3f}"
    )
    return round(raw, 2), note


def _feature_balance_score(
    top_global_features: list[dict[str, Any]],
) -> tuple[float, str]:
    """30 points max. Penalises concentration of influence in few features."""
    if not top_global_features:
        return 22.0, "No feature data available."

    vals = [abs(f.get("mean_abs_shap", f.get("importance", 0))) for f in top_global_features[:10]]
    if not vals or sum(vals) == 0:
        return 22.0, "Feature importance values are zero."

    total = sum(vals)
    top1_share = vals[0] / total if total > 0 else 0
    top3_share = sum(vals[:3]) / total if total > 0 else 0

    concentration_penalty = min(30.0, top1_share * 45 + top3_share * 15)
    raw = max(0.0, 30.0 - concentration_penalty)
    note = f"Top feature share={top1_share:.1%}, top-3 share={top3_share:.1%}"
    return round(raw, 2), note


def _bias_detection_score(
    fairness_index: float,
    verdict: str,
) -> tuple[float, str]:
    """20 points max. Based directly on the computed fairness index."""
    raw = min(20.0, fairness_index * 20)
    if verdict == "FAIL":
        raw = max(0.0, raw - 8.0)
    elif verdict == "REVIEW":
        raw = max(0.0, raw - 3.0)
    note = f"Fairness index={fairness_index:.3f}, verdict={verdict}"
    return round(raw, 2), note


def _model_calibration_score(
    accuracy: float,
    true_positive_rate_by_group: dict[str, float],
) -> tuple[float, str]:
    """10 points max. Penalises large TPR gaps across groups."""
    tprs = list(true_positive_rate_by_group.values())
    tpr_gap = (max(tprs) - min(tprs)) if len(tprs) >= 2 else 0.0
    raw = max(0.0, 10.0 - tpr_gap * 60)
    raw = min(raw, accuracy * 11)       # cap by overall accuracy
    note = f"Overall accuracy={accuracy:.3f}, TPR gap={tpr_gap:.3f}"
    return round(raw, 2), note


# ── Main function ─────────────────────────────────────────────────────────────

def compute_fairness_score(
    demographic_parity_difference: float,
    equal_opportunity_difference: float,
    selection_rate_by_group: dict[str, float],
    true_positive_rate_by_group: dict[str, float],
    fairness_index: float,
    verdict: str,
    accuracy: float,
    top_global_features: list[dict[str, Any]] | None = None,
) -> FairnessScoreResult:
    """
    Compute the composite Fairness Score (0–100).

    Parameters mirror the output of bias + explain + train endpoints.
    """
    sp_score, sp_note = _selection_parity_score(
        demographic_parity_difference, equal_opportunity_difference, selection_rate_by_group,
    )
    fb_score, fb_note = _feature_balance_score(top_global_features or [])
    bd_score, bd_note = _bias_detection_score(fairness_index, verdict)
    mc_score, mc_note = _model_calibration_score(accuracy, true_positive_rate_by_group)

    overall = round(sp_score + fb_score + bd_score + mc_score, 1)
    grade, label = _grade(overall)

    breakdown = [
        {"dimension": "Selection Parity",   "max": 40, "score": sp_score, "weight": "40%", "note": sp_note},
        {"dimension": "Feature Balance",     "max": 30, "score": fb_score, "weight": "30%", "note": fb_note},
        {"dimension": "Bias Detection",      "max": 20, "score": bd_score, "weight": "20%", "note": bd_note},
        {"dimension": "Model Calibration",   "max": 10, "score": mc_score, "weight": "10%", "note": mc_note},
    ]

    strengths, weaknesses, recs = [], [], []

    if sp_score >= 30: strengths.append("Strong selection parity across demographic groups.")
    else:              weaknesses.append(f"Selection parity needs improvement (score {sp_score}/40).")

    if fb_score >= 22: strengths.append("Feature influence is well-distributed across predictors.")
    else:              weaknesses.append(f"Feature concentration detected (score {fb_score}/30). One feature dominates.")

    if bd_score >= 15: strengths.append("Fairness index is healthy — low bias detected.")
    else:              weaknesses.append(f"Fairness index is low (score {bd_score}/20). Significant bias present.")

    if mc_score >= 7:  strengths.append("Model predictions are consistent across demographic groups.")
    else:              weaknesses.append(f"TPR gap between groups is large (score {mc_score}/10).")

    # Recommendations
    if sp_score < 30:
        recs.append("Apply reweighting or threshold calibration to reduce demographic parity gap.")
    if fb_score < 20:
        recs.append("Remove or reduce weight of dominant features — check for proxy variables.")
    if bd_score < 14:
        recs.append("Run the Dual Evaluation system to isolate bias sources.")
    if mc_score < 6:
        recs.append("Calibrate model separately per group using stratified thresholds.")

    if not recs:
        recs.append("Maintain current fairness posture. Re-evaluate after every model retrain.")

    return FairnessScoreResult(
        overall_score=overall,
        grade=grade,
        label=label,
        breakdown=breakdown,
        strengths=strengths,
        weaknesses=weaknesses,
        recommendations=recs,
        raw_inputs={
            "demographic_parity_difference": demographic_parity_difference,
            "equal_opportunity_difference": equal_opportunity_difference,
            "fairness_index": fairness_index,
            "verdict": verdict,
            "accuracy": accuracy,
        },
    )
