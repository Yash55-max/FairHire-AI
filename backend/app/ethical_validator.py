"""
Ethical Decision Validator
==========================
Classifies each hiring decision as Fair, Needs Review, or Biased and
provides human-readable justification for each classification.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Classification labels
# ---------------------------------------------------------------------------
FAIR         = "Fair"
NEEDS_REVIEW = "Needs Review"
BIASED       = "Biased"


@dataclass
class DecisionValidation:
    candidate_index: str
    decision: str           # Recommended | Borderline | Not Recommended
    classification: str     # Fair | Needs Review | Biased
    confidence: float
    justification: str
    bias_signals: list[str]
    recommended_action: str


@dataclass
class ValidationReport:
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_bias_signals(
    row_contributions: list[dict[str, Any]],
    sensitive_keywords: list[str],
) -> list[str]:
    signals: list[str] = []
    for contrib in row_contributions:
        feat = contrib.get("feature", "").lower()
        val  = contrib.get("shap_value", 0)
        if abs(val) >= 0.05 and any(kw in feat for kw in sensitive_keywords):
            direction = "positively" if val > 0 else "negatively"
            signals.append(f"'{contrib['feature']}' influenced decision {direction} (SHAP={val:.3f})")
    return signals


def _classify_decision(
    confidence: float,
    bias_signals: list[str],
    group_disparity: float,
) -> tuple[str, str, str]:
    """Return (classification, justification, recommended_action)."""

    if len(bias_signals) >= 2 or group_disparity >= 0.15:
        cls = BIASED
        justification = (
            f"Decision shows {len(bias_signals)} bias signal(s): "
            + "; ".join(bias_signals[:2])
            + (f". Group disparity detected: {group_disparity:.0%}." if group_disparity >= 0.15 else "")
        )
        action = "Escalate to HR review. Remove flagged attributes and reassess candidate."
    elif len(bias_signals) == 1 or group_disparity >= 0.08:
        cls = NEEDS_REVIEW
        justification = (
            f"One bias signal detected: {bias_signals[0] if bias_signals else ''}. "
            f"Group disparity: {group_disparity:.0%}."
        )
        action = "Manual review recommended before finalising decision."
    else:
        cls = FAIR
        justification = (
            f"No significant bias signals detected. Decision confidence: {confidence:.0%}. "
            "Decision appears to be based on merit-relevant features."
        )
        action = "No action required."

    return cls, justification, action


# ---------------------------------------------------------------------------
# Main validator
# ---------------------------------------------------------------------------

def validate_decisions(
    candidate_scores: list[dict[str, Any]],
    test_frame: pd.DataFrame,
    sensitive_column: str | None = None,
    selection_rate_by_group: dict[str, float] | None = None,
    run_id: str = "",
) -> ValidationReport:
    """
    Validate every hiring decision for fairness and bias.

    Parameters
    ----------
    candidate_scores : output from candidate_scorer.score_candidates()
    test_frame       : original test dataframe (to reference group membership)
    sensitive_column : name of the protected attribute column
    selection_rate_by_group : from bias analysis, if already computed
    run_id           : parent run identifier
    """

    sensitive_kws = [
        "gender", "sex", "race", "ethnicity", "age", "name", "zip",
        "location", "college", "school", "referral", "gap",
    ]

    # ── Compute group disparity per candidate ─────────────────────────────
    max_disparity = 0.0
    if selection_rate_by_group and len(selection_rate_by_group) >= 2:
        rates = list(selection_rate_by_group.values())
        max_disparity = max(rates) - min(rates)

    validated: list[DecisionValidation] = []
    for cs in candidate_scores:
        contributions = cs.get("feature_contributions", [])
        bias_signals  = _detect_bias_signals(contributions, sensitive_kws)
        prob          = cs.get("confidence", 0.5)

        cls, justification, action = _classify_decision(prob, bias_signals, max_disparity)

        validated.append(DecisionValidation(
            candidate_index=str(cs["candidate_index"]),
            decision=cs["decision"],
            classification=cls,
            confidence=prob,
            justification=justification,
            bias_signals=bias_signals,
            recommended_action=action,
        ))

    fair_count   = sum(1 for v in validated if v.classification == FAIR)
    review_count = sum(1 for v in validated if v.classification == NEEDS_REVIEW)
    biased_count = sum(1 for v in validated if v.classification == BIASED)
    bias_rate    = biased_count / max(len(validated), 1)

    # ── Group disparity table ─────────────────────────────────────────────
    disparities: list[dict[str, Any]] = []
    if selection_rate_by_group:
        rates  = list(selection_rate_by_group.values())
        groups = list(selection_rate_by_group.keys())
        for i, g in enumerate(groups):
            for j in range(i + 1, len(groups)):
                gap = abs(rates[i] - rates[j])
                disparities.append({
                    "group_a": g, "group_b": groups[j],
                    "selection_rate_a": round(rates[i], 4),
                    "selection_rate_b": round(rates[j], 4),
                    "gap": round(gap, 4),
                    "significant": gap >= 0.10,
                })

    # ── Statistical significance tests ───────────────────────────────────
    sig_patterns: list[dict[str, Any]] = []
    if sensitive_column and sensitive_column in test_frame.columns:
        groups_in_data = test_frame[sensitive_column].unique()
        if len(groups_in_data) >= 2:
            group_scores: dict[str, list[float]] = {}
            for cs in candidate_scores:
                try:
                    idx = cs["candidate_index"]
                    if idx in test_frame.index:
                        grp = str(test_frame.loc[idx, sensitive_column])
                    else:
                        continue
                    group_scores.setdefault(grp, []).append(cs["score"])
                except Exception:  # noqa: BLE001
                    pass

            group_list = list(group_scores.keys())
            for i in range(len(group_list)):
                for j in range(i + 1, len(group_list)):
                    g1, g2 = group_list[i], group_list[j]
                    s1, s2 = group_scores[g1], group_scores[g2]
                    if len(s1) >= 5 and len(s2) >= 5:
                        t_stat, p_val = stats.ttest_ind(s1, s2, equal_var=False)
                        sig_patterns.append({
                            "group_a": g1,
                            "group_b": g2,
                            "mean_score_a": round(float(np.mean(s1)), 2),
                            "mean_score_b": round(float(np.mean(s2)), 2),
                            "t_statistic": round(float(t_stat), 4),
                            "p_value": round(float(p_val), 4),
                            "statistically_significant": p_val < 0.05,
                            "interpretation": (
                                f"Significant score difference between {g1} and {g2} (p={p_val:.3f})"
                                if p_val < 0.05 else
                                f"No significant score difference between {g1} and {g2} (p={p_val:.3f})"
                            ),
                        })

    # ── Overall assessment ────────────────────────────────────────────────
    if bias_rate >= 0.25:
        overall = f"HIGH BIAS RISK — {bias_rate:.0%} of decisions flagged as Biased. Immediate remediation required."
    elif bias_rate >= 0.10 or review_count / max(len(validated), 1) >= 0.30:
        overall = f"MODERATE BIAS RISK — {review_count} decisions need review. Process audit recommended."
    else:
        overall = f"LOW BIAS RISK — {fair_count}/{len(validated)} decisions are Fair. Continue monitoring."

    return ValidationReport(
        run_id=run_id,
        total_decisions=len(validated),
        fair_count=fair_count,
        needs_review_count=review_count,
        biased_count=biased_count,
        bias_rate=round(bias_rate, 4),
        validated_decisions=[v.__dict__ for v in validated],
        group_disparities=disparities,
        statistically_significant_patterns=sig_patterns,
        overall_assessment=overall,
    )
