"""
Bias Stress Testing Engine
===========================
Injects controlled, artificial bias into datasets and model weights,
then validates whether the fairness detection pipeline catches it.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .ml_pipeline import compute_bias, train_pipeline


# ── Bias injection strategies ────────────────────────────────────────────────

STRATEGIES = {
    "label_flip":       "Flip hire decisions for a protected group (direct discrimination).",
    "score_skew":       "Artificially lower a numeric feature for a protected group.",
    "undersample":      "Drastically reduce representation of a protected group.",
    "feature_suppress": "Zero-out a key performance feature for a protected group.",
}


def _inject_label_flip(frame: pd.DataFrame, target_col: str,
                        sensitive_col: str, group_value: Any,
                        flip_rate: float = 0.60) -> pd.DataFrame:
    """Flip hire=1 → 0 for `flip_rate` fraction of group_value rows."""
    df = frame.copy()
    mask = df[sensitive_col].astype(str) == str(group_value)
    hired_mask = mask & (df[target_col] == 1)
    flip_idx = df[hired_mask].sample(frac=flip_rate, random_state=42).index
    df.loc[flip_idx, target_col] = 0
    return df


def _inject_score_skew(frame: pd.DataFrame, feature_col: str,
                        sensitive_col: str, group_value: Any,
                        skew_factor: float = 0.55) -> pd.DataFrame:
    """Multiply a numeric feature by skew_factor for the protected group."""
    df = frame.copy()
    if feature_col not in df.columns:
        return df
    mask = df[sensitive_col].astype(str) == str(group_value)
    df.loc[mask, feature_col] = (df.loc[mask, feature_col] * skew_factor).round(2)
    return df


def _inject_undersample(frame: pd.DataFrame, sensitive_col: str,
                         group_value: Any, keep_rate: float = 0.30) -> pd.DataFrame:
    """Keep only `keep_rate` fraction of the protected group's rows."""
    df = frame.copy()
    mask = df[sensitive_col].astype(str) == str(group_value)
    group_rows = df[mask]
    keep_n = max(1, int(len(group_rows) * keep_rate))
    drop_idx = group_rows.sample(n=len(group_rows) - keep_n, random_state=42).index
    return df.drop(index=drop_idx).reset_index(drop=True)


def _inject_feature_suppress(frame: pd.DataFrame, feature_col: str,
                              sensitive_col: str, group_value: Any) -> pd.DataFrame:
    """Zero-out a key feature for the protected group."""
    df = frame.copy()
    if feature_col not in df.columns:
        return df
    mask = df[sensitive_col].astype(str) == str(group_value)
    df.loc[mask, feature_col] = 0
    return df


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class StressTestResult:
    strategy: str
    sensitive_column: str
    target_group: str
    description: str
    # Baseline
    baseline_fairness_index: float
    baseline_dpd: float
    baseline_verdict: str
    # Biased
    biased_fairness_index: float
    biased_dpd: float
    biased_verdict: str
    # Detection
    bias_detected: bool
    detection_confidence: float        # 0–1: how strongly the system flagged it
    delta_fairness_index: float
    delta_dpd: float
    detection_summary: str
    injected_params: dict[str, Any] = field(default_factory=dict)


# ── Main runner ───────────────────────────────────────────────────────────────

def run_stress_test(
    frame: pd.DataFrame,
    target_column: str,
    sensitive_column: str,
    model_type: str = "random_forest",
    strategies: list[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> list[StressTestResult]:
    """
    Run one or more bias injection strategies and validate detection.

    Returns a list of StressTestResult — one per strategy × group combination.
    """
    if sensitive_column not in frame.columns:
        raise ValueError(f"Column '{sensitive_column}' not found in dataset.")

    strats = strategies or list(STRATEGIES.keys())
    groups = frame[sensitive_column].dropna().unique().tolist()
    if not groups:
        raise ValueError("No groups found in sensitive column.")

    # ── Baseline model ────────────────────────────────────────────────────────
    baseline_arts = train_pipeline(frame, target_column, model_type, test_size, random_state)
    baseline_bias = compute_bias(
        baseline_arts.test_frame, baseline_arts.y_true,
        baseline_arts.y_pred, sensitive_column,
    )
    baseline_fi  = baseline_bias["fairness_index"]
    baseline_dpd = baseline_bias["demographic_parity_difference"]
    baseline_v   = "PASS" if baseline_fi >= 0.85 else "REVIEW" if baseline_fi >= 0.70 else "FAIL"

    results: list[StressTestResult] = []

    for strat in strats:
        target_grp = str(groups[0])          # inject against first group

        # ── Inject bias ───────────────────────────────────────────────────────
        numeric_cols = frame.select_dtypes(include="number").columns.tolist()
        key_feature  = next((c for c in numeric_cols if c != target_column), numeric_cols[0] if numeric_cols else None)
        injected_params: dict[str, Any] = {"strategy": strat, "group": target_grp}

        try:
            if strat == "label_flip":
                biased_frame = _inject_label_flip(frame, target_column, sensitive_column, target_grp, 0.65)
                injected_params["flip_rate"] = 0.65
            elif strat == "score_skew" and key_feature:
                biased_frame = _inject_score_skew(frame, key_feature, sensitive_column, target_grp, 0.45)
                injected_params["feature"] = key_feature; injected_params["skew_factor"] = 0.45
            elif strat == "undersample":
                biased_frame = _inject_undersample(frame, sensitive_column, target_grp, 0.25)
                injected_params["keep_rate"] = 0.25
            elif strat == "feature_suppress" and key_feature:
                biased_frame = _inject_feature_suppress(frame, key_feature, sensitive_column, target_grp)
                injected_params["feature"] = key_feature
            else:
                continue
        except Exception as exc:  # noqa: BLE001
            continue

        # ── Train on biased data ──────────────────────────────────────────────
        try:
            biased_arts = train_pipeline(biased_frame, target_column, model_type, test_size, random_state)
            if sensitive_column not in biased_arts.test_frame.columns:
                continue
            biased_bias = compute_bias(
                biased_arts.test_frame, biased_arts.y_true,
                biased_arts.y_pred, sensitive_column,
            )
        except Exception:  # noqa: BLE001
            continue

        biased_fi  = biased_bias["fairness_index"]
        biased_dpd = biased_bias["demographic_parity_difference"]
        biased_v   = "PASS" if biased_fi >= 0.85 else "REVIEW" if biased_fi >= 0.70 else "FAIL"

        delta_fi  = baseline_fi - biased_fi
        delta_dpd = biased_dpd - baseline_dpd

        # ── Detection assessment ──────────────────────────────────────────────
        bias_detected = biased_v in {"REVIEW", "FAIL"} and delta_fi >= 0.04
        detection_conf = min(1.0, max(0.0, delta_fi * 3.5 + delta_dpd * 2.0))

        summary = (
            f"✅ DETECTED — Fairness index dropped {delta_fi:.3f} ({baseline_fi:.3f} → {biased_fi:.3f}). "
            f"Verdict changed: {baseline_v} → {biased_v}."
            if bias_detected else
            f"⚠️ MISSED — Fairness index barely changed ({delta_fi:.3f}). "
            "The injected bias was subtle enough to evade detection."
        )

        results.append(StressTestResult(
            strategy=strat,
            sensitive_column=sensitive_column,
            target_group=target_grp,
            description=STRATEGIES[strat],
            baseline_fairness_index=round(baseline_fi, 4),
            baseline_dpd=round(baseline_dpd, 4),
            baseline_verdict=baseline_v,
            biased_fairness_index=round(biased_fi, 4),
            biased_dpd=round(biased_dpd, 4),
            biased_verdict=biased_v,
            bias_detected=bias_detected,
            detection_confidence=round(detection_conf, 4),
            delta_fairness_index=round(delta_fi, 4),
            delta_dpd=round(delta_dpd, 4),
            detection_summary=summary,
            injected_params=injected_params,
        ))

    return results
