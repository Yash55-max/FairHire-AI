"""
Data De-biasing Engine
======================
Identifies sensitive attributes, proxy variables, and flags features that
may indirectly introduce bias (college prestige, career gaps, name, location).
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Attribute catalogs
# ---------------------------------------------------------------------------

SENSITIVE_ATTRS: set[str] = {
    "gender", "sex", "race", "ethnicity", "nationality", "religion",
    "age", "marital_status", "married", "disability", "pregnancy",
    "sexual_orientation", "national_origin", "color",
}

PROXY_KEYWORDS: list[str] = [
    "name", "first_name", "last_name", "surname", "full_name",
    "zip", "zipcode", "postal", "city", "location", "address", "neighborhood",
    "school", "college", "university", "alma_mater", "institution",
    "club", "fraternity", "sorority", "extracurricular",
    "gap", "career_gap", "employment_gap", "break",
    "photo", "image", "profile_pic",
    "linkedin", "social",
]

# Features that correlate with protected attributes but look neutral
PROXY_RISK_MAP: dict[str, str] = {
    "college": "College prestige correlates with socioeconomic status and geography.",
    "university": "University name may encode regional and socioeconomic bias.",
    "school": "School prestige can proxy race and socioeconomic background.",
    "alma_mater": "Alma mater data can encode socioeconomic + demographic bias.",
    "zip": "ZIP/postal code closely proxies race and income.",
    "zipcode": "ZIP/postal code closely proxies race and income.",
    "neighborhood": "Neighborhood name is a strong racial proxy.",
    "city": "City of residence can proxy ethnicity and socioeconomic class.",
    "name": "Candidate name encodes perceived ethnicity and gender.",
    "first_name": "First name signals perceived gender and ethnicity.",
    "last_name": "Last name signals perceived ethnicity.",
    "career_gap": "Career gaps correlate with caregiving, which correlates with gender.",
    "employment_gap": "Employment gaps correlate with caregiving responsibilities.",
    "gap": "Gap columns may reflect caregiving breaks — a gender proxy.",
    "photo": "Profile photos directly expose demographic attributes.",
    "age": "Age is a protected characteristic.",
    "marital_status": "Marital status is a protected characteristic.",
    "married": "Marital / relationship status is protected.",
    "referral_source": "Referral networks often reflect existing demographic imbalances.",
}


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

class DebiasEngine:
    """Analyzes a DataFrame and suggests / applies de-biasing steps."""

    def __init__(self, frame: pd.DataFrame, target_column: str | None = None) -> None:
        self.frame = frame.copy()
        self.target = target_column
        self._columns = [c for c in frame.columns if c != target_column]

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect_sensitive(self) -> list[dict[str, Any]]:
        """Return columns that are likely sensitive attributes."""
        results = []
        for col in self._columns:
            low = col.lower()
            matched = next((s for s in SENSITIVE_ATTRS if s in low), None)
            if matched:
                results.append({
                    "column": col,
                    "type": "sensitive",
                    "reason": f"'{col}' matches protected attribute '{matched}'.",
                    "risk": "high",
                })
        return results

    def detect_proxies(self) -> list[dict[str, Any]]:
        """Return columns that act as proxy variables for protected attributes."""
        results = []
        for col in self._columns:
            low = col.lower()
            for kw, explanation in PROXY_RISK_MAP.items():
                if kw in low and not any(r["column"] == col for r in results):
                    results.append({
                        "column": col,
                        "type": "proxy",
                        "reason": explanation,
                        "risk": "high" if kw in {"name", "first_name", "last_name", "zip", "zipcode"} else "medium",
                    })
        return results

    def detect_high_correlation_features(self, threshold: float = 0.35) -> list[dict[str, Any]]:
        """Return features highly correlated with known sensitive columns."""
        results: list[dict[str, Any]] = []
        sensitive_cols = [f["column"] for f in self.detect_sensitive()]
        if not sensitive_cols:
            return results

        numeric = self.frame.select_dtypes(include="number")
        if numeric.empty:
            return results

        for sens in sensitive_cols:
            if sens not in numeric.columns:
                continue
            for col in numeric.columns:
                if col == sens or col == self.target:
                    continue
                try:
                    corr = abs(float(numeric[sens].corr(numeric[col])))
                    if corr >= threshold:
                        results.append({
                            "column": col,
                            "type": "correlated",
                            "reason": f"Pearson |r| = {corr:.2f} with '{sens}'.",
                            "risk": "medium",
                            "correlated_with": sens,
                        })
                except Exception:  # noqa: BLE001
                    pass
        return results

    def full_audit(self) -> dict[str, Any]:
        """Run the full de-biasing audit and return a structured report."""
        sensitive = self.detect_sensitive()
        proxies = self.detect_proxies()
        correlated = self.detect_high_correlation_features()

        all_flagged = {f["column"] for f in sensitive + proxies + correlated}
        safe = [c for c in self._columns if c not in all_flagged]

        return {
            "total_columns": len(self._columns),
            "safe_columns": safe,
            "sensitive_columns": sensitive,
            "proxy_columns": proxies,
            "correlated_columns": correlated,
            "flagged_count": len(all_flagged),
            "safe_count": len(safe),
            "risk_summary": {
                "high": sum(1 for f in sensitive + proxies if f.get("risk") == "high"),
                "medium": sum(1 for f in correlated + proxies if f.get("risk") == "medium"),
            },
        }

    # ------------------------------------------------------------------
    # Masking / removal
    # ------------------------------------------------------------------

    def mask_columns(self, columns_to_remove: list[str]) -> pd.DataFrame:
        """Return a copy of the frame with listed columns removed."""
        existing = [c for c in columns_to_remove if c in self.frame.columns]
        return self.frame.drop(columns=existing)

    def auto_mask(self) -> tuple[pd.DataFrame, list[str]]:
        """Auto-remove all detected sensitive + high-risk proxy columns."""
        audit = self.full_audit()
        to_remove = (
            [f["column"] for f in audit["sensitive_columns"]]
            + [f["column"] for f in audit["proxy_columns"] if f["risk"] == "high"]
        )
        # Never remove the target
        if self.target in to_remove:
            to_remove.remove(self.target)
        masked = self.mask_columns(to_remove)
        return masked, to_remove
