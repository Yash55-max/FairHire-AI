"""
Conversational Fairness Assistant
===================================
Answers natural-language questions about bias, candidate decisions,
and hiring fairness using local rule-based NLU + ML models.
Falls back to a structured template when no model is available.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Intent taxonomy
# ---------------------------------------------------------------------------
INTENT_PATTERNS: list[tuple[str, list[str]]] = [
    ("why_rejected",   ["why.*reject", "reason.*reject", "rejected.*candidate", "not selected", "why.*failed"]),
    ("why_selected",   ["why.*select", "why.*hired", "why.*accepted", "why.*passed"]),
    ("is_biased",      ["is.*bias", "bias.*detect", "any.*bias", "biased.*process", "fair.*process"]),
    ("explain_score",  ["explain.*score", "score.*breakdown", "how.*scored", "what.*score"]),
    ("group_parity",   ["group.*parity", "parity", "selection.*rate", "disparity", "demographic"]),
    ("top_features",   ["top.*feature", "important.*feature", "what.*feature", "feature.*impact"]),
    ("fairness_index", ["fairness.*index", "fairness.*score", "how.*fair", "fairness.*metric"]),
    ("what_if",        ["what.*if", "if.*remove", "scenario", "simulate", "change.*feature"]),
    ("recommend_fix",  ["how.*fix", "fix.*bias", "mitigate", "reduce.*bias", "improve.*fairness", "recommendation"]),
    ("verdict",        ["verdict", "pass.*fail", "fail", "review.*decision", "decision.*classify"]),
    ("help",           ["help", "what can you", "capabilities", "commands", "usage"]),
]


def detect_intent(question: str) -> str:
    """Rule-based intent classifier using regex patterns."""
    q = question.lower().strip()
    for intent, patterns in INTENT_PATTERNS:
        for pat in patterns:
            if re.search(pat, q):
                return intent
    return "general"


# ---------------------------------------------------------------------------
# Response templates
# ---------------------------------------------------------------------------

def _response_why_rejected(context: dict[str, Any]) -> str:
    cid = context.get("candidate_id", "the candidate")
    neg = context.get("top_negative_factors", ["experience gap", "assessment score"])
    score = context.get("score", "N/A")
    explain = context.get("explanation", "")
    neg_str = ", ".join(neg) if isinstance(neg, list) else neg
    return (
        f"Candidate {cid} received a score of {score}/100 and was not recommended. "
        f"The primary negative drivers were: **{neg_str}**. "
        f"{explain} "
        "If you believe this decision is unfair, use the What-If Simulator to explore alternative feature weights."
    )


def _response_why_selected(context: dict[str, Any]) -> str:
    cid = context.get("candidate_id", "the candidate")
    pos = context.get("top_positive_factors", ["strong experience", "high assessment score"])
    score = context.get("score", "N/A")
    pos_str = ", ".join(pos) if isinstance(pos, list) else pos
    return (
        f"Candidate {cid} scored {score}/100 and was recommended. "
        f"Key positive drivers: **{pos_str}**. "
        "The decision is based on merit-relevant features evaluated by the trained model."
    )


def _response_is_biased(context: dict[str, Any]) -> str:
    verdict = context.get("bias_verdict", "REVIEW")
    fi = context.get("fairness_index", "N/A")
    dpd = context.get("demographic_parity_difference", "N/A")
    recos = context.get("recommendations", [])
    reco_str = (" Recommended steps: " + "; ".join(recos[:2]) + ".") if recos else ""
    icon = "✅" if verdict == "PASS" else "🚨" if verdict == "FAIL" else "⚠️"
    return (
        f"{icon} Fairness Verdict: **{verdict}**. "
        f"Fairness Index: {fi} (1.0 = perfect). "
        f"Demographic Parity Gap: {dpd}. "
        f"{reco_str} "
        "Run the Dual Evaluation to compare full vs bias-masked model decisions."
    )


def _response_explain_score(context: dict[str, Any]) -> str:
    cid = context.get("candidate_id", "the candidate")
    score = context.get("score", "N/A")
    pos = context.get("top_positive_factors", [])
    neg = context.get("top_negative_factors", [])
    pos_str = ", ".join(pos) or "none identified"
    neg_str = ", ".join(neg) or "none identified"
    return (
        f"Score for {cid}: **{score}/100**. "
        f"Positive contributors: {pos_str}. "
        f"Negative contributors: {neg_str}. "
        "SHAP values quantify each feature's contribution to the model output on a scale from −1 to +1."
    )


def _response_group_parity(context: dict[str, Any]) -> str:
    srg = context.get("selection_rate_by_group", {})
    if srg:
        lines = [f"  • {g}: {v:.1%}" for g, v in srg.items()]
        return (
            "Selection rates by group:\n" + "\n".join(lines) + "\n\n"
            f"Demographic Parity Difference: **{context.get('demographic_parity_difference', 'N/A')}**. "
            "A gap > 10% typically indicates a fairness concern."
        )
    return "No group parity data available yet. Run the Fairness Audit first."


def _response_top_features(context: dict[str, Any]) -> str:
    features = context.get("top_global_features", [])
    if features:
        lines = [f"  {i+1}. **{f['feature']}** — importance {f.get('mean_abs_shap', f.get('importance', 0)):.3f}" for i, f in enumerate(features[:5])]
        return "Top features driving hiring decisions:\n" + "\n".join(lines) + "\n\nFeatures with high SHAP values have the most influence on outcomes."
    return "No feature importance data available. Train a model and run explainability first."


def _response_fairness_index(context: dict[str, Any]) -> str:
    fi = context.get("fairness_index", None)
    if fi is None:
        return "Fairness index not yet computed. Run the Bias Audit from the Fairness Audit page."
    level = "Excellent" if fi >= 0.90 else "Good" if fi >= 0.80 else "Moderate" if fi >= 0.70 else "Poor"
    return (
        f"**Fairness Index: {fi:.3f}** ({level}). "
        "Scale: 0.0 (fully biased) → 1.0 (perfectly fair). "
        f"Threshold for PASS verdict: ≥ 0.85. "
        f"Current status: {'✅ PASS' if fi >= 0.85 else '⚠️ REVIEW' if fi >= 0.70 else '🚨 FAIL'}."
    )


def _response_what_if(context: dict[str, Any]) -> str:
    return (
        "To run a What-If simulation:\n"
        "1. Navigate to **Fairness Audit → What-If Simulator**.\n"
        "2. Adjust the decision threshold and reweighting sliders.\n"
        "3. For candidate-level simulation, go to **Candidate Scoring → What-If** and select a candidate to modify.\n"
        "The simulator will recompute scores and rankings without retraining the model."
    )


def _response_recommend(context: dict[str, Any]) -> str:
    recos = context.get("recommendations", [])
    verdict = context.get("bias_verdict", "REVIEW")
    base = [
        "1. **Reweight training data** — oversample underrepresented groups.",
        "2. **Remove proxy features** — college name, ZIP code, referral source.",
        "3. **Apply threshold calibration** — use the What-If Simulator to find a fairer threshold.",
        "4. **Enforce explainability** — require SHAP justification for every adverse decision.",
        "5. **Run Dual Evaluation** — compare Model A vs B to isolate bias-driven differences.",
    ]
    if recos:
        base = [f"{i+1}. {r}" for i, r in enumerate(recos)] + base[len(recos):]
    return (
        f"Recommended fairness improvement steps for verdict **{verdict}**:\n"
        + "\n".join(base[:5])
    )


def _response_verdict(context: dict[str, Any]) -> str:
    v = context.get("bias_verdict") or context.get("dual_verdict") or "Not yet computed"
    fi = context.get("fairness_index", "N/A")
    vd = context.get("verdict_detail", "")
    return f"**Current verdict: {v}**. Fairness index: {fi}. {vd}"


def _response_help() -> str:
    return (
        "I can answer questions about this hiring audit platform. Try asking:\n"
        '• "Why was candidate CAND-003 rejected?"\n'
        '• "Is this hiring process biased?"\n'
        '• "What are the top features driving decisions?"\n'
        '• "What is the fairness index?"\n'
        '• "How can I fix the bias?"\n'
        '• "What is the group parity?"\n'
        '• "Run a what-if simulation"\n'
        '• "What is the current verdict?"\n'
    )


def _response_general(question: str, context: dict[str, Any]) -> str:
    return (
        f"I received your question: \"{question}\"\n\n"
        "I'm a fairness-focused assistant. I can help with bias detection, candidate explanations, "
        "fairness metrics, and remediation recommendations. "
        "Try asking: \"Is this process biased?\" or \"Why was a candidate rejected?\""
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

INTENT_HANDLERS = {
    "why_rejected":   _response_why_rejected,
    "why_selected":   _response_why_selected,
    "is_biased":      _response_is_biased,
    "explain_score":  _response_explain_score,
    "group_parity":   _response_group_parity,
    "top_features":   _response_top_features,
    "fairness_index": _response_fairness_index,
    "what_if":        _response_what_if,
    "recommend_fix":  _response_recommend,
    "verdict":        _response_verdict,
}


def chat(question: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Process a natural-language question and return a structured response.

    Parameters
    ----------
    question : user's natural language question
    context  : flat dict containing relevant audit data (fairness index,
               top features, candidate score, etc.)
    """
    ctx = context or {}
    intent = detect_intent(question)

    if intent == "help":
        answer = _response_help()
    elif intent in INTENT_HANDLERS:
        handler = INTENT_HANDLERS[intent]
        import inspect
        if len(inspect.signature(handler).parameters) == 0:
            answer = handler()          # type: ignore[call-arg]
        else:
            answer = handler(ctx)       # type: ignore[call-arg]
    else:
        answer = _response_general(question, ctx)

    # Suggested follow-up questions based on intent
    followups_map: dict[str, list[str]] = {
        "is_biased":      ["How can I fix the bias?", "What are the top features?", "Show group parity"],
        "why_rejected":   ["Show score breakdown", "Run a what-if simulation", "Is this biased?"],
        "why_selected":   ["Show score breakdown", "What are the top features?"],
        "top_features":   ["Explain the fairness index", "Is this biased?", "Show group parity"],
        "fairness_index": ["How can I improve fairness?", "What is the verdict?", "Show group parity"],
        "recommend_fix":  ["Run a what-if simulation", "What is the fairness index?"],
        "general":        ["Is this process biased?", "What are the top features?", "How do I improve fairness?"],
    }
    followups = followups_map.get(intent, ["Is this biased?", "How can I fix the bias?", "What are the top features?"])

    return {
        "question": question,
        "intent": intent,
        "answer": answer,
        "suggested_followups": followups[:3],
        "data_used": list(ctx.keys()),
    }
