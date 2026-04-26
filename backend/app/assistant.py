"""
Conversational Fairness Assistant (Gemini-Powered)
===================================================
Answers natural-language questions about bias, candidate decisions,
and hiring fairness using Google Gemini AI with a rule-based fallback.
"""
from __future__ import annotations

import os
import re
from typing import Any
import json

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    genai = None
    GEMINI_AVAILABLE = False

# ---------------------------------------------------------------------------
# Intent taxonomy (for fallback and suggestion logic)
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
# Gemini Integration
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are the FairHire AI Auditor, a sophisticated AI assistant designed to help HR managers and data scientists understand hiring fairness, model bias, and explainability.

Your tone is professional, authoritative, yet helpful and calm. You are an expert in Responsible AI, SHAP (SHapley Additive exPlanations), and fairness metrics like Demographic Parity and Equal Opportunity.

When answering:
1. Use the provided audit context (metrics, scores, features) to give specific, data-driven answers.
2. If asked about bias, explain the implications of the "Fairness Index" and "Demographic Parity Gap."
3. If asked about a candidate, explain the drivers (positive and negative) behind their score using available data.
4. Always suggest actionable next steps for bias mitigation (e.g., reweighting, feature masking, threshold adjustment).
5. Be transparent about what the data shows. Do not hide potential risks.
6. Keep your responses concise but insightful (around 2-4 sentences).

Audit Context:
{context_json}
"""

def _call_gemini(question: str, context: dict[str, Any]) -> str | None:
    if not GEMINI_AVAILABLE:
        return None
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None

    try:
        client = genai.Client(api_key=api_key)
        
        # Prepare context for the prompt
        # Filter out very large objects to keep prompt size manageable
        safe_context = {k: v for k, v in context.items() if not isinstance(v, (list, dict)) or len(str(v)) < 1000}
        context_json = json.dumps(safe_context, indent=2)
        
        prompt = f"System: {SYSTEM_PROMPT.format(context_json=context_json)}\n\nUser Question: {question}\n\nAuditor Response:"
        
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"Gemini Error: {e}")
        return None

# ---------------------------------------------------------------------------
# Fallback Response templates (Legacy logic)
# ---------------------------------------------------------------------------

def _response_why_rejected(context: dict[str, Any]) -> str:
    cid = context.get("candidate_id", "the candidate")
    neg = context.get("top_negative_factors", ["experience gap", "assessment score"])
    score = context.get("score", "N/A")
    neg_str = ", ".join(neg) if isinstance(neg, list) else str(neg)
    return (
        f"Candidate {cid} received a score of {score}/100 and was not recommended. "
        f"The primary negative drivers were: **{neg_str}**. "
        "If you believe this decision is unfair, use the What-If Simulator to explore alternative feature weights."
    )

def _response_is_biased(context: dict[str, Any]) -> str:
    verdict = context.get("bias_verdict", "REVIEW")
    fi = context.get("fairness_index", "N/A")
    dpd = context.get("demographic_parity_difference", "N/A")
    icon = "✅" if verdict == "PASS" else "🚨" if verdict == "FAIL" else "⚠️"
    return (
        f"{icon} Fairness Verdict: **{verdict}**. "
        f"Fairness Index: {fi} (1.0 = perfect). "
        f"Demographic Parity Gap: {dpd}. "
        "Run the Dual Evaluation to compare full vs bias-masked model decisions."
    )

def _response_help() -> str:
    return (
        "I am the FairHire AI Auditor. I can answer questions about your hiring audit. Try asking:\n"
        '• "Is this hiring process biased?"\n'
        '• "Why was candidate CAND-003 rejected?"\n'
        '• "What are the top features driving decisions?"\n'
        '• "How can I fix the detected bias?"'
    )

# ---------------------------------------------------------------------------
# Main Chat Entry Point
# ---------------------------------------------------------------------------

def chat(question: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Process a natural-language question using Gemini (if available) or fallback logic.
    """
    ctx = context or {}
    
    # 1. Try Gemini
    answer = _call_gemini(question, ctx)
    
    # 2. Fallback to rule-based if Gemini failed or is unavailable
    if not answer:
        intent = detect_intent(question)
        if intent == "help":
            answer = _response_help()
        elif intent == "why_rejected":
            answer = _response_why_rejected(ctx)
        elif intent == "is_biased":
            answer = _response_is_biased(ctx)
        else:
            answer = f"I'm currently operating in fallback mode. Regarding your question: \"{question}\", please ensure your audit data is fully loaded in the dashboard."

    # Suggested follow-up questions
    intent = detect_intent(question)
    followups_map: dict[str, list[str]] = {
        "is_biased":      ["How can I fix the bias?", "What are the top features?", "Show group parity"],
        "why_rejected":   ["Show score breakdown", "Run a what-if simulation", "Is this biased?"],
        "general":        ["Is this process biased?", "What are the top features?", "How do I improve fairness?"],
    }
    followups = followups_map.get(intent, ["Is this biased?", "How can I fix the bias?", "What are the top features?"])

    return {
        "question": question,
        "intent": intent,
        "answer": answer,
        "suggested_followups": followups[:3],
        "powered_by": "Gemini AI" if answer and GEMINI_AVAILABLE and os.getenv("GOOGLE_API_KEY") else "Rule-based Engine"
    }
