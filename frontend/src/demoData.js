export const DEMO_REPORT = {
  session: {
    token: 'demo-token-123',
    user: { email: 'judge@solutionchallenge.org', name: 'Solution Challenge Reviewer' }
  },
  train: {
    run_id: 'demo-run-2026',
    model_name: 'RandomForest_Enterprise_v2',
    target_column: 'hired',
    accuracy: 0.942,
    precision: 0.925,
    recall: 0.891,
    f1_score: 0.908,
    status: 'completed',
    timestamp: '2026-04-26T10:00:00Z',
    feature_importance: {
      'Experience_Years': 0.32,
      'Assessment_Score': 0.28,
      'Technical_Interview': 0.21,
      'Education_Level': 0.12,
      'Referral_Source': 0.05,
      'Location_Region': 0.02
    }
  },
  bias: {
    sensitive_column: 'gender',
    fairness_index: 0.912,
    disparate_impact_ratio: 0.88,
    selection_rate_by_group: {
      'Male': 0.62,
      'Female': 0.58,
      'Non-Binary': 0.60
    },
    bias_detected: false,
    recommendations: [
      "Model demonstrates high demographic parity (91%).",
      "Minor selection gap detected in 'referral_source' correlation.",
      "Recommendation: Maintain current weighting but monitor 'Experience_Years' for proxy drift."
    ]
  },
  explain: {
    summary: "The model primarily relies on objective performance metrics (Assessment and Technical Interview scores). 'Experience_Years' is the strongest predictor, contributing 32% to the final decision. No protected attributes were found to have a significant causal impact on the ranking.",
    shap_values: {
      'Experience_Years': 0.31,
      'Assessment_Score': 0.24,
      'Technical_Interview': 0.19,
      'Referral_Source': -0.05,
      'Education_Level': 0.08
    }
  },
  report: {
    summary: "Audit successful. The 'Enterprise_v2' model meets all corporate fairness standards for the 2026 hiring cycle. Accuracy remains above 94% with a fairness index of 91.2%. No systemic bias was detected across gender or regional subgroups.",
    narrative: "This audit was performed on a dataset of 45,000 candidate records. The Gemini Auditor identified 'Referral_Source' as a potential proxy but confirmed that after mitigation, the demographic parity ratio improved from 0.82 to 0.88. The model is cleared for production deployment."
  }
}
