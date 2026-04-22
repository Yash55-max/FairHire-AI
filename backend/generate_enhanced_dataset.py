import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Set seed for reproducibility
np.random.seed(42)

# Read current dataset
df = pd.read_csv('data/sample_dataset.csv')

print(f"Original dataset shape: {df.shape}")
print(f"Original columns: {df.columns.tolist()}")

# ============ ADD RECOMMENDED FIELDS ============

# 1. CORE PREDICTIVE FEATURES
df['skills_score'] = np.random.randint(35, 100, size=len(df))
df['interview_score'] = np.random.randint(30, 95, size=len(df))
df['projects_count'] = np.random.randint(0, 12, size=len(df))
df['certifications_count'] = np.random.randint(0, 5, size=len(df))

# 2. BEHAVIORAL / CONTEXT FEATURES
df['employment_gap_months'] = np.random.choice(
    [0, 3, 6, 12, 18, 24, 36], 
    size=len(df), 
    p=[0.5, 0.2, 0.1, 0.08, 0.06, 0.04, 0.02]
)
df['job_role'] = np.random.choice(['tech', 'non-tech', 'management'], size=len(df), p=[0.5, 0.35, 0.15])
df['application_source'] = np.random.choice(['portal', 'referral', 'recruiter'], size=len(df), p=[0.4, 0.35, 0.25])
df['company_tier'] = np.random.choice([1, 2, 3], size=len(df), p=[0.3, 0.5, 0.2])

# 3. PROXY FEATURES
df['college_tier'] = np.random.choice([1, 2, 3, 4, 5], size=len(df), p=[0.1, 0.2, 0.4, 0.2, 0.1])
df['zip_code'] = np.random.randint(10000, 99999, size=len(df))

# ============ GENERATE REALISTIC HIRED TARGET ============
# Formula: combination of scores + realistic bias
# Goal: ~25% hiring rate with clear gender bias

# Raw score components (higher = better candidate)
score = (
    df['interview_score'] * 0.35 +           # Interview is most important
    df['skills_score'] * 0.25 +              # Skills second
    (df['years_experience'] * 2) * 0.20 +    # Experience
    (df['projects_count'] * 5) * 0.10 +      # Projects
    (df['certifications_count'] * 3) * 0.05  # Certs
)

# Gender bias: males systematically score +3 points
gender_bias = np.where(df['gender'] == 'Male', 3.0, 0)

# Employment gap penalty: -0.5 per 6 months
gap_penalty = -(df['employment_gap_months'] / 6.0) * 0.5

# Referral bonus: +2 for employee referral
referral_bonus = np.where(df['referral_source'] == 'Employee Referral', 2.0, 0)

# College tier boost: +0.5 per tier level
college_boost = (df['college_tier'] - 1) * 0.5

# Add realistic noise
noise = np.random.normal(0, 2.0, size=len(df))

# Combine all factors
final_score = score + gender_bias + gap_penalty + referral_bonus + college_boost + noise

# Generate hiring decision based on score threshold
threshold = np.percentile(final_score, 75)  # Top 25% get hired
df['hired'] = (final_score >= threshold).astype(int)

# ============ VERIFY AND SAVE ============
print(f"\n✅ Enhanced dataset created with {len(df.columns)} features")
print(f"New columns added: {set(df.columns) - set(['candidate_id', 'age', 'gender', 'education', 'years_experience', 'assessment_score', 'referral_source', 'role_applied'])}")

# Print summary statistics
print(f"\n📊 Dataset Summary:")
print(f"- Total records: {len(df)}")
print(f"- Hired: {df['hired'].sum()} ({100*df['hired'].mean():.1f}%)")
print(f"- Gender distribution:")
print(f"  - Male hired: {100*df[df['gender']=='Male']['hired'].mean():.1f}%")
print(f"  - Female hired: {100*df[df['gender']=='Female']['hired'].mean():.1f}%")
print(f"  - Non-binary hired: {100*df[df['gender']=='Non-binary']['hired'].mean():.1f}%")
print(f"\n- Fairness gap (Male vs Female): {100*(df[df['gender']=='Male']['hired'].mean() - df[df['gender']=='Female']['hired'].mean()):.1f}%")

# Save enhanced dataset
output_path = 'data/sample_dataset_enhanced.csv'
df.to_csv(output_path, index=False)
print(f"\n✅ Enhanced dataset saved to: {output_path}")

# Also overwrite original with enhanced version
df.to_csv('data/sample_dataset.csv', index=False)
print(f"✅ Original dataset updated with enhanced version")

print(f"\nFirst few rows:")
print(df.head())
