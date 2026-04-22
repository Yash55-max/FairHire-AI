import requests
import json

csv_file_path = 'data/sample_dataset.csv'
with open(csv_file_path, 'rb') as f:
    response = requests.post(
        'http://127.0.0.1:8000/train',
        files={'file': ('sample_dataset.csv', f, 'text/csv')},
        data={
            'model_type': 'xgboost',
            'sensitive_column': 'gender',
            'include_fairness_proof': 'true'
        },
        timeout=120
    )

result = response.json()

print(f'Response Status: {response.status_code}')
if response.status_code != 200:
    print(f'Error response: {result}')
    exit(1)

# Extract key metrics
print('=' * 70)
print('FAIRNESS AUDIT REPORT — Enhanced Dataset')
print('=' * 70)
print(f'\n🧠 MODEL PERFORMANCE:')
print(f'  Accuracy: {result.get("accuracy", 0):.4f}')
print(f'  F1 Score: {result.get("f1_score", 0):.4f}')
print(f'  CV Score: {result.get("cv_score", 0):.4f}')

fairness = result.get('fairness', {})
if fairness:
    before = fairness.get('before', {})
    after = fairness.get('after', {})
    method = fairness.get('method', 'N/A')
    acc_impact = fairness.get('accuracy_impact', 'N/A')
    
    before_gap = before.get('demographic_parity_difference', 0)
    after_gap = after.get('demographic_parity_difference', 0)
    
    print(f'\n⚖️  FAIRNESS METRICS (DEMOGRAPHIC PARITY):')
    print(f'  Before mitigation: {before_gap:.4f}')
    print(f'  After mitigation:  {after_gap:.4f}')
    print(f'  Improvement:       {((before_gap - after_gap) / before_gap * 100):.1f}% reduction')
    
    print(f'\n🔧 MITIGATION APPLIED:')
    print(f'  Method:    {method}')
    print(f'  Parameter: {fairness.get("parameter", "N/A")}')
    print(f'  Acc Impact: {acc_impact:.4f}')
    
    print(f'\n📊 BEFORE vs AFTER SELECTION RATES:')
    
    try:
        before_rates = before.get('selection_rate_by_group', {})
        after_rates = after.get('selection_rate_by_group', {})
        
        for group in sorted(set(list(before_rates.keys()) + list(after_rates.keys()))):
            before_rate = before_rates.get(group, 0)
            after_rate = after_rates.get(group, 0)
            print(f'  {group:12} → {before_rate:.1%} → {after_rate:.1%} (delta: {(after_rate - before_rate):+.1%})')
    except:
        pass

print(f'\n📋 DIAGNOSTICS:')
for i, diag in enumerate(result.get('diagnostics', []), 1):
    print(f'  {i}. {diag}')

print(f'\n✅ FAIRNESS IMPROVEMENTS AVAILABLE:')
candidates = fairness.get('candidate_strategies', [])
for i, cand in enumerate(candidates[:3], 1):
    print(f'  {i}. {cand.get("method")} (param={cand.get("parameter")}) → gap={cand.get("after_gap", 0):.4f}, acc_impact={cand.get("accuracy_impact", 0):.4f}')

print(f'\n' + '=' * 70)
print('🎯 NARRATIVE FOR JUDGES:')
print('=' * 70)
print(f'''
We detected a fairness gap of {before_gap:.1%} in our baseline hiring model.
Using automated mitigation ({method}), we reduced this gap to {after_gap:.1%}
while maintaining model accuracy with only {acc_impact:.2%} performance trade-off.

This demonstrates responsible AI: we balance predictive power with equitable outcomes.
''')
