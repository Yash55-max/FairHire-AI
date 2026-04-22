import requests
import json

# Read CSV file
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
        timeout=60
    )

result = response.json()

print(f'Response Status: {response.status_code}')
if response.status_code != 200:
    print(f'Error response: {result}')
    exit(1)

# Extract key metrics
print('=== METRICS VALIDATION ===')
acc = result.get("accuracy", 0)
f1 = result.get("f1_score", 0)
cv = result.get("cv_score", 0)
print(f'Accuracy: {acc:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'CV Score: {cv:.4f}')

fairness = result.get('fairness', {})
if fairness:
    before_gap = fairness.get('before', {}).get('demographic_parity_difference', 'N/A')
    after_gap = fairness.get('after', {}).get('demographic_parity_difference', 'N/A')
    method = fairness.get('method', 'N/A')
    acc_impact = fairness.get('accuracy_impact', 'N/A')
    
    print(f'\nFairness Before Gap: {before_gap}')
    print(f'Fairness After Gap: {after_gap}')
    print(f'Method: {method}')
    print(f'Accuracy Impact: {acc_impact}')
    
    print(f'\n=== VALIDATION AGAINST TARGETS ===')
    accuracy = result.get("accuracy", 0)
    print(f'✅ Accuracy in range (0.86-0.91): {0.86 <= accuracy <= 0.91}')
    print(f'✅ Fairness gap after mitigation <= 0.08: {after_gap <= 0.08 if isinstance(after_gap, (int, float)) else "N/A"}')
    
    if isinstance(before_gap, (int, float)) and isinstance(after_gap, (int, float)) and before_gap > 0:
        improvement = (before_gap - after_gap) / before_gap * 100
        print(f'✅ Improvement >= 30%: {improvement >= 30} ({improvement:.1f}% improvement)')

print(f'\nDiagnostics:')
for diag in result.get('diagnostics', []):
    print(f'  - {diag}')

print(f'\nFull response structure:')
print(json.dumps({k: v for k, v in result.items() if k != 'probabilities'}, indent=2, default=str))
