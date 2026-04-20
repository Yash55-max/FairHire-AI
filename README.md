# FairHire AI — Enterprise Hiring Audit Platform

> Detect bias · Explain decisions · Generate board-ready compliance reports

[![Python](https://img.shields.io/badge/Python-3.10%20|%203.11%20|%203.12-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-latest-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19-61DAFB?logo=react)](https://react.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Overview

**FairHire AI** is an end-to-end responsible-AI platform for auditing hiring models. It allows HR teams, data scientists, and compliance officers to:

| Capability | Description |
|---|---|
| 📤 **Dataset Upload** | Ingest CSV / JSON / Excel hiring data with schema preview |
| 🤖 **Model Training** | Train Random Forest, Logistic Regression, Gradient Boosting, or Decision Tree classifiers |
| ⚖️ **Bias Detection** | Measure demographic parity, equal opportunity, and a composite fairness index |
| 💡 **SHAP Explainability** | Understand global feature importance and individual candidate decisions |
| 🔍 **What-If Simulator** | Test how threshold adjustments and dataset reweighting change fairness |
| 📋 **Compliance Reports** | Generate executive-ready audit packages with AI recommendations |
| 🎯 **Fairness Verdicts** | Automated PASS / REVIEW / FAIL verdicts with remediation steps |

---

## Architecture

```
FairHire-AI/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI routes (upload/train/bias/explain/report/simulate)
│   │   ├── ml_pipeline.py   # Model training, SHAP explainability, fairness metrics
│   │   ├── schemas.py       # Pydantic request/response models
│   │   └── store.py         # Thread-safe in-memory store
│   ├── data/
│   │   ├── sample_dataset.csv   # 50-candidate demo dataset
│   │   └── runs.json            # Persisted run history (auto-created)
│   └── main.py              # Uvicorn entry point
├── frontend/
│   └── src/
│       ├── App.jsx          # Full React SPA (1 file, all pages)
│       └── index.css        # Design system + component styles
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Quick Start

### Prerequisites
- **Python 3.10 – 3.12** (3.11 recommended)
- **Node.js 18+**
- `pip` and `npm`

### 1 — Clone & configure

```bash
git clone <repo-url>
cd FairHire-AI
cp .env.example .env          # review ALLOWED_ORIGINS if needed
```

### 2 — Backend

```bash
# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# Start the API server
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API documentation: http://localhost:8000/docs

### 3 — Frontend

```bash
cd frontend
npm install
npm run dev
```

App: http://localhost:5173

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe |
| `GET` | `/runs` | List all persisted run summaries |
| `POST` | `/upload` | Upload a dataset (multipart/form-data) |
| `POST` | `/train` | Train a classifier on an uploaded dataset |
| `GET` | `/train/{run_id}` | Retrieve training metrics for a run |
| `GET` | `/bias?run_id=&sensitive_column=` | Compute fairness / bias metrics |
| `GET` | `/explain?run_id=&sample_size=` | SHAP-based feature importance |
| `GET` | `/report?run_id=&sensitive_column=` | Full consolidated audit report |
| `POST` | `/simulate/whatif` | Single what-if simulation |
| `POST` | `/simulate/batch` | Batch threshold × reweight simulations |

Full interactive docs: `http://localhost:8000/docs` (Swagger UI)

---

## Supported Models

| Key | Algorithm |
|---|---|
| `random_forest` | Random Forest (default, recommended) |
| `logistic_regression` | Logistic Regression |
| `gradient_boosting` | Gradient Boosting |
| `decision_tree` | Decision Tree |

---

## Fairness Metrics

| Metric | Formula | Threshold |
|---|---|---|
| Demographic Parity Difference | max(SR) − min(SR) | ≤ 0.10 |
| Equal Opportunity Difference | max(TPR) − min(TPR) | ≤ 0.10 |
| Fairness Index | 1 − ((DPD + EOD) / 2) | ≥ 0.85 → PASS |

**Verdict mapping:**
- ✅ **PASS** — FI ≥ 0.85 and DPD ≤ 0.08
- ⚠️ **REVIEW** — FI ≥ 0.70 (moderate disparity)
- 🚨 **FAIL** — FI < 0.70 (significant bias)

---

## Demo Dataset

A sample 50-candidate dataset is pre-loaded at `backend/data/sample_dataset.csv`.

Columns: `candidate_id, age, gender, education, years_experience, assessment_score, referral_source, role_applied, hired`

Use this to test the full audit workflow end-to-end.

---

## Docker

```bash
# Build & run both services
docker-compose up --build

# Backend only
docker build -t fairhire-backend .
docker run -p 8000:8000 fairhire-backend
```

---

## Development

```bash
# Run backend with auto-reload
cd backend && uvicorn main:app --reload --port 8000

# Run frontend with HMR
cd frontend && npm run dev

# Lint frontend
cd frontend && npm run lint
```

---

## License

MIT © FairHire AI Contributors
