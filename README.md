# FairHire AI ⚖️
**Enterprise Hiring Intelligence for Fairness Auditing & Ethical AI Compliance.**

FairHire AI unifies model evaluation, bias auditing, explainability, and compliance-grade reporting into one guided enterprise workflow. Move from raw datasets to board-ready fairness narratives in minutes.

[![Live Demo](https://img.shields.io/badge/Live-Firebase%20Hosting-FFCA28?logo=firebase&logoColor=black)](https://fairhire-67f38.web.app)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)](https://react.dev)
[![Vite](https://img.shields.io/badge/Vite-Build-646CFF?logo=vite&logoColor=white)](https://vite.dev)
[![Firebase](https://img.shields.io/badge/Firebase-Firestore%20%26%20Hosting-FFCA28?logo=firebase&logoColor=black)](https://firebase.google.com)

---

## 🚀 Key Capabilities

| Capability | Impact |
|---|---|
| **Fairness Verdict Banner** | Immediate risk signal with actionable remediation guidance. |
| **Guided Audit Flow** | Seamless progression from Data Ingest → Training → Bias Audit → Decision Rationale. |
| **Interactive What-if Simulator** | Real-time threshold tuning and reweighting to visualize fairness impact. |
| **FairHire Auditor (LLM)** | Gemini-powered conversational assistant to investigate bias and candidate decisions. |
| **Decision Rationale (SHAP)** | Local and global explainability to justify every model prediction. |
| **Enterprise Persistence** | Hybrid storage with Firebase Firestore and local JSON fallback for high reliability. |

---

## 🛠️ Tech Stack

- **Frontend**: React 19, Framer Motion (premium animations), Recharts, Tailwind CSS.
- **Backend**: FastAPI, Scikit-learn, Fairlearn, SHAP (Explainability).
- **AI Layer**: Google Gemini 2.0 (Conversational Auditor).
- **Storage**: Firebase Firestore (Cloud) + Local JSON Fallback (Persistence).

---

## 🏗️ Architecture

```text
FairHire AI/
├── backend/
│   ├── app/
│   │   ├── assistant.py    # Gemini-powered conversational auditor
│   │   ├── main.py         # API orchestration & CORS configuration
│   │   ├── ml_pipeline.py  # Training, bias metrics, & SHAP logic
│   │   └── persistence.py  # Hybrid storage (Firestore + Local fallback)
│   └── data/
│       └── runs.json       # Local persistence store
├── frontend/
│   ├── src/
│   │   ├── App.jsx         # SPA Shell & Navigation logic
│   │   ├── index.css       # Aurora-theme design system
│   │   └── firebase.js     # Cloud configuration
├── .gitignore              # Comprehensive environment & data exclusion
└── README.md               # Product documentation
```

---

## 🚦 Getting Started (Local Development)

### 1. Prerequisites
- Python 3.10+
- Node.js 18+
- [Google Gemini API Key](https://aistudio.google.com/) (for Auditor Chat)

### 2. Backend Setup
```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Set GOOGLE_API_KEY in .env
python -m uvicorn app.main:app --port 8000 --reload
```

### 3. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

---

## ☁️ Production Deployment (Firebase Hosting + Cloud Run)

### Backend Deployment
Deploy the FastAPI backend to Cloud Run with CORS enabled for your Firebase domain.
See [DEPLOY_BACKEND.md](DEPLOY_BACKEND.md) for gcloud commands and post-deployment testing.

### Frontend Deployment

#### 1. Configure Frontend API Base URL
Create `frontend/.env.production`:

```env
VITE_API_URL=https://fairhire-backend-796656775802.us-central1.run.app
```

This makes production builds call Cloud Run directly instead of `/api`.

#### 2. Build Frontend
```bash
cd frontend
npm run build
```

#### 3. Deploy to Firebase Hosting
```bash
cd ..
firebase deploy --only hosting
```

#### 4. Live URLs
- Frontend: https://fairhire-67f38.web.app
- API: https://fairhire-backend-796656775802.us-central1.run.app

#### Notes
- Hosting project is mapped in `.firebaserc` (`fairhire-67f38`).
- Hosting serves static files from `frontend/dist` (configured in `firebase.json`).
- Backend must have `ALLOWED_ORIGINS` env var set to include your Firebase domain for CORS to work.

#### Production-safe mode (Cloud Run)
- Keep frontend training requests synchronous in production (`async_job: false`) to avoid cross-instance `/jobs/{id}` polling issues on stateless Cloud Run.
- Keep Cloud Run-safe training enabled in `backend/app/ml_pipeline.py` (skip CV tuning when `K_SERVICE` is present).
- If you switch back to async jobs later, move job storage from in-memory to shared persistence (Firestore/Redis/DB) before enabling it in production.

---

## 📊 Product Walkthrough

### 1. Landing & Authentication
Premium entry point with aurora gradients and secure access controls.
![Landing](screenshots/8_landing_1776685849657.png)

### 2. Guided Ingest & Training
Automatic target detection and protected class flagging during data upload.
![Upload](screenshots/2_upload_1776685934495.png)

### 3. Deep Fairness Audit
Demographic parity tracking and interactive what-if simulations.
![Audit](screenshots/4_fairness_audit_1776685970889.png)

### 4. Conversational Auditor
Ask Gemini about specific bias gaps or candidate justifications in real-time.
![Chat](screenshots/5_decision_insights_1776685987986.png)

---

## 📄 License
MIT License - Developed for Ethical AI Compliance & Responsible Hiring.
