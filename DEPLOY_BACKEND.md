# Backend Deployment to Cloud Run

## From GCP Console or Local gcloud CLI

### Option 1: Deploy from Git (via Cloud Build)
```bash
gcloud run deploy fairhire-backend \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars "ALLOWED_ORIGINS=http://localhost:5173,http://127.0.0.1:5173,https://fairhire-67f38.web.app,https://fairhire-67f38.firebaseapp.com,https://fairhire-67f38.firebaseapp.com"
```

### Option 2: Deploy Docker Image (if already built)
```bash
gcloud run deploy fairhire-backend \
  --image gcr.io/fairhire-67f38/fairhire-backend:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars "ALLOWED_ORIGINS=http://localhost:5173,http://127.0.0.1:5173,https://fairhire-67f38.web.app,https://fairhire-67f38.firebaseapp.com"
```

## CORS Origins Included
- `http://localhost:5173` — Local frontend dev
- `http://127.0.0.1:5173` — Local frontend dev (IP)
- `https://fairhire-67f38.web.app` — Firebase Hosting production
- `https://fairhire-67f38.firebaseapp.com` — Firebase alternate domain

## After Deployment
1. Test CORS from browser console:
   ```javascript
   fetch('https://fairhire-backend-796656775802.us-central1.run.app/auth/exists?email=test@example.com')
     .then(r => r.json())
     .then(d => console.log(d))
   ```

2. Check response headers include:
   ```
   Access-Control-Allow-Origin: https://fairhire-67f38.web.app
   Access-Control-Allow-Credentials: true
   ```

3. Reload [https://fairhire-67f38.web.app](https://fairhire-67f38.web.app) and test login again.

## Code Changes Applied
- ✅ [backend/app/main.py](backend/app/main.py) — Updated CORS defaults
- ✅ [backend/app/server.py](backend/app/server.py) — Updated CORS defaults  
- ✅ [backend/Dockerfile](backend/Dockerfile) — Added ALLOWED_ORIGINS env var
