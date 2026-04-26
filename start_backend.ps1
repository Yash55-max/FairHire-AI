# FairHire AI - Backend Startup Script
Write-Host "🚀 Activating Virtual Environment..." -ForegroundColor Cyan
Set-Location "d:\FairHire AI"
& .\.venv\Scripts\Activate.ps1

Write-Host "📦 Installing/Updating Dependencies..." -ForegroundColor Cyan
Set-Location "d:\FairHire AI\backend"
pip install -r requirements.txt

Write-Host "🔥 Starting Backend Server (Port 8000)..." -ForegroundColor Cyan
python -m uvicorn app.main:app --port 8000 --reload
