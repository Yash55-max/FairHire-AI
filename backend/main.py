"""Entry point for the FairHire AI backend service.

Run with:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
or via the Makefile:
    make dev-backend
"""
from __future__ import annotations

from app.main import app  # noqa: F401 – re-exported for uvicorn

__all__ = ["app"]
