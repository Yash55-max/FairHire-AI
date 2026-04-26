import os
import json
from pathlib import Path
from datetime import datetime
from typing import Any
try:
    from google.cloud import firestore
    FIRESTORE_AVAILABLE = True
except ImportError:
    firestore = None
    FIRESTORE_AVAILABLE = False

class FirestorePersistence:
    def __init__(self):
        self.db = None
        if FIRESTORE_AVAILABLE:
            try:
                # Uses GOOGLE_APPLICATION_CREDENTIALS or default auth
                self.db = firestore.Client()
            except Exception as e:
                print(f"Firestore Error: {e}")

    def save_analysis(self, run_id: str, email: str, metrics: dict[str, Any], bias_data: dict[str, Any] = None):
        if not self.db:
            return
        
        try:
            doc_ref = self.db.collection("analysis_history").document(run_id)
            doc_ref.set({
                "run_id": run_id,
                "user_email": email,
                "timestamp": datetime.utcnow(),
                "metrics": metrics,
                "bias_data": bias_data,
                "status": "completed"
            })
        except Exception as e:
            print(f"Failed to save to Firestore: {e}")

    def get_history(self, email: str = None, limit: int = 10):
        # Try Firestore first
        if self.db:
            try:
                query = self.db.collection("analysis_history")
                if email:
                    query = query.where("user_email", "==", email)
                
                docs = query.order_by("timestamp", direction=firestore.Query.DESCENDING).limit(limit).stream()
                results = [doc.to_dict() for doc in docs]
                if results:
                    return results
            except Exception as e:
                print(f"Firestore Fetch Error: {e}")

        # Fallback to local runs.json
        try:
            # Path relative to backend/app/persistence.py -> backend/data/runs.json
            runs_file = Path(__file__).resolve().parents[1] / "data" / "runs.json"
            if runs_file.exists():
                with runs_file.open("r", encoding="utf-8") as f:
                    records = json.load(f)
                    
                    # Normalize for frontend (map created_at to timestamp if missing)
                    for r in records:
                        if "created_at" in r and "timestamp" not in r:
                            r["timestamp"] = r["created_at"]
                    
                    # Filter by email if provided
                    if email:
                        records = [r for r in records if r.get("user_email") == email or r.get("user_email") == "anonymous"]
                    
                    # Sort by timestamp descending
                    records.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
                    
                    # Limit results
                    return records[:limit]
        except Exception as e:
            print(f"Local History Fetch Error: {e}")

        return []

# Singleton instance
persistence = FirestorePersistence()
