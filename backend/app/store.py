from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock
from typing import Any

import pandas as pd


@dataclass
class TrainingRun:
    run_id: str
    dataset_id: str
    model_type: str
    target_column: str
    model: Any
    test_frame: pd.DataFrame
    y_true: pd.Series
    y_pred: pd.Series
    metrics: dict[str, Any]
    created_at: datetime
    tags: list[str] = field(default_factory=list)
    notes: str = ""


class InMemoryStore:
    """Thread-safe in-memory store for datasets and training runs."""

    def __init__(self) -> None:
        self._lock = Lock()
        self.datasets: dict[str, pd.DataFrame] = {}
        self.runs: dict[str, TrainingRun] = {}

    # ---- Datasets ----------------------------------------------------------

    def put_dataset(self, dataset_id: str, frame: pd.DataFrame) -> None:
        with self._lock:
            self.datasets[dataset_id] = frame

    def get_dataset(self, dataset_id: str) -> pd.DataFrame:
        with self._lock:
            if dataset_id not in self.datasets:
                raise KeyError(f"Dataset '{dataset_id}' not found. Upload it first.")
            return self.datasets[dataset_id]

    def list_datasets(self) -> list[str]:
        with self._lock:
            return list(self.datasets.keys())

    def delete_dataset(self, dataset_id: str) -> None:
        with self._lock:
            self.datasets.pop(dataset_id, None)

    # ---- Runs --------------------------------------------------------------

    def put_run(self, run: TrainingRun) -> None:
        with self._lock:
            self.runs[run.run_id] = run

    def get_run(self, run_id: str) -> TrainingRun:
        with self._lock:
            if run_id not in self.runs:
                raise KeyError(f"Run '{run_id}' not found. Train a model first.")
            return self.runs[run_id]

    def list_runs(self) -> list[str]:
        with self._lock:
            return list(self.runs.keys())

    def delete_run(self, run_id: str) -> None:
        with self._lock:
            self.runs.pop(run_id, None)

    def recent_run(self) -> TrainingRun | None:
        """Return the most recently created run, or None."""
        with self._lock:
            if not self.runs:
                return None
            return max(self.runs.values(), key=lambda r: r.created_at)
