"""
tracking.py — MLflow experiment tracking helpers.

Wraps mlflow calls so the rest of the codebase doesn't import mlflow directly,
making it easy to swap or disable tracking without touching pipeline logic.
"""

import time
from typing import Any, Dict

import mlflow


def start_experiment(name: str = "rag_pipeline") -> None:
    """Create or select an MLflow experiment by name."""
    mlflow.set_experiment(name)


def log_params(params: Dict[str, Any]) -> None:
    """Log pipeline configuration parameters to the active run."""
    mlflow.log_params(params)


def log_metrics(metrics: Dict[str, float]) -> None:
    """Log numeric evaluation metrics to the active run."""
    mlflow.log_metrics(metrics)


class Timer:
    """Context manager that records wall-clock elapsed time in seconds."""

    def __init__(self) -> None:
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        self.elapsed = time.perf_counter() - self._start
