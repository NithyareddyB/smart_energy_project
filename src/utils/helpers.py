"""
src/utils/helpers.py
--------------------
Shared utility functions used across all modules.

Includes:
  - Timer context manager
  - Metric formatting helpers
  - Data validation utilities
  - File I/O helpers
"""

from __future__ import annotations

import time
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

@contextmanager
def timer(label: str = "block") -> Generator[None, None, None]:
    """
    Context manager that logs elapsed time for any code block.

    Usage:
        with timer("LSTM training"):
            model.fit(...)
        # logs: "LSTM training completed in 42.3s"
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        log.info("%s completed in %.2fs", label, elapsed)


# ---------------------------------------------------------------------------
# Metric formatting
# ---------------------------------------------------------------------------

def format_metrics(metrics: dict[str, float], prefix: str = "") -> str:
    """
    Return a human-readable string for a dict of metric values.

    Example:
        format_metrics({"RMSE": 12.4, "MAE": 8.1, "MAPE": 5.2})
        → "RMSE=12.4000  MAE=8.1000  MAPE=5.2000"
    """
    parts = [f"{prefix}{k}={v:.4f}" for k, v in metrics.items()]
    return "  ".join(parts)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error (returns %, e.g. 5.2 means 5.2%)."""
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str = "model",
) -> dict[str, float]:
    """Compute RMSE, MAE, MAPE and log them."""
    metrics = {
        "RMSE": rmse(y_true, y_pred),
        "MAE":  mae(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
    }
    log.info("[%s] %s", label, format_metrics(metrics))
    return metrics


# ---------------------------------------------------------------------------
# Data validation
# ---------------------------------------------------------------------------

def check_dataframe(
    df: pd.DataFrame,
    required_columns: list[str],
    name: str = "DataFrame",
) -> bool:
    """
    Validate that a DataFrame has the required columns and is non-empty.
    Logs a warning for missing columns.
    Returns True if valid, False otherwise.
    """
    if df.empty:
        log.warning("%s is empty", name)
        return False

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        log.warning("%s missing columns: %s", name, missing)
        return False

    missing_pct = df[required_columns].isnull().mean() * 100
    for col, pct in missing_pct.items():
        if pct > 0:
            log.debug("%s | %s missing: %.1f%%", name, col, pct)

    return True


def validate_date_range(df: pd.DataFrame, date_col: str = "timestamp") -> None:
    """Log the date range and total rows of a time-series DataFrame."""
    if date_col in df.columns:
        log.info(
            "Date range: %s → %s  (%d rows)",
            df[date_col].min(), df[date_col].max(), len(df),
        )
    else:
        log.info("DataFrame shape: %s", df.shape)


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def save_json(data: dict[str, Any], path: str | Path) -> None:
    """Save a dictionary as a pretty-printed JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    log.info("Saved JSON → %s", path)


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON file into a dictionary."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def ensure_parent(path: str | Path) -> Path:
    """Create parent directories for a file path if they don't exist."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def file_hash(path: str | Path) -> str:
    """Return the MD5 hash of a file — useful for cache invalidation."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Timer
    with timer("test sleep"):
        time.sleep(0.1)

    # Metrics
    y_true = np.array([100, 200, 300, 400, 500], dtype=float)
    y_pred = np.array([110, 190, 310, 390, 510], dtype=float)
    metrics = compute_all_metrics(y_true, y_pred, label="test")
    print(format_metrics(metrics))

    # DataFrame validation
    df = pd.DataFrame({"timestamp": pd.date_range("2023-01-01", periods=5, freq="h"),
                        "ghi": [100, 200, None, 400, 500]})
    check_dataframe(df, required_columns=["timestamp", "ghi", "demand_mw"], name="test_df")

    print("\nhelpers.py OK")
