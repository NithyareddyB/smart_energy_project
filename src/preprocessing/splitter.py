"""
src/preprocessing/splitter.py
-------------------------------
Temporal (chronological) train / validation / test split.

IMPORTANT: Random shuffling is intentionally NOT used — time-series data
must be split chronologically to prevent data leakage.

Splits are determined by ratios in config.yaml (default 70/15/15).

The module returns both the raw X/y arrays AND retains a numpy save so
that models can load the splits without re-running the full pipeline.

Usage (standalone)
------------------
    python -m src.preprocessing.splitter
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from src.utils.config import get
from src.utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FEATURES_FILE = get("preprocessing.processed_file", "data/processed/features.csv")
SPLIT_RATIOS  = get("preprocessing.split_ratios", {"train": 0.70, "val": 0.15, "test": 0.15})
SPLITS_DIR    = Path("data/processed/splits")

# These are the primary regression targets; exclude from X, use for y
TARGET_SOLAR  = get("lstm.target_column",          "ghi")
TARGET_WIND   = get("lstm_wind.target_column",     "wind_speed")
TARGET_DEMAND = get("random_forest.target_column", "demand_mw")


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DataSplitter:
    """
    Temporal train/val/test splitter for the feature DataFrame.

    Parameters
    ----------
    features_path : Path to the scaled feature CSV.
    splits_dir    : Directory to persist numpy split files.
    """

    def __init__(
        self,
        features_path: str = FEATURES_FILE,
        splits_dir: str = str(SPLITS_DIR),
    ) -> None:
        self.features_path = Path(features_path)
        self.splits_dir = Path(splits_dir)
        self.splits_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def split(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Split *df* chronologically into train / val / test.

        Parameters
        ----------
        df : Feature DataFrame (scaled, DatetimeIndex).

        Returns
        -------
        dict with keys: X_train, X_val, X_test, y_solar_train, y_solar_val,
        y_solar_test, y_wind_train, y_wind_val, y_wind_test,
        y_demand_train, y_demand_val, y_demand_test
        Also saves .npy files to splits_dir for fast reload.
        """
        log.info("=== DataSplitter: creating temporal splits ===")

        n = len(df)
        train_end = int(n * SPLIT_RATIOS["train"])
        val_end   = train_end + int(n * SPLIT_RATIOS["val"])

        log.info(
            "Total rows=%d  train=%d  val=%d  test=%d",
            n, train_end, val_end - train_end, n - val_end,
        )

        # ── Feature matrix (X) — drop all target columns ──────────────────
        target_cols = {TARGET_SOLAR, TARGET_WIND, TARGET_DEMAND}
        feature_cols = [c for c in df.columns if c not in target_cols]
        X = df[feature_cols].values.astype(np.float32)

        splits: Dict[str, np.ndarray] = {
            "X_train": X[:train_end],
            "X_val":   X[train_end:val_end],
            "X_test":  X[val_end:],
        }

        # ── Target vectors (y) ─────────────────────────────────────────────
        for label, col in (
            ("y_solar",  TARGET_SOLAR),
            ("y_wind",   TARGET_WIND),
            ("y_demand", TARGET_DEMAND),
        ):
            if col not in df.columns:
                log.warning("Target column '%s' not in DataFrame — skipping.", col)
                continue
            y = df[col].values.astype(np.float32)
            splits[f"{label}_train"] = y[:train_end]
            splits[f"{label}_val"]   = y[train_end:val_end]
            splits[f"{label}_test"]  = y[val_end:]

        # ── Save timestamps for later plotting ─────────────────────────────
        splits["ts_train"] = df.index[:train_end]
        splits["ts_val"]   = df.index[train_end:val_end]
        splits["ts_test"]  = df.index[val_end:]

        # ── Persist to disk ────────────────────────────────────────────────
        for key, arr in splits.items():
            if isinstance(arr, np.ndarray):
                np.save(self.splits_dir / f"{key}.npy", arr)
        # Save timestamps as CSV
        for split_name in ("train", "val", "test"):
            splits[f"ts_{split_name}"].to_frame().to_csv(
                self.splits_dir / f"ts_{split_name}.csv"
            )

        log.info("Splits saved to %s", self.splits_dir)
        return splits

    # ------------------------------------------------------------------
    @staticmethod
    def load(splits_dir: str = str(SPLITS_DIR)) -> Dict[str, np.ndarray]:
        """
        Load previously saved splits from .npy files.

        Returns
        -------
        dict mirroring the output of ``split()``.
        """
        d = Path(splits_dir)
        splits: Dict[str, np.ndarray] = {}
        for p in d.glob("*.npy"):
            splits[p.stem] = np.load(p, allow_pickle=True)
        for p in d.glob("ts_*.csv"):
            splits[p.stem] = pd.read_csv(p, index_col=0, parse_dates=True).index
        log.info("Splits loaded from %s  keys=%s", d, list(splits.keys()))
        return splits


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = pd.read_csv(FEATURES_FILE, index_col="timestamp", parse_dates=True)
    splitter = DataSplitter()
    splits = splitter.split(df)
    for k, v in splits.items():
        if hasattr(v, "shape"):
            print(f"  {k}: {v.shape}")
        else:
            print(f"  {k}: {len(v)} timestamps")
