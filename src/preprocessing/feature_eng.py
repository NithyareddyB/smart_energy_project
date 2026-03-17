"""
src/preprocessing/feature_eng.py
----------------------------------
Transforms the clean hourly DataFrame into a rich feature matrix
used by both the LSTM and Random Forest models.

Features added
--------------
Time features
  hour_sin, hour_cos           — cyclical hour encoding
  dow_sin, dow_cos             — cyclical day-of-week
  month_sin, month_cos         — cyclical month
  is_weekend                   — 0/1 flag

Lag features (for columns in config lag_features list)
  {col}_lag_{h}h  for h in config.preprocessing.lag_hours

Rolling statistics
  {col}_roll{w}h_mean, _std   for w in config.preprocessing.rolling_windows

Physical / derived
  cloud_index                  — (clearsky_ghi - ghi) / (clearsky_ghi + 1e-6)
  solar_fraction               — ghi / (clearsky_ghi + 1e-6)  clamped [0,1]
  wind_power_proxy             — wind_speed³  (proportional to wind power)

Output
------
data/processed/features.csv

Usage (standalone)
------------------
    python -m src.preprocessing.feature_eng
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from src.utils.config import get
from src.utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CLEANED_FILE  = "data/processed/cleaned.csv"
FEATURES_FILE = get("preprocessing.processed_file", "data/processed/features.csv")
LAG_FEATURES: List[str] = get("preprocessing.lag_features",
                               ["ghi", "temperature", "wind_speed", "demand_mw"])
LAG_HOURS: List[int]    = get("preprocessing.lag_hours",
                               [1, 2, 3, 6, 12, 24, 48, 168])
ROLL_WINDOWS: List[int]  = get("preprocessing.rolling_windows", [6, 24, 168])


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class FeatureEngineer:
    """
    Build engineered features from the cleaned DataFrame.

    Parameters
    ----------
    input_path  : Path to cleaned CSV.  Defaults to data/processed/cleaned.csv.
    output_path : Path to save features CSV.
    """

    def __init__(
        self,
        input_path: str = CLEANED_FILE,
        output_path: str = FEATURES_FILE,
    ) -> None:
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    @staticmethod
    def _cyclical(series: pd.Series, period: float) -> tuple[pd.Series, pd.Series]:
        """Convert a numeric series to sin/cos cyclical encoding."""
        radians = 2 * np.pi * series / period
        return np.sin(radians), np.cos(radians)

    # ------------------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all engineered features in-place (on a copy) and save.

        Parameters
        ----------
        df : Clean DataFrame with DatetimeIndex.

        Returns
        -------
        pd.DataFrame with all original + new feature columns.
        """
        log.info("=== FeatureEngineer: building feature matrix ===")
        df = df.copy()

        idx = df.index

        # ── Time features ─────────────────────────────────────────────────────
        df["hour_sin"], df["hour_cos"]   = self._cyclical(idx.hour, 24)
        df["dow_sin"],  df["dow_cos"]    = self._cyclical(idx.dayofweek, 7)
        df["month_sin"], df["month_cos"] = self._cyclical(idx.month, 12)
        df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
        df["hour"] = idx.hour
        log.debug("  Time features added.")

        # ── Lag features ─────────────────────────────────────────────────────
        existing_lag_cols = [c for c in LAG_FEATURES if c in df.columns]
        for col in existing_lag_cols:
            for h in LAG_HOURS:
                df[f"{col}_lag_{h}h"] = df[col].shift(h)
        log.debug("  Lag features added for: %s  lags=%s", existing_lag_cols, LAG_HOURS)

        # ── Rolling statistics ────────────────────────────────────────────────
        for col in existing_lag_cols:
            for w in ROLL_WINDOWS:
                df[f"{col}_roll{w}h_mean"] = df[col].shift(1).rolling(w).mean()
                df[f"{col}_roll{w}h_std"]  = df[col].shift(1).rolling(w).std()
        log.debug("  Rolling features added for windows: %s", ROLL_WINDOWS)

        # ── Physical / derived features ───────────────────────────────────────
        if "clearsky_ghi" in df.columns and "ghi" in df.columns:
            denom = df["clearsky_ghi"] + 1e-6
            df["cloud_index"]    = ((df["clearsky_ghi"] - df["ghi"]) / denom).clip(0, 1)
            df["solar_fraction"] = (df["ghi"] / denom).clip(0, 1)
            log.debug("  cloud_index, solar_fraction added.")

        ws_col = "wind_speed" if "wind_speed" in df.columns else "wind_speed_10m"
        if ws_col in df.columns:
            df["wind_power_proxy"] = df[ws_col] ** 3
            log.debug("  wind_power_proxy added.")

        n_features = df.shape[1]
        n_rows_before = len(df)
        # Drop rows with NaN caused by lags/rolling (all NaN from the lag period)
        df.dropna(inplace=True)
        n_dropped = n_rows_before - len(df)

        log.info(
            "Feature matrix: %d rows × %d cols (dropped %d NaN rows from lag warmup)",
            len(df), n_features, n_dropped,
        )

        df.to_csv(self.output_path)
        log.info("Features saved → %s", self.output_path)
        return df


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.preprocessing.cleaner import DataCleaner
    cleaner = DataCleaner()
    df_clean = cleaner.clean()

    engineer = FeatureEngineer()
    df_features = engineer.transform(df_clean)
    print(df_features.shape)
    print(df_features.columns.tolist())
