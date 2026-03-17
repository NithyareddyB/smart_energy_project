"""
src/preprocessing/cleaner.py
------------------------------
Merges the three raw CSVs (solar, weather, demand) on their timestamps,
fills short gaps, removes sensor outliers, and returns a single clean
DataFrame ready for feature engineering.

Steps
-----
1. Load nasa_solar_raw.csv, weather_raw.csv, demand_raw.csv
2. Resample each to hourly frequency (outer join on timestamp)
3. Forward-fill short gaps (≤ ffill_limit_hours from config)
4. Linearly interpolate remaining NaNs
5. IQR-based outlier clipping for positive-only sensor columns
6. Save to data/processed/cleaned.csv and return DataFrame

Usage (standalone)
------------------
    python -m src.preprocessing.cleaner
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.config import get
from src.utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SOLAR_FILE:   str = get("data.nasa_power.output_file", "data/raw/nasa_solar_raw.csv")
WEATHER_FILE: str = get("data.open_meteo.output_file", "data/raw/weather_raw.csv")
DEMAND_FILE:  str = get("data.grid_demand.output_file", "data/raw/demand_raw.csv")
CLEANED_FILE  = "data/processed/cleaned.csv"
FFILL_LIMIT   = int(get("preprocessing.ffill_limit_hours", 3))

# Columns that must be ≥ 0 (physical irradiance / demand)
_NON_NEGATIVE = {"ghi", "dni", "dhi", "clearsky_ghi", "wind_speed",
                 "wind_speed_10m", "demand_mw", "precipitation",
                 "shortwave_radiation"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_csv(path: str, label: str) -> Optional[pd.DataFrame]:
    p = Path(path)
    if not p.exists():
        log.warning("%s file not found at %s — will be absent from merged set.", label, p)
        return None
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    df.index.name = "timestamp"
    df = df.resample("h").mean()  # ensure hourly frequency
    log.info("  Loaded %s: %d rows, %d cols", label, len(df), df.shape[1])
    return df


def _iqr_clip(df: pd.DataFrame) -> pd.DataFrame:
    """Clip values beyond 1.5×IQR for sensor columns that should be ≥ 0."""
    for col in df.columns:
        if col in _NON_NEGATIVE:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            upper = q3 + 3.0 * iqr  # generous; mainly catches wild spikes
            df[col] = df[col].clip(lower=0.0, upper=upper)
    return df


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DataCleaner:
    """Merge, fill, and clean the three raw data sources."""

    def __init__(self) -> None:
        self.output_path = Path(CLEANED_FILE)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def clean(self, force: bool = False) -> pd.DataFrame:
        """
        Full cleaning pipeline.

        Returns
        -------
        pd.DataFrame  — clean, hourly, DatetimeIndex
        """
        if self.output_path.exists() and not force:
            log.info("Cleaned data already exists at %s — loading.", self.output_path)
            return pd.read_csv(self.output_path, index_col="timestamp",
                               parse_dates=True)

        log.info("=== DataCleaner: merging and cleaning raw data ===")

        # 1. Load
        df_solar   = _load_csv(SOLAR_FILE,   "NASA solar")
        df_weather = _load_csv(WEATHER_FILE, "Open-Meteo weather")
        df_demand  = _load_csv(DEMAND_FILE,  "Grid demand")

        # 2. Merge on timestamp (outer so we don't drop rows if one source is absent)
        frames = [df for df in (df_solar, df_weather, df_demand) if df is not None]
        if not frames:
            raise RuntimeError("No raw data files found.  Run --collect first.")

        df = frames[0]
        for other in frames[1:]:
            df = df.join(other, how="outer")

        # Drop fully-duplicate columns (e.g. wind_speed from both sources)
        df = df.loc[:, ~df.columns.duplicated(keep="first")]

        log.info("Merged shape: %s", df.shape)
        log.info("NaN counts per column before cleaning:\n%s", df.isnull().sum())

        # 3. Forward-fill short gaps
        df = df.ffill(limit=FFILL_LIMIT)

        # 4. Interpolate remaining NaNs linearly
        df = df.interpolate(method="time", limit_direction="both")

        # Any remaining NaNs (at boundaries) → fill with column median
        for col in df.columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

        # 5. Clip outliers
        df = _iqr_clip(df)

        # 6. Sort and drop any leftovers
        df.sort_index(inplace=True)
        df.dropna(how="all", inplace=True)

        nan_remaining = df.isnull().sum().sum()
        log.info("NaN count after cleaning: %d", nan_remaining)

        df.to_csv(self.output_path)
        log.info("Clean data saved → %s   shape=%s", self.output_path, df.shape)
        return df


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cleaner = DataCleaner()
    df = cleaner.clean(force=True)
    print(df.head())
    print(f"Shape: {df.shape}")
    print(df.dtypes)
