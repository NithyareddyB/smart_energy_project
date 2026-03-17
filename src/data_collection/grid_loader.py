"""
src/data_collection/grid_loader.py
------------------------------------
Loads hourly grid demand data from a local CSV (e.g. POSOCO data).
If the source file is missing it synthesises a realistic Telangana-like
demand profile so the rest of the pipeline can run without real data.

Expected input columns (flexible — normalised internally):
    timestamp / date / datetime  →  normalised to "timestamp"
    demand / demand_mw / load    →  normalised to "demand_mw"
    frequency / freq_hz          →  normalised to "frequency_hz" (optional)

Output
------
data/raw/demand_raw.csv  — timestamp, demand_mw, [frequency_hz]

Usage (standalone)
------------------
    python -m src.data_collection.grid_loader
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from src.utils.config import get
from src.utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SOURCE_FILE: str  = get("data.grid_demand.source",      "data/external/grid_demand.csv")
OUTPUT_FILE: str  = get("data.grid_demand.output_file", "data/raw/demand_raw.csv")

# Column synonyms for robust loading
_TIMESTAMP_COLS = {"timestamp", "date", "datetime", "time", "Date", "Timestamp"}
_DEMAND_COLS    = {"demand", "demand_mw", "load", "load_mw", "Load", "Demand_MW",
                   "DemandMW", "demand_MW"}
_FREQ_COLS      = {"frequency", "freq_hz", "frequency_hz", "Hz", "Freq"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to the canonical names used by the rest of the pipeline."""
    col_map: dict[str, str] = {}

    for col in df.columns:
        if col in _TIMESTAMP_COLS:
            col_map[col] = "timestamp"
        elif col in _DEMAND_COLS:
            col_map[col] = "demand_mw"
        elif col in _FREQ_COLS:
            col_map[col] = "frequency_hz"

    df = df.rename(columns=col_map)

    # Drop columns we don't need
    keep = [c for c in ("timestamp", "demand_mw", "frequency_hz") if c in df.columns]
    return df[keep]


def _synthesise_demand(start: str, end: str, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic hourly demand time-series that mimics Telangana's grid:

    * Base ~8000 MW with seasonal warmth peaks (Apr–Jun  +15 %)
    * Daily profile: low at night, peaks at 09:00 and 20:00
    * Day-of-week effect: weekends ~10 % lower
    * Gaussian noise ± 2 %
    * Frequency drawn from Normal(50.0, 0.03) Hz clamped to [49.7, 50.3]
    """
    log.warning(
        "grid_demand source file not found at '%s'. "
        "Generating SYNTHETIC demand data — replace with real POSOCO data "
        "for publication-quality results.",
        SOURCE_FILE,
    )
    rng = np.random.default_rng(seed)

    idx = pd.date_range(start=start, end=end, freq="h")
    n = len(idx)

    # ── Hourly shape ────────────────────────────────────────────────────────
    hour = idx.hour
    hourly_profile = (
        0.65
        + 0.20 * np.sin(np.pi * (hour - 4) / 12)       # broad daytime bump
        + 0.15 * np.exp(-0.5 * ((hour - 9) / 2) ** 2)  # morning peak
        + 0.20 * np.exp(-0.5 * ((hour - 20) / 2) ** 2) # evening peak
    )

    # ── Seasonal (month) factor ──────────────────────────────────────────────
    month = idx.month
    seasonal = 1.0 + 0.15 * np.where(month.isin([4, 5, 6]), 1, 0)  # summer peak

    # ── Day-of-week effect ───────────────────────────────────────────────────
    dow = idx.dayofweek  # 0=Mon … 6=Sun
    weekend = 1.0 - 0.10 * (dow >= 5).astype(float)

    # ── Combine ─────────────────────────────────────────────────────────────
    base = 8000  # MW
    noise = 1.0 + rng.normal(0, 0.02, n)
    demand = base * hourly_profile * seasonal * weekend * noise

    # ── Frequency ───────────────────────────────────────────────────────────
    freq = np.clip(rng.normal(50.0, 0.03, n), 49.7, 50.3)

    df = pd.DataFrame({
        "timestamp":    idx,
        "demand_mw":    demand.round(1),
        "frequency_hz": freq.round(3),
    })
    df.set_index("timestamp", inplace=True)
    return df


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class GridDemandLoader:
    """
    Load (or synthesise) hourly grid demand and save a normalised CSV.
    """

    def __init__(self) -> None:
        self.source_path = Path(SOURCE_FILE)
        self.output_path = Path(OUTPUT_FILE)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.start: str = get("data.start_date", "2021-01-01")
        self.end: str   = get("data.end_date",   "2023-12-31")

    # ------------------------------------------------------------------
    def load(self, force: bool = False) -> pd.DataFrame:
        """
        Load grid demand data from a CSV file or synthesise it.

        Parameters
        ----------
        force : Reprocess even if the output file already exists.

        Returns
        -------
        pd.DataFrame with a DatetimeIndex and columns: demand_mw, [frequency_hz]
        """
        if self.output_path.exists() and not force:
            log.info("Grid demand data already exists at %s — skipping.",
                     self.output_path)
            return pd.read_csv(self.output_path, index_col="timestamp",
                               parse_dates=True)

        # ── Try loading real data ────────────────────────────────────────────
        if self.source_path.exists():
            log.info("Loading grid demand from %s", self.source_path)
            df = pd.read_csv(self.source_path)
            df = _normalise_columns(df)

            if "timestamp" not in df.columns:
                raise ValueError(
                    f"Could not find a timestamp column in {self.source_path}. "
                    f"Rename one column to 'timestamp' or add it."
                )
            if "demand_mw" not in df.columns:
                raise ValueError(
                    f"Could not find a demand column in {self.source_path}. "
                    f"Rename one column to 'demand_mw'."
                )

            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)

            # Filter to configured date range
            df = df.sort_index()
            df = df.loc[self.start:self.end]
        else:
            # ── Synthesise ────────────────────────────────────────────────────
            df = _synthesise_demand(self.start, self.end)

        # Resample to hourly in case input has finer resolution
        df = df.resample("h").mean()

        df.to_csv(self.output_path)
        log.info(
            "Grid demand saved → %s   rows=%d  cols=%s",
            self.output_path, len(df), list(df.columns),
        )
        return df


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    loader = GridDemandLoader()
    df = loader.load()
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(df.describe())
