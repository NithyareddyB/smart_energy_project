"""
src/data_collection/nasa_api.py
--------------------------------
Fetches hourly solar irradiance (GHI, DNI, DHI), temperature, wind-speed,
humidity, and clear-sky GHI from the NASA POWER API for the lat/lon and
date-range specified in config.yaml.

NASA POWER limits each request to ~366 days, so the collector automatically
chunks the date range into segments of `chunk_months` months each.

Output
------
data/raw/nasa_solar_raw.csv  — hourly rows, UTC timestamps, SI units

Usage (standalone)
------------------
    python -m src.data_collection.nasa_api
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import requests

from src.utils.config import cfg, get
from src.utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_URL: str = get("data.nasa_power.base_url",
                    "https://power.larc.nasa.gov/api/temporal/hourly/point")
PARAMETERS: List[str] = get("data.nasa_power.parameters", [
    "ALLSKY_SFC_SW_DWN",
    "ALLSKY_SFC_SW_DNI",
    "ALLSKY_SFC_SW_DIFF",
    "T2M",
    "WS10M",
    "WD10M",
    "RH2M",
    "CLRSKY_SFC_SW_DWN",
])
COMMUNITY: str = get("data.nasa_power.community", "RE")
CHUNK_MONTHS: int = int(get("data.nasa_power.chunk_months", 6))
OUTPUT_FILE: str = get("data.nasa_power.output_file", "data/raw/nasa_solar_raw.csv")

# Friendly column name mapping
_COL_MAP = {
    "ALLSKY_SFC_SW_DWN":  "ghi",
    "ALLSKY_SFC_SW_DNI":  "dni",
    "ALLSKY_SFC_SW_DIFF": "dhi",
    "T2M":                "temperature",
    "WS10M":              "wind_speed",
    "WD10M":              "wind_direction",
    "RH2M":               "humidity",
    "CLRSKY_SFC_SW_DWN":  "clearsky_ghi",
}

# Sentinel value NASA uses for missing data
_FILL_VALUE = -999.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _date_chunks(start: str, end: str, months: int) -> List[Tuple[str, str]]:
    """Split [start, end] into chunks of at most `months` calendar months."""
    from dateutil.relativedelta import relativedelta  # lightweight dep via python-dateutil

    fmt = "%Y-%m-%d"
    s = datetime.strptime(start, fmt)
    e = datetime.strptime(end, fmt)
    chunks: List[Tuple[str, str]] = []
    cur = s
    while cur <= e:
        chunk_end = min(cur + relativedelta(months=months) - timedelta(days=1), e)
        chunks.append((cur.strftime(fmt), chunk_end.strftime(fmt)))
        cur = chunk_end + timedelta(days=1)
    return chunks


def _fetch_chunk(lat: float, lon: float, start: str, end: str,
                 retries: int = 3, backoff: float = 5.0) -> pd.DataFrame:
    """Call NASA POWER API for a single date chunk, return tidy DataFrame."""
    params = {
        "parameters": ",".join(PARAMETERS),
        "community":  COMMUNITY,
        "longitude":  lon,
        "latitude":   lat,
        "start":      start.replace("-", ""),
        "end":        end.replace("-", ""),
        "format":     "JSON",
    }

    for attempt in range(1, retries + 1):
        try:
            log.debug("NASA POWER request: %s → %s (attempt %d)", start, end, attempt)
            resp = requests.get(BASE_URL, params=params, timeout=120)
            resp.raise_for_status()
            data_json = resp.json()
            break
        except (requests.RequestException, ValueError) as exc:
            log.warning("Attempt %d failed: %s", attempt, exc)
            if attempt == retries:
                raise RuntimeError(
                    f"NASA POWER API failed after {retries} attempts for {start}–{end}. "
                    "Check your internet connection or NASA service status."
                ) from exc
            time.sleep(backoff * attempt)

    # Parse the nested JSON structure
    hourly_data = data_json["properties"]["parameter"]
    rows = {}
    for var, hour_dict in hourly_data.items():
        for yyyymmddhh, value in hour_dict.items():
            # NASA key format: YYYYMMDDHHH (last 3 digits are hour in local solar)
            dt_str = yyyymmddhh[:8] + " " + yyyymmddhh[8:10] + ":00:00"
            if dt_str not in rows:
                rows[dt_str] = {}
            rows[dt_str][var] = float(value)

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index = pd.to_datetime(df.index, format="%Y%m%d %H:%M:%S")
    df.index.name = "timestamp"
    df.sort_index(inplace=True)

    # Replace fill values with NaN
    df.replace(_FILL_VALUE, float("nan"), inplace=True)
    return df


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class NASASolarCollector:
    """
    Download hourly solar & meteorological data from NASA POWER API.

    Parameters are read from ``config.yaml`` at instantiation.
    """

    def __init__(self) -> None:
        self.lat: float = float(get("location.latitude", 17.385044))
        self.lon: float = float(get("location.longitude", 78.486671))
        self.start: str  = get("data.start_date", "2021-01-01")
        self.end:   str  = get("data.end_date",   "2023-12-31")
        self.output_path = Path(OUTPUT_FILE)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def collect(self, force: bool = False) -> pd.DataFrame:
        """
        Fetch data for the full date range and save to CSV.

        Parameters
        ----------
        force : Re-download and overwrite even if the output file already exists.

        Returns
        -------
        pd.DataFrame with a DatetimeIndex and renamed columns.
        """
        if self.output_path.exists() and not force:
            log.info("NASA solar data already exists at %s — skipping download.",
                     self.output_path)
            return pd.read_csv(self.output_path, index_col="timestamp",
                               parse_dates=True)

        chunks = _date_chunks(self.start, self.end, CHUNK_MONTHS)
        log.info(
            "Fetching NASA POWER data for %s  lat=%.4f lon=%.4f  chunks=%d",
            get("location.name", "Hyderabad"),
            self.lat, self.lon, len(chunks),
        )

        dfs: List[pd.DataFrame] = []
        for i, (s, e) in enumerate(chunks, 1):
            log.info("  Chunk %d/%d  %s → %s", i, len(chunks), s, e)
            df_chunk = _fetch_chunk(self.lat, self.lon, s, e)
            dfs.append(df_chunk)
            time.sleep(1)  # polite delay between requests

        df = pd.concat(dfs)
        df = df[~df.index.duplicated(keep="first")]  # drop overlap if any
        df.sort_index(inplace=True)

        # Rename columns to friendly names
        df.rename(columns={k: v for k, v in _COL_MAP.items() if k in df.columns},
                  inplace=True)

        df.to_csv(self.output_path)
        log.info(
            "NASA solar data saved → %s   rows=%d  cols=%s",
            self.output_path, len(df), list(df.columns),
        )
        return df


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    collector = NASASolarCollector()
    result = collector.collect()
    print(result.head())
    print(f"\nShape: {result.shape}")
