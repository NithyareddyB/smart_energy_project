"""
src/data_collection/weather_api.py
------------------------------------
Fetches historical hourly weather data from the Open-Meteo Archive API
(https://archive-api.open-meteo.com) — no API key required.

Variables collected
-------------------
  wind_speed_10m          m/s
  wind_direction_10m      degrees
  relative_humidity_2m    %
  cloud_cover             %
  precipitation           mm
  surface_pressure        hPa
  shortwave_radiation     W/m²  (cross-check with NASA GHI)

Output
------
data/raw/weather_raw.csv  — hourly rows, UTC timestamps

Usage (standalone)
------------------
    python -m src.data_collection.weather_api
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List

import pandas as pd
import requests

from src.utils.config import get
from src.utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARS = [
    "wind_speed_10m",
    "wind_direction_10m",
    "relative_humidity_2m",
    "cloud_cover",
    "precipitation",
    "surface_pressure",
    "shortwave_radiation",
]
OUTPUT_FILE: str = get("data.open_meteo.output_file", "data/raw/weather_raw.csv")


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class WeatherCollector:
    """
    Download hourly historical weather from Open-Meteo Archive API.

    No API key needed.  Works for any lat/lon and date range.
    """

    def __init__(self) -> None:
        self.lat: float = float(get("location.latitude", 17.385044))
        self.lon: float = float(get("location.longitude", 78.486671))
        self.start: str = get("data.start_date", "2021-01-01")
        self.end: str   = get("data.end_date",   "2023-12-31")
        self.output_path = Path(OUTPUT_FILE)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def _fetch(self, retries: int = 3, backoff: float = 5.0) -> pd.DataFrame:
        """Hit the Open-Meteo Archive endpoint, return a tidy DataFrame."""
        params = {
            "latitude":       self.lat,
            "longitude":      self.lon,
            "start_date":     self.start,
            "end_date":       self.end,
            "hourly":         ",".join(HOURLY_VARS),
            "timezone":       "UTC",
            "wind_speed_unit": "ms",
        }

        for attempt in range(1, retries + 1):
            try:
                log.debug("Open-Meteo request (attempt %d)…", attempt)
                resp = requests.get(ARCHIVE_URL, params=params, timeout=120)
                resp.raise_for_status()
                data_json = resp.json()
                break
            except (requests.RequestException, ValueError) as exc:
                log.warning("Attempt %d failed: %s", attempt, exc)
                if attempt == retries:
                    raise RuntimeError(
                        f"Open-Meteo API failed after {retries} attempts. "
                        "Check your internet connection."
                    ) from exc
                time.sleep(backoff * attempt)

        hourly = data_json.get("hourly", {})
        df = pd.DataFrame(hourly)
        df["timestamp"] = pd.to_datetime(df["time"])
        df.drop(columns=["time"], inplace=True)
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
        return df

    # ------------------------------------------------------------------
    def collect(self, force: bool = False) -> pd.DataFrame:
        """
        Fetch weather data for the full date range and save to CSV.

        Parameters
        ----------
        force : Re-download and overwrite even if the output file exists.

        Returns
        -------
        pd.DataFrame with a DatetimeIndex.
        """
        if self.output_path.exists() and not force:
            log.info("Weather data already exists at %s — skipping download.",
                     self.output_path)
            return pd.read_csv(self.output_path, index_col="timestamp",
                               parse_dates=True)

        log.info(
            "Fetching Open-Meteo weather data  %s → %s  lat=%.4f lon=%.4f",
            self.start, self.end, self.lat, self.lon,
        )
        df = self._fetch()

        df.to_csv(self.output_path)
        log.info(
            "Weather data saved → %s   rows=%d  cols=%s",
            self.output_path, len(df), list(df.columns),
        )
        return df


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    wc = WeatherCollector()
    df = wc.collect()
    print(df.head())
    print(f"\nShape: {df.shape}")
