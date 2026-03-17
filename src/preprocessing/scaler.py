"""
src/preprocessing/scaler.py
-----------------------------
Fits MinMax or Standard scaler on the training set only and persists it
so that the same scaler can be reloaded at inference time.

The scaler is fitted ONLY on numeric feature columns (boolean/int flags
like is_weekend are excluded from scaling).

Usage (standalone)
------------------
    python -m src.preprocessing.scaler
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.utils.config import get
from src.utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCALER_TYPE: str  = get("preprocessing.scaler_type", "minmax")
SCALER_PATH: str  = get("preprocessing.scaler_save_path", "models/saved/scaler.pkl")
FEATURES_FILE     = get("preprocessing.processed_file", "data/processed/features.csv")
SCALED_FILE       = "data/processed/scaled.csv"

# Columns to exclude from scaling (binary flags, integer encodings)
_SKIP_COLS = {"is_weekend", "hour"}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DataScaler:
    """
    Fit-and-transform (or load-and-transform) a feature scaler.

    Parameters
    ----------
    scaler_type : ``"minmax"`` or ``"standard"``
    scaler_path : Where to save / load the fitted scaler.
    """

    def __init__(
        self,
        scaler_type: str = SCALER_TYPE,
        scaler_path: str = SCALER_PATH,
    ) -> None:
        self.scaler_type = scaler_type.lower()
        self.scaler_path = Path(scaler_path)
        self.scaler_path.parent.mkdir(parents=True, exist_ok=True)
        self._scaler: Optional[MinMaxScaler | StandardScaler] = None
        self._cols_to_scale: List[str] = []

    # ------------------------------------------------------------------
    def _build_scaler(self) -> MinMaxScaler | StandardScaler:
        if self.scaler_type == "minmax":
            return MinMaxScaler(feature_range=(0, 1))
        elif self.scaler_type == "standard":
            return StandardScaler()
        else:
            raise ValueError(f"Unknown scaler_type: {self.scaler_type!r}. "
                             "Choose 'minmax' or 'standard'.")

    # ------------------------------------------------------------------
    def fit_transform(self, df: pd.DataFrame,
                      train_end_idx: Optional[int] = None) -> pd.DataFrame:
        """
        Fit the scaler on the first ``train_end_idx`` rows (or the whole df
        if None), then scale all rows.

        Parameters
        ----------
        df            : Full feature DataFrame with DatetimeIndex.
        train_end_idx : Row index boundary — scaler fitted only up to here.
                        Prevents data leakage from val/test into the scaler.

        Returns
        -------
        pd.DataFrame — same shape, numeric columns scaled.
        """
        log.info("=== DataScaler: fitting %s scaler ===", self.scaler_type.upper())

        self._cols_to_scale = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in _SKIP_COLS
        ]
        log.debug("Columns to scale (%d): %s", len(self._cols_to_scale),
                  self._cols_to_scale[:10])

        self._scaler = self._build_scaler()
        fit_slice = df[self._cols_to_scale].iloc[:train_end_idx]
        self._scaler.fit(fit_slice)

        # Save fitted scaler
        joblib.dump((self._scaler, self._cols_to_scale), self.scaler_path)
        log.info("Scaler saved → %s", self.scaler_path)

        return self._apply(df)

    # ------------------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform using a previously fitted (or loaded) scaler."""
        if self._scaler is None:
            self.load()
        return self._apply(df)

    # ------------------------------------------------------------------
    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Invert scaling for a DataFrame."""
        if self._scaler is None:
            self.load()
        out = df.copy()
        scaled_present = [c for c in self._cols_to_scale if c in out.columns]
        out[scaled_present] = self._scaler.inverse_transform(
            out[scaled_present].values.reshape(-1, len(self._cols_to_scale))
        )[:, :len(scaled_present)]
        return out

    # ------------------------------------------------------------------
    def _apply(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        present = [c for c in self._cols_to_scale if c in out.columns]
        out[present] = self._scaler.transform(out[present])
        return out

    # ------------------------------------------------------------------
    def load(self) -> None:
        """Load a previously saved scaler from disk."""
        if not self.scaler_path.exists():
            raise FileNotFoundError(
                f"Scaler not found at {self.scaler_path}. "
                "Run the preprocessing pipeline first."
            )
        self._scaler, self._cols_to_scale = joblib.load(self.scaler_path)
        log.info("Scaler loaded from %s", self.scaler_path)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = pd.read_csv(FEATURES_FILE, index_col="timestamp", parse_dates=True)
    scaler = DataScaler()
    df_scaled = scaler.fit_transform(df)
    print(df_scaled.describe().round(3))
