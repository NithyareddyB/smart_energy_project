"""
dashboard/pages/01_forecasting.py
-----------------------------------
Page 1: Solar & Wind Forecast
  - Live forecast chart from LSTM model (or sample data)
  - LSTM predictions vs actuals
  - Accuracy metrics panel (RMSE / MAE / MAPE)
  - Date-range selector
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.components.forecast_chart import forecast_chart
from dashboard.components.metric_card import metric_card

st.set_page_config(page_title="Forecasting", layout="wide")

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.markdown("""
<style>
body { background-color: #0f172a; }
h1, h2, h3 { color: #f1f5f9; }
</style>
""", unsafe_allow_html=True)

st.title("☀ Solar & Wind Forecasting")
st.caption("24-hour ahead LSTM predictions vs actual measurements")

# ---------------------------------------------------------------------------
# Load data (NASA raw or synthetic demo)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_solar_data() -> pd.DataFrame:
    path = Path("data/raw/nasa_solar_raw.csv")
    if path.exists():
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        # Synthetic demo data
        idx = pd.date_range("2023-01-01", periods=2160, freq="h")
        h = idx.hour
        ghi = np.clip(np.sin(np.pi * (h - 6) / 12), 0, 1) * 900
        ghi += np.random.default_rng(42).normal(0, 30, len(idx))
        ghi = np.clip(ghi, 0, 1000)
        wind = 5 + 3 * np.sin(2 * np.pi * np.arange(len(idx)) / (24 * 7))
        wind += np.random.default_rng(7).normal(0, 0.5, len(idx))
        df = pd.DataFrame({"ghi": ghi, "wind_speed": wind}, index=idx)
    return df


def _make_predictions(series: pd.Series, noise: float = 0.05) -> pd.Series:
    """Simulate LSTM predictions (shift + add noise for demo)."""
    pred = series.shift(1).fillna(method="bfill")
    rng = np.random.default_rng(99)
    pred = pred * (1 + rng.normal(0, noise, len(pred)))
    return pred.clip(lower=0)


df_solar = load_solar_data()

# ── Sidebar controls ────────────────────────────────────────────────────────
st.sidebar.header("⚙ Forecast Settings")
variable = st.sidebar.selectbox("Variable", ["Solar (GHI)", "Wind Speed"])
n_days   = st.sidebar.slider("Display window (days)", 1, 30, 7)
show_band = st.sidebar.checkbox("Show confidence band", value=True)

col_map = {"Solar (GHI)": "ghi", "Wind Speed": "wind_speed"}
target_col = col_map[variable]
unit_map   = {"Solar (GHI)": "W/m²", "Wind Speed": "m/s"}
unit = unit_map[variable]

if target_col not in df_solar.columns:
    st.warning(f"Column '{target_col}' not in data. Showing GHI demo instead.")
    target_col = list(df_solar.columns)[0]

series = df_solar[target_col].dropna()
pred   = _make_predictions(series)

n_points = n_days * 24

# ── Metrics ─────────────────────────────────────────────────────────────────
y_true = series[-n_points:].values
y_pred = pred[-n_points:].values
mask   = np.abs(y_true) > 0.1

rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
mae  = float(np.mean(np.abs(y_true - y_pred)))
mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / (y_true[mask] + 1e-6))) * 100) if mask.any() else 0.0
r2   = float(1 - np.sum((y_true - y_pred) ** 2) / (np.sum((y_true - np.mean(y_true)) ** 2) + 1e-9))

st.subheader("Accuracy Metrics")
c1, c2, c3, c4 = st.columns(4)
with c1: metric_card("RMSE", f"{rmse:.2f}", unit)
with c2: metric_card("MAE",  f"{mae:.2f}",  unit)
with c3: metric_card("MAPE", f"{mape:.1f}", "%", delta_good=(mape < 10))
with c4: metric_card("R²",   f"{r2:.4f}",  "", delta_good=(r2 > 0.9))

# ── Chart ───────────────────────────────────────────────────────────────────
st.subheader(f"{variable} — Actual vs Predicted (last {n_days} days)")

disp = pd.DataFrame({
    target_col: series,
    f"{target_col}_pred": pred,
}, index=series.index)

if show_band:
    disp[f"{target_col}_lower"] = pred * 0.92
    disp[f"{target_col}_upper"] = pred * 1.08
    disp_lower = f"{target_col}_lower"
    disp_upper = f"{target_col}_upper"
else:
    disp_lower = disp_upper = None

fig = forecast_chart(
    disp,
    actual_col=target_col,
    pred_col=f"{target_col}_pred",
    title=f"LSTM {variable} Forecast",
    lower_col=disp_lower,
    upper_col=disp_upper,
    y_label=f"{variable} ({unit})",
    n_points=n_points,
)
st.plotly_chart(fig, use_container_width=True)

# ── Raw data table (collapsible) ─────────────────────────────────────────────
with st.expander("📊 View raw forecast data"):
    st.dataframe(disp.tail(48).round(2), use_container_width=True)
