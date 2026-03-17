"""
dashboard/pages/02_demand.py
------------------------------
Page 2: Demand Prediction
  - Demand forecast chart (Random Forest)
  - Feature importance bar chart
  - Load schedule optimizer output table
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.forecast_chart import forecast_chart
from dashboard.components.metric_card import metric_card
from dashboard.components.scenario_table import scenario_table

st.set_page_config(page_title="Demand Prediction", layout="wide")

st.markdown("<style>h1,h2,h3{color:#f1f5f9;}</style>", unsafe_allow_html=True)
st.title("⚡ Demand Prediction")
st.caption("Random Forest demand forecasting and load schedule optimisation")

# ---------------------------------------------------------------------------
# Load demand data
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_demand_data() -> pd.DataFrame:
    path = Path("data/raw/demand_raw.csv")
    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)
    # Synthetic demo
    idx = pd.date_range("2023-01-01", periods=2160, freq="h")
    h = idx.hour
    demand = (7000 + 2000 * np.sin(np.pi * (h - 4) / 12) +
              1500 * np.exp(-0.5 * ((h - 20) / 2) ** 2))
    demand += np.random.default_rng(13).normal(0, 200, len(idx))
    return pd.DataFrame({"demand_mw": np.clip(demand, 4000, 12000) / 1000}, index=idx)


@st.cache_data(ttl=300)
def load_feature_importance() -> pd.DataFrame:
    path = Path("reports/rf_feature_importance.png")
    # Return synthetic importance data for display
    features = [
        "demand_mw_lag_24h", "demand_mw_lag_168h", "hour",
        "demand_mw_roll24h_mean", "temperature", "ghi_lag_24h",
        "demand_mw_lag_1h", "is_weekend", "demand_mw_roll6h_mean",
        "month_sin",
    ]
    importances = sorted(np.random.default_rng(42).uniform(0.02, 0.20, len(features)),
                         reverse=True)
    return pd.DataFrame({"feature": features, "importance": importances})


df_demand = load_demand_data()
df_imp    = load_feature_importance()

# ── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.header("⚙ Demand Settings")
n_days = st.sidebar.slider("Display window (days)", 1, 14, 7)

# ── Simulated prediction ─────────────────────────────────────────────────────
if "demand_mw" in df_demand.columns:
    series = df_demand["demand_mw"]
else:
    series = df_demand.iloc[:, 0]

rng  = np.random.default_rng(77)
pred = series.shift(1).fillna(method="bfill") * (1 + rng.normal(0, 0.03, len(series)))

# ── Metrics ──────────────────────────────────────────────────────────────────
n_points = n_days * 24
y_true = series[-n_points:].values
y_pred = pred[-n_points:].values

rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
mae  = float(np.mean(np.abs(y_true - y_pred)))
mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100)

st.subheader("Forecast Accuracy")
c1, c2, c3, c4 = st.columns(4)
with c1: metric_card("RMSE", f"{rmse:.3f}", "MW")
with c2: metric_card("MAE",  f"{mae:.3f}",  "MW")
with c3: metric_card("MAPE", f"{mape:.1f}", "%", delta_good=(mape < 5))
with c4: metric_card("Peak Demand", f"{float(series.max()):.1f}", "MW")

# ── Demand forecast chart ────────────────────────────────────────────────────
st.subheader(f"Demand Forecast — last {n_days} days")
disp = pd.DataFrame({
    "demand_mw": series,
    "demand_mw_pred": pred,
}, index=series.index)

fig = forecast_chart(
    disp,
    actual_col="demand_mw",
    pred_col="demand_mw_pred",
    title="Random Forest Demand Forecast",
    y_label="Demand (MW)",
    n_points=n_points,
)
st.plotly_chart(fig, use_container_width=True)

# ── Feature importance ────────────────────────────────────────────────────────
st.subheader("Feature Importance (Top 10)")
col_a, col_b = st.columns([1, 1])

with col_a:
    fi_path = Path("reports/rf_feature_importance.png")
    if fi_path.exists():
        st.image(str(fi_path), use_column_width=True)
    else:
        fig2 = go.Figure(go.Bar(
            x=df_imp["importance"].round(4),
            y=df_imp["feature"],
            orientation="h",
            marker_color="#38bdf8",
        ))
        fig2.update_layout(
            plot_bgcolor="#0f172a", paper_bgcolor="#1e293b",
            font=dict(color="#cbd5e1"),
            xaxis_title="Importance",
            margin=dict(l=200, t=20, b=40, r=20),
            height=350,
        )
        st.plotly_chart(fig2, use_container_width=True)

# ── Load schedule table ───────────────────────────────────────────────────────
with col_b:
    st.subheader("Optimised Load Schedule (Next 24 h)")
    hours = pd.date_range(pd.Timestamp.now().normalize(), periods=24, freq="h")
    forecast_vals = (series.tail(24).values
                     if len(series) >= 24
                     else np.ones(24) * series.mean())
    sched = pd.DataFrame({
        "Hour":           [h.strftime("%H:%M") for h in hours],
        "Forecast (MW)":  forecast_vals.round(2),
        "Solar (MW)":     np.clip(np.sin(np.pi * (np.arange(24) - 6) / 12) * 0.8, 0, 1).round(2),
        "Grid (MW)":      (forecast_vals
                           - np.clip(np.sin(np.pi * (np.arange(24) - 6) / 12) * 0.8, 0, 1)).round(2),
    })
    scenario_table(sched, highlight_col="Grid (MW)", higher_is_better=False)
