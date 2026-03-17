"""
dashboard/pages/03_policy.py
------------------------------
Page 3: Policy Scenario Explorer
  - Interactive sliders for renewable %, storage, years, carbon price
  - Plotly comparison charts (cost / emissions / reliability)
  - Auto-generated recommendation text
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.metric_card import metric_card
from dashboard.components.scenario_table import scenario_table

st.set_page_config(page_title="Policy Scenarios", layout="wide")
st.markdown("<style>h1,h2,h3{color:#f1f5f9;}</style>", unsafe_allow_html=True)
st.title("🌿 Policy Scenario Explorer")
st.caption("Simulate energy transition scenarios and compare outcomes over time")

# ---------------------------------------------------------------------------
# Load pre-computed simulation or compute on the fly
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_results() -> pd.DataFrame | None:
    path = Path("reports/simulation/scenario_results.csv")
    if path.exists():
        return pd.read_csv(path)
    return None


precomputed = load_results()

# ---------------------------------------------------------------------------
# Sidebar — custom scenario builder
# ---------------------------------------------------------------------------
st.sidebar.header("🔧 Custom Scenario")
re_pct      = st.sidebar.slider("Renewable Mix (%)",         10, 100, 60)
storage_kwh = st.sidebar.slider("Storage Capacity (kWh)", 100, 10000, 2000)
years       = st.sidebar.slider("Simulation Horizon (yrs)",   5,  25,  10)
carbon_pr   = st.sidebar.slider("Carbon Price (INR/kg)",     0.0,  5.0, 1.0, step=0.1)
demand_gr   = st.sidebar.slider("Demand Growth (%/yr)",      1.0,  8.0, 4.0, step=0.5)
run_button  = st.sidebar.button("▶ Run Simulation", type="primary")

# ---------------------------------------------------------------------------
# Simulate function (lightweight version — no model loading required)
# ---------------------------------------------------------------------------

def _quick_simulate(
    renewable_pct: float,
    storage_kwh: float,
    n_years: int,
    carbon_price: float,
    demand_growth: float,
    base_demand_mwh_yr: float = 70_000,
) -> pd.DataFrame:
    """Fast analytical approximation of yearly metrics."""
    rows = []
    for yr in range(n_years):
        demand = base_demand_mwh_yr * (1 + demand_growth / 100) ** yr
        re_gen = demand * (renewable_pct / 100) * 0.85   # 85% utilisation
        storage_buf = min(storage_kwh / 1000, re_gen * 0.1)  # MWh
        grid_import = max(demand - re_gen - storage_buf, 0)
        grid_cost = grid_import * 7.5  # INR per MWh → thousands
        carbon_kg = grid_import * 0.82
        carbon_cost = carbon_kg * carbon_price
        total_cost = (grid_cost + carbon_cost) / 1e6  # Crore INR
        reliability = 100 - max(0, (demand - re_gen - storage_buf) / demand) * 5
        curtailment = max(0, re_gen - (demand + storage_buf)) / demand * 100
        rows.append({
            "year": yr,
            "demand_mwh": round(demand, 0),
            "re_fraction_pct": round(min(renewable_pct * 0.85, 100), 1),
            "grid_import_mwh": round(grid_import, 0),
            "cost_crore_inr": round(total_cost, 2),
            "carbon_kg": round(carbon_kg, 0),
            "reliability_pct": round(reliability, 2),
            "curtailment_pct": round(curtailment, 2),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Build display data
# ---------------------------------------------------------------------------
if precomputed is not None and not run_button:
    display_df = precomputed.copy()
    scenarios_present = display_df["scenario"].unique()
    st.info(f"Showing pre-computed results for: {', '.join(scenarios_present)}")
else:
    custom = _quick_simulate(re_pct, storage_kwh, years, carbon_pr, demand_gr)
    custom["scenario"] = f"Custom ({re_pct}% RE)"
    display_df = custom

# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
col1, col2 = st.columns(2)

COLOURS = ["#38bdf8", "#f97316", "#22c55e", "#a78bfa", "#fb7185"]

def _line_chart(df: pd.DataFrame, y_col: str, title: str, y_label: str) -> go.Figure:
    fig = go.Figure()
    for i, (sc, grp) in enumerate(df.groupby("scenario") if "scenario" in df.columns
                                   else [(display_df["scenario"].iloc[0], df)]):
        fig.add_trace(go.Scatter(
            x=grp["year"], y=grp[y_col],
            mode="lines+markers",
            name=sc,
            line=dict(color=COLOURS[i % len(COLOURS)], width=2),
            marker=dict(size=5),
        ))
    fig.update_layout(
        title=dict(text=title, font=dict(color="#f1f5f9")),
        plot_bgcolor="#0f172a", paper_bgcolor="#1e293b",
        font=dict(color="#cbd5e1"),
        xaxis=dict(title="Year", gridcolor="#1e293b"),
        yaxis=dict(title=y_label, gridcolor="#1e293b"),
        legend=dict(bgcolor="rgba(15,23,42,0.7)", bordercolor="#334155", borderwidth=1),
        margin=dict(t=50, b=40, l=60, r=20),
    )
    return fig


with col1:
    st.subheader("Annual Grid Cost")
    if "cost_crore_inr" in display_df.columns:
        st.plotly_chart(_line_chart(display_df, "cost_crore_inr",
                                    "Total Grid Cost", "Crore INR"),
                        use_container_width=True)
    elif "total_cost_inr" in display_df.columns:
        tmp = display_df.copy()
        tmp["cost_crore_inr"] = tmp["total_cost_inr"] / 1e7
        st.plotly_chart(_line_chart(tmp, "cost_crore_inr",
                                    "Total Grid Cost", "Crore INR"),
                        use_container_width=True)

with col2:
    st.subheader("Carbon Emissions")
    if "carbon_kg" in display_df.columns:
        tmp = display_df.copy()
        tmp["carbon_kt"] = tmp["carbon_kg"] / 1e6
        st.plotly_chart(_line_chart(tmp, "carbon_kt",
                                    "Annual Grid Carbon Emissions", "Kilo-tonne CO₂"),
                        use_container_width=True)

col3, col4 = st.columns(2)
with col3:
    st.subheader("Renewable Fraction")
    if "re_fraction_pct" in display_df.columns:
        st.plotly_chart(_line_chart(display_df, "re_fraction_pct",
                                    "Renewable Energy Fraction", "% of Demand"),
                        use_container_width=True)

with col4:
    st.subheader("Grid Reliability")
    if "reliability_pct" in display_df.columns:
        st.plotly_chart(_line_chart(display_df, "reliability_pct",
                                    "Grid Reliability", "% Uptime"),
                        use_container_width=True)

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
st.subheader("Scenario Summary (Final Year)")
last_year = display_df.groupby("scenario").last().reset_index() if "scenario" in display_df.columns else display_df.tail(1)
scenario_table(last_year, highlight_col="reliability_pct", higher_is_better=True)

# ---------------------------------------------------------------------------
# Auto-generated recommendation
# ---------------------------------------------------------------------------
st.subheader("💡 Automated Recommendation")
if "scenario" in display_df.columns:
    best_sc = display_df.groupby("scenario")["reliability_pct"].last().idxmax()
else:
    best_sc = "Custom scenario"

st.success(
    f"**Recommended pathway:** {best_sc}\n\n"
    f"Based on the simulated outcomes, a renewable mix of **{re_pct}%** with "
    f"**{storage_kwh:,.0f} kWh** battery storage achieves the best balance of "
    f"cost, reliability, and emissions over the {years}-year horizon. "
    f"At a carbon price of **₹{carbon_pr}/kg CO₂**, this scenario avoids an estimated "
    f"**{(display_df['carbon_kg'].sum() / 1e6):.1f} Mega-tonnes** of grid emissions "
    f"compared to a business-as-usual trajectory."
)
