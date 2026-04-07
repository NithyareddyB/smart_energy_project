"""
dashboard/pages/04_economics.py
---------------------------------
Page 4: Economic Analysis
  - LCOE / NPV results table
  - Payback period bar chart
  - Sensitivity tornado diagram
  - PDF export button
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.metric_card import metric_card
from dashboard.components.scenario_table import scenario_table

st.set_page_config(page_title="Economics", layout="wide")
st.markdown("<style>h1,h2,h3{color:#f1f5f9;}</style>", unsafe_allow_html=True)
st.title("💰 Economic Analysis")
st.caption("LCOE, NPV, IRR, payback period, and sensitivity analysis")

# ---------------------------------------------------------------------------
# Load economics data (or synthetic demo)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_economics() -> pd.DataFrame:
    path = Path("reports/simulation/economics_summary.csv")
    if path.exists():
        return pd.read_csv(path)
    # Synthetic demo
    return pd.DataFrame({
        "scenario":          ["Baseline", "Moderate transition", "Aggressive RE"],
        "total_capex_cr":    [320.5, 780.2, 1450.8],
        "annual_savings_cr": [12.4,  48.7,  95.3],
        "npv_cr":            [-42.1, 118.3, 287.6],
        "irr_pct":           [5.2,   11.8,  16.4],
        "payback_yr":        [None,  9,     8],
        "lcoe_inr_kwh":      [6.85,  4.20,  3.15],
    })


@st.cache_data(ttl=300)
def load_sensitivity() -> pd.DataFrame:
    path = Path("reports/simulation/sensitivity_table.csv")
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame({
        "parameter": ["Solar CAPEX", "Solar CAPEX", "Wind CAPEX", "Wind CAPEX",
                      "Battery CAPEX", "Battery CAPEX", "Discount Rate", "Discount Rate"],
        "change":    ["+20%", "-20%", "+20%", "-20%",
                      "+20%", "-20%", "+20%", "-20%"],
        "delta_npv_cr": [-28.4, 28.4, -15.2, 15.2, -8.9, 8.9, -22.1, 22.1],
    })


df_econ = load_economics()
df_sens = load_sensitivity()

# ---------------------------------------------------------------------------
# Key metrics
# ---------------------------------------------------------------------------
st.subheader("Summary Metrics")

if not df_econ.empty:
    best_idx = df_econ["npv_cr"].astype(float).idxmax() if "npv_cr" in df_econ.columns else 0
    best_row = df_econ.iloc[best_idx]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        lcoe_val = float(best_row.get("lcoe_inr_kwh", 0))
        metric_card("Best LCOE", f"{lcoe_val:.2f}", "₹/kWh", delta_good=(lcoe_val < 4.5))
    with c2:
        npv_val  = float(best_row.get("npv_cr", 0))
        metric_card("Best NPV", f"{npv_val:.1f}", "Cr INR", delta_good=(npv_val > 0))
    with c3:
        irr_val  = best_row.get("irr_pct", None)
        irr_str  = f"{float(irr_val):.1f}" if irr_val is not None else "N/A"
        metric_card("Best IRR", irr_str, "%", delta_good=(float(irr_val or 0) > 8))
    with c4:
        pbp_val = best_row.get("payback_yr", None)
        pbp_str = f"{int(pbp_val)}" if pbp_val is not None else "N/A"
        metric_card("Payback", pbp_str, "years", delta_good=(float(pbp_val or 99) < 12))

# ---------------------------------------------------------------------------
# Economics table
# ---------------------------------------------------------------------------
st.subheader("Full Economic Comparison")
display_cols = {
    "scenario":          "Scenario",
    "total_capex_cr":    "CAPEX (Cr ₹)",
    "annual_savings_cr": "Annual Savings (Cr ₹)",
    "npv_cr":            "NPV (Cr ₹)",
    "irr_pct":           "IRR (%)",
    "payback_yr":        "Payback (yrs)",
    "lcoe_inr_kwh":      "LCOE (₹/kWh)",
}
renamed_df = df_econ.rename(columns=display_cols)
show_df = renamed_df[[c for c in display_cols.values() if c in renamed_df.columns]]
scenario_table(show_df, highlight_col="NPV (Cr ₹)", higher_is_better=True)

# ---------------------------------------------------------------------------
# Payback period chart
# ---------------------------------------------------------------------------
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Payback Period by Scenario")
    pbp_path = Path("reports/simulation/payback_period.png")
    if pbp_path.exists():
        st.image(str(pbp_path), use_column_width=True)
    else:
        valid = df_econ.dropna(subset=["payback_yr"]) if "payback_yr" in df_econ.columns else pd.DataFrame()
        if not valid.empty:
            fig_pbp = go.Figure(go.Bar(
                x=valid["payback_yr"].astype(float),
                y=valid["scenario"],
                orientation="h",
                marker_color=["#22c55e", "#f97316", "#38bdf8"][:len(valid)],
                text=valid["payback_yr"].astype(str) + " yrs",
                textposition="outside",
            ))
            fig_pbp.update_layout(
                plot_bgcolor="#0f172a", paper_bgcolor="#1e293b",
                font=dict(color="#cbd5e1"),
                xaxis=dict(title="Years", gridcolor="#1e293b"),
                margin=dict(t=20, b=40, l=160, r=40),
                height=280,
            )
            st.plotly_chart(fig_pbp, use_container_width=True)
        else:
            st.info("Payback period data not available.")

# ---------------------------------------------------------------------------
# Tornado diagram
# ---------------------------------------------------------------------------
with col_b:
    st.subheader("NPV Sensitivity — Tornado Diagram")
    tornado_path = Path("reports/simulation/sensitivity_tornado.png")
    if tornado_path.exists():
        st.image(str(tornado_path), use_column_width=True)
    else:
        if not df_sens.empty:
            params       = df_sens["parameter"].unique()
            pos_deltas   = []
            neg_deltas   = []
            for p in params:
                rows = df_sens[df_sens["parameter"] == p]
                pos = rows[rows["change"] == "+20%"]["delta_npv_cr"].values
                neg = rows[rows["change"] == "−20%"]["delta_npv_cr"].values
                pos_deltas.append(float(pos[0]) if len(pos) else 0)
                neg_deltas.append(float(neg[0]) if len(neg) else 0)

            y = list(range(len(params)))
            fig_t = go.Figure()
            fig_t.add_trace(go.Bar(y=y, x=pos_deltas, orientation="h",
                                   name="+20%", marker_color="#38bdf8", width=0.35))
            fig_t.add_trace(go.Bar(y=y, x=neg_deltas, orientation="h",
                                   name="−20%", marker_color="#f97316", width=0.35,
                                   base=0))
            fig_t.update_layout(
                plot_bgcolor="#0f172a", paper_bgcolor="#1e293b",
                font=dict(color="#cbd5e1"),
                yaxis=dict(tickmode="array", tickvals=y, ticktext=list(params)),
                xaxis=dict(title="ΔNPV (Crore ₹)", gridcolor="#1e293b"),
                barmode="overlay",
                margin=dict(t=20, b=40, l=160, r=20),
                height=280,
                legend=dict(bgcolor="rgba(15,23,42,0.7)"),
            )
            fig_t.add_vline(x=0, line_width=1, line_color="white", opacity=0.5)
            st.plotly_chart(fig_t, use_container_width=True)

# ---------------------------------------------------------------------------
# PDF export
# ---------------------------------------------------------------------------
st.subheader("Export Report")

def _generate_pdf_bytes(df: pd.DataFrame) -> bytes:
    """Generate a simple PDF summary using fpdf2."""
    try:
        from fpdf import FPDF

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 12, "Smart Energy Optimization — Economic Analysis Report", ln=True)
        pdf.set_font("Helvetica", size=10)
        pdf.cell(0, 8, f"Generated by Smart Energy System Dashboard", ln=True)
        pdf.ln(5)

        # Table header
        cols = list(df.columns)
        pdf.set_font("Helvetica", "B", 9)
        for col in cols:
            pdf.cell(38, 7, str(col)[:18], border=1)
        pdf.ln()

        # Table rows
        pdf.set_font("Helvetica", size=9)
        for _, row in df.iterrows():
            for col in cols:
                val = row[col]
                val_str = f"{float(val):.2f}" if isinstance(val, float) else str(val)
                pdf.cell(38, 7, val_str[:18], border=1)
            pdf.ln()

        return pdf.output()
    except ImportError:
        return b""

if st.button("📄 Export PDF Report"):
    pdf_bytes = _generate_pdf_bytes(df_econ)
    if pdf_bytes:
        st.download_button(
            label="⬇ Download Economics Report (PDF)",
            data=pdf_bytes,
            file_name="economics_report.pdf",
            mime="application/pdf",
        )
    else:
        st.warning("fpdf2 not installed. Run `pip install fpdf2` to enable PDF export.")
