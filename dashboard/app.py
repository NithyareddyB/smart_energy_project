"""
dashboard/app.py
-----------------
Streamlit entry point for the Smart Energy Optimization dashboard.

Navigation
----------
  ☀ Forecasting   — Solar & wind LSTM predictions
  ⚡ Demand        — Random Forest demand forecasting
  🌿 Policy         — Scenario comparison & recommendation
  💰 Economics      — LCOE / NPV / IRR / sensitivity

Run
---
    # From the project root:
    streamlit run dashboard/app.py
    # or via main.py:
    python main.py --dashboard
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Ensure project root is importable when `streamlit run dashboard/app.py` is used
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Energy Optimization",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global dark-mode style override
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ───── Base colours ───── */
:root {
    --bg-primary:   #0f172a;
    --bg-secondary: #1e293b;
    --border:       #334155;
    --text-primary: #f1f5f9;
    --text-muted:   #94a3b8;
    --accent:       #38bdf8;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}
[data-testid="stSidebar"] {
    background-color: var(--bg-secondary) !important;
    border-right: 1px solid var(--border);
}
h1, h2, h3, h4 { color: var(--text-primary) !important; }
p, li, label, span, div { color: var(--text-primary) !important; }
.stCaption { color: var(--text-muted) !important; }
[data-testid="stMetricValue"] { color: var(--accent) !important; }

/* ── sidebar nav links ── */
.css-1d391kg a { color: var(--text-primary) !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar — branding + navigation hint
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 16px 0 8px;">
        <div style="font-size:2rem;">⚡</div>
        <div style="font-size:1.1rem;font-weight:700;color:#f1f5f9;">Smart Energy</div>
        <div style="font-size:0.75rem;color:#64748b;">Optimization System</div>
    </div>
    <hr style="border-color:#334155;margin:8px 0 16px;">
    """, unsafe_allow_html=True)

    region = "Hyderabad, Telangana"
    st.caption(f"📍 Region: **{region}**")
    st.caption("Use the **Pages** menu above to navigate.")

    st.markdown("---")
    st.caption("**Pipeline Stages**")
    stages = {
        "data/raw/nasa_solar_raw.csv":   "✅ Solar data",
        "data/raw/weather_raw.csv":      "✅ Weather data",
        "data/raw/demand_raw.csv":       "✅ Demand data",
        "data/processed/features.csv":  "✅ Features",
        "models/saved/lstm_solar.keras": "✅ LSTM Solar",
        "models/saved/rf_demand.pkl":    "✅ RF Demand",
        "models/saved/rl_optimizer.zip": "✅ RL Optimizer",
    }
    for path, label in stages.items():
        exists = Path(path).exists()
        icon = "✅" if exists else "⏳"
        st.markdown(
            f'<div style="font-size:0.75rem;color:{"#22c55e" if exists else "#64748b"};">'
            f'{icon} {label.split(" ", 1)[1]}</div>',
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Home page content
# ---------------------------------------------------------------------------
st.title("⚡ Smart Energy Optimization System")
st.caption("Telangana Grid • AI-Powered Forecasting & Policy Analysis")

st.markdown("""
> **Navigate using the sidebar** to explore the four analysis pages.
""")

col1, col2, col3, col4 = st.columns(4)

cards = [
    ("☀", "Forecasting",  "01_forecasting",
     "24-hour solar & wind forecasts using LSTM neural networks"),
    ("⚡", "Demand",       "02_demand",
     "Random Forest demand prediction with feature analysis"),
    ("🌿", "Policy",        "03_policy",
     "Interactive scenario explorer with cost & emissions projections"),
    ("💰", "Economics",    "04_economics",
     "LCOE, NPV, IRR and sensitivity analysis for investment decisions"),
]

for col, (icon, title, page, desc) in zip([col1, col2, col3, col4], cards):
    with col:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border: 1px solid #334155;
            border-radius: 14px;
            padding: 20px 16px;
            text-align: center;
            cursor: pointer;
            transition: border-color 0.2s;
        ">
            <div style="font-size: 2.2rem; margin-bottom: 8px;">{icon}</div>
            <div style="font-size: 1rem; font-weight: 700; color: #f1f5f9;
                        margin-bottom: 6px;">{title}</div>
            <div style="font-size: 0.78rem; color: #94a3b8; line-height: 1.4;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
### Quick Start

1. **Collect data**: `python main.py --collect`
2. **Preprocess**: `python main.py --preprocess`
3. **Train models**: `python main.py --train && python main.py --rl`
4. **Simulate**: `python main.py --simulate`
5. **Dashboard**: already running! Navigate via the sidebar.

*All pages work with demo data even before the pipeline has been run.*
""")
