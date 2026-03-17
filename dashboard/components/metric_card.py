"""
dashboard/components/metric_card.py
-------------------------------------
Reusable metric card widget for Streamlit.

Usage
-----
    from dashboard.components.metric_card import metric_card
    metric_card("RMSE", "12.4", "kWh", delta="-3.2%", delta_good=True)
"""

from __future__ import annotations

import streamlit as st


def metric_card(
    label: str,
    value: str | float,
    unit: str = "",
    delta: str | None = None,
    delta_good: bool = True,
    help_text: str | None = None,
) -> None:
    """
    Render a styled metric card with optional delta badge.

    Parameters
    ----------
    label      : Card title (e.g. "RMSE")
    value      : Main metric value (rendered large)
    unit       : Unit suffix (e.g. "kWh", "%", "INR/kWh")
    delta      : Change string, e.g. "−3.2%". If None, badge is hidden.
    delta_good : True → green badge, False → red badge.
    help_text  : Optional tooltip text.
    """
    if isinstance(value, float):
        value = f"{value:,.2f}"

    delta_color = "#22c55e" if delta_good else "#ef4444"
    delta_html  = (
        f'<span style="font-size:0.8rem;color:{delta_color};font-weight:600;">'
        f'{delta}</span>'
        if delta is not None else ""
    )
    help_html = (
        f'<span title="{help_text}" style="cursor:help;opacity:0.6;font-size:0.75rem;">ℹ</span>'
        if help_text else ""
    )

    card_html = f"""
    <div style="
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    ">
        <div style="color:#94a3b8;font-size:0.78rem;font-weight:500;
                    text-transform:uppercase;letter-spacing:0.05em;margin-bottom:6px;">
            {label} {help_html}
        </div>
        <div style="display:flex;align-items:baseline;gap:6px;">
            <span style="color:#f1f5f9;font-size:1.8rem;font-weight:700;line-height:1;">
                {value}
            </span>
            <span style="color:#64748b;font-size:0.9rem;">{unit}</span>
        </div>
        {f'<div style="margin-top:4px;">{delta_html}</div>' if delta else ''}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)
