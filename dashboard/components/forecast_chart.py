"""
dashboard/components/forecast_chart.py
----------------------------------------
Reusable Plotly forecast chart with confidence band.

Usage
-----
    from dashboard.components.forecast_chart import forecast_chart
    fig = forecast_chart(df, actual_col="ghi", pred_col="ghi_pred",
                         title="Solar GHI Forecast")
    st.plotly_chart(fig, use_container_width=True)
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go


def forecast_chart(
    df: pd.DataFrame,
    actual_col: str,
    pred_col: str,
    title: str = "Forecast",
    lower_col: Optional[str] = None,
    upper_col: Optional[str] = None,
    x_label: str = "Time",
    y_label: str = "Value",
    n_points: int = 168,  # show last 7 days by default
) -> go.Figure:
    """
    Build a Plotly line chart comparing actuals vs predictions.

    Parameters
    ----------
    df         : DataFrame with DatetimeIndex (or any index).
    actual_col : Column name for actual values.
    pred_col   : Column name for predicted values.
    title      : Chart title.
    lower_col  : (optional) Column for lower confidence bound.
    upper_col  : (optional) Column for upper confidence bound.
    n_points   : Number of most-recent points to display.
    """
    display_df = df.tail(n_points)
    x = display_df.index

    fig = go.Figure()

    # ── Confidence band ────────────────────────────────────────────────────
    if lower_col and upper_col and lower_col in df.columns and upper_col in df.columns:
        fig.add_trace(go.Scatter(
            x=list(x) + list(x[::-1]),
            y=list(display_df[upper_col]) + list(display_df[lower_col][::-1]),
            fill="toself",
            fillcolor="rgba(59,130,246,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="95% Confidence",
            showlegend=True,
        ))

    # ── Actual ─────────────────────────────────────────────────────────────
    if actual_col in display_df.columns:
        fig.add_trace(go.Scatter(
            x=x, y=display_df[actual_col],
            mode="lines",
            name="Actual",
            line=dict(color="#38bdf8", width=2),
        ))

    # ── Predicted ──────────────────────────────────────────────────────────
    if pred_col in display_df.columns:
        fig.add_trace(go.Scatter(
            x=x, y=display_df[pred_col],
            mode="lines",
            name="Predicted",
            line=dict(color="#f97316", width=2, dash="dot"),
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#f1f5f9")),
        plot_bgcolor="#0f172a",
        paper_bgcolor="#1e293b",
        font=dict(color="#cbd5e1"),
        xaxis=dict(
            title=x_label,
            gridcolor="#1e293b",
            tickfont=dict(color="#94a3b8"),
        ),
        yaxis=dict(
            title=y_label,
            gridcolor="#1e293b",
            tickfont=dict(color="#94a3b8"),
        ),
        legend=dict(
            bgcolor="rgba(15,23,42,0.7)",
            bordercolor="#334155",
            borderwidth=1,
        ),
        hovermode="x unified",
        margin=dict(t=50, b=40, l=50, r=20),
    )
    return fig
