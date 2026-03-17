"""
dashboard/components/scenario_table.py
----------------------------------------
Reusable styled scenario comparison table.

Usage
-----
    from dashboard.components.scenario_table import scenario_table
    scenario_table(df, highlight_col="npv_cr", higher_is_better=True)
"""

from __future__ import annotations

import pandas as pd
import streamlit as st


def scenario_table(
    df: pd.DataFrame,
    highlight_col: str | None = None,
    higher_is_better: bool = True,
    caption: str | None = None,
    num_format: str = "{:.2f}",
) -> None:
    """
    Render a styled DataFrame table with optional column highlighting.

    Parameters
    ----------
    df                : DataFrame to display.
    highlight_col     : Name of column to colour-code (best value highlighted).
    higher_is_better  : Direction for highlight_col colouring.
    caption           : Optional caption shown above the table.
    num_format        : Format string applied to float columns.
    """
    if caption:
        st.caption(caption)

    if df.empty:
        st.info("No data to display.")
        return

    styled = df.style.format(num_format, subset=df.select_dtypes("float").columns)

    if highlight_col and highlight_col in df.columns:
        if higher_is_better:
            styled = styled.highlight_max(subset=[highlight_col], color="#166534", axis=0)
            styled = styled.highlight_min(subset=[highlight_col], color="#7f1d1d", axis=0)
        else:
            styled = styled.highlight_min(subset=[highlight_col], color="#166534", axis=0)
            styled = styled.highlight_max(subset=[highlight_col], color="#7f1d1d", axis=0)

    st.dataframe(styled, use_container_width=True, hide_index=True)
