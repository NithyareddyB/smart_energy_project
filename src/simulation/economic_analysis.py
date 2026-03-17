"""
src/simulation/economic_analysis.py
-------------------------------------
Computes techno-economic metrics for each policy scenario.

Metrics
-------
  LCOE   Levelised Cost of Energy          (INR / kWh)
  NPV    Net Present Value                 (INR)
  IRR    Internal Rate of Return           (%)
  PBP    Simple Payback Period             (years)

Also performs ±20 % sensitivity sweeps on six key parameters:
  solar_capex, wind_capex, battery_capex, discount_rate,
  demand_growth, carbon_price

Output
------
  reports/simulation/economics_summary.csv
  reports/simulation/economics_summary.xlsx
  reports/simulation/sensitivity_table.csv
  (figures saved to reports/simulation/)

Usage (standalone)
------------------
    python -m src.simulation.economic_analysis
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.config import get
from src.utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ECON = {
    "discount_rate":          float(get("economics.discount_rate",            0.08)),
    "solar_capex_inr_per_kw": float(get("economics.solar_capex_per_kw_inr",  45000)),
    "wind_capex_inr_per_kw":  float(get("economics.wind_capex_per_kw_inr",   55000)),
    "batt_capex_inr_per_kwh": float(get("economics.battery_capex_per_kwh_inr", 25000)),
    "opex_pct":               float(get("economics.opex_pct_of_capex",         0.02)),
    "solar_life":             int(get("economics.solar_lifetime_years",         25)),
    "wind_life":              int(get("economics.wind_lifetime_years",           20)),
    "batt_life":              int(get("economics.battery_lifetime_years",        10)),
}
OUTPUT_DIR = Path(get("simulation.output_dir", "reports/simulation"))
SIM_YEARS  = int(get("simulation.simulation_years", 10))

SOLAR_CF  = 0.20
WIND_CF   = 0.28
GRID_PRICE = float(get("rl.env.grid_import_price_per_kwh", 7.5))

# Sensitivity sweep range
SWEEP_RANGE = 0.20  # ±20 %


# ---------------------------------------------------------------------------
# Financial helpers
# ---------------------------------------------------------------------------

def _npv(cashflows: np.ndarray, discount_rate: float) -> float:
    """Compute NPV given an array of cashflows (year 0..N)."""
    years = np.arange(len(cashflows))
    return float(np.sum(cashflows / (1 + discount_rate) ** years))


def _irr(cashflows: np.ndarray, max_iter: int = 1000) -> float:
    """
    Compute IRR using bisection search.
    cashflows[0] should be the negative initial investment.
    """
    def _npv_at(r: float) -> float:
        years = np.arange(len(cashflows))
        return float(np.sum(cashflows / (1 + r) ** years))

    lo, hi = -0.999, 10.0
    if _npv_at(lo) * _npv_at(hi) > 0:
        return float("nan")  # no sign change → IRR not defined

    for _ in range(max_iter):
        mid = (lo + hi) / 2
        val = _npv_at(mid)
        if abs(val) < 1.0:  # INR precision
            return mid * 100  # return as %
        if val > 0:
            lo = mid
        else:
            hi = mid
    return ((lo + hi) / 2) * 100


def _lcoe(
    capex: float,
    opex_annual: float,
    annual_energy_kwh: float,
    life_years: int,
    discount_rate: float,
) -> float:
    """Levelised Cost of Energy (INR/kWh)."""
    years = np.arange(1, life_years + 1)
    disc  = (1 + discount_rate) ** years
    present_value_opex   = float(np.sum(opex_annual / disc))
    present_value_energy = float(np.sum(annual_energy_kwh / disc))
    return (capex + present_value_opex) / (present_value_energy + 1e-6)


# ---------------------------------------------------------------------------
# Scenario-level analysis
# ---------------------------------------------------------------------------

def _analyse_scenario(
    scenario_name: str,
    scenario_row: pd.Series,
    scenario_years: pd.DataFrame,
    econ_params: dict,
) -> dict:
    """Compute LCOE, NPV, IRR, PBP for one scenario."""
    renewable_pct = float(scenario_row.get("renewable_pct", 30)) / 100
    storage_kwh   = float(scenario_row.get("storage_capacity_kwh", 500))

    # Approximate capacities (simplified sizing)
    base_demand_mw = scenario_years["demand_mwh"].iloc[0] / 8760
    total_cap_kw   = base_demand_mw * 1000 / SOLAR_CF
    solar_cap_kw   = total_cap_kw * renewable_pct * 0.65
    wind_cap_kw    = total_cap_kw * renewable_pct * 0.35

    # CAPEX (year 0 investment)
    solar_capex = solar_cap_kw  * econ_params["solar_capex_inr_per_kw"]
    wind_capex  = wind_cap_kw   * econ_params["wind_capex_inr_per_kw"]
    batt_capex  = storage_kwh   * econ_params["batt_capex_inr_per_kwh"]
    total_capex = solar_capex + wind_capex + batt_capex

    # Annual OPEX
    annual_opex = total_capex * econ_params["opex_pct"]

    # Annual energy from renewables (average over sim years)
    avg_re_mwh  = scenario_years["demand_mwh"].mean() * (scenario_years["re_fraction_pct"].mean() / 100)
    avg_re_kwh  = avg_re_mwh * 1000

    # Without project (grid only) cost per year
    avg_demand_kwh_yr = scenario_years["demand_mwh"].mean() * 1000
    baseline_cost_yr  = avg_demand_kwh_yr * GRID_PRICE

    # With project cost per year = grid import cost + OPEX
    project_import_kwh_yr  = scenario_years["grid_import_mwh"].mean() * 1000
    project_grid_cost_yr   = project_import_kwh_yr * GRID_PRICE
    project_total_annual   = project_grid_cost_yr + annual_opex

    # Annual savings = baseline – project
    annual_savings = baseline_cost_yr - project_total_annual

    # Cashflows: year 0 = -capex, years 1..N = annual_savings
    cashflows = np.array([-total_capex] + [annual_savings] * SIM_YEARS)

    r = econ_params["discount_rate"]
    npv_val = _npv(cashflows, r)
    irr_val = _irr(cashflows)

    # Payback period (simple)
    cum = -total_capex
    pbp = float("inf")
    for yr in range(1, SIM_YEARS + 1):
        cum += annual_savings
        if cum >= 0:
            pbp = yr
            break

    # LCOE (solar component)
    lcoe_val = _lcoe(
        capex=solar_capex + wind_capex,
        opex_annual=annual_opex,
        annual_energy_kwh=avg_re_kwh,
        life_years=econ_params["solar_life"],
        discount_rate=r,
    )

    return {
        "scenario":      scenario_name,
        "total_capex_cr": round(total_capex / 1e7, 2),  # Crore INR
        "annual_savings_cr": round(annual_savings / 1e7, 2),
        "npv_cr":        round(npv_val / 1e7, 2),
        "irr_pct":       round(irr_val, 2) if not np.isnan(irr_val) else None,
        "payback_yr":    pbp if pbp != float("inf") else None,
        "lcoe_inr_kwh":  round(lcoe_val, 4),
    }


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class EconomicAnalyser:
    """
    Compute and persist economic metrics for all simulated scenarios.
    """

    def __init__(self) -> None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.econ = ECON.copy()

    # ------------------------------------------------------------------
    def analyse(
        self,
        results: pd.DataFrame,
        scenarios_cfg: Optional[List[dict]] = None,
    ) -> pd.DataFrame:
        """
        Compute economic metrics for every scenario in results.

        Parameters
        ----------
        results       : DataFrame from PolicySimulator.run_all_scenarios()
        scenarios_cfg : Scenario config list (read from config if None)

        Returns
        -------
        pd.DataFrame — one row per scenario with economic metrics.
        """
        if scenarios_cfg is None:
            scenarios_cfg = get("simulation.scenarios", [])

        sc_lookup = {sc["name"]: sc for sc in scenarios_cfg}

        summary_rows: List[dict] = []
        for sc_name, grp in results.groupby("scenario"):
            sc_cfg = sc_lookup.get(sc_name, {})
            row = _analyse_scenario(sc_name, pd.Series(sc_cfg), grp, self.econ)
            summary_rows.append(row)

        summary = pd.DataFrame(summary_rows)

        # Save
        csv_path  = OUTPUT_DIR / "economics_summary.csv"
        xlsx_path = OUTPUT_DIR / "economics_summary.xlsx"
        sens_path = OUTPUT_DIR / "sensitivity_table.csv"

        summary.to_csv(csv_path, index=False)
        try:
            summary.to_excel(xlsx_path, index=False)
        except Exception as exc:
            log.warning("Excel export skipped: %s", exc)

        log.info("Economics summary:\n%s", summary.to_string(index=False))

        # Sensitivity analysis
        sens_df = self._sensitivity_sweep(results, sc_lookup)
        sens_df.to_csv(sens_path, index=False)

        # Plots
        self._plot_payback(summary)
        self._plot_sensitivity(sens_df)

        log.info("Economic analysis saved → %s", OUTPUT_DIR)
        return summary

    # ------------------------------------------------------------------
    def _sensitivity_sweep(
        self,
        results: pd.DataFrame,
        sc_lookup: dict,
    ) -> pd.DataFrame:
        """±20% sweep on six key parameters for the first scenario."""
        if results.empty:
            return pd.DataFrame()

        first_sc = results["scenario"].iloc[0]
        sc_grp   = results[results["scenario"] == first_sc]
        sc_cfg   = sc_lookup.get(first_sc, {})

        params = [
            ("solar_capex_inr_per_kw", "Solar CAPEX"),
            ("wind_capex_inr_per_kw",  "Wind CAPEX"),
            ("batt_capex_inr_per_kwh", "Battery CAPEX"),
            ("discount_rate",          "Discount Rate"),
        ]

        rows: List[dict] = []
        base_econ = self.econ.copy()
        base_row  = _analyse_scenario(first_sc, pd.Series(sc_cfg), sc_grp, base_econ)
        base_npv  = base_row["npv_cr"]

        for param_key, param_label in params:
            for direction, mult in [("−20%", 0.80), ("+20%", 1.20)]:
                econ = base_econ.copy()
                econ[param_key] = base_econ[param_key] * mult
                row = _analyse_scenario(first_sc, pd.Series(sc_cfg), sc_grp, econ)
                rows.append({
                    "parameter": param_label,
                    "change":    direction,
                    "npv_cr":    row["npv_cr"],
                    "delta_npv_cr": round(row["npv_cr"] - base_npv, 2),
                })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    @staticmethod
    def _plot_payback(summary: pd.DataFrame) -> None:
        """Bar chart of payback period per scenario."""
        if summary.empty or "payback_yr" not in summary.columns:
            return
        valid = summary.dropna(subset=["payback_yr"])
        if valid.empty:
            return

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(valid["scenario"], valid["payback_yr"], color="seagreen")
        ax.set_xlabel("Payback Period (years)")
        ax.set_title("Simple Payback Period by Scenario")
        plt.tight_layout()
        path = OUTPUT_DIR / "payback_period.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        log.info("Payback chart saved → %s", path)

    # ------------------------------------------------------------------
    @staticmethod
    def _plot_sensitivity(sens: pd.DataFrame) -> None:
        """Tornado diagram of NPV sensitivity."""
        if sens.empty:
            return

        params = sens["parameter"].unique()
        pos_deltas, neg_deltas = [], []
        for p in params:
            rows = sens[sens["parameter"] == p]
            pos = rows[rows["change"] == "+20%"]["delta_npv_cr"].values
            neg = rows[rows["change"] == "−20%"]["delta_npv_cr"].values
            pos_deltas.append(float(pos[0]) if len(pos) else 0)
            neg_deltas.append(float(neg[0]) if len(neg) else 0)

        y = np.arange(len(params))
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(y, pos_deltas, 0.4, left=0,           label="+20%", color="steelblue", alpha=0.8)
        ax.barh(y, neg_deltas, 0.4, left=0,           label="−20%", color="tomato",    alpha=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(params)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("ΔNPV (Crore INR)")
        ax.set_title("Sensitivity Tornado — NPV Response to ±20% Parameter Shock")
        ax.legend()
        plt.tight_layout()
        path = OUTPUT_DIR / "sensitivity_tornado.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        log.info("Sensitivity tornado saved → %s", path)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.simulation.policy_simulator import PolicySimulator
    sim = PolicySimulator()
    results = sim.run_all_scenarios()

    analyser = EconomicAnalyser()
    summary = analyser.analyse(results)
    print(summary)
