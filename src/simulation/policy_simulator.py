"""
src/simulation/policy_simulator.py
------------------------------------
Runs parameterised energy-policy scenarios forward in time,
returning annual cost, emissions, reliability, and curtailment
metrics for each year of the simulation horizon.

Each scenario is defined in config.yaml under simulation.scenarios.
The simulator uses a simplified hourly energy balance:
  ─ Renewable generation = capacity × capacity_factor(t)
  ─ Storage dispatched by a rule-based controller (RL model optionally)
  ─ Grid serves residual demand + handles excess

Output
------
reports/simulation/scenario_results.csv
reports/simulation/scenario_results.json

Usage (standalone)
------------------
    python -m src.simulation.policy_simulator
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.utils.config import get
from src.utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SIM_YEARS     = int(get("simulation.simulation_years", 10))
SCENARIOS     = get("simulation.scenarios", [])
OUTPUT_DIR    = Path(get("simulation.output_dir", "reports/simulation"))

SOLAR_CAP_KW  = float(get("rl.env.solar_capacity_kw",  1000))
WIND_CAP_KW   = float(get("rl.env.wind_capacity_kw",    500))
BATT_CAP_KWH  = float(get("rl.env.battery_capacity_kwh",500))

# Base capacity factors (annual averages for Hyderabad)
SOLAR_CF      = 0.20  # 20% — typical Telangana
WIND_CF       = 0.28  # 28%
# Hourly profile shape (used for 8760-point simulation)
HOURS         = 8760

CARBON_GRID_KG_KWH = 0.82  # India grid emission factor

GRID_IMPORT_PRICE  = float(get("rl.env.grid_import_price_per_kwh",  7.5))
GRID_EXPORT_PRICE  = float(get("rl.env.grid_export_price_per_kwh",  3.5))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hourly_solar_profile() -> np.ndarray:
    """Normalised hourly solar profile over 8760 hours (0–1)."""
    rng = np.random.default_rng(42)
    h = np.arange(HOURS) % 24
    day_of_year = np.arange(HOURS) // 24
    seasonal = 1.0 + 0.15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    profile = np.clip(np.sin(np.pi * (h - 6) / 12), 0, 1) * seasonal
    profile += rng.normal(0, 0.03, HOURS)
    return np.clip(profile, 0, 1).astype(np.float32)


def _hourly_wind_profile() -> np.ndarray:
    """Normalised hourly wind profile."""
    rng = np.random.default_rng(7)
    profile = 0.28 + 0.12 * np.sin(2 * np.pi * np.arange(HOURS) / (24 * 7))
    profile += rng.normal(0, 0.05, HOURS)
    return np.clip(profile, 0, 1).astype(np.float32)


def _hourly_demand_profile(base_demand_kw: float) -> np.ndarray:
    """Hourly demand profile centred around base_demand_kw."""
    rng = np.random.default_rng(13)
    h = np.arange(HOURS) % 24
    shape = (0.65 + 0.20 * np.sin(np.pi * (h - 4) / 12)
             + 0.15 * np.exp(-0.5 * ((h - 9) / 2) ** 2)
             + 0.20 * np.exp(-0.5 * ((h - 20) / 2) ** 2))
    demand = base_demand_kw * shape * (1 + rng.normal(0, 0.02, HOURS))
    return np.clip(demand, 0.3 * base_demand_kw, 1.8 * base_demand_kw).astype(np.float32)


def _simulate_year(
    scenario: dict,
    year: int,
    base_demand_kw: float,
) -> dict:
    """
    Simulate one year of microgrid operation for a given scenario.

    Returns a dict of summary metrics for that year.
    """
    renewable_pct      = float(scenario.get("renewable_pct", 30)) / 100
    storage_kwh        = float(scenario.get("storage_capacity_kwh", 500))
    demand_growth      = float(scenario.get("demand_growth_pct_per_year", 4)) / 100
    carbon_price       = float(scenario.get("carbon_price_inr_per_kg", 0.5))

    # Scale demand for this year
    demand_kw = base_demand_kw * (1 + demand_growth) ** year

    # Scale renewable capacity proportional to renewable_pct target
    total_cap_kw = demand_kw / SOLAR_CF  # rough sizing
    solar_cap_kw = total_cap_kw * renewable_pct * 0.65  # 65% solar, 35% wind
    wind_cap_kw  = total_cap_kw * renewable_pct * 0.35

    solar_profile  = _hourly_solar_profile()
    wind_profile   = _hourly_wind_profile()
    demand_profile = _hourly_demand_profile(demand_kw)

    solar_gen  = solar_profile  * solar_cap_kw
    wind_gen   = wind_profile   * wind_cap_kw
    total_re   = solar_gen + wind_gen

    # Simple battery dispatch: charge when surplus, discharge when deficit
    soc = storage_kwh * 0.5
    grid_import   = np.zeros(HOURS)
    grid_export   = np.zeros(HOURS)
    curtailment   = np.zeros(HOURS)
    unserved      = np.zeros(HOURS)

    for t in range(HOURS):
        net = total_re[t] - demand_profile[t]  # positive → surplus

        if net > 0:
            charge = min(net, storage_kwh - soc)
            soc += charge
            curtailment[t] = net - charge
            grid_export[t] = 0.0  # assume no export in this simplified model
        else:
            deficit = -net
            discharge = min(deficit, soc)
            soc -= discharge
            residual = deficit - discharge
            grid_import[t] = residual

    # Metrics
    total_demand_kwh     = demand_profile.sum()
    total_import_kwh     = grid_import.sum()
    total_curtail_kwh    = curtailment.sum()
    total_re_kwh         = total_re.sum()
    unserved_energy_kwh  = float(unserved.sum())  # always 0 in this model

    reliability_pct = 100.0 * (1 - unserved_energy_kwh / (total_demand_kwh + 1e-6))
    curtailment_pct = 100.0 * total_curtail_kwh / (total_re_kwh + 1e-6)
    re_fraction     = 100.0 * (total_demand_kwh - total_import_kwh) / (total_demand_kwh + 1e-6)

    grid_cost_inr    = total_import_kwh * GRID_IMPORT_PRICE
    carbon_kg        = total_import_kwh * CARBON_GRID_KG_KWH
    carbon_cost_inr  = carbon_kg * carbon_price
    total_cost_inr   = grid_cost_inr + carbon_cost_inr

    return {
        "year":             year,
        "demand_mwh":       round(total_demand_kwh / 1000, 1),
        "re_fraction_pct":  round(re_fraction, 2),
        "grid_import_mwh":  round(total_import_kwh / 1000, 1),
        "curtailment_pct":  round(curtailment_pct, 2),
        "reliability_pct":  round(reliability_pct, 3),
        "total_cost_inr":   round(total_cost_inr, 0),
        "carbon_kg":        round(carbon_kg, 0),
    }


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PolicySimulator:
    """
    Run multiple policy scenarios and accumulate yearly metrics.
    """

    def __init__(self) -> None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        # Estimate base demand from demand profile (year 0)
        demand_path = Path(get("data.grid_demand.output_file", "data/raw/demand_raw.csv"))
        if demand_path.exists():
            df = pd.read_csv(demand_path, index_col=0, parse_dates=True)
            # Convert MW → kW for simulation
            self.base_demand_kw = float(df["demand_mw"].mean() * 1000) if "demand_mw" in df.columns else 8000.0
        else:
            self.base_demand_kw = 8000.0  # ~8 MW base load
        log.info("Base demand: %.1f kW", self.base_demand_kw)

    # ------------------------------------------------------------------
    def run_scenario(self, scenario: dict) -> pd.DataFrame:
        """Run one named scenario for all simulation years."""
        name = scenario.get("name", "unnamed")
        log.info("  Running scenario: %s", name)
        rows: List[dict] = []
        for year in range(SIM_YEARS):
            row = _simulate_year(scenario, year, self.base_demand_kw)
            row["scenario"] = name
            rows.append(row)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    def run_all_scenarios(self) -> pd.DataFrame:
        """
        Run every scenario defined in config.yaml.

        Returns
        -------
        pd.DataFrame — all scenarios concatenated, with columns:
            scenario, year, demand_mwh, re_fraction_pct, …
        """
        log.info("=== PolicySimulator: running %d scenarios ===", len(SCENARIOS))
        all_dfs: List[pd.DataFrame] = []

        for sc in SCENARIOS:
            df = self.run_scenario(sc)
            all_dfs.append(df)

        if not all_dfs:
            log.warning("No scenarios found in config.yaml. "
                        "Add entries under simulation.scenarios.")
            return pd.DataFrame()

        results = pd.concat(all_dfs, ignore_index=True)

        # Save
        results.to_csv(OUTPUT_DIR / "scenario_results.csv", index=False)
        results.to_json(OUTPUT_DIR / "scenario_results.json", orient="records", indent=2)
        log.info("Simulation results saved → %s", OUTPUT_DIR)
        return results


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sim = PolicySimulator()
    df = sim.run_all_scenarios()
    print(df.to_string(index=False))
