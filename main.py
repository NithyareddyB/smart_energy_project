"""
main.py
-------
CLI entry point for the Smart Energy Optimization System.

Run individual pipeline stages or the full end-to-end flow.

Usage examples
--------------
# Show all available commands
python main.py --help

# Run the full pipeline from data collection to simulation
python main.py --all

# Run only data collection
python main.py --collect

# Run preprocessing (assumes raw data already downloaded)
python main.py --preprocess

# Train all models
python main.py --train

# Run policy simulation (assumes models trained)
python main.py --simulate

# Launch the Streamlit dashboard
python main.py --dashboard

# Run with verbose debug logging
python main.py --train --log-level DEBUG
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

# ── Ensure project root is on the path (needed when running as script) ────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.config import cfg, get, ensure_dirs
from src.utils.logger import configure_logging, get_logger
from src.utils.helpers import timer

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Pipeline stage functions
# ---------------------------------------------------------------------------
# Each function is a thin orchestrator that imports the relevant module and
# runs it.  Heavy imports are kept inside the functions so that
# `python main.py --help` is instant even if TensorFlow is slow to import.
# ---------------------------------------------------------------------------


def run_collect() -> None:
    """Stage 1 — Download raw data from NASA POWER and Open-Meteo."""
    log.info("=" * 60)
    log.info("STAGE 1: Data Collection")
    log.info("=" * 60)
    with timer("Data collection"):
        from src.data_collection.nasa_api import NASASolarCollector
        from src.data_collection.weather_api import WeatherCollector
        from src.data_collection.grid_loader import GridDemandLoader

        # Solar & meteorological data
        solar = NASASolarCollector()
        solar.collect()

        # Additional weather variables (if separate from NASA)
        weather = WeatherCollector()
        weather.collect()

        # Grid demand data
        grid = GridDemandLoader()
        grid.load()

    log.info("Data collection complete. Check data/raw/")


def run_preprocess() -> None:
    """Stage 2 — Clean, engineer features, and split data."""
    log.info("=" * 60)
    log.info("STAGE 2: Preprocessing")
    log.info("=" * 60)
    with timer("Preprocessing"):
        from src.preprocessing.cleaner import DataCleaner
        from src.preprocessing.feature_eng import FeatureEngineer
        from src.preprocessing.scaler import DataScaler
        from src.preprocessing.splitter import DataSplitter

        cleaner = DataCleaner()
        df_clean = cleaner.clean()

        engineer = FeatureEngineer()
        df_features = engineer.transform(df_clean)

        scaler = DataScaler()
        df_scaled = scaler.fit_transform(df_features)

        splitter = DataSplitter()
        splits = splitter.split(df_scaled)

    log.info(
        "Preprocessing complete | train=%d  val=%d  test=%d rows",
        len(splits["X_train"]), len(splits["X_val"]), len(splits["X_test"]),
    )


def run_train() -> None:
    """Stage 3 — Train all 7 supervised models (LSTM, GRU, CNN-LSTM, RF, XGBoost, SVR, SARIMA)."""
    log.info("=" * 60)
    log.info("STAGE 3: Model Training (8-Model Suite)")
    log.info("=" * 60)

    # ── Deep Learning models ────────────────────────────────────────────────
    with timer("LSTM solar forecaster training"):
        from src.models.lstm_forecaster import LSTMForecaster
        LSTMForecaster(target="solar").train()

    with timer("LSTM wind forecaster training"):
        from src.models.lstm_forecaster import LSTMForecaster
        LSTMForecaster(target="wind").train()

    with timer("GRU wind forecaster training"):
        from src.models.gru_forecaster import GRUForecaster
        GRUForecaster().train()

    with timer("CNN-LSTM solar multi-step forecaster training"):
        from src.models.cnn_lstm_solar import CNNLSTMSolarForecaster
        CNNLSTMSolarForecaster().train()

    # ── Classical ML models ─────────────────────────────────────────────────
    with timer("Random Forest demand predictor training"):
        from src.models.rf_demand import RFDemandPredictor
        RFDemandPredictor().train()

    with timer("XGBoost demand predictor training"):
        from src.models.xgboost_demand import XGBoostDemandPredictor
        XGBoostDemandPredictor().train()

    with timer("SVR wind predictor training"):
        from src.models.svr_wind import SVRWindPredictor
        SVRWindPredictor().train()

    # ── Statistical baseline ────────────────────────────────────────────────
    with timer("SARIMA demand baseline training"):
        from src.models.sarima_demand import SARIMADemandForecaster
        SARIMADemandForecaster().train()

    log.info("All 7 supervised models trained. Saved to models/saved/")


def run_compare() -> None:
    """Print a side-by-side metric comparison table for all supervised models."""
    log.info("=" * 60)
    log.info("MODEL COMPARISON TABLE")
    log.info("=" * 60)
    import pandas as pd

    results = []
    model_map = {
        "LSTM Solar":     ("src.models.lstm_forecaster", "LSTMForecaster",         {"target": "solar"}),
        "LSTM Wind":      ("src.models.lstm_forecaster", "LSTMForecaster",         {"target": "wind"}),
        "GRU Wind":       ("src.models.gru_forecaster",  "GRUForecaster",          {}),
        "CNN-LSTM Solar": ("src.models.cnn_lstm_solar",  "CNNLSTMSolarForecaster", {}),
        "RF Demand":      ("src.models.rf_demand",       "RFDemandPredictor",      {}),
        "XGBoost Demand": ("src.models.xgboost_demand",  "XGBoostDemandPredictor", {}),
        "SVR Wind":       ("src.models.svr_wind",        "SVRWindPredictor",       {}),
        "SARIMA Demand":  ("src.models.sarima_demand",   "SARIMADemandForecaster", {}),
    }
    for name, (module_path, cls_name, kwargs) in model_map.items():
        try:
            import importlib
            mod = importlib.import_module(module_path)
            cls = getattr(mod, cls_name)
            instance = cls(**kwargs) if kwargs else cls()
            metrics = instance.evaluate()
            metrics["model"] = name
            results.append(metrics)
        except Exception as exc:
            log.warning("Could not evaluate %s: %s", name, exc)

    if results:
        df = pd.DataFrame(results).set_index("model")
        log.info("\n" + df.to_string())
        out = Path("reports") / "model_comparison.csv"
        df.to_csv(out)
        log.info("Comparison saved → %s", out)


def run_rl() -> None:
    """Stage 4 — Train the Reinforcement Learning load optimizer."""
    log.info("=" * 60)
    log.info("STAGE 4: RL Load Optimizer Training")
    log.info("=" * 60)
    with timer("RL optimizer training"):
        from src.models.rl_optimizer import RLLoadOptimizer
        rl = RLLoadOptimizer()
        rl.train()
        rl.evaluate()

    log.info("RL training complete. Model saved to models/saved/rl_optimizer.zip")


def run_simulate() -> None:
    """Stage 5 — Run policy scenarios and economic analysis."""
    log.info("=" * 60)
    log.info("STAGE 5: Policy Simulation & Economic Analysis")
    log.info("=" * 60)
    with timer("Policy simulation"):
        from src.simulation.policy_simulator import PolicySimulator
        sim = PolicySimulator()
        results = sim.run_all_scenarios()

    with timer("Economic analysis"):
        from src.simulation.economic_analysis import EconomicAnalyser
        econ = EconomicAnalyser()
        econ.analyse(results)

    log.info("Simulation complete. Reports saved to reports/simulation/")


def run_dashboard() -> None:
    """Launch the Streamlit dashboard."""
    log.info("Launching Streamlit dashboard on http://localhost:%s", get("dashboard.port", 8501))
    import subprocess
    dashboard_path = Path(__file__).parent / "dashboard" / "app.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(dashboard_path),
         "--server.port", str(get("dashboard.port", 8501))],
        check=True,
    )


# ---------------------------------------------------------------------------
# CLI definition
# ---------------------------------------------------------------------------

@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--all",        "run_all",   is_flag=True, help="Run the full pipeline end to end.")
@click.option("--collect",               is_flag=True, help="Stage 1: collect raw data.")
@click.option("--preprocess",            is_flag=True, help="Stage 2: clean & engineer features.")
@click.option("--train",                 is_flag=True, help="Stage 3: train all 7 supervised models.")
@click.option("--rl",                    is_flag=True, help="Stage 4: train RL optimizer.")
@click.option("--simulate",              is_flag=True, help="Stage 5: policy simulation & economics.")
@click.option("--compare",               is_flag=True, help="Print side-by-side metrics for all models.")
@click.option("--dashboard",             is_flag=True, help="Launch Streamlit dashboard.")
@click.option("--log-level", default=None,
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
              help="Override log level from config.yaml.")
def main(
    run_all: bool,
    collect: bool,
    preprocess: bool,
    train: bool,
    rl: bool,
    simulate: bool,
    compare: bool,
    dashboard: bool,
    log_level: str | None,
) -> None:
    """
    Smart Energy Optimization System — pipeline runner.

    Run stages individually or use --all for the full pipeline.
    """
    # ── Setup ─────────────────────────────────────────────────────────────
    configure_logging(level=log_level)
    ensure_dirs()

    log.info("Smart Energy Optimization System  |  region: %s", get("project.region"))

    # ── Decide which stages to run ────────────────────────────────────────
    if not any([run_all, collect, preprocess, train, rl, simulate, compare, dashboard]):
        click.echo(click.get_current_context().get_help())
        return

    stages = {
        "collect":    collect    or run_all,
        "preprocess": preprocess or run_all,
        "train":      train      or run_all,
        "rl":         rl         or run_all,
        "simulate":   simulate   or run_all,
        "compare":    compare,
        "dashboard":  dashboard,
    }

    # ── Execute stages in order ───────────────────────────────────────────
    with timer("Total pipeline"):
        if stages["collect"]:
            run_collect()

        if stages["preprocess"]:
            run_preprocess()

        if stages["train"]:
            run_train()

        if stages["rl"]:
            run_rl()

        if stages["simulate"]:
            run_simulate()

        if stages["compare"]:
            run_compare()

        if stages["dashboard"]:
            run_dashboard()

    log.info("Pipeline finished successfully.")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
