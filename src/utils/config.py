"""
src/utils/config.py
-------------------
Central configuration loader.

Usage anywhere in the project:
    from src.utils.config import cfg
    lat = cfg["location"]["latitude"]
    model_path = cfg["lstm"]["model_save_path"]

The module reads config.yaml once at import time and caches the result.
Environment variables in .env override yaml values when both exist
(useful for keeping API keys out of version control).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
# PROJECT_ROOT is the directory that contains this repo, regardless of where
# Python is invoked from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
ENV_PATH = PROJECT_ROOT / ".env"


def _load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file and return its contents as a dict."""
    if not path.exists():
        raise FileNotFoundError(
            f"config.yaml not found at {path}. "
            "Make sure you are running from the project root."
        )
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _resolve_paths(cfg: dict[str, Any], root: Path) -> dict[str, Any]:
    """
    Walk the config dict and convert every value that ends with a known
    file-extension or directory keyword into an absolute Path string.
    This means modules can call Path(cfg["lstm"]["model_save_path"]) and
    it will always resolve correctly no matter where main.py is run from.
    """
    path_keywords = {".csv", ".pkl", ".h5", ".keras", ".zip", ".log", "_dir", "_path", "_file"}

    def _resolve(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _resolve(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_resolve(item) for item in obj]
        if isinstance(obj, str):
            if any(obj.endswith(kw) or kw in obj for kw in path_keywords):
                resolved = (root / obj).resolve()
                return str(resolved)
        return obj

    return _resolve(cfg)


def _load_config() -> dict[str, Any]:
    """Load yaml, apply .env overrides, resolve relative paths."""
    # Load .env so os.environ is populated before we read it
    load_dotenv(ENV_PATH)

    raw = _load_yaml(CONFIG_PATH)
    cfg = _resolve_paths(raw, PROJECT_ROOT)

    # Inject any env-var overrides for sensitive fields
    # Pattern: ENV var NASA_API_KEY overrides cfg["nasa"]["api_key"]
    env_overrides = {
        "NASA_API_KEY":      ("nasa_power", "api_key"),
        "OPEN_METEO_KEY":   ("open_meteo", "api_key"),
    }
    for env_var, (section, key) in env_overrides.items():
        value = os.getenv(env_var)
        if value and section in cfg.get("data", {}):
            cfg["data"][section][key] = value

    return cfg


# Module-level singleton — imported by all other modules
cfg: dict[str, Any] = _load_config()


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def get(key_path: str, default: Any = None) -> Any:
    """
    Dot-notation getter for nested config values.

    Example:
        get("lstm.units")          → [128, 64]
        get("location.latitude")   → 17.385044
        get("missing.key", 0)      → 0
    """
    keys = key_path.split(".")
    node = cfg
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


def ensure_dirs() -> None:
    """
    Create all output directories declared in config if they don't exist.
    Call this once at the start of main.py.
    """
    dir_keys = [
        "data.raw_dir",
        "data.processed_dir",
        "data.external_dir",
        "rl.log_dir",
        "simulation.output_dir",
        "logging.log_dir",
    ]
    for key in dir_keys:
        path_str = get(key)
        if path_str:
            Path(path_str).mkdir(parents=True, exist_ok=True)

    # Also ensure models/saved exists
    models_dir = PROJECT_ROOT / "models" / "saved"
    models_dir.mkdir(parents=True, exist_ok=True)


def project_root() -> Path:
    """Return the absolute project root Path object."""
    return PROJECT_ROOT


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Config loader self-test ===")
    print(f"Project root : {project_root()}")
    print(f"Region       : {get('project.region')}")
    print(f"Latitude     : {get('location.latitude')}")
    print(f"LSTM units   : {get('lstm.units')}")
    print(f"Train split  : {get('preprocessing.split_ratios.train')}")
    print(f"Log level    : {get('logging.level')}")
    print("\nAll config keys at top level:")
    for k in cfg:
        print(f"  {k}")
    print("\nconfig.py OK")
