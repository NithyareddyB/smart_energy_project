"""
src/utils/logger.py
-------------------
Centralised logging setup.

Every module gets a logger like this:
    from src.utils.logger import get_logger
    log = get_logger(__name__)
    log.info("Training started")
    log.warning("Missing values found: %d", n_missing)
    log.error("API call failed", exc_info=True)

Logs are written to:
  - Console  (coloured, human-readable)
  - File     (plain text, rotating, path from config.yaml)

Call configure_logging() once in main.py before any other imports.
After that every get_logger(__name__) call is pre-configured.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

from src.utils.config import cfg, get

# ---------------------------------------------------------------------------
# ANSI colour codes for console output
# ---------------------------------------------------------------------------
_RESET = "\033[0m"
_BOLD  = "\033[1m"
_COLOURS = {
    "DEBUG":    "\033[36m",   # cyan
    "INFO":     "\033[32m",   # green
    "WARNING":  "\033[33m",   # yellow
    "ERROR":    "\033[31m",   # red
    "CRITICAL": "\033[35m",   # magenta
}


class _ColourFormatter(logging.Formatter):
    """Formatter that adds ANSI colour to the levelname in console output."""

    FMT = "{colour}{bold}[{levelname:<8}]{reset} {asctime} | {name} | {message}"
    DATE_FMT = "%Y-%m-%d %H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:
        colour = _COLOURS.get(record.levelname, "")
        record.colour = colour
        record.bold   = _BOLD
        record.reset  = _RESET
        formatter = logging.Formatter(
            fmt=self.FMT.format(
                colour=colour, bold=_BOLD, reset=_RESET,
                levelname="{levelname}", asctime="{asctime}",
                name="{name}", message="{message}",
            ),
            datefmt=self.DATE_FMT,
            style="{",
        )
        return formatter.format(record)


class _PlainFormatter(logging.Formatter):
    """Plain formatter for file output (no ANSI codes)."""

    FMT      = "[{levelname:<8}] {asctime} | {name} | {message}"
    DATE_FMT = "%Y-%m-%d %H:%M:%S"

    def __init__(self) -> None:
        super().__init__(fmt=self.FMT, datefmt=self.DATE_FMT, style="{")


# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------
_configured = False


def configure_logging(
    level: Optional[str] = None,
    log_file: Optional[str] = None,
) -> None:
    """
    Configure the root logger. Call this ONCE in main.py at startup.

    Parameters
    ----------
    level    : Override log level (default taken from config.yaml).
    log_file : Override log file path (default taken from config.yaml).
    """
    global _configured

    # ── Resolve settings ──────────────────────────────────────────────────
    level_str = (level or get("logging.level", "INFO")).upper()
    numeric_level = getattr(logging, level_str, logging.INFO)

    log_dir  = Path(get("logging.log_dir", "reports/logs"))
    filename = log_file or get("logging.log_file", "pipeline.log")
    log_path = log_dir / filename
    log_dir.mkdir(parents=True, exist_ok=True)

    max_bytes    = int(get("logging.max_bytes",    10_485_760))  # 10 MB
    backup_count = int(get("logging.backup_count", 5))

    # ── Root logger ───────────────────────────────────────────────────────
    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Remove any handlers added by previous configure_logging calls
    root.handlers.clear()

    # ── Console handler ───────────────────────────────────────────────────
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(numeric_level)
    # Disable colour on Windows without ANSI support
    use_colour = sys.stdout.isatty() or sys.platform != "win32"
    console.setFormatter(_ColourFormatter() if use_colour else _PlainFormatter())
    root.addHandler(console)

    # ── Rotating file handler ─────────────────────────────────────────────
    file_handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(_PlainFormatter())
    root.addHandler(file_handler)

    # ── Silence noisy third-party loggers ─────────────────────────────────
    for noisy in ("urllib3", "matplotlib", "PIL", "absl", "tensorflow",
                  "h5py", "numexpr", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True
    logging.getLogger(__name__).info(
        "Logging configured | level=%s | file=%s", level_str, log_path
    )


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger.  If configure_logging() has not been called yet
    (e.g. in a standalone script or notebook), apply a minimal default setup
    so log calls still produce visible output.

    Parameters
    ----------
    name : Typically pass __name__ so log lines show the module path.
    """
    if not _configured:
        # Minimal fallback — console only, INFO level
        logging.basicConfig(
            level=logging.INFO,
            format="[%(levelname)-8s] %(asctime)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            stream=sys.stdout,
        )
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Convenience: a project-level logger for one-liners in scripts
# ---------------------------------------------------------------------------
log = get_logger("smart_energy")


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    configure_logging(level="DEBUG")
    test_log = get_logger("test.module")
    test_log.debug("This is a DEBUG message")
    test_log.info("This is an INFO message")
    test_log.warning("This is a WARNING message")
    test_log.error("This is an ERROR message")
    test_log.critical("This is a CRITICAL message")
    print("\nlogger.py OK — check reports/logs/pipeline.log for file output")
