"""
File Logging Setup for Jotty
=============================

Configures Python logging to write to files based on SwarmConfig.
"""
import logging
from pathlib import Path
from typing import Optional


def setup_file_logging(
    output_dir: str,
    enable_beautified: bool = True,
    enable_debug: bool = True,
    log_level: str = "INFO"
) -> None:
    """
    Setup file logging for Jotty.

    Args:
        output_dir: Base output directory (e.g., ./outputs/run_20260111_123456)
        enable_beautified: Create human-readable logs/beautified.log
        enable_debug: Create detailed logs/debug.log
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    # Create logs directory
    logs_dir = Path(output_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Get root logger
    root_logger = logging.getLogger()

    # Set level
    level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            root_logger.removeHandler(handler)

    # Beautified log (human-readable)
    if enable_beautified:
        beautified_path = logs_dir / "beautified.log"
        beautified_handler = logging.FileHandler(beautified_path, mode='w', encoding='utf-8')
        beautified_handler.setLevel(logging.INFO)

        # Clean format
        beautified_formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        beautified_handler.setFormatter(beautified_formatter)
        root_logger.addHandler(beautified_handler)

    # Debug log (detailed)
    if enable_debug:
        debug_path = logs_dir / "debug.log"
        debug_handler = logging.FileHandler(debug_path, mode='w', encoding='utf-8')
        debug_handler.setLevel(logging.DEBUG)

        # Detailed format
        debug_formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        debug_handler.setFormatter(debug_formatter)
        root_logger.addHandler(debug_handler)

    # Log that file logging is active
    root_logger.info(f" File logging enabled: {logs_dir}/")
    if enable_beautified:
        root_logger.info(f"   • Beautified log: {beautified_path}")
    if enable_debug:
        root_logger.info(f"   • Debug log: {debug_path}")


def close_file_logging() -> None:
    """Close all file handlers to flush logs."""
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            handler.close()
            root_logger.removeHandler(handler)
