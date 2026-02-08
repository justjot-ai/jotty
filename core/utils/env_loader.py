"""
Environment Loader
==================

Centralized .env loading for all Jotty components.
Eliminates duplicate dotenv loading code across skills.

Usage:
    from Jotty.core.utils.env_loader import load_jotty_env
    load_jotty_env()
"""

import os
from pathlib import Path
from typing import Optional

# Track if already loaded to avoid redundant loads
_env_loaded = False


def get_jotty_root() -> Path:
    """Get the Jotty root directory."""
    # Navigate from this file: core/utils/env_loader.py -> core/utils -> core -> Jotty
    return Path(__file__).resolve().parent.parent.parent


def load_jotty_env(env_file: Optional[str] = None, override: bool = False) -> bool:
    """
    Load environment variables from Jotty's .env file.

    Args:
        env_file: Optional path to .env file. Defaults to Jotty/.env
        override: If True, override existing environment variables

    Returns:
        True if .env was loaded, False otherwise
    """
    global _env_loaded

    # Skip if already loaded (unless override requested)
    if _env_loaded and not override:
        return True

    try:
        from dotenv import load_dotenv
    except ImportError:
        return False

    if env_file:
        env_path = Path(env_file)
    else:
        env_path = get_jotty_root() / ".env"

    if env_path.exists():
        load_dotenv(env_path, override=override)
        _env_loaded = True
        return True

    return False


def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get environment variable, loading .env if not already loaded.

    Args:
        key: Environment variable name
        default: Default value if not set

    Returns:
        Environment variable value or default
    """
    load_jotty_env()
    return os.getenv(key, default)


def get_env_bool(key: str, default: bool = False) -> bool:
    """Get environment variable as boolean."""
    value = get_env(key, "").lower()
    if value in ("true", "1", "yes", "on"):
        return True
    elif value in ("false", "0", "no", "off"):
        return False
    return default


def get_env_int(key: str, default: int = 0) -> int:
    """Get environment variable as integer."""
    value = get_env(key)
    if value is not None:
        try:
            return int(value)
        except ValueError:
            pass
    return default
