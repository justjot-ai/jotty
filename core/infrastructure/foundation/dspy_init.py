"""
Shared DSPy LM initialization and API key loading.

Consolidates the duplicated .env loading and DSPy configuration logic
that was previously copy-pasted between BaseAgent and SwarmLearning.

Thread-safe: uses a module-level lock to prevent races when multiple
agents/swarms initialize concurrently.

Author: Jotty Team
Date: February 2026
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Module-level lock for DSPy LM initialization.
# Prevents race conditions when multiple agents/swarms concurrently
# try to configure the global dspy.settings.lm.
_dspy_lm_lock = threading.Lock()

# Project root: 4 levels up from this file
# (foundation -> infrastructure -> core -> Jotty)
_PROJECT_ROOT = Path(__file__).parents[3]


def load_api_keys(dotenv_paths: list[Path] | None = None) -> None:
    """Load API keys from .env files into os.environ if not already set.

    Reads key=value pairs from dotenv files, skipping comments and blank
    lines.  Only sets keys that are not already present in the environment.

    Args:
        dotenv_paths: Paths to try, in order.  Defaults to
            ``[<project_root>/.env.anthropic, <project_root>/.env]``.
    """
    if dotenv_paths is None:
        dotenv_paths = [_PROJECT_ROOT / ".env.anthropic", _PROJECT_ROOT / ".env"]

    for env_file in dotenv_paths:
        if not env_file.exists():
            continue
        try:
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key_name, key_val = line.split("=", 1)
                        key_name, key_val = key_name.strip(), key_val.strip()
                        if key_val and key_name not in os.environ:
                            os.environ[key_name] = key_val
                            logger.debug("Loaded %s from %s", key_name, env_file)
        except Exception as e:
            logger.warning("Failed to load API keys from %s: %s", env_file, e)


def auto_detect_provider() -> str:
    """Auto-detect available LM provider from environment variables.

    Priority: Anthropic > OpenAI > Groq > OpenRouter > Zen > anthropic (default).
    """
    for env_var, provider in (
        ("ANTHROPIC_API_KEY", "anthropic"),
        ("OPENAI_API_KEY", "openai"),
        ("GROQ_API_KEY", "groq"),
        ("OPENROUTER_API_KEY", "openrouter"),
        ("OPENCODE_ZEN_API_KEY", "zen"),
    ):
        if os.environ.get(env_var):
            return provider
    return "anthropic"


def init_dspy_lm(
    provider: str | None = None,
    model: str | None = None,
    max_tokens: int = 8192,
    temperature: float | None = None,
    timeout: int | None = None,
    force: bool = False,
) -> Optional[Any]:
    """Thread-safe DSPy LM initialization.

    If ``dspy.settings.lm`` is already configured, returns the existing
    instance without reconfiguring (unless *force=True*).  Otherwise loads
    API keys, creates an LM via :class:`UnifiedLMProvider`, and calls
    ``dspy.configure()``.

    Args:
        provider: LLM provider name (auto-detected if *None*).
        model: Model alias (e.g. ``"haiku"``, ``"sonnet"``).  Defaults vary by caller.
        max_tokens: Maximum output tokens.
        temperature: Sampling temperature.
        timeout: Request timeout in seconds.
        force: If *True*, reconfigure even if already set.

    Returns:
        The configured ``dspy.LM`` instance, or *None* if DSPy is unavailable.
    """
    try:
        import dspy
    except ImportError:
        logger.debug("DSPy not available, skipping LM configuration")
        return None

    with _dspy_lm_lock:
        # Re-check inside lock — another thread may have configured it
        if not force and hasattr(dspy.settings, "lm") and dspy.settings.lm is not None:
            return dspy.settings.lm

        # Load API keys from .env files if not in environment
        if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("OPENROUTER_API_KEY"):
            load_api_keys()

        # Resolve provider
        resolved_provider = provider or auto_detect_provider()

        # Use UnifiedLMProvider for consistent LM initialization
        try:
            from Jotty.core.infrastructure.foundation.unified_lm_provider import (
                UnifiedLMProvider,
            )

            kwargs: dict[str, Any] = {
                "provider": resolved_provider,
                "model": model or "haiku",
                "max_tokens": max_tokens,
            }
            if temperature is not None:
                kwargs["temperature"] = temperature
            if timeout is not None:
                kwargs["timeout"] = timeout

            lm = UnifiedLMProvider.create_lm(**kwargs)
            dspy.configure(lm=lm)
            logger.info(
                "Configured DSPy LM via UnifiedLMProvider " "(provider=%s, model=%s)",
                resolved_provider,
                kwargs["model"],
            )
            return lm

        except Exception as e:
            logger.warning("UnifiedLMProvider initialization failed: %s", e)
            # Fall back to configure_dspy_lm (auto-detect)
            try:
                from Jotty.core.infrastructure.foundation.unified_lm_provider import (
                    configure_dspy_lm,
                )

                lm = configure_dspy_lm()
                logger.info("Configured DSPy LM via auto-detection")
                return lm
            except Exception as e2:
                logger.error("LM initialization failed completely: %s", e2)
                return None


__all__ = [
    "load_api_keys",
    "auto_detect_provider",
    "init_dspy_lm",
    "_dspy_lm_lock",
]
