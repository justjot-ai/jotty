"""
Jotty Skill SDK
===============

Standalone utilities for skill development. Skills should import from here
instead of from core.utils to minimize framework coupling.

All exports are pure utility functions with zero framework dependencies
(except tool_helpers which has a lightweight foundation import for param aliases).

Usage:
    from Jotty.core.capabilities.sdk import tool_helpers, env_loader, SkillStatus
    from Jotty.core.capabilities.sdk.smart_fetcher import smart_fetch
    from Jotty.core.capabilities.sdk.api_client import BaseAPIClient
"""

# Re-export submodules for easy access
from . import api_client, async_utils, env_loader, skill_status, smart_fetcher, tool_helpers
from .async_utils import StatusCallback, StatusReporter, safe_status
from .env_loader import get_env, get_env_bool, get_env_int, load_jotty_env

# Convenience re-exports
from .skill_status import SkillStatus, get_status

__all__ = [
    # Submodules
    "tool_helpers",
    "env_loader",
    "skill_status",
    "api_client",
    "async_utils",
    "smart_fetcher",
    # Convenience
    "SkillStatus",
    "get_status",
    "load_jotty_env",
    "get_env",
    "get_env_bool",
    "get_env_int",
    "safe_status",
    "StatusReporter",
    "StatusCallback",
]
