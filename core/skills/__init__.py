"""
Skills Layer - Provider Management
====================================

Pluggable skill providers (browser-use, openhands, agent-s, etc.)
with RL-based provider selection.

For the unified registry (skills + UI + tools), see:
    from Jotty.core.registry import get_unified_registry

For the skills facade:
    from Jotty.core.skills.facade import get_registry, list_providers
"""

from .facade import (
    get_registry,
    list_providers,
    get_provider,
    list_skills,
    list_components,
)

__all__ = [
    'get_registry',
    'list_providers',
    'get_provider',
    'list_skills',
    'list_components',
]
