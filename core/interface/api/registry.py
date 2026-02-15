"""
Registry Facade for SDK
========================

Provides registry access to SDK via interface layer.
SDK should import from here, not directly from core.capabilities.

Architecture:
    SDK → core/interface/api/registry.py → core/capabilities/registry/

This ensures:
- SDK respects layer boundaries
- Core can refactor registry without breaking SDK
- Clean separation of concerns
"""

from Jotty.core.capabilities.registry.unified_registry import get_unified_registry


def get_registry():
    """
    Get the unified skill registry.

    Returns:
        UnifiedRegistry: The unified skill and provider registry
    """
    return get_unified_registry()


__all__ = ["get_registry", "get_unified_registry"]
