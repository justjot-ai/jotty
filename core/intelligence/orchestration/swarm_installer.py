"""
SwarmInstaller - Backward compatibility re-export.

Canonical location: Jotty.core.infrastructure.utils.package_installer

This file re-exports SwarmInstaller and InstallationResult so that
existing imports continue to work.
"""

from Jotty.core.infrastructure.utils.package_installer import (
    InstallationResult,
    SwarmInstaller,
)

__all__ = ["SwarmInstaller", "InstallationResult"]
