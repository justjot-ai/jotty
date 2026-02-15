"""
DEPRECATED: CLI has moved to apps/cli/
============================================

The Jotty CLI has been migrated to follow clean architecture principles.

OLD LOCATION (Deprecated):
    Jotty/core/interface/cli/

NEW LOCATION:
    Jotty/apps/cli/

WHY THIS CHANGE:
- Follows world-class architecture (Google, Amazon, Stripe, GitHub, etc.)
- CLI is an APPLICATION, not part of core framework
- Apps should use SDK, not import from core directly
- Enables proper "dogfooding" of the Jotty SDK

MIGRATION:
    # OLD (deprecated)
    from Jotty.core.interface.cli.app import JottyCLI

    # NEW (correct)
    from Jotty.apps.cli.app import JottyCLI

For more information, see:
    - Jotty/ARCHITECTURE_RECOMMENDATION.md
    - Jotty/ARCHITECTURE_WORLD_CLASS_EXAMPLES.md

This directory is kept for backward compatibility but will be removed
in a future release.
"""

import warnings
import sys


def __getattr__(name):
    """
    Redirect imports to new location with deprecation warning.

    This provides backward compatibility while warning users to update.
    """
    warnings.warn(
        f"\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"⚠️  DEPRECATED: Jotty.core.interface.cli has moved!\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"\n"
        f"OLD: from Jotty.core.interface.cli.{name}\n"
        f"NEW: from Jotty.apps.cli.{name}\n"
        f"\n"
        f"The CLI has been moved to apps/ to follow clean architecture.\n"
        f"Please update your imports.\n"
        f"\n"
        f"See Jotty/ARCHITECTURE_RECOMMENDATION.md for details.\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n",
        DeprecationWarning,
        stacklevel=2
    )

    # Try to import from new location
    try:
        import importlib
        new_module = importlib.import_module(f"Jotty.apps.cli.{name}")
        return new_module
    except ImportError:
        # If not a module, try to import the attribute
        try:
            new_module = importlib.import_module("Jotty.apps.cli")
            return getattr(new_module, name)
        except (ImportError, AttributeError):
            raise ImportError(
                f"Cannot import '{name}' from either old or new location. "
                f"Please check that Jotty.apps.cli is properly installed."
            )


# For direct imports like "import Jotty.core.interface.cli"
warnings.warn(
    "\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "⚠️  DEPRECATED: Jotty.core.interface.cli has moved to apps/cli/\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "\n"
    "The CLI has been migrated to follow clean architecture.\n"
    "\n"
    "Update your imports:\n"
    "  OLD: from Jotty.core.interface.cli import ...\n"
    "  NEW: from Jotty.apps.cli import ...\n"
    "\n"
    "This follows the same pattern used by Google, Amazon, Stripe,\n"
    "GitHub, and other world-class companies.\n"
    "\n"
    "See Jotty/ARCHITECTURE_RECOMMENDATION.md for details.\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n",
    DeprecationWarning,
    stacklevel=2
)
