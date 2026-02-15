"""
DEPRECATED: Use Cases moved to core/modes/use_cases/
============================================================

Use cases have been reorganized following clean architecture principles.

OLD LOCATION (Deprecated):
    Jotty/core/interface/use_cases/

NEW LOCATION:
    Jotty/core/modes/use_cases/

WHY THIS CHANGE:
- All execution logic now in core/modes/ (Layer 2)
- core/interface/ is thin API layer (Layer 3)
- Follows clean architecture separation of concerns

MIGRATION:
    # OLD (deprecated)
    from Jotty.core.interface.use_cases import ChatUseCase, WorkflowUseCase
    from Jotty.core.interface.use_cases.chat import ChatExecutor

    # NEW (correct)
    from Jotty.core.modes.use_cases import ChatUseCase, WorkflowUseCase
    from Jotty.core.modes.use_cases.chat import ChatExecutor

This backward compatibility shim will be removed in a future release.
"""

import warnings
import sys


def __getattr__(name):
    """
    Redirect imports to new location with deprecation warning.

    Provides backward compatibility while warning users to update.
    """
    warnings.warn(
        f"\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"⚠️  DEPRECATED: Jotty.core.interface.use_cases has moved!\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"\n"
        f"OLD: from Jotty.core.interface.use_cases.{name}\n"
        f"NEW: from Jotty.core.modes.use_cases.{name}\n"
        f"\n"
        f"Use cases moved to modes/ for clean architecture.\n"
        f"Please update your imports.\n"
        f"\n"
        f"See LAYER3_ANALYSIS.md for details.\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n",
        DeprecationWarning,
        stacklevel=2
    )

    # Try to import from new location
    try:
        import importlib
        new_module = importlib.import_module(f"Jotty.core.modes.use_cases.{name}")
        return new_module
    except ImportError:
        # If not a module, try to import the attribute
        try:
            new_module = importlib.import_module("Jotty.core.modes.use_cases")
            return getattr(new_module, name)
        except (ImportError, AttributeError):
            raise ImportError(
                f"Cannot import '{name}' from either old or new location. "
                f"Please check that Jotty.core.modes.use_cases is properly installed."
            )


# For direct imports like "import Jotty.core.interface.use_cases"
warnings.warn(
    "\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "⚠️  DEPRECATED: use_cases moved to core/modes/use_cases/\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "\n"
    "Use cases have been reorganized for clean architecture.\n"
    "\n"
    "Update your imports:\n"
    "  OLD: from Jotty.core.interface.use_cases import ...\n"
    "  NEW: from Jotty.core.modes.use_cases import ...\n"
    "\n"
    "This follows clean architecture - all execution in core/modes/.\n"
    "\n"
    "See LAYER3_ANALYSIS.md for details.\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n",
    DeprecationWarning,
    stacklevel=2
)
