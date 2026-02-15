"""
Learning Manager - DEPRECATED
=============================

This module is deprecated. Use LearningManager instead:

    from core.learning.learning_coordinator import LearningManager

All classes and functions are re-exported from learning_coordinator for
backward compatibility.
"""

import warnings

warnings.warn(
    "core.learning.learning_manager is deprecated. "
    "Use core.learning.learning_coordinator instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from learning_coordinator for backward compatibility
from .learning_coordinator import (
    LearningManager,
    LearningSession,
    LearningUpdate,
    get_learning_coordinator,
)

__all__ = [
    "LearningManager",
    "LearningSession",
    "LearningUpdate",
    "get_learning_coordinator",
]
