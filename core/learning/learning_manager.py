"""
Learning Manager - DEPRECATED
=============================

This module is deprecated. Use LearningCoordinator instead:

    from core.learning.learning_coordinator import LearningCoordinator

All classes and functions are re-exported from learning_coordinator for
backward compatibility.
"""

import warnings

warnings.warn(
    "core.learning.learning_manager is deprecated. "
    "Use core.learning.learning_coordinator instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from learning_coordinator for backward compatibility
from .learning_coordinator import (
    LearningCoordinator,
    LearningCoordinator as LearningManager,  # Alias
    LearningSession,
    LearningUpdate,
    get_learning_coordinator as get_learning_manager,
    reset_learning_coordinator as reset_learning_manager,
)

__all__ = [
    'LearningManager',
    'LearningSession',
    'LearningUpdate',
    'get_learning_manager',
    'reset_learning_manager',
]
