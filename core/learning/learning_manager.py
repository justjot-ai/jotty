"""
Learning Manager - DEPRECATED
=============================

This module is deprecated. Use SwarmLearningManager instead:

    from core.learning.learning_coordinator import SwarmLearningManager

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
    SwarmLearningManager,
    LearningSession,
    LearningUpdate,
    get_learning_coordinator as get_learning_manager,
    reset_learning_coordinator as reset_learning_manager,
)

__all__ = [
    'SwarmLearningManager',
    'LearningSession',
    'LearningUpdate',
    'get_learning_manager',
    'reset_learning_manager',
]
