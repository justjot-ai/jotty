"""
RL Learning Manager - DEPRECATED
================================

This module is deprecated. Use LearningCoordinator instead:

    from core.learning.learning_coordinator import LearningCoordinator

All classes are re-exported from learning_coordinator for backward compatibility.
"""

import warnings

warnings.warn(
    "core.orchestration.v2.rl_learning_manager is deprecated. "
    "Use core.learning.learning_coordinator instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from learning_coordinator for backward compatibility
from ...learning.learning_coordinator import (
    LearningCoordinator as LearningManager,
    LearningUpdate,
)

__all__ = [
    'LearningManager',
    'LearningUpdate',
]
