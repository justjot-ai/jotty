"""
SwarmEnsemble - Extracted from SwarmManager
=============================================

Multi-perspective ensemble execution and auto-ensemble detection.

Optima-inspired (Chen et al., 2024): Adaptive communication efficiency.
Returns (should_ensemble, max_perspectives) to control cost vs. quality.
"""

import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)


def should_auto_ensemble(goal: str) -> Tuple[bool, int]:
    """
    Determine if ensemble should be auto-enabled and with how many perspectives.

    Optima-inspired adaptive sizing (Chen et al., 2024):
    - Simple comparison: 2 perspectives (fast, saves ~80s)
    - Complex analysis/decision: 4 perspectives (thorough)
    - Creation tasks: 0 (no ensemble)

    Returns:
        (should_ensemble, max_perspectives) tuple
    """
    goal_lower = goal.lower()

    # EXCLUSION: Don't auto-ensemble for creation/generation tasks
    creation_keywords = [
        'create ', 'generate ', 'write ', 'build ', 'make ',
        'checklist', 'template', 'document', 'report',
        'draft ', 'prepare ', 'compile ',
    ]
    for keyword in creation_keywords:
        if keyword in goal_lower:
            logger.debug(f"Auto-ensemble SKIPPED for creation task: {keyword}")
            return False, 0

    # Complex decision indicators → 4 perspectives (thorough)
    complex_keywords = [
        'should i ', 'should we ',
        'which is better', 'what is best',
        'choose between', 'decide between',
        'evaluate ', 'assess ', 'recommend',
    ]

    # Simple comparison indicators → 2 perspectives (fast)
    simple_comparison_keywords = [
        ' vs ', ' versus ', 'compare ',
        'difference between', 'differences between',
        'pros and cons', 'advantages and disadvantages',
    ]

    for keyword in complex_keywords:
        if keyword in goal_lower:
            logger.debug(f"Auto-ensemble: 4 perspectives (complex decision: {keyword})")
            return True, 4

    for keyword in simple_comparison_keywords:
        if keyword in goal_lower:
            logger.debug(f"Auto-ensemble: 2 perspectives (simple comparison: {keyword})")
            return True, 2

    return False, 0
