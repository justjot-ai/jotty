"""
SwarmEnsemble - Extracted from SwarmManager
=============================================

Multi-perspective ensemble execution and auto-ensemble detection.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def should_auto_ensemble(goal: str) -> bool:
    """
    Determine if ensemble should be auto-enabled based on task type.

    BE CONSERVATIVE - ensemble adds significant latency (4x LLM calls).
    Only enable for comparison and decision tasks.
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
            return False

    # Comparison indicators (STRONG signal)
    comparison_keywords = [
        ' vs ', ' versus ', 'compare ',
        'difference between', 'differences between',
        'pros and cons', 'advantages and disadvantages',
    ]

    # Decision indicators (STRONG signal)
    decision_keywords = [
        'should i ', 'should we ',
        'which is better', 'what is best',
        'choose between', 'decide between',
    ]

    for keyword in comparison_keywords:
        if keyword in goal_lower:
            logger.debug(f"Auto-ensemble triggered by comparison: {keyword}")
            return True

    for keyword in decision_keywords:
        if keyword in goal_lower:
            logger.debug(f"Auto-ensemble triggered by decision: {keyword}")
            return True

    return False
