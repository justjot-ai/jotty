"""
SwarmEnsemble - Extracted from Orchestrator
=============================================

Multi-perspective ensemble execution and auto-ensemble detection.

Optima-inspired (Chen et al., 2024): Adaptive communication efficiency.
Returns (should_ensemble, max_perspectives) to control cost vs. quality.
"""

import logging
from typing import Any, Dict, Tuple

logger = logging.getLogger(__name__)


def should_auto_ensemble(goal: str) -> Tuple[bool, int]:
    """
    Determine if ensemble should be auto-enabled and with how many perspectives.

    CONSERVATIVE: Only triggers for EXPLICIT debate/brainstorm/comparison tasks.
    Most tasks should NOT use ensemble — it burns 5-6 LLM calls and is slow
    and expensive, especially on rate-limited free providers.

    Ensemble is for:
    - Explicit "debate", "brainstorm", "devil's advocate" requests
    - Explicit "pros and cons" / "advantages and disadvantages" analysis
    - Multi-option decision-making ("should I X vs Y")

    Ensemble is NOT for:
    - Simple questions, lookups, summaries
    - Creation/generation tasks
    - Single-option evaluations ("evaluate X")
    - Research tasks

    Returns:
        (should_ensemble, max_perspectives) tuple
    """
    goal_lower = goal.lower()

    # Only trigger for EXPLICIT multi-perspective requests
    # These are phrases where the user is clearly asking for multiple viewpoints
    explicit_debate_keywords = [
        "debate ",
        "brainstorm",
        "devil's advocate",
        "devils advocate",
        "multiple perspectives",
        "multi-perspective",
        "pros and cons of",
        "advantages and disadvantages of",
    ]

    # Explicit comparison between two named alternatives
    explicit_comparison_keywords = [
        " vs ",
        " versus ",
        "choose between",
        "decide between",
    ]

    for keyword in explicit_debate_keywords:
        if keyword in goal_lower:
            logger.debug(f"Auto-ensemble: 4 perspectives (explicit debate: {keyword})")
            return True, 4

    for keyword in explicit_comparison_keywords:
        if keyword in goal_lower:
            logger.debug(f"Auto-ensemble: 2 perspectives (explicit comparison: {keyword})")
            return True, 2

    # Everything else: NO ensemble
    return False, 0
