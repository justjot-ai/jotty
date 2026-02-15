"""
Orchestration Prompts - Swarm Coordination & Validation
=========================================================

Provides access to swarm-level prompts for:
- Orchestration readiness (architect)
- Actor coordination (auditor)
- Goal alignment (auditor)
- Generic validation (fallback)

Usage:
    from Jotty.core.intelligence.orchestration.prompts import (
        get_swarm_architect_prompt,
        get_swarm_auditor_prompt,
        get_generic_auditor_prompt,
    )

    # Use in swarm orchestration
    architect_prompt = get_swarm_architect_prompt()
    auditor_prompt = get_swarm_auditor_prompt('coordination')
"""

from pathlib import Path
from typing import Literal

_PROMPTS_DIR = Path(__file__).parent


def get_swarm_architect_prompt() -> str:
    """
    Get swarm architect prompt for orchestration readiness validation.

    Validates:
    - Task clarity
    - Actor availability
    - Context sufficiency
    - Dependencies

    Returns:
        Markdown prompt text
    """
    prompt_file = _PROMPTS_DIR / "swarm" / "architect_orchestration.md"
    return prompt_file.read_text()


def get_swarm_auditor_prompt(
    auditor_type: Literal['coordination', 'goal_alignment'] = 'coordination'
) -> str:
    """
    Get swarm auditor prompt for actor coordination or goal alignment.

    Args:
        auditor_type:
            - 'coordination': Validates actor coordination and info flow
            - 'goal_alignment': Validates goal achievement and completeness

    Returns:
        Markdown prompt text
    """
    if auditor_type == 'coordination':
        prompt_file = _PROMPTS_DIR / "swarm" / "auditor_coordination.md"
    elif auditor_type == 'goal_alignment':
        prompt_file = _PROMPTS_DIR / "swarm" / "auditor_goal_alignment.md"
    else:
        raise ValueError(f"Invalid auditor_type: {auditor_type}")

    return prompt_file.read_text()


def get_generic_auditor_prompt() -> str:
    """
    Get generic auditor prompt for output validation.

    Handles:
    - Execution status validation
    - Error classification (infrastructure/logic/data)
    - Semantic validation
    - Edge cases (empty results, nulls, etc.)

    Returns validation status: pass/fail/external_error/enquiry

    Returns:
        Markdown prompt text
    """
    prompt_file = _PROMPTS_DIR / "validation" / "generic_auditor.md"
    return prompt_file.read_text()


__all__ = [
    'get_swarm_architect_prompt',
    'get_swarm_auditor_prompt',
    'get_generic_auditor_prompt',
]
