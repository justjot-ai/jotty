"""
Jotty v6.0 - Validation-Related Types
======================================

All validation-related dataclasses including validation results
and multi-round validation tracking.
Extracted from data_structures.py for better organization.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .enums import OutputTag, ValidationRound


# =============================================================================
# VALIDATION RESULTS (Enhanced)
# =============================================================================

@dataclass
class ValidationResult:
    """
    Enhanced validation result with reasoning trace.
    """
    agent_name: str
    is_valid: bool
    confidence: float
    reasoning: str

    # For Architect
    should_proceed: Optional[bool] = None
    injected_context: Optional[str] = None
    injected_instructions: Optional[str] = None

    # For Auditor
    output_tag: Optional[OutputTag] = None
    why_useful: Optional[str] = None

    # Execution info
    tool_calls: List[Dict] = field(default_factory=list)
    execution_time: float = 0.0

    # NEW: Reasoning quality for credit assignment
    reasoning_steps: List[str] = field(default_factory=list)
    reasoning_quality: float = 0.5  # How well-reasoned was the decision

    # NEW: Multi-round info
    validation_round: ValidationRound = ValidationRound.INITIAL
    previous_rounds: List['ValidationResult'] = field(default_factory=list)
