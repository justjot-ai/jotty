"""
Jotty v6.0 - Workflow-Related Types
====================================

All workflow-related dataclasses including rich observations
and execution context.
Extracted from data_structures.py for better organization.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


# =============================================================================
# RICH OBSERVATION (A-Team Enhancement)
# =============================================================================

@dataclass
class RichObservation:
    """
    A-Team Enhancement: Rich observation with linguistic context for LLM understanding.

    Instead of a simple string observation, this captures:
    - Natural language summary (for LLM comprehension)
    - State deltas (what changed)
    - Entities affected
    - Confidence signals
    - Anomalies detected

    Usage:
        obs = RichObservation(
            raw_result={"success": True, "rows": 100},
            natural_summary="Successfully mapped 100 rows to target schema",
            action_taken="Applied column mapping for 'bank_code'",
            outcome_type="success"
        )

        # For LLM context
        context = obs.to_linguistic_string()
    """

    # Core
    raw_result: Any = None

    # Linguistic (for LLM understanding)
    natural_summary: str = ""     # "The validation agent found 3 issues..."
    action_taken: str = ""        # "Validated column 'bank_code' against schema"
    outcome_type: str = "unknown" # "success", "partial", "failure", "unknown"

    # State delta
    state_before: Dict[str, Any] = field(default_factory=dict)
    state_after: Dict[str, Any] = field(default_factory=dict)
    delta_summary: str = ""       # "Added 100 rows, filled 3 columns"

    # Entities
    entities_affected: List[str] = field(default_factory=list)
    columns_touched: List[str] = field(default_factory=list)
    agents_involved: List[str] = field(default_factory=list)

    # Signals
    confidence_reason: str = ""
    anomalies: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # For learning
    should_remember: bool = True  # Can be set False for routine observations
    memory_level_hint: Optional[str] = None  # Hint for MemoryLevelClassifier

    def to_linguistic_string(self) -> str:
        """Convert to rich linguistic representation for LLM context."""
        parts = []

        if self.action_taken:
            parts.append(f"ACTION: {self.action_taken}")

        if self.outcome_type:
            parts.append(f"OUTCOME: {self.outcome_type.upper()}")

        if self.natural_summary:
            parts.append(f"SUMMARY: {self.natural_summary}")

        if self.delta_summary:
            parts.append(f"CHANGES: {self.delta_summary}")

        if self.entities_affected:
            parts.append(f"ENTITIES: {', '.join(self.entities_affected)}")

        if self.columns_touched:
            parts.append(f"COLUMNS: {', '.join(self.columns_touched)}")

        if self.anomalies:
            parts.append(f"⚠️ ANOMALIES: {'; '.join(self.anomalies)}")

        if self.warnings:
            parts.append(f"⚠️ WARNINGS: {'; '.join(self.warnings)}")

        if self.confidence_reason:
            parts.append(f"CONFIDENCE: {self.confidence_reason}")

        return "\n".join(parts) if parts else str(self.raw_result)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "natural_summary": self.natural_summary,
            "action_taken": self.action_taken,
            "outcome_type": self.outcome_type,
            "delta_summary": self.delta_summary,
            "entities_affected": self.entities_affected,
            "columns_touched": self.columns_touched,
            "anomalies": self.anomalies,
            "warnings": self.warnings,
            "confidence_reason": self.confidence_reason
        }

    @classmethod
    def from_simple(cls, result: Any, action: str, outcome: str) -> 'RichObservation':
        """Create a simple RichObservation from basic info."""
        return cls(
            raw_result=result,
            natural_summary=str(result) if result else "",
            action_taken=action,
            outcome_type=outcome
        )
