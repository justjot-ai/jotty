"""
Jotty v6.0 - Enumeration Types
==============================

All enum types used across the Jotty framework.
Extracted from data_structures.py for better organization.
"""

from enum import Enum, auto


class MemoryLevel(Enum):
    """Aristotelian knowledge hierarchy."""
    EPISODIC = "episodic"      # Raw experiences (Techne source)
    SEMANTIC = "semantic"      # Abstracted patterns (Episteme)
    PROCEDURAL = "procedural"  # Action sequences (Techne)
    META = "meta"              # Learning wisdom (Phronesis)
    CAUSAL = "causal"          # NEW: Why things work (Episteme+)


class OutputTag(Enum):
    """Post-validation output classification."""
    ENQUIRY = "enquiry"
    FAIL = "fail"
    USEFUL = "useful"
    PARTIAL = "partial"  # NEW: Partially correct


class AlertType(Enum):
    """Health monitoring alert types."""
    REWARD_HACKING = "reward_hacking"
    DISTRIBUTION_SHIFT = "distribution_shift"
    CONSERVATIVE_COLLAPSE = "conservative_collapse"
    FORGETTING = "forgetting"
    LEARNING_STALL = "learning_stall"  # NEW
    GOAL_DRIFT = "goal_drift"          # NEW


class CommunicationType(Enum):
    """NEW: Inter-agent communication types."""
    TOOL_RESULT = "tool_result"        # Share tool call results
    INSIGHT = "insight"                 # Share discovered insight
    WARNING = "warning"                 # Share concern
    REQUEST = "request"                 # Request info from other agent


class ValidationRound(Enum):
    """NEW: Multi-round validation phases."""
    INITIAL = "initial"
    REFINEMENT = "refinement"
    FINAL = "final"


class ContextType(Enum):
    """A-Team Enhancement: Context types for memory retrieval prioritization."""
    VALIDATION = "validation"     # Prefer PROCEDURAL, META
    DEBUGGING = "debugging"       # Prefer CAUSAL, EPISODIC
    PLANNING = "planning"         # Prefer META, SEMANTIC
    EXPLORATION = "exploration"   # Prefer EPISODIC, CAUSAL
    TRANSFORMATION = "transformation"  # Prefer PROCEDURAL, SEMANTIC
    DEFAULT = "default"           # Equal priority


class TaskStatus(Enum):
    """
    Task status enum - unified across all subsystems.

    Consolidated from roadmap.py, task.py, and workflow_context.py.
    Contains all status values from all three sources for backward compatibility.
    """
    SUGGESTED = "suggested"       # From task.py - task suggested but not yet planned
    BACKLOG = "backlog"           # From task.py - in backlog queue
    PENDING = "pending"           # Common - waiting to start
    IN_PROGRESS = "in_progress"   # Common - currently executing
    COMPLETED = "completed"       # Common - successfully finished
    FAILED = "failed"             # Common - execution failed
    BLOCKED = "blocked"           # Common - blocked by dependencies
    CANCELLED = "cancelled"       # From task.py - manually cancelled
    RETRYING = "retrying"         # From task.py - retry in progress
    SKIPPED = "skipped"           # From roadmap.py - skipped execution
