"""
Jotty Safety & Compliance Layer
================================

Provides safety constraints and validation for production deployments.

Modules:
- validators: Safety constraint definitions (PII, cost, quality, etc.)
- validator_agent: Validator agent for pre/post execution checks
- red_team: Ethical red-teaming for bias detection
"""

from .red_team import BiasReport, EthicalRedTeam, FairnessAudit
from .validator_agent import ValidatorAgent
from .validators import (
    CostBudgetConstraint,
    MaliciousInputConstraint,
    PIIConstraint,
    QualityThresholdConstraint,
    RateLimitConstraint,
    SafetyConstraint,
    ValidationReport,
    ValidationResult,
)

__all__ = [
    # Validators
    "SafetyConstraint",
    "PIIConstraint",
    "CostBudgetConstraint",
    "QualityThresholdConstraint",
    "RateLimitConstraint",
    "MaliciousInputConstraint",
    "ValidationResult",
    "ValidationReport",
    "ValidatorAgent",
    # Red Team
    "EthicalRedTeam",
    "BiasReport",
    "FairnessAudit",
    # Adaptive Thresholds
    "AdaptiveThresholdManager",
    "ThresholdHistory",
    "get_adaptive_threshold_manager",
]

from .adaptive_thresholds import (
    AdaptiveThresholdManager,
    ThresholdHistory,
    get_adaptive_threshold_manager,
)
