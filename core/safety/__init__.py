"""
Jotty Safety & Compliance Layer
================================

Provides safety constraints and validation for production deployments.

Modules:
- validators: Safety constraint definitions (PII, cost, quality, etc.)
- validator_agent: Validator agent for pre/post execution checks
- red_team: Ethical red-teaming for bias detection
"""

from .validators import (
    SafetyConstraint,
    PIIConstraint,
    CostBudgetConstraint,
    QualityThresholdConstraint,
    RateLimitConstraint,
    MaliciousInputConstraint,
    ValidationResult,
    ValidationReport
)

from .validator_agent import ValidatorAgent

from .red_team import (
    EthicalRedTeam,
    BiasReport,
    FairnessAudit
)

__all__ = [
    # Validators
    'SafetyConstraint',
    'PIIConstraint',
    'CostBudgetConstraint',
    'QualityThresholdConstraint',
    'RateLimitConstraint',
    'MaliciousInputConstraint',
    'ValidationResult',
    'ValidationReport',
    'ValidatorAgent',
    # Red Team
    'EthicalRedTeam',
    'BiasReport',
    'FairnessAudit'
]
