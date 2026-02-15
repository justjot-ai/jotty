"""
Adaptive Validator - Learn When to Skip Validation

Insight: High-confidence operations from proven agents don't need validation!

Current Jotty: Always runs Architect + Auditor (2 LLM calls per agent)
With Adaptive: Skip validation for well-behaved agents → 50-90% fewer validation calls

Features:
- Track validation outcomes per (agent, operation) pair
- Learn success rates over time
- Skip validation when success rate exceeds threshold
- Periodic sampling to catch regressions
"""

import logging
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .config import LotusConfig

logger = logging.getLogger(__name__)


@dataclass
class ValidationHistory:
    """Validation history for an (agent, operation) pair."""

    total_validations: int = 0
    successful_validations: int = 0
    failed_validations: int = 0
    skipped_validations: int = 0
    last_validation_time: float = 0.0
    last_failure_time: float = 0.0

    # Rolling window for recent performance
    recent_results: List[bool] = field(default_factory=list)
    max_recent: int = 50

    @property
    def success_rate(self) -> float:
        """Overall success rate."""
        if self.total_validations == 0:
            return 0.0
        return self.successful_validations / self.total_validations

    @property
    def recent_success_rate(self) -> float:
        """Recent success rate (rolling window)."""
        if not self.recent_results:
            return 0.0
        return sum(self.recent_results) / len(self.recent_results)

    def add_result(self, success: bool) -> None:
        """Record a validation result."""
        self.total_validations += 1
        self.last_validation_time = time.time()

        if success:
            self.successful_validations += 1
        else:
            self.failed_validations += 1
            self.last_failure_time = time.time()

        # Update rolling window
        self.recent_results.append(success)
        if len(self.recent_results) > self.max_recent:
            self.recent_results.pop(0)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_validations": self.total_validations,
            "successful_validations": self.successful_validations,
            "failed_validations": self.failed_validations,
            "skipped_validations": self.skipped_validations,
            "success_rate": self.success_rate,
            "recent_success_rate": self.recent_success_rate,
            "last_validation_time": self.last_validation_time,
            "last_failure_time": self.last_failure_time,
        }


@dataclass
class ValidationDecision:
    """Decision about whether to validate."""

    should_validate: bool
    reason: str
    confidence: float
    is_sample: bool = False  # True if this is a sampling check


class AdaptiveValidator:
    """
    Adaptive validation controller that learns when validation is needed.

    Tracks validation outcomes per (agent, operation) and skips validation
    when the combination has a high success rate.

    DRY: Reuses threshold from LotusConfig.

    Usage:
        validator = AdaptiveValidator(config)

        # Check if validation is needed
        decision = validator.should_validate("my_agent", "filter")

        if decision.should_validate:
            # Run full validation
            is_valid = await run_validation(result)
            validator.record_result("my_agent", "filter", is_valid)
        else:
            # Skip validation (trusted)
            validator.record_skip("my_agent", "filter")
    """

    def __init__(
        self,
        config: Optional[LotusConfig] = None,
        skip_threshold: float = 0.95,
        sample_rate: float = 0.1,
        min_samples: int = 10,
    ) -> None:
        """
        Initialize adaptive validator.

        Args:
            config: LOTUS configuration
            skip_threshold: Skip validation if success rate >= this
            sample_rate: Rate of validation sampling when skipped
            min_samples: Minimum validations before skipping allowed
        """
        self.config = config or LotusConfig()
        self.skip_threshold = skip_threshold or self.config.validation_skip_threshold
        self.sample_rate = sample_rate or self.config.validation_sample_rate
        self.min_samples = min_samples

        # History per (agent, operation) pair
        self._history: Dict[Tuple[str, str], ValidationHistory] = defaultdict(ValidationHistory)

        # Global stats
        self._total_decisions = 0
        self._total_skips = 0
        self._total_samples = 0

        logger.info(
            f"AdaptiveValidator initialized: skip_threshold={self.skip_threshold}, "
            f"sample_rate={self.sample_rate}, min_samples={self.min_samples}"
        )

    def should_validate(
        self,
        agent: str,
        operation: str,
        force: bool = False,
    ) -> ValidationDecision:
        """
        Decide whether to validate a result.

        Args:
            agent: Agent name
            operation: Operation type
            force: Force validation regardless of history

        Returns:
            ValidationDecision with should_validate and reason
        """
        self._total_decisions += 1
        key = (agent, operation)
        history = self._history[key]

        # Force validation if requested
        if force:
            return ValidationDecision(
                should_validate=True,
                reason="forced",
                confidence=1.0,
            )

        # Not enough samples yet
        if history.total_validations < self.min_samples:
            return ValidationDecision(
                should_validate=True,
                reason=f"insufficient_samples ({history.total_validations}/{self.min_samples})",
                confidence=0.0,
            )

        # Check recent success rate
        success_rate = history.recent_success_rate

        if success_rate >= self.skip_threshold:
            # High success rate - consider skipping

            # But sample occasionally to catch regressions
            if random.random() < self.sample_rate:
                self._total_samples += 1
                return ValidationDecision(
                    should_validate=True,
                    reason="sampling_check",
                    confidence=success_rate,
                    is_sample=True,
                )

            # Skip validation
            self._total_skips += 1
            return ValidationDecision(
                should_validate=False,
                reason=f"trusted (success_rate={success_rate:.1%})",
                confidence=success_rate,
            )

        # Recent failure or low success rate - validate
        time_since_failure = time.time() - history.last_failure_time
        if time_since_failure < 300:  # 5 minutes
            return ValidationDecision(
                should_validate=True,
                reason=f"recent_failure ({time_since_failure:.0f}s ago)",
                confidence=success_rate,
            )

        return ValidationDecision(
            should_validate=True,
            reason=f"low_success_rate ({success_rate:.1%})",
            confidence=success_rate,
        )

    def record_result(self, agent: str, operation: str, success: bool) -> Any:
        """
        Record a validation result.

        Args:
            agent: Agent name
            operation: Operation type
            success: Whether validation passed
        """
        key = (agent, operation)
        self._history[key].add_result(success)

        logger.debug(
            f"Validation recorded: {agent}/{operation} = {success} "
            f"(rate: {self._history[key].success_rate:.1%})"
        )

    def record_skip(self, agent: str, operation: str) -> Any:
        """
        Record a skipped validation.

        Args:
            agent: Agent name
            operation: Operation type
        """
        key = (agent, operation)
        self._history[key].skipped_validations += 1

    def get_history(
        self,
        agent: str,
        operation: str,
    ) -> ValidationHistory:
        """Get validation history for agent/operation pair."""
        return self._history[(agent, operation)]

    def get_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return {
            "total_decisions": self._total_decisions,
            "total_skips": self._total_skips,
            "total_samples": self._total_samples,
            "skip_rate": self._total_skips / max(self._total_decisions, 1),
            "agents_tracked": len(set(k[0] for k in self._history)),
            "operations_tracked": len(set(k[1] for k in self._history)),
            "history_by_agent": {
                k[0]: self._history[k].to_dict() for k in sorted(self._history.keys())
            },
        }

    def reset_agent(self, agent: str) -> None:
        """Reset history for an agent (e.g., after code change)."""
        keys_to_remove = [k for k in self._history if k[0] == agent]
        for key in keys_to_remove:
            del self._history[key]
        logger.info(f"Reset validation history for agent: {agent}")

    def get_trusted_agents(
        self,
        operation: Optional[str] = None,
    ) -> List[str]:
        """
        Get list of agents that can skip validation.

        Args:
            operation: Filter by operation type (optional)

        Returns:
            List of trusted agent names
        """
        trusted = []

        for (agent, op), history in self._history.items():
            if operation and op != operation:
                continue

            if history.total_validations >= self.min_samples:
                if history.recent_success_rate >= self.skip_threshold:
                    trusted.append(agent)

        return list(set(trusted))

    def export_history(self) -> Dict[str, Any]:
        """Export all history for persistence."""
        return {
            f"{agent}:{operation}": history.to_dict()
            for (agent, operation), history in self._history.items()
        }

    def import_history(self, data: Dict[str, Any]) -> None:
        """Import history from persistence."""
        for key_str, history_data in data.items():
            agent, operation = key_str.split(":", 1)
            history = ValidationHistory(
                total_validations=history_data.get("total_validations", 0),
                successful_validations=history_data.get("successful_validations", 0),
                failed_validations=history_data.get("failed_validations", 0),
                skipped_validations=history_data.get("skipped_validations", 0),
                last_validation_time=history_data.get("last_validation_time", 0),
                last_failure_time=history_data.get("last_failure_time", 0),
            )
            self._history[(agent, operation)] = history

        logger.info(f"Imported validation history for {len(data)} agent/operation pairs")
