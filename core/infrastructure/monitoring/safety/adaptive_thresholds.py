"""
Adaptive Safety Thresholds
===========================

Auto-tune safety thresholds based on historical data.

KISS PRINCIPLE: Uses simple percentiles (no complex ML models).
DRY PRINCIPLE: Reuses existing SafetyConstraint classes.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import statistics

logger = logging.getLogger(__name__)


@dataclass
class ThresholdHistory:
    """Historical data for a single threshold."""
    constraint_name: str
    current_threshold: float
    violations: List[float] = field(default_factory=list)
    non_violations: List[float] = field(default_factory=list)
    last_adjusted: Optional[str] = None
    adjustment_count: int = 0


class AdaptiveThresholdManager:
    """
    Auto-tune safety thresholds based on observed data.

    ALGORITHM (KISS - simple percentile):
    1. Collect violations and non-violations
    2. Every N observations, check if threshold is appropriate:
       - If 95th percentile of non-violations > threshold → too strict (raise)
       - If 5th percentile of violations < threshold → too loose (lower)
    3. Adjust by 10% increments (gradual, conservative)

    EXAMPLE:
    Cost threshold = $0.50
    Observed non-violations: [$0.05, $0.08, $0.10, $0.12, ..., $0.45]
    95th percentile = $0.42
    → Threshold is appropriate (42 < 50, no adjustment)

    Observed violations: [$0.55, $0.60, $0.70, $2.00]
    5th percentile = $0.55
    → Violations are close to threshold (55 vs 50)
    → Lower threshold to $0.48 (catch anomalies sooner)
    """

    def __init__(self, adaptation_interval: int = 100, percentile_margin: float = 0.1, max_adjustment: float = 0.2) -> None:
        """
        Args:
            adaptation_interval: Review thresholds every N observations
            percentile_margin: Safety margin for percentiles (10% = conservative)
            max_adjustment: Max threshold change per adjustment (20% max)
        """
        self.adaptation_interval = adaptation_interval
        self.percentile_margin = percentile_margin
        self.max_adjustment = max_adjustment

        self.history: Dict[str, ThresholdHistory] = {}
        self.observation_count = 0

        logger.info(
            f"🎯 AdaptiveThresholdManager initialized "
            f"(interval={adaptation_interval}, margin={percentile_margin:.0%})"
        )

    def record_observation(self, constraint_name: str, value: float, violated: bool, current_threshold: float) -> Any:
        """
        Record an observation for threshold adaptation.

        Args:
            constraint_name: Name of constraint (e.g., "cost_budget", "latency")
            value: Observed value
            violated: Whether this observation violated the threshold
            current_threshold: Current threshold value
        """
        # Initialize history if needed
        if constraint_name not in self.history:
            self.history[constraint_name] = ThresholdHistory(
                constraint_name=constraint_name,
                current_threshold=current_threshold,
            )

        hist = self.history[constraint_name]

        # Record observation
        if violated:
            hist.violations.append(value)
        else:
            hist.non_violations.append(value)

        self.observation_count += 1

        # Check if it's time to adapt
        if self.observation_count % self.adaptation_interval == 0:
            self._adapt_thresholds()

    def _adapt_thresholds(self) -> Any:
        """Review and adapt all thresholds based on accumulated data."""
        logger.info(f"\n📊 Reviewing thresholds ({self.observation_count} observations)...")

        for constraint_name, hist in self.history.items():
            if len(hist.non_violations) < 10:
                logger.debug(f"Skipping {constraint_name} (insufficient data)")
                continue

            # Calculate percentiles (KISS - use standard library)
            p95_non_violations = self._percentile(hist.non_violations, 95)

            # Check if threshold is too strict
            threshold = hist.current_threshold
            if p95_non_violations > threshold * (1 - self.percentile_margin):
                # 95% of valid requests are close to threshold → too strict
                # Raise threshold by 10%
                new_threshold = min(
                    threshold * (1 + self.max_adjustment),
                    threshold * 1.10
                )
                self._apply_adjustment(hist, new_threshold, "too_strict")

            # Check if threshold is too loose (if we have violations)
            elif len(hist.violations) >= 5:
                p5_violations = self._percentile(hist.violations, 5)
                if p5_violations < threshold * (1 + self.percentile_margin):
                    # Violations are close to threshold → too loose
                    # Lower threshold by 10%
                    new_threshold = max(
                        threshold * (1 - self.max_adjustment),
                        threshold * 0.90
                    )
                    self._apply_adjustment(hist, new_threshold, "too_loose")

            # Reset buffers (keep only recent history)
            hist.non_violations = hist.non_violations[-100:]
            hist.violations = hist.violations[-20:]

    def _apply_adjustment(self, hist: ThresholdHistory, new_threshold: float, reason: str) -> Any:
        """Apply threshold adjustment."""
        old = hist.current_threshold
        hist.current_threshold = new_threshold
        hist.last_adjusted = datetime.now().isoformat()
        hist.adjustment_count += 1

        logger.info(
            f"🔧 Adjusted {hist.constraint_name}: "
            f"{old:.3f} → {new_threshold:.3f} ({reason})"
        )

    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile (KISS - simple implementation)."""
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        idx = int(len(sorted_vals) * percentile / 100.0)
        return sorted_vals[min(idx, len(sorted_vals) - 1)]

    def get_threshold(self, constraint_name: str) -> Optional[float]:
        """Get current adaptive threshold for a constraint."""
        hist = self.history.get(constraint_name)
        return hist.current_threshold if hist else None

    def get_stats(self) -> Dict[str, Any]:
        """Get adaptation statistics."""
        return {
            'total_observations': self.observation_count,
            'constraints_tracked': len(self.history),
            'thresholds': {
                name: {
                    'current': hist.current_threshold,
                    'adjustments': hist.adjustment_count,
                    'last_adjusted': hist.last_adjusted,
                    'observations': len(hist.violations) + len(hist.non_violations),
                }
                for name, hist in self.history.items()
            }
        }


# Singleton
_adaptive_manager = None


def get_adaptive_threshold_manager() -> AdaptiveThresholdManager:
    """Get or create adaptive threshold manager singleton."""
    global _adaptive_manager
    if _adaptive_manager is None:
        _adaptive_manager = AdaptiveThresholdManager()
    return _adaptive_manager


__all__ = [
    'AdaptiveThresholdManager',
    'ThresholdHistory',
    'get_adaptive_threshold_manager',
]
