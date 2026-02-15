"""
Robust Parsing - A-Team Approved Generic Parsing
================================================

A-Team Decision: NO REGEX for LLM output parsing.

This module provides robust parsing that works with any LLM output format:
1. Tries JSON parsing first
2. Falls back to structured extraction
3. Returns None instead of magic numbers on failure
4. Never uses regex for value extraction

Generic enough for any agentic system.
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

logger = logging.getLogger(__name__)


def parse_float_robust(value: Any, default: Optional[float] = None) -> Optional[float]:
    """
    Robustly parse a float from any input.

    A-Team Approved: No regex, handles all edge cases.

    Works with:
    - float/int: 0.7, 1
    - string: "0.7", "0.7%", "70%", "approximately 0.7"
    - dict: {"value": 0.7}
    - None/empty: returns default

    Returns None on failure (not magic number).
    """
    if value is None:
        return default

    # Already a number
    if isinstance(value, (int, float)):
        return float(value)

    # Dict with value key
    if isinstance(value, dict):
        for key in ["value", "score", "q_value", "confidence", "result"]:
            if key in value:
                return parse_float_robust(value[key], default)
        return default

    # String parsing
    if isinstance(value, str):
        value = value.strip()

        if not value:
            return default

        # Try direct float conversion
        try:
            return float(value)
        except ValueError:
            pass

        # Handle percentage
        if value.endswith("%"):
            try:
                return float(value[:-1]) / 100.0
            except ValueError:
                pass

        # Try JSON parsing
        try:
            parsed = json.loads(value)
            return parse_float_robust(parsed, default)
        except (json.JSONDecodeError, ValueError):
            pass

        # Extract first number-like substring (no regex!)
        # Walk through string finding digits and decimal points
        num_str = ""
        in_number = False
        has_decimal = False

        for char in value:
            if char.isdigit():
                num_str += char
                in_number = True
            elif char == "." and in_number and not has_decimal:
                num_str += char
                has_decimal = True
            elif in_number:
                # End of number
                break

        if num_str:
            try:
                result = float(num_str)
                # Normalize if looks like percentage
                if result > 1.0 and "percent" in value.lower():
                    result /= 100.0
                return result
            except ValueError:
                pass

    return default


def parse_bool_robust(value: Any, default: bool = False) -> bool:
    """
    Robustly parse a boolean from any input.

    Works with:
    - bool: True, False
    - string: "true", "yes", "1", "proceed", "valid"
    - int: 0, 1
    - None: returns default
    """
    if value is None:
        return default

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return value > 0

    if isinstance(value, str):
        value_lower = value.lower().strip()

        positive = {"true", "yes", "1", "proceed", "valid", "accept", "approved", "pass"}
        negative = {"false", "no", "0", "block", "invalid", "reject", "denied", "fail"}

        if value_lower in positive:
            return True
        if value_lower in negative:
            return False

    return default


def parse_json_robust(value: Any) -> Optional[Dict]:
    """
    Robustly parse JSON from any input.

    Works with:
    - dict: returns as-is
    - string: tries JSON parsing
    - string with markdown code blocks: extracts JSON
    """
    if value is None:
        return None

    if isinstance(value, dict):
        return value

    if isinstance(value, str):
        value = value.strip()

        # Try direct parsing
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code blocks (no regex!)
        if "```" in value:
            # Find content between ``` markers
            parts = value.split("```")
            for i, part in enumerate(parts):
                if i % 2 == 1:  # Odd indices are inside code blocks
                    # Remove language tag if present
                    lines = part.strip().split("\n")
                    if lines and lines[0] in ["json", "JSON", ""]:
                        content = "\n".join(lines[1:])
                    else:
                        content = part.strip()

                    try:
                        return json.loads(content)
                    except json.JSONDecodeError:
                        continue

        # Try finding JSON object in string (no regex!)
        # Look for {...} pattern
        start = value.find("{")
        if start >= 0:
            # Find matching closing brace
            depth = 0
            end = start
            for i, char in enumerate(value[start:], start):
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break

            if end > start:
                try:
                    return json.loads(value[start:end])
                except json.JSONDecodeError:
                    pass

    return None


class AdaptiveThreshold:
    """
    Adaptive threshold that learns from data.

    A-Team Approved: No hardcoded thresholds like 0.8 or 0.2.
    Uses running statistics to determine what's "high" or "low".
    """

    def __init__(self, initial_mean: float = 0.5, initial_std: float = 0.2) -> None:
        self.mean = initial_mean
        self.std = initial_std
        self.count = 0
        self.m2 = 0  # For Welford's algorithm

    def update(self, value: float) -> None:
        """Update running statistics (Welford's algorithm)."""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

        if self.count > 1:
            self.std = (self.m2 / (self.count - 1)) ** 0.5

    def is_high(self, value: float, sigma: float = 1.5) -> bool:
        """Check if value is significantly above mean."""
        return value > self.mean + sigma * max(self.std, 0.1)

    def is_low(self, value: float, sigma: float = 1.5) -> bool:
        """Check if value is significantly below mean."""
        return value < self.mean - sigma * max(self.std, 0.1)

    def is_extreme(self, value: float, sigma: float = 2.0) -> bool:
        """Check if value is extreme (either direction)."""
        deviation = abs(value - self.mean)
        return deviation > sigma * max(self.std, 0.1)


class EpsilonGreedy:
    """
    Deterministic epsilon-greedy decision making.

    A-Team Approved: No random.random() > 0.5 fallbacks.
    """

    def __init__(
        self, initial_epsilon: float = 0.3, decay: float = 0.99, min_epsilon: float = 0.05
    ) -> None:
        self.epsilon = initial_epsilon
        self.decay = decay
        self.min_epsilon = min_epsilon
        self.decision_count = 0

    def should_explore(self) -> bool:
        """
        Deterministic exploration based on decision count.

        Returns True for exploration with frequency epsilon.
        NOT random - uses hash of decision count for determinism.
        """
        self.decision_count += 1

        # Use hash for deterministic pseudo-randomness
        hash_val = hash(self.decision_count) % 1000 / 1000.0

        should_explore = hash_val < self.epsilon

        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)

        return should_explore

    def decide(self, exploit_decision: bool) -> bool:
        """
        Make decision: explore (True) or exploit (use exploit_decision).
        """
        if self.should_explore():
            return True  # Explore: try new thing
        return exploit_decision  # Exploit: use learned strategy


def safe_hash(content: Any, max_length: Optional[int] = None) -> int:
    """
    Safe hash that handles any input.

    A-Team Approved: No hash(content) assumptions.
    """
    if content is None:
        return 0

    # Convert to string if needed
    if not isinstance(content, str):
        content = str(content)

    # Truncate if needed, but handle edge cases
    if max_length and len(content) > max_length:
        content = content[:max_length]

    return hash(content)


def content_similarity(content1: Any, content2: Any, threshold: float = 0.8) -> bool:
    """
    Simple content similarity check without external dependencies.

    Uses character overlap, not regex.
    """
    if content1 is None or content2 is None:
        return content1 is content2

    s1 = str(content1).lower()
    s2 = str(content2).lower()

    if not s1 or not s2:
        return s1 == s2

    # Simple word overlap
    words1 = set(s1.split())
    words2 = set(s2.split())

    if not words1 or not words2:
        return s1 == s2

    intersection = len(words1 & words2)
    union = len(words1 | words2)

    jaccard = intersection / union if union > 0 else 0
    return jaccard >= threshold


# =============================================================================
# ADAPTIVE WEIGHT CLASSES (Real Learning, Not Text Logging)
# =============================================================================


@dataclass
class AdaptiveWeight:
    """
    Persistent adaptive weight that learns from feedback.

    A-Team v8.0: Real numeric weight updates, not text logging.
    Uses momentum-based gradient descent for stable learning.
    """

    name: str
    value: float = 0.3
    momentum: float = 0.0
    learning_rate: float = 0.01
    updates: int = 0

    def update(self, gradient: float, reward: float = 0.0) -> None:
        """
        Update weight using momentum-based gradient descent.

        Args:
            gradient: Direction and magnitude of update (-1 to 1)
            reward: Optional reward signal for adaptive learning rate
        """
        # Momentum update (smooths noisy gradients)
        self.momentum = 0.9 * self.momentum + 0.1 * gradient

        # Adaptive learning rate based on reward (higher reward = more confident update)
        effective_lr = self.learning_rate * (0.5 + 0.5 * max(0, reward))

        # Update value with clipping to [0.05, 0.95]
        self.value = max(0.05, min(0.95, self.value + effective_lr * self.momentum))
        self.updates += 1

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for persistence."""
        return {
            "name": self.name,
            "value": self.value,
            "momentum": self.momentum,
            "learning_rate": self.learning_rate,
            "updates": self.updates,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdaptiveWeight":
        """Deserialize from persistence."""
        return cls(
            name=data["name"],
            value=data.get("value", 0.3),
            momentum=data.get("momentum", 0.0),
            learning_rate=data.get("learning_rate", 0.01),
            updates=data.get("updates", 0),
        )


class AdaptiveWeightGroup:
    """
    Group of weights that sum to 1.0 (for credit assignment).

    A-Team v8.0: Replaces hardcoded weights like 0.3/0.4/0.3 with
    learned weights that adapt based on feedback.
    """

    def __init__(self, weights: Dict[str, float], learning_rate: float = 0.01) -> None:
        """
        Initialize with initial weights (will be normalized to sum=1.0).

        Args:
            weights: Dict of weight names to initial values
            learning_rate: Learning rate for all weights
        """
        self.learning_rate = learning_rate

        # Normalize and create adaptive weights
        total = sum(weights.values())
        if total <= 0:
            total = len(weights)
            weights = {k: 1.0 for k in weights}

        self._weights: Dict[str, AdaptiveWeight] = {}
        for name, value in weights.items():
            normalized = value / total
            self._weights[name] = AdaptiveWeight(
                name=name, value=normalized, learning_rate=learning_rate
            )

    def get(self, name: str) -> float:
        """Get current weight value."""
        if name in self._weights:
            return self._weights[name].value
        return 0.0

    def get_all(self) -> Dict[str, float]:
        """Get all weight values."""
        return {name: w.value for name, w in self._weights.items()}

    def update_from_feedback(self, name: str, gradient: float, reward: float = 0.0) -> None:
        """
        Update one weight and renormalize all to sum to 1.0.

        Args:
            name: Weight to update
            gradient: Positive = increase this weight's importance
            reward: Episode reward for adaptive learning rate
        """
        if name not in self._weights:
            return

        # Update the target weight
        self._weights[name].update(gradient, reward)

        # Renormalize all weights to sum to 1.0
        self._renormalize()

    def _renormalize(self) -> Any:
        """Ensure all weights sum to 1.0."""
        total = sum(w.value for w in self._weights.values())
        if total > 0:
            for w in self._weights.values():
                w.value = w.value / total

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for persistence."""
        return {
            "weights": {name: w.to_dict() for name, w in self._weights.items()},
            "learning_rate": self.learning_rate,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdaptiveWeightGroup":
        """Deserialize from persistence."""
        instance = cls.__new__(cls)
        instance.learning_rate = data.get("learning_rate", 0.01)
        instance._weights = {}

        for name, w_data in data.get("weights", {}).items():
            instance._weights[name] = AdaptiveWeight.from_dict(w_data)

        return instance

    def __repr__(self) -> str:
        weights_str = ", ".join(f"{n}={w.value:.3f}" for n, w in self._weights.items())
        return f"AdaptiveWeightGroup({weights_str})"


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "parse_float_robust",
    "parse_bool_robust",
    "parse_json_robust",
    "AdaptiveThreshold",
    "EpsilonGreedy",
    "safe_hash",
    "content_similarity",
    "AdaptiveWeight",
    "AdaptiveWeightGroup",
]
