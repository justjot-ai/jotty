"""
Learning Utilities - Consolidated utility classes for Jotty learning
=====================================================================

This module provides a unified interface to learning utilities:

1. Pattern Extraction - Extract patterns from experiences/trajectories
2. Experience Buffers - Store and replay experiences
3. State Description - Convert states to natural language
4. Similarity Scoring - Compare states, actions, patterns

Import from here for clean access:

    from core.learning.utils import (
        PatternExtractor,
        PrioritizedEpisodeBuffer,
        StateDescriber,
        SimilarityEngine,
        ExperienceRecord,
    )
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# RE-EXPORTS FROM EXISTING MODULES (DRY - don't duplicate)
# =============================================================================

# PatternExtractor from transfer_learning
try:
    from .transfer_learning import PatternExtractor
except ImportError:
    PatternExtractor = None

# PrioritizedEpisodeBuffer from offline_learning
try:
    from .offline_learning import PrioritizedEpisodeBuffer
except ImportError:
    PrioritizedEpisodeBuffer = None

# SemanticEmbedder from transfer_learning (for similarity)
try:
    from .transfer_learning import SemanticEmbedder
except ImportError:
    SemanticEmbedder = None


# =============================================================================
# NEW UTILITIES
# =============================================================================


@dataclass
class ExperienceRecord:
    """
    Unified experience record for learning systems.

    Standard format used across all learning components.
    """

    state: Dict[str, Any]
    action: Dict[str, Any]
    reward: float
    next_state: Optional[Dict[str, Any]] = None
    done: bool = False
    timestamp: float = field(default_factory=time.time)
    agent: str = ""
    episode_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def experience_id(self) -> str:
        """Generate unique ID for this experience."""
        content = f"{self.state}:{self.action}:{self.timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "state": self.state,
            "action": self.action,
            "reward": self.reward,
            "next_state": self.next_state,
            "done": self.done,
            "timestamp": self.timestamp,
            "agent": self.agent,
            "episode_id": self.episode_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperienceRecord":
        """Create from dictionary."""
        return cls(
            state=data.get("state", {}),
            action=data.get("action", {}),
            reward=data.get("reward", 0.0),
            next_state=data.get("next_state"),
            done=data.get("done", False),
            timestamp=data.get("timestamp", time.time()),
            agent=data.get("agent", ""),
            episode_id=data.get("episode_id", ""),
            metadata=data.get("metadata", {}),
        )


class StateDescriber:
    """
    Converts structured states to natural language descriptions.

    Used to provide context to LLM agents about current state,
    enabling learning to manifest in prompts.

    Example:
        describer = StateDescriber()
        state = {'task': 'analyze', 'progress': 0.5, 'errors': 2}
        text = describer.describe(state)
        # "Current task: analyze (50% progress, 2 errors encountered)"
    """

    # Default templates for common state keys
    TEMPLATES = {
        "task": "Current task: {value}",
        "goal": "Goal: {value}",
        "progress": "Progress: {value:.0%}" if isinstance(0, float) else "Progress: {value}",
        "errors": "{value} error(s) encountered",
        "attempts": "{value} attempt(s) made",
        "agent": "Agent: {value}",
        "phase": "Phase: {value}",
        "status": "Status: {value}",
        "confidence": "Confidence: {value:.0%}" if isinstance(0, float) else "Confidence: {value}",
    }

    def __init__(self, custom_templates: Dict[str, str] = None) -> None:
        """
        Initialize describer.

        Args:
            custom_templates: Additional or override templates
        """
        self.templates = {**self.TEMPLATES}
        if custom_templates:
            self.templates.update(custom_templates)

    def describe(self, state: Dict[str, Any], include_keys: List[str] = None) -> str:
        """
        Convert state dict to natural language description.

        Args:
            state: State dictionary
            include_keys: If provided, only include these keys

        Returns:
            Natural language description
        """
        if not state:
            return "No state information available."

        parts = []
        keys = include_keys or state.keys()

        for key in keys:
            if key not in state:
                continue

            value = state[key]

            # Skip None values
            if value is None:
                continue

            # Use template if available
            if key in self.templates:
                template = self.templates[key]
                try:
                    if "{value:" in template:
                        # Format with value specifier
                        parts.append(template.format(value=value))
                    else:
                        parts.append(template.format(value=value))
                except (ValueError, KeyError):
                    parts.append(f"{key}: {value}")
            else:
                # Default formatting
                parts.append(f"{self._format_key(key)}: {self._format_value(value)}")

        return ". ".join(parts) if parts else "State: " + str(state)

    def describe_action(self, action: Dict[str, Any]) -> str:
        """Describe an action in natural language."""
        if not action:
            return "No action specified."

        parts = []

        if "actor" in action:
            parts.append(f"Agent '{action['actor']}'")

        if "type" in action:
            parts.append(f"performing {action['type']}")
        elif "task" in action:
            parts.append(f"executing {action['task']}")

        if "target" in action:
            parts.append(f"on {action['target']}")

        return " ".join(parts) if parts else f"Action: {action}"

    def describe_transition(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        next_state: Dict[str, Any],
        reward: float,
    ) -> str:
        """Describe a full state transition."""
        parts = [
            f"From: {self.describe(state)}",
            f"Action: {self.describe_action(action)}",
            f"To: {self.describe(next_state)}",
            f"Reward: {reward:+.2f}",
        ]
        return " | ".join(parts)

    def _format_key(self, key: str) -> str:
        """Format key for display."""
        return key.replace("_", " ").title()

    def _format_value(self, value: Any) -> str:
        """Format value for display."""
        if isinstance(value, float):
            if 0 <= value <= 1:
                return f"{value:.0%}"
            return f"{value:.2f}"
        elif isinstance(value, bool):
            return "Yes" if value else "No"
        elif isinstance(value, list):
            if len(value) <= 3:
                return ", ".join(str(v) for v in value)
            return f"{len(value)} items"
        elif isinstance(value, dict):
            return f"{len(value)} attributes"
        return str(value)


class SimilarityEngine:
    """
    Unified similarity scoring for states, actions, and patterns.

    Supports multiple backends:
    - Semantic embeddings (if available)
    - Keyword/topic overlap
    - Structural similarity

    Example:
        engine = SimilarityEngine()
        score = engine.similarity(state1, state2)
    """

    def __init__(self, use_embeddings: bool = True) -> None:
        """
        Initialize similarity engine.

        Args:
            use_embeddings: Whether to use semantic embeddings if available
        """
        self._embedder = None
        if use_embeddings and SemanticEmbedder:
            try:
                self._embedder = SemanticEmbedder(use_embeddings=True)
            except Exception as e:
                logger.debug(f"Could not initialize embedder: {e}")

    def similarity(self, item1: Any, item2: Any, method: str = "auto") -> float:
        """
        Compute similarity between two items.

        Args:
            item1: First item (dict, str, or list)
            item2: Second item
            method: 'auto', 'semantic', 'structural', or 'keyword'

        Returns:
            Similarity score between 0 and 1
        """
        if item1 == item2:
            return 1.0

        # Convert to strings if dicts
        if isinstance(item1, dict) and isinstance(item2, dict):
            if method == "structural":
                return self._structural_similarity(item1, item2)
            # For other methods, convert to string
            str1 = self._dict_to_string(item1)
            str2 = self._dict_to_string(item2)
        else:
            str1 = str(item1)
            str2 = str(item2)

        if method == "auto":
            if self._embedder:
                return self._semantic_similarity(str1, str2)
            return self._keyword_similarity(str1, str2)
        elif method == "semantic":
            if self._embedder:
                return self._semantic_similarity(str1, str2)
            logger.warning("Semantic similarity requested but embedder not available")
            return self._keyword_similarity(str1, str2)
        elif method == "keyword":
            return self._keyword_similarity(str1, str2)
        else:
            return self._keyword_similarity(str1, str2)

    def find_similar(
        self, query: Any, candidates: List[Any], threshold: float = 0.5, top_k: int = 5
    ) -> List[Tuple[Any, float]]:
        """
        Find similar items from candidates.

        Args:
            query: Query item
            candidates: List of candidate items
            threshold: Minimum similarity threshold
            top_k: Maximum results to return

        Returns:
            List of (item, score) tuples sorted by similarity
        """
        results = []
        for candidate in candidates:
            score = self.similarity(query, candidate)
            if score >= threshold:
                results.append((candidate, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity using embeddings."""
        if not self._embedder:
            return self._keyword_similarity(text1, text2)
        return self._embedder.similarity(text1, text2)

    def _keyword_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity based on keyword overlap."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        # Remove very short words
        words1 = {w for w in words1 if len(w) > 2}
        words2 = {w for w in words2 if len(w) > 2}

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union) if union else 0.0

    def _structural_similarity(self, dict1: Dict, dict2: Dict) -> float:
        """Compute structural similarity between dicts."""
        keys1 = set(dict1.keys())
        keys2 = set(dict2.keys())

        # Key overlap
        common_keys = keys1 & keys2
        all_keys = keys1 | keys2

        if not all_keys:
            return 1.0 if not keys1 and not keys2 else 0.0

        key_score = len(common_keys) / len(all_keys)

        # Value similarity for common keys
        value_scores = []
        for key in common_keys:
            v1, v2 = dict1[key], dict2[key]
            if v1 == v2:
                value_scores.append(1.0)
            elif type(v1) == type(v2):
                if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                    # Numeric similarity
                    max_val = max(abs(v1), abs(v2), 1)
                    value_scores.append(1 - abs(v1 - v2) / max_val)
                elif isinstance(v1, str):
                    value_scores.append(self._keyword_similarity(v1, v2))
                else:
                    value_scores.append(0.5)
            else:
                value_scores.append(0.0)

        value_score = sum(value_scores) / len(value_scores) if value_scores else 0.0

        return 0.5 * key_score + 0.5 * value_score

    def _dict_to_string(self, d: Dict) -> str:
        """Convert dict to string for text-based similarity."""
        parts = []
        for key, value in d.items():
            if isinstance(value, (dict, list)):
                parts.append(f"{key}")
            else:
                parts.append(f"{key} {value}")
        return " ".join(parts)


class SimpleExperienceBuffer:
    """
    Simple experience buffer for storing and replaying experiences.

    For prioritized replay, use PrioritizedEpisodeBuffer from offline_learning.
    This is a lightweight alternative for basic replay needs.
    """

    def __init__(self, capacity: int = 1000) -> None:
        """
        Initialize buffer.

        Args:
            capacity: Maximum experiences to store
        """
        self.capacity = capacity
        self.experiences: List[ExperienceRecord] = []

    def add(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        reward: float,
        next_state: Dict[str, Any] = None,
        done: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Add experience to buffer."""
        exp = ExperienceRecord(
            state=state, action=action, reward=reward, next_state=next_state, done=done, **kwargs
        )
        self.experiences.append(exp)

        # Enforce capacity (remove oldest)
        while len(self.experiences) > self.capacity:
            self.experiences.pop(0)

    def sample(self, batch_size: int) -> List[ExperienceRecord]:
        """Sample random batch of experiences."""
        import random

        n = min(batch_size, len(self.experiences))
        return random.sample(self.experiences, n) if n > 0 else []

    def get_recent(self, n: int) -> List[ExperienceRecord]:
        """Get most recent n experiences."""
        return self.experiences[-n:]

    def get_by_agent(self, agent: str) -> List[ExperienceRecord]:
        """Get all experiences for an agent."""
        return [e for e in self.experiences if e.agent == agent]

    def get_successful(self) -> List[ExperienceRecord]:
        """Get experiences with positive reward."""
        return [e for e in self.experiences if e.reward > 0]

    def get_failed(self) -> List[ExperienceRecord]:
        """Get experiences with negative reward."""
        return [e for e in self.experiences if e.reward <= 0]

    def clear(self) -> None:
        """Clear all experiences."""
        self.experiences = []

    def __len__(self) -> int:
        return len(self.experiences)

    def __iter__(self) -> Any:
        return iter(self.experiences)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Re-exported (from existing modules)
    "PatternExtractor",
    "PrioritizedEpisodeBuffer",
    "SemanticEmbedder",
    # New utilities
    "ExperienceRecord",
    "StateDescriber",
    "SimilarityEngine",
    "SimpleExperienceBuffer",
]
