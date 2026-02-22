"""
SwarmLearner - Online Prompt Learning for V2
=============================================

Extracted from conductor.py for v2 independence.

SOTA: Treats prompt updates as weight updates.
- Learns from each episode
- Updates prompts with new patterns
- Accumulates wisdom over time
- Like fine-tuning but at prompt level
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

# Check for dspy availability
try:
    import dspy

    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    dspy = None


if DSPY_AVAILABLE:

    class SwarmLearnerSignature(dspy.Signature):
        """Update system prompts based on episode outcomes (online learning)."""

        current_prompt = dspy.InputField(desc="Current Architect/Auditor prompt")
        episode_trajectory = dspy.InputField(desc="What happened this episode")
        outcome = dspy.InputField(desc="Success/failure and why")
        patterns_observed = dspy.InputField(desc="Patterns that led to success/failure")

        updated_prompt = dspy.OutputField(desc="Updated prompt incorporating learnings")
        changes_made = dspy.OutputField(desc="List of specific changes made")
        learning_summary = dspy.OutputField(desc="What the system learned")

else:
    SwarmLearnerSignature = None  # type: ignore[assignment, misc]


class SwarmLearner:
    """
    SOTA: Treats prompt updates as weight updates.

    Instead of static prompts:
    - Learns from each episode
    - Updates prompts with new patterns
    - Accumulates wisdom over time
    - Like fine-tuning but at prompt level
    - Persists evolved prompts to disk for cross-session learning
    """

    _PERSIST_DIR = Path.home() / "jotty" / "learning"
    _PERSIST_FILE = "evolved_prompts.json"

    def __init__(self, config: Any) -> None:
        """
        Initialize SwarmLearner.

        Args:
            config: SwarmConfig with policy_update_threshold
        """
        self.config = config
        self.learner = (
            dspy.ChainOfThought(SwarmLearnerSignature)
            if DSPY_AVAILABLE and SwarmLearnerSignature  # type: ignore[truthy-function]
            else None
        )

        # Learning state
        self.learned_patterns: List[Dict] = []
        self.prompt_versions: Dict[str, List[str]] = {}  # prompt_name -> versions
        self.update_threshold = getattr(config, "policy_update_threshold", 3)
        self.episode_buffer: List[Dict] = []

        # Load persisted prompt evolution from disk
        self._load_persisted_prompts()

    def record_episode(self, trajectory: List[Dict], outcome: bool, insights: List[str]) -> Any:
        """
        Record episode for learning.

        Args:
            trajectory: List of episode steps
            outcome: Whether episode succeeded
            insights: List of insights from episode
        """
        self.episode_buffer.append(
            {
                "trajectory": trajectory,
                "outcome": outcome,
                "insights": insights,
                "timestamp": time.time(),
            }
        )

        # Extract patterns
        pattern_type = "success" if outcome else "failure"
        self.learned_patterns.append(
            {
                "type": pattern_type,
                "pattern": self._extract_pattern(trajectory),
                "timestamp": time.time(),
            }
        )

    def should_update_prompts(self) -> bool:
        """Check if we should update prompts."""
        return len(self.episode_buffer) >= self.update_threshold

    def update_prompt(self, prompt_name: str, current_prompt: str) -> Tuple[str, List[str]]:
        """
        Update a prompt based on learned patterns.

        Args:
            prompt_name: Name of prompt being updated
            current_prompt: Current prompt text

        Returns:
            (updated_prompt, changes_made)
        """
        if not self.learner or not self.episode_buffer:
            return current_prompt, []

        # Summarize episodes
        successes = [e for e in self.episode_buffer if e["outcome"]]
        failures = [e for e in self.episode_buffer if not e["outcome"]]

        patterns = []
        for p in self.learned_patterns[-20:]:
            patterns.append(f"{p['type']}: {p['pattern']}")

        try:
            result = self.learner(
                current_prompt=current_prompt[:3000],
                episode_trajectory=json.dumps(
                    {
                        "success_count": len(successes),
                        "failure_count": len(failures),
                        "recent_trajectories": [
                            e["trajectory"][:3] for e in self.episode_buffer[-3:]
                        ],
                    },
                    default=str,
                )[:1500],
                outcome=f"Successes: {len(successes)}, Failures: {len(failures)}",
                patterns_observed="\n".join(patterns[-10:])[:1000],
            )

            updated = result.updated_prompt or current_prompt
            changes = result.changes_made.split("\n") if result.changes_made else []

            # Track versions
            if prompt_name not in self.prompt_versions:
                self.prompt_versions[prompt_name] = []
            self.prompt_versions[prompt_name].append(updated)

            # Persist to disk for cross-session learning
            self._save_persisted_prompts()

            # Clear buffer after update
            self.episode_buffer = []

            logger.info(f"Prompt '{prompt_name}' updated with {len(changes)} changes")
            return updated, changes

        except Exception as e:
            logger.warning(f"Prompt update failed: {e}")
            return current_prompt, []

    def _extract_pattern(self, trajectory: List) -> str:
        """Extract pattern from trajectory (handles dicts, strings, ints, etc.)."""
        steps = []
        for step in trajectory[:5]:
            if isinstance(step, dict):
                steps.append(str(step.get("step", "unknown")))
            else:
                steps.append(str(step))
        return " -> ".join(steps)

    def get_learned_patterns(self) -> List[Dict]:
        """Get all learned patterns."""
        return self.learned_patterns

    def get_prompt_history(self, prompt_name: str) -> List[str]:
        """Get version history for a prompt."""
        return self.prompt_versions.get(prompt_name, [])

    def get_latest_prompt(self, prompt_name: str) -> str:
        """Get the latest evolved prompt (or empty string if none)."""
        versions = self.prompt_versions.get(prompt_name, [])
        return versions[-1] if versions else ""

    def clear_buffer(self) -> None:
        """Clear episode buffer without updating prompts."""
        self.episode_buffer = []

    # ─── Persistence ──────────────────────────────────────────────────────

    def _persist_path(self) -> Path:
        self._PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        return self._PERSIST_DIR / self._PERSIST_FILE

    def _load_persisted_prompts(self) -> None:
        """Load prompt versions and patterns from disk."""
        path = self._persist_path()
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            self.prompt_versions = data.get("prompt_versions", {})
            self.learned_patterns = data.get("learned_patterns", [])[-100:]
            logger.info(
                "Loaded %d prompt versions, %d patterns from disk",
                sum(len(v) for v in self.prompt_versions.values()),
                len(self.learned_patterns),
            )
        except Exception as exc:
            logger.warning("Failed to load persisted prompts: %s", exc)

    def _save_persisted_prompts(self) -> None:
        """Write prompt versions and patterns to disk."""
        try:
            path = self._persist_path()
            # Keep last 5 versions per prompt, last 100 patterns
            trimmed_versions = {k: v[-5:] for k, v in self.prompt_versions.items()}
            data = {
                "prompt_versions": trimmed_versions,
                "learned_patterns": self.learned_patterns[-100:],
                "updated_at": time.time(),
            }
            path.write_text(json.dumps(data, indent=2, default=str))
            logger.debug("Persisted prompt evolution to %s", path)
        except Exception as exc:
            logger.warning("Failed to persist prompts: %s", exc)


__all__ = ["SwarmLearner", "SwarmLearnerSignature"]
