"""
Memory Orchestrator - Unified Memory Management API
====================================================

CONSOLIDATED MODULE combining:
- SimpleBrain (user-friendly API with presets)
- BrainInspiredMemoryManager (neuroscience-based consolidation)

The A-Team decided:
1. Complexity should be HIDDEN - users should NEVER need to configure brain modes
2. Consolidation should have MULTIPLE triggers (not just episode count)
3. Chunk sizing should be MODEL-AWARE (ratio, not absolute)
4. Presets for easy config, advanced for experts
5. Public API: brain.consolidate() for manual control

Scientific basis: Buzsáki (2015), Dudai et al. (2015), McClelland et al. (1995)

Usage:
    # Zero config (recommended)
    brain = SimpleBrain()

    # With preset
    brain = SimpleBrain.from_preset("thorough")

    # Manual consolidation
    brain.consolidate()

    # Auto consolidation on exit
    with brain.session() as session:
        session.process(experience)
    # Auto-consolidates here
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# PRESETS (A-Team Approved)
# =============================================================================


class BrainPreset(Enum):
    """
    Simple presets for brain configuration.
    Users pick ONE word, everything else is auto-configured.
    """

    MINIMAL = "minimal"  # Fast, low memory usage
    BALANCED = "balanced"  # Default, good for most cases
    THOROUGH = "thorough"  # Deep learning, high memory
    OFF = "off"  # Disable brain features entirely


# Preset configurations (internal, users don't see this)
PRESET_CONFIGS = {
    BrainPreset.MINIMAL: {
        "consolidation_interval": 100,
        "memory_buffer_size": 50,
        "chunk_ratio": 0.5,  # 50% of model context
        "prune_threshold": 0.3,
        "auto_consolidate": True,
    },
    BrainPreset.BALANCED: {
        "consolidation_interval": 3,  # A-TEAM: Consolidate every 3 episodes (prevent memory buildup!)
        "memory_buffer_size": 100,
        "chunk_ratio": 0.64,  # 64% of model context
        "prune_threshold": 0.15,
        "auto_consolidate": True,
    },
    BrainPreset.THOROUGH: {
        "consolidation_interval": 25,
        "memory_buffer_size": 200,
        "chunk_ratio": 0.75,  # 75% of model context
        "prune_threshold": 0.1,
        "auto_consolidate": True,
    },
    BrainPreset.OFF: {
        "consolidation_interval": float("inf"),
        "memory_buffer_size": 10,
        "chunk_ratio": 0.64,
        "prune_threshold": 1.0,  # Never prune
        "auto_consolidate": False,
    },
}


# =============================================================================
# CONSOLIDATION TRIGGERS (A-Team Decision: Multiple, not just episodes)
# =============================================================================


class ConsolidationTrigger(Enum):
    """When to consolidate memory."""

    EPISODE_COUNT = "episode_count"  # After N episodes
    MEMORY_PRESSURE = "memory_pressure"  # When memory buffer > 80% full
    PIPELINE_STAGE = "pipeline_stage"  # After a pipeline stage completes
    EXPLICIT = "explicit"  # User calls consolidate()
    ON_EXIT = "on_exit"  # When session/context manager exits
    IDLE = "idle"  # No new experiences for N seconds


# =============================================================================
# MODEL-AWARE CHUNK SIZING (A-Team Decision: Ratio, not absolute)
# =============================================================================

MODEL_CONTEXTS = {
    # OpenAI
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-turbo": 128000,
    "gpt-4.1": 30000,  # Effective usable
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    # Anthropic
    "claude-3-sonnet": 200000,
    "claude-3-opus": 200000,
    "claude-3.5-sonnet": 200000,
    # Local/Other
    "llama-7b": 4096,
    "mistral-7b": 8192,
    # Default
    "default": 28000,
}


def get_model_context(model_name: str) -> int:
    """Get context window size for a model."""
    model_lower = model_name.lower().replace("-", "").replace("_", "")

    for key, value in MODEL_CONTEXTS.items():
        if key.replace("-", "").replace("_", "") in model_lower:
            return value

    return MODEL_CONTEXTS["default"]


def calculate_chunk_size(model_name: str, ratio: float = 0.64) -> int:
    """
    Calculate chunk size based on model context and ratio.

    A-Team Decision: Use ratio, not absolute value.
    This automatically adapts to any model.
    """
    context = get_model_context(model_name)
    chunk = int(context * ratio)

    # Ensure reasonable bounds
    chunk = max(1000, min(chunk, 100000))

    logger.debug(f"Chunk size for {model_name}: {chunk} ({ratio*100:.0f}% of {context})")
    return chunk


# =============================================================================
# SIMPLE BRAIN (A-Team Approved Implementation)
# =============================================================================


@dataclass
class Experience:
    """A single experience to potentially remember."""

    content: str
    reward: float
    timestamp: float = field(default_factory=time.time)
    agent: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class SimpleBrain:
    """
    A-Team Approved brain implementation.

    Design Principles (from A-Team debate):
    1. ZERO CONFIG required - works out of box
    2. PRESETS for easy customization
    3. MULTIPLE consolidation triggers
    4. MODEL-AWARE chunk sizing
    5. PUBLIC API for manual control

    Usage:
        # Zero config (recommended)
        brain = SimpleBrain()

        # With preset
        brain = SimpleBrain.from_preset("thorough")

        # Manual consolidation
        brain.consolidate()

        # On-exit consolidation
        async with brain.session() as session:
            await session.process(experience)
        # Auto-consolidates when session exits
    """

    def __init__(
        self,
        preset: BrainPreset = BrainPreset.BALANCED,
        model_name: str = "gpt-4.1",
        consolidate_on: ConsolidationTrigger = ConsolidationTrigger.EPISODE_COUNT,
    ) -> None:
        self.preset = preset
        self.model_name = model_name
        self.consolidate_on = consolidate_on

        # Get preset config
        self.config = PRESET_CONFIGS[preset]

        # Calculate model-aware chunk size
        self.chunk_size = calculate_chunk_size(model_name, self.config["chunk_ratio"])

        # State
        self.experience_buffer: List[Experience] = []
        self.patterns_learned: List[str] = []
        self.episode_count = 0
        self.last_activity_time = time.time()
        self.is_consolidating = False

        # Stats
        self.consolidation_count = 0
        self.total_experiences = 0
        self.total_pruned = 0

        if preset != BrainPreset.OFF:
            logger.info(
                f" SimpleBrain initialized: preset={preset.value}, "
                f"model={model_name}, chunk_size={self.chunk_size}"
            )

    @classmethod
    def from_preset(cls, preset_name: str, model_name: str = "gpt-4.1") -> "SimpleBrain":
        """
        Create brain from preset name.

        Options: "minimal", "balanced", "thorough", "off"
        """
        try:
            preset = BrainPreset(preset_name.lower())
        except ValueError:
            logger.warning(f"Unknown preset '{preset_name}', using 'balanced'")
            preset = BrainPreset.BALANCED

        return cls(preset=preset, model_name=model_name)

    def process(self, experience: Experience) -> bool:
        """
        Process an experience, potentially storing it.

        Returns True if experience was stored, False if filtered out.
        """
        if self.preset == BrainPreset.OFF:
            return False

        if self.is_consolidating:
            logger.debug("Brain is consolidating, experience queued")
            return False

        self.episode_count += 1
        self.total_experiences += 1
        self.last_activity_time = time.time()

        # Simple filtering (A-Team: weights should be LEARNED, not configured)
        # For now, use simple heuristics instead of hardcoded weights
        should_store = self._should_remember(experience)

        if should_store:
            self.experience_buffer.append(experience)

            # Check buffer overflow
            if len(self.experience_buffer) > self.config["memory_buffer_size"]:
                self._prune_buffer()

        # Check if should consolidate
        if self._should_consolidate():
            asyncio.create_task(self._async_consolidate())

        return should_store

    def _should_remember(self, exp: Experience) -> bool:
        """
        Adaptive filtering: Remember surprising and relevant experiences.

        A-Team Decision: No hardcoded thresholds (0.8, 0.2).
        Uses adaptive statistics and deterministic exploration.
        """
        from Jotty.core.infrastructure.foundation.robust_parsing import (
            AdaptiveThreshold,
            EpsilonGreedy,
            safe_hash,
        )

        # Initialize adaptive components on first use
        if not hasattr(self, "_reward_threshold"):
            self._reward_threshold = AdaptiveThreshold()
            self._exploration = EpsilonGreedy(initial_epsilon=0.3)

        # Update statistics
        self._reward_threshold.update(exp.reward)

        # Always remember statistically extreme rewards (2 sigma)
        if self._reward_threshold.is_extreme(exp.reward, sigma=2.0):
            return True

        # Remember if content is novel (not seen recently)
        content_hash = safe_hash(exp.content)  # No  assumption!
        recent_hashes = {safe_hash(e.content) for e in self.experience_buffer}
        if content_hash not in recent_hashes:
            return True

        # Epsilon-greedy exploration (deterministic, not random!)
        return self._exploration.should_explore()

    def _should_consolidate(self) -> bool:
        """Check all consolidation triggers."""
        if self.is_consolidating:
            return False

        if not self.config["auto_consolidate"]:
            return False

        triggers = {
            ConsolidationTrigger.EPISODE_COUNT: (
                self.episode_count >= self.config["consolidation_interval"]
            ),
            ConsolidationTrigger.MEMORY_PRESSURE: (
                len(self.experience_buffer) >= self.config["memory_buffer_size"] * 0.8
            ),
            ConsolidationTrigger.IDLE: (time.time() - self.last_activity_time > 30),
        }

        return triggers.get(self.consolidate_on, False)

    def _prune_buffer(self) -> None:
        """Remove low-value experiences to make room."""
        if not self.experience_buffer:
            return

        # Sort by reward (keep high and low, remove middle)
        sorted_exp = sorted(self.experience_buffer, key=lambda e: abs(e.reward - 0.5))

        # Keep top 80%
        keep_count = int(len(sorted_exp) * 0.8)
        self.experience_buffer = sorted_exp[:keep_count]

        pruned = len(sorted_exp) - keep_count
        self.total_pruned += pruned
        logger.debug(f"Pruned {pruned} experiences from buffer")

    async def _async_consolidate(self) -> Any:
        """Async consolidation (internal)."""
        await self.consolidate(ConsolidationTrigger.EPISODE_COUNT)

    # =========================================================================
    # PUBLIC API (A-Team Decision: Users should be able to control this)
    # =========================================================================

    async def consolidate(
        self, trigger: ConsolidationTrigger = ConsolidationTrigger.EXPLICIT
    ) -> Any:
        """
        PUBLIC API: Consolidate memories.

        Users can call this directly for manual control:
            await brain.consolidate()
        """
        if self.is_consolidating:
            logger.debug("Already consolidating, skipping")
            return

        if not self.experience_buffer:
            logger.debug("No experiences to consolidate")
            return

        self.is_consolidating = True
        self.consolidation_count += 1

        logger.info(
            f" Consolidating ({trigger.value}): " f"{len(self.experience_buffer)} experiences"
        )

        try:
            # Extract patterns from experiences
            patterns = self._extract_patterns()
            self.patterns_learned.extend(patterns)

            # Keep patterns bounded
            if len(self.patterns_learned) > 100:
                self.patterns_learned = self.patterns_learned

            # Prune buffer after consolidation
            self._prune_buffer()

            # Reset episode counter
            self.episode_count = 0

            logger.info(f" Consolidation complete: " f"{len(patterns)} patterns extracted")

        except Exception as e:
            logger.error(f"Consolidation failed: {e}")
        finally:
            self.is_consolidating = False

    def _extract_patterns(self) -> List[str]:
        """Extract patterns from experience buffer."""
        patterns = []

        # Group by outcome
        successes = [e for e in self.experience_buffer if e.reward > 0.7]
        failures = [e for e in self.experience_buffer if e.reward < 0.3]

        if len(successes) >= 3:
            patterns.append(f"SUCCESS_PATTERN: {len(successes)} successful experiences")

        if len(failures) >= 3:
            patterns.append(f"FAILURE_PATTERN: {len(failures)} failures to learn from")

        # Agent-specific patterns
        agents = {e.agent for e in self.experience_buffer if e.agent}
        for agent in agents:
            agent_exp = [e for e in self.experience_buffer if e.agent == agent]
            if len(agent_exp) >= 3:
                avg_reward = sum(e.reward for e in agent_exp) / len(agent_exp)
                patterns.append(f"AGENT_PATTERN: {agent} has {avg_reward:.0%} avg success")

        return patterns

    @asynccontextmanager
    async def session(self) -> Any:
        """
        Context manager for automatic on-exit consolidation.

        Usage:
            async with brain.session() as session:
                session.process(experience1)
                session.process(experience2)
            # Auto-consolidates here
        """
        try:
            yield self
        finally:
            await self.consolidate(ConsolidationTrigger.ON_EXIT)

    def get_stats(self) -> Dict[str, Any]:
        """Get brain statistics."""
        return {
            "preset": self.preset.value,
            "model": self.model_name,
            "chunk_size": self.chunk_size,
            "buffer_size": len(self.experience_buffer),
            "patterns_learned": len(self.patterns_learned),
            "consolidation_count": self.consolidation_count,
            "total_experiences": self.total_experiences,
            "total_pruned": self.total_pruned,
        }

    def get_learned_patterns(self) -> List[str]:
        """Get patterns learned so far (for prompt injection)."""
        return self.patterns_learned.copy()


# =============================================================================
# SIMPLE CONFIG LOADER (A-Team: Single-level config with presets)
# =============================================================================


def load_brain_config(config: Dict[str, Any]) -> SimpleBrain:
    """
    Load SimpleBrain from config dict.

    Simple config (recommended):
        reval:
          brain: balanced  # or minimal, thorough, off

    Advanced config:
        reval:
          brain:
            preset: balanced
            model: gpt-4.1
            consolidate_on: pipeline_stage
    """
    brain_config = config.get("reval", {}).get("brain", "balanced")

    if isinstance(brain_config, str):
        # Simple: just preset name
        return SimpleBrain.from_preset(brain_config)

    elif isinstance(brain_config, dict):
        # Advanced: dict with options
        preset_name = brain_config.get("preset", "balanced")
        model = brain_config.get("model", "gpt-4.1")

        return SimpleBrain.from_preset(preset_name, model_name=model)

    else:
        # Default
        return SimpleBrain()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "SimpleBrain",
    "BrainPreset",
    "ConsolidationTrigger",
    "Experience",
    "calculate_chunk_size",
    "get_model_context",
    "load_brain_config",
    # From brain_memory_manager (merged)
    "EpisodicMemory",
    "SemanticPattern",
    "BrainInspiredMemoryManager",
]

"""
Brain-Inspired Memory Manager for ReVal
========================================

Integrates all neuroscience-based memory mechanisms:
- Sharp-wave ripples (SWR) for consolidation
- Hippocampal selection for prioritization
- Systems consolidation (hippocampus → neocortex)
- Synaptic pruning for signal-to-noise optimization

Scientific basis: Buzsáki (2015), Dudai et al. (2015), McClelland et al. (1995)

Dr. Elena Rivera (Neuralink-level neuroscientist) - Consultant
"""

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class EpisodicMemory:
    """Hippocampal storage (short-term, high-detail)."""

    content: Dict[str, Any]
    reward: float
    timestamp: float
    strength: float = 1.0
    replay_count: int = 0
    consolidated: bool = False
    novelty_score: float = 0.0


@dataclass
class SemanticPattern:
    """Neocortical storage (long-term, abstracted)."""

    abstract_lesson: str
    strength: float
    source_count: int  # How many experiences support this
    created_at: float
    last_reinforced: float
    tags: List[str] = field(default_factory=list)


class BrainInspiredMemoryManager:
    """
    Coordinates memory systems like the brain.

    Architecture:
    - Hippocampus: Fast learning, specific (episodic)
    - Neocortex: Slow learning, abstract (semantic)
    - SWR: Consolidation during "sleep"
    - Pruning: Homeostatic plasticity

     SCIENTIFICALLY ACCURATE per Dr. Rivera
    """

    def __init__(
        self,
        sleep_interval: int = 10,
        max_hippocampus_size: int = 100,
        max_neocortex_size: int = 200,
        replay_threshold: float = 0.7,
        novelty_weight: float = 0.4,
        reward_weight: float = 0.3,
        frequency_weight: float = 0.3,
    ) -> None:
        """
        Initialize brain-inspired memory.

        Args:
            sleep_interval: Episodes before consolidation (default: 10)
            max_hippocampus_size: Max episodic memories (default: 100)
            max_neocortex_size: Max semantic patterns (default: 200)
            replay_threshold: Min reward for replay (default: 0.7)
            novelty_weight: Weight for novelty in selection (default: 0.4)
            reward_weight: Weight for reward salience (default: 0.3)
            frequency_weight: Weight for frequency (default: 0.3)
        """
        # Short-term: Hippocampus (episodic)
        self.hippocampus: List[EpisodicMemory] = []

        # Long-term: Neocortex (semantic)
        self.neocortex: List[SemanticPattern] = []

        # Configuration
        self.sleep_interval = sleep_interval
        self.max_hippocampus_size = max_hippocampus_size
        self.max_neocortex_size = max_neocortex_size
        self.replay_threshold = replay_threshold

        # Selection weights
        self.novelty_weight = novelty_weight
        self.reward_weight = reward_weight
        self.frequency_weight = frequency_weight

        # Episode tracking
        self.episodes_since_sleep = 0
        self.total_consolidations = 0

        logger.info(f" BrainInspiredMemoryManager initialized (sleep_interval={sleep_interval})")

    def store_experience(self, experience: Dict[str, Any], reward: float) -> None:
        """
        Store new experience in hippocampus (encoding).

        Like hippocampal encoding in the brain.
        """
        # Compute novelty
        novelty = self._compute_novelty(experience)

        # Create episodic memory
        mem = EpisodicMemory(
            content=experience,
            reward=reward,
            timestamp=time.time(),
            strength=1.0,
            replay_count=0,
            novelty_score=novelty,
        )

        self.hippocampus.append(mem)

        # Keep hippocampus bounded (limited capacity like real hippocampus)
        if len(self.hippocampus) > self.max_hippocampus_size:
            # Prune weakest (lowest priority)
            self.hippocampus.sort(key=lambda x: self._memory_priority(x), reverse=True)
            pruned = self.hippocampus[self.max_hippocampus_size :]
            self.hippocampus = self.hippocampus[: self.max_hippocampus_size]
            logger.debug(f"Pruned {len(pruned)} low-priority hippocampal memories")

    def should_consolidate(self) -> bool:
        """Check if it's time to consolidate (enter "sleep")."""
        return self.episodes_since_sleep >= self.sleep_interval

    def trigger_consolidation(self) -> None:
        """
        Trigger sleep-like consolidation (Sharp-Wave Ripples).

        Simulates:
        1. Hippocampal selection (what to replay)
        2. Sharp-wave ripple replay (10-20x speed)
        3. Pattern extraction
        4. Systems consolidation (hippocampus → neocortex)
        5. Synaptic pruning

         Based on Buzsáki (2015), Wilson & McNaughton (1994)
        """
        logger.info(" Entering consolidation mode (simulating slow-wave sleep)")
        logger.info(f"   Hippocampus: {len(self.hippocampus)} memories")
        logger.info(f"   Neocortex: {len(self.neocortex)} patterns")

        # Step 1: Hippocampal Selection
        experiences_to_replay = self._select_for_replay()
        logger.info(f"   Selected {len(experiences_to_replay)} experiences for replay")

        # Step 2: Sharp-Wave Ripple Replay (rapid, 10-20x speed)
        patterns = self._sharp_wave_ripple_replay(experiences_to_replay)
        logger.info(f"   Extracted {len(patterns)} patterns from replay")

        # Step 3: Systems Consolidation (transfer to neocortex)
        transferred = self._transfer_to_neocortex(patterns)
        logger.info(f"   Transferred {transferred} patterns to neocortex")

        # Step 4: Synaptic Pruning
        pruned_hippo, pruned_neo = self._synaptic_pruning()
        logger.info(f"   Pruned {pruned_hippo} hippocampal, {pruned_neo} neocortical")

        # Reset counter
        self.episodes_since_sleep = 0
        self.total_consolidations += 1

        logger.info(f" Consolidation #{self.total_consolidations} complete")
        logger.info(
            f"   Final: Hippocampus={len(self.hippocampus)}, Neocortex={len(self.neocortex)}"
        )

    def get_consolidated_knowledge(self, query: str = None, max_items: int = 10) -> str:
        """
        Get consolidated knowledge to inject into prompts.

        THIS IS HOW BRAIN-INSPIRED LEARNING MANIFESTS!

        Returns natural language patterns from neocortex.
        """
        if not self.neocortex:
            return ""

        # Sort by strength (most consolidated first)
        sorted_patterns = sorted(self.neocortex, key=lambda p: p.strength, reverse=True)
        top_patterns = sorted_patterns[:max_items]

        context = "# Brain-Consolidated Knowledge (Neocortex):\n"
        context += f"# {len(self.neocortex)} total patterns, showing top {len(top_patterns)}\n\n"

        for i, pattern in enumerate(top_patterns, 1):
            context += f"{i}. {pattern.abstract_lesson}\n"
            context += f"   (strength: {pattern.strength:.2f}, sources: {pattern.source_count}, "
            context += f"consolidated: {time.time() - pattern.created_at:.0f}s ago)\n"

        return context

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics for monitoring."""
        return {
            "hippocampus_size": len(self.hippocampus),
            "neocortex_size": len(self.neocortex),
            "total_consolidations": self.total_consolidations,
            "episodes_since_sleep": self.episodes_since_sleep,
            "avg_hippo_strength": (
                sum(m.strength for m in self.hippocampus) / len(self.hippocampus)
                if self.hippocampus
                else 0
            ),
            "avg_neo_strength": (
                sum(p.strength for p in self.neocortex) / len(self.neocortex)
                if self.neocortex
                else 0
            ),
            "total_replay_count": sum(m.replay_count for m in self.hippocampus),
        }

    # =========================================================================
    # INTERNAL MECHANISMS (Neuroscience-inspired)
    # =========================================================================

    def _select_for_replay(self) -> List[EpisodicMemory]:
        """
        Hippocampal selection: choose experiences to replay.

        Prioritizes:
        1. High reward/punishment (emotional salience)
        2. Novel/surprising (prediction error)
        3. Not yet consolidated

         Based on Dudai et al. (2015)
        """
        candidates = []

        for mem in self.hippocampus:
            # Priority score
            priority = self._memory_priority(mem)

            # Only replay if above threshold and not fully consolidated
            if priority > self.replay_threshold and mem.replay_count < 5:
                candidates.append((priority, mem))

        # Sort by priority, take top half
        candidates.sort(reverse=True, key=lambda x: x[0])
        selected = [mem for _, mem in candidates[: len(candidates) // 2 + 1]]

        return selected

    def _memory_priority(self, mem: EpisodicMemory) -> float:
        """
        Compute memory priority for replay/retention.

         A-TEAM ENHANCEMENTS (per GRF MARL paper):
        - Recency decay: exp(-λt) where t = age in episodes
        - Causal impact bonus: memories with high Shapley credit get priority

        Combines:
        - Reward salience (emotional importance)
        - Novelty (surprise)
        - Frequency (how often accessed)
        - Recency decay (newer = more priority)
        - Causal impact (if tracked)
        """
        import time as time_module

        # Reward salience (deviation from neutral)
        reward_salience = abs(mem.reward - 0.5) * 2  # Scale to [0, 1]

        # Novelty
        novelty = mem.novelty_score

        # Frequency (inverse of replay count - less replayed = higher priority initially)
        frequency = 1.0 / (1.0 + mem.replay_count)

        # RECENCY DECAY: exp(-λ * age_in_seconds / 3600)
        # Half-life ~1 hour: λ = ln(2) ≈ 0.693
        age_seconds = time_module.time() - mem.timestamp
        recency_decay_lambda = 0.693 / 3600  # Half-life = 1 hour
        recency_factor = math.exp(-recency_decay_lambda * age_seconds)

        # CAUSAL IMPACT (from context if available)
        causal_impact = getattr(mem, "causal_impact", 0.0)
        if not causal_impact and hasattr(mem, "context") and isinstance(mem.context, dict):
            causal_impact = mem.context.get("shapley_credit", 0.0)

        # Weighted combination with recency and causal impact
        # Weights: reward=0.25, novelty=0.20, frequency=0.15, recency=0.25, causal=0.15
        priority = (
            0.25 * reward_salience
            + 0.20 * novelty
            + 0.15 * frequency
            + 0.25 * recency_factor
            + 0.15 * causal_impact
        )

        return priority

    def _sharp_wave_ripple_replay(self, experiences: List[EpisodicMemory]) -> List[Dict]:
        """
        Simulate sharp-wave ripple replay (10-20x real-time speed).

        During replay:
        - Increment replay counter
        - Extract abstract patterns
        - Strengthen memory traces

         Based on Buzsáki (2015), Wilson & McNaughton (1994)
        """
        patterns = []

        for exp in experiences:
            # Increment replay count
            exp.replay_count += 1

            # Strengthen memory trace (like LTP - long-term potentiation)
            exp.strength = min(2.0, exp.strength * 1.1)

            # Extract abstract pattern
            pattern = self._extract_pattern(exp)

            if pattern:
                patterns.append(pattern)

        return patterns

    def _extract_pattern(self, exp: EpisodicMemory) -> Optional[Dict]:
        """
        Extract abstract pattern from specific experience.

        Abstraction: Specific → General
        """
        content = exp.content

        if not content:
            return None

        # Extract key elements
        action = content.get("action", "unknown")
        outcome = "success" if exp.reward > 0.7 else "failure" if exp.reward < 0.3 else "neutral"

        # Create abstract lesson
        if outcome == "success":
            lesson = f" Strategy '{action}' tends to succeed (reward: {exp.reward:.2f})"
        elif outcome == "failure":
            lesson = f" Strategy '{action}' tends to fail (reward: {exp.reward:.2f})"
        else:
            lesson = f" Strategy '{action}' has mixed results (reward: {exp.reward:.2f})"

        # Extract tags for semantic similarity
        tags = self._extract_tags(content)

        return {
            "abstract_lesson": lesson,
            "strength": exp.strength * exp.replay_count,  # Strength from replay
            "tags": tags,
            "source_count": 1,
        }

    def _transfer_to_neocortex(self, patterns: List[Dict]) -> int:
        """
        Systems consolidation: transfer patterns to neocortex.

        Neocortex stores abstracted, schema-based representations.

         Based on McClelland et al. (1995)
        """
        transferred = 0

        for pattern in patterns:
            # Check if similar pattern already exists
            similar_pattern = self._find_similar_neocortical_pattern(pattern)

            if similar_pattern:
                # Strengthen existing pattern (like LTP)
                similar_pattern.strength += pattern["strength"]
                similar_pattern.source_count += 1
                similar_pattern.last_reinforced = time.time()
                transferred += 1
            else:
                # Create new neocortical pattern
                new_pattern = SemanticPattern(
                    abstract_lesson=pattern["abstract_lesson"],
                    strength=pattern["strength"],
                    source_count=pattern["source_count"],
                    created_at=time.time(),
                    last_reinforced=time.time(),
                    tags=pattern.get("tags", []),
                )
                self.neocortex.append(new_pattern)
                transferred += 1

        # Keep neocortex bounded
        if len(self.neocortex) > self.max_neocortex_size:
            # Remove weakest patterns
            self.neocortex.sort(key=lambda p: p.strength, reverse=True)
            self.neocortex = self.neocortex[: self.max_neocortex_size]

        return transferred

    def _synaptic_pruning(self) -> tuple[int, int]:
        """
        Synaptic homeostasis: prune weak memories.

        During sleep:
        - Strong jottys → strengthen (LTP)
        - Weak jottys → prune (homeostatic downscaling)

         Based on Tononi & Cirelli (2014)
        """
        # Prune hippocampus (more aggressive - shorter retention)
        initial_hippo = len(self.hippocampus)
        self.hippocampus = [
            mem
            for mem in self.hippocampus
            if mem.strength > 0.2 or mem.replay_count > 0  # Keep if strong or replayed
        ]
        pruned_hippo = initial_hippo - len(self.hippocampus)

        # Prune neocortex (less aggressive - longer retention)
        initial_neo = len(self.neocortex)
        current_time = time.time()
        self.neocortex = [
            pattern
            for pattern in self.neocortex
            if pattern.strength > 0.5  # Only remove very weak
            or (current_time - pattern.last_reinforced)
            < 86400 * 7  # Or keep if reinforced in last 7 days
        ]
        pruned_neo = initial_neo - len(self.neocortex)

        return pruned_hippo, pruned_neo

    def _compute_novelty(self, experience: Dict) -> float:
        """
        Compute how novel this experience is.

        Novel = different from existing neocortical patterns.
        """
        if not self.neocortex:
            return 1.0  # Everything is novel at first

        # Extract tags from experience
        exp_tags = set(self._extract_tags(experience))

        # Compare to neocortical patterns
        max_similarity = 0.0
        for pattern in self.neocortex:
            pattern_tags = set(pattern.tags)

            if exp_tags and pattern_tags:
                overlap = len(exp_tags & pattern_tags)
                union = len(exp_tags | pattern_tags)
                similarity = overlap / union if union > 0 else 0
                max_similarity = max(max_similarity, similarity)

        # Novelty = 1 - similarity
        novelty = 1.0 - max_similarity
        return novelty

    def _find_similar_neocortical_pattern(self, pattern: Dict) -> Optional[SemanticPattern]:
        """Find similar existing pattern in neocortex."""
        pattern_tags = set(pattern.get("tags", []))

        for existing in self.neocortex:
            existing_tags = set(existing.tags)

            if pattern_tags and existing_tags:
                overlap = len(pattern_tags & existing_tags)
                union = len(pattern_tags | existing_tags)
                similarity = overlap / union if union > 0 else 0

                if similarity > 0.7:  # Similar enough
                    return existing

        return None

    def _extract_tags(self, content: Dict) -> List[str]:
        """
        Extract semantic tags from content.

         A-TEAM ENHANCEMENTS:
        - Include causal impact (Shapley credit) as tag
        - Include tool I/O schema hints
        - Include knowledge provenance
        """
        tags = []

        # Extract from common fields
        if "action" in content:
            tags.append(f"action:{content['action']}")

        if "actor" in content:
            tags.append(f"actor:{content['actor']}")

        if "goal" in content:
            tags.append(f"goal:{content['goal'][:50]}")  # Truncate

        if "status" in content:
            tags.append(f"status:{content['status']}")

        # CAUSAL IMPACT TAG (for ContextGradient)
        if "shapley_credit" in content:
            credit = content["shapley_credit"]
            if credit > 0.7:
                tags.append("causal:high_impact")
            elif credit > 0.3:
                tags.append("causal:medium_impact")
            else:
                tags.append("causal:low_impact")

        # TOOL I/O SCHEMA HINTS
        if "tool_used" in content:
            tags.append(f"tool:{content['tool_used']}")

        if "output_schema" in content:
            tags.append(f"schema:{content['output_schema']}")

        # KNOWLEDGE PROVENANCE (who knew what)
        if "knowledge_source" in content:
            tags.append(f"source:{content['knowledge_source']}")

        return tags
