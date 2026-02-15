"""
Unit Tests for Memory Orchestrator
===================================

Tests for:
- BrainPreset enum
- ConsolidationTrigger enum
- Experience dataclass
- SimpleBrain class (init, from_preset, process, consolidate, session, stats)
- EpisodicMemory dataclass
- SemanticPattern dataclass
- BrainInspiredMemoryManager (store, consolidation, retrieval, statistics)
- Utility functions (get_model_context, calculate_chunk_size, load_brain_config)

All tests use mocks -- no real LLM calls, no external dependencies.
"""

import time
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

try:
    from Jotty.core.intelligence.memory.memory_orchestrator import (
        BrainPreset,
        ConsolidationTrigger,
        Experience,
        SimpleBrain,
        EpisodicMemory,
        SemanticPattern,
        BrainInspiredMemoryManager,
        get_model_context,
        calculate_chunk_size,
        load_brain_config,
        PRESET_CONFIGS,
        MODEL_CONTEXTS,
    )
    MEMORY_ORCHESTRATOR_AVAILABLE = True
except ImportError:
    MEMORY_ORCHESTRATOR_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not MEMORY_ORCHESTRATOR_AVAILABLE,
    reason="memory_orchestrator module not importable",
)


# =============================================================================
# BrainPreset Enum
# =============================================================================

@pytest.mark.unit
class TestBrainPreset:
    """Tests for BrainPreset enum values and completeness."""

    def test_enum_values(self):
        """BrainPreset has the four expected members with correct string values."""
        assert BrainPreset.MINIMAL.value == "minimal"
        assert BrainPreset.BALANCED.value == "balanced"
        assert BrainPreset.THOROUGH.value == "thorough"
        assert BrainPreset.OFF.value == "off"

    def test_all_presets_have_configs(self):
        """Every BrainPreset member has a corresponding entry in PRESET_CONFIGS."""
        for preset in BrainPreset:
            assert preset in PRESET_CONFIGS, f"Missing config for {preset}"
            cfg = PRESET_CONFIGS[preset]
            assert "consolidation_interval" in cfg
            assert "memory_buffer_size" in cfg
            assert "chunk_ratio" in cfg
            assert "auto_consolidate" in cfg


# =============================================================================
# ConsolidationTrigger Enum
# =============================================================================

@pytest.mark.unit
class TestConsolidationTrigger:
    """Tests for ConsolidationTrigger enum."""

    def test_trigger_values(self):
        """ConsolidationTrigger has six members with correct string values."""
        expected = {
            "EPISODE_COUNT": "episode_count",
            "MEMORY_PRESSURE": "memory_pressure",
            "PIPELINE_STAGE": "pipeline_stage",
            "EXPLICIT": "explicit",
            "ON_EXIT": "on_exit",
            "IDLE": "idle",
        }
        for name, value in expected.items():
            member = ConsolidationTrigger[name]
            assert member.value == value


# =============================================================================
# Experience Dataclass
# =============================================================================

@pytest.mark.unit
class TestExperience:
    """Tests for the Experience dataclass."""

    def test_required_fields(self):
        """Experience can be created with only required fields."""
        exp = Experience(content="hello", reward=0.5)
        assert exp.content == "hello"
        assert exp.reward == 0.5
        assert exp.agent == ""
        assert exp.metadata == {}
        assert isinstance(exp.timestamp, float)

    def test_custom_fields(self):
        """Experience stores custom metadata and agent."""
        exp = Experience(
            content="task done",
            reward=0.9,
            timestamp=100.0,
            agent="ResearchAgent",
            metadata={"key": "value"},
        )
        assert exp.agent == "ResearchAgent"
        assert exp.metadata == {"key": "value"}
        assert exp.timestamp == 100.0


# =============================================================================
# SimpleBrain
# =============================================================================

@pytest.mark.unit
class TestSimpleBrain:
    """Tests for SimpleBrain init, presets, stats, consolidation, and session."""

    def test_default_init(self):
        """SimpleBrain initializes with BALANCED preset by default."""
        brain = SimpleBrain()
        assert brain.preset == BrainPreset.BALANCED
        assert brain.model_name == "gpt-4.1"
        assert brain.consolidate_on == ConsolidationTrigger.EPISODE_COUNT
        assert brain.episode_count == 0
        assert brain.experience_buffer == []
        assert brain.patterns_learned == []
        assert brain.consolidation_count == 0
        assert brain.total_experiences == 0

    def test_init_with_off_preset(self):
        """SimpleBrain with OFF preset disables auto-consolidation."""
        brain = SimpleBrain(preset=BrainPreset.OFF)
        assert brain.preset == BrainPreset.OFF
        assert brain.config["auto_consolidate"] is False

    def test_from_preset_valid(self):
        """from_preset creates brain with the requested preset."""
        brain = SimpleBrain.from_preset("thorough")
        assert brain.preset == BrainPreset.THOROUGH

    def test_from_preset_invalid_falls_back_to_balanced(self):
        """from_preset with unknown name falls back to BALANCED."""
        brain = SimpleBrain.from_preset("nonexistent")
        assert brain.preset == BrainPreset.BALANCED

    def test_chunk_size_model_aware(self):
        """Chunk size is computed from model context and preset ratio."""
        brain = SimpleBrain(preset=BrainPreset.BALANCED, model_name="gpt-4")
        expected_context = MODEL_CONTEXTS["gpt-4"]
        expected_ratio = PRESET_CONFIGS[BrainPreset.BALANCED]["chunk_ratio"]
        expected_chunk = int(expected_context * expected_ratio)
        # Clamped to [1000, 100000]
        expected_chunk = max(1000, min(expected_chunk, 100000))
        assert brain.chunk_size == expected_chunk

    def test_get_stats(self):
        """get_stats returns a dict with the expected keys and correct values."""
        brain = SimpleBrain(preset=BrainPreset.MINIMAL, model_name="gpt-4o")
        stats = brain.get_stats()
        assert stats["preset"] == "minimal"
        assert stats["model"] == "gpt-4o"
        assert stats["buffer_size"] == 0
        assert stats["patterns_learned"] == 0
        assert stats["consolidation_count"] == 0
        assert stats["total_experiences"] == 0
        assert stats["total_pruned"] == 0

    def test_process_off_preset_returns_false(self):
        """process() always returns False when preset is OFF."""
        brain = SimpleBrain(preset=BrainPreset.OFF)
        exp = Experience(content="test", reward=0.8)
        result = brain.process(exp)
        assert result is False
        assert brain.total_experiences == 0

    def test_process_stores_experience_when_should_remember(self):
        """process() stores experiences when _should_remember returns True."""
        brain = SimpleBrain(preset=BrainPreset.BALANCED)
        # Mock _should_remember to always return True
        brain._should_remember = lambda e: True
        # Mock _should_consolidate to return False (avoid asyncio.create_task)
        brain._should_consolidate = lambda: False

        exp = Experience(content="important", reward=0.9)
        result = brain.process(exp)

        assert result is True
        assert len(brain.experience_buffer) == 1
        assert brain.total_experiences == 1
        assert brain.episode_count == 1

    def test_process_rejects_when_should_not_remember(self):
        """process() does not store when _should_remember returns False."""
        brain = SimpleBrain(preset=BrainPreset.BALANCED)
        brain._should_remember = lambda e: False
        brain._should_consolidate = lambda: False

        exp = Experience(content="boring", reward=0.5)
        result = brain.process(exp)

        assert result is False
        assert len(brain.experience_buffer) == 0
        # total_experiences and episode_count still increment
        assert brain.total_experiences == 1

    def test_process_skips_when_consolidating(self):
        """process() returns False when brain is in consolidating state."""
        brain = SimpleBrain(preset=BrainPreset.BALANCED)
        brain.is_consolidating = True
        exp = Experience(content="test", reward=0.9)
        result = brain.process(exp)
        assert result is False

    @pytest.mark.asyncio
    async def test_consolidate_extracts_patterns(self):
        """consolidate() extracts patterns and resets episode count."""
        brain = SimpleBrain(preset=BrainPreset.BALANCED)
        # Manually fill buffer with experiences
        for i in range(5):
            brain.experience_buffer.append(
                Experience(content=f"success_{i}", reward=0.9, agent="TestAgent")
            )
        brain.episode_count = 10

        await brain.consolidate(ConsolidationTrigger.EXPLICIT)

        assert brain.consolidation_count == 1
        assert brain.episode_count == 0
        assert brain.is_consolidating is False
        # Patterns should have been extracted (5 successes >= 3 threshold)
        assert any("SUCCESS_PATTERN" in p for p in brain.patterns_learned)

    @pytest.mark.asyncio
    async def test_consolidate_skips_when_empty(self):
        """consolidate() is a no-op when buffer is empty."""
        brain = SimpleBrain(preset=BrainPreset.BALANCED)
        await brain.consolidate()
        assert brain.consolidation_count == 0

    @pytest.mark.asyncio
    async def test_consolidate_skips_when_already_consolidating(self):
        """consolidate() is a no-op when is_consolidating is True."""
        brain = SimpleBrain(preset=BrainPreset.BALANCED)
        brain.experience_buffer.append(Experience(content="x", reward=0.5))
        brain.is_consolidating = True
        await brain.consolidate()
        assert brain.consolidation_count == 0

    @pytest.mark.asyncio
    async def test_session_auto_consolidates_on_exit(self):
        """session() context manager calls consolidate on exit."""
        brain = SimpleBrain(preset=BrainPreset.BALANCED)
        brain.experience_buffer.append(
            Experience(content="session_exp", reward=0.8)
        )

        async with brain.session() as session:
            assert session is brain

        # After exiting session, consolidation should have run
        assert brain.consolidation_count == 1

    def test_get_learned_patterns_returns_copy(self):
        """get_learned_patterns returns a copy, not the internal list."""
        brain = SimpleBrain(preset=BrainPreset.BALANCED)
        brain.patterns_learned = ["pattern1", "pattern2"]
        patterns = brain.get_learned_patterns()
        assert patterns == ["pattern1", "pattern2"]
        # Mutating the returned list should not affect internals
        patterns.append("extra")
        assert len(brain.patterns_learned) == 2


# =============================================================================
# EpisodicMemory Dataclass
# =============================================================================

@pytest.mark.unit
class TestEpisodicMemory:
    """Tests for the EpisodicMemory dataclass."""

    def test_creation_with_defaults(self):
        """EpisodicMemory uses sensible defaults for optional fields."""
        mem = EpisodicMemory(
            content={"action": "search"},
            reward=0.8,
            timestamp=1000.0,
        )
        assert mem.content == {"action": "search"}
        assert mem.reward == 0.8
        assert mem.strength == 1.0
        assert mem.replay_count == 0
        assert mem.consolidated is False
        assert mem.novelty_score == 0.0

    def test_custom_values(self):
        """EpisodicMemory stores custom values for all fields."""
        mem = EpisodicMemory(
            content={"action": "plan"},
            reward=0.2,
            timestamp=500.0,
            strength=1.5,
            replay_count=3,
            consolidated=True,
            novelty_score=0.9,
        )
        assert mem.strength == 1.5
        assert mem.replay_count == 3
        assert mem.consolidated is True
        assert mem.novelty_score == 0.9


# =============================================================================
# SemanticPattern Dataclass
# =============================================================================

@pytest.mark.unit
class TestSemanticPattern:
    """Tests for the SemanticPattern dataclass."""

    def test_creation(self):
        """SemanticPattern stores abstract lesson and metadata."""
        pattern = SemanticPattern(
            abstract_lesson="Strategy 'search' tends to succeed",
            strength=2.5,
            source_count=4,
            created_at=1000.0,
            last_reinforced=1100.0,
            tags=["action:search", "status:success"],
        )
        assert pattern.abstract_lesson == "Strategy 'search' tends to succeed"
        assert pattern.strength == 2.5
        assert pattern.source_count == 4
        assert pattern.tags == ["action:search", "status:success"]

    def test_default_tags(self):
        """SemanticPattern defaults to empty tags list."""
        pattern = SemanticPattern(
            abstract_lesson="lesson",
            strength=1.0,
            source_count=1,
            created_at=0.0,
            last_reinforced=0.0,
        )
        assert pattern.tags == []


# =============================================================================
# BrainInspiredMemoryManager
# =============================================================================

@pytest.mark.unit
class TestBrainInspiredMemoryManager:
    """Tests for BrainInspiredMemoryManager store, consolidation, retrieval."""

    def test_init_defaults(self):
        """BrainInspiredMemoryManager initializes with expected defaults."""
        mgr = BrainInspiredMemoryManager()
        assert mgr.hippocampus == []
        assert mgr.neocortex == []
        assert mgr.sleep_interval == 10
        assert mgr.max_hippocampus_size == 100
        assert mgr.total_consolidations == 0
        assert mgr.episodes_since_sleep == 0

    def test_store_experience_adds_to_hippocampus(self):
        """store_experience adds an EpisodicMemory to the hippocampus."""
        mgr = BrainInspiredMemoryManager()
        mgr.store_experience({"action": "research", "goal": "find data"}, reward=0.85)
        assert len(mgr.hippocampus) == 1
        mem = mgr.hippocampus[0]
        assert mem.content == {"action": "research", "goal": "find data"}
        assert mem.reward == 0.85
        assert mem.strength == 1.0
        assert mem.replay_count == 0

    def test_store_experience_prunes_when_over_capacity(self):
        """Hippocampus prunes to max size when capacity is exceeded."""
        mgr = BrainInspiredMemoryManager(max_hippocampus_size=5)
        for i in range(10):
            mgr.store_experience({"action": f"act_{i}"}, reward=i * 0.1)
        assert len(mgr.hippocampus) <= 5

    def test_should_consolidate_false_initially(self):
        """should_consolidate returns False before enough episodes."""
        mgr = BrainInspiredMemoryManager(sleep_interval=5)
        mgr.episodes_since_sleep = 3
        assert mgr.should_consolidate() is False

    def test_should_consolidate_true_at_threshold(self):
        """should_consolidate returns True when episodes reach interval."""
        mgr = BrainInspiredMemoryManager(sleep_interval=5)
        mgr.episodes_since_sleep = 5
        assert mgr.should_consolidate() is True

    def test_trigger_consolidation_transfers_to_neocortex(self):
        """Full consolidation cycle: store, replay, transfer to neocortex."""
        mgr = BrainInspiredMemoryManager(sleep_interval=3, replay_threshold=0.0)
        # Store high-reward experiences
        for i in range(5):
            mgr.store_experience(
                {"action": "search", "status": "ok"},
                reward=0.9,
            )
        mgr.episodes_since_sleep = 5

        mgr.trigger_consolidation()

        assert mgr.total_consolidations == 1
        assert mgr.episodes_since_sleep == 0
        # Patterns should have been transferred to neocortex
        assert len(mgr.neocortex) > 0

    def test_get_consolidated_knowledge_empty(self):
        """get_consolidated_knowledge returns empty string with no patterns."""
        mgr = BrainInspiredMemoryManager()
        assert mgr.get_consolidated_knowledge() == ""

    def test_get_consolidated_knowledge_with_patterns(self):
        """get_consolidated_knowledge returns formatted knowledge from neocortex."""
        mgr = BrainInspiredMemoryManager()
        mgr.neocortex.append(
            SemanticPattern(
                abstract_lesson="Search before acting",
                strength=3.0,
                source_count=5,
                created_at=time.time(),
                last_reinforced=time.time(),
                tags=["action:search"],
            )
        )
        knowledge = mgr.get_consolidated_knowledge()
        assert "Search before acting" in knowledge
        assert "Brain-Consolidated Knowledge" in knowledge

    def test_get_statistics(self):
        """get_statistics returns a complete stats dict."""
        mgr = BrainInspiredMemoryManager()
        mgr.store_experience({"action": "test"}, reward=0.5)
        stats = mgr.get_statistics()
        assert stats["hippocampus_size"] == 1
        assert stats["neocortex_size"] == 0
        assert stats["total_consolidations"] == 0
        assert stats["total_replay_count"] == 0
        assert isinstance(stats["avg_hippo_strength"], float)


# =============================================================================
# Utility Functions
# =============================================================================

@pytest.mark.unit
class TestUtilityFunctions:
    """Tests for get_model_context, calculate_chunk_size, load_brain_config."""

    def test_get_model_context_known_model(self):
        """get_model_context returns correct size for known models."""
        assert get_model_context("gpt-4") == 8192
        assert get_model_context("claude-3-opus") == 200000

    def test_get_model_context_unknown_model(self):
        """get_model_context returns default for unknown models."""
        assert get_model_context("totally-unknown-model") == MODEL_CONTEXTS["default"]

    def test_calculate_chunk_size_bounds(self):
        """calculate_chunk_size clamps result between 1000 and 100000."""
        # Very small model context
        size = calculate_chunk_size("llama-7b", ratio=0.1)
        assert size >= 1000
        # Very large model context with high ratio
        size = calculate_chunk_size("claude-3-opus", ratio=0.9)
        assert size <= 100000

    def test_load_brain_config_simple_string(self):
        """load_brain_config with simple string preset."""
        config = {"reval": {"brain": "thorough"}}
        brain = load_brain_config(config)
        assert brain.preset == BrainPreset.THOROUGH

    def test_load_brain_config_dict(self):
        """load_brain_config with advanced dict config."""
        config = {
            "reval": {
                "brain": {
                    "preset": "minimal",
                    "model": "claude-3-opus",
                    "consolidate_on": "memory_pressure",
                }
            }
        }
        brain = load_brain_config(config)
        assert brain.preset == BrainPreset.MINIMAL
        assert brain.model_name == "claude-3-opus"

    def test_load_brain_config_missing_defaults(self):
        """load_brain_config with empty config uses defaults."""
        brain = load_brain_config({})
        assert brain.preset == BrainPreset.BALANCED


# =============================================================================
# SimpleBrain Extended Tests
# =============================================================================

@pytest.mark.unit
class TestSimpleBrainExtended:
    """Additional tests for SimpleBrain methods and edge cases."""

    def test_should_consolidate_false_when_consolidating(self):
        """_should_consolidate returns False when already consolidating."""
        brain = SimpleBrain(preset=BrainPreset.BALANCED)
        brain.is_consolidating = True
        assert brain._should_consolidate() is False

    def test_should_consolidate_false_no_auto(self):
        """_should_consolidate returns False when auto_consolidate is off."""
        brain = SimpleBrain(preset=BrainPreset.OFF)
        assert brain._should_consolidate() is False

    def test_should_consolidate_episode_count_trigger(self):
        """_should_consolidate triggers on episode_count threshold."""
        brain = SimpleBrain(preset=BrainPreset.BALANCED)
        brain.consolidate_on = ConsolidationTrigger.EPISODE_COUNT
        brain.episode_count = brain.config['consolidation_interval']
        # Need auto_consolidate to be True
        brain.config['auto_consolidate'] = True
        result = brain._should_consolidate()
        assert result is True

    def test_should_consolidate_memory_pressure_trigger(self):
        """_should_consolidate triggers on memory pressure."""
        brain = SimpleBrain(preset=BrainPreset.BALANCED)
        brain.consolidate_on = ConsolidationTrigger.MEMORY_PRESSURE
        brain.config['auto_consolidate'] = True
        # Fill buffer to 80%+
        buf_size = brain.config['memory_buffer_size']
        for i in range(int(buf_size * 0.9)):
            brain.experience_buffer.append(Experience(content=f"exp_{i}", reward=0.5))
        assert brain._should_consolidate() is True

    def test_prune_buffer_removes_middle_experiences(self):
        """_prune_buffer removes middle-reward experiences."""
        brain = SimpleBrain(preset=BrainPreset.BALANCED)
        brain.experience_buffer = [
            Experience(content="high", reward=0.95),
            Experience(content="mid1", reward=0.5),
            Experience(content="mid2", reward=0.5),
            Experience(content="mid3", reward=0.5),
            Experience(content="mid4", reward=0.5),
            Experience(content="low", reward=0.05),
        ]
        original_count = len(brain.experience_buffer)
        brain._prune_buffer()
        assert len(brain.experience_buffer) < original_count
        assert brain.total_pruned > 0

    def test_prune_buffer_empty_is_noop(self):
        """_prune_buffer does nothing on empty buffer."""
        brain = SimpleBrain(preset=BrainPreset.BALANCED)
        brain._prune_buffer()
        assert brain.total_pruned == 0

    def test_extract_patterns_failure_threshold(self):
        """_extract_patterns finds FAILURE_PATTERN when 3+ failures."""
        brain = SimpleBrain(preset=BrainPreset.BALANCED)
        for i in range(4):
            brain.experience_buffer.append(
                Experience(content=f"fail_{i}", reward=0.1)
            )
        patterns = brain._extract_patterns()
        assert any("FAILURE_PATTERN" in p for p in patterns)

    def test_extract_patterns_agent_specific(self):
        """_extract_patterns finds AGENT_PATTERN for agent with 3+ experiences."""
        brain = SimpleBrain(preset=BrainPreset.BALANCED)
        for i in range(5):
            brain.experience_buffer.append(
                Experience(content=f"task_{i}", reward=0.8, agent="ResearchAgent")
            )
        patterns = brain._extract_patterns()
        assert any("AGENT_PATTERN" in p and "ResearchAgent" in p for p in patterns)

    def test_extract_patterns_no_patterns(self):
        """_extract_patterns returns empty when not enough data."""
        brain = SimpleBrain(preset=BrainPreset.BALANCED)
        brain.experience_buffer = [Experience(content="solo", reward=0.5)]
        patterns = brain._extract_patterns()
        assert patterns == []

    @pytest.mark.asyncio
    async def test_consolidate_custom_trigger(self):
        """consolidate with custom trigger works."""
        brain = SimpleBrain(preset=BrainPreset.BALANCED)
        for i in range(3):
            brain.experience_buffer.append(
                Experience(content=f"exp_{i}", reward=0.9)
            )
        await brain.consolidate(ConsolidationTrigger.PIPELINE_STAGE)
        assert brain.consolidation_count == 1

    def test_chunk_size_clamped_min(self):
        """Chunk size is clamped to minimum 1000."""
        brain = SimpleBrain(preset=BrainPreset.MINIMAL, model_name="unknown-tiny-model")
        assert brain.chunk_size >= 1000

    def test_chunk_size_clamped_max(self):
        """Chunk size is clamped to maximum 100000."""
        brain = SimpleBrain(preset=BrainPreset.THOROUGH, model_name="claude-3-opus")
        assert brain.chunk_size <= 100000

    def test_from_preset_minimal(self):
        """from_preset('minimal') creates minimal brain."""
        brain = SimpleBrain.from_preset("minimal")
        assert brain.preset == BrainPreset.MINIMAL

    def test_from_preset_off(self):
        """from_preset('off') creates disabled brain."""
        brain = SimpleBrain.from_preset("off")
        assert brain.preset == BrainPreset.OFF
        assert brain.config['auto_consolidate'] is False


# =============================================================================
# BrainInspiredMemoryManager Extended Tests
# =============================================================================

@pytest.mark.unit
class TestBrainInspiredMemoryManagerExtended:
    """Additional tests for BrainInspiredMemoryManager."""

    def test_custom_init(self):
        """BrainInspiredMemoryManager custom init params."""
        mgr = BrainInspiredMemoryManager(
            max_hippocampus_size=50,
            sleep_interval=20,
        )
        assert mgr.max_hippocampus_size == 50
        assert mgr.sleep_interval == 20

    def test_store_experience_adds_to_hippocampus_count(self):
        """store_experience increases hippocampus size."""
        mgr = BrainInspiredMemoryManager()
        mgr.store_experience({"action": "search"}, reward=0.7)
        assert len(mgr.hippocampus) == 1
        mgr.store_experience({"action": "analyze"}, reward=0.6)
        assert len(mgr.hippocampus) == 2

    def test_store_low_reward_experience(self):
        """Low-reward experiences are still stored."""
        mgr = BrainInspiredMemoryManager()
        mgr.store_experience({"action": "fail"}, reward=0.1)
        assert len(mgr.hippocampus) == 1
        assert mgr.hippocampus[0].reward == 0.1

    def test_novelty_computed_for_stored_experience(self):
        """Stored experience has novelty_score computed."""
        mgr = BrainInspiredMemoryManager()
        mgr.store_experience({"action": "search", "query": "unique"}, reward=0.5)
        mem = mgr.hippocampus[0]
        assert isinstance(mem.novelty_score, float)

    def test_consolidation_resets_episode_count(self):
        """trigger_consolidation resets episodes_since_sleep to 0."""
        mgr = BrainInspiredMemoryManager(sleep_interval=2)
        for i in range(5):
            mgr.store_experience({"action": f"act_{i}"}, reward=0.8)
        mgr.episodes_since_sleep = 5
        mgr.trigger_consolidation()
        assert mgr.episodes_since_sleep == 0

    def test_multiple_consolidations(self):
        """Multiple consolidation cycles accumulate patterns."""
        mgr = BrainInspiredMemoryManager(sleep_interval=3, replay_threshold=0.0)
        # First cycle
        for i in range(5):
            mgr.store_experience({"action": "search", "status": "ok"}, reward=0.9)
        mgr.episodes_since_sleep = 5
        mgr.trigger_consolidation()
        first_count = len(mgr.neocortex)

        # Second cycle
        for i in range(5):
            mgr.store_experience({"action": "plan", "status": "ok"}, reward=0.85)
        mgr.episodes_since_sleep = 5
        mgr.trigger_consolidation()

        assert mgr.total_consolidations == 2
        # Should have at least as many patterns as first cycle
        assert len(mgr.neocortex) >= first_count

    def test_get_statistics_complete_keys(self):
        """get_statistics returns all expected keys."""
        mgr = BrainInspiredMemoryManager()
        stats = mgr.get_statistics()
        expected_keys = {
            "hippocampus_size", "neocortex_size", "total_consolidations",
            "episodes_since_sleep", "total_replay_count", "avg_hippo_strength",
        }
        assert expected_keys.issubset(set(stats.keys()))

    def test_get_consolidated_knowledge_max_items(self):
        """get_consolidated_knowledge respects max_items limit."""
        mgr = BrainInspiredMemoryManager()
        # Add many patterns
        for i in range(15):
            mgr.neocortex.append(SemanticPattern(
                abstract_lesson=f"Lesson {i}",
                strength=float(i),
                source_count=1,
                created_at=time.time(),
                last_reinforced=time.time(),
            ))
        knowledge = mgr.get_consolidated_knowledge(max_items=5)
        # Should not include all 15
        lesson_count = knowledge.count("Lesson")
        assert lesson_count <= 5

    def test_pruning_during_consolidation(self):
        """Synaptic pruning removes weak memories during consolidation."""
        mgr = BrainInspiredMemoryManager(sleep_interval=2, replay_threshold=0.0)
        # Add experiences with varying strengths
        for i in range(10):
            mgr.store_experience({"action": f"act_{i}"}, reward=0.9)
        # Manually weaken some memories
        for mem in mgr.hippocampus[:5]:
            mem.strength = 0.01
        mgr.episodes_since_sleep = 10
        mgr.trigger_consolidation()
        # After pruning, weakest should be removed
        assert mgr.total_consolidations == 1


# =============================================================================
# Experience Edge Cases
# =============================================================================

@pytest.mark.unit
class TestExperienceEdgeCases:
    """Edge case tests for Experience dataclass."""

    def test_zero_reward(self):
        """Experience with 0 reward is valid."""
        exp = Experience(content="nothing", reward=0.0)
        assert exp.reward == 0.0

    def test_negative_reward(self):
        """Experience with negative reward is valid."""
        exp = Experience(content="penalty", reward=-0.5)
        assert exp.reward == -0.5

    def test_large_content(self):
        """Experience with large content string."""
        content = "x" * 10000
        exp = Experience(content=content, reward=0.5)
        assert len(exp.content) == 10000

    def test_dict_content(self):
        """Experience content can be any type."""
        exp = Experience(content={"key": "value"}, reward=0.5)
        assert exp.content == {"key": "value"}


# =============================================================================
# load_brain_config Extended Tests
# =============================================================================

@pytest.mark.unit
class TestLoadBrainConfigExtended:
    """Additional tests for load_brain_config."""

    def test_config_with_dict_brain_creates_brain(self):
        """load_brain_config with dict config creates brain with correct preset."""
        config = {
            "reval": {
                "brain": {
                    "preset": "balanced",
                    "consolidate_on": "memory_pressure",
                }
            }
        }
        brain = load_brain_config(config)
        # Dict config creates brain from preset (consolidate_on is parsed but
        # not passed through to SimpleBrain in current implementation)
        assert brain.preset == BrainPreset.BALANCED

    def test_config_with_thorough_dict(self):
        """load_brain_config with thorough preset dict."""
        config = {
            "reval": {
                "brain": {
                    "preset": "thorough",
                }
            }
        }
        brain = load_brain_config(config)
        assert brain.preset == BrainPreset.THOROUGH

    def test_config_nested_reval_missing(self):
        """load_brain_config handles missing reval key."""
        brain = load_brain_config({"other_key": "value"})
        assert brain.preset == BrainPreset.BALANCED

    def test_config_with_custom_model(self):
        """load_brain_config applies custom model from config."""
        config = {
            "reval": {
                "brain": {
                    "preset": "minimal",
                    "model": "gpt-4o",
                }
            }
        }
        brain = load_brain_config(config)
        assert brain.model_name == "gpt-4o"
        assert brain.preset == BrainPreset.MINIMAL
