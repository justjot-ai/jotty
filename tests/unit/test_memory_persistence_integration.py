"""
Tests for memory persistence integration across BaseAgent and SwarmResources.

Verifies that:
- BaseAgent creates SwarmMemory with persistence
- BaseAgent.save_memory() persists to disk
- Memories survive reload (new agent, same name)
- SwarmResources provides non-None memory with persistence
- SwarmLearning wires _memory_persistence from resources
- Auto-save fires after _record_trace()
"""

import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tmp_home(tmp_path):
    """Return a tmp_path-based home so persistence files don't pollute ~."""
    return tmp_path / "fakehome"


# ---------------------------------------------------------------------------
# BaseAgent tests
# ---------------------------------------------------------------------------


class TestBaseAgentMemoryPersistence:
    """Memory persistence through BaseAgent.memory property."""

    def _make_agent(self, tmp_path, name="test_agent"):
        from Jotty.core.intelligence.reasoning.base.base_agent import (
            AgentRuntimeConfig,
            BaseAgent,
        )

        class StubAgent(BaseAgent):
            async def _execute_impl(self, **kwargs):
                return "ok"

        agent = StubAgent(config=AgentRuntimeConfig(name=name, enable_safety_gates=False))

        # Redirect persistence dir to tmp_path
        fake_home = _make_tmp_home(tmp_path)
        with patch("pathlib.Path.home", return_value=fake_home):
            _ = agent.memory  # triggers lazy init

        return agent

    def test_memory_property_creates_persistence(self, tmp_path):
        """Accessing .memory should set _memory_persistence."""
        from Jotty.core.intelligence.reasoning.base.base_agent import (
            AgentRuntimeConfig,
            BaseAgent,
        )

        class StubAgent(BaseAgent):
            async def _execute_impl(self, **kwargs):
                return "ok"

        agent = StubAgent(config=AgentRuntimeConfig(name="persist_test", enable_safety_gates=False))
        assert agent._memory is None
        assert agent._memory_persistence is None

        fake_home = _make_tmp_home(tmp_path)
        with patch("pathlib.Path.home", return_value=fake_home):
            mem = agent.memory

        assert mem is not None
        assert agent._memory_persistence is not None

    def test_save_memory_writes_files(self, tmp_path):
        """save_memory() should write JSON files to disk."""
        from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel
        from Jotty.core.intelligence.reasoning.base.base_agent import (
            AgentRuntimeConfig,
            BaseAgent,
        )

        class StubAgent(BaseAgent):
            async def _execute_impl(self, **kwargs):
                return "ok"

        agent = StubAgent(config=AgentRuntimeConfig(name="saver", enable_safety_gates=False))

        fake_home = _make_tmp_home(tmp_path)
        with patch("pathlib.Path.home", return_value=fake_home):
            mem = agent.memory
            # Store something
            mem.store(content="test fact", level=MemoryLevel.SEMANTIC, context={}, goal="test")
            result = agent.save_memory()

        assert result is True

        # Check that files exist
        persistence_dir = fake_home / "jotty" / "memory" / "saver"
        assert persistence_dir.exists()
        semantic_file = persistence_dir / "semantic_memories.json"
        assert semantic_file.exists()

        data = json.loads(semantic_file.read_text())
        assert len(data) >= 1
        assert any("test fact" in entry.get("content", "") for entry in data)

    def test_memory_survives_reload(self, tmp_path):
        """Memories saved by one agent should be loaded by another with the same name."""
        from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel
        from Jotty.core.intelligence.reasoning.base.base_agent import (
            AgentRuntimeConfig,
            BaseAgent,
        )

        class StubAgent(BaseAgent):
            async def _execute_impl(self, **kwargs):
                return "ok"

        fake_home = _make_tmp_home(tmp_path)

        # Agent 1: store and save
        agent1 = StubAgent(config=AgentRuntimeConfig(name="reload_test", enable_safety_gates=False))
        with patch("pathlib.Path.home", return_value=fake_home):
            mem1 = agent1.memory
            mem1.store(content="persistent fact", level=MemoryLevel.SEMANTIC, context={}, goal="g")
            agent1.save_memory()

        # Agent 2: new instance, same name — should load saved data
        agent2 = StubAgent(config=AgentRuntimeConfig(name="reload_test", enable_safety_gates=False))
        with patch("pathlib.Path.home", return_value=fake_home):
            mem2 = agent2.memory

        # The loaded memory should contain "persistent fact"
        entries = list(mem2.memories[MemoryLevel.SEMANTIC].values())
        assert len(entries) >= 1
        assert any("persistent fact" in e.content for e in entries)

    def test_save_memory_returns_false_without_persistence(self):
        """save_memory() returns False when no persistence is set up."""
        from Jotty.core.intelligence.reasoning.base.base_agent import (
            AgentRuntimeConfig,
            BaseAgent,
        )

        class StubAgent(BaseAgent):
            async def _execute_impl(self, **kwargs):
                return "ok"

        agent = StubAgent(
            config=AgentRuntimeConfig(
                name="no_persist", enable_memory=False, enable_safety_gates=False
            )
        )
        assert agent.save_memory() is False


# ---------------------------------------------------------------------------
# SwarmResources tests
# ---------------------------------------------------------------------------


class TestSwarmResourcesMemory:
    """SwarmResources should provide real SwarmMemory with persistence."""

    def setup_method(self):
        from Jotty.core.intelligence.reasoning.planners.swarm_resources_stub import SwarmResources

        SwarmResources.reset()

    def teardown_method(self):
        from Jotty.core.intelligence.reasoning.planners.swarm_resources_stub import SwarmResources

        SwarmResources.reset()

    def test_memory_is_not_none(self, tmp_path):
        from Jotty.core.intelligence.reasoning.planners.swarm_resources_stub import SwarmResources

        fake_home = _make_tmp_home(tmp_path)
        with patch("pathlib.Path.home", return_value=fake_home):
            SwarmResources.reset()
            resources = SwarmResources.get_instance()

        assert resources.memory is not None

    def test_memory_persistence_is_set(self, tmp_path):
        from Jotty.core.intelligence.reasoning.planners.swarm_resources_stub import SwarmResources

        fake_home = _make_tmp_home(tmp_path)
        with patch("pathlib.Path.home", return_value=fake_home):
            SwarmResources.reset()
            resources = SwarmResources.get_instance()

        assert resources._memory_persistence is not None

    def test_save_memory(self, tmp_path):
        from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel
        from Jotty.core.intelligence.reasoning.planners.swarm_resources_stub import SwarmResources

        fake_home = _make_tmp_home(tmp_path)
        with patch("pathlib.Path.home", return_value=fake_home):
            SwarmResources.reset()
            resources = SwarmResources.get_instance()
            resources.memory.store(
                content="swarm fact", level=MemoryLevel.EPISODIC, context={}, goal="test"
            )
            result = resources.save_memory()

        assert result is True
        episodic_file = fake_home / "jotty" / "memory" / "swarm_shared" / "episodic_memories.json"
        assert episodic_file.exists()


# ---------------------------------------------------------------------------
# SwarmLearning wiring tests
# ---------------------------------------------------------------------------


class TestSwarmLearningPersistenceWiring:
    """SwarmLearning._init_shared_resources() should wire _memory_persistence."""

    def setup_method(self):
        from Jotty.core.intelligence.reasoning.planners.swarm_resources_stub import SwarmResources

        SwarmResources.reset()

    def teardown_method(self):
        from Jotty.core.intelligence.reasoning.planners.swarm_resources_stub import SwarmResources

        SwarmResources.reset()

    def test_memory_persistence_wired_from_resources(self, tmp_path):
        from Jotty.core.intelligence.orchestration.swarms._base.swarm_learning import SwarmLearning
        from Jotty.core.intelligence.orchestration.swarms._base.swarm_types import SwarmBaseConfig

        class StubSwarm(SwarmLearning):
            async def execute(self, *args, **kwargs):
                return None

        config = SwarmBaseConfig(
            name="test_swarm",
            domain="test",
            enable_self_improvement=False,
        )
        swarm = StubSwarm(config=config)

        fake_home = _make_tmp_home(tmp_path)
        with patch("pathlib.Path.home", return_value=fake_home):
            swarm._init_shared_resources()

        assert swarm._memory is not None
        assert swarm._memory_persistence is not None


# ---------------------------------------------------------------------------
# Auto-save after _record_trace tests
# ---------------------------------------------------------------------------


class TestAutoSaveAfterTrace:
    """_record_trace should auto-save memory persistence."""

    def test_auto_save_called_after_trace(self):
        """Mock _memory_persistence.save and verify it's called after trace recording."""
        from Jotty.core.intelligence.orchestration.swarms._base.swarm_learning import SwarmLearning
        from Jotty.core.intelligence.orchestration.swarms._base.swarm_types import (
            AgentRole,
            SwarmBaseConfig,
        )

        class StubSwarm(SwarmLearning):
            async def execute(self, *args, **kwargs):
                return None

        config = SwarmBaseConfig(
            name="trace_test",
            domain="test",
            enable_self_improvement=False,
            enable_learning=True,
        )
        swarm = StubSwarm(config=config)

        # Give it a mock memory and persistence
        mock_memory = MagicMock()
        mock_memory.store_with_outcome = MagicMock()
        swarm._memory = mock_memory

        mock_persistence = MagicMock()
        mock_persistence.save = MagicMock(return_value=True)
        swarm._memory_persistence = mock_persistence

        # Also need _swarm_intelligence for the trace to work
        swarm._swarm_intelligence = None
        swarm._traces = []

        swarm._record_trace(
            agent_name="test_agent",
            agent_role=AgentRole.ACTOR,
            input_data={"query": "test"},
            output_data={"result": "ok"},
            execution_time=1.0,
            success=True,
        )

        # Verify auto-save was called
        mock_persistence.save.assert_called_once()
