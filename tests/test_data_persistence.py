"""
Tests for Data Persistence & Data Registry Modules
====================================================

Covers:
1. Vault (core/persistence/persistence.py)
2. SessionManager (core/persistence/session_manager.py)
3. ScratchpadPersistence (core/persistence/scratchpad_persistence.py)
4. DataRegistry (core/data/data_registry.py)
5. AgenticParameterResolver (core/data/parameter_resolver.py)
6. AgenticFeedbackRouter (core/data/feedback_router.py)
7. InformationTheoreticStorage (core/data/information_storage.py)

All tests use mocks -- no LLM calls, no API keys, runs offline.
"""

import json
import math
import os
import time
import asyncio
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock, PropertyMock
from dataclasses import dataclass, field
from enum import Enum

# ---------------------------------------------------------------------------
# Safe imports with skip guards
# ---------------------------------------------------------------------------
try:
    from Jotty.core.persistence.persistence import Vault
    HAS_VAULT = True
except ImportError:
    HAS_VAULT = False

try:
    from Jotty.core.persistence.session_manager import SessionManager
    HAS_SESSION_MANAGER = True
except ImportError:
    HAS_SESSION_MANAGER = False

try:
    from Jotty.core.persistence.scratchpad_persistence import ScratchpadPersistence
    HAS_SCRATCHPAD = True
except ImportError:
    HAS_SCRATCHPAD = False

try:
    from Jotty.core.foundation.types.agent_types import (
        SharedScratchpad,
        AgentMessage,
    )
    from Jotty.core.foundation.types.enums import CommunicationType
    HAS_AGENT_TYPES = True
except ImportError:
    HAS_AGENT_TYPES = False

try:
    from Jotty.core.data.data_registry import DataArtifact, DataRegistry
    HAS_DATA_REGISTRY = True
except ImportError:
    HAS_DATA_REGISTRY = False

try:
    from Jotty.core.data.parameter_resolver import AgenticParameterResolver
    HAS_PARAM_RESOLVER = True
except ImportError:
    HAS_PARAM_RESOLVER = False

try:
    from Jotty.core.data.feedback_router import AgenticFeedbackRouter
    HAS_FEEDBACK_ROUTER = True
except ImportError:
    HAS_FEEDBACK_ROUTER = False

try:
    from Jotty.core.data.information_storage import (
        InformationTheoreticStorage,
        InformationWeightedMemory,
        SurpriseEstimator,
    )
    HAS_INFO_STORAGE = True
except ImportError:
    HAS_INFO_STORAGE = False

try:
    from Jotty.core.agents.feedback_channel import FeedbackType, FeedbackMessage
    HAS_FEEDBACK_CHANNEL = True
except ImportError:
    HAS_FEEDBACK_CHANNEL = False


# ===========================================================================
# Helpers -- lightweight mock objects used across test classes
# ===========================================================================

class _MockTaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


def _make_mock_task(**overrides):
    """Create a mock task object with all attributes the Vault serialiser expects."""
    defaults = dict(
        task_id="task_1",
        description="Test task",
        actor="TestActor",
        status=_MockTaskStatus.PENDING,
        priority=0.8,
        estimated_reward=0.5,
        confidence=0.9,
        attempts=0,
        max_attempts=3,
        progress=0.0,
        depends_on=[],
        blocks=[],
        started_at=None,
        completed_at=None,
        estimated_duration=10.0,
        intermediary_values={},
        predicted_next_task=None,
        predicted_duration=None,
        predicted_reward=None,
        failure_reasons=[],
        result=None,
        error=None,
    )
    defaults.update(overrides)
    task = Mock()
    for k, v in defaults.items():
        setattr(task, k, v)
    return task


def _make_mock_todo(**overrides):
    """Create a mock SwarmTaskBoard (todo) object."""
    task = _make_mock_task()
    defaults = dict(
        root_task="Root Task",
        todo_id="todo_1",
        subtasks={"task_1": task},
        execution_order=["task_1"],
        completed_tasks=set(),
        failed_tasks=set(),
        current_task_id="task_1",
        estimated_remaining_steps=1,
        completion_probability=0.75,
    )
    defaults.update(overrides)
    todo = Mock()
    for k, v in defaults.items():
        setattr(todo, k, v)
    return todo


def _make_mock_config(tmp_path, **overrides):
    """Create a mock SwarmConfig suitable for SessionManager."""
    defaults = dict(
        output_base_dir=str(tmp_path),
        create_run_folder=True,
        auto_load_on_start=False,
        persist_memories=True,
        persist_q_tables=True,
        persist_brain_state=True,
        persist_todos=True,
        max_runs_to_keep=5,
    )
    defaults.update(overrides)
    config = Mock()
    for k, v in defaults.items():
        setattr(config, k, v)
    config.to_flat_dict.return_value = defaults
    return config


# ===========================================================================
# 1. Vault (core/persistence/persistence.py)
# ===========================================================================

@pytest.mark.unit
@pytest.mark.skipif(not HAS_VAULT, reason="Vault not importable")
class TestVault:
    """Tests for the Vault persistence manager."""

    def test_init_creates_directories(self, tmp_path):
        """Vault.__init__ creates the expected directory tree."""
        vault = Vault(str(tmp_path))
        jotty_dir = tmp_path / "jotty_state"
        assert jotty_dir.is_dir()
        assert (jotty_dir / "markovian_todos").is_dir()
        assert (jotty_dir / "q_tables").is_dir()
        assert (jotty_dir / "memories" / "local_memories").is_dir()
        assert (jotty_dir / "episode_history").is_dir()
        assert (jotty_dir / "brain_state").is_dir()

    def test_init_auto_save_interval_default(self, tmp_path):
        vault = Vault(str(tmp_path))
        assert vault.auto_save_interval == 10

    def test_init_custom_auto_save_interval(self, tmp_path):
        vault = Vault(str(tmp_path), auto_save_interval=5)
        assert vault.auto_save_interval == 5

    # -- save_markovian_todo ------------------------------------------------

    def test_save_markovian_todo_creates_json_and_md(self, tmp_path):
        vault = Vault(str(tmp_path))
        todo = _make_mock_todo()
        vault.save_markovian_todo(todo)

        state_file = tmp_path / "jotty_state" / "markovian_todos" / "todo_state.json"
        display_file = tmp_path / "jotty_state" / "markovian_todos" / "todo_display.md"
        assert state_file.exists()
        assert display_file.exists()

        data = json.loads(state_file.read_text())
        assert data["root_task"] == "Root Task"
        assert "task_1" in data["subtasks"]

    def test_save_markovian_todo_serialises_task_fields(self, tmp_path):
        vault = Vault(str(tmp_path))
        todo = _make_mock_todo()
        vault.save_markovian_todo(todo)

        data = json.loads(
            (tmp_path / "jotty_state" / "markovian_todos" / "todo_state.json").read_text()
        )
        task_data = data["subtasks"]["task_1"]
        assert task_data["actor"] == "TestActor"
        assert task_data["status"] == "pending"
        assert task_data["priority"] == 0.8

    def test_save_markovian_todo_markdown_contains_overview(self, tmp_path):
        vault = Vault(str(tmp_path))
        todo = _make_mock_todo()
        vault.save_markovian_todo(todo)

        md = (tmp_path / "jotty_state" / "markovian_todos" / "todo_display.md").read_text()
        assert "Root Task" in md
        assert "Progress Overview" in md

    # -- save_q_predictor ---------------------------------------------------

    def test_save_q_predictor(self, tmp_path):
        vault = Vault(str(tmp_path))
        q_predictor = Mock()
        q_predictor.experience_buffer = [{"state": "s1", "reward": 0.5}]
        vault.save_q_predictor(q_predictor)

        buf_file = tmp_path / "jotty_state" / "q_tables" / "q_predictor_buffer.json"
        assert buf_file.exists()
        data = json.loads(buf_file.read_text())
        assert data["buffer_size"] == 1
        assert len(data["experience_buffer"]) == 1

    # -- save_memory --------------------------------------------------------

    def test_save_memory_shared(self, tmp_path):
        vault = Vault(str(tmp_path))
        memory = Mock()
        memory.storage = {
            "episodic": [{"content": "hello", "context": {}, "goal": "g", "value": 1.0, "timestamp": 0}]
        }
        vault.save_memory(memory, name="shared")

        mem_file = tmp_path / "jotty_state" / "memories" / "shared_memory.json"
        assert mem_file.exists()
        data = json.loads(mem_file.read_text())
        assert "episodic" in data
        assert data["metadata"]["name"] == "shared"

    def test_save_memory_local(self, tmp_path):
        vault = Vault(str(tmp_path))
        memory = Mock()
        memory.storage = {"semantic": [{"content": "fact", "context": {}, "goal": "", "value": 0, "timestamp": 0}]}
        vault.save_memory(memory, name="loader")

        local_file = tmp_path / "jotty_state" / "memories" / "local_memories" / "loader.json"
        assert local_file.exists()

    def test_save_memory_max_per_level(self, tmp_path):
        vault = Vault(str(tmp_path))
        entries = [{"content": f"m{i}", "context": {}, "goal": "", "value": 0, "timestamp": 0} for i in range(200)]
        memory = Mock()
        memory.storage = {"level": entries}
        vault.save_memory(memory, name="shared", max_per_level=50)

        data = json.loads((tmp_path / "jotty_state" / "memories" / "shared_memory.json").read_text())
        assert len(data["level"]) == 50

    # -- save_brain_state ---------------------------------------------------

    def test_save_brain_state(self, tmp_path):
        vault = Vault(str(tmp_path))
        brain = Mock(spec=[])
        brain.preset = Mock(value="default")
        brain.chunk_size = 10
        brain.consolidation_count = 3
        brain.sleep_interval = 60
        vault.save_brain_state(brain)

        brain_file = tmp_path / "jotty_state" / "brain_state" / "consolidated_memories.json"
        assert brain_file.exists()
        data = json.loads(brain_file.read_text())
        assert data["preset"] == "default"
        assert data["chunk_size"] == 10

    def test_save_brain_state_none(self, tmp_path):
        vault = Vault(str(tmp_path))
        vault.save_brain_state(None)  # Should not raise

    # -- save_episode -------------------------------------------------------

    def test_save_episode(self, tmp_path):
        vault = Vault(str(tmp_path))
        trajectory = [{"step": 1, "action": "a1"}]
        metadata = {"total_episodes": 5}
        vault.save_episode(1, trajectory, metadata)

        ep_file = tmp_path / "jotty_state" / "episode_history" / "episode_1.json"
        assert ep_file.exists()
        data = json.loads(ep_file.read_text())
        assert data["episode"] == 1
        assert data["trajectory_length"] == 1

    # -- save_all -----------------------------------------------------------

    def test_save_all(self, tmp_path):
        vault = Vault(str(tmp_path))
        conductor = Mock()
        conductor.todo = _make_mock_todo()
        conductor.q_predictor = Mock(experience_buffer=[])
        conductor.shared_memory = Mock(storage={})
        conductor.local_memories = {}
        conductor.episode_count = 1
        conductor.trajectory = []
        conductor.brain = None
        vault.save_all(conductor)

        assert (tmp_path / "jotty_state" / "markovian_todos" / "todo_state.json").exists()
        assert (tmp_path / "jotty_state" / "q_tables" / "q_predictor_buffer.json").exists()

    # -- should_auto_save ---------------------------------------------------

    def test_should_auto_save(self, tmp_path):
        vault = Vault(str(tmp_path), auto_save_interval=5)
        assert vault.should_auto_save(0) is True
        assert vault.should_auto_save(5) is True
        assert vault.should_auto_save(3) is False


# ===========================================================================
# 2. SessionManager (core/persistence/session_manager.py)
# ===========================================================================

@pytest.mark.unit
@pytest.mark.skipif(not HAS_SESSION_MANAGER, reason="SessionManager not importable")
class TestSessionManager:
    """Tests for the SessionManager."""

    def test_init_creates_session_folder(self, tmp_path):
        config = _make_mock_config(tmp_path)
        sm = SessionManager(config)
        assert sm.session_dir.exists()
        assert sm.session_dir.name.startswith("run_")

    def test_init_creates_latest_symlink(self, tmp_path):
        config = _make_mock_config(tmp_path)
        sm = SessionManager(config)
        latest = tmp_path / "latest"
        assert latest.is_symlink()
        assert latest.resolve() == sm.session_dir.resolve()

    def test_init_no_run_folder(self, tmp_path):
        config = _make_mock_config(tmp_path, create_run_folder=False)
        sm = SessionManager(config)
        assert sm.session_dir == tmp_path

    def test_setup_directories(self, tmp_path):
        config = _make_mock_config(tmp_path)
        sm = SessionManager(config)
        assert (sm.session_dir / "jotty_state" / "memories" / "agent_memories").is_dir()
        assert (sm.session_dir / "jotty_state" / "q_tables").is_dir()
        assert (sm.session_dir / "logs" / "raw").is_dir()
        assert (sm.session_dir / "results").is_dir()

    def test_save_config_snapshot(self, tmp_path):
        config = _make_mock_config(tmp_path)
        sm = SessionManager(config)
        snapshot = sm.session_dir / "config_snapshot.json"
        assert snapshot.exists()

    # -- load_previous_state ------------------------------------------------

    def test_load_previous_state_disabled(self, tmp_path):
        config = _make_mock_config(tmp_path, auto_load_on_start=False)
        sm = SessionManager(config)
        result = sm.load_previous_state()
        assert result is None

    def test_load_previous_state_no_latest(self, tmp_path):
        config = _make_mock_config(tmp_path, auto_load_on_start=True)
        sm = SessionManager(config)
        # Remove the latest symlink so "no previous state" path is taken
        latest = tmp_path / "latest"
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        result = sm.load_previous_state()
        assert result is None

    def test_load_previous_state_with_data(self, tmp_path):
        """Create a previous run with memories, then load them."""
        config = _make_mock_config(tmp_path, auto_load_on_start=True)
        sm = SessionManager(config)

        # Write sample memories to the current session
        mem_dir = sm.session_dir / "jotty_state" / "memories"
        (mem_dir / "shared_memory.json").write_text(json.dumps({"key": "value"}))
        agent_dir = mem_dir / "agent_memories"
        agent_dir.mkdir(parents=True, exist_ok=True)
        (agent_dir / "AgentA.json").write_text(json.dumps({"mem": 1}))

        # Write q-table
        q_dir = sm.session_dir / "jotty_state" / "q_tables"
        (q_dir / "predictor.json").write_text(json.dumps({"q": 1}))

        # Now load -- latest should point at current session
        result = sm.load_previous_state()
        assert result is not None
        assert "memories" in result
        assert result["memories"]["shared"] == {"key": "value"}
        assert "AgentA" in result["memories"]["agents"]
        assert "q_tables" in result

    # -- save_state ---------------------------------------------------------

    def test_save_state_memories(self, tmp_path):
        config = _make_mock_config(tmp_path)
        sm = SessionManager(config)
        state = {
            "memories": {
                "shared": {"data": "hello"},
                "agents": {"A": {"m": 1}},
            }
        }
        sm.save_state(state)
        shared_file = sm.session_dir / "jotty_state" / "memories" / "shared_memory.json"
        assert shared_file.exists()
        assert json.loads(shared_file.read_text())["data"] == "hello"

    def test_save_state_q_tables(self, tmp_path):
        config = _make_mock_config(tmp_path)
        sm = SessionManager(config)
        sm.save_state({"q_tables": {"predictor": {"q": 42}}})
        q_file = sm.session_dir / "jotty_state" / "q_tables" / "predictor.json"
        assert q_file.exists()

    def test_save_state_brain(self, tmp_path):
        config = _make_mock_config(tmp_path)
        sm = SessionManager(config)
        sm.save_state({"brain_state": {"consolidated": True}})
        brain_file = sm.session_dir / "jotty_state" / "brain_state" / "consolidated.json"
        assert brain_file.exists()

    def test_save_state_todos(self, tmp_path):
        config = _make_mock_config(tmp_path)
        sm = SessionManager(config)
        sm.save_state({"todos": "# TODO\n- something"})
        todo_file = sm.session_dir / "jotty_state" / "markovian_todos" / "session_todo.md"
        assert todo_file.exists()
        assert "something" in todo_file.read_text()

    # -- get_session_info ---------------------------------------------------

    def test_get_session_info(self, tmp_path):
        config = _make_mock_config(tmp_path)
        sm = SessionManager(config)
        info = sm.get_session_info()
        assert "session_dir" in info
        assert "session_name" in info
        assert "has_memories" in info

    # -- cleanup_old_runs ---------------------------------------------------

    def test_cleanup_old_runs(self, tmp_path):
        config = _make_mock_config(tmp_path, max_runs_to_keep=2)
        # Create 4 run folders
        for i in range(4):
            d = tmp_path / f"run_2025010{i}_000000"
            d.mkdir()
            (d / "marker.txt").write_text(str(i))
            # Touch to set modification time
            os.utime(d, (time.time() + i, time.time() + i))

        sm = SessionManager(config)
        sm.cleanup_old_runs()

        remaining = sorted([d for d in tmp_path.glob("run_*") if d.is_dir()])
        # Should keep newest 2 + the session's own run folder
        assert len(remaining) <= 3

    def test_cleanup_old_runs_no_limit(self, tmp_path):
        config = _make_mock_config(tmp_path, max_runs_to_keep=0)
        sm = SessionManager(config)
        sm.cleanup_old_runs()  # Should not raise / do nothing

    # -- utility methods ----------------------------------------------------

    def test_save_results(self, tmp_path):
        config = _make_mock_config(tmp_path)
        sm = SessionManager(config)
        sm.save_results({"query": "SELECT 1", "rows": 42})
        result_file = sm.session_dir / "results" / "query_results.json"
        assert result_file.exists()

    def test_get_log_path(self, tmp_path):
        config = _make_mock_config(tmp_path)
        sm = SessionManager(config)
        log_path = sm.get_log_path("raw")
        assert "raw" in str(log_path)


# ===========================================================================
# 3. ScratchpadPersistence (core/persistence/scratchpad_persistence.py)
# ===========================================================================

@pytest.mark.unit
@pytest.mark.skipif(
    not (HAS_SCRATCHPAD and HAS_AGENT_TYPES),
    reason="ScratchpadPersistence or AgentTypes not importable",
)
class TestScratchpadPersistence:
    """Tests for the ScratchpadPersistence."""

    def _make_message(self, sender="A", receiver="B", content=None):
        return AgentMessage(
            sender=sender,
            receiver=receiver,
            message_type=CommunicationType.INSIGHT,
            content=content or {"text": "hello"},
            timestamp=datetime(2026, 1, 1, 12, 0, 0),
        )

    def test_init_creates_workspace(self, tmp_path):
        sp = ScratchpadPersistence(workspace_dir=str(tmp_path / "sp"))
        assert (tmp_path / "sp").is_dir()

    def test_create_session_default_name(self, tmp_path):
        sp = ScratchpadPersistence(workspace_dir=str(tmp_path / "sp"))
        session_file = sp.create_session()
        assert session_file.exists()
        assert session_file.suffix == ".jsonl"

    def test_create_session_custom_name(self, tmp_path):
        sp = ScratchpadPersistence(workspace_dir=str(tmp_path / "sp"))
        session_file = sp.create_session("my_session")
        assert session_file.name == "my_session.jsonl"

    def test_save_message(self, tmp_path):
        sp = ScratchpadPersistence(workspace_dir=str(tmp_path / "sp"))
        session_file = sp.create_session("test")
        msg = self._make_message()
        sp.save_message(session_file, msg)

        lines = session_file.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["sender"] == "A"
        assert data["receiver"] == "B"

    def test_save_scratchpad(self, tmp_path):
        sp = ScratchpadPersistence(workspace_dir=str(tmp_path / "sp"))
        session_file = sp.create_session("test")

        scratchpad = SharedScratchpad()
        msg = self._make_message()
        scratchpad.add_message(msg)
        scratchpad.shared_insights = ["insight1"]

        sp.save_scratchpad(session_file, scratchpad)

        lines = session_file.read_text().strip().split("\n")
        # 1 message + 1 metadata
        assert len(lines) == 2
        metadata = json.loads(lines[-1])
        assert metadata["type"] == "metadata"
        assert metadata["total_messages"] == 1

    def test_load_scratchpad(self, tmp_path):
        sp = ScratchpadPersistence(workspace_dir=str(tmp_path / "sp"))
        session_file = sp.create_session("test")

        scratchpad = SharedScratchpad()
        msg = self._make_message()
        scratchpad.add_message(msg)
        scratchpad.shared_insights = ["insight1"]
        sp.save_scratchpad(session_file, scratchpad)

        loaded = sp.load_scratchpad(session_file)
        assert len(loaded.messages) == 1
        assert loaded.messages[0].sender == "A"
        assert loaded.shared_insights == ["insight1"]

    def test_load_scratchpad_missing_file(self, tmp_path):
        sp = ScratchpadPersistence(workspace_dir=str(tmp_path / "sp"))
        loaded = sp.load_scratchpad(tmp_path / "nonexistent.jsonl")
        assert len(loaded.messages) == 0

    def test_list_sessions(self, tmp_path):
        sp = ScratchpadPersistence(workspace_dir=str(tmp_path / "sp"))
        sp.create_session("session_a")
        sp.create_session("session_b")
        sessions = sp.list_sessions()
        assert len(sessions) == 2

    def test_export_session_json(self, tmp_path):
        sp = ScratchpadPersistence(workspace_dir=str(tmp_path / "sp"))
        session_file = sp.create_session("test")

        scratchpad = SharedScratchpad()
        msg = self._make_message()
        scratchpad.add_message(msg)
        sp.save_scratchpad(session_file, scratchpad)

        exported = sp.export_session(session_file, output_format="json")
        data = json.loads(exported)
        assert data["total_messages"] == 1
        assert len(data["messages"]) == 1

    def test_export_session_markdown(self, tmp_path):
        sp = ScratchpadPersistence(workspace_dir=str(tmp_path / "sp"))
        session_file = sp.create_session("test")

        scratchpad = SharedScratchpad()
        msg = self._make_message()
        scratchpad.add_message(msg)
        sp.save_scratchpad(session_file, scratchpad)

        exported = sp.export_session(session_file, output_format="markdown")
        assert "Message 1" in exported
        assert "From" in exported

    def test_export_session_invalid_format(self, tmp_path):
        sp = ScratchpadPersistence(workspace_dir=str(tmp_path / "sp"))
        session_file = sp.create_session("test")
        scratchpad = SharedScratchpad()
        sp.save_scratchpad(session_file, scratchpad)

        with pytest.raises(ValueError, match="Unknown format"):
            sp.export_session(session_file, output_format="xml")


# ===========================================================================
# 4. DataRegistry (core/data/data_registry.py)
# ===========================================================================

@pytest.mark.unit
@pytest.mark.skipif(not HAS_DATA_REGISTRY, reason="DataRegistry not importable")
class TestDataArtifact:
    """Tests for the DataArtifact dataclass."""

    def test_default_fields(self):
        artifact = DataArtifact(
            id="a1", name="Test", source_actor="Actor1",
            data="hello", data_type="str",
        )
        assert artifact.id == "a1"
        assert artifact.tags == []
        assert artifact.confidence == 1.0

    def test_matches_query_type(self):
        artifact = DataArtifact(
            id="a1", name="Test", source_actor="Actor1",
            data="hello", data_type="html",
        )
        assert artifact.matches_query({"type": "html"}) == pytest.approx(0.5)
        assert artifact.matches_query({"type": "json"}) == pytest.approx(0.0)

    def test_matches_query_tag(self):
        artifact = DataArtifact(
            id="a1", name="Test", source_actor="Actor1",
            data="hello", data_type="str", tags=["finance", "report"],
        )
        assert artifact.matches_query({"tag": "finance"}) == pytest.approx(0.3)
        assert artifact.matches_query({"tag": "unknown"}) == pytest.approx(0.0)

    def test_matches_query_tags_list(self):
        artifact = DataArtifact(
            id="a1", name="Test", source_actor="Actor1",
            data="hello", data_type="str", tags=["finance", "report"],
        )
        score = artifact.matches_query({"tags": ["finance", "report", "chart"]})
        assert score == pytest.approx(0.2, abs=0.01)

    def test_matches_query_actor(self):
        artifact = DataArtifact(
            id="a1", name="Test", source_actor="Actor1",
            data="hello", data_type="str",
        )
        assert artifact.matches_query({"actor": "Actor1"}) == pytest.approx(0.4)

    def test_matches_query_fields(self):
        artifact = DataArtifact(
            id="a1", name="Test", source_actor="Actor1",
            data="hello", data_type="str",
            schema={"col1": "int", "col2": "str"},
        )
        score = artifact.matches_query({"fields": ["col1"]})
        assert score == pytest.approx(0.2, abs=0.01)

    def test_matches_query_combined(self):
        artifact = DataArtifact(
            id="a1", name="Test", source_actor="Actor1",
            data="hello", data_type="html",
            tags=["finance"],
        )
        score = artifact.matches_query({"type": "html", "tag": "finance", "actor": "Actor1"})
        assert score == pytest.approx(1.2, abs=0.01)


@pytest.mark.unit
@pytest.mark.skipif(not HAS_DATA_REGISTRY, reason="DataRegistry not importable")
class TestDataRegistry:
    """Tests for the DataRegistry."""

    def _make_artifact(self, id="a1", name="Art1", actor="Actor1",
                       data_type="html", tags=None):
        return DataArtifact(
            id=id, name=name, source_actor=actor,
            data="<div>test</div>", data_type=data_type,
            tags=tags or [],
        )

    def test_register_and_get(self):
        reg = DataRegistry()
        art = self._make_artifact()
        reg.register(art)
        assert reg.get("a1") is art

    def test_get_nonexistent(self):
        reg = DataRegistry()
        assert reg.get("missing") is None

    def test_get_by_name(self):
        reg = DataRegistry()
        art = self._make_artifact()
        reg.register(art)
        assert reg.get_by_name("Art1") is art

    def test_get_by_name_most_recent(self):
        reg = DataRegistry()
        art1 = self._make_artifact(id="a1", name="Report", data_type="html")
        art1.timestamp = 1.0
        art2 = self._make_artifact(id="a2", name="Report", data_type="html")
        art2.timestamp = 2.0
        reg.register(art1)
        reg.register(art2)
        result = reg.get_by_name("Report")
        assert result.id == "a2"

    def test_register_updates_indices(self):
        reg = DataRegistry()
        art = self._make_artifact(tags=["finance", "report"])
        reg.register(art)
        assert "html" in reg.index_by_type
        assert "finance" in reg.index_by_tag
        assert "Actor1" in reg.index_by_actor

    def test_discover(self):
        reg = DataRegistry()
        reg.register(self._make_artifact())
        result = reg.discover()
        assert result["total_artifacts"] == 1
        assert "html" in result["types_available"]
        assert len(result["artifacts"]) == 1

    def test_search_by_type(self):
        reg = DataRegistry()
        reg.register(self._make_artifact(id="a1", data_type="html", tags=["x"]))
        reg.register(self._make_artifact(id="a2", data_type="json", tags=["y"]))
        results = reg.search({"type": "html"})
        assert len(results) == 1
        assert results[0].id == "a1"

    def test_search_by_tag(self):
        reg = DataRegistry()
        reg.register(self._make_artifact(id="a1", tags=["finance"]))
        reg.register(self._make_artifact(id="a2", tags=["tech"]))
        results = reg.search({"tag": "finance"})
        assert len(results) == 1
        assert results[0].id == "a1"

    def test_search_by_actor(self):
        reg = DataRegistry()
        reg.register(self._make_artifact(id="a1", actor="Alpha"))
        reg.register(self._make_artifact(id="a2", actor="Beta"))
        results = reg.search({"actor": "Alpha"})
        assert len(results) == 1

    def test_search_full(self):
        reg = DataRegistry()
        reg.register(self._make_artifact(id="a1", tags=["x"]))
        results = reg.search({"tag": "x"})
        assert len(results) == 1

    def test_list_types(self):
        reg = DataRegistry()
        reg.register(self._make_artifact(id="a1", data_type="html"))
        reg.register(self._make_artifact(id="a2", data_type="json"))
        assert sorted(reg.list_types()) == ["html", "json"]

    def test_list_tags(self):
        reg = DataRegistry()
        reg.register(self._make_artifact(id="a1", tags=["a", "b"]))
        assert sorted(reg.list_tags()) == ["a", "b"]

    def test_list_actors(self):
        reg = DataRegistry()
        reg.register(self._make_artifact(id="a1", actor="X"))
        reg.register(self._make_artifact(id="a2", actor="Y"))
        assert sorted(reg.list_actors()) == ["X", "Y"]


# ===========================================================================
# 5. AgenticParameterResolver (core/data/parameter_resolver.py)
# ===========================================================================

@pytest.mark.unit
@pytest.mark.skipif(not HAS_PARAM_RESOLVER, reason="AgenticParameterResolver not importable")
class TestAgenticParameterResolver:
    """Tests for the AgenticParameterResolver (mocked LLM)."""

    def _make_resolver(self):
        """Create a resolver with mocked internals."""
        with patch("dspy.settings"):
            resolver = AgenticParameterResolver.__new__(AgenticParameterResolver)
            resolver.llm = Mock()
            resolver.resolver = Mock()
            return resolver

    def test_resolve_parameter_match(self):
        resolver = self._make_resolver()
        resolver.resolver.return_value = Mock(
            best_match_key="html_data",
            confidence=0.9,
            reasoning="Semantic match found",
            semantic_explanation="The HTML data matches the need",
        )

        key, confidence, explanation = resolver.resolve_parameter(
            actor_name="Renderer",
            parameter_name="html_content",
            parameter_type=str,
            parameter_purpose="Render the HTML page",
            available_data={"html_data": {"value": "<div/>", "type": "str", "description": "Page HTML", "tags": [], "source": "Scraper"}},
        )
        assert key == "html_data"
        assert confidence == pytest.approx(0.9)

    def test_resolve_parameter_no_match(self):
        resolver = self._make_resolver()
        resolver.resolver.return_value = Mock(
            best_match_key="NO_MATCH",
            confidence=0.3,
            reasoning="Nothing matched semantically",
            semantic_explanation="No match found",
        )

        key, confidence, reasoning = resolver.resolve_parameter(
            actor_name="Renderer",
            parameter_name="html_content",
            parameter_type=str,
            parameter_purpose="Render page",
            available_data={},
        )
        assert key is None
        assert confidence == pytest.approx(0.3)

    def test_resolve_parameter_low_confidence(self):
        resolver = self._make_resolver()
        resolver.resolver.return_value = Mock(
            best_match_key="some_key",
            confidence=0.5,
            reasoning="Partial match",
            semantic_explanation="Somewhat related",
        )

        key, confidence, reasoning = resolver.resolve_parameter(
            actor_name="Renderer",
            parameter_name="data",
            parameter_type=list,
            parameter_purpose="Process data",
            available_data={"some_key": {"value": [], "type": "list", "description": "data", "tags": [], "source": "x"}},
            min_confidence=0.7,
        )
        assert key is None

    def test_resolve_parameter_exception(self):
        resolver = self._make_resolver()
        resolver.resolver.side_effect = RuntimeError("LLM down")

        key, confidence, reasoning = resolver.resolve_parameter(
            actor_name="Renderer",
            parameter_name="data",
            parameter_type=str,
            parameter_purpose="Render",
            available_data={},
        )
        assert key is None
        assert confidence == 0.0
        assert "Resolver failed" in reasoning

    def test_build_self_describing_data(self):
        resolver = self._make_resolver()
        data = resolver.build_self_describing_data(
            key="my_key",
            value=[1, 2, 3],
            description="A list of numbers",
            tags=["numbers", "input"],
            source="Calculator",
        )
        assert data["key"] == "my_key"
        assert data["type"] == "list"
        assert "numbers" in data["tags"]
        assert data["source"] == "Calculator"


# ===========================================================================
# 6. AgenticFeedbackRouter (core/data/feedback_router.py)
# ===========================================================================

@pytest.mark.unit
@pytest.mark.skipif(
    not (HAS_FEEDBACK_ROUTER and HAS_FEEDBACK_CHANNEL),
    reason="AgenticFeedbackRouter or FeedbackChannel not importable",
)
class TestAgenticFeedbackRouter:
    """Tests for the AgenticFeedbackRouter (mocked LLM)."""

    def _make_router(self):
        lm = Mock()
        router = AgenticFeedbackRouter(lm)
        router.router = Mock()
        return router

    # -- helper method tests -----------------------------------------------

    def test_describe_agents(self):
        router = self._make_router()
        agents = {
            "SQLGen": {"provides": ["sql"], "dependencies": ["schema"], "goal": "Generate SQL"},
            "Resolver": {"provides": ["terms"], "dependencies": [], "goal": "Resolve terms"},
        }
        desc = router._describe_agents(agents)
        assert "SQLGen" in desc
        assert "Generate SQL" in desc

    def test_describe_dependencies(self):
        router = self._make_router()
        deps = {"SQLGen": ["Resolver", "Schema"], "Resolver": []}
        desc = router._describe_dependencies(deps)
        assert "SQLGen depends on" in desc

    def test_describe_swarm_state(self):
        router = self._make_router()
        state = {
            "completed_tasks": ["TaskA"],
            "pending_tasks": ["TaskB"],
            "failed_tasks": ["TaskC"],
        }
        desc = router._describe_swarm_state(state)
        assert "TaskA" in desc
        assert "TaskC" in desc

    def test_describe_scratchpad(self):
        router = self._make_router()
        scratchpad = {"Agent1": "output data here"}
        desc = router._describe_scratchpad(scratchpad)
        assert "Agent1" in desc

    def test_parse_target_agents_json_valid(self):
        router = self._make_router()
        result = router._parse_target_agents_json('["AgentA", "AgentB"]')
        assert result == ["AgentA", "AgentB"]

    def test_parse_target_agents_json_empty(self):
        router = self._make_router()
        result = router._parse_target_agents_json("[]")
        assert result == []

    def test_parse_target_agents_json_invalid(self):
        router = self._make_router()
        result = router._parse_target_agents_json("not json at all")
        assert result == []

    def test_parse_target_agents_json_deduplicates(self):
        router = self._make_router()
        result = router._parse_target_agents_json('["A", "A", "B"]')
        assert result == ["A", "B"]

    def test_parse_target_agents_json_filters_non_strings(self):
        router = self._make_router()
        result = router._parse_target_agents_json('[123, "A", null]')
        assert result == ["A"]

    def test_parse_feedback_type_error(self):
        """Source code references FeedbackType.ERROR_CORRECTION which does not exist.
        This test documents the known bug -- the method raises AttributeError."""
        router = self._make_router()
        with pytest.raises(AttributeError):
            router._parse_feedback_type("error_correction")

    def test_parse_feedback_type_default(self):
        """Default path also references FeedbackType.ERROR_CORRECTION (known bug)."""
        router = self._make_router()
        with pytest.raises(AttributeError):
            router._parse_feedback_type("unknown_type")

    def test_routing_history_starts_empty(self):
        router = self._make_router()
        assert router.routing_history == []

    def test_get_routing_history_context_empty(self):
        router = self._make_router()
        context = router._get_routing_history_context()
        assert context == ""

    def test_get_routing_history_context_with_entries(self):
        router = self._make_router()
        router.routing_history = [
            {"failing_actor": "A", "error": "timeout", "targets": ["B"], "reasoning": "B can help"},
        ]
        context = router._get_routing_history_context()
        assert "Past Routing Decisions" in context
        assert "timeout" in context


# ===========================================================================
# 7. InformationTheoreticStorage (core/data/information_storage.py)
# ===========================================================================

@pytest.mark.unit
@pytest.mark.skipif(not HAS_INFO_STORAGE, reason="InformationTheoreticStorage not importable")
class TestInformationWeightedMemory:
    """Tests for the InformationWeightedMemory dataclass."""

    def test_default_creation(self):
        mem = InformationWeightedMemory(
            key="k1", content="test", information_content=2.0,
            frequency_estimate=0.25, llm_surprise=0.5, detail_level="normal",
        )
        assert mem.key == "k1"
        assert mem.detail_level == "normal"

    def test_negative_info_content_clamped(self):
        mem = InformationWeightedMemory(
            key="k1", content="test", information_content=-1.0,
            frequency_estimate=0.5, llm_surprise=0.5, detail_level="minimal",
        )
        assert mem.information_content == 0


@pytest.mark.unit
@pytest.mark.skipif(not HAS_INFO_STORAGE, reason="InformationTheoreticStorage not importable")
class TestSurpriseEstimator:
    """Tests for the SurpriseEstimator."""

    @pytest.mark.asyncio
    async def test_fallback_failure_event(self):
        """Without dspy, failures should return higher surprise."""
        estimator = SurpriseEstimator()
        estimator.estimator = None  # Force fallback
        score, reason = await estimator.estimate_surprise(
            event={"outcome": "failure"},
            context={},
            historical_patterns=[],
        )
        assert score == pytest.approx(0.7)
        assert "less common" in reason.lower()

    @pytest.mark.asyncio
    async def test_fallback_success_event(self):
        estimator = SurpriseEstimator()
        estimator.estimator = None
        score, reason = await estimator.estimate_surprise(
            event={"success": True},
            context={},
            historical_patterns=[],
        )
        assert score == pytest.approx(0.3)

    @pytest.mark.asyncio
    async def test_fallback_error_event(self):
        estimator = SurpriseEstimator()
        estimator.estimator = None
        score, _ = await estimator.estimate_surprise(
            event={"outcome": "error occurred"},
            context={},
            historical_patterns=[],
        )
        assert score == pytest.approx(0.7)

    def test_parse_score_float(self):
        estimator = SurpriseEstimator()
        assert estimator._parse_score(0.42) == pytest.approx(0.42)

    def test_parse_score_string(self):
        estimator = SurpriseEstimator()
        assert estimator._parse_score("0.75") == pytest.approx(0.75)

    def test_parse_score_invalid_string(self):
        estimator = SurpriseEstimator()
        # Should return 0.5 as default
        assert estimator._parse_score("not a number") == pytest.approx(0.5)

    def test_parse_score_embedded_number(self):
        estimator = SurpriseEstimator()
        result = estimator._parse_score("about 0.8 surprise")
        assert result == pytest.approx(0.8, abs=0.01)


@pytest.mark.unit
@pytest.mark.skipif(not HAS_INFO_STORAGE, reason="InformationTheoreticStorage not importable")
class TestInformationTheoreticStorage:
    """Tests for the InformationTheoreticStorage class."""

    def test_init_defaults(self):
        storage = InformationTheoreticStorage()
        assert storage.alpha == 0.5
        assert storage.min_info_threshold == 0.5
        assert storage.max_info_threshold == 3.0
        assert storage.total_events == 0
        assert len(storage.memories) == 0

    def test_init_custom_params(self):
        storage = InformationTheoreticStorage(alpha=0.7, min_info_threshold=1.0, max_info_threshold=5.0)
        assert storage.alpha == 0.7
        assert storage.min_info_threshold == 1.0
        assert storage.max_info_threshold == 5.0

    def test_compute_event_signature_deterministic(self):
        storage = InformationTheoreticStorage()
        event = {"type": "action", "outcome": "success", "agent": "A"}
        sig1 = storage._compute_event_signature(event)
        sig2 = storage._compute_event_signature(event)
        assert sig1 == sig2

    def test_compute_event_signature_different_events(self):
        storage = InformationTheoreticStorage()
        sig1 = storage._compute_event_signature({"type": "action", "outcome": "success", "agent": "A"})
        sig2 = storage._compute_event_signature({"type": "action", "outcome": "failure", "agent": "A"})
        assert sig1 != sig2

    def test_summarize_event(self):
        storage = InformationTheoreticStorage()
        summary = storage._summarize_event({"agent": "TestAgent", "action": "fetch", "outcome": "ok"})
        assert "TestAgent" in summary
        assert "fetch" in summary
        assert "ok" in summary

    def test_generate_key_unique(self):
        storage = InformationTheoreticStorage()
        event = {"agent": "A", "action": "run"}
        key1 = storage._generate_key(event)
        assert isinstance(key1, str)
        assert len(key1) == 32  # MD5 hex digest

    def test_maximum_detail(self):
        storage = InformationTheoreticStorage()
        content = storage._maximum_detail(
            {"type": "failure", "agent": "A"},
            {"task": "research"},
            "Raw error log text",
        )
        assert "HIGH INFORMATION EVENT" in content
        assert "failure" in content
        assert "Raw error log text" in content

    def test_high_detail(self):
        storage = InformationTheoreticStorage()
        content = storage._high_detail(
            {"type": "warning"},
            {"context": "c"},
            "Some important content",
        )
        assert "NOTABLE EVENT" in content

    def test_normal_detail(self):
        storage = InformationTheoreticStorage()
        content = storage._normal_detail(
            {"type": "success", "outcome": "done"},
            "Brief summary",
        )
        assert "success" in content
        assert "done" in content

    def test_minimal_detail(self):
        storage = InformationTheoreticStorage()
        content = storage._minimal_detail({"agent": "Bot", "outcome": "ok"})
        assert "Bot" in content
        assert "ok" in content

    @pytest.mark.asyncio
    async def test_compute_information_content(self):
        """Test information content computation with mocked surprise."""
        storage = InformationTheoreticStorage(alpha=0.5)
        storage.surprise_estimator = Mock()
        storage.surprise_estimator.estimate_surprise = AsyncMock(return_value=(0.7, "Somewhat surprising"))

        event = {"type": "action", "outcome": "failure", "agent": "A"}
        context = {"task": "test"}

        info, detail_level, freq_est, surprise = await storage.compute_information_content(event, context)

        assert storage.total_events == 1
        assert surprise == pytest.approx(0.7)
        assert freq_est > 0
        assert info > 0
        assert detail_level in ("maximum", "high", "normal", "minimal")

    @pytest.mark.asyncio
    async def test_compute_information_content_repeated_events(self):
        """Repeated events should have lower information content when
        frequency rises relative to total events."""
        storage = InformationTheoreticStorage(alpha=0.5)
        storage.surprise_estimator = Mock()
        storage.surprise_estimator.estimate_surprise = AsyncMock(return_value=(0.3, "Common event"))

        event = {"type": "query", "outcome": "success", "agent": "B"}
        other_event = {"type": "write", "outcome": "success", "agent": "C"}

        # First occurrence: freq = 1/1 = 1.0
        info1, _, _, _ = await storage.compute_information_content(event, {})
        # Insert a different event so total_events grows but target event count does not
        await storage.compute_information_content(other_event, {})
        # Now event again: freq = 2/3 = 0.667 < 1.0 -- BUT that actually means
        # lower freq = higher info, not lower. Let us instead test that repeating
        # the SAME event many times (while other events also occur) converges to
        # a stable info value, and verify the formula produces expected numbers.
        # Simpler: just verify info content is positive and consistent with formula.
        info3, _, freq3, _ = await storage.compute_information_content(event, {})
        # freq should be 2/3
        assert freq3 == pytest.approx(2.0 / 3.0, abs=0.01)
        # After many repeats of only the same event, frequency -> 1.0 and info drops
        for _ in range(20):
            await storage.compute_information_content(event, {})
        info_final, _, freq_final, _ = await storage.compute_information_content(event, {})
        # freq should be high (22/24 ~ 0.917)
        assert freq_final > 0.9
        # With high freq (high P), info content should be LOW
        assert info_final < info3

    @pytest.mark.asyncio
    async def test_store(self):
        """Test storing a memory with mocked surprise."""
        storage = InformationTheoreticStorage()
        storage.surprise_estimator = Mock()
        storage.surprise_estimator.estimate_surprise = AsyncMock(return_value=(0.9, "Very surprising"))

        event = {"type": "crash", "outcome": "failure", "agent": "C"}
        memory = await storage.store(event, {"task": "deployment"}, "Full crash log")

        assert isinstance(memory, InformationWeightedMemory)
        assert memory.llm_surprise == pytest.approx(0.9)
        assert len(storage.memories) == 1

    @pytest.mark.asyncio
    async def test_store_detail_levels(self):
        """Test that detail level changes based on information content."""
        storage = InformationTheoreticStorage(
            alpha=0.0,  # Use only LLM surprise
            min_info_threshold=0.5,
            max_info_threshold=3.0,
        )
        storage.surprise_estimator = Mock()

        # Very surprising = high info content
        storage.surprise_estimator.estimate_surprise = AsyncMock(return_value=(0.95, "Very rare"))
        mem_high = await storage.store(
            {"type": "a", "outcome": "crash", "agent": "X"}, {}, "content"
        )
        assert mem_high.detail_level in ("maximum", "high")

        # Not surprising = low info content
        storage.surprise_estimator.estimate_surprise = AsyncMock(return_value=(0.1, "Very common"))
        mem_low = await storage.store(
            {"type": "b", "outcome": "success", "agent": "Y"}, {}, "content"
        )
        assert mem_low.detail_level in ("normal", "minimal")

    def test_get_statistics_empty(self):
        storage = InformationTheoreticStorage()
        stats = storage.get_statistics()
        assert stats["total_events"] == 0
        assert stats["memories_stored"] == 0
        assert stats["average_info_content"] == 0

    @pytest.mark.asyncio
    async def test_get_statistics_after_storage(self):
        storage = InformationTheoreticStorage()
        storage.surprise_estimator = Mock()
        storage.surprise_estimator.estimate_surprise = AsyncMock(return_value=(0.5, "Medium"))

        await storage.store({"type": "a", "outcome": "ok", "agent": "A"}, {}, "content")
        await storage.store({"type": "b", "outcome": "fail", "agent": "B"}, {}, "content")

        stats = storage.get_statistics()
        assert stats["total_events"] == 2
        assert stats["memories_stored"] == 2
        assert stats["average_info_content"] > 0
        assert isinstance(stats["detail_distribution"], dict)

    @pytest.mark.asyncio
    async def test_recent_patterns_capped(self):
        """Verify recent_patterns list respects max_patterns limit."""
        storage = InformationTheoreticStorage()
        storage.max_patterns = 5
        storage.surprise_estimator = Mock()
        storage.surprise_estimator.estimate_surprise = AsyncMock(return_value=(0.5, "ok"))

        for i in range(10):
            await storage.compute_information_content(
                {"type": f"t{i}", "outcome": "ok", "agent": "A"}, {}
            )

        assert len(storage.recent_patterns) <= 5
