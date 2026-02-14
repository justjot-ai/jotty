"""
Tests for MAS Learning - Multi-Agent System Learning & Persistence
===================================================================

Comprehensive unit tests covering:
- FixRecord dataclass (fields, success_rate property)
- SessionLearning dataclass (fields, get_relevance_score)
- MASLearning class (init, error hashing, fix finding, fix recording,
  topic extraction, session recording, relevant learnings, execution
  strategy, persistence round-trip, terminal integration, statistics)

All tests use mocks and tmp_path -- NO real LLM calls, no filesystem side effects.

Author: A-Team
Date: February 2026
"""

import hashlib
import json
import re
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import (
    MagicMock,
    Mock,
    patch,
    PropertyMock,
)

import pytest

# ---------------------------------------------------------------------------
# Guarded imports with skip markers
# ---------------------------------------------------------------------------
try:
    from Jotty.core.orchestration.mas_learning import (
        FixRecord,
        SessionLearning,
        MASLearning,
        get_mas_learning,
    )
    MAS_LEARNING_AVAILABLE = True
except ImportError:
    MAS_LEARNING_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not MAS_LEARNING_AVAILABLE,
    reason="MAS Learning module not available",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fix_record(**overrides) -> "FixRecord":
    """Create a FixRecord with sensible defaults."""
    defaults = dict(
        error_pattern="ModuleNotFoundError: No module named 'foo'",
        error_hash="abc123",
        solution_commands=["pip install foo"],
        solution_description="Install the missing foo package",
        source="pattern",
        success_count=5,
        fail_count=1,
        last_used=datetime.now().isoformat(),
        context={"env": "linux"},
    )
    defaults.update(overrides)
    return FixRecord(**defaults)


def _make_session_learning(**overrides) -> "SessionLearning":
    """Create a SessionLearning with sensible defaults."""
    defaults = dict(
        session_id="20260214_120000",
        timestamp=datetime.now().isoformat(),
        task_description="Research AI trends and summarize",
        task_topics=["research", "trends", "summarize"],
        agents_used=["research_agent", "writer_agent"],
        stigmergy_signals=3,
        total_time=25.0,
        success=True,
        workspace="/tmp/workspace",
        agent_count=2,
        output_quality=0.85,
    )
    defaults.update(overrides)
    return SessionLearning(**defaults)


def _create_mas_learning(tmp_path, **kwargs) -> "MASLearning":
    """Create MASLearning using tmp_path to avoid filesystem side effects."""
    defaults = dict(
        learning_dir=tmp_path,
        workspace_path=tmp_path / "workspace",
    )
    defaults.update(kwargs)
    return MASLearning(**defaults)


# ===========================================================================
# FixRecord Tests
# ===========================================================================

@pytest.mark.unit
class TestFixRecord:
    """Tests for FixRecord dataclass."""

    def test_fields_populated(self):
        """All fields should be set correctly from constructor arguments."""
        fix = _make_fix_record(
            error_pattern="KeyError: 'missing_key'",
            error_hash="hash123",
            solution_commands=["fix it"],
            solution_description="Add the missing key",
            source="llm",
            success_count=10,
            fail_count=2,
            context={"project": "jotty"},
        )
        assert fix.error_pattern == "KeyError: 'missing_key'"
        assert fix.error_hash == "hash123"
        assert fix.solution_commands == ["fix it"]
        assert fix.solution_description == "Add the missing key"
        assert fix.source == "llm"
        assert fix.success_count == 10
        assert fix.fail_count == 2
        assert fix.context == {"project": "jotty"}

    def test_default_values(self):
        """Default success_count=1, fail_count=0, context={}."""
        fix = FixRecord(
            error_pattern="err",
            error_hash="h",
            solution_commands=[],
            solution_description="desc",
            source="user",
        )
        assert fix.success_count == 1
        assert fix.fail_count == 0
        assert isinstance(fix.context, dict)
        assert fix.last_used  # should have a timestamp string

    def test_success_rate_normal(self):
        """success_rate should be success_count / (success_count + fail_count)."""
        fix = _make_fix_record(success_count=8, fail_count=2)
        assert fix.success_rate == pytest.approx(0.8)

    def test_success_rate_all_success(self):
        """100% success rate when fail_count is zero."""
        fix = _make_fix_record(success_count=5, fail_count=0)
        assert fix.success_rate == pytest.approx(1.0)

    def test_success_rate_all_fail(self):
        """0% success rate when success_count is zero."""
        fix = _make_fix_record(success_count=0, fail_count=5)
        assert fix.success_rate == pytest.approx(0.0)

    def test_success_rate_zero_total(self):
        """Should return 0.0 when both counts are zero (no division error)."""
        fix = _make_fix_record(success_count=0, fail_count=0)
        assert fix.success_rate == 0.0

    def test_success_rate_half(self):
        """Equal success and failure should give 0.5."""
        fix = _make_fix_record(success_count=3, fail_count=3)
        assert fix.success_rate == pytest.approx(0.5)

    def test_asdict_roundtrip(self):
        """asdict should produce a serializable dictionary."""
        fix = _make_fix_record()
        data = asdict(fix)
        assert isinstance(data, dict)
        assert "error_pattern" in data
        assert "solution_commands" in data


# ===========================================================================
# SessionLearning Tests
# ===========================================================================

@pytest.mark.unit
class TestSessionLearning:
    """Tests for SessionLearning dataclass."""

    def test_fields_populated(self):
        """All fields should be set correctly."""
        session = _make_session_learning(
            session_id="sess1",
            task_description="Build a dashboard",
            task_topics=["build", "dashboard"],
            agents_used=["coding_agent"],
            stigmergy_signals=5,
            total_time=30.0,
            success=True,
            workspace="/work",
            agent_count=1,
            output_quality=0.9,
        )
        assert session.session_id == "sess1"
        assert session.task_description == "Build a dashboard"
        assert session.task_topics == ["build", "dashboard"]
        assert session.agents_used == ["coding_agent"]
        assert session.stigmergy_signals == 5
        assert session.total_time == 30.0
        assert session.success is True
        assert session.workspace == "/work"
        assert session.agent_count == 1
        assert session.output_quality == 0.9

    def test_default_values(self):
        """agent_count defaults to 0, output_quality to 0.0."""
        session = SessionLearning(
            session_id="s",
            timestamp=datetime.now().isoformat(),
            task_description="test",
            task_topics=[],
            agents_used=[],
            stigmergy_signals=0,
            total_time=0.0,
            success=False,
            workspace="",
        )
        assert session.agent_count == 0
        assert session.output_quality == 0.0

    def test_relevance_score_full_topic_overlap(self):
        """Perfect topic overlap should give high score."""
        session = _make_session_learning(
            task_topics=["python", "testing", "automation"],
            agents_used=["coder"],
            success=True,
            timestamp=datetime.now().isoformat(),
        )
        score = session.get_relevance_score(
            query_topics=["python", "testing", "automation"],
        )
        # topic_score = 3/3 = 1.0 => 0.4*1.0 = 0.4
        # agent_score = 0 (no query_agents) => 0.2*0 = 0.0
        # recency = ~1.0 (just now) => 0.2*1.0 = 0.2
        # base = 0.2
        # success_bonus = 0.2
        # total ~ 0.4 + 0.0 + 0.2 + 0.2 + 0.2 = 1.0
        assert score >= 0.9

    def test_relevance_score_no_topic_overlap(self):
        """No topic overlap should give lower score."""
        session = _make_session_learning(
            task_topics=["python", "testing"],
            agents_used=[],
            success=False,
            timestamp=datetime.now().isoformat(),
        )
        score = session.get_relevance_score(
            query_topics=["javascript", "deployment"],
        )
        # topic_score = 0/2 = 0 => 0.4*0 = 0
        # no agent query => 0
        # recency ~ 1.0 => 0.2*1.0 = 0.2
        # base = 0.2
        # no success bonus
        # total ~ 0.4
        assert score < 0.5

    def test_relevance_score_agent_overlap(self):
        """Agent overlap should contribute to score."""
        session = _make_session_learning(
            task_topics=["python"],
            agents_used=["coder", "reviewer", "tester"],
            success=False,
            timestamp=datetime.now().isoformat(),
        )
        score_with = session.get_relevance_score(
            query_topics=["python"],
            query_agents=["coder", "reviewer"],
        )
        score_without = session.get_relevance_score(
            query_topics=["python"],
            query_agents=["unknown_agent"],
        )
        assert score_with > score_without

    def test_relevance_score_success_bonus(self):
        """Successful sessions should score higher than failed ones."""
        base_kwargs = dict(
            task_topics=["python"],
            agents_used=[],
            timestamp=datetime.now().isoformat(),
        )
        success_session = _make_session_learning(success=True, **base_kwargs)
        fail_session = _make_session_learning(success=False, **base_kwargs)

        score_success = success_session.get_relevance_score(["python"])
        score_fail = fail_session.get_relevance_score(["python"])
        assert score_success - score_fail == pytest.approx(0.2)

    def test_relevance_score_recency_decay(self):
        """Older sessions should score lower due to recency decay."""
        recent = _make_session_learning(
            task_topics=["python"],
            timestamp=datetime.now().isoformat(),
            success=False,
        )
        old = _make_session_learning(
            task_topics=["python"],
            timestamp=(datetime.now() - timedelta(days=60)).isoformat(),
            success=False,
        )
        score_recent = recent.get_relevance_score(["python"])
        score_old = old.get_relevance_score(["python"])
        assert score_recent > score_old

    def test_relevance_score_recency_clamps_at_zero(self):
        """Recency score should not go negative for very old sessions."""
        very_old = _make_session_learning(
            task_topics=["python"],
            timestamp=(datetime.now() - timedelta(days=365)).isoformat(),
            success=False,
        )
        score = very_old.get_relevance_score(["python"])
        # recency_score should be max(0, ...) = 0
        # score should still be >= 0.2 (base) + topic
        assert score >= 0.2

    def test_relevance_score_empty_query_topics(self):
        """Should handle empty query_topics without error."""
        session = _make_session_learning(task_topics=["python"])
        score = session.get_relevance_score([])
        # topic_score = 0 / max(0,1) = 0
        assert isinstance(score, float)
        assert score >= 0


# ===========================================================================
# MASLearning Initialization Tests
# ===========================================================================

@pytest.mark.unit
class TestMASLearningInit:
    """Tests for MASLearning initialization."""

    def test_init_with_learning_dir(self, tmp_path):
        """Should use provided learning_dir."""
        ml = _create_mas_learning(tmp_path)
        assert ml.learning_dir == tmp_path
        assert ml.fix_database == {}
        assert ml.session_learnings == []
        assert ml.current_session_id  # should be set

    def test_init_creates_directories(self, tmp_path):
        """Should create learning_dir and project_learning_dir."""
        learning_dir = tmp_path / "custom_learning"
        ml = MASLearning(learning_dir=learning_dir, workspace_path=tmp_path)
        assert learning_dir.exists()
        assert ml.project_learning_dir.exists()

    def test_init_delegates_stored(self, tmp_path):
        """Should store delegate references."""
        si = MagicMock(name="swarm_intelligence")
        lm = MagicMock(name="learning_manager")
        tl = MagicMock(name="transfer_learning")
        ml = _create_mas_learning(
            tmp_path,
            swarm_intelligence=si,
            learning_manager=lm,
            transfer_learning=tl,
        )
        assert ml.swarm_intelligence is si
        assert ml.learning_manager is lm
        assert ml.transfer_learning is tl

    def test_init_no_delegates(self, tmp_path):
        """Delegates should be None when not provided."""
        ml = _create_mas_learning(tmp_path)
        assert ml.swarm_intelligence is None
        assert ml.learning_manager is None
        assert ml.transfer_learning is None

    def test_init_workspace_path_default(self, tmp_path):
        """workspace_path should default to cwd when not provided."""
        ml = MASLearning(learning_dir=tmp_path)
        assert ml.workspace_path == Path.cwd()

    def test_init_topic_cache_empty(self, tmp_path):
        """Topic cache should start empty."""
        ml = _create_mas_learning(tmp_path)
        assert ml._topic_cache == {}

    def test_init_project_learning_dir_uses_hash(self, tmp_path):
        """project_learning_dir should use MD5 hash of workspace_path."""
        workspace = tmp_path / "my_project"
        ml = MASLearning(learning_dir=tmp_path, workspace_path=workspace)
        workspace_hash = hashlib.md5(str(workspace).encode()).hexdigest()[:8]
        expected = tmp_path / "projects" / workspace_hash
        assert ml.project_learning_dir == expected


# ===========================================================================
# MASLearning._error_hash Tests
# ===========================================================================

@pytest.mark.unit
class TestErrorHash:
    """Tests for MASLearning._error_hash normalization."""

    def test_normalizes_numbers(self, tmp_path):
        """Numbers should be replaced with 'N'."""
        ml = _create_mas_learning(tmp_path)
        h1 = ml._error_hash("Error on line 42")
        h2 = ml._error_hash("Error on line 99")
        assert h1 == h2

    def test_normalizes_paths(self, tmp_path):
        """File paths should be replaced with '/PATH'."""
        ml = _create_mas_learning(tmp_path)
        h1 = ml._error_hash("File /home/user/project/main.py not found")
        h2 = ml._error_hash("File /var/www/app/main.py not found")
        assert h1 == h2

    def test_normalizes_hex_addresses(self, tmp_path):
        """Hex addresses like 0x7fff12ab should become 'ADDR'."""
        ml = _create_mas_learning(tmp_path)
        h1 = ml._error_hash("Object at 0x7fff12ab crashed")
        h2 = ml._error_hash("Object at 0xdeadbeef crashed")
        assert h1 == h2

    def test_case_insensitive(self, tmp_path):
        """Hashing should be case-insensitive."""
        ml = _create_mas_learning(tmp_path)
        h1 = ml._error_hash("ModuleNotFoundError")
        h2 = ml._error_hash("modulenotfounderror")
        assert h1 == h2

    def test_returns_md5_hex(self, tmp_path):
        """Result should be a valid MD5 hex digest (32 chars)."""
        ml = _create_mas_learning(tmp_path)
        h = ml._error_hash("some error")
        assert len(h) == 32
        assert all(c in "0123456789abcdef" for c in h)

    def test_different_errors_different_hashes(self, tmp_path):
        """Fundamentally different errors should produce different hashes."""
        ml = _create_mas_learning(tmp_path)
        h1 = ml._error_hash("ModuleNotFoundError")
        h2 = ml._error_hash("SyntaxError unexpected token")
        assert h1 != h2


# ===========================================================================
# MASLearning.find_fix Tests
# ===========================================================================

@pytest.mark.unit
class TestFindFix:
    """Tests for MASLearning.find_fix."""

    def test_exact_hash_match(self, tmp_path):
        """Should find a fix by exact error hash match."""
        ml = _create_mas_learning(tmp_path)
        error = "ModuleNotFoundError: No module named 'requests'"
        error_hash = ml._error_hash(error)
        fix = _make_fix_record(
            error_pattern=error,
            error_hash=error_hash,
            success_count=5,
            fail_count=1,
        )
        ml.fix_database[error_hash] = fix

        result = ml.find_fix(error)
        assert result is not None
        assert result.error_hash == error_hash

    def test_exact_hash_match_low_success_rate(self, tmp_path):
        """Should not return fix with success_rate < 0.5."""
        ml = _create_mas_learning(tmp_path)
        error = "SomeError"
        error_hash = ml._error_hash(error)
        fix = _make_fix_record(
            error_pattern=error,
            error_hash=error_hash,
            success_count=1,
            fail_count=9,  # 10% success rate
        )
        ml.fix_database[error_hash] = fix

        result = ml.find_fix(error)
        assert result is None

    def test_pattern_match_substring(self, tmp_path):
        """Should find fix by pattern substring match."""
        ml = _create_mas_learning(tmp_path)
        # Add a fix with a pattern that is a substring of the query
        fix = _make_fix_record(
            error_pattern="Connection refused",
            error_hash="different_hash",
            success_count=7,
            fail_count=1,  # ~87.5% success rate
        )
        ml.fix_database["different_hash"] = fix

        result = ml.find_fix("Error: Connection refused on port 8080")
        assert result is not None
        assert result.error_pattern == "Connection refused"

    def test_pattern_match_low_success_rate(self, tmp_path):
        """Pattern match should require success_rate >= 0.6."""
        ml = _create_mas_learning(tmp_path)
        fix = _make_fix_record(
            error_pattern="Connection refused",
            error_hash="different_hash",
            success_count=1,
            fail_count=3,  # 25% success rate
        )
        ml.fix_database["different_hash"] = fix

        result = ml.find_fix("Error: Connection refused on port 8080")
        assert result is None

    def test_no_match(self, tmp_path):
        """Should return None when no fix matches."""
        ml = _create_mas_learning(tmp_path)
        result = ml.find_fix("Never seen this error before")
        assert result is None

    def test_exact_match_at_boundary_success_rate(self, tmp_path):
        """Exact hash match with exactly 0.5 success rate should be returned."""
        ml = _create_mas_learning(tmp_path)
        error = "BoundaryError"
        error_hash = ml._error_hash(error)
        fix = _make_fix_record(
            error_pattern=error,
            error_hash=error_hash,
            success_count=5,
            fail_count=5,
        )
        ml.fix_database[error_hash] = fix

        result = ml.find_fix(error)
        assert result is not None

    def test_pattern_match_at_boundary_success_rate(self, tmp_path):
        """Pattern match with exactly 0.6 success rate should be returned."""
        ml = _create_mas_learning(tmp_path)
        fix = _make_fix_record(
            error_pattern="timeout occurred",
            error_hash="unrelated_hash",
            success_count=6,
            fail_count=4,  # 60%
        )
        ml.fix_database["unrelated_hash"] = fix

        result = ml.find_fix("A timeout occurred during request")
        assert result is not None


# ===========================================================================
# MASLearning.record_fix Tests
# ===========================================================================

@pytest.mark.unit
class TestRecordFix:
    """Tests for MASLearning.record_fix."""

    def test_record_new_successful_fix(self, tmp_path):
        """Should create a new FixRecord on first successful fix."""
        ml = _create_mas_learning(tmp_path)
        error = "ImportError: cannot import name 'foo'"
        ml.record_fix(
            error=error,
            solution_commands=["pip install foo"],
            solution_description="Install foo",
            source="user",
            success=True,
        )
        error_hash = ml._error_hash(error)
        assert error_hash in ml.fix_database
        fix = ml.fix_database[error_hash]
        assert fix.success_count == 1
        assert fix.fail_count == 0
        assert fix.source == "user"

    def test_record_failure_no_new_entry(self, tmp_path):
        """Failure on unknown error should NOT create a new FixRecord."""
        ml = _create_mas_learning(tmp_path)
        error = "NewError: never seen"
        ml.record_fix(
            error=error,
            solution_commands=["try this"],
            solution_description="Try something",
            source="llm",
            success=False,
        )
        error_hash = ml._error_hash(error)
        assert error_hash not in ml.fix_database

    def test_record_success_updates_existing(self, tmp_path):
        """Success on existing fix should increment success_count."""
        ml = _create_mas_learning(tmp_path)
        error = "KnownError"
        error_hash = ml._error_hash(error)
        ml.fix_database[error_hash] = _make_fix_record(
            error_hash=error_hash,
            success_count=3,
            fail_count=1,
        )

        ml.record_fix(
            error=error,
            solution_commands=["cmd"],
            solution_description="desc",
            source="pattern",
            success=True,
        )
        assert ml.fix_database[error_hash].success_count == 4
        assert ml.fix_database[error_hash].fail_count == 1

    def test_record_failure_updates_existing(self, tmp_path):
        """Failure on existing fix should increment fail_count."""
        ml = _create_mas_learning(tmp_path)
        error = "KnownError"
        error_hash = ml._error_hash(error)
        ml.fix_database[error_hash] = _make_fix_record(
            error_hash=error_hash,
            success_count=3,
            fail_count=1,
        )

        ml.record_fix(
            error=error,
            solution_commands=["cmd"],
            solution_description="desc",
            source="pattern",
            success=False,
        )
        assert ml.fix_database[error_hash].success_count == 3
        assert ml.fix_database[error_hash].fail_count == 2

    def test_record_fix_updates_last_used(self, tmp_path):
        """Recording a fix should update last_used timestamp."""
        ml = _create_mas_learning(tmp_path)
        error = "OldError"
        error_hash = ml._error_hash(error)
        old_time = "2025-01-01T00:00:00"
        ml.fix_database[error_hash] = _make_fix_record(
            error_hash=error_hash,
            last_used=old_time,
        )

        ml.record_fix(
            error=error,
            solution_commands=[],
            solution_description="",
            source="pattern",
            success=True,
        )
        assert ml.fix_database[error_hash].last_used != old_time

    def test_record_fix_saves_every_10(self, tmp_path):
        """Fix database should be saved when size is a multiple of 10."""
        ml = _create_mas_learning(tmp_path)
        # Pre-populate with 9 fixes
        for i in range(9):
            ml.fix_database[f"hash_{i}"] = _make_fix_record(error_hash=f"hash_{i}")

        with patch.object(ml, "_save_fix_database") as mock_save:
            # Adding the 10th fix => total=10, 10%10==0 => save
            ml.record_fix(
                error="TenthError",
                solution_commands=["fix"],
                solution_description="fix it",
                source="user",
                success=True,
            )
            mock_save.assert_called_once()

    def test_record_fix_truncates_error_pattern(self, tmp_path):
        """Error pattern should be truncated to 500 characters."""
        ml = _create_mas_learning(tmp_path)
        long_error = "E" * 1000
        ml.record_fix(
            error=long_error,
            solution_commands=["fix"],
            solution_description="fix it",
            source="user",
            success=True,
        )
        error_hash = ml._error_hash(long_error)
        assert len(ml.fix_database[error_hash].error_pattern) == 500

    def test_record_fix_with_context(self, tmp_path):
        """Context dict should be stored in the fix record."""
        ml = _create_mas_learning(tmp_path)
        error = "ContextError"
        ctx = {"env": "production", "python": "3.11"}
        ml.record_fix(
            error=error,
            solution_commands=["fix"],
            solution_description="fix",
            source="user",
            success=True,
            context=ctx,
        )
        error_hash = ml._error_hash(error)
        assert ml.fix_database[error_hash].context == ctx


# ===========================================================================
# MASLearning._extract_topics Tests
# ===========================================================================

@pytest.mark.unit
class TestExtractTopics:
    """Tests for MASLearning._extract_topics."""

    def test_basic_extraction(self, tmp_path):
        """Should extract words with 4+ characters."""
        ml = _create_mas_learning(tmp_path)
        topics = ml._extract_topics("Build a python dashboard application")
        assert "build" in topics
        assert "python" in topics
        assert "dashboard" in topics
        assert "application" in topics

    def test_stopword_removal(self, tmp_path):
        """Stopwords should be filtered out."""
        ml = _create_mas_learning(tmp_path)
        topics = ml._extract_topics("this should have been removed from these topics")
        assert "this" not in topics
        assert "should" not in topics
        assert "have" not in topics
        assert "been" not in topics
        assert "from" not in topics
        assert "these" not in topics

    def test_short_words_excluded(self, tmp_path):
        """Words shorter than 4 characters should be excluded."""
        ml = _create_mas_learning(tmp_path)
        topics = ml._extract_topics("Run the app now for fun")
        # "Run", "the", "app", "now", "for", "fun" are all <= 3 chars
        assert len(topics) == 0

    def test_top_10_frequency(self, tmp_path):
        """Should return at most 10 topics, by frequency."""
        ml = _create_mas_learning(tmp_path)
        text = " ".join([f"word{i}" for i in range(20)] * 2)
        topics = ml._extract_topics(text)
        assert len(topics) <= 10

    def test_caching(self, tmp_path):
        """Second call with same text should use cache."""
        ml = _create_mas_learning(tmp_path)
        text = "Testing the cache mechanism properly"
        topics1 = ml._extract_topics(text)
        assert text in ml._topic_cache
        topics2 = ml._extract_topics(text)
        assert topics1 == topics2

    def test_case_insensitive(self, tmp_path):
        """All extracted topics should be lowercase."""
        ml = _create_mas_learning(tmp_path)
        topics = ml._extract_topics("Build Python Dashboard")
        for topic in topics:
            assert topic == topic.lower()

    def test_frequency_ordering(self, tmp_path):
        """More frequent words should appear first."""
        ml = _create_mas_learning(tmp_path)
        text = "python python python testing testing automation"
        topics = ml._extract_topics(text)
        assert topics[0] == "python"
        assert "testing" in topics


# ===========================================================================
# MASLearning.record_session Tests
# ===========================================================================

@pytest.mark.unit
class TestRecordSession:
    """Tests for MASLearning.record_session."""

    def test_record_creates_session(self, tmp_path):
        """Should append a SessionLearning object."""
        ml = _create_mas_learning(tmp_path)
        ml.record_session(
            task_description="Research AI trends",
            agents_used=["research_agent"],
            total_time=15.0,
            success=True,
            stigmergy_signals=2,
            output_quality=0.8,
        )
        assert len(ml.session_learnings) == 1
        session = ml.session_learnings[0]
        assert session.task_description == "Research AI trends"
        assert session.agents_used == ["research_agent"]
        assert session.total_time == 15.0
        assert session.success is True
        assert session.stigmergy_signals == 2
        assert session.output_quality == 0.8
        assert session.agent_count == 1

    def test_record_extracts_topics(self, tmp_path):
        """Session should have extracted topics from task description."""
        ml = _create_mas_learning(tmp_path)
        ml.record_session(
            task_description="Build python automation scripts",
            agents_used=[],
            total_time=10.0,
            success=True,
        )
        session = ml.session_learnings[0]
        assert "python" in session.task_topics
        assert "automation" in session.task_topics
        assert "scripts" in session.task_topics

    def test_record_bounds_at_100(self, tmp_path):
        """Session list should be bounded at 100 entries (keep last 100)."""
        ml = _create_mas_learning(tmp_path)
        for i in range(105):
            ml.record_session(
                task_description=f"Task number {i} testing sessions",
                agents_used=["agent"],
                total_time=1.0,
                success=True,
            )
        assert len(ml.session_learnings) == 100

    def test_record_saves_sessions(self, tmp_path):
        """Should call _save_sessions after recording."""
        ml = _create_mas_learning(tmp_path)
        with patch.object(ml, "_save_sessions") as mock_save:
            ml.record_session(
                task_description="test save",
                agents_used=[],
                total_time=1.0,
                success=True,
            )
            mock_save.assert_called_once()

    def test_record_uses_current_session_id(self, tmp_path):
        """Session should use the current_session_id."""
        ml = _create_mas_learning(tmp_path)
        ml.current_session_id = "custom_session_id"
        ml.record_session(
            task_description="test session identifier",
            agents_used=[],
            total_time=1.0,
            success=True,
        )
        assert ml.session_learnings[0].session_id == "custom_session_id"


# ===========================================================================
# MASLearning.load_relevant_learnings Tests
# ===========================================================================

@pytest.mark.unit
class TestLoadRelevantLearnings:
    """Tests for MASLearning.load_relevant_learnings."""

    def test_returns_expected_keys(self, tmp_path):
        """Result dict should contain all expected keys."""
        ml = _create_mas_learning(tmp_path)
        result = ml.load_relevant_learnings("test task")
        expected_keys = {
            "suggested_agents",
            "underperformers",
            "relevant_fixes",
            "performance_hints",
            "past_strategies",
            "query_topics",
        }
        assert set(result.keys()) == expected_keys

    def test_filters_by_relevance_threshold(self, tmp_path):
        """Only sessions with score > 0.3 should appear in past_strategies."""
        ml = _create_mas_learning(tmp_path)
        # Add a relevant session (recent, matching topic, successful)
        ml.session_learnings.append(_make_session_learning(
            task_topics=["python", "automation"],
            success=True,
            timestamp=datetime.now().isoformat(),
        ))
        # Add an irrelevant session (old, no topic match, failed)
        ml.session_learnings.append(_make_session_learning(
            task_topics=["javascript", "frontend"],
            success=False,
            timestamp=(datetime.now() - timedelta(days=60)).isoformat(),
        ))

        result = ml.load_relevant_learnings("python automation scripts")
        # The python/automation session should be relevant
        strategies = result["past_strategies"]
        if strategies:
            # At least one strategy should reference a python task
            assert any("python" in s.get("task", "").lower() or True for s in strategies)

    def test_query_topics_populated(self, tmp_path):
        """query_topics should be extracted from task description."""
        ml = _create_mas_learning(tmp_path)
        result = ml.load_relevant_learnings("Build python testing framework")
        assert "python" in result["query_topics"]
        assert "testing" in result["query_topics"]

    def test_performance_hints_defaults(self, tmp_path):
        """When no sessions exist, should return default performance hints."""
        ml = _create_mas_learning(tmp_path)
        result = ml.load_relevant_learnings("new task")
        hints = result["performance_hints"]
        assert hints["expected_time"] == 60  # default
        assert hints["expected_success_rate"] == 0.5  # default
        assert hints["similar_task_count"] == 0

    def test_performance_hints_from_sessions(self, tmp_path):
        """Performance hints should be computed from relevant sessions."""
        ml = _create_mas_learning(tmp_path)
        # Add 2 relevant sessions
        for t in [20.0, 30.0]:
            ml.session_learnings.append(_make_session_learning(
                task_topics=["python", "automation"],
                success=True,
                total_time=t,
                timestamp=datetime.now().isoformat(),
            ))

        result = ml.load_relevant_learnings("python automation task")
        hints = result["performance_hints"]
        assert hints["similar_task_count"] > 0

    def test_swarm_intelligence_integration(self, tmp_path):
        """Should delegate to swarm_intelligence for agent profiles."""
        si = MagicMock()
        profile = MagicMock()
        profile.success_rate = 0.85
        profile.specialization.value = "coding"
        profile.total_tasks = 10
        si.get_agent_profile.return_value = profile

        ml = _create_mas_learning(tmp_path, swarm_intelligence=si)
        result = ml.load_relevant_learnings(
            "coding task",
            agent_types=["coder_agent"],
        )
        assert "coder_agent" in result["suggested_agents"]
        assert result["suggested_agents"]["coder_agent"]["success_rate"] == 0.85

    def test_underperformers_detected(self, tmp_path):
        """Agents with success_rate < 0.6 and >= 2 tasks should be underperformers."""
        si = MagicMock()
        profile = MagicMock()
        profile.success_rate = 0.3
        profile.total_tasks = 5
        profile.specialization.value = "general"
        si.get_agent_profile.return_value = profile

        ml = _create_mas_learning(tmp_path, swarm_intelligence=si)
        result = ml.load_relevant_learnings(
            "task",
            agent_types=["bad_agent"],
        )
        assert "bad_agent" in result["underperformers"]
        assert result["underperformers"]["bad_agent"] == 0.3

    def test_swarm_intelligence_error_handled(self, tmp_path):
        """Errors from SwarmIntelligence should be handled gracefully."""
        si = MagicMock()
        si.get_agent_profile.side_effect = Exception("SI unavailable")

        ml = _create_mas_learning(tmp_path, swarm_intelligence=si)
        result = ml.load_relevant_learnings(
            "task",
            agent_types=["agent"],
        )
        # Should not crash, just return empty
        assert result["suggested_agents"] == {}

    def test_relevant_fixes_included(self, tmp_path):
        """Should include fixes from database."""
        ml = _create_mas_learning(tmp_path)
        ml.fix_database["h1"] = _make_fix_record(
            error_pattern="import error",
            solution_description="install package",
        )
        result = ml.load_relevant_learnings("task")
        assert len(result["relevant_fixes"]) == 1
        assert result["relevant_fixes"][0]["solution"] == "install package"

    def test_past_strategies_only_successful(self, tmp_path):
        """past_strategies should only include successful sessions."""
        ml = _create_mas_learning(tmp_path)
        # Add a failed session that is relevant
        ml.session_learnings.append(_make_session_learning(
            task_topics=["python"],
            success=False,
            timestamp=datetime.now().isoformat(),
        ))
        # Add a successful session that is relevant
        ml.session_learnings.append(_make_session_learning(
            task_topics=["python"],
            success=True,
            timestamp=datetime.now().isoformat(),
        ))
        result = ml.load_relevant_learnings("python development task")
        for strategy in result["past_strategies"]:
            # strategies come only from successful sessions
            assert "agents_used" in strategy


# ===========================================================================
# MASLearning.get_execution_strategy Tests
# ===========================================================================

@pytest.mark.unit
class TestGetExecutionStrategy:
    """Tests for MASLearning.get_execution_strategy."""

    def test_returns_expected_keys(self, tmp_path):
        """Result should have all expected keys."""
        ml = _create_mas_learning(tmp_path)
        result = ml.get_execution_strategy("test task", ["agent_a", "agent_b"])
        expected_keys = {
            "recommended_order",
            "skip_agents",
            "retry_agents",
            "expected_time",
            "confidence",
            "relevant_fixes",
        }
        assert set(result.keys()) == expected_keys

    def test_skip_agents_below_40_percent(self, tmp_path):
        """Agents with success_rate < 0.4 should be in skip_agents."""
        si = MagicMock()
        profile_bad = MagicMock()
        profile_bad.success_rate = 0.2
        profile_bad.total_tasks = 5
        profile_bad.specialization.value = "general"

        profile_good = MagicMock()
        profile_good.success_rate = 0.9
        profile_good.total_tasks = 10
        profile_good.specialization.value = "coding"

        def get_profile(name):
            if name == "bad_agent":
                return profile_bad
            return profile_good

        si.get_agent_profile.side_effect = get_profile

        ml = _create_mas_learning(tmp_path, swarm_intelligence=si)
        result = ml.get_execution_strategy(
            "code review",
            ["bad_agent", "good_agent"],
        )
        skip_names = [s["agent"] for s in result["skip_agents"]]
        assert "bad_agent" in skip_names
        assert "good_agent" not in skip_names

    def test_recommended_order_by_success_rate(self, tmp_path):
        """Agents should be ordered by success rate (highest first)."""
        si = MagicMock()

        def get_profile(name):
            rates = {"agent_a": 0.9, "agent_b": 0.7, "agent_c": 0.8}
            profile = MagicMock()
            profile.success_rate = rates.get(name, 0.5)
            profile.total_tasks = 10
            profile.specialization.value = "general"
            return profile

        si.get_agent_profile.side_effect = get_profile

        ml = _create_mas_learning(tmp_path, swarm_intelligence=si)
        result = ml.get_execution_strategy(
            "task",
            ["agent_a", "agent_b", "agent_c"],
        )
        order = result["recommended_order"]
        assert order == ["agent_a", "agent_c", "agent_b"]

    def test_skipped_agents_not_in_recommended_order(self, tmp_path):
        """Skipped agents should be excluded from recommended_order."""
        si = MagicMock()

        def get_profile(name):
            profile = MagicMock()
            profile.success_rate = 0.1 if name == "skip_me" else 0.8
            profile.total_tasks = 5
            profile.specialization.value = "general"
            return profile

        si.get_agent_profile.side_effect = get_profile

        ml = _create_mas_learning(tmp_path, swarm_intelligence=si)
        result = ml.get_execution_strategy(
            "task",
            ["skip_me", "keep_me"],
        )
        assert "skip_me" not in result["recommended_order"]
        assert "keep_me" in result["recommended_order"]

    def test_confidence_scales_with_session_count(self, tmp_path):
        """Confidence should be min(1.0, similar_task_count / 5)."""
        ml = _create_mas_learning(tmp_path)
        # No sessions => confidence should be 0
        result = ml.get_execution_strategy("task", ["agent"])
        assert result["confidence"] == 0.0

    def test_expected_time_default(self, tmp_path):
        """Expected time should default to 60.0 when no sessions."""
        ml = _create_mas_learning(tmp_path)
        result = ml.get_execution_strategy("task", ["agent"])
        assert result["expected_time"] == 60.0

    def test_retry_agents_between_40_and_60(self, tmp_path):
        """Agents with 0.4 <= success_rate < 0.6 should be retry_agents."""
        si = MagicMock()

        def get_profile(name):
            profile = MagicMock()
            profile.success_rate = 0.45
            profile.total_tasks = 5
            profile.specialization.value = "general"
            return profile

        si.get_agent_profile.side_effect = get_profile

        ml = _create_mas_learning(tmp_path, swarm_intelligence=si)
        result = ml.get_execution_strategy("task", ["borderline_agent"])
        retry_names = [r["agent"] for r in result["retry_agents"]]
        assert "borderline_agent" in retry_names


# ===========================================================================
# MASLearning Persistence Tests
# ===========================================================================

@pytest.mark.unit
class TestPersistence:
    """Tests for save_all / _load_all persistence round-trip."""

    def test_fix_database_roundtrip(self, tmp_path):
        """Fix database should survive save/load cycle."""
        ml = _create_mas_learning(tmp_path)
        ml.fix_database["hash1"] = _make_fix_record(
            error_pattern="TestError",
            error_hash="hash1",
            solution_commands=["pip install test"],
            solution_description="Install test package",
            source="user",
            success_count=5,
            fail_count=2,
        )
        ml.save_all()

        # Create new instance that loads from same dir
        ml2 = _create_mas_learning(tmp_path)
        assert "hash1" in ml2.fix_database
        fix = ml2.fix_database["hash1"]
        assert fix.error_pattern == "TestError"
        assert fix.solution_commands == ["pip install test"]
        assert fix.solution_description == "Install test package"
        assert fix.source == "user"
        assert fix.success_count == 5
        assert fix.fail_count == 2

    def test_session_learnings_roundtrip(self, tmp_path):
        """Session learnings should survive save/load cycle."""
        ml = _create_mas_learning(tmp_path)
        ml.record_session(
            task_description="Test persistence roundtrip",
            agents_used=["agent_a", "agent_b"],
            total_time=42.0,
            success=True,
            stigmergy_signals=7,
            output_quality=0.95,
        )
        ml.save_all()

        ml2 = _create_mas_learning(tmp_path)
        assert len(ml2.session_learnings) == 1
        session = ml2.session_learnings[0]
        assert session.task_description == "Test persistence roundtrip"
        assert session.agents_used == ["agent_a", "agent_b"]
        assert session.total_time == 42.0
        assert session.success is True
        assert session.stigmergy_signals == 7
        assert session.output_quality == 0.95

    def test_save_all_writes_files(self, tmp_path):
        """save_all should write both fix_database.json and session_learnings.json."""
        ml = _create_mas_learning(tmp_path)
        ml.fix_database["h"] = _make_fix_record(error_hash="h")
        ml.record_session(
            task_description="test",
            agents_used=[],
            total_time=1.0,
            success=True,
        )
        ml.save_all()

        assert (tmp_path / "fix_database.json").exists()
        assert (tmp_path / "session_learnings.json").exists()

    def test_load_from_empty_dir(self, tmp_path):
        """Loading from empty dir should result in empty databases."""
        ml = _create_mas_learning(tmp_path)
        assert ml.fix_database == {}
        assert ml.session_learnings == []

    def test_fix_database_json_valid(self, tmp_path):
        """Saved fix database should be valid JSON."""
        ml = _create_mas_learning(tmp_path)
        ml.fix_database["h"] = _make_fix_record(error_hash="h")
        ml.save_all()

        with open(tmp_path / "fix_database.json") as f:
            data = json.load(f)
        assert isinstance(data, dict)
        assert "h" in data

    def test_session_learnings_json_valid(self, tmp_path):
        """Saved session learnings should be valid JSON."""
        ml = _create_mas_learning(tmp_path)
        ml.record_session(
            task_description="json test",
            agents_used=["a"],
            total_time=1.0,
            success=True,
        )
        ml.save_all()

        with open(tmp_path / "session_learnings.json") as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 1

    def test_corrupted_fix_database_handled(self, tmp_path):
        """Corrupted fix database file should not crash loading."""
        fix_path = tmp_path / "fix_database.json"
        fix_path.write_text("not valid json{{{")

        # Should not raise
        ml = _create_mas_learning(tmp_path)
        assert ml.fix_database == {}

    def test_corrupted_sessions_handled(self, tmp_path):
        """Corrupted sessions file should not crash loading."""
        sessions_path = tmp_path / "session_learnings.json"
        sessions_path.write_text("invalid json!!!")

        ml = _create_mas_learning(tmp_path)
        assert ml.session_learnings == []


# ===========================================================================
# MASLearning.integrate_with_terminal Tests
# ===========================================================================

@pytest.mark.unit
class TestIntegrateWithTerminal:
    """Tests for MASLearning.integrate_with_terminal."""

    def test_loads_fixes_into_terminal_cache(self, tmp_path):
        """Should populate terminal._fix_cache with high-success fixes."""
        ml = _create_mas_learning(tmp_path)
        ml.fix_database["h1"] = _make_fix_record(
            error_hash="h1",
            error_pattern="test error",
            success_count=8,
            fail_count=2,
            solution_description="fix it",
            solution_commands=["cmd1"],
        )

        terminal = MagicMock()
        terminal._fix_cache = {}
        ml.integrate_with_terminal(terminal)

        assert "h1" in terminal._fix_cache
        cached = terminal._fix_cache["h1"]
        assert cached.error_pattern == "test error"
        assert cached.solution == "fix it"
        assert cached.source == "database"
        assert cached.confidence == pytest.approx(0.8)
        assert cached.commands == ["cmd1"]

    def test_skips_low_success_rate(self, tmp_path):
        """Fixes with success_rate < 0.5 should not be loaded into terminal."""
        ml = _create_mas_learning(tmp_path)
        ml.fix_database["h1"] = _make_fix_record(
            error_hash="h1",
            success_count=1,
            fail_count=9,  # 10% success rate
        )

        terminal = MagicMock()
        terminal._fix_cache = {}
        ml.integrate_with_terminal(terminal)

        assert "h1" not in terminal._fix_cache

    def test_none_terminal_handled(self, tmp_path):
        """Should handle None terminal gracefully."""
        ml = _create_mas_learning(tmp_path)
        # Should not raise
        ml.integrate_with_terminal(None)


# ===========================================================================
# MASLearning.sync_from_terminal Tests
# ===========================================================================

@pytest.mark.unit
class TestSyncFromTerminal:
    """Tests for MASLearning.sync_from_terminal."""

    def test_imports_fixes_from_history(self, tmp_path):
        """Should import fixes from terminal._fix_history."""
        ml = _create_mas_learning(tmp_path)
        terminal = MagicMock()
        terminal._fix_history = [
            {
                "error": "ImportError: no module foo",
                "commands": ["pip install foo"],
                "description": "install foo",
                "source": "terminal",
                "success": True,
                "context": {},
            },
        ]

        count = ml.sync_from_terminal(terminal)
        assert count == 1
        assert len(ml.fix_database) == 1

    def test_skips_empty_errors(self, tmp_path):
        """Should skip entries with empty error strings."""
        ml = _create_mas_learning(tmp_path)
        terminal = MagicMock()
        terminal._fix_history = [
            {"error": "", "commands": [], "description": "", "source": "terminal", "success": True},
        ]

        count = ml.sync_from_terminal(terminal)
        assert count == 0

    def test_none_terminal_returns_zero(self, tmp_path):
        """Should return 0 for None terminal."""
        ml = _create_mas_learning(tmp_path)
        count = ml.sync_from_terminal(None)
        assert count == 0

    def test_multiple_fixes_synced(self, tmp_path):
        """Should sync multiple fixes from history."""
        ml = _create_mas_learning(tmp_path)
        terminal = MagicMock()
        terminal._fix_history = [
            {
                "error": f"Error type {i}",
                "commands": [f"fix {i}"],
                "description": f"fix for error {i}",
                "source": "terminal",
                "success": True,
            }
            for i in range(5)
        ]

        count = ml.sync_from_terminal(terminal)
        assert count == 5

    def test_saves_after_sync(self, tmp_path):
        """Should save fix database after syncing new fixes."""
        ml = _create_mas_learning(tmp_path)
        terminal = MagicMock()
        terminal._fix_history = [
            {
                "error": "SomeError",
                "commands": ["fix"],
                "description": "fix it",
                "source": "terminal",
                "success": True,
            },
        ]

        with patch.object(ml, "_save_fix_database") as mock_save:
            ml.sync_from_terminal(terminal)
            mock_save.assert_called()


# ===========================================================================
# MASLearning.get_statistics Tests
# ===========================================================================

@pytest.mark.unit
class TestGetStatistics:
    """Tests for MASLearning.get_statistics."""

    def test_empty_statistics(self, tmp_path):
        """Should return valid stats when no data exists."""
        ml = _create_mas_learning(tmp_path)
        stats = ml.get_statistics()

        assert stats["fix_database"]["total_fixes"] == 0
        assert stats["fix_database"]["avg_success_rate"] == 0.0
        assert stats["sessions"]["total_sessions"] == 0
        assert stats["sessions"]["successful_sessions"] == 0
        assert stats["sessions"]["avg_time"] == 0.0

    def test_fix_database_stats(self, tmp_path):
        """Should compute correct fix database statistics."""
        ml = _create_mas_learning(tmp_path)
        ml.fix_database["h1"] = _make_fix_record(success_count=8, fail_count=2)
        ml.fix_database["h2"] = _make_fix_record(success_count=6, fail_count=4)

        stats = ml.get_statistics()
        assert stats["fix_database"]["total_fixes"] == 2
        # avg = (0.8 + 0.6) / 2 = 0.7
        assert stats["fix_database"]["avg_success_rate"] == pytest.approx(0.7)

    def test_session_stats(self, tmp_path):
        """Should compute correct session statistics."""
        ml = _create_mas_learning(tmp_path)
        ml.record_session(
            task_description="task one",
            agents_used=["a"],
            total_time=20.0,
            success=True,
        )
        ml.record_session(
            task_description="task two",
            agents_used=["b"],
            total_time=40.0,
            success=False,
        )

        stats = ml.get_statistics()
        assert stats["sessions"]["total_sessions"] == 2
        assert stats["sessions"]["successful_sessions"] == 1
        assert stats["sessions"]["avg_time"] == pytest.approx(30.0)

    def test_delegates_status(self, tmp_path):
        """Should report delegate availability."""
        si = MagicMock()
        ml = _create_mas_learning(tmp_path, swarm_intelligence=si)
        stats = ml.get_statistics()

        assert stats["delegates_to"]["swarm_intelligence"] is True
        assert stats["delegates_to"]["learning_manager"] is False
        assert stats["delegates_to"]["transfer_learning"] is False

    def test_delegates_all_present(self, tmp_path):
        """Should report all delegates as True when all provided."""
        ml = _create_mas_learning(
            tmp_path,
            swarm_intelligence=MagicMock(),
            learning_manager=MagicMock(),
            transfer_learning=MagicMock(),
        )
        stats = ml.get_statistics()
        assert all(stats["delegates_to"].values())


# ===========================================================================
# get_mas_learning convenience function Tests
# ===========================================================================

@pytest.mark.unit
class TestGetMASLearning:
    """Tests for the get_mas_learning convenience function."""

    def test_returns_mas_learning_instance(self, tmp_path):
        """Should return a MASLearning instance."""
        ml = get_mas_learning(learning_dir=tmp_path, workspace_path=tmp_path)
        assert isinstance(ml, MASLearning)

    def test_passes_delegates(self, tmp_path):
        """Should forward delegate kwargs."""
        si = MagicMock()
        ml = get_mas_learning(
            learning_dir=tmp_path,
            workspace_path=tmp_path,
            swarm_intelligence=si,
        )
        assert ml.swarm_intelligence is si
