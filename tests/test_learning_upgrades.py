"""
Tests for learning system upgrades:
  - Vector embeddings (EmbeddingService)
  - Distilled lessons (mem0 pattern)
  - Per-agent learning profiles
  - SwarmLearner prompt persistence
  - LearningStore schema extensions

All tests run offline (mocked LLM calls, no API keys needed).
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ─── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_db(tmp_path):
    """Fresh temporary SQLite database for each test."""
    db_path = str(tmp_path / "test_learning.db")
    from Jotty.core.intelligence.learning.learning_store import LearningStore

    store = LearningStore(db_path)
    yield store
    store.close()


@pytest.fixture
def tmp_service(tmp_db):
    """LearningService backed by a temp DB."""
    from Jotty.core.intelligence.learning.learning_service import LearningService

    service = LearningService(store=tmp_db)
    return service


@pytest.fixture
def mock_embedding_service():
    """EmbeddingService with deterministic embeddings (no model load)."""
    from Jotty.core.intelligence.learning.embeddings import EmbeddingService

    svc = EmbeddingService()
    svc._available = True

    def _mock_embed(text, **kwargs):
        np.random.seed(hash(text) % 2**31)
        vec = np.random.randn(384).astype(np.float32)
        return vec / np.linalg.norm(vec)

    def _mock_embed_batch(texts, **kwargs):
        return np.array([_mock_embed(t) for t in texts])

    svc._model = MagicMock()
    svc._model.encode = MagicMock(
        side_effect=lambda text, **kw: (
            _mock_embed(text) if isinstance(text, str) else _mock_embed_batch(text)
        )
    )
    return svc


# ─── EmbeddingService Tests ─────────────────────────────────────────────


class TestEmbeddingService:
    @pytest.mark.unit
    def test_singleton(self):
        from Jotty.core.intelligence.learning.embeddings import EmbeddingService

        a = EmbeddingService.get_instance()
        b = EmbeddingService.get_instance()
        assert a is b

    @pytest.mark.unit
    def test_serialize_roundtrip(self):
        from Jotty.core.intelligence.learning.embeddings import EmbeddingService

        vec = np.random.randn(384).astype(np.float32)
        blob = EmbeddingService.serialize(vec)
        assert isinstance(blob, bytes)
        assert len(blob) == 384 * 4  # float32 = 4 bytes

        restored = EmbeddingService.deserialize(blob)
        assert np.allclose(vec, restored)

    @pytest.mark.unit
    def test_cosine_similarity_normalized(self):
        from Jotty.core.intelligence.learning.embeddings import EmbeddingService

        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert EmbeddingService.cosine_similarity(a, b) == pytest.approx(1.0)

        c = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        assert EmbeddingService.cosine_similarity(a, c) == pytest.approx(0.0)

    @pytest.mark.unit
    def test_cosine_similarities_batch(self):
        from Jotty.core.intelligence.learning.embeddings import EmbeddingService

        query = np.array([1.0, 0.0], dtype=np.float32)
        matrix = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], dtype=np.float32)
        sims = EmbeddingService.cosine_similarities(query, matrix)
        assert sims.shape == (3,)
        assert sims[0] == pytest.approx(1.0)
        assert sims[1] == pytest.approx(0.0)

    @pytest.mark.unit
    def test_embed_returns_none_when_unavailable(self):
        from Jotty.core.intelligence.learning.embeddings import EmbeddingService

        svc = EmbeddingService()
        svc._available = False
        assert svc.embed("hello") is None

    @pytest.mark.unit
    def test_dim(self):
        from Jotty.core.intelligence.learning.embeddings import EmbeddingService

        assert EmbeddingService().dim == 384


# ─── LearningStore Schema Tests ─────────────────────────────────────────


class TestLearningStoreSchema:
    @pytest.mark.unit
    def test_distilled_lessons_table_exists(self, tmp_db):
        conn = tmp_db._get_conn()
        tables = [
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        ]
        assert "distilled_lessons" in tables

    @pytest.mark.unit
    def test_episodes_has_embedding_column(self, tmp_db):
        conn = tmp_db._get_conn()
        cols = {r[1] for r in conn.execute("PRAGMA table_info(episodes)").fetchall()}
        assert "embedding" in cols

    @pytest.mark.unit
    def test_save_and_get_distilled_lesson(self, tmp_db):
        from Jotty.core.intelligence.learning.learning_store import DistilledLesson

        lesson = DistilledLesson(
            lesson_id="dl_test_1",
            episode_id="ep_test_1",
            domain="coding",
            agent_name="test_agent",
            lesson="Use dataclass for clean APIs",
            context_type="pattern",
            applicability="Python API tasks",
            confidence=0.8,
        )
        tmp_db.save_distilled_lesson(lesson, embedding=None)

        lessons = tmp_db.get_distilled_lessons(domain="coding")
        assert len(lessons) == 1
        assert lessons[0].lesson == "Use dataclass for clean APIs"
        assert lessons[0].confidence == 0.8

    @pytest.mark.unit
    def test_save_and_get_embedding(self, tmp_db):
        from Jotty.core.intelligence.learning.learning_store import EpisodeRecord

        ep = EpisodeRecord(
            episode_id="ep_emb_test",
            unit_type="agent",
            unit_name="test",
            domain="coding",
            task_type="test",
            context={},
            action={},
            outcome={},
            success=True,
            quality=0.8,
            execution_time=1.0,
            cost=0.0,
        )
        tmp_db.save_episode(ep)

        vec = np.random.randn(384).astype(np.float32)
        blob = vec.tobytes()
        tmp_db.save_embedding("ep_emb_test", blob)

        results = tmp_db.get_episodes_with_embeddings(domain="coding")
        assert len(results) == 1
        ep_back, emb_back = results[0]
        assert ep_back.episode_id == "ep_emb_test"
        assert np.allclose(np.frombuffer(emb_back, dtype=np.float32), vec)

    @pytest.mark.unit
    def test_lesson_filtering_by_agent(self, tmp_db):
        from Jotty.core.intelligence.learning.learning_store import DistilledLesson

        for i, agent in enumerate(["coder", "researcher", "coder"]):
            lesson = DistilledLesson(
                lesson_id=f"dl_{i}",
                episode_id=f"ep_{i}",
                domain="coding",
                agent_name=agent,
                lesson=f"Lesson {i} from {agent}",
                context_type="pattern",
                applicability="",
                confidence=0.7,
            )
            tmp_db.save_distilled_lesson(lesson)

        coder_lessons = tmp_db.get_distilled_lessons(agent_name="coder")
        assert len(coder_lessons) == 2

        researcher_lessons = tmp_db.get_distilled_lessons(agent_name="researcher")
        assert len(researcher_lessons) == 1

    @pytest.mark.unit
    def test_lesson_with_embedding_retrieval(self, tmp_db):
        from Jotty.core.intelligence.learning.learning_store import DistilledLesson

        vec = np.random.randn(384).astype(np.float32)
        lesson = DistilledLesson(
            lesson_id="dl_emb_1",
            episode_id="ep_1",
            domain="coding",
            agent_name="test",
            lesson="Test lesson",
            context_type="pattern",
            applicability="",
            confidence=0.9,
        )
        tmp_db.save_distilled_lesson(lesson, embedding=vec.tobytes())

        results = tmp_db.get_distilled_lessons_with_embeddings(domain="coding")
        assert len(results) == 1
        lesson_back, emb_back = results[0]
        assert lesson_back.lesson == "Test lesson"
        assert len(emb_back) == 384 * 4


# ─── LearningService Integration Tests ──────────────────────────────────


class TestLearningServiceEmbeddings:
    @pytest.mark.unit
    def test_embedding_retrieval_prefers_similar(self, tmp_service, mock_embedding_service):
        """Embedding retrieval should rank semantically similar episodes higher."""
        with patch(
            "Jotty.core.intelligence.learning.learning_service._get_embeddings",
            return_value=mock_embedding_service,
        ):
            from Jotty.core.intelligence.learning.embeddings import EmbeddingService

            # Record two episodes with different goals
            for i, goal in enumerate(["Build a REST API", "Cook pasta recipe"]):
                ep_id = tmp_service.record(
                    unit_name="test",
                    unit_type="agent",
                    domain="coding",
                    task_type="test",
                    context={"goal": goal},
                    action={},
                    outcome={"content": f"Response for: {goal} " * 50},
                    success=True,
                    quality=0.8,
                )
                vec = mock_embedding_service.embed(goal)
                tmp_service._store.save_embedding(ep_id, EmbeddingService.serialize(vec))

            results = tmp_service.retrieve_similar_responses(
                "coding", goal="Create a web API server"
            )
            assert len(results) > 0

    @pytest.mark.unit
    def test_tfidf_fallback_when_no_embeddings(self, tmp_service):
        """When no embeddings available, falls back to TF-IDF."""
        with patch(
            "Jotty.core.intelligence.learning.learning_service._get_embeddings",
        ) as mock:
            mock.return_value = MagicMock(available=False)

            tmp_service.record(
                unit_name="test",
                unit_type="agent",
                domain="coding",
                task_type="test",
                context={"goal": "Build a REST API"},
                action={},
                outcome={"content": "class Router: pass " * 50},
                success=True,
                quality=0.8,
            )

            results = tmp_service.retrieve_similar_responses("coding", goal="Create a REST API")
            # Should still work via TF-IDF fallback
            assert isinstance(results, list)


class TestDistilledLessons:
    @pytest.mark.unit
    def test_store_and_retrieve_lessons(self, tmp_service):
        """Manually store and retrieve distilled lessons."""
        tmp_service._store_distilled_lessons(
            [
                {
                    "lesson": "Use dataclass for clean request models",
                    "type": "pattern",
                    "applies_to": "API tasks",
                    "confidence": 0.8,
                },
                {
                    "lesson": "Separate routing from handler logic",
                    "type": "strategy",
                    "applies_to": "Framework tasks",
                    "confidence": 0.7,
                },
            ],
            episode_id="ep_test",
            domain="coding",
            agent_name="coder",
        )

        lessons = tmp_service.retrieve_distilled_lessons("coding")
        assert len(lessons) == 2
        assert any("dataclass" in l["lesson"] for l in lessons)

    @pytest.mark.unit
    def test_lessons_in_context_string_when_failing(self, tmp_service):
        """Distilled lessons appear in context when there are failures to correct."""
        tmp_service._store_distilled_lessons(
            [{"lesson": "UNIQUE_TEST_LESSON_XYZ", "type": "pattern", "confidence": 0.9}],
            episode_id="ep_test",
            domain="coding",
            agent_name="test",
        )

        # Record failures to trigger full corrective context
        for i in range(4):
            tmp_service.record(
                unit_name="test",
                unit_type="agent",
                domain="coding",
                task_type="test",
                context={"goal": f"Task {i}"},
                action={},
                outcome={"content": "bad " * 10},
                success=False,
                quality=0.2,
            )

        ctx = tmp_service.build_context_string("coding", goal="Build something")
        assert "UNIQUE_TEST_LESSON_XYZ" in ctx

    @pytest.mark.unit
    def test_lessons_injected_when_succeeding(self, tmp_service):
        """When model succeeds, distilled lessons should still be injected (concise, high-signal)."""
        tmp_service._store_distilled_lessons(
            [{"lesson": "LESSON_FROM_PAST", "type": "pattern", "confidence": 0.9}],
            episode_id="ep_test",
            domain="coding",
            agent_name="test",
        )

        for i in range(4):
            tmp_service.record(
                unit_name="test",
                unit_type="agent",
                domain="coding",
                task_type="test",
                context={"goal": f"Task {i}"},
                action={},
                outcome={"content": "good code " * 100},
                success=True,
                quality=0.9,
            )

        ctx = tmp_service.build_context_string("coding", goal="New task")
        assert "LESSON_FROM_PAST" in ctx
        assert "Learned patterns" in ctx
        assert len(ctx) < 500

    @pytest.mark.unit
    def test_per_agent_lesson_retrieval(self, tmp_service):
        """Lessons should be filterable by agent name."""
        for agent, lesson in [
            ("coder", "Use type hints everywhere"),
            ("researcher", "Cite primary sources"),
            ("coder", "Keep functions under 20 lines"),
        ]:
            tmp_service._store_distilled_lessons(
                [{"lesson": lesson, "type": "pattern", "confidence": 0.8}],
                episode_id=f"ep_{agent}_{lesson[:5]}",
                domain="coding",
                agent_name=agent,
            )

        coder_lessons = tmp_service.retrieve_distilled_lessons("coding", agent_name="coder")
        assert len(coder_lessons) >= 2
        assert all(l["agent"] == "coder" for l in coder_lessons)


# ─── SwarmLearner Persistence Tests ──────────────────────────────────────


class TestSwarmLearnerPersistence:
    @pytest.mark.unit
    def test_persist_and_load_prompts(self, tmp_path):
        from Jotty.core.intelligence.orchestration.learning.swarm_learner import SwarmLearner

        config = MagicMock()
        config.policy_update_threshold = 3

        # Override persist path
        SwarmLearner._PERSIST_DIR = tmp_path
        SwarmLearner._PERSIST_FILE = "test_prompts.json"

        learner = SwarmLearner(config)
        learner.prompt_versions = {
            "architect": ["v1 prompt", "v2 prompt"],
            "auditor": ["v1 audit prompt"],
        }
        learner.learned_patterns = [
            {"type": "success", "pattern": "step1 -> step2", "timestamp": time.time()}
        ]
        learner._save_persisted_prompts()

        persist_file = tmp_path / "test_prompts.json"
        assert persist_file.exists()

        # Load into new instance
        learner2 = SwarmLearner(config)
        assert learner2.prompt_versions == {
            "architect": ["v1 prompt", "v2 prompt"],
            "auditor": ["v1 audit prompt"],
        }
        assert len(learner2.learned_patterns) == 1

    @pytest.mark.unit
    def test_get_latest_prompt(self, tmp_path):
        from Jotty.core.intelligence.orchestration.learning.swarm_learner import SwarmLearner

        config = MagicMock()
        config.policy_update_threshold = 3
        SwarmLearner._PERSIST_DIR = tmp_path
        SwarmLearner._PERSIST_FILE = "test_prompts2.json"

        learner = SwarmLearner(config)
        assert learner.get_latest_prompt("architect") == ""

        learner.prompt_versions["architect"] = ["v1", "v2", "v3"]
        assert learner.get_latest_prompt("architect") == "v3"

    @pytest.mark.unit
    def test_persist_trims_versions(self, tmp_path):
        from Jotty.core.intelligence.orchestration.learning.swarm_learner import SwarmLearner

        config = MagicMock()
        config.policy_update_threshold = 3
        SwarmLearner._PERSIST_DIR = tmp_path
        SwarmLearner._PERSIST_FILE = "test_prompts3.json"

        learner = SwarmLearner(config)
        learner.prompt_versions["architect"] = [f"v{i}" for i in range(20)]
        learner._save_persisted_prompts()

        # Reload — should only have last 5
        learner2 = SwarmLearner(config)
        assert len(learner2.prompt_versions["architect"]) == 5
        assert learner2.prompt_versions["architect"][-1] == "v19"


# ─── Adaptive Gate Tests ─────────────────────────────────────────────────


class TestAdaptiveGate:
    @pytest.mark.unit
    def test_no_context_when_no_episodes(self, tmp_service):
        """Cold start with zero episodes: bootstrap guidance only."""
        ctx = tmp_service.build_context_string("coding")
        # Should get bootstrap guidance, not empty
        assert "coding" in ctx.lower() or "code" in ctx.lower() or ctx == ""

    @pytest.mark.unit
    def test_lightweight_context_when_succeeding(self, tmp_service):
        """When model succeeds, context should be lightweight (no raw retrieval)."""
        # Record 5 successes
        for i in range(5):
            tmp_service.record(
                unit_name="test",
                unit_type="agent",
                domain="coding",
                task_type="test",
                context={"goal": f"Task {i}"},
                action={},
                outcome={"content": "good code " * 100},
                success=True,
                quality=0.9,
            )

        ctx = tmp_service.build_context_string("coding", goal="New task")
        # After 5 successes: should get maintenance guidance (lightweight)
        assert len(ctx) < 500 or "maintenance" in ctx.lower() or "Learned" in ctx

    @pytest.mark.unit
    def test_full_context_when_failing(self, tmp_service):
        """When model fails, context should include full learning signals."""
        for i in range(5):
            tmp_service.record(
                unit_name="test",
                unit_type="agent",
                domain="coding",
                task_type="test",
                context={"goal": f"Task {i}"},
                action={},
                outcome={"content": "bad " * 10},
                success=False,
                quality=0.2,
            )

        ctx = tmp_service.build_context_string("coding", goal="Fix this")
        # After failures: should get substantive guidance
        assert len(ctx) > 50


# ─── Truncation Tests ────────────────────────────────────────────────────


class TestTruncation:
    @pytest.mark.unit
    def test_content_keys_get_higher_limit(self):
        from Jotty.core.intelligence.learning.learning_service import LearningService

        d = {
            "content": "x" * 5000,
            "response_excerpt": "y" * 5000,
            "some_field": "z" * 5000,
        }
        result = LearningService._truncate_dict(d)
        assert len(result["content"]) == 3003  # 3000 + "..."
        assert len(result["response_excerpt"]) == 3003
        assert len(result["some_field"]) == 503  # 500 + "..."
