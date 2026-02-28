"""
Tests for the architecture reorganization:
1. LearningService - unified learning for all units
2. Team elimination - swarm IS coordinated agents
3. Pipeline integration into Orchestrator
4. Backward compatibility of all aliases

Author: Jotty Team
Date: February 2026
"""

import asyncio
import os
import tempfile
import time
import unittest.mock

import pytest

# =============================================================================
# 1. LearningStore Tests
# =============================================================================


class TestLearningStore:
    """Test the SQLite-backed persistent learning store."""

    @pytest.fixture
    def store(self):
        from Jotty.core.intelligence.learning.learning_store import LearningStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_learning.db")
            s = LearningStore(db_path)
            yield s
            s.close()

    def test_save_and_query_episode(self, store):
        from Jotty.core.intelligence.learning.learning_store import EpisodeRecord

        episode = EpisodeRecord(
            episode_id="ep_test_001",
            unit_type="agent",
            unit_name="TestAgent",
            domain="coding",
            task_type="code_gen",
            context={"task": "Build REST API"},
            action={"model": "claude-sonnet"},
            outcome={"code_lines": 150},
            success=True,
            quality=0.85,
            execution_time=2.5,
            cost=0.01,
        )
        store.save_episode(episode)

        results = store.query_episodes(domain="coding")
        assert len(results) == 1
        assert results[0].unit_name == "TestAgent"
        assert results[0].success is True
        assert results[0].quality == 0.85

    def test_success_rate(self, store):
        from Jotty.core.intelligence.learning.learning_store import EpisodeRecord

        for i in range(10):
            store.save_episode(
                EpisodeRecord(
                    episode_id=f"ep_{i}",
                    unit_type="swarm",
                    unit_name="CodingSwarm",
                    domain="coding",
                    task_type="code_gen",
                    context={},
                    action={},
                    outcome={},
                    success=i < 7,  # 70% success
                    quality=0.8 if i < 7 else 0.2,
                    execution_time=1.0,
                    cost=0.01,
                    timestamp=time.time() + i,
                )
            )

        rate, total = store.get_success_rate("coding")
        assert total == 10
        assert abs(rate - 0.7) < 0.01

    def test_save_and_query_pattern(self, store):
        from Jotty.core.intelligence.learning.learning_store import PatternRecord

        pattern = PatternRecord(
            pattern_id="pat_001",
            source_domain="coding",
            pattern_type="success_strategy",
            description="Pipeline pattern works best for coding tasks",
            conditions={"domain": "coding"},
            recommendation="Use PIPELINE coordination",
            confidence=0.85,
            evidence_count=15,
            applicable_domains=["coding", "devops"],
        )
        store.save_pattern(pattern)

        patterns = store.get_patterns(domain="coding")
        assert len(patterns) == 1
        assert patterns[0].recommendation == "Use PIPELINE coordination"

    def test_improvement_report(self, store):
        from Jotty.core.intelligence.learning.learning_store import EpisodeRecord

        for i in range(30):
            store.save_episode(
                EpisodeRecord(
                    episode_id=f"ep_{i}",
                    unit_type="swarm",
                    unit_name="TestSwarm",
                    domain="research",
                    task_type="analysis",
                    context={},
                    action={},
                    outcome={},
                    success=i >= 15,  # Last 15 succeed, first 15 fail
                    quality=0.8 if i >= 15 else 0.2,
                    execution_time=1.0,
                    cost=0.01,
                    timestamp=time.time() + i,
                )
            )

        report = store.get_improvement_report("research")
        assert report["total"] == 30
        assert report["recent_success_rate"] > report["historical_success_rate"]
        assert report["improving"] is True

    def test_reflections(self, store):
        from Jotty.core.intelligence.learning.learning_store import (
            EpisodeRecord,
            ReflectionRecord,
        )

        store.save_episode(
            EpisodeRecord(
                episode_id="ep_ref",
                unit_type="pipeline",
                unit_name="Pipeline",
                domain="coding",
                task_type="pipeline",
                context={},
                action={},
                outcome={},
                success=False,
                quality=0.0,
                execution_time=5.0,
                cost=0.02,
            )
        )

        store.save_reflection(
            ReflectionRecord(
                reflection_id="ref_001",
                episode_id="ep_ref",
                step=2,
                unit_name="TestWriter",
                observation="Edge case tests failing",
                analysis="Missing validation for null inputs",
                adjustment="Add null checks before testing",
                applied=True,
                improvement=0.3,
            )
        )

        refs = store.get_reflections(episode_id="ep_ref")
        assert len(refs) == 1
        assert refs[0].observation == "Edge case tests failing"
        assert refs[0].applied is True


# =============================================================================
# 2. LearningService Tests
# =============================================================================


class TestLearningService:
    """Test the unified learning service."""

    @pytest.fixture
    def service(self):
        from Jotty.core.intelligence.learning.learning_service import LearningService
        from Jotty.core.intelligence.learning.learning_store import LearningStore

        LearningService.reset_instance()
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_learning.db")
            store = LearningStore(db_path)
            svc = LearningService(store)
            yield svc
            store.close()
        LearningService.reset_instance()

    def test_record_and_query(self, service):
        service.record(
            unit_name="TestAgent",
            unit_type="agent",
            domain="coding",
            task_type="code_gen",
            context={"task": "Build API"},
            action={"model": "claude-sonnet"},
            outcome={"lines": 100},
            success=True,
            quality=0.9,
        )

        guidance = service.query("coding", "code_gen")
        assert guidance["has_learning"] is True
        assert guidance["success_rate"] > 0

    def test_reflect(self, service):
        ep_id = service.record(
            unit_name="CodingSwarm",
            unit_type="swarm",
            domain="coding",
            task_type="code_gen",
            context={},
            action={},
            outcome={},
            success=False,
            error_type="execution_failure",
            error_message="Tests failing on edge cases",
        )

        result = service.reflect(
            episode_id=ep_id,
            step=3,
            observation="Tests failing on edge cases",
            unit_name="TestWriter",
        )
        assert "adjustment" in result
        assert isinstance(result["adjustment"], str)

    def test_episode_lifecycle(self, service):
        ep_id = service.start_episode(
            unit_name="PipelineStage",
            unit_type="pipeline_stage",
            domain="coding",
            task_type="design",
            context={"goal": "Design API"},
        )

        service.record_step(ep_id, "analyze", {"tool": "analyzer"}, {"result": "ok"}, True)
        service.record_step(ep_id, "design", {"tool": "architect"}, {"design": "..."}, True)

        final_id = service.end_episode(
            episode_id=ep_id,
            success=True,
            quality=0.85,
            outcome={"design": "REST API with auth"},
        )
        assert final_id is not None

    def test_transfer(self, service):
        from Jotty.core.intelligence.learning.learning_store import PatternRecord

        service._store.save_pattern(
            PatternRecord(
                pattern_id="pat_transfer",
                source_domain="coding",
                pattern_type="success_strategy",
                description="Pipeline coordination works for multi-step tasks",
                conditions={},
                recommendation="Use PIPELINE for multi-step workflows",
                confidence=0.85,
                evidence_count=20,
                applicable_domains=["coding"],
            )
        )

        transferred = service.transfer("coding", "devops")
        assert len(transferred) > 0
        assert any("PIPELINE" in t["recommendation"] for t in transferred)

    def test_build_context_string(self, service):
        # Record a mix of successes and failures so adaptive gate triggers
        for i in range(5):
            service.record(
                unit_name="Agent",
                unit_type="agent",
                domain="research",
                task_type="analysis",
                context={},
                action={},
                outcome={},
                success=(i % 2 == 0),  # 60% failure rate triggers guidance
                quality=0.8 if i % 2 == 0 else 0.2,
            )

        # Disable the 10% online holdout so test is deterministic
        with unittest.mock.patch("random.random", return_value=0.99):
            ctx = service.build_context_string("research", "analysis")
        assert "research" in ctx
        assert len(ctx) > 0

    def test_improvement_report(self, service):
        report = service.improvement_report("coding")
        assert "total" in report


# =============================================================================
# 3. Team Elimination — Backward Compatibility Tests
# =============================================================================


class TestTeamElimination:
    """Verify backward compatibility of Team → Swarm rename."""

    def test_agent_coordinator_alias(self):
        """AgentCoordinator is alias for TeamCoordinator."""
        from Jotty.core.intelligence.orchestration.swarms.base import (
            AgentCoordinator,
            TeamCoordinator,
        )

        assert AgentCoordinator is TeamCoordinator

    def test_coordination_result_alias(self):
        """CoordinationResult is alias for TeamResult."""
        from Jotty.core.intelligence.orchestration.swarms.base import (
            CoordinationResult,
            TeamResult,
        )

        assert CoordinationResult is TeamResult

    def test_swarm_exports_new_names(self):
        """Swarm __init__ exports both new and old names."""
        from Jotty.core.intelligence.orchestration.swarms import (
            AgentCoordinator,
            CoordinationResult,
            SwarmTemplate,
            TeamCoordinator,
            TeamResult,
        )

        assert AgentCoordinator is TeamCoordinator
        assert CoordinationResult is TeamResult

    def test_agent_team_backward_compat(self):
        """AGENT_TEAM still works as class attribute."""
        from Jotty.core.intelligence.orchestration.swarms.base import (
            CoordinationPattern,
            SwarmTemplate,
            TeamCoordinator,
        )
        from Jotty.core.intelligence.orchestration.swarms._base.swarm_learning import (
            SwarmBaseConfig,
        )

        class MockAgent:
            def __init__(self, **kwargs):
                pass

        # Old style: AGENT_TEAM
        class OldSwarm(SwarmTemplate):
            AGENT_TEAM = TeamCoordinator.define(
                (MockAgent, "Agent1"),
            )

            async def _execute_domain(self, *a, **kw):
                return None

        config = SwarmBaseConfig(name="test", domain="test")
        swarm = OldSwarm(config)
        coordinator = swarm._resolve_agents()
        assert coordinator is not None
        assert len(coordinator) == 1

    def test_agents_new_style(self):
        """AGENTS works as class attribute (new preferred style)."""
        from Jotty.core.intelligence.orchestration.swarms.base import (
            AgentCoordinator,
            CoordinationPattern,
            SwarmTemplate,
        )
        from Jotty.core.intelligence.orchestration.swarms._base.swarm_learning import (
            SwarmBaseConfig,
        )

        class MockAgent:
            def __init__(self, **kwargs):
                pass

        class NewSwarm(SwarmTemplate):
            AGENTS = AgentCoordinator.define(
                (MockAgent, "Agent1"),
                (MockAgent, "Agent2"),
                pattern=CoordinationPattern.PARALLEL,
            )

            async def _execute_domain(self, *a, **kw):
                return None

        config = SwarmBaseConfig(name="test", domain="test")
        swarm = NewSwarm(config)
        coordinator = swarm._resolve_agents()
        assert coordinator is not None
        assert len(coordinator) == 2
        assert coordinator.pattern == CoordinationPattern.PARALLEL

    def test_execute_team_is_coordinate(self):
        """execute_team is an alias for coordinate."""
        from Jotty.core.intelligence.orchestration.swarms.base import SwarmTemplate

        assert SwarmTemplate.execute_team is SwarmTemplate.coordinate

    def test_has_team_coordination_is_has_coordination(self):
        """has_team_coordination is an alias for has_coordination."""
        from Jotty.core.intelligence.orchestration.swarms.base import SwarmTemplate

        assert SwarmTemplate.has_team_coordination is SwarmTemplate.has_coordination


# =============================================================================
# 4. Import Tests — Everything still importable
# =============================================================================


class TestImports:
    """Verify all existing imports still work."""

    def test_learning_service_import(self):
        from Jotty.core.intelligence.learning import LearningService, get_learning_service

        assert LearningService is not None

    def test_learning_store_import(self):
        from Jotty.core.intelligence.learning import (
            EpisodeRecord,
            LearningStore,
            PatternRecord,
            ReflectionRecord,
            ValueEstimate,
        )

        assert LearningStore is not None
        assert EpisodeRecord is not None

    def test_swarm_base_imports(self):
        from Jotty.core.intelligence.orchestration.swarms.base import (
            AgentCoordinator,
            AgentSpec,
            CoordinationPattern,
            CoordinationResult,
            MergeStrategy,
            PhaseExecutor,
            SwarmTemplate,
            TeamCoordinator,
            TeamResult,
        )

        assert all(
            x is not None
            for x in [
                AgentCoordinator,
                AgentSpec,
                CoordinationPattern,
                CoordinationResult,
                MergeStrategy,
                PhaseExecutor,
                SwarmTemplate,
                TeamCoordinator,
                TeamResult,
            ]
        )

    def test_facade_get_learning_service(self):
        from Jotty.core.intelligence.learning.facade import get_learning_service

        assert callable(get_learning_service)

    def test_old_team_coordinator_import(self):
        """Old import path still works."""
        from Jotty.core.intelligence.orchestration.swarms.base.team_coordinator import (
            TeamCoordinator,
            TeamResult,
        )

        assert TeamCoordinator is not None
        assert TeamResult is not None

    def test_pipeline_import_still_works(self):
        """Old pipeline import still works."""
        from Jotty.core.intelligence.orchestration.pipelines.multi_stage import (
            MultiStagePipeline,
            PipelineResult,
            StageConfig,
        )

        assert MultiStagePipeline is not None


# =============================================================================
# 5. Architecture Integrity Tests
# =============================================================================


class TestArchitectureIntegrity:
    """Verify the new architecture relationships."""

    def test_swarm_has_learning_service_slot(self):
        """SwarmTemplate has _learning_service attribute."""
        from Jotty.core.intelligence.orchestration.swarms.base import SwarmTemplate
        from Jotty.core.intelligence.orchestration.swarms._base.swarm_learning import (
            SwarmBaseConfig,
        )

        class MinimalSwarm(SwarmTemplate):
            async def _execute_domain(self, *a, **kw):
                return None

        config = SwarmBaseConfig(name="test", domain="test")
        swarm = MinimalSwarm(config)
        assert hasattr(swarm, "_learning_service")

    def test_base_agent_has_learning_recording(self):
        """BaseAgent has _record_to_learning_service method."""
        from Jotty.core.intelligence.reasoning.base.base_agent import BaseAgent

        assert hasattr(BaseAgent, "_record_to_learning_service")
        assert callable(BaseAgent._record_to_learning_service)

    def test_base_agent_has_error_classifier(self):
        """BaseAgent has _classify_error_type method."""
        from Jotty.core.intelligence.reasoning.base.base_agent import BaseAgent

        assert BaseAgent._classify_error_type("timeout error") == "timeout"
        assert BaseAgent._classify_error_type("rate limit exceeded") == "rate_limit"
        assert BaseAgent._classify_error_type("connection refused") == "infrastructure"
        assert BaseAgent._classify_error_type("some random error") == "execution_failure"
