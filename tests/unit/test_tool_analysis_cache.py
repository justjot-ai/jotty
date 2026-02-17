"""Tests for tool analysis caching between pre/post execution."""

from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from Jotty.core.intelligence.orchestration.swarms._base._learning_mixin import (
    SwarmLearningMixin,
)


class FakeSwarm(SwarmLearningMixin):
    """Minimal class for testing the mixin's learning lifecycle."""

    def __init__(self):
        self._swarm_intelligence = None
        self._memory = None
        self._memory_persistence = None
        self._learned_context = None
        self._traces = []
        self._gold_db = None
        self._improvement_history = None
        self._evaluation_history = MagicMock()
        self._expert = None
        self._reviewer = None
        self._planner = None
        self._actor = None
        self._auditor = None
        self._learner = None
        self._cached_tool_analysis = None
        self.config = MagicMock()
        self.config.name = "test_swarm"
        self.config.domain = "test"
        self.config.enable_self_improvement = False
        self.config.enable_learning = False
        self.config.improvement_threshold = 0.7

    def _manage_tools(self):
        return {
            "weak_tools": [],
            "strong_tools": [],
            "suggested_removals": [],
            "replacements": {},
        }

    def connect_swarm_intelligence(self):
        pass

    def _get_intelligence_save_path(self):
        return "/tmp/test_si.json"

    def _run_auto_warmup(self):
        return {"seeded": 0}

    def _retrieve_expert_knowledge(self):
        return []

    def _analyze_prior_failures(self):
        return []

    def _send_executor_feedback(self, **kwargs):
        pass

    def _coordinate_post_execution(self, **kwargs):
        pass

    def _store_execution_as_improvement(self, **kwargs):
        pass


class TestToolAnalysisCache:
    """Verify _manage_tools() is not called redundantly in post if cached."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_pre_execution_caches_tool_analysis(self) -> None:
        """_pre_execute_learning() stores cached tool analysis."""
        swarm = FakeSwarm()

        # Set up a minimal SwarmIntelligence mock
        si = MagicMock()
        si.agent_profiles = {}
        si.curriculum_generator.get_curriculum_stats.return_value = {
            "feedback_count": 1,
            "tool_success_rates": {},
        }
        si.morph_score_history = []
        si.coalitions = {}
        swarm._swarm_intelligence = si

        await swarm._pre_execute_learning()

        # Verify the cache was populated
        assert swarm._cached_tool_analysis is not None
        assert isinstance(swarm._cached_tool_analysis, dict)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_post_execution_skips_manage_tools_when_cached(self) -> None:
        """_post_execute_learning() skips _manage_tools() when cache exists."""
        swarm = FakeSwarm()

        # Pre-populate the cache
        swarm._cached_tool_analysis = {
            "weak_tools": [],
            "strong_tools": [{"tool": "web-search", "success_rate": 0.95}],
            "suggested_removals": [],
            "replacements": {},
        }

        # Mock SwarmIntelligence
        si = MagicMock()
        si.agent_profiles = {}
        si.morph_score_history = []
        si.coalitions = {}
        swarm._swarm_intelligence = si

        # Track _manage_tools calls
        call_count = 0
        original = swarm._manage_tools

        def counting_manage_tools():
            nonlocal call_count
            call_count += 1
            return original()

        swarm._manage_tools = counting_manage_tools

        await swarm._post_execute_learning(
            success=True,
            execution_time=1.0,
            tools_used=["web-search"],
            task_type="test",
        )

        # _manage_tools should NOT have been called (cache was non-empty)
        assert call_count == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_post_execution_calls_manage_tools_when_not_cached(self) -> None:
        """_post_execute_learning() calls _manage_tools() when no cache."""
        swarm = FakeSwarm()
        swarm._cached_tool_analysis = None  # No cache

        si = MagicMock()
        si.agent_profiles = {}
        si.morph_score_history = []
        si.coalitions = {}
        swarm._swarm_intelligence = si

        call_count = 0
        original = swarm._manage_tools

        def counting_manage_tools():
            nonlocal call_count
            call_count += 1
            return original()

        swarm._manage_tools = counting_manage_tools

        await swarm._post_execute_learning(
            success=True,
            execution_time=1.0,
            tools_used=[],
            task_type="test",
        )

        # _manage_tools SHOULD have been called (no cache)
        assert call_count == 1
