"""Tests for double coalition formation guard in execute_team()."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from Jotty.core.intelligence.orchestration.swarms.base.swarm_template import (
    SwarmTemplate,
)
from Jotty.core.intelligence.orchestration.swarms.base.team_coordinator import (
    CoordinationPattern,
    TeamCoordinator,
    TeamResult,
)
from Jotty.core.intelligence.orchestration.swarms._base.swarm_types import (
    SwarmBaseConfig,
    SwarmResult,
)


class _DummyAgent:
    """Minimal agent for team tests."""

    def __init__(self, **kwargs):
        pass


class DummySwarm(SwarmTemplate):
    """Minimal SwarmTemplate subclass with PARALLEL team for testing."""

    AGENT_TEAM = TeamCoordinator.define(
        (_DummyAgent, "AgentA"),
        (_DummyAgent, "AgentB"),
        pattern=CoordinationPattern.PARALLEL,
    )

    async def _execute_domain(self, *args, **kwargs) -> SwarmResult:
        return SwarmResult(
            success=True,
            swarm_name="dummy",
            domain="test",
            output={},
            execution_time=0.1,
        )


class TestCoalitionGuard:
    """Verify coalitions are formed at most once regardless of call order."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_team_skips_coalition_when_already_formed(self) -> None:
        """execute_team() does NOT form coalition when si.coalitions is non-empty."""
        config = SwarmBaseConfig(name="test", domain="test")
        swarm = DummySwarm(config)
        swarm._agents_initialized = True
        swarm._agent_a = _DummyAgent()
        swarm._agent_b = _DummyAgent()

        # Mock SwarmIntelligence with existing coalitions
        si = MagicMock()
        si.agent_profiles = {"AgentA": MagicMock(), "AgentB": MagicMock()}
        si.coalitions = {"existing_coalition": MagicMock()}  # Non-empty
        si.smart_route.return_value = {}
        swarm._swarm_intelligence = si

        # Mock team execution to return a result
        mock_result = TeamResult(
            success=True,
            outputs={"AgentA": "ok", "AgentB": "ok"},
            merged_output="ok ok",
        )
        swarm.AGENT_TEAM.set_instances({"_agent_a": swarm._agent_a, "_agent_b": swarm._agent_b})
        with patch.object(swarm.AGENT_TEAM, "execute", return_value=mock_result):
            await swarm.execute_team(task="test task")

        # form_coalition should NOT have been called because coalitions was non-empty
        si.form_coalition.assert_not_called()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_team_forms_coalition_when_empty(self) -> None:
        """execute_team() DOES form coalition when si.coalitions is empty."""
        config = SwarmBaseConfig(name="test", domain="test")
        swarm = DummySwarm(config)
        swarm._agents_initialized = True
        swarm._agent_a = _DummyAgent()
        swarm._agent_b = _DummyAgent()

        # Mock SwarmIntelligence with NO existing coalitions
        si = MagicMock()
        si.agent_profiles = {"AgentA": MagicMock(), "AgentB": MagicMock()}
        si.coalitions = {}  # Empty — should trigger formation
        si.smart_route.return_value = {}
        si.form_coalition.return_value = MagicMock(coalition_id="new", leader="AgentA")
        swarm._swarm_intelligence = si

        mock_result = TeamResult(
            success=True,
            outputs={"AgentA": "ok", "AgentB": "ok"},
            merged_output="ok ok",
        )
        swarm.AGENT_TEAM.set_instances({"_agent_a": swarm._agent_a, "_agent_b": swarm._agent_b})
        with patch.object(swarm.AGENT_TEAM, "execute", return_value=mock_result):
            await swarm.execute_team(task="test task")

        # form_coalition SHOULD have been called
        si.form_coalition.assert_called_once()
