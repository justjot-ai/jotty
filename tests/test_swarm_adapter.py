#!/usr/bin/env python3
"""
Tests for SwarmAdapter
======================

Tests the zero-wrapper multi-swarm integration.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch


@pytest.fixture
def mock_swarm():
    """Create a mock swarm with run() method."""
    swarm = Mock()
    swarm.name = "TestSwarm"
    swarm.run = AsyncMock(return_value={
        'output': 'Test output',
        'success': True,
        'confidence': 0.8
    })
    return swarm


@pytest.fixture
def mock_agent():
    """Create a mock agent with execute() method."""
    agent = Mock()
    agent.name = "TestAgent"
    agent.execute = AsyncMock(return_value="Agent output")
    return agent


class TestSwarmAdapter:
    """Test SwarmAdapter functionality."""

    @pytest.mark.asyncio
    async def test_from_swarms_with_run_method(self, mock_swarm):
        """Test adapting swarm with run() method."""
        from Jotty.core.intelligence.orchestration.swarm_adapter import SwarmAdapter

        # Adapt swarms
        adapted = SwarmAdapter.from_swarms([mock_swarm])

        assert len(adapted) == 1
        assert hasattr(adapted[0], 'execute')

        # Execute adapted swarm
        result = await adapted[0].execute("test task")

        assert result.swarm_name == "TestSwarm"
        assert result.output == "Test output"
        assert result.success is True
        assert result.confidence == 0.8

        # Verify run() was called
        mock_swarm.run.assert_called_once_with(goal="test task")

    @pytest.mark.asyncio
    async def test_from_swarms_with_execute_method(self):
        """Test swarm that already has execute() method."""
        from Jotty.core.intelligence.orchestration.swarm_adapter import SwarmAdapter
        from Jotty.core.intelligence.orchestration.multi_swarm_coordinator import SwarmResult

        # Create swarm with execute()
        class AlreadyCompatible:
            name = "Compatible"

            async def execute(self, task):
                return SwarmResult(
                    swarm_name=self.name,
                    output="Already compatible",
                    success=True,
                    confidence=0.9
                )

        swarm = AlreadyCompatible()

        # Adapt (should use directly)
        adapted = SwarmAdapter.from_swarms([swarm])

        assert len(adapted) == 1
        assert adapted[0] is swarm  # Should be same object

    @pytest.mark.asyncio
    async def test_from_agents(self, mock_agent):
        """Test adapting agents to swarms."""
        from Jotty.core.intelligence.orchestration.swarm_adapter import SwarmAdapter

        # Adapt agents
        adapted = SwarmAdapter.from_agents([mock_agent])

        assert len(adapted) == 1
        assert hasattr(adapted[0], 'execute')

        # Execute adapted agent
        result = await adapted[0].execute("test task")

        assert result.swarm_name == "TestAgent"
        assert "Agent output" in result.output
        assert result.success is True

        # Verify execute() was called
        mock_agent.execute.assert_called_once_with("test task")

    @pytest.mark.asyncio
    @patch('os.getenv')
    @patch('anthropic.AsyncAnthropic')
    async def test_quick_swarms(self, mock_anthropic_class, mock_getenv):
        """Test creating quick swarms from prompts."""
        from Jotty.core.intelligence.orchestration.swarm_adapter import SwarmAdapter

        # Mock API key
        mock_getenv.return_value = "sk-test-key"

        # Mock Anthropic response
        mock_response = Mock()
        mock_response.content = [Mock(text="Test response")]
        mock_response.usage = Mock(input_tokens=10, output_tokens=20)

        mock_client = Mock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        mock_anthropic_class.return_value = mock_client

        # Create quick swarms
        swarms = SwarmAdapter.quick_swarms([
            ("Swarm1", "Prompt 1"),
            ("Swarm2", "Prompt 2"),
        ])

        assert len(swarms) == 2
        assert swarms[0].name == "Swarm1"
        assert swarms[1].name == "Swarm2"

        # Execute one
        result = await swarms[0].execute("test task")

        assert result.swarm_name == "Swarm1"
        assert result.output == "Test response"
        assert result.success is True
        assert 'cost_usd' in result.metadata

        # Verify API was called
        mock_client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_adapter_error_handling(self):
        """Test error handling in adapters."""
        from Jotty.core.intelligence.orchestration.swarm_adapter import SwarmAdapter

        # Create swarm that raises error
        class FailingSwarm:
            name = "Failing"

            async def run(self, goal):
                raise ValueError("Simulated failure")

        swarm = FailingSwarm()

        # Adapt
        adapted = SwarmAdapter.from_swarms([swarm])

        # Execute (should return failed SwarmResult, not raise)
        result = await adapted[0].execute("test")

        assert result.success is False
        assert "Error" in result.output
        assert result.confidence == 0.0


class TestSwarmAdapterIntegration:
    """Integration tests with Multi-Swarm Coordinator."""

    @pytest.mark.asyncio
    @patch('os.getenv')
    @patch('anthropic.AsyncAnthropic')
    async def test_full_integration(self, mock_anthropic_class, mock_getenv):
        """Test complete workflow: SwarmAdapter + Coordinator."""
        from Jotty.core.intelligence.orchestration import (
            SwarmAdapter,
            get_multi_swarm_coordinator,
            MergeStrategy
        )

        # Mock API
        mock_getenv.return_value = "sk-test-key"

        mock_response = Mock()
        mock_response.content = [Mock(text="Integration test response")]
        mock_response.usage = Mock(input_tokens=10, output_tokens=20)

        mock_client = Mock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        mock_anthropic_class.return_value = mock_client

        # Create swarms
        swarms = SwarmAdapter.quick_swarms([
            ("Test1", "Prompt 1"),
            ("Test2", "Prompt 2"),
        ])

        # Execute with coordinator
        coordinator = get_multi_swarm_coordinator()
        result = await coordinator.execute_parallel(
            swarms=swarms,
            task="test integration",
            merge_strategy=MergeStrategy.VOTING
        )

        # Verify result
        assert result.success is True
        assert result.output is not None

        # Verify stats
        stats = coordinator.get_stats()
        assert stats['total_executions'] >= 1


@pytest.mark.asyncio
async def test_swarm_adapter_exports():
    """Test that SwarmAdapter is properly exported."""
    # Should be importable from orchestration
    from Jotty.core.intelligence.orchestration import SwarmAdapter

    assert SwarmAdapter is not None
    assert hasattr(SwarmAdapter, 'quick_swarms')
    assert hasattr(SwarmAdapter, 'from_swarms')
    assert hasattr(SwarmAdapter, 'from_agents')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
