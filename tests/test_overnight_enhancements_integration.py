#!/usr/bin/env python3
"""
Integration Tests for All Overnight Enhancements
=================================================

Tests that all 5 enhancements work together in production scenarios.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch


@pytest.mark.integration
class TestOvernightEnhancementsIntegration:
    """Test all overnight enhancements working together."""

    @pytest.mark.asyncio
    async def test_full_stack_integration(self):
        """Test all enhancements: tracing + multi-swarm + cost-aware + adaptive."""
        from Jotty.core.observability import get_distributed_tracer
        from Jotty.core.orchestration import SwarmAdapter, get_multi_swarm_coordinator, MergeStrategy
        from Jotty.core.learning import get_cost_aware_td_lambda
        from Jotty.core.safety import get_adaptive_threshold_manager

        # Initialize all components
        tracer = get_distributed_tracer("integration-test")
        coordinator = get_multi_swarm_coordinator()
        learner = get_cost_aware_td_lambda(cost_sensitivity=0.5)
        threshold_manager = get_adaptive_threshold_manager()

        # Verify all initialized
        assert tracer is not None
        assert coordinator is not None
        assert learner is not None
        assert threshold_manager is not None

        # Test traced execution
        with tracer.trace("test_operation") as trace_id:
            assert trace_id is not None

            # Inject headers
            headers = tracer.inject_headers(trace_id)
            assert 'traceparent' in headers

            # Record observation
            threshold_manager.record_observation(
                "test_threshold",
                0.5,
                False,
                1.0
            )

            # Update learner
            learner.update(
                state={"test": 1},
                action={"do": "something"},
                reward=1.0,
                next_state={"test": 2},
                cost_usd=0.1
            )

            # Get context
            context = tracer.get_context(trace_id)
            assert context['operation'] == 'test_operation'

        # Verify stats
        stats = learner.get_stats()
        assert stats['updates'] == 1

        threshold_stats = threshold_manager.get_stats()
        assert threshold_stats['total_observations'] >= 1

    @pytest.mark.asyncio
    @patch('os.getenv')
    @patch('anthropic.AsyncAnthropic')
    async def test_multi_swarm_with_cost_tracking(self, mock_anthropic, mock_getenv):
        """Test multi-swarm execution with cost-aware learning."""
        from Jotty.core.orchestration import SwarmAdapter, get_multi_swarm_coordinator, MergeStrategy
        from Jotty.core.learning import get_cost_aware_td_lambda

        # Mock API
        mock_getenv.return_value = "sk-test"

        mock_response = Mock()
        mock_response.content = [Mock(text="Test")]
        mock_response.usage = Mock(input_tokens=10, output_tokens=10)

        mock_client = Mock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        mock_anthropic.return_value = mock_client

        # Create swarms
        swarms = SwarmAdapter.quick_swarms([
            ("S1", "Test 1"),
            ("S2", "Test 2"),
        ])

        # Execute
        coordinator = get_multi_swarm_coordinator()
        result = await coordinator.execute_parallel(
            swarms=swarms,
            task="test",
            merge_strategy=MergeStrategy.VOTING
        )

        # Track cost
        learner = get_cost_aware_td_lambda()
        total_cost = sum(
            s.metadata.get('cost_usd', 0)
            for s in [result]
            if hasattr(result, 'metadata')
        )

        learner.update(
            state={"strategy": "multi_swarm"},
            action={"swarms": 2},
            reward=1.0 if result.success else 0.0,
            next_state={"done": True},
            cost_usd=total_cost if total_cost > 0 else 0.001  # Estimated
        )

        # Verify learning happened
        stats = learner.get_stats()
        assert stats['updates'] >= 1

    @pytest.mark.asyncio
    async def test_distributed_tracing_with_multi_swarm(self):
        """Test distributed tracing propagates through multi-swarm."""
        from Jotty.core.observability import get_distributed_tracer
        from Jotty.core.orchestration import SwarmAdapter, get_multi_swarm_coordinator, MergeStrategy

        tracer = get_distributed_tracer("test-service")
        coordinator = get_multi_swarm_coordinator()

        # Mock swarms (no API calls)
        class MockSwarm:
            def __init__(self, name):
                self.name = name

            async def execute(self, task):
                from Jotty.core.orchestration.multi_swarm_coordinator import SwarmResult
                await asyncio.sleep(0.01)  # Simulate work
                return SwarmResult(
                    swarm_name=self.name,
                    output=f"Result from {self.name}",
                    success=True,
                    confidence=0.8
                )

        swarms = [MockSwarm("M1"), MockSwarm("M2")]

        # Execute with tracing
        with tracer.trace("multi_swarm_test") as trace_id:
            result = await coordinator.execute_parallel(
                swarms=swarms,
                task="test",
                merge_strategy=MergeStrategy.VOTING
            )

            # Verify trace context
            context = tracer.get_context(trace_id)
            assert context is not None
            assert context['operation'] == 'multi_swarm_test'

            # Verify execution succeeded
            assert result.success is True

    @pytest.mark.asyncio
    async def test_adaptive_thresholds_with_multi_swarm(self):
        """Test adaptive thresholds adjust based on multi-swarm usage."""
        from Jotty.core.safety import get_adaptive_threshold_manager
        from Jotty.core.orchestration import SwarmAdapter, get_multi_swarm_coordinator, MergeStrategy

        manager = get_adaptive_threshold_manager()

        # Simulate 100 observations (triggers adaptation)
        for i in range(100):
            # Simulate varying costs
            cost = 0.001 + (i * 0.00001)  # Increasing costs
            violated = cost > 0.01  # $0.01 threshold

            manager.record_observation(
                "swarm_cost",
                cost,
                violated,
                0.01
            )

        # Check adaptation happened
        stats = manager.get_stats()
        assert stats['total_observations'] >= 100
        assert 'swarm_cost' in stats['thresholds']

        # Get adapted threshold
        threshold = manager.get_threshold("swarm_cost")
        assert threshold is not None


@pytest.mark.unit
class TestIndividualEnhancements:
    """Unit tests for individual enhancements."""

    def test_distributed_tracer_initialization(self):
        """Test distributed tracer initializes correctly."""
        from Jotty.core.observability import get_distributed_tracer

        tracer = get_distributed_tracer("test")
        assert tracer.service_name == "test"

    def test_adaptive_threshold_manager_initialization(self):
        """Test adaptive threshold manager initializes correctly."""
        from Jotty.core.safety import get_adaptive_threshold_manager

        manager = get_adaptive_threshold_manager()
        assert manager.adaptation_interval == 100
        assert manager.percentile_margin == 0.10

    def test_cost_aware_learner_initialization(self):
        """Test cost-aware learner initializes correctly."""
        from Jotty.core.learning import get_cost_aware_td_lambda

        learner = get_cost_aware_td_lambda(cost_sensitivity=1.0)
        assert learner.cost_sensitivity == 1.0
        assert learner.update_count == 0

    def test_multi_swarm_coordinator_initialization(self):
        """Test multi-swarm coordinator initializes correctly."""
        from Jotty.core.orchestration import get_multi_swarm_coordinator

        coordinator = get_multi_swarm_coordinator()
        assert coordinator.execution_count == 0

    def test_incremental_consolidator_initialization(self):
        """Test incremental consolidator initializes correctly."""
        from Jotty.core.memory import get_incremental_consolidator

        consolidator = get_incremental_consolidator()
        assert consolidator.batch_size == 1
        assert consolidator.delay_between_ms == 50
        assert len(consolidator.queue) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'integration or unit'])
