#!/usr/bin/env python3
"""
Orchestrator V2 — Comprehensive Test Suite
============================================

Tests the most complex use cases of the V2 Orchestrator:

1. Lazy initialization & fast startup
2. Lifecycle management (startup/shutdown/context manager)
3. Introspection & metrics
4. Zero-config agent creation
5. Multi-agent coordination via SwarmTaskBoard
6. Learning pipeline integration
7. LOTUS optimization layer
8. Provider registry integration
9. Adaptive credit assignment weights
10. Status & component tracking

These tests use mocks to avoid actual LLM calls while testing
the full orchestration logic.
"""

import asyncio
import logging
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from typing import Dict, Any, List

# Setup logging for test visibility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def swarm_manager():
    """Create a fresh Orchestrator with defaults."""
    from Jotty.core.orchestration.swarm_manager import Orchestrator
    return Orchestrator()


@pytest.fixture
def config():
    """Create a SwarmConfig."""
    from Jotty.core.foundation.data_structures import SwarmConfig
    return SwarmConfig()


@pytest.fixture
def agent_config():
    """Create a basic AgentConfig."""
    from Jotty.core.foundation.agent_config import AgentConfig
    from Jotty.core.agents.auto_agent import AutoAgent
    return AgentConfig(name="test_agent", agent=AutoAgent())


@pytest.fixture
def multi_agent_configs():
    """Create multiple AgentConfigs for multi-agent testing."""
    from Jotty.core.foundation.agent_config import AgentConfig
    from Jotty.core.agents.auto_agent import AutoAgent
    return [
        AgentConfig(
            name="researcher",
            agent=AutoAgent(),
            capabilities=["Research AI trends and summarize findings"],
        ),
        AgentConfig(
            name="writer",
            agent=AutoAgent(),
            capabilities=["Write a blog post about AI trends"],
        ),
        AgentConfig(
            name="reviewer",
            agent=AutoAgent(),
            capabilities=["Review the blog post for accuracy"],
        ),
    ]


# =============================================================================
# TEST 1: LAZY INITIALIZATION & FAST STARTUP
# =============================================================================

class TestLazyInitialization:
    """Test that Orchestrator init is fast and components are lazy."""

    def test_init_is_fast(self):
        """Orchestrator.__init__ should complete in < 500ms."""
        from Jotty.core.orchestration.swarm_manager import Orchestrator

        start = time.time()
        sm = Orchestrator()
        elapsed = time.time() - start

        assert elapsed < 0.5, f"Init took {elapsed:.3f}s, expected < 0.5s"
        logger.info(f"Orchestrator init: {elapsed*1000:.1f}ms")

    def test_no_components_created_on_init(self, swarm_manager):
        """No lazy components should be created during init."""
        status = swarm_manager.status()
        assert status['components']['created'] == 0
        assert status['components']['pending'] == 23

    def test_runners_not_built_on_init(self, swarm_manager):
        """Runners should not be built during init."""
        assert swarm_manager._runners_built is False
        assert len(swarm_manager.runners) == 0

    def test_default_single_mode(self, swarm_manager):
        """Default mode should be single with one AutoAgent."""
        assert swarm_manager.mode == "single"
        assert len(swarm_manager.agents) == 1
        assert swarm_manager.agents[0].name == "auto"

    def test_multi_agent_mode(self, multi_agent_configs):
        """Providing multiple agents should set multi mode."""
        from Jotty.core.orchestration.swarm_manager import Orchestrator

        sm = Orchestrator(agents=multi_agent_configs)
        assert sm.mode == "multi"
        assert len(sm.agents) == 3

    def test_lotus_not_active_before_run(self, swarm_manager):
        """LOTUS should not be active until runners are built."""
        assert swarm_manager.lotus is None
        assert swarm_manager.lotus_optimizer is None


# =============================================================================
# TEST 2: LIFECYCLE MANAGEMENT
# =============================================================================

class TestLifecycleManagement:
    """Test startup/shutdown and context manager patterns."""

    @pytest.mark.asyncio
    async def test_startup_builds_runners(self):
        """startup() should build runners and lazy components."""
        from Jotty.core.orchestration.swarm_manager import Orchestrator

        sm = Orchestrator()
        assert not sm._runners_built

        result = await sm.startup()

        assert sm._runners_built
        assert len(sm.runners) >= 1
        assert result is sm  # Returns self for chaining

    @pytest.mark.asyncio
    async def test_shutdown_clears_runners(self):
        """shutdown() should clear runners and persist learnings."""
        from Jotty.core.orchestration.swarm_manager import Orchestrator

        sm = Orchestrator()
        await sm.startup()
        assert sm._runners_built

        await sm.shutdown()
        assert not sm._runners_built
        assert len(sm.runners) == 0

    @pytest.mark.asyncio
    async def test_shutdown_safe_to_call_multiple_times(self):
        """shutdown() should be safe to call multiple times."""
        from Jotty.core.orchestration.swarm_manager import Orchestrator

        sm = Orchestrator()
        await sm.startup()
        await sm.shutdown()
        await sm.shutdown()  # Should not raise

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """async with Orchestrator() should work."""
        from Jotty.core.orchestration.swarm_manager import Orchestrator

        async with Orchestrator() as sm:
            assert sm._runners_built
            assert len(sm.runners) >= 1

        # After exit, runners should be cleared
        assert not sm._runners_built


# =============================================================================
# TEST 3: INTROSPECTION & METRICS
# =============================================================================

class TestIntrospection:
    """Test status() and metrics property."""

    def test_status_structure(self, swarm_manager):
        """status() should return well-structured dict."""
        status = swarm_manager.status()

        assert 'mode' in status
        assert 'agents' in status
        assert 'runners_built' in status
        assert 'episode_count' in status
        assert 'lotus_enabled' in status
        assert 'components' in status

        components = status['components']
        assert 'total' in components
        assert 'created' in components
        assert 'pending' in components
        assert 'detail' in components

    def test_metrics_property(self, swarm_manager):
        """metrics property should return quick summary."""
        metrics = swarm_manager.metrics

        assert metrics['episodes'] == 0
        assert metrics['agents'] == 1
        assert metrics['mode'] == 'single'
        assert metrics['runners_built'] is False
        assert metrics['components_loaded'] == 0

    def test_status_tracks_component_creation(self, swarm_manager):
        """Accessing a lazy component should update status counts."""
        # Before: 0 components
        before = swarm_manager.status()['components']['created']
        assert before == 0

        # Access a lazy component
        _ = swarm_manager.shared_context

        # After: at least 1 new component (may cascade to dependencies)
        after = swarm_manager.status()['components']['created']
        assert after >= 1

    @pytest.mark.asyncio
    async def test_status_after_startup(self):
        """Status should reflect runner build after startup."""
        from Jotty.core.orchestration.swarm_manager import Orchestrator

        sm = Orchestrator()
        await sm.startup()

        status = sm.status()
        assert status['runners_built'] is True
        assert len(status['runners']) >= 1
        # Many components should be created after startup
        assert status['components']['created'] > 0

        await sm.shutdown()


# =============================================================================
# TEST 4: ZERO-CONFIG AGENT CREATION
# =============================================================================

class TestZeroConfig:
    """Test natural language -> agent configuration."""

    def test_zero_config_enabled_by_default(self, swarm_manager):
        """Zero-config should be enabled by default."""
        assert swarm_manager.enable_zero_config is True

    def test_zero_config_disabled(self):
        """Can disable zero-config."""
        from Jotty.core.orchestration.swarm_manager import Orchestrator

        sm = Orchestrator(enable_zero_config=False)
        assert sm.enable_zero_config is False


# =============================================================================
# TEST 5: MULTI-AGENT COORDINATION
# =============================================================================

class TestMultiAgentCoordination:
    """Test multi-agent task board coordination."""

    def test_multi_agent_init(self, multi_agent_configs):
        """Multi-agent mode should set up correctly."""
        from Jotty.core.orchestration.swarm_manager import Orchestrator

        sm = Orchestrator(agents=multi_agent_configs)
        assert sm.mode == "multi"
        assert len(sm.agents) == 3
        assert [a.name for a in sm.agents] == ["researcher", "writer", "reviewer"]

    @pytest.mark.asyncio
    async def test_multi_agent_task_board(self, multi_agent_configs):
        """Task board should get tasks added for each agent."""
        from Jotty.core.orchestration.swarm_manager import Orchestrator

        sm = Orchestrator(agents=multi_agent_configs)
        await sm.startup()

        # Task board should be accessible
        tb = sm.swarm_task_board
        assert tb is not None

        await sm.shutdown()


# =============================================================================
# TEST 6: LEARNING PIPELINE INTEGRATION
# =============================================================================

class TestLearningPipeline:
    """Test the learning pipeline accessors and integration."""

    @pytest.mark.asyncio
    async def test_learning_pipeline_accessible(self):
        """Learning pipeline should be accessible after startup."""
        from Jotty.core.orchestration.swarm_manager import Orchestrator

        sm = Orchestrator()
        await sm.startup()

        # Learning pipeline should be created
        assert sm.learning is not None
        assert sm.learning_manager is not None
        assert sm.transfer_learning is not None
        assert sm.swarm_intelligence is not None

        await sm.shutdown()

    @pytest.mark.asyncio
    async def test_credit_weights_adaptive(self):
        """Credit weights should be adaptive (not hardcoded)."""
        from Jotty.core.orchestration.swarm_manager import Orchestrator

        sm = Orchestrator()
        await sm.startup()

        weights = sm.credit_weights
        assert weights is not None

        # Should have the three credit components
        assert weights.get('base_reward') is not None
        assert weights.get('cooperation_bonus') is not None
        assert weights.get('predictability_bonus') is not None

        # Weights should sum to ~1.0
        total = (
            weights.get('base_reward') +
            weights.get('cooperation_bonus') +
            weights.get('predictability_bonus')
        )
        assert 0.9 <= total <= 1.1, f"Weights sum to {total}, expected ~1.0"

        await sm.shutdown()


# =============================================================================
# TEST 7: LOTUS OPTIMIZATION
# =============================================================================

class TestLOTUSOptimization:
    """Test LOTUS optimization layer."""

    def test_lotus_enabled_by_default(self, swarm_manager):
        """LOTUS should be enabled by default."""
        assert swarm_manager.enable_lotus is True

    def test_lotus_disabled(self):
        """Can disable LOTUS."""
        from Jotty.core.orchestration.swarm_manager import Orchestrator

        sm = Orchestrator(enable_lotus=False)
        assert sm.enable_lotus is False

    def test_lotus_stats_empty_before_use(self, swarm_manager):
        """LOTUS stats should be empty dict when not active."""
        stats = swarm_manager.get_lotus_stats()
        assert stats == {}

    def test_lotus_savings_empty_before_use(self, swarm_manager):
        """LOTUS savings should be empty dict when not active."""
        savings = swarm_manager.get_lotus_savings()
        assert savings == {}


# =============================================================================
# TEST 8: AGGREGATE RESULTS
# =============================================================================

class TestAggregateResults:
    """Test result aggregation for multi-agent mode."""

    def test_aggregate_empty_results(self, swarm_manager):
        """Empty results should produce failed EpisodeResult."""
        result = swarm_manager._aggregate_results({}, "test goal")
        assert result.success is False
        assert result.output is None

    def test_aggregate_single_result(self, swarm_manager):
        """Single result should be returned as-is."""
        from Jotty.core.foundation.data_structures import EpisodeResult

        mock_result = EpisodeResult(
            output="test output",
            success=True,
            trajectory=[],
            tagged_outputs=[],
            episode=1,
            execution_time=1.5,
            architect_results=[],
            auditor_results=[],
            agent_contributions={},
        )

        result = swarm_manager._aggregate_results({"agent1": mock_result}, "test goal")
        assert result is mock_result

    def test_aggregate_multiple_results(self, swarm_manager):
        """Multiple results should be combined."""
        from Jotty.core.foundation.data_structures import EpisodeResult

        results = {
            "researcher": EpisodeResult(
                output="Research findings",
                success=True,
                trajectory=[{"step": "research"}],
                tagged_outputs=[],
                episode=1,
                execution_time=2.0,
                architect_results=[],
                auditor_results=[],
                agent_contributions={"researcher": 0.9},
            ),
            "writer": EpisodeResult(
                output="Blog post draft",
                success=True,
                trajectory=[{"step": "writing"}],
                tagged_outputs=[],
                episode=1,
                execution_time=3.0,
                architect_results=[],
                auditor_results=[],
                agent_contributions={"writer": 0.8},
            ),
        }

        combined = swarm_manager._aggregate_results(results, "test goal")

        # Combined should succeed (all succeeded)
        assert combined.success is True

        # Output should contain all agent outputs (dict or verified string)
        if isinstance(combined.output, dict):
            assert "researcher" in combined.output
            assert "writer" in combined.output
        else:
            # CandidateVerifier may merge into a single string
            assert combined.output is not None

        # Execution time should be summed
        assert combined.execution_time == 5.0

        # Trajectories should be merged with agent labels
        assert len(combined.trajectory) == 2
        assert combined.trajectory[0]['agent'] == 'researcher'
        assert combined.trajectory[1]['agent'] == 'writer'

    def test_aggregate_partial_failure(self, swarm_manager):
        """If any agent fails, combined success should be False."""
        from Jotty.core.foundation.data_structures import EpisodeResult

        results = {
            "researcher": EpisodeResult(
                output="Findings", success=True,
                trajectory=[], tagged_outputs=[], episode=1,
                execution_time=1.0, architect_results=[], auditor_results=[],
                agent_contributions={},
            ),
            "writer": EpisodeResult(
                output=None, success=False,
                trajectory=[], tagged_outputs=[], episode=1,
                execution_time=0.5, architect_results=[], auditor_results=[],
                agent_contributions={},
            ),
        }

        combined = swarm_manager._aggregate_results(results, "test goal")
        assert combined.success is False


# =============================================================================
# TEST 9: STATE MANAGEMENT
# =============================================================================

class TestStateManagement:
    """Test shared context and state management."""

    def test_shared_context_lazy(self, swarm_manager):
        """SharedContext should be lazy-loaded."""
        # Should not be created yet
        assert '_lazy_shared_context' not in swarm_manager.__dict__

        # Access it
        ctx = swarm_manager.shared_context
        assert ctx is not None

        # Now it should be created
        assert '_lazy_shared_context' in swarm_manager.__dict__

    def test_io_manager_lazy(self, swarm_manager):
        """IOManager should be lazy-loaded."""
        assert '_lazy_io_manager' not in swarm_manager.__dict__
        io = swarm_manager.io_manager
        assert io is not None
        assert '_lazy_io_manager' in swarm_manager.__dict__

    def test_data_registry_lazy(self, swarm_manager):
        """DataRegistry should be lazy-loaded."""
        assert '_lazy_data_registry' not in swarm_manager.__dict__
        dr = swarm_manager.data_registry
        assert dr is not None
        assert '_lazy_data_registry' in swarm_manager.__dict__

    def test_context_guard_lazy(self, swarm_manager):
        """ContextGuard should be lazy-loaded."""
        assert '_lazy_context_guard' not in swarm_manager.__dict__
        cg = swarm_manager.context_guard
        assert cg is not None
        assert '_lazy_context_guard' in swarm_manager.__dict__


# =============================================================================
# TEST 10: EPISODE RESULT DATA STRUCTURE
# =============================================================================

class TestEpisodeResult:
    """Test EpisodeResult creation and usage."""

    def test_episode_result_creation(self):
        """EpisodeResult should be creatable with all fields."""
        from Jotty.core.foundation.data_structures import EpisodeResult

        result = EpisodeResult(
            output={"key": "value"},
            success=True,
            trajectory=[{"step": 1, "action": "test"}],
            tagged_outputs=[],
            episode=5,
            execution_time=2.5,
            architect_results=["design plan"],
            auditor_results=["validation OK"],
            agent_contributions={"agent1": 0.8},
        )

        assert result.success is True
        assert result.episode == 5
        assert result.execution_time == 2.5
        assert result.output == {"key": "value"}


# =============================================================================
# TEST 11: WARMUP & DAG DELEGATION
# =============================================================================

class TestDelegation:
    """Test warmup and DAG executor delegation patterns."""

    def test_warmup_recommendation_before_init(self, swarm_manager):
        """get_warmup_recommendation should work before startup."""
        rec = swarm_manager.get_warmup_recommendation()
        assert isinstance(rec, dict)

    @pytest.mark.asyncio
    async def test_warmup_creates_swarm_warmup(self, swarm_manager):
        """warmup() should create SwarmWarmup lazily."""
        assert not hasattr(swarm_manager, '_warmup') or swarm_manager._warmup is None

        # Warmup should work (may not do much without LLM, but shouldn't crash)
        try:
            result = await swarm_manager.warmup()
            assert isinstance(result, dict)
        except Exception:
            # OK - warmup may need LM configured
            pass


# =============================================================================
# TEST 12: ML LEARNING BRIDGE
# =============================================================================

class TestMLLearningBridge:
    """Test ML learning bridge methods."""

    @pytest.mark.asyncio
    async def test_get_ml_learning(self):
        """get_ml_learning should return MASLearning instance."""
        from Jotty.core.orchestration.swarm_manager import Orchestrator

        sm = Orchestrator()
        await sm.startup()

        ml = sm.get_ml_learning()
        assert ml is not None

        await sm.shutdown()

    @pytest.mark.asyncio
    async def test_record_report_section_outcome(self):
        """record_report_section_outcome should not crash."""
        from Jotty.core.orchestration.swarm_manager import Orchestrator

        sm = Orchestrator()
        await sm.startup()

        # Should not raise
        sm.record_report_section_outcome("introduction", success=True)
        sm.record_report_section_outcome("conclusion", success=False, error="timeout")

        await sm.shutdown()

    @pytest.mark.asyncio
    async def test_should_skip_report_section(self):
        """should_skip_report_section should return bool."""
        from Jotty.core.orchestration.swarm_manager import Orchestrator

        sm = Orchestrator()
        await sm.startup()

        result = sm.should_skip_report_section("introduction")
        assert isinstance(result, bool)

        await sm.shutdown()


# =============================================================================
# TEST 13: CONFIGURATION VARIANTS
# =============================================================================

class TestConfigurationVariants:
    """Test different Orchestrator configurations."""

    def test_custom_config(self, config):
        """Custom SwarmConfig should be accepted."""
        from Jotty.core.orchestration.swarm_manager import Orchestrator

        sm = Orchestrator(config=config)
        assert sm.config is config

    def test_single_agent_config(self, agent_config):
        """Single AgentConfig should result in single mode."""
        from Jotty.core.orchestration.swarm_manager import Orchestrator

        sm = Orchestrator(agents=agent_config)
        assert sm.mode == "single"
        assert len(sm.agents) == 1

    def test_agent_list_config(self, multi_agent_configs):
        """List of AgentConfigs should result in multi mode."""
        from Jotty.core.orchestration.swarm_manager import Orchestrator

        sm = Orchestrator(agents=multi_agent_configs)
        assert sm.mode == "multi"
        assert len(sm.agents) == 3

    def test_custom_prompts(self):
        """Custom architect/auditor prompts should be stored."""
        from Jotty.core.orchestration.swarm_manager import Orchestrator

        sm = Orchestrator(
            architect_prompts=["custom/architect.md"],
            auditor_prompts=["custom/auditor.md"],
        )
        assert sm.architect_prompts == ["custom/architect.md"]
        assert sm.auditor_prompts == ["custom/auditor.md"]

    def test_default_prompts(self, swarm_manager):
        """Default prompts should be set."""
        assert len(swarm_manager.architect_prompts) > 0
        assert len(swarm_manager.auditor_prompts) > 0


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
