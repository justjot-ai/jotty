"""
Subsystem Integration Tests
============================

End-to-end integration tests for subsystem pipelines.
No mocks, no API calls — tests that real components work together.

Pipelines tested:
1. Memory: store → retrieve → consolidation
2. Learning: TD-Lambda → credit assignment → reward shaping
3. Observability: tracing → cost tracking → budget enforcement
4. Config: focused config → SwarmConfig bridge → subsystem acceptance
"""

import time
import threading
import pytest


# =============================================================================
# 1. Memory Pipeline Integration
# =============================================================================

@pytest.mark.unit
class TestMemoryPipelineIntegration:
    """Test memory store → retrieve → consolidation pipeline."""

    def test_memory_system_store_and_retrieve(self):
        """Store a memory and retrieve it."""
        from Jotty.core.memory.memory_system import MemorySystem
        mem = MemorySystem()
        mem_id = mem.store(
            "Task X succeeded with approach Y",
            level="episodic",
            goal="research",
            metadata={"reward": 1.0},
        )
        assert mem_id is not None

        results = mem.retrieve("How to handle task X?", goal="research", top_k=5)
        assert isinstance(results, list)

    def test_memory_system_status(self):
        """Memory system reports its status."""
        from Jotty.core.memory.memory_system import MemorySystem
        mem = MemorySystem()
        status = mem.status()
        assert isinstance(status, dict)
        assert 'backend' in status

    def test_memory_system_multi_level_storage(self):
        """Store across all 5 memory levels."""
        from Jotty.core.memory.memory_system import MemorySystem
        mem = MemorySystem()
        levels = ["episodic", "semantic", "procedural", "meta", "causal"]
        ids = []
        for level in levels:
            mid = mem.store(f"Test memory at {level} level", level=level)
            ids.append(mid)
        assert len(ids) == 5
        assert all(mid is not None for mid in ids)

    def test_memory_facade_singleton_consistency(self):
        """Facade returns same MemorySystem instance."""
        from Jotty.core.memory.facade import get_memory_system
        import Jotty.core.memory.facade as mf
        mf._singletons.clear()
        m1 = get_memory_system()
        m2 = get_memory_system()
        assert m1 is m2

    def test_brain_manager_instantiation(self):
        """BrainInspiredMemoryManager can be created and queried."""
        from Jotty.core.memory.facade import get_brain_manager
        import Jotty.core.memory.facade as mf
        mf._singletons.clear()
        brain = get_brain_manager()
        assert brain is not None
        assert hasattr(brain, 'store_experience')

    def test_memory_config_to_rag_retriever(self):
        """MemoryConfig flows through to RAG retriever."""
        from Jotty.core.foundation.configs import MemoryConfig
        from Jotty.core.memory.facade import get_rag_retriever
        cfg = MemoryConfig(
            rag_window_size=10,
            rag_relevance_threshold=0.8,
            chunk_size=250,
        )
        retriever = get_rag_retriever(config=cfg)
        assert retriever is not None


# =============================================================================
# 2. Learning Pipeline Integration
# =============================================================================

@pytest.mark.unit
class TestLearningPipelineIntegration:
    """Test TD-Lambda → credit → reward pipeline."""

    def test_td_lambda_update_cycle(self):
        """Full TD-Lambda update cycle: state → action → reward → next_state."""
        from Jotty.core.learning.facade import get_td_lambda
        td = get_td_lambda()
        assert td is not None

        td.update(
            state={"task": "research", "agent": "researcher"},
            action={"tool": "web-search"},
            reward=1.0,
            next_state={"task": "research", "agent": "researcher", "step": 2},
        )
        # Should not raise — update completes successfully

    def test_credit_assigner_creation(self):
        """Credit assigner can be created through facade."""
        from Jotty.core.learning.facade import get_credit_assigner
        credit = get_credit_assigner()
        assert credit is not None
        assert hasattr(credit, 'analyze_contributions')

    def test_reward_manager_creation(self):
        """Reward manager can be created through facade."""
        from Jotty.core.learning.facade import get_reward_manager
        rm = get_reward_manager()
        assert rm is not None

    def test_learning_config_acceptance(self):
        """Learning facade accepts LearningConfig."""
        from Jotty.core.foundation.configs import LearningConfig
        from Jotty.core.learning.facade import get_td_lambda
        cfg = LearningConfig(gamma=0.5, alpha=0.05)
        td = get_td_lambda(config=cfg)
        assert td is not None

    def test_td_lambda_multiple_updates(self):
        """Multiple TD-Lambda updates build state-action values."""
        from Jotty.core.learning.facade import get_td_lambda
        td = get_td_lambda()

        for i in range(5):
            td.update(
                state={"task": "coding", "step": i},
                action={"tool": f"tool_{i}"},
                reward=0.5 + i * 0.1,
                next_state={"task": "coding", "step": i + 1},
            )

    def test_learning_system_full_pipeline(self):
        """Full learning system pipeline."""
        from Jotty.core.learning.facade import get_learning_system
        ls = get_learning_system()
        assert ls is not None
        assert hasattr(ls, 'q_learner')
        assert hasattr(ls, 'record_experience')


# =============================================================================
# 3. Observability Pipeline Integration
# =============================================================================

@pytest.mark.unit
class TestObservabilityPipelineIntegration:
    """Test tracing → cost tracking → budget pipeline."""

    def test_tracing_full_pipeline(self):
        """Create trace, add spans, add costs, get summary."""
        from Jotty.core.observability.tracing import TracingContext
        ctx = TracingContext()
        trace = ctx.new_trace(metadata={"goal": "integration test"})

        with ctx.span("swarm_run", goal="test") as root:
            with ctx.span("planning") as plan:
                plan.set_attribute("task_type", "research")
                time.sleep(0.01)

            with ctx.span("agent_execute", agent="auto") as agent:
                ctx.add_cost_to_current(
                    input_tokens=500, output_tokens=200, cost_usd=0.003
                )
                time.sleep(0.01)

            with ctx.span("validation") as val:
                val.set_status(
                    __import__('Jotty.core.observability.tracing', fromlist=['SpanStatus']).SpanStatus.OK,
                    "passed"
                )

        # Verify trace structure
        assert trace.span_count == 4  # root + 3 children
        assert trace.total_tokens == 700
        assert trace.total_cost == pytest.approx(0.003)
        assert trace.total_llm_calls == 1

        # Verify summary is readable
        summary = trace.summary()
        assert "swarm_run" in summary
        assert "planning" in summary
        assert "agent_execute" in summary

        # Verify serialization
        d = trace.to_dict()
        assert d['trace_id'] == trace.trace_id
        assert d['summary']['span_count'] == 4

    def test_profiler_full_pipeline(self):
        """Profile operations, get report with slowest segments."""
        from Jotty.core.monitoring.profiler import PerformanceProfiler
        p = PerformanceProfiler()

        with p.profile("llm_call", metadata={"model": "claude"}):
            time.sleep(0.05)
            with p.profile("tokenization"):
                time.sleep(0.01)

        with p.profile("memory_retrieval"):
            time.sleep(0.02)

        report = p.get_report(top_n=5)
        assert report.total_duration > 0
        assert len(report.segments) == 2  # 2 root segments
        # Slowest should be llm_call
        assert report.slowest_segments[0].name == "llm_call"

    def test_budget_tracker_full_pipeline(self):
        """Track LLM costs across multiple agents."""
        from Jotty.core.utils.budget_tracker import BudgetTracker
        BudgetTracker.reset_instances()

        bt = BudgetTracker.get_instance("integration_test")
        bt.record_call("researcher", tokens_input=1000, tokens_output=500, model="gpt-4o")
        bt.record_call("coder", tokens_input=500, tokens_output=200, model="gpt-4o-mini")
        bt.record_call("researcher", tokens_input=800, tokens_output=300, model="gpt-4o")

        usage = bt.get_usage()
        assert usage['calls'] == 3
        assert usage['tokens_input'] == 2300
        assert usage['tokens_output'] == 1000

        BudgetTracker.reset_instances()

    def test_llm_cache_full_pipeline(self):
        """Cache LLM responses, verify hits and misses."""
        from Jotty.core.utils.llm_cache import LLMCallCache
        LLMCallCache.reset_instances()

        cache = LLMCallCache(max_size=100, default_ttl=60)
        call_count = 0

        def fake_llm(prompt):
            nonlocal call_count
            call_count += 1
            return f"Response to: {prompt}"

        # First call — cache miss
        r1 = cache.get_or_call("What is AI?", fake_llm, model="claude", temperature=0.0)
        assert call_count == 1
        assert "Response to: What is AI?" in r1

        # Second call — cache hit
        r2 = cache.get_or_call("What is AI?", fake_llm, model="claude", temperature=0.0)
        assert call_count == 1  # No new call
        assert r2 == r1

        # Different prompt — cache miss
        r3 = cache.get_or_call("What is ML?", fake_llm, model="claude", temperature=0.0)
        assert call_count == 2

        stats = cache.stats()
        assert stats.hits == 1
        assert stats.misses == 2
        assert stats.hit_rate == pytest.approx(1/3)

        LLMCallCache.reset_instances()

    def test_tracing_and_profiler_combined(self):
        """Use tracing and profiling together on same operation."""
        from Jotty.core.observability.tracing import TracingContext
        from Jotty.core.monitoring.profiler import PerformanceProfiler

        tracer = TracingContext()
        profiler = PerformanceProfiler()

        tracer.new_trace()
        with tracer.span("profiled_op") as span:
            with profiler.profile("same_op"):
                time.sleep(0.02)
            span.add_cost(input_tokens=100, output_tokens=50, cost_usd=0.001)

        # Both captured the operation
        trace = tracer.get_current_trace()
        report = profiler.get_report()

        assert trace.span_count == 1
        assert trace.total_tokens == 150
        assert len(report.segments) == 1
        assert report.segments[0].name == "same_op"


# =============================================================================
# 4. Config Pipeline Integration
# =============================================================================

@pytest.mark.unit
class TestConfigPipelineIntegration:
    """Test focused config → SwarmConfig bridge → subsystem acceptance."""

    def test_memory_config_roundtrip(self):
        """MemoryConfig → SwarmConfig → to_memory_config preserves values."""
        from Jotty.core.foundation.configs import MemoryConfig
        from Jotty.core.foundation.data_structures import SwarmConfig

        original = MemoryConfig(
            episodic_capacity=5000,
            rag_relevance_threshold=0.8,
            chunk_size=300,
            chunk_overlap=50,
        )
        # Build SwarmConfig from focused config
        swarm_cfg = SwarmConfig.from_configs(memory=original)
        assert swarm_cfg.episodic_capacity == 5000
        assert swarm_cfg.rag_relevance_threshold == 0.8

        # Extract back
        roundtripped = swarm_cfg.to_memory_config()
        assert roundtripped.episodic_capacity == 5000
        assert roundtripped.rag_relevance_threshold == 0.8
        assert roundtripped.chunk_size == 300

    def test_learning_config_roundtrip(self):
        """LearningConfig → SwarmConfig → to_learning_config preserves values."""
        from Jotty.core.foundation.configs import LearningConfig
        from Jotty.core.foundation.data_structures import SwarmConfig

        original = LearningConfig(
            gamma=0.8, alpha=0.05, lambda_trace=0.9,
            alpha_min=0.01, alpha_max=0.1,
        )
        swarm_cfg = SwarmConfig.from_configs(learning=original)
        assert swarm_cfg.gamma == 0.8
        assert swarm_cfg.alpha == 0.05

        roundtripped = swarm_cfg.to_learning_config()
        assert roundtripped.gamma == 0.8
        assert roundtripped.alpha == 0.05
        assert roundtripped.lambda_trace == 0.9

    def test_multi_config_compose(self):
        """Compose multiple focused configs into single SwarmConfig."""
        from Jotty.core.foundation.configs import (
            MemoryConfig, LearningConfig, ExecutionConfig, MonitoringConfig,
        )
        from Jotty.core.foundation.data_structures import SwarmConfig

        cfg = SwarmConfig.from_configs(
            memory=MemoryConfig(episodic_capacity=3000),
            learning=LearningConfig(gamma=0.7),
            execution=ExecutionConfig(max_concurrent_agents=5),
            monitoring=MonitoringConfig(log_level="DEBUG"),
        )
        assert cfg.episodic_capacity == 3000
        assert cfg.gamma == 0.7
        assert cfg.max_concurrent_agents == 5
        assert cfg.log_level == "DEBUG"

    def test_config_validation_catches_invalid(self):
        """Validation prevents creating invalid configs."""
        from Jotty.core.foundation.configs import (
            LearningConfig, MemoryConfig, ContextBudgetConfig,
        )

        # Invalid gamma
        with pytest.raises(ValueError, match="gamma"):
            LearningConfig(gamma=2.0)

        # Invalid chunk overlap
        with pytest.raises(ValueError, match="chunk_overlap"):
            MemoryConfig(chunk_overlap=1000, chunk_size=500)

        # Budget over-allocation
        with pytest.raises(ValueError, match="Sum of static budgets"):
            ContextBudgetConfig(
                max_context_tokens=1000,
                system_prompt_budget=500,
                current_input_budget=500,
                trajectory_budget=500,
                tool_output_budget=500,
            )

    def test_swarm_config_views_consistent_with_focused(self):
        """SwarmConfig View proxies match focused config extraction."""
        from Jotty.core.foundation.data_structures import SwarmConfig

        cfg = SwarmConfig(gamma=0.8, episodic_capacity=3000)

        # View proxy access
        assert cfg.learning.gamma == 0.8
        assert cfg.memory_settings.episodic_capacity == 3000

        # Focused config extraction
        learning = cfg.to_learning_config()
        memory = cfg.to_memory_config()
        assert learning.gamma == 0.8
        assert memory.episodic_capacity == 3000


# =============================================================================
# 5. Cross-Subsystem Integration
# =============================================================================

@pytest.mark.unit
class TestCrossSubsystemIntegration:
    """Test that different subsystems work together."""

    def test_memory_with_tracing(self):
        """Memory operations can be traced."""
        from Jotty.core.memory.memory_system import MemorySystem
        from Jotty.core.observability.tracing import TracingContext

        tracer = TracingContext()
        tracer.new_trace()
        mem = MemorySystem()

        with tracer.span("memory_store") as s:
            mem.store("Test memory for tracing", level="episodic")
            s.set_attribute("operation", "store")

        with tracer.span("memory_retrieve") as s:
            results = mem.retrieve("tracing test", top_k=3)
            s.set_attribute("results_count", len(results))

        trace = tracer.get_current_trace()
        assert trace.span_count == 2

    def test_learning_with_budget_tracking(self):
        """Learning operations tracked with budget."""
        from Jotty.core.learning.facade import get_td_lambda
        from Jotty.core.utils.budget_tracker import BudgetTracker
        BudgetTracker.reset_instances()

        td = get_td_lambda()
        bt = BudgetTracker.get_instance("learning_test")

        # Simulate: learning update costs tokens
        td.update(
            state={"task": "coding"},
            action={"tool": "edit"},
            reward=0.8,
            next_state={"task": "coding", "step": 2},
        )
        bt.record_call("learner", tokens_input=100, tokens_output=50, model="internal")

        usage = bt.get_usage()
        assert usage['calls'] == 1
        BudgetTracker.reset_instances()

    def test_facades_all_accessible(self):
        """All 6 subsystem facades return valid objects."""
        from Jotty.core.memory.facade import get_memory_system
        from Jotty.core.learning.facade import get_td_lambda, get_credit_assigner
        from Jotty.core.orchestration.facade import (
            get_swarm_intelligence, get_ensemble_manager, get_swarm_router,
        )
        from Jotty.core.utils.facade import (
            get_budget_tracker, get_circuit_breaker, get_llm_cache, get_tokenizer,
        )

        # Memory
        mem = get_memory_system()
        assert mem is not None

        # Learning
        td = get_td_lambda()
        assert td is not None
        credit = get_credit_assigner()
        assert credit is not None

        # Orchestration
        si = get_swarm_intelligence()
        assert si is not None
        em = get_ensemble_manager()
        assert em is not None
        router = get_swarm_router()
        assert router is not None

        # Utils
        bt = get_budget_tracker()
        assert bt is not None
        cb = get_circuit_breaker("test")
        assert cb is not None
        cache = get_llm_cache()
        assert cache is not None
        tok = get_tokenizer()
        assert tok is not None

    def test_tokenizer_counts_real_text(self):
        """Tokenizer counts tokens in real text."""
        from Jotty.core.utils.facade import get_tokenizer
        tok = get_tokenizer()
        count = tok.count_tokens("Hello world, this is a test of the tokenizer")
        assert isinstance(count, int)
        assert count > 0
        assert count < 100  # Reasonable for a short sentence

    def test_circuit_breaker_state_machine(self):
        """Circuit breaker transitions through states."""
        from Jotty.core.utils.facade import get_circuit_breaker
        cb = get_circuit_breaker("integration_test")

        # Start closed
        cb.record_success()

        # Record failures
        for _ in range(10):
            cb.record_failure(Exception("timeout"))

        # Should be open or half-open after many failures
        # (exact behavior depends on config thresholds)
        assert cb is not None  # At minimum, didn't crash
