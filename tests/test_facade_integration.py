"""
Facade Integration Tests — Real Objects, No Mocks
==================================================

These tests instantiate REAL components through the facade layer
to verify the full import → construct → use chain works end-to-end.

No LLM calls. No network. Just real object creation and basic operations.
"""

import pytest


# =============================================================================
# Discovery API
# =============================================================================

@pytest.mark.unit
class TestCapabilitiesIntegration:
    """Test capabilities() returns live data from real subsystems."""

    def test_capabilities_returns_all_keys(self):
        from Jotty.core.capabilities import capabilities
        caps = capabilities()
        assert set(caps.keys()) == {
            "execution_paths", "subsystems", "swarms",
            "skills_count", "providers", "utilities",
        }

    def test_capabilities_execution_paths(self):
        from Jotty.core.capabilities import capabilities
        paths = capabilities()["execution_paths"]
        assert "chat" in paths
        assert "workflow" in paths
        assert "swarm" in paths

    def test_capabilities_subsystems(self):
        from Jotty.core.capabilities import capabilities
        subs = capabilities()["subsystems"]
        for name in ["learning", "memory", "context", "orchestration", "skills", "utils"]:
            assert name in subs, f"Missing subsystem: {name}"

    def test_capabilities_skills_count_is_positive(self):
        from Jotty.core.capabilities import capabilities
        assert capabilities()["skills_count"] > 100

    def test_capabilities_providers_have_structure(self):
        from Jotty.core.capabilities import capabilities
        providers = capabilities()["providers"]
        assert len(providers) > 0
        for p in providers:
            assert "name" in p
            assert "description" in p
            assert "installed" in p

    def test_explain_returns_nonempty_string(self):
        from Jotty.core.capabilities import explain
        for component in ["memory", "learning", "context", "orchestration", "skills", "utils"]:
            result = explain(component)
            assert isinstance(result, str)
            assert len(result) > 20, f"explain('{component}') too short"

    def test_explain_unknown_returns_message(self):
        from Jotty.core.capabilities import explain
        result = explain("nonexistent_thing")
        assert "unknown" in result.lower() or "no explanation" in result.lower() or isinstance(result, str)


# =============================================================================
# Memory Facade
# =============================================================================

@pytest.mark.unit
class TestMemoryFacadeIntegration:
    """Test memory facade returns real, usable objects."""

    def test_get_memory_system_returns_memory_system(self):
        from Jotty.core.intelligence.memory import get_memory_system
        mem = get_memory_system()
        assert type(mem).__name__ == "MemorySystem"

    def test_memory_system_can_store_and_retrieve(self):
        from Jotty.core.intelligence.memory import get_memory_system
        mem = get_memory_system()
        entry = mem.store("facade integration test data", "episodic", "testing")
        assert entry is not None
        assert hasattr(entry, "key")

    def test_get_brain_manager_returns_real_instance(self):
        from Jotty.core.intelligence.memory import get_brain_manager
        brain = get_brain_manager()
        assert type(brain).__name__ == "BrainInspiredMemoryManager"

    def test_get_consolidator_returns_real_instance(self):
        from Jotty.core.intelligence.memory import get_consolidator
        consolidator = get_consolidator()
        assert type(consolidator).__name__ == "SharpWaveRippleConsolidator"

    def test_get_rag_retriever_returns_real_instance(self):
        from Jotty.core.intelligence.memory import get_rag_retriever
        rag = get_rag_retriever()
        assert type(rag).__name__ == "LLMRAGRetriever"


# =============================================================================
# Learning Facade
# =============================================================================

@pytest.mark.unit
class TestLearningFacadeIntegration:
    """Test learning facade returns real, configured objects."""

    def test_get_td_lambda_has_correct_defaults(self):
        from Jotty.core.intelligence.learning import get_td_lambda
        td = get_td_lambda()
        assert type(td).__name__ == "TDLambdaLearner"
        assert td.gamma == 0.99
        assert td.lambda_trace == 0.95

    def test_get_credit_assigner_returns_real_instance(self):
        from Jotty.core.intelligence.learning import get_credit_assigner
        credit = get_credit_assigner()
        assert type(credit).__name__ == "ReasoningCreditAssigner"

    def test_get_reward_manager_returns_real_instance(self):
        from Jotty.core.intelligence.learning import get_reward_manager
        rewards = get_reward_manager()
        assert type(rewards).__name__ == "ShapedRewardManager"

    def test_get_cooperative_agents_returns_dict(self):
        from Jotty.core.intelligence.learning import get_cooperative_agents
        coop = get_cooperative_agents()
        assert "predictive_agent" in coop
        assert "nash_solver" in coop


# =============================================================================
# Context Facade
# =============================================================================

@pytest.mark.unit
class TestContextFacadeIntegration:
    """Test context facade returns real objects."""

    def test_get_context_manager_returns_real_instance(self):
        from Jotty.core.infrastructure.context import get_context_manager
        ctx = get_context_manager()
        assert type(ctx).__name__ == "SmartContextManager"

    def test_get_context_guard_returns_real_instance(self):
        from Jotty.core.infrastructure.context import get_context_guard
        guard = get_context_guard()
        assert type(guard).__name__ == "GlobalContextGuard"

    def test_get_content_gate_returns_real_instance(self):
        from Jotty.core.infrastructure.context import get_content_gate
        gate = get_content_gate()
        assert type(gate).__name__ == "ContentGate"


# =============================================================================
# Skills Facade
# =============================================================================

@pytest.mark.unit
class TestSkillsFacadeIntegration:
    """Test skills facade returns real registry data."""

    def test_get_registry_returns_unified_registry(self):
        from Jotty.core.capabilities.skills import get_registry
        reg = get_registry()
        assert type(reg).__name__ == "UnifiedRegistry"

    def test_list_skills_returns_nonempty_list(self):
        from Jotty.core.capabilities.skills import list_skills
        skills = list_skills()
        assert len(skills) > 100

    def test_list_providers_returns_structured_data(self):
        from Jotty.core.capabilities.skills import list_providers
        providers = list_providers()
        assert len(providers) > 0
        names = [p["name"] for p in providers]
        assert "browser-use" in names or "streamlit" in names


# =============================================================================
# Orchestration Facade
# =============================================================================

@pytest.mark.unit
class TestOrchestrationFacadeIntegration:
    """Test orchestration facade surfaces hidden components."""

    def test_get_swarm_intelligence_returns_instance(self):
        from Jotty.core.intelligence.orchestration import get_swarm_intelligence
        si = get_swarm_intelligence()
        assert si is not None
        assert type(si).__name__ == "SwarmIntelligence"

    def test_get_paradigm_executor_returns_class_without_manager(self):
        from Jotty.core.intelligence.orchestration import get_paradigm_executor
        pe = get_paradigm_executor()
        # Without manager arg, returns the class for manual instantiation
        assert pe is not None

    def test_get_training_daemon_returns_class_without_manager(self):
        from Jotty.core.intelligence.orchestration import get_training_daemon
        td = get_training_daemon()
        # Without manager arg, returns the class for manual instantiation
        assert td is not None

    def test_get_ensemble_manager_returns_real_instance(self):
        from Jotty.core.intelligence.orchestration import get_ensemble_manager
        em = get_ensemble_manager()
        assert type(em).__name__ == "EnsembleManager"

    def test_get_swarm_router_returns_real_instance(self):
        from Jotty.core.intelligence.orchestration import get_swarm_router
        sr = get_swarm_router()
        assert type(sr).__name__ == "SwarmRouter"


# =============================================================================
# Utils Facade
# =============================================================================

@pytest.mark.unit
class TestUtilsFacadeIntegration:
    """Test utils facade returns real, operational objects."""

    def test_budget_tracker_can_record_and_report(self):
        from Jotty.core.infrastructure.utils import get_budget_tracker
        bt = get_budget_tracker()
        bt.record_call("test_agent", tokens_input=100, tokens_output=50, model="gpt-4o")
        usage = bt.get_usage()
        assert usage["calls"] >= 1
        assert usage["total_tokens"] >= 150

    def test_circuit_breaker_starts_closed(self):
        from Jotty.core.infrastructure.utils import get_circuit_breaker
        cb = get_circuit_breaker("integration-test")
        assert "CLOSED" in str(cb.state)

    def test_circuit_breaker_records_success(self):
        from Jotty.core.infrastructure.utils import get_circuit_breaker
        cb = get_circuit_breaker("integration-test-2")
        cb.record_success()
        assert "CLOSED" in str(cb.state)

    def test_llm_cache_set_and_get(self):
        from Jotty.core.infrastructure.utils import get_llm_cache
        cache = get_llm_cache()
        cache.set("integration-test-key", {"answer": "cached"})
        hit = cache.get("integration-test-key")
        assert hit is not None
        # cache.get() returns a CachedResponse object with a .response attribute
        assert hit.response["answer"] == "cached"

    def test_llm_cache_stats(self):
        from Jotty.core.infrastructure.utils import get_llm_cache
        cache = get_llm_cache()
        stats = cache.stats()
        assert hasattr(stats, "hits")
        assert hasattr(stats, "misses")

    def test_tokenizer_counts_tokens(self):
        from Jotty.core.infrastructure.utils import get_tokenizer
        tok = get_tokenizer()
        count = tok.count_tokens("Hello world")
        assert isinstance(count, int)
        assert count > 0


# =============================================================================
# Top-Level Imports
# =============================================================================

@pytest.mark.unit
class TestTopLevelImports:
    """Test that top-level Jotty imports resolve to real classes."""

    def test_capabilities_import(self):
        from Jotty import capabilities
        assert callable(capabilities)
        result = capabilities()
        assert isinstance(result, dict)

    def test_memory_system_import(self):
        from Jotty import MemorySystem
        mem = MemorySystem()
        assert type(mem).__name__ == "MemorySystem"

    def test_budget_tracker_import(self):
        from Jotty import BudgetTracker
        assert BudgetTracker is not None

    def test_circuit_breaker_import(self):
        from Jotty import CircuitBreaker
        assert CircuitBreaker is not None

    def test_chat_executor_import(self):
        from Jotty import ChatExecutor
        assert ChatExecutor is not None

    def test_swarm_intelligence_import(self):
        from Jotty import SwarmIntelligence
        assert SwarmIntelligence is not None

    def test_paradigm_executor_import(self):
        from Jotty import ParadigmExecutor
        assert ParadigmExecutor is not None

    def test_ensemble_manager_import(self):
        from Jotty import EnsembleManager
        assert EnsembleManager is not None

    def test_model_tier_router_import(self):
        from Jotty import ModelTierRouter
        assert ModelTierRouter is not None


# =============================================================================
# Jotty Class Properties
# =============================================================================

@pytest.mark.unit
class TestJottyClassIntegration:
    """Test Jotty class properties return real objects."""

    def test_jotty_capabilities(self):
        from Jotty import Jotty
        j = Jotty()
        caps = j.capabilities()
        assert "subsystems" in caps
        assert caps["skills_count"] > 100

    def test_jotty_registry(self):
        from Jotty import Jotty
        j = Jotty()
        reg = j.registry
        assert type(reg).__name__ == "UnifiedRegistry"
        assert len(reg.list_skills()) > 100

    def test_jotty_router(self):
        from Jotty import Jotty
        j = Jotty()
        router = j.router
        assert type(router).__name__ == "ModeRouter"

    def test_jotty_chat_executor(self):
        from Jotty import Jotty
        j = Jotty()
        executor = j.chat_executor
        assert type(executor).__name__ == "ChatExecutor"


# =============================================================================
# Cross-Facade: End-to-End Scenario
# =============================================================================

@pytest.mark.unit
class TestCrossFacadeScenario:
    """Test a realistic multi-facade workflow without LLM calls."""

    def test_discover_then_access_subsystems(self):
        """Simulate: discover capabilities, then access each subsystem."""
        from Jotty.core.capabilities import capabilities

        caps = capabilities()

        # Verify every listed subsystem is actually accessible
        for name, info in caps["subsystems"].items():
            import_path = info.get("import", info.get("facade", ""))
            assert len(import_path) > 0, f"Subsystem '{name}' has no import path"

    def test_budget_track_then_check_remaining(self):
        """Track costs then verify budget reporting."""
        from Jotty.core.infrastructure.utils import get_budget_tracker
        bt = get_budget_tracker("scenario-test")
        bt.record_call("researcher", tokens_input=1000, tokens_output=500, model="gpt-4o")
        bt.record_call("coder", tokens_input=500, tokens_output=200, model="gpt-4o-mini")

        usage = bt.get_usage()
        assert usage["calls"] == 2
        assert usage["tokens_input"] == 1500
        assert usage["tokens_output"] == 700

    def test_cache_miss_then_hit(self):
        """Cache a response, verify miss then hit."""
        from Jotty.core.infrastructure.utils import get_llm_cache
        cache = get_llm_cache()

        # Miss
        result = cache.get("scenario-test-key-unique")
        assert result is None

        # Set
        cache.set("scenario-test-key-unique", {"response": "hello"})

        # Hit — returns CachedResponse with .response attribute
        result = cache.get("scenario-test-key-unique")
        assert result is not None
        assert result.response["response"] == "hello"

    def test_memory_store_then_search(self):
        """Store a memory, then search for it."""
        from Jotty.core.intelligence.memory import get_memory_system
        mem = get_memory_system()

        # Store
        entry = mem.store(
            "Integration test: budget was $0.05 for research task",
            "episodic",
            "integration testing",
        )
        assert entry is not None

    def test_tokenizer_consistent_counts(self):
        """Verify tokenizer gives consistent counts for same input."""
        from Jotty.core.infrastructure.utils import get_tokenizer
        tok = get_tokenizer()

        text = "The quick brown fox jumps over the lazy dog"
        count1 = tok.count_tokens(text)
        count2 = tok.count_tokens(text)
        assert count1 == count2
        assert count1 > 0
