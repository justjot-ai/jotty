"""
Modularity Tests — Import Boundaries, Focused Configs, Circular Deps
=====================================================================

Tests for the modularity improvements (Phases 1-5):
1. Import boundary linter runs cleanly
2. CoordinationPattern/MergeStrategy extracted to foundation
3. Focused config dataclasses match SwarmConfig fields
4. SwarmConfig bridge methods (to_*_config, from_configs)
5. Plugin skill discovery infrastructure
"""

import pytest
import subprocess
import sys
from pathlib import Path
from dataclasses import fields as dc_fields


# =============================================================================
# Phase 1: Import Boundary Linter
# =============================================================================

@pytest.mark.unit
class TestImportBoundaryLinter:
    """Verify the import boundary linter script."""

    def test_linter_script_exists(self):
        script = Path(__file__).resolve().parent.parent / "scripts" / "check_import_boundaries.py"
        assert script.exists(), f"Linter script not found at {script}"

    def test_linter_passes(self):
        script = Path(__file__).resolve().parent.parent / "scripts" / "check_import_boundaries.py"
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"Linter failed:\n{result.stdout}\n{result.stderr}"
        assert "PASSED" in result.stdout

    def test_linter_reports_import_count(self):
        script = Path(__file__).resolve().parent.parent / "scripts" / "check_import_boundaries.py"
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True, text=True, timeout=30,
        )
        assert "cross-module imports" in result.stdout


# =============================================================================
# Phase 2: Circular Dependency Fix — Execution Types in Foundation
# =============================================================================

@pytest.mark.unit
class TestExecutionTypesExtraction:
    """CoordinationPattern and MergeStrategy live in foundation, not swarms."""

    def test_import_from_foundation(self):
        from Jotty.core.infrastructure.foundation.types.execution_types import CoordinationPattern, MergeStrategy
        assert hasattr(CoordinationPattern, 'PIPELINE')
        assert hasattr(MergeStrategy, 'COMBINE')

    def test_foundation_types_reexport(self):
        from Jotty.core.infrastructure.foundation.types import CoordinationPattern, MergeStrategy
        assert CoordinationPattern.PARALLEL.value == "parallel"
        assert MergeStrategy.VOTE.value == "vote"

    def test_swarms_reexport_still_works(self):
        """Backward compat: importing from swarms still works."""
        from Jotty.core.intelligence.swarms.base.agent_team import CoordinationPattern, MergeStrategy
        assert CoordinationPattern.PIPELINE.value == "pipeline"
        assert MergeStrategy.FIRST.value == "first"

    def test_swarms_init_reexport(self):
        from Jotty.core.intelligence.swarms import CoordinationPattern, MergeStrategy
        assert len(CoordinationPattern) == 7
        assert len(MergeStrategy) == 5

    def test_composite_agent_uses_foundation(self):
        """composite_agent.py should import from foundation, not from swarms."""
        import inspect
        from Jotty.core.modes.agent.base import composite_agent
        source = inspect.getsource(composite_agent)
        assert "from Jotty.core.infrastructure.foundation.types.execution_types import" in source
        assert "from Jotty.core.intelligence.swarms.base.agent_team import CoordinationPattern" not in source

    def test_all_coordination_patterns(self):
        from Jotty.core.infrastructure.foundation.types.execution_types import CoordinationPattern
        expected = {'none', 'pipeline', 'parallel', 'consensus',
                    'hierarchical', 'blackboard', 'round_robin'}
        actual = {p.value for p in CoordinationPattern}
        assert actual == expected

    def test_all_merge_strategies(self):
        from Jotty.core.infrastructure.foundation.types.execution_types import MergeStrategy
        expected = {'combine', 'first', 'best', 'vote', 'concat'}
        actual = {s.value for s in MergeStrategy}
        assert actual == expected

    def test_identity_across_import_paths(self):
        """Same enum class regardless of import path."""
        from Jotty.core.infrastructure.foundation.types.execution_types import CoordinationPattern as CP1
        from Jotty.core.intelligence.swarms.base.agent_team import CoordinationPattern as CP2
        from Jotty.core.intelligence.swarms import CoordinationPattern as CP3
        assert CP1 is CP2 is CP3


# =============================================================================
# Phase 3: Focused Config Dataclasses
# =============================================================================

@pytest.mark.unit
class TestFocusedConfigs:
    """Standalone config dataclasses for each subsystem."""

    def test_all_configs_importable(self):
        from Jotty.core.infrastructure.foundation.configs import (
            PersistenceConfig, ExecutionConfig, MemoryConfig,
            ContextBudgetConfig, LearningConfig, ValidationConfig,
            MonitoringConfig, IntelligenceConfig,
        )
        # All should be instantiable with defaults
        for cls in [PersistenceConfig, ExecutionConfig, MemoryConfig,
                    ContextBudgetConfig, LearningConfig, ValidationConfig,
                    MonitoringConfig, IntelligenceConfig]:
            obj = cls()
            assert obj is not None, f"Failed to instantiate {cls.__name__}"

    def test_memory_config_defaults(self):
        from Jotty.core.infrastructure.foundation.configs import MemoryConfig
        cfg = MemoryConfig()
        assert cfg.episodic_capacity == 1000
        assert cfg.semantic_capacity == 500
        assert cfg.enable_llm_rag is True
        assert cfg.retrieval_mode == "synthesize"

    def test_learning_config_defaults(self):
        from Jotty.core.infrastructure.foundation.configs import LearningConfig
        cfg = LearningConfig()
        assert cfg.gamma == 0.99
        assert cfg.lambda_trace == 0.95
        assert cfg.alpha == 0.01
        assert cfg.enable_rl is True

    def test_execution_config_defaults(self):
        from Jotty.core.infrastructure.foundation.configs import ExecutionConfig
        cfg = ExecutionConfig()
        assert cfg.max_actor_iters == 50
        assert cfg.max_concurrent_agents == 10
        assert cfg.enable_deterministic is True

    def test_context_budget_config_defaults(self):
        from Jotty.core.infrastructure.foundation.configs import ContextBudgetConfig
        cfg = ContextBudgetConfig()
        assert cfg.max_context_tokens == 100000
        assert cfg.enable_dynamic_budget is True

    def test_validation_config_defaults(self):
        from Jotty.core.infrastructure.foundation.configs import ValidationConfig
        cfg = ValidationConfig()
        assert cfg.enable_validation is True
        assert cfg.max_validation_rounds == 3

    def test_monitoring_config_defaults(self):
        from Jotty.core.infrastructure.foundation.configs import MonitoringConfig
        cfg = MonitoringConfig()
        assert cfg.enable_debug_logging is False  # Off for production
        assert cfg.enable_metrics is True

    def test_intelligence_config_defaults(self):
        from Jotty.core.infrastructure.foundation.configs import IntelligenceConfig
        cfg = IntelligenceConfig()
        assert cfg.trust_min == 0.1
        assert cfg.local_mode is False

    def test_focused_config_customization(self):
        from Jotty.core.infrastructure.foundation.configs import MemoryConfig
        cfg = MemoryConfig(episodic_capacity=5000, enable_llm_rag=False)
        assert cfg.episodic_capacity == 5000
        assert cfg.enable_llm_rag is False


@pytest.mark.unit
class TestSwarmConfigBridge:
    """SwarmConfig to/from focused config conversion."""

    def test_to_memory_config(self):
        from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig
        cfg = SwarmConfig(episodic_capacity=2000)
        mem = cfg.to_memory_config()
        assert mem.episodic_capacity == 2000
        assert mem.enable_llm_rag is True

    def test_to_learning_config(self):
        from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig
        cfg = SwarmConfig(gamma=0.95)
        learn = cfg.to_learning_config()
        assert learn.gamma == 0.95
        assert learn.lambda_trace == 0.95

    def test_to_execution_config(self):
        from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig
        cfg = SwarmConfig(max_actor_iters=100)
        exe = cfg.to_execution_config()
        assert exe.max_actor_iters == 100

    def test_to_context_budget_config(self):
        from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig
        cfg = SwarmConfig(max_context_tokens=200000)
        ctx = cfg.to_context_budget_config()
        assert ctx.max_context_tokens == 200000

    def test_to_validation_config(self):
        from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig
        cfg = SwarmConfig(enable_validation=False)
        val = cfg.to_validation_config()
        assert val.enable_validation is False

    def test_to_monitoring_config(self):
        from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig
        cfg = SwarmConfig(verbose=2)
        mon = cfg.to_monitoring_config()
        assert mon.verbose == 2

    def test_to_intelligence_config(self):
        from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig
        cfg = SwarmConfig(local_mode=True)
        intel = cfg.to_intelligence_config()
        assert intel.local_mode is True

    def test_to_persistence_config(self):
        from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig
        cfg = SwarmConfig(storage_format="sqlite")
        pers = cfg.to_persistence_config()
        assert pers.storage_format == "sqlite"

    def test_from_configs_memory(self):
        from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig
        from Jotty.core.infrastructure.foundation.configs import MemoryConfig
        cfg = SwarmConfig.from_configs(memory=MemoryConfig(episodic_capacity=3000))
        assert cfg.episodic_capacity == 3000

    def test_from_configs_multiple(self):
        from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig
        from Jotty.core.infrastructure.foundation.configs import MemoryConfig, LearningConfig
        cfg = SwarmConfig.from_configs(
            memory=MemoryConfig(episodic_capacity=3000),
            learning=LearningConfig(gamma=0.5),
        )
        assert cfg.episodic_capacity == 3000
        assert cfg.gamma == 0.5

    def test_from_configs_with_overrides(self):
        from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig
        from Jotty.core.infrastructure.foundation.configs import MemoryConfig
        cfg = SwarmConfig.from_configs(
            memory=MemoryConfig(episodic_capacity=3000),
            schema_version="3.0",
        )
        assert cfg.episodic_capacity == 3000
        assert cfg.schema_version == "3.0"

    def test_from_configs_override_beats_subconfig(self):
        from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig
        from Jotty.core.infrastructure.foundation.configs import MemoryConfig
        cfg = SwarmConfig.from_configs(
            memory=MemoryConfig(episodic_capacity=3000),
            episodic_capacity=5000,  # Override beats sub-config
        )
        assert cfg.episodic_capacity == 5000

    def test_roundtrip_memory_config(self):
        """Extract -> modify -> compose back preserves other fields."""
        from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig
        original = SwarmConfig(gamma=0.5, episodic_capacity=2000)
        mem = original.to_memory_config()
        mem.episodic_capacity = 4000
        rebuilt = SwarmConfig.from_configs(memory=mem, gamma=0.5)
        assert rebuilt.episodic_capacity == 4000
        assert rebuilt.gamma == 0.5

    def test_flat_dict_unchanged(self):
        """to_flat_dict() still works after adding bridge methods."""
        from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig
        cfg = SwarmConfig()
        flat = cfg.to_flat_dict()
        assert 'gamma' in flat
        assert 'episodic_capacity' in flat
        assert 'schema_version' in flat

    def test_views_still_work(self):
        """View proxy access unchanged."""
        from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig
        cfg = SwarmConfig(gamma=0.5)
        assert cfg.learning.gamma == 0.5
        assert cfg.memory_settings.episodic_capacity == 1000

    def test_lazy_reexport_from_data_structures(self):
        """Focused configs importable from data_structures module."""
        from Jotty.core.infrastructure.foundation.data_structures import MemoryConfig
        assert MemoryConfig().episodic_capacity == 1000


# =============================================================================
# Phase 5: Plugin Skill Discovery
# =============================================================================

@pytest.mark.unit
class TestPluginSkillDiscovery:
    """Skill registry plugin discovery infrastructure."""

    def test_registry_has_scan_plugin_method(self):
        from Jotty.core.capabilities.registry.skills_registry import SkillsRegistry
        assert hasattr(SkillsRegistry, '_scan_plugin_skills')

    def test_registry_init_calls_plugin_scan(self):
        """Plugin scan is called during init (no plugins installed = no error)."""
        from Jotty.core.capabilities.registry.skills_registry import get_skills_registry
        reg = get_skills_registry()
        reg.initialized = False  # Force re-init
        reg.loaded_skills.clear()
        reg.init()
        assert reg.initialized
        assert len(reg.loaded_skills) > 100  # Built-in skills still load

    def test_plugin_scan_handles_no_plugins(self):
        """No installed plugins = no error."""
        from Jotty.core.capabilities.registry.skills_registry import SkillsRegistry
        reg = SkillsRegistry.__new__(SkillsRegistry)
        reg.loaded_skills = {}
        # Should not raise
        reg._scan_plugin_skills({'__pycache__'})
        # No plugins installed, so no new skills
        assert len(reg.loaded_skills) == 0


# =============================================================================
# Phase A: Memory Subsystem Accepts Focused Configs
# =============================================================================

@pytest.mark.unit
class TestMemoryFocusedConfigs:
    """Memory subsystem accepts MemoryConfig instead of SwarmConfig."""

    def test_memory_facade_accepts_memory_config(self):
        """get_rag_retriever accepts MemoryConfig."""
        from Jotty.core.infrastructure.foundation.configs import MemoryConfig
        from Jotty.core.intelligence.memory.facade import _resolve_memory_config
        cfg = MemoryConfig(episodic_capacity=2000, enable_llm_rag=False)
        resolved = _resolve_memory_config(cfg)
        # Should be a SwarmConfig with memory fields applied
        assert resolved.episodic_capacity == 2000
        assert resolved.enable_llm_rag is False

    def test_memory_facade_accepts_swarm_config_backward_compat(self):
        """get_rag_retriever still accepts SwarmConfig."""
        from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig
        from Jotty.core.intelligence.memory.facade import _resolve_memory_config
        cfg = SwarmConfig(episodic_capacity=3000)
        resolved = _resolve_memory_config(cfg)
        assert resolved is cfg  # Pass-through, not converted

    def test_memory_facade_accepts_none(self):
        """get_rag_retriever accepts None (defaults)."""
        from Jotty.core.intelligence.memory.facade import _resolve_memory_config
        resolved = _resolve_memory_config(None)
        assert resolved.episodic_capacity == 1000  # Default

    def test_cortex_accepts_memory_config(self):
        """SwarmMemory.__init__ accepts MemoryConfig via _ensure_swarm_config."""
        from Jotty.core.intelligence.memory.cortex import _ensure_swarm_config
        from Jotty.core.infrastructure.foundation.configs import MemoryConfig
        cfg = MemoryConfig(episodic_capacity=5000)
        resolved = _ensure_swarm_config(cfg)
        assert resolved.episodic_capacity == 5000
        assert hasattr(resolved, 'gamma')  # SwarmConfig field

    def test_cortex_accepts_swarm_config(self):
        """SwarmMemory.__init__ still accepts SwarmConfig."""
        from Jotty.core.intelligence.memory.cortex import _ensure_swarm_config
        from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig
        cfg = SwarmConfig()
        resolved = _ensure_swarm_config(cfg)
        assert resolved is cfg  # Pass-through

    def test_llm_rag_accepts_memory_config(self):
        """LLM RAG components accept MemoryConfig."""
        from Jotty.core.intelligence.memory.llm_rag import _ensure_swarm_config
        from Jotty.core.infrastructure.foundation.configs import MemoryConfig
        cfg = MemoryConfig(rag_window_size=10, chunk_size=1000)
        resolved = _ensure_swarm_config(cfg)
        assert resolved.rag_window_size == 10
        assert resolved.chunk_size == 1000


# =============================================================================
# Phase B: Learning Subsystem Accepts Focused Configs
# =============================================================================

@pytest.mark.unit
class TestLearningFocusedConfigs:
    """Learning subsystem accepts LearningConfig instead of SwarmConfig."""

    def test_learning_facade_accepts_learning_config(self):
        """Facade resolver converts LearningConfig to SwarmConfig."""
        from Jotty.core.infrastructure.foundation.configs import LearningConfig
        from Jotty.core.intelligence.learning.facade import _resolve_learning_config
        cfg = LearningConfig(gamma=0.5, lambda_trace=0.8)
        resolved = _resolve_learning_config(cfg)
        assert resolved.gamma == 0.5
        assert resolved.lambda_trace == 0.8

    def test_learning_facade_accepts_swarm_config_backward_compat(self):
        """Facade resolver still passes through SwarmConfig."""
        from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig
        from Jotty.core.intelligence.learning.facade import _resolve_learning_config
        cfg = SwarmConfig(gamma=0.7)
        resolved = _resolve_learning_config(cfg)
        assert resolved is cfg

    def test_learning_facade_accepts_none(self):
        """Facade resolver accepts None (defaults)."""
        from Jotty.core.intelligence.learning.facade import _resolve_learning_config
        resolved = _resolve_learning_config(None)
        assert resolved.gamma == 0.99  # Default

    def test_td_lambda_accepts_learning_config(self):
        """TDLambdaLearner accepts LearningConfig."""
        from Jotty.core.intelligence.learning.td_lambda import _ensure_swarm_config
        from Jotty.core.infrastructure.foundation.configs import LearningConfig
        cfg = LearningConfig(gamma=0.9, alpha=0.05)
        resolved = _ensure_swarm_config(cfg)
        assert resolved.gamma == 0.9
        assert resolved.alpha == 0.05

    def test_td_lambda_accepts_swarm_config(self):
        """TDLambdaLearner still accepts SwarmConfig."""
        from Jotty.core.intelligence.learning.td_lambda import _ensure_swarm_config
        from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig
        cfg = SwarmConfig()
        resolved = _ensure_swarm_config(cfg)
        assert resolved is cfg

    def test_reasoning_credit_accepts_learning_config(self):
        """ReasoningCreditAssigner accepts LearningConfig."""
        from Jotty.core.intelligence.learning.reasoning_credit import _ensure_swarm_config
        from Jotty.core.infrastructure.foundation.configs import LearningConfig
        cfg = LearningConfig(reasoning_weight=0.5, evidence_weight=0.3)
        resolved = _ensure_swarm_config(cfg)
        assert resolved.reasoning_weight == 0.5
        assert resolved.evidence_weight == 0.3

    def test_adaptive_components_accept_learning_config(self):
        """Adaptive components accept LearningConfig."""
        from Jotty.core.intelligence.learning.adaptive_components import _ensure_swarm_config
        from Jotty.core.infrastructure.foundation.configs import LearningConfig
        cfg = LearningConfig(alpha=0.02, alpha_min=0.005)
        resolved = _ensure_swarm_config(cfg)
        assert resolved.alpha == 0.02
        assert resolved.alpha_min == 0.005

    def test_offline_learning_accepts_learning_config(self):
        """Offline learning components accept LearningConfig."""
        from Jotty.core.intelligence.learning.offline_learning import _ensure_swarm_config
        from Jotty.core.infrastructure.foundation.configs import LearningConfig
        cfg = LearningConfig(episode_buffer_size=500)
        resolved = _ensure_swarm_config(cfg)
        assert resolved.episode_buffer_size == 500


# =============================================================================
# Phase C: Skill SDK Package
# =============================================================================

@pytest.mark.unit
class TestSkillSDK:
    """Skill SDK package provides standalone skill utilities."""

    def test_skill_sdk_importable(self):
        """core.skill_sdk is importable."""
        from Jotty.core.capabilities.sdk import (
            tool_helpers, env_loader, skill_status,
            api_client, async_utils, smart_fetcher,
        )
        assert tool_helpers is not None
        assert env_loader is not None
        assert skill_status is not None
        assert api_client is not None
        assert async_utils is not None
        assert smart_fetcher is not None

    def test_skill_sdk_tool_helpers_works(self):
        """tool_helpers from skill_sdk has expected functions."""
        from Jotty.core.capabilities.sdk.tool_helpers import (
            tool_response, tool_error, require_params,
        )
        resp = tool_response(data={"ok": True})
        assert resp["success"] is True
        err = tool_error("bad input")
        assert err["success"] is False

    def test_skill_sdk_skill_status_works(self):
        """SkillStatus from skill_sdk works."""
        from Jotty.core.capabilities.sdk import SkillStatus
        status = SkillStatus("test-skill")
        assert status.skill_name == "test-skill"

    def test_skill_sdk_env_loader_works(self):
        """env_loader from skill_sdk works."""
        from Jotty.core.capabilities.sdk import get_env
        # Should not raise (returns None if not set)
        result = get_env("NONEXISTENT_VAR_12345")
        assert result is None

    def test_skill_sdk_api_client_works(self):
        """BaseAPIClient from skill_sdk works."""
        from Jotty.core.capabilities.sdk.api_client import BaseAPIClient
        assert hasattr(BaseAPIClient, '_make_request')

    def test_utils_reexport_backward_compat(self):
        """Importing from core.utils still works (backward compat)."""
        from Jotty.core.infrastructure.utils.skill_status import SkillStatus
        from Jotty.core.infrastructure.utils.tool_helpers import tool_response
        from Jotty.core.infrastructure.utils.env_loader import get_env
        assert SkillStatus is not None
        assert tool_response is not None
        assert get_env is not None

    def test_skill_sdk_smart_fetcher(self):
        """smart_fetcher from skill_sdk is accessible."""
        from Jotty.core.capabilities.sdk.smart_fetcher import smart_fetch, FetchResult
        assert callable(smart_fetch)
        assert FetchResult is not None


# =============================================================================
# Phase D: Tightened Boundary Linter
# =============================================================================

@pytest.mark.unit
class TestTightenedBoundaryLinter:
    """Boundary linter with explicit orchestration deps."""

    def test_linter_passes(self):
        """Linter passes with tighter rules."""
        script = Path(__file__).resolve().parent.parent / "scripts" / "check_import_boundaries.py"
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"Linter failed:\n{result.stdout}\n{result.stderr}"
        assert "PASSED" in result.stdout

    def test_orchestration_no_wildcard(self):
        """Orchestration no longer uses wildcard deps."""
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        try:
            import check_import_boundaries as linter
            deps = linter.ALLOWED_DEPS["orchestration"]
            assert "*" not in deps, "Orchestration should not use wildcard deps"
            # Should have explicit deps
            assert "foundation" in deps
            assert "agents" in deps
            assert "memory" in deps
        finally:
            sys.path.pop(0)

    def test_skill_sdk_in_allowed_deps(self):
        """skill_sdk module is tracked in boundary linter."""
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        try:
            import check_import_boundaries as linter
            assert "skill_sdk" in linter.ALLOWED_DEPS
            deps = linter.ALLOWED_DEPS["skill_sdk"]
            assert "foundation" in deps
        finally:
            sys.path.pop(0)

    def test_linter_rejects_bad_import(self):
        """Linter detects violations programmatically."""
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        try:
            import check_import_boundaries as linter
            # Create a fake violation
            fake_import = linter.ImportInfo(
                source_file="test.py", line=1,
                from_module="memory", to_module="agents",
                is_deferred=False,
            )
            top, _ = linter.check_violations([fake_import])
            assert len(top) == 1, "Should detect memory -> agents as violation"
        finally:
            sys.path.pop(0)


# =============================================================================
# Phase E: Orchestration Sub-Boundaries
# =============================================================================

@pytest.mark.unit
class TestOrchestrationSubBoundaries:
    """Orchestration internal sub-module boundaries."""

    def test_orchestration_has_sub_module_docs(self):
        """orchestration/__init__.py documents sub-module structure."""
        import inspect
        from Jotty.core import orchestration
        doc = inspect.getmodule(orchestration).__doc__
        assert "Sub-module Structure" in doc
        assert "llm_providers" in doc
        assert "intelligence" in doc
        assert "routing" in doc

    def test_internal_boundaries_defined(self):
        """Internal boundary rules exist in linter."""
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        try:
            import check_import_boundaries as linter
            assert hasattr(linter, 'ORCHESTRATION_SUB_MODULES')
            assert hasattr(linter, 'INTERNAL_ALLOWED_DEPS')
            assert "llm_providers" in linter.ORCHESTRATION_SUB_MODULES
            assert "intelligence" in linter.ORCHESTRATION_SUB_MODULES
        finally:
            sys.path.pop(0)

    def test_internal_boundaries_pass(self):
        """Internal boundary check passes."""
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        try:
            import check_import_boundaries as linter
            core_root = str(Path(__file__).resolve().parent.parent / "core")
            violations = linter.check_internal_boundaries(core_root)
            assert len(violations) == 0, f"Internal violations: {violations}"
        finally:
            sys.path.pop(0)

    def test_llm_providers_is_leaf(self):
        """llm_providers has no intra-orchestration deps."""
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        try:
            import check_import_boundaries as linter
            deps = linter.INTERNAL_ALLOWED_DEPS.get("llm_providers", set())
            assert len(deps) == 0, "llm_providers should be a leaf"
        finally:
            sys.path.pop(0)


# =============================================================================
# Part C: Thread Safety for Facade Singletons
# =============================================================================

@pytest.mark.unit
class TestFacadeThreadSafety:
    """Verify facade files have thread-safe singleton patterns."""

    def test_memory_facade_has_lock(self):
        """memory/facade.py uses threading.Lock for singletons."""
        import Jotty.core.intelligence.memory.facade as mf
        assert hasattr(mf, '_lock'), "memory facade missing _lock"
        assert hasattr(mf, '_singletons'), "memory facade missing _singletons"
        import threading
        assert isinstance(mf._lock, type(threading.Lock()))

    def test_orchestration_facade_has_lock(self):
        """orchestration/facade.py uses threading.Lock for singletons."""
        import Jotty.core.intelligence.orchestration.facade as of
        assert hasattr(of, '_lock'), "orchestration facade missing _lock"
        assert hasattr(of, '_singletons'), "orchestration facade missing _singletons"
        import threading
        assert isinstance(of._lock, type(threading.Lock()))

    def test_utils_facade_has_lock(self):
        """utils/facade.py uses threading.Lock for singletons."""
        import Jotty.core.infrastructure.utils.facade as uf
        assert hasattr(uf, '_lock'), "utils facade missing _lock"
        assert hasattr(uf, '_singletons'), "utils facade missing _singletons"
        import threading
        assert isinstance(uf._lock, type(threading.Lock()))

    def test_memory_facade_returns_same_instance(self):
        """get_memory_system() returns same singleton across calls."""
        import Jotty.core.intelligence.memory.facade as mf
        mf._singletons.clear()
        ms1 = mf.get_memory_system()
        ms2 = mf.get_memory_system()
        assert ms1 is ms2

    def test_orchestration_facade_returns_same_instance(self):
        """get_ensemble_manager() returns same singleton across calls."""
        import Jotty.core.intelligence.orchestration.facade as of
        of._singletons.clear()
        em1 = of.get_ensemble_manager()
        em2 = of.get_ensemble_manager()
        assert em1 is em2

    def test_orchestration_facade_config_bypass(self):
        """get_swarm_intelligence(config=...) bypasses cache."""
        import Jotty.core.intelligence.orchestration.facade as of
        of._singletons.clear()
        from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig
        si1 = of.get_swarm_intelligence()
        si2 = of.get_swarm_intelligence(config=SwarmConfig())
        assert si1 is not si2  # Config-parameterized call returns fresh instance

    def test_budget_tracker_thread_safe_singleton(self):
        """BudgetTracker.get_instance() has class-level lock."""
        from Jotty.core.infrastructure.utils.budget_tracker import BudgetTracker
        import threading
        assert hasattr(BudgetTracker, '_instances_lock')
        assert isinstance(BudgetTracker._instances_lock, type(threading.Lock()))

    def test_llm_cache_thread_safe_singleton(self):
        """LLMCallCache.get_instance() has class-level lock."""
        from Jotty.core.infrastructure.utils.llm_cache import LLMCallCache
        import threading
        assert hasattr(LLMCallCache, '_instances_lock')
        assert isinstance(LLMCallCache._instances_lock, type(threading.Lock()))

    def test_concurrent_memory_facade_access(self):
        """Multiple threads getting memory system don't race."""
        import Jotty.core.intelligence.memory.facade as mf
        import threading
        mf._singletons.clear()
        results = []
        errors = []

        def _get():
            try:
                results.append(mf.get_memory_system())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_get) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0, f"Errors during concurrent access: {errors}"
        assert len(results) == 8
        # All threads got the same singleton
        assert all(r is results[0] for r in results)

    def test_concurrent_budget_tracker_access(self):
        """Multiple threads getting budget tracker don't race."""
        from Jotty.core.infrastructure.utils.budget_tracker import BudgetTracker
        import threading
        BudgetTracker.reset_instances()
        results = []
        errors = []

        def _get():
            try:
                results.append(BudgetTracker.get_instance("concurrent_test"))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_get) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0, f"Errors during concurrent access: {errors}"
        assert len(results) == 8
        assert all(r is results[0] for r in results)
        BudgetTracker.reset_instances()


# =============================================================================
# Config Validation (__post_init__)
# =============================================================================

@pytest.mark.unit
class TestConfigValidation:
    """Validate __post_init__ on all focused config dataclasses."""

    # --- LearningConfig ---

    def test_learning_config_defaults_valid(self):
        from Jotty.core.infrastructure.foundation.configs import LearningConfig
        cfg = LearningConfig()  # Should not raise
        assert cfg.gamma == 0.99

    def test_learning_config_gamma_out_of_range(self):
        from Jotty.core.infrastructure.foundation.configs import LearningConfig
        with pytest.raises(ValueError, match="gamma"):
            LearningConfig(gamma=1.5)

    def test_learning_config_gamma_negative(self):
        from Jotty.core.infrastructure.foundation.configs import LearningConfig
        with pytest.raises(ValueError, match="gamma"):
            LearningConfig(gamma=-0.1)

    def test_learning_config_alpha_min_gt_max(self):
        from Jotty.core.infrastructure.foundation.configs import LearningConfig
        with pytest.raises(ValueError, match="alpha_min"):
            LearningConfig(alpha_min=0.5, alpha_max=0.1)

    def test_learning_config_epsilon_end_gt_start(self):
        from Jotty.core.infrastructure.foundation.configs import LearningConfig
        with pytest.raises(ValueError, match="epsilon_end"):
            LearningConfig(epsilon_end=0.5, epsilon_start=0.1)

    def test_learning_config_replay_gt_buffer(self):
        from Jotty.core.infrastructure.foundation.configs import LearningConfig
        with pytest.raises(ValueError, match="replay_batch_size"):
            LearningConfig(replay_batch_size=2000, episode_buffer_size=100)

    def test_learning_config_negative_q_table(self):
        from Jotty.core.infrastructure.foundation.configs import LearningConfig
        with pytest.raises(ValueError, match="max_q_table_size"):
            LearningConfig(max_q_table_size=0)

    def test_learning_config_valid_custom(self):
        from Jotty.core.infrastructure.foundation.configs import LearningConfig
        cfg = LearningConfig(
            gamma=0.5, alpha=0.05, alpha_min=0.01, alpha_max=0.1,
            epsilon_start=0.5, epsilon_end=0.01,
            replay_batch_size=10, episode_buffer_size=100,
        )
        assert cfg.gamma == 0.5

    # --- MemoryConfig ---

    def test_memory_config_defaults_valid(self):
        from Jotty.core.infrastructure.foundation.configs import MemoryConfig
        cfg = MemoryConfig()  # Should not raise
        assert cfg.episodic_capacity == 1000

    def test_memory_config_zero_capacity(self):
        from Jotty.core.infrastructure.foundation.configs import MemoryConfig
        # Zero capacity is allowed (means disabled); negative is not
        cfg = MemoryConfig(episodic_capacity=0)
        assert cfg.episodic_capacity == 0
        with pytest.raises(ValueError, match="episodic_capacity"):
            MemoryConfig(episodic_capacity=-1)

    def test_memory_config_bad_threshold(self):
        from Jotty.core.infrastructure.foundation.configs import MemoryConfig
        with pytest.raises(ValueError, match="rag_relevance_threshold"):
            MemoryConfig(rag_relevance_threshold=1.5)

    def test_memory_config_chunk_overlap_gte_size(self):
        from Jotty.core.infrastructure.foundation.configs import MemoryConfig
        with pytest.raises(ValueError, match="chunk_overlap"):
            MemoryConfig(chunk_overlap=500, chunk_size=500)

    def test_memory_config_bad_retrieval_mode(self):
        from Jotty.core.infrastructure.foundation.configs import MemoryConfig
        with pytest.raises(ValueError, match="retrieval_mode"):
            MemoryConfig(retrieval_mode="invalid")

    # --- ContextBudgetConfig ---

    def test_context_budget_defaults_valid(self):
        from Jotty.core.infrastructure.foundation.configs import ContextBudgetConfig
        cfg = ContextBudgetConfig()  # Should not raise
        assert cfg.max_context_tokens == 100000

    def test_context_budget_min_gt_max_memory(self):
        from Jotty.core.infrastructure.foundation.configs import ContextBudgetConfig
        with pytest.raises(ValueError, match="min_memory_budget"):
            ContextBudgetConfig(min_memory_budget=70000, max_memory_budget=50000)

    def test_context_budget_sum_exceeds_max(self):
        from Jotty.core.infrastructure.foundation.configs import ContextBudgetConfig
        # Sum exceeding max now produces a warning instead of a ValueError,
        # because dynamic budgeting clamps memory_budget at runtime.
        cfg = ContextBudgetConfig(
            max_context_tokens=10000,
            system_prompt_budget=5000,
            current_input_budget=5000,
            trajectory_budget=5000,
            tool_output_budget=5000,
        )
        assert cfg.max_context_tokens == 10000

    def test_context_budget_zero_budget(self):
        from Jotty.core.infrastructure.foundation.configs import ContextBudgetConfig
        with pytest.raises(ValueError, match="system_prompt_budget"):
            ContextBudgetConfig(system_prompt_budget=0)

    # --- ExecutionConfig ---

    def test_execution_config_defaults_valid(self):
        from Jotty.core.infrastructure.foundation.configs import ExecutionConfig
        cfg = ExecutionConfig()
        assert cfg.max_actor_iters == 50

    def test_execution_config_zero_iters(self):
        from Jotty.core.infrastructure.foundation.configs import ExecutionConfig
        # Zero is allowed (means unlimited/disabled); negative is not
        cfg = ExecutionConfig(max_actor_iters=0)
        assert cfg.max_actor_iters == 0
        with pytest.raises(ValueError, match="max_actor_iters"):
            ExecutionConfig(max_actor_iters=-1)

    def test_execution_config_negative_timeout(self):
        from Jotty.core.infrastructure.foundation.configs import ExecutionConfig
        with pytest.raises(ValueError, match="async_timeout"):
            ExecutionConfig(async_timeout=-1.0)

    # --- PersistenceConfig ---

    def test_persistence_config_defaults_valid(self):
        from Jotty.core.infrastructure.foundation.configs import PersistenceConfig
        cfg = PersistenceConfig()
        assert cfg.storage_format == "json"

    def test_persistence_config_bad_format(self):
        from Jotty.core.infrastructure.foundation.configs import PersistenceConfig
        with pytest.raises(ValueError, match="storage_format"):
            PersistenceConfig(storage_format="xml")

    def test_persistence_config_zero_interval(self):
        from Jotty.core.infrastructure.foundation.configs import PersistenceConfig
        with pytest.raises(ValueError, match="auto_save_interval"):
            PersistenceConfig(auto_save_interval=0)

    # --- ValidationConfig ---

    def test_validation_config_defaults_valid(self):
        from Jotty.core.infrastructure.foundation.configs import ValidationConfig
        cfg = ValidationConfig()
        assert cfg.enable_validation is True

    def test_validation_config_bad_confidence(self):
        from Jotty.core.infrastructure.foundation.configs import ValidationConfig
        with pytest.raises(ValueError, match="min_confidence"):
            ValidationConfig(min_confidence=2.0)

    def test_validation_config_bad_mode(self):
        from Jotty.core.infrastructure.foundation.configs import ValidationConfig
        with pytest.raises(ValueError, match="validation_mode"):
            ValidationConfig(validation_mode="turbo")

    def test_validation_config_negative_timeout(self):
        from Jotty.core.infrastructure.foundation.configs import ValidationConfig
        with pytest.raises(ValueError, match="refinement_timeout"):
            ValidationConfig(refinement_timeout=-5.0)

    # --- MonitoringConfig ---

    def test_monitoring_config_defaults_valid(self):
        from Jotty.core.infrastructure.foundation.configs import MonitoringConfig
        cfg = MonitoringConfig()
        assert cfg.log_level == "INFO"

    def test_monitoring_config_bad_log_level(self):
        from Jotty.core.infrastructure.foundation.configs import MonitoringConfig
        with pytest.raises(ValueError, match="log_level"):
            MonitoringConfig(log_level="VERBOSE")

    def test_monitoring_config_bad_threshold(self):
        from Jotty.core.infrastructure.foundation.configs import MonitoringConfig
        with pytest.raises(ValueError, match="budget_warning_threshold"):
            MonitoringConfig(budget_warning_threshold=1.5)

    def test_monitoring_config_negative_verbose(self):
        from Jotty.core.infrastructure.foundation.configs import MonitoringConfig
        with pytest.raises(ValueError, match="verbose"):
            MonitoringConfig(verbose=-1)

    def test_monitoring_config_bad_baseline_cost(self):
        from Jotty.core.infrastructure.foundation.configs import MonitoringConfig
        with pytest.raises(ValueError, match="baseline_cost_per_success"):
            MonitoringConfig(baseline_cost_per_success=-0.5)

    # --- IntelligenceConfig ---

    def test_intelligence_config_defaults_valid(self):
        from Jotty.core.infrastructure.foundation.configs import IntelligenceConfig
        cfg = IntelligenceConfig()
        assert cfg.trust_min == 0.1

    def test_intelligence_config_bad_trust(self):
        from Jotty.core.infrastructure.foundation.configs import IntelligenceConfig
        with pytest.raises(ValueError, match="trust_min"):
            IntelligenceConfig(trust_min=1.5)

    def test_intelligence_config_zero_budget(self):
        from Jotty.core.infrastructure.foundation.configs import IntelligenceConfig
        with pytest.raises(ValueError, match="memory_retrieval_budget"):
            IntelligenceConfig(memory_retrieval_budget=0)


# =============================================================================
# Phase 9.5-1: SwarmConfig Validation Delegation
# =============================================================================

@pytest.mark.unit
class TestSwarmConfigValidationDelegation:
    """SwarmConfig.__post_init__ delegates to focused config validation."""

    def test_swarm_config_default_passes_validation(self):
        """Default SwarmConfig() must pass all focused-config validation."""
        from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig
        cfg = SwarmConfig()  # Should not raise
        assert cfg.gamma == 0.99

    def test_swarm_config_validates_via_focused_configs(self):
        """Bad values caught by focused-config __post_init__."""
        from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig
        with pytest.raises(ValueError, match="SwarmConfig validation failed"):
            SwarmConfig(gamma=1.5)

    def test_swarm_config_catches_multiple_errors(self):
        """Multiple invalid fields produce multiple error lines."""
        from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig
        with pytest.raises(ValueError) as exc_info:
            SwarmConfig(gamma=1.5, episodic_capacity=-1, max_actor_iters=-1)
        msg = str(exc_info.value)
        assert "to_learning_config" in msg
        assert "to_memory_config" in msg
        assert "to_execution_config" in msg

    def test_swarm_config_single_subsystem_error(self):
        """Error from one subsystem only mentions that subsystem."""
        from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig
        with pytest.raises(ValueError, match="to_persistence_config"):
            SwarmConfig(storage_format="xml")

    def test_validate_method_exists(self):
        """SwarmConfig has _validate_via_focused_configs method."""
        from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig
        assert hasattr(SwarmConfig, '_validate_via_focused_configs')


# =============================================================================
# Phase 9.5-2a: Trigger Discovery
# =============================================================================

@pytest.mark.unit
class TestTriggerDiscovery:
    """Skills trigger parsing and discovery scoring."""

    def test_triggers_parsed_from_skill_md(self):
        """Skills with ## Triggers section have triggers in metadata."""
        from Jotty.core.capabilities.registry.skills_registry import SkillsRegistry
        reg = SkillsRegistry()
        reg.init()
        ws = reg.get_skill('web-search')
        triggers = ws.metadata.get('triggers', [])
        assert len(triggers) > 0
        assert any('search' in t for t in triggers)

    def test_triggers_boost_discovery_score(self):
        """Skills with matching triggers score higher in discover()."""
        from Jotty.core.capabilities.registry.skills_registry import SkillsRegistry
        reg = SkillsRegistry()
        reg.init()
        results = reg.discover('search for AI trends')
        # web-search should be in top results due to trigger match
        names = [r['name'] for r in results[:10]]
        assert 'web-search' in names

    def test_category_parsed_from_skill_md(self):
        """Skills with ## Category section have category set."""
        from Jotty.core.capabilities.registry.skills_registry import SkillsRegistry
        reg = SkillsRegistry()
        reg.init()
        ws = reg.get_skill('web-search')
        assert ws.category != "general"  # Should be workflow-automation


# =============================================================================
# Phase 9.5-2b: @tool_wrapper Migration Verification
# =============================================================================

@pytest.mark.unit
class TestToolWrapperMigration:
    """Verify ALL skills use @tool_wrapper."""

    def test_all_skills_have_tool_wrapper(self):
        """Every skill tools.py imports tool_wrapper."""
        skills_dir = Path(__file__).resolve().parent.parent / "skills"
        missing = []
        for skill_dir in sorted(skills_dir.iterdir()):
            tools_py = skill_dir / "tools.py"
            if not tools_py.exists():
                continue
            content = tools_py.read_text()
            if "tool_wrapper" not in content:
                missing.append(skill_dir.name)
        assert not missing, (
            f"{len(missing)} skills missing tool_wrapper: {missing[:10]}"
        )

    def test_all_skills_parse_correctly(self):
        """Every skill tools.py parses without syntax errors."""
        import ast
        skills_dir = Path(__file__).resolve().parent.parent / "skills"
        errors = []
        for skill_dir in sorted(skills_dir.iterdir()):
            tools_py = skill_dir / "tools.py"
            if not tools_py.exists():
                continue
            content = tools_py.read_text()
            try:
                ast.parse(content)
            except SyntaxError as e:
                errors.append(f"{skill_dir.name}: {e}")
        assert not errors, f"Syntax errors in: {errors}"

    def test_all_tool_functions_decorated(self):
        """Every public tool function (params arg) has @*_wrapper decorator."""
        import ast
        skills_dir = Path(__file__).resolve().parent.parent / "skills"
        undecorated = []
        for skill_dir in sorted(skills_dir.iterdir()):
            tools_py = skill_dir / "tools.py"
            if not tools_py.exists():
                continue
            try:
                tree = ast.parse(tools_py.read_text())
            except SyntaxError:
                continue
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name.startswith('_'):
                        continue
                    if node.args.args and node.args.args[0].arg == 'params':
                        has_wrapper = any(
                            (isinstance(d, ast.Call) and isinstance(d.func, ast.Name)
                             and 'wrapper' in d.func.id)
                            or (isinstance(d, ast.Name) and 'wrapper' in d.id)
                            for d in node.decorator_list
                        )
                        if not has_wrapper:
                            undecorated.append(
                                f"{skill_dir.name}:{node.name}"
                            )
        assert not undecorated, (
            f"{len(undecorated)} undecorated tool functions: {undecorated[:10]}"
        )

    def test_all_skills_have_triggers(self):
        """Every SKILL.md has a ## Triggers section."""
        skills_dir = Path(__file__).resolve().parent.parent / "skills"
        missing = []
        for skill_dir in sorted(skills_dir.iterdir()):
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue
            content = skill_md.read_text()
            if "## Triggers" not in content:
                missing.append(skill_dir.name)
        assert not missing, (
            f"{len(missing)} SKILL.md files missing triggers: {missing[:10]}"
        )


# =============================================================================
# Phase 9.5-2c: Parameter Schema Parsing
# =============================================================================

@pytest.mark.unit
class TestParameterSchemaParsing:
    """Verify parameter schemas parsed from SKILL.md."""

    def test_web_search_has_tool_schemas(self):
        """web-search skill has parsed tool parameter schemas."""
        from Jotty.core.capabilities.registry.skills_registry import SkillsRegistry
        reg = SkillsRegistry()
        reg.init()
        ws = reg.get_skill('web-search')
        assert len(ws._tool_metadata) > 0
        # search_web_tool should have query parameter
        search_meta = ws._tool_metadata.get('search_web_tool')
        assert search_meta is not None
        props = search_meta.parameters.get('properties', {})
        assert 'query' in props

    def test_schema_has_required_params(self):
        """Parsed schemas include required parameter list."""
        from Jotty.core.capabilities.registry.skills_registry import SkillsRegistry
        reg = SkillsRegistry()
        reg.init()
        ws = reg.get_skill('web-search')
        search_meta = ws._tool_metadata.get('search_web_tool')
        required = search_meta.parameters.get('required', [])
        assert 'query' in required

    def test_claude_tool_format_output(self):
        """to_claude_tools() produces valid Claude API tool format."""
        from Jotty.core.capabilities.registry.skills_registry import SkillsRegistry
        reg = SkillsRegistry()
        reg.init()
        ws = reg.get_skill('web-search')
        tools = ws.to_claude_tools()
        assert len(tools) > 0
        for tool in tools:
            assert 'name' in tool
            assert 'description' in tool
            assert 'input_schema' in tool
            schema = tool['input_schema']
            assert schema['type'] == 'object'


# =============================================================================
# Phase 9.5-3: Facade Return Type Annotations
# =============================================================================

@pytest.mark.unit
class TestFacadeReturnAnnotations:
    """Verify all facade public functions have return type annotations."""

    def test_all_facades_have_return_annotations(self):
        """Every public get_* and list_* function has a return annotation."""
        import inspect
        from Jotty.core.intelligence.memory import facade as mem_f
        from Jotty.core.intelligence.learning import facade as learn_f
        from Jotty.core.infrastructure.context import facade as ctx_f
        from Jotty.core.capabilities.skills import facade as skills_f
        from Jotty.core.intelligence.orchestration import facade as orch_f
        from Jotty.core.infrastructure.utils import facade as utils_f

        missing = []
        for mod in [mem_f, learn_f, ctx_f, skills_f, orch_f, utils_f]:
            for name in dir(mod):
                if name.startswith('_'):
                    continue
                obj = getattr(mod, name)
                if not callable(obj) or not (name.startswith('get_') or name.startswith('list_')):
                    continue
                sig = inspect.signature(obj)
                if sig.return_annotation is inspect.Parameter.empty:
                    missing.append(f"{mod.__name__}.{name}")

        assert len(missing) == 0, f"Missing return annotations: {missing}"


# =============================================================================
# TaskClassifier — Smart Swarm Routing (110+ scenarios)
# =============================================================================

# Shared classifier instance (avoid re-creating per test for speed)
_tc_instance = None

def _get_tc():
    global _tc_instance
    if _tc_instance is None:
        from Jotty.core.modes.execution.intent_classifier import TaskClassifier
        _tc_instance = TaskClassifier()
    return _tc_instance


# --- Coding swarm (15 scenarios) ---
_CODING_SCENARIOS = [
    ("Write a Python REST API", "coding"),
    ("Implement a binary search tree in Java", "coding"),
    ("Debug this segfault in my C++ program", "coding"),
    ("Refactor the authentication module", "coding"),
    ("Create a React component for user profiles", "coding"),
    ("Fix the null pointer exception in UserService.java", "coding"),
    ("Build a microservice with FastAPI", "coding"),
    ("Optimize this SQL query for performance", "coding"),
    ("Add error handling to the file upload endpoint", "coding"),
    ("Convert this JavaScript to TypeScript", "coding"),
    ("Implement OAuth2 login flow", "coding"),
    ("Write a CLI tool in Go that parses JSON logs", "coding"),
    ("Create a Python script to rename files in bulk", "coding"),
    ("Build a REST API backend with authentication", "coding"),
    ("Write a recursive algorithm to solve Tower of Hanoi", "coding"),
]

# --- Research swarm (10 scenarios) ---
_RESEARCH_SCENARIOS = [
    ("Research AI trends 2024", "research"),
    ("Investigate the impact of climate change on agriculture", "research"),
    ("Research competitor pricing strategies", "research"),
    ("Investigate why our conversion rate dropped last quarter", "research"),
    ("Do a deep dive into quantum computing applications", "research"),
    ("Research the regulatory landscape for fintech in India", "research"),
    ("Investigate supply chain disruptions in semiconductor industry", "research"),
    ("Research the history of neural network architectures", "research"),
    ("Research emerging technologies in renewable energy", "research"),
    ("Investigate the root cause of the production outage", "research"),
]

# --- Testing swarm (10 scenarios) ---
_TESTING_SCENARIOS = [
    ("Write unit tests for the payment gateway", "testing"),
    ("Write integration tests for the checkout flow", "testing"),
    ("Increase test coverage to 90%", "testing"),
    ("Run pytest on the authentication module", "testing"),
    ("Write regression tests for the billing bug fix", "testing"),
    ("Set up end-to-end testing with Cypress", "testing"),
    ("Create load tests with Locust for the API", "testing"),
    ("Write unit tests for the data validation layer", "testing"),
    ("Add integration test for the payment webhook", "testing"),
    ("Benchmark the database query performance", "testing"),
]

# --- Review swarm (8 scenarios) ---
_REVIEW_SCENARIOS = [
    ("Review the pull request for feature/auth-redesign", "review"),
    ("Audit the codebase for security vulnerabilities", "review"),
    ("Do a code review of the new caching layer", "review"),
    ("Review this PR and check for memory leaks", "review"),
    ("Audit our API for OWASP top 10 vulnerabilities", "review"),
    ("Review the database migration scripts", "review"),
    ("Check code quality of the reporting module", "review"),
    ("Peer review the machine learning pipeline code", "review"),
]

# --- Data analysis swarm (12 scenarios) ---
_DATA_ANALYSIS_SCENARIOS = [
    ("Analyze this CSV dataset of sales figures", "data_analysis"),
    ("Create a visualization of monthly revenue trends", "data_analysis"),
    ("Build a dashboard showing user engagement metrics", "data_analysis"),
    ("Plot a histogram of response times from the logs", "data_analysis"),
    ("Create a pandas dataframe from this JSON and find outliers", "data_analysis"),
    ("Analyze customer churn patterns in this spreadsheet", "data_analysis"),
    ("Generate statistics on website traffic by region", "data_analysis"),
    ("Visualize the correlation matrix for these features", "data_analysis"),
    ("Analyze this Excel file of employee satisfaction surveys", "data_analysis"),
    ("Create a bar chart comparing quarterly performance", "data_analysis"),
    ("Process this dataset and identify trends", "data_analysis"),
    ("Analyze the A/B test results from last sprint", "data_analysis"),
]

# --- DevOps swarm (12 scenarios) ---
_DEVOPS_SCENARIOS = [
    ("Deploy the application to Kubernetes cluster", "devops"),
    ("Set up a CI/CD pipeline with GitHub Actions", "devops"),
    ("Configure Nginx as a reverse proxy", "devops"),
    ("Set up Terraform for AWS infrastructure", "devops"),
    ("Configure monitoring with Prometheus and Grafana", "devops"),
    ("Deploy to AWS using CloudFormation", "devops"),
    ("Set up auto-scaling for the web server cluster", "devops"),
    ("Configure the CI pipeline to run on every commit", "devops"),
    ("Create an Ansible playbook for server provisioning", "devops"),
    ("Migrate the database to a new cloud region", "devops"),
    ("Set up log aggregation with ELK stack", "devops"),
    ("Deploy with Docker and Kubernetes", "devops"),
]

# --- Idea writer swarm (10 scenarios) ---
_IDEA_WRITER_SCENARIOS = [
    ("Write a blog post about AI trends in healthcare", "idea_writer"),
    ("Draft a newsletter about our product launch", "idea_writer"),
    ("Create an editorial about remote work culture", "idea_writer"),
    ("Write a technical blog post about GraphQL vs REST", "idea_writer"),
    ("Draft an article about sustainable technology", "idea_writer"),
    ("Create content for our social media campaign", "idea_writer"),
    ("Write a whitepaper on zero-trust security", "idea_writer"),
    ("Draft a case study about our biggest client", "idea_writer"),
    ("Compose a company announcement for the new feature", "idea_writer"),
    ("Write copy for the landing page hero section", "idea_writer"),
]

# --- Fundamental swarm (10 scenarios) ---
_FUNDAMENTAL_SCENARIOS = [
    ("Analyze the fundamentals of Reliance Industries stock", "fundamental"),
    ("Calculate the intrinsic valuation of Apple Inc", "fundamental"),
    ("Review the earnings report for Tesla Q4", "fundamental"),
    ("Analyze the PE ratio trends for HDFC Bank", "fundamental"),
    ("Evaluate the dividend yield of this portfolio", "fundamental"),
    ("Do a fundamental analysis of Infosys stock", "fundamental"),
    ("Assess the market cap and revenue growth of Amazon", "fundamental"),
    ("Analyze the investment potential of NVIDIA stock", "fundamental"),
    ("Review the balance sheet of JPMorgan Chase", "fundamental"),
    ("Compare the financial ratios of TCS vs Wipro", "fundamental"),
]

# --- Learning swarm (8 scenarios) ---
_LEARNING_SCENARIOS = [
    ("Create a curriculum for learning Python", "learning"),
    ("Build a lesson plan for teaching machine learning basics", "learning"),
    ("Design a training program for new hires on our tech stack", "learning"),
    ("Create a study guide for AWS certifications", "learning"),
    ("Develop an education program on cybersecurity fundamentals", "learning"),
    ("Create a teaching module on data structures", "learning"),
    ("Design a training course for junior developers", "learning"),
    ("Create a tutorial on React hooks for beginners", "learning"),
]

# --- ArXiv learning swarm (8 scenarios) ---
_ARXIV_SCENARIOS = [
    ("Get the arXiv preprint 2401.12345", "arxiv_learning"),
    ("Summarize this arXiv paper on diffusion models", "arxiv_learning"),
    ("Read the academic paper on transformer architectures", "arxiv_learning"),
    ("Find recent preprints on reinforcement learning from human feedback", "arxiv_learning"),
    ("Review this scientific paper about protein folding", "arxiv_learning"),
    ("Summarize the key findings from this journal article on LLMs", "arxiv_learning"),
    ("Analyze this arXiv preprint about multi-agent systems", "arxiv_learning"),
    ("Do a literature review of papers on graph neural networks", "arxiv_learning"),
]

# --- Olympiad learning swarm (8 scenarios) ---
_OLYMPIAD_SCENARIOS = [
    ("Prepare for the math olympiad with practice problems", "olympiad_learning"),
    ("Create olympiad-level problems in combinatorics", "olympiad_learning"),
    ("Help me prepare for the International Math Olympiad", "olympiad_learning"),
    ("Generate competition problems for science olympiad", "olympiad_learning"),
    ("Practice problems for the International Olympiad in Informatics", "olympiad_learning"),
    ("Create challenging contest problems in number theory", "olympiad_learning"),
    ("Prepare materials for competitive programming", "olympiad_learning"),
    ("Design olympiad training exercises for algebra", "olympiad_learning"),
]

# --- Fallback / None (15 scenarios) ---
_FALLBACK_SCENARIOS = [
    ("Hello, how are you?", None),
    ("What time is it?", None),
    ("Tell me a joke", None),
    ("Thanks for your help!", None),
    ("Good morning", None),
    ("Who are you?", None),
    ("Can you help me?", None),
    ("What can you do?", None),
    ("I am bored", None),
    ("Explain quantum physics to me", None),
    ("Translate this to French: Hello world", None),
    ("Send an email to the team about the deadline", None),
    ("Schedule a meeting for next Tuesday", None),
    ("Remind me to buy groceries", None),
    ("I learned a lot from the coding workshop", None),
]

# Combine all "strict" scenarios — classifier must get these right.
_ALL_STRICT_SCENARIOS = (
    _CODING_SCENARIOS
    + _RESEARCH_SCENARIOS
    + _TESTING_SCENARIOS
    + _REVIEW_SCENARIOS
    + _DATA_ANALYSIS_SCENARIOS
    + _DEVOPS_SCENARIOS
    + _IDEA_WRITER_SCENARIOS
    + _FUNDAMENTAL_SCENARIOS
    + _LEARNING_SCENARIOS
    + _ARXIV_SCENARIOS
    + _OLYMPIAD_SCENARIOS
    + _FALLBACK_SCENARIOS
)

# "Flexible" scenarios — genuinely ambiguous, accept multiple valid answers.
# Format: (goal, set_of_acceptable_swarm_names)
_FLEXIBLE_SCENARIOS = [
    # Multi-domain overlap (both answers are defensible)
    ("Study the correlation between sleep and productivity",
        {"research", "data_analysis"}),
    ("Analyze market trends for electric vehicles",
        {"research", "data_analysis"}),
    ("Research best practices for microservice architecture",
        {"research", "coding"}),
    ("Create a Docker Compose file for the microservices",
        {"devops", "coding"}),
    ("Build a step-by-step tutorial on Docker basics",
        {"learning", "devops"}),
    ("Create tests for the Docker deployment scripts",
        {"testing", "devops"}),
    ("Review the data analysis pipeline code",
        {"review", "coding"}),
    ("Write Python code to analyze a CSV dataset",
        {"coding", "data_analysis"}),
    ("Can you deploy a simple hello world script?",
        {"devops", "coding"}),
    ("Research and write a report on AI safety",
        {"research", "idea_writer"}),
    ("Build a dashboard and deploy it to AWS",
        {"devops", "data_analysis", "coding"}),
    ("Set up a WebSocket server in Node.js",
        {"coding", "devops"}),
    # Edge cases — context-dependent or idiomatic
    ("Summarize this meeting transcript",
        {"idea_writer", None}),
    ("Test the waters with a new market strategy",
        {"testing", None}),
    ("The data shows that our API is fast",
        {"coding", None}),
    ("test",
        {"testing", None}),
    ("code",
        {"coding", None}),
    ("deploy",
        {"devops", None}),
]


@pytest.mark.unit
class TestTaskClassifierStrict:
    """110+ parametrized scenarios where classifier must return the exact swarm."""

    @pytest.mark.parametrize("goal,expected", _ALL_STRICT_SCENARIOS,
                             ids=[f"{e}:{g[:50]}" for g, e in _ALL_STRICT_SCENARIOS])
    def test_strict_routing(self, goal, expected):
        tc = _get_tc()
        result = tc.classify_swarm(goal)
        assert result.swarm_name == expected, (
            f"Goal: '{goal}'\n"
            f"  Expected: {expected}\n"
            f"  Got:      {result.swarm_name}\n"
            f"  Conf:     {result.confidence:.2f}\n"
            f"  Reason:   {result.reasoning}"
        )


@pytest.mark.unit
class TestTaskClassifierFlexible:
    """18 ambiguous scenarios where multiple answers are acceptable."""

    @pytest.mark.parametrize("goal,acceptable", _FLEXIBLE_SCENARIOS,
                             ids=[f"flex:{g[:50]}" for g, _ in _FLEXIBLE_SCENARIOS])
    def test_flexible_routing(self, goal, acceptable):
        tc = _get_tc()
        result = tc.classify_swarm(goal)
        assert result.swarm_name in acceptable, (
            f"Goal: '{goal}'\n"
            f"  Acceptable: {acceptable}\n"
            f"  Got:        {result.swarm_name}\n"
            f"  Conf:       {result.confidence:.2f}\n"
            f"  Reason:     {result.reasoning}"
        )


@pytest.mark.unit
class TestTaskClassifierProperties:
    """Structural properties that must hold for ALL inputs."""

    @pytest.mark.parametrize("goal,_", _ALL_STRICT_SCENARIOS[:20],
                             ids=[f"prop:{g[:40]}" for g, _ in _ALL_STRICT_SCENARIOS[:20]])
    def test_confidence_in_range(self, goal, _):
        """Confidence is always in [0.0, 1.0]."""
        tc = _get_tc()
        result = tc.classify_swarm(goal)
        assert 0.0 <= result.confidence <= 1.0, (
            f"confidence={result.confidence} out of range for '{goal}'"
        )

    @pytest.mark.parametrize("goal,_", _ALL_STRICT_SCENARIOS[:20],
                             ids=[f"intent:{g[:40]}" for g, _ in _ALL_STRICT_SCENARIOS[:20]])
    def test_has_intent(self, goal, _):
        """Result always has a valid TaskIntent."""
        from Jotty.core.modes.execution.intent_classifier import TaskIntent
        tc = _get_tc()
        result = tc.classify_swarm(goal)
        assert isinstance(result.intent, TaskIntent), (
            f"intent is not TaskIntent for '{goal}'"
        )

    @pytest.mark.parametrize("goal,_", _ALL_STRICT_SCENARIOS[:20],
                             ids=[f"reason:{g[:40]}" for g, _ in _ALL_STRICT_SCENARIOS[:20]])
    def test_has_reasoning(self, goal, _):
        """Result always has non-empty reasoning."""
        tc = _get_tc()
        result = tc.classify_swarm(goal)
        assert result.reasoning, f"Empty reasoning for '{goal}'"

    def test_none_result_has_zero_or_low_confidence(self):
        """When swarm_name is None, confidence should be below threshold."""
        from Jotty.core.modes.execution.intent_classifier import CONFIDENCE_THRESHOLD
        tc = _get_tc()
        for goal in ["Hello", "Good morning", "Thanks"]:
            result = tc.classify_swarm(goal)
            if result.swarm_name is None:
                assert result.confidence < CONFIDENCE_THRESHOLD, (
                    f"None result has high confidence {result.confidence} for '{goal}'"
                )

    def test_singleton_returns_same_instance(self):
        """get_task_classifier() returns the same singleton."""
        from Jotty.core.modes.execution.intent_classifier import get_task_classifier
        tc1 = get_task_classifier()
        tc2 = get_task_classifier()
        assert tc1 is tc2


@pytest.mark.unit
class TestResearchSwarmRegistration:
    """Verify ResearchSwarm now registers properly."""

    def test_research_swarm_registered(self):
        """Importing research_swarm registers 'research' in SwarmRegistry."""
        import importlib
        importlib.import_module('Jotty.core.swarms.research_swarm')
        from Jotty.core.intelligence.swarms.registry import SwarmRegistry
        assert 'research' in SwarmRegistry.list_all(), (
            f"'research' not registered. Available: {SwarmRegistry.list_all()}"
        )

    def test_ensure_swarms_registered_all_11(self):
        """After _ensure_swarms_registered(), all 11 swarms are present."""
        from Jotty.core.modes.execution.executor import TierExecutor
        from Jotty.core.intelligence.swarms.registry import SwarmRegistry

        # Reset registration state
        TierExecutor._swarms_registered = False
        executor = TierExecutor.__new__(TierExecutor)
        executor._ensure_swarms_registered()

        registered = set(SwarmRegistry.list_all())
        expected = {
            'coding', 'research', 'testing', 'review', 'data_analysis',
            'devops', 'idea_writer', 'fundamental', 'learning',
            'arxiv_learning', 'olympiad_learning',
        }
        missing = expected - registered
        assert not missing, f"Missing swarms after registration: {missing}"


@pytest.mark.unit
class TestSelectSwarmIntegration:
    """Verify _select_swarm() uses TaskClassifier."""

    def test_select_swarm_uses_classifier(self):
        """_select_swarm() delegates to TaskClassifier and returns a swarm."""
        from unittest.mock import patch, MagicMock
        from Jotty.core.modes.execution.intent_classifier import TaskClassification, TaskIntent
        from Jotty.core.modes.execution.executor import TierExecutor

        mock_classification = TaskClassification(
            swarm_name="coding",
            confidence=0.75,
            reasoning="test",
            intent=TaskIntent.CODE_GENERATION,
        )

        executor = TierExecutor.__new__(TierExecutor)

        with patch.object(TierExecutor, '_ensure_swarms_registered'):
            with patch(
                'Jotty.core.execution.intent_classifier.get_task_classifier'
            ) as mock_get_tc:
                mock_tc = MagicMock()
                mock_tc.classify_swarm.return_value = mock_classification
                mock_get_tc.return_value = mock_tc

                with patch('Jotty.core.swarms.registry.SwarmRegistry.create') as mock_create:
                    mock_swarm = MagicMock()
                    mock_create.return_value = mock_swarm

                    result = executor._select_swarm("Write Python code")

                    mock_tc.classify_swarm.assert_called_once_with("Write Python code")
                    mock_create.assert_called_with("coding")
                    assert result == mock_swarm
