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
        from Jotty.core.foundation.types.execution_types import CoordinationPattern, MergeStrategy
        assert hasattr(CoordinationPattern, 'PIPELINE')
        assert hasattr(MergeStrategy, 'COMBINE')

    def test_foundation_types_reexport(self):
        from Jotty.core.foundation.types import CoordinationPattern, MergeStrategy
        assert CoordinationPattern.PARALLEL.value == "parallel"
        assert MergeStrategy.VOTE.value == "vote"

    def test_swarms_reexport_still_works(self):
        """Backward compat: importing from swarms still works."""
        from Jotty.core.swarms.base.agent_team import CoordinationPattern, MergeStrategy
        assert CoordinationPattern.PIPELINE.value == "pipeline"
        assert MergeStrategy.FIRST.value == "first"

    def test_swarms_init_reexport(self):
        from Jotty.core.swarms import CoordinationPattern, MergeStrategy
        assert len(CoordinationPattern) == 7
        assert len(MergeStrategy) == 5

    def test_composite_agent_uses_foundation(self):
        """composite_agent.py should import from foundation, not from swarms."""
        import inspect
        from Jotty.core.agents.base import composite_agent
        source = inspect.getsource(composite_agent)
        assert "from Jotty.core.foundation.types.execution_types import" in source
        assert "from Jotty.core.swarms.base.agent_team import CoordinationPattern" not in source

    def test_all_coordination_patterns(self):
        from Jotty.core.foundation.types.execution_types import CoordinationPattern
        expected = {'none', 'pipeline', 'parallel', 'consensus',
                    'hierarchical', 'blackboard', 'round_robin'}
        actual = {p.value for p in CoordinationPattern}
        assert actual == expected

    def test_all_merge_strategies(self):
        from Jotty.core.foundation.types.execution_types import MergeStrategy
        expected = {'combine', 'first', 'best', 'vote', 'concat'}
        actual = {s.value for s in MergeStrategy}
        assert actual == expected

    def test_identity_across_import_paths(self):
        """Same enum class regardless of import path."""
        from Jotty.core.foundation.types.execution_types import CoordinationPattern as CP1
        from Jotty.core.swarms.base.agent_team import CoordinationPattern as CP2
        from Jotty.core.swarms import CoordinationPattern as CP3
        assert CP1 is CP2 is CP3


# =============================================================================
# Phase 3: Focused Config Dataclasses
# =============================================================================

@pytest.mark.unit
class TestFocusedConfigs:
    """Standalone config dataclasses for each subsystem."""

    def test_all_configs_importable(self):
        from Jotty.core.foundation.configs import (
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
        from Jotty.core.foundation.configs import MemoryConfig
        cfg = MemoryConfig()
        assert cfg.episodic_capacity == 1000
        assert cfg.semantic_capacity == 500
        assert cfg.enable_llm_rag is True
        assert cfg.retrieval_mode == "synthesize"

    def test_learning_config_defaults(self):
        from Jotty.core.foundation.configs import LearningConfig
        cfg = LearningConfig()
        assert cfg.gamma == 0.99
        assert cfg.lambda_trace == 0.95
        assert cfg.alpha == 0.01
        assert cfg.enable_rl is True

    def test_execution_config_defaults(self):
        from Jotty.core.foundation.configs import ExecutionConfig
        cfg = ExecutionConfig()
        assert cfg.max_actor_iters == 50
        assert cfg.max_concurrent_agents == 10
        assert cfg.enable_deterministic is True

    def test_context_budget_config_defaults(self):
        from Jotty.core.foundation.configs import ContextBudgetConfig
        cfg = ContextBudgetConfig()
        assert cfg.max_context_tokens == 100000
        assert cfg.enable_dynamic_budget is True

    def test_validation_config_defaults(self):
        from Jotty.core.foundation.configs import ValidationConfig
        cfg = ValidationConfig()
        assert cfg.enable_validation is True
        assert cfg.max_validation_rounds == 3

    def test_monitoring_config_defaults(self):
        from Jotty.core.foundation.configs import MonitoringConfig
        cfg = MonitoringConfig()
        assert cfg.enable_debug_logging is False  # Off for production
        assert cfg.enable_metrics is True

    def test_intelligence_config_defaults(self):
        from Jotty.core.foundation.configs import IntelligenceConfig
        cfg = IntelligenceConfig()
        assert cfg.trust_min == 0.1
        assert cfg.local_mode is False

    def test_focused_config_customization(self):
        from Jotty.core.foundation.configs import MemoryConfig
        cfg = MemoryConfig(episodic_capacity=5000, enable_llm_rag=False)
        assert cfg.episodic_capacity == 5000
        assert cfg.enable_llm_rag is False


@pytest.mark.unit
class TestSwarmConfigBridge:
    """SwarmConfig to/from focused config conversion."""

    def test_to_memory_config(self):
        from Jotty.core.foundation.data_structures import SwarmConfig
        cfg = SwarmConfig(episodic_capacity=2000)
        mem = cfg.to_memory_config()
        assert mem.episodic_capacity == 2000
        assert mem.enable_llm_rag is True

    def test_to_learning_config(self):
        from Jotty.core.foundation.data_structures import SwarmConfig
        cfg = SwarmConfig(gamma=0.95)
        learn = cfg.to_learning_config()
        assert learn.gamma == 0.95
        assert learn.lambda_trace == 0.95

    def test_to_execution_config(self):
        from Jotty.core.foundation.data_structures import SwarmConfig
        cfg = SwarmConfig(max_actor_iters=100)
        exe = cfg.to_execution_config()
        assert exe.max_actor_iters == 100

    def test_to_context_budget_config(self):
        from Jotty.core.foundation.data_structures import SwarmConfig
        cfg = SwarmConfig(max_context_tokens=50000)
        ctx = cfg.to_context_budget_config()
        assert ctx.max_context_tokens == 50000

    def test_to_validation_config(self):
        from Jotty.core.foundation.data_structures import SwarmConfig
        cfg = SwarmConfig(enable_validation=False)
        val = cfg.to_validation_config()
        assert val.enable_validation is False

    def test_to_monitoring_config(self):
        from Jotty.core.foundation.data_structures import SwarmConfig
        cfg = SwarmConfig(verbose=2)
        mon = cfg.to_monitoring_config()
        assert mon.verbose == 2

    def test_to_intelligence_config(self):
        from Jotty.core.foundation.data_structures import SwarmConfig
        cfg = SwarmConfig(local_mode=True)
        intel = cfg.to_intelligence_config()
        assert intel.local_mode is True

    def test_to_persistence_config(self):
        from Jotty.core.foundation.data_structures import SwarmConfig
        cfg = SwarmConfig(storage_format="sqlite")
        pers = cfg.to_persistence_config()
        assert pers.storage_format == "sqlite"

    def test_from_configs_memory(self):
        from Jotty.core.foundation.data_structures import SwarmConfig
        from Jotty.core.foundation.configs import MemoryConfig
        cfg = SwarmConfig.from_configs(memory=MemoryConfig(episodic_capacity=3000))
        assert cfg.episodic_capacity == 3000

    def test_from_configs_multiple(self):
        from Jotty.core.foundation.data_structures import SwarmConfig
        from Jotty.core.foundation.configs import MemoryConfig, LearningConfig
        cfg = SwarmConfig.from_configs(
            memory=MemoryConfig(episodic_capacity=3000),
            learning=LearningConfig(gamma=0.5),
        )
        assert cfg.episodic_capacity == 3000
        assert cfg.gamma == 0.5

    def test_from_configs_with_overrides(self):
        from Jotty.core.foundation.data_structures import SwarmConfig
        from Jotty.core.foundation.configs import MemoryConfig
        cfg = SwarmConfig.from_configs(
            memory=MemoryConfig(episodic_capacity=3000),
            schema_version="3.0",
        )
        assert cfg.episodic_capacity == 3000
        assert cfg.schema_version == "3.0"

    def test_from_configs_override_beats_subconfig(self):
        from Jotty.core.foundation.data_structures import SwarmConfig
        from Jotty.core.foundation.configs import MemoryConfig
        cfg = SwarmConfig.from_configs(
            memory=MemoryConfig(episodic_capacity=3000),
            episodic_capacity=5000,  # Override beats sub-config
        )
        assert cfg.episodic_capacity == 5000

    def test_roundtrip_memory_config(self):
        """Extract -> modify -> compose back preserves other fields."""
        from Jotty.core.foundation.data_structures import SwarmConfig
        original = SwarmConfig(gamma=0.5, episodic_capacity=2000)
        mem = original.to_memory_config()
        mem.episodic_capacity = 4000
        rebuilt = SwarmConfig.from_configs(memory=mem, gamma=0.5)
        assert rebuilt.episodic_capacity == 4000
        assert rebuilt.gamma == 0.5

    def test_flat_dict_unchanged(self):
        """to_flat_dict() still works after adding bridge methods."""
        from Jotty.core.foundation.data_structures import SwarmConfig
        cfg = SwarmConfig()
        flat = cfg.to_flat_dict()
        assert 'gamma' in flat
        assert 'episodic_capacity' in flat
        assert 'schema_version' in flat

    def test_views_still_work(self):
        """View proxy access unchanged."""
        from Jotty.core.foundation.data_structures import SwarmConfig
        cfg = SwarmConfig(gamma=0.5)
        assert cfg.learning.gamma == 0.5
        assert cfg.memory_settings.episodic_capacity == 1000

    def test_lazy_reexport_from_data_structures(self):
        """Focused configs importable from data_structures module."""
        from Jotty.core.foundation.data_structures import MemoryConfig
        assert MemoryConfig().episodic_capacity == 1000


# =============================================================================
# Phase 5: Plugin Skill Discovery
# =============================================================================

@pytest.mark.unit
class TestPluginSkillDiscovery:
    """Skill registry plugin discovery infrastructure."""

    def test_registry_has_scan_plugin_method(self):
        from Jotty.core.registry.skills_registry import SkillsRegistry
        assert hasattr(SkillsRegistry, '_scan_plugin_skills')

    def test_registry_init_calls_plugin_scan(self):
        """Plugin scan is called during init (no plugins installed = no error)."""
        from Jotty.core.registry.skills_registry import get_skills_registry
        reg = get_skills_registry()
        reg.initialized = False  # Force re-init
        reg.loaded_skills.clear()
        reg.init()
        assert reg.initialized
        assert len(reg.loaded_skills) > 100  # Built-in skills still load

    def test_plugin_scan_handles_no_plugins(self):
        """No installed plugins = no error."""
        from Jotty.core.registry.skills_registry import SkillsRegistry
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
        from Jotty.core.foundation.configs import MemoryConfig
        from Jotty.core.memory.facade import _resolve_memory_config
        cfg = MemoryConfig(episodic_capacity=2000, enable_llm_rag=False)
        resolved = _resolve_memory_config(cfg)
        # Should be a SwarmConfig with memory fields applied
        assert resolved.episodic_capacity == 2000
        assert resolved.enable_llm_rag is False

    def test_memory_facade_accepts_swarm_config_backward_compat(self):
        """get_rag_retriever still accepts SwarmConfig."""
        from Jotty.core.foundation.data_structures import SwarmConfig
        from Jotty.core.memory.facade import _resolve_memory_config
        cfg = SwarmConfig(episodic_capacity=3000)
        resolved = _resolve_memory_config(cfg)
        assert resolved is cfg  # Pass-through, not converted

    def test_memory_facade_accepts_none(self):
        """get_rag_retriever accepts None (defaults)."""
        from Jotty.core.memory.facade import _resolve_memory_config
        resolved = _resolve_memory_config(None)
        assert resolved.episodic_capacity == 1000  # Default

    def test_cortex_accepts_memory_config(self):
        """SwarmMemory.__init__ accepts MemoryConfig via _ensure_swarm_config."""
        from Jotty.core.memory.cortex import _ensure_swarm_config
        from Jotty.core.foundation.configs import MemoryConfig
        cfg = MemoryConfig(episodic_capacity=5000)
        resolved = _ensure_swarm_config(cfg)
        assert resolved.episodic_capacity == 5000
        assert hasattr(resolved, 'gamma')  # SwarmConfig field

    def test_cortex_accepts_swarm_config(self):
        """SwarmMemory.__init__ still accepts SwarmConfig."""
        from Jotty.core.memory.cortex import _ensure_swarm_config
        from Jotty.core.foundation.data_structures import SwarmConfig
        cfg = SwarmConfig()
        resolved = _ensure_swarm_config(cfg)
        assert resolved is cfg  # Pass-through

    def test_llm_rag_accepts_memory_config(self):
        """LLM RAG components accept MemoryConfig."""
        from Jotty.core.memory.llm_rag import _ensure_swarm_config
        from Jotty.core.foundation.configs import MemoryConfig
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
        from Jotty.core.foundation.configs import LearningConfig
        from Jotty.core.learning.facade import _resolve_learning_config
        cfg = LearningConfig(gamma=0.5, lambda_trace=0.8)
        resolved = _resolve_learning_config(cfg)
        assert resolved.gamma == 0.5
        assert resolved.lambda_trace == 0.8

    def test_learning_facade_accepts_swarm_config_backward_compat(self):
        """Facade resolver still passes through SwarmConfig."""
        from Jotty.core.foundation.data_structures import SwarmConfig
        from Jotty.core.learning.facade import _resolve_learning_config
        cfg = SwarmConfig(gamma=0.7)
        resolved = _resolve_learning_config(cfg)
        assert resolved is cfg

    def test_learning_facade_accepts_none(self):
        """Facade resolver accepts None (defaults)."""
        from Jotty.core.learning.facade import _resolve_learning_config
        resolved = _resolve_learning_config(None)
        assert resolved.gamma == 0.99  # Default

    def test_td_lambda_accepts_learning_config(self):
        """TDLambdaLearner accepts LearningConfig."""
        from Jotty.core.learning.td_lambda import _ensure_swarm_config
        from Jotty.core.foundation.configs import LearningConfig
        cfg = LearningConfig(gamma=0.9, alpha=0.05)
        resolved = _ensure_swarm_config(cfg)
        assert resolved.gamma == 0.9
        assert resolved.alpha == 0.05

    def test_td_lambda_accepts_swarm_config(self):
        """TDLambdaLearner still accepts SwarmConfig."""
        from Jotty.core.learning.td_lambda import _ensure_swarm_config
        from Jotty.core.foundation.data_structures import SwarmConfig
        cfg = SwarmConfig()
        resolved = _ensure_swarm_config(cfg)
        assert resolved is cfg

    def test_reasoning_credit_accepts_learning_config(self):
        """ReasoningCreditAssigner accepts LearningConfig."""
        from Jotty.core.learning.reasoning_credit import _ensure_swarm_config
        from Jotty.core.foundation.configs import LearningConfig
        cfg = LearningConfig(reasoning_weight=0.5, evidence_weight=0.3)
        resolved = _ensure_swarm_config(cfg)
        assert resolved.reasoning_weight == 0.5
        assert resolved.evidence_weight == 0.3

    def test_adaptive_components_accept_learning_config(self):
        """Adaptive components accept LearningConfig."""
        from Jotty.core.learning.adaptive_components import _ensure_swarm_config
        from Jotty.core.foundation.configs import LearningConfig
        cfg = LearningConfig(alpha=0.02, alpha_min=0.005)
        resolved = _ensure_swarm_config(cfg)
        assert resolved.alpha == 0.02
        assert resolved.alpha_min == 0.005

    def test_offline_learning_accepts_learning_config(self):
        """Offline learning components accept LearningConfig."""
        from Jotty.core.learning.offline_learning import _ensure_swarm_config
        from Jotty.core.foundation.configs import LearningConfig
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
        from Jotty.core.skill_sdk import (
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
        from Jotty.core.skill_sdk.tool_helpers import (
            tool_response, tool_error, require_params,
        )
        resp = tool_response(data={"ok": True})
        assert resp["success"] is True
        err = tool_error("bad input")
        assert err["success"] is False

    def test_skill_sdk_skill_status_works(self):
        """SkillStatus from skill_sdk works."""
        from Jotty.core.skill_sdk import SkillStatus
        status = SkillStatus("test-skill")
        assert status.skill_name == "test-skill"

    def test_skill_sdk_env_loader_works(self):
        """env_loader from skill_sdk works."""
        from Jotty.core.skill_sdk import get_env
        # Should not raise (returns None if not set)
        result = get_env("NONEXISTENT_VAR_12345")
        assert result is None

    def test_skill_sdk_api_client_works(self):
        """BaseAPIClient from skill_sdk works."""
        from Jotty.core.skill_sdk.api_client import BaseAPIClient
        assert hasattr(BaseAPIClient, '_make_request')

    def test_utils_reexport_backward_compat(self):
        """Importing from core.utils still works (backward compat)."""
        from Jotty.core.utils.skill_status import SkillStatus
        from Jotty.core.utils.tool_helpers import tool_response
        from Jotty.core.utils.env_loader import get_env
        assert SkillStatus is not None
        assert tool_response is not None
        assert get_env is not None

    def test_skill_sdk_smart_fetcher(self):
        """smart_fetcher from skill_sdk is accessible."""
        from Jotty.core.skill_sdk.smart_fetcher import smart_fetch, FetchResult
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
