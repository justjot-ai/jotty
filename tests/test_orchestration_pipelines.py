"""
Tests for Orchestration Pipeline Modules (Phase 7)
====================================================
Tests for:
- ParadigmExecutor (core/orchestration/paradigm_executor.py)
- TrustLevel, SandboxType, SandboxConfig, SandboxResult, SandboxManager
  (core/orchestration/sandbox_manager.py)
- EffectivenessTracker (core/orchestration/learning_pipeline.py)
- OptimizationConfig, IterationResult, OptimizationPipeline
  (core/orchestration/optimization_pipeline.py)
"""

import pytest
import asyncio
import time
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, MagicMock, AsyncMock, patch, PropertyMock
from dataclasses import fields, asdict

# ---------------------------------------------------------------------------
# Conditional imports with pytest.skip support
# ---------------------------------------------------------------------------
try:
    from Jotty.core.orchestration.sandbox_manager import (
        TrustLevel, SandboxType, SandboxConfig, SandboxResult, SandboxManager,
    )
    HAS_SANDBOX = True
except ImportError:
    HAS_SANDBOX = False

try:
    from Jotty.core.orchestration.learning_pipeline import EffectivenessTracker
    HAS_EFFECTIVENESS = True
except ImportError:
    HAS_EFFECTIVENESS = False

try:
    from Jotty.core.orchestration.paradigm_executor import ParadigmExecutor
    HAS_PARADIGM = True
except ImportError:
    HAS_PARADIGM = False

try:
    from Jotty.core.orchestration.optimization_pipeline import (
        OptimizationConfig, IterationResult, OptimizationPipeline,
    )
    HAS_OPTIMIZATION = True
except ImportError:
    HAS_OPTIMIZATION = False

try:
    from Jotty.core.foundation.data_structures import EpisodeResult, SwarmConfig
    from Jotty.core.foundation.agent_config import AgentConfig
    HAS_FOUNDATION = True
except ImportError:
    HAS_FOUNDATION = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_episode_result(**overrides):
    """Create a minimal EpisodeResult with sensible defaults."""
    defaults = dict(
        output="test output",
        success=True,
        trajectory=[],
        tagged_outputs=[],
        episode=0,
        execution_time=1.0,
        architect_results=[],
        auditor_results=[],
        agent_contributions={},
    )
    defaults.update(overrides)
    return EpisodeResult(**defaults)


def _make_mock_manager(**overrides):
    """Create a mock Orchestrator (manager) for ParadigmExecutor."""
    m = MagicMock()
    m.agents = overrides.get("agents", [])
    m.runners = overrides.get("runners", {})
    m.agent_semaphore = asyncio.Semaphore(5)
    m.episode_count = overrides.get("episode_count", 0)
    m._schedule_background_learning = MagicMock()
    m._mas_zero_verify = MagicMock(return_value=None)
    m.credit_weights = MagicMock()
    m.credit_weights.get = MagicMock(side_effect=lambda k: {"base_reward": 0.3, "cooperation_bonus": 0.4, "predictability_bonus": 0.3}.get(k, 0.0))
    m.credit_weights.update_from_feedback = MagicMock()
    m.learning_manager = MagicMock()
    m.learning = MagicMock()
    m.learning.adaptive_learning = MagicMock()
    m.learning.adaptive_learning.should_stop_early = MagicMock(return_value=False)
    return m


# =============================================================================
# TrustLevel Tests
# =============================================================================

@pytest.mark.unit
class TestTrustLevel:
    """Tests for the TrustLevel enum."""

    @pytest.mark.skipif(not HAS_SANDBOX, reason="sandbox_manager not importable")
    def test_trusted_value(self):
        assert TrustLevel.TRUSTED.value == "trusted"

    @pytest.mark.skipif(not HAS_SANDBOX, reason="sandbox_manager not importable")
    def test_sandboxed_value(self):
        assert TrustLevel.SANDBOXED.value == "sandboxed"

    @pytest.mark.skipif(not HAS_SANDBOX, reason="sandbox_manager not importable")
    def test_dangerous_value(self):
        assert TrustLevel.DANGEROUS.value == "dangerous"

    @pytest.mark.skipif(not HAS_SANDBOX, reason="sandbox_manager not importable")
    def test_all_members(self):
        names = [m.name for m in TrustLevel]
        assert set(names) == {"TRUSTED", "SANDBOXED", "DANGEROUS"}


# =============================================================================
# SandboxType Tests
# =============================================================================

@pytest.mark.unit
class TestSandboxType:
    """Tests for the SandboxType enum."""

    @pytest.mark.skipif(not HAS_SANDBOX, reason="sandbox_manager not importable")
    def test_none_value(self):
        assert SandboxType.NONE.value == "none"

    @pytest.mark.skipif(not HAS_SANDBOX, reason="sandbox_manager not importable")
    def test_e2b_value(self):
        assert SandboxType.E2B.value == "e2b"

    @pytest.mark.skipif(not HAS_SANDBOX, reason="sandbox_manager not importable")
    def test_docker_value(self):
        assert SandboxType.DOCKER.value == "docker"

    @pytest.mark.skipif(not HAS_SANDBOX, reason="sandbox_manager not importable")
    def test_subprocess_value(self):
        assert SandboxType.SUBPROCESS.value == "subprocess"


# =============================================================================
# SandboxConfig Tests
# =============================================================================

@pytest.mark.unit
class TestSandboxConfig:
    """Tests for the SandboxConfig dataclass."""

    @pytest.mark.skipif(not HAS_SANDBOX, reason="sandbox_manager not importable")
    def test_defaults(self):
        cfg = SandboxConfig(sandbox_type=SandboxType.NONE)
        assert cfg.timeout == 120
        assert cfg.memory_limit == "512m"
        assert cfg.cpu_limit == 1.0
        assert cfg.network_enabled is False
        assert cfg.filesystem_access == []
        assert cfg.environment == {}

    @pytest.mark.skipif(not HAS_SANDBOX, reason="sandbox_manager not importable")
    def test_custom_values(self):
        cfg = SandboxConfig(
            sandbox_type=SandboxType.DOCKER,
            timeout=60,
            memory_limit="1g",
            cpu_limit=2.0,
            network_enabled=True,
            filesystem_access=["/tmp"],
            environment={"FOO": "bar"},
        )
        assert cfg.sandbox_type == SandboxType.DOCKER
        assert cfg.timeout == 60
        assert cfg.memory_limit == "1g"
        assert cfg.cpu_limit == 2.0
        assert cfg.network_enabled is True
        assert cfg.filesystem_access == ["/tmp"]
        assert cfg.environment == {"FOO": "bar"}

    @pytest.mark.skipif(not HAS_SANDBOX, reason="sandbox_manager not importable")
    def test_sandbox_type_required(self):
        """sandbox_type is a required positional field."""
        with pytest.raises(TypeError):
            SandboxConfig()

    @pytest.mark.skipif(not HAS_SANDBOX, reason="sandbox_manager not importable")
    def test_filesystem_access_is_independent(self):
        """Each instance gets its own list for filesystem_access."""
        a = SandboxConfig(sandbox_type=SandboxType.NONE)
        b = SandboxConfig(sandbox_type=SandboxType.NONE)
        a.filesystem_access.append("/mnt")
        assert b.filesystem_access == []

    @pytest.mark.skipif(not HAS_SANDBOX, reason="sandbox_manager not importable")
    def test_environment_is_independent(self):
        """Each instance gets its own dict for environment."""
        a = SandboxConfig(sandbox_type=SandboxType.NONE)
        b = SandboxConfig(sandbox_type=SandboxType.NONE)
        a.environment["KEY"] = "val"
        assert b.environment == {}

    @pytest.mark.skipif(not HAS_SANDBOX, reason="sandbox_manager not importable")
    def test_all_sandbox_types_accepted(self):
        for st in SandboxType:
            cfg = SandboxConfig(sandbox_type=st)
            assert cfg.sandbox_type is st


# =============================================================================
# SandboxResult Tests
# =============================================================================

@pytest.mark.unit
class TestSandboxResult:
    """Tests for the SandboxResult dataclass."""

    @pytest.mark.skipif(not HAS_SANDBOX, reason="sandbox_manager not importable")
    def test_defaults(self):
        r = SandboxResult(success=True, output="hello")
        assert r.success is True
        assert r.output == "hello"
        assert r.error == ""
        assert r.execution_time == 0.0
        assert r.sandbox_type == ""
        assert r.exit_code == 0
        assert r.stdout == ""
        assert r.stderr == ""
        assert r.metadata == {}

    @pytest.mark.skipif(not HAS_SANDBOX, reason="sandbox_manager not importable")
    def test_custom_values(self):
        r = SandboxResult(
            success=False,
            output=None,
            error="timeout",
            execution_time=5.5,
            sandbox_type="docker",
            exit_code=1,
            stdout="out",
            stderr="err",
            metadata={"key": "val"},
        )
        assert r.success is False
        assert r.output is None
        assert r.error == "timeout"
        assert r.execution_time == 5.5
        assert r.sandbox_type == "docker"
        assert r.exit_code == 1
        assert r.stdout == "out"
        assert r.stderr == "err"
        assert r.metadata == {"key": "val"}

    @pytest.mark.skipif(not HAS_SANDBOX, reason="sandbox_manager not importable")
    def test_success_and_output_required(self):
        with pytest.raises(TypeError):
            SandboxResult()

    @pytest.mark.skipif(not HAS_SANDBOX, reason="sandbox_manager not importable")
    def test_metadata_independent(self):
        a = SandboxResult(success=True, output="x")
        b = SandboxResult(success=True, output="y")
        a.metadata["k"] = 1
        assert "k" not in b.metadata

    @pytest.mark.skipif(not HAS_SANDBOX, reason="sandbox_manager not importable")
    def test_output_can_be_any_type(self):
        r = SandboxResult(success=True, output={"key": [1, 2, 3]})
        assert r.output == {"key": [1, 2, 3]}

    @pytest.mark.skipif(not HAS_SANDBOX, reason="sandbox_manager not importable")
    def test_fields_count(self):
        """SandboxResult has 9 fields."""
        assert len(fields(SandboxResult)) == 9


# =============================================================================
# SandboxManager Init Tests
# =============================================================================

@pytest.mark.unit
class TestSandboxManagerInit:
    """Tests for SandboxManager initialization."""

    @pytest.mark.skipif(not HAS_SANDBOX, reason="sandbox_manager not importable")
    @patch("Jotty.core.orchestration.sandbox_manager.subprocess.run")
    def test_basic_init_no_backends(self, mock_run):
        """Init with no docker / no e2b available."""
        mock_run.side_effect = FileNotFoundError
        mgr = SandboxManager()
        assert mgr.e2b_available is False
        assert mgr.docker_available is False
        assert mgr.default_timeout == 120

    @pytest.mark.skipif(not HAS_SANDBOX, reason="sandbox_manager not importable")
    @patch("Jotty.core.orchestration.sandbox_manager.subprocess.run")
    def test_custom_config(self, mock_run):
        mock_run.side_effect = FileNotFoundError
        mgr = SandboxManager(config={"timeout": 60, "docker_image": "node:18"})
        assert mgr.default_timeout == 60
        assert mgr.docker_image == "node:18"

    @pytest.mark.skipif(not HAS_SANDBOX, reason="sandbox_manager not importable")
    @patch("Jotty.core.orchestration.sandbox_manager.subprocess.run")
    def test_docker_detected(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        mgr = SandboxManager()
        assert mgr.docker_available is True

    @pytest.mark.skipif(not HAS_SANDBOX, reason="sandbox_manager not importable")
    @patch("Jotty.core.orchestration.sandbox_manager.subprocess.run")
    def test_docker_not_running(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1)
        mgr = SandboxManager()
        assert mgr.docker_available is False

    @pytest.mark.skipif(not HAS_SANDBOX, reason="sandbox_manager not importable")
    @patch("Jotty.core.orchestration.sandbox_manager.subprocess.run")
    def test_default_docker_image(self, mock_run):
        mock_run.side_effect = FileNotFoundError
        mgr = SandboxManager()
        assert mgr.docker_image == "python:3.11-slim"

    @pytest.mark.skipif(not HAS_SANDBOX, reason="sandbox_manager not importable")
    @patch("Jotty.core.orchestration.sandbox_manager.subprocess.run")
    def test_e2b_api_key_from_config(self, mock_run):
        mock_run.side_effect = FileNotFoundError
        with patch.dict("os.environ", {}, clear=False):
            mgr = SandboxManager(config={"e2b_api_key": "test-key"})
            # e2b_available depends on whether e2b_code_interpreter is importable
            # but the api key should be stored if the module is importable
            # Since e2b is not installed in test env, e2b_available should be False
            assert mgr.e2b_available is False


# =============================================================================
# SandboxManager Config Tests
# =============================================================================

@pytest.mark.unit
class TestSandboxManagerConfig:
    """Tests for SandboxManager.get_sandbox_config()."""

    @pytest.mark.skipif(not HAS_SANDBOX, reason="sandbox_manager not importable")
    @patch("Jotty.core.orchestration.sandbox_manager.subprocess.run")
    def test_trusted_config(self, mock_run):
        mock_run.side_effect = FileNotFoundError
        mgr = SandboxManager()
        cfg = mgr.get_sandbox_config(TrustLevel.TRUSTED)
        assert cfg.sandbox_type == SandboxType.NONE
        assert cfg.network_enabled is True

    @pytest.mark.skipif(not HAS_SANDBOX, reason="sandbox_manager not importable")
    @patch("Jotty.core.orchestration.sandbox_manager.subprocess.run")
    def test_sandboxed_no_docker(self, mock_run):
        mock_run.side_effect = FileNotFoundError
        mgr = SandboxManager()
        cfg = mgr.get_sandbox_config(TrustLevel.SANDBOXED)
        assert cfg.sandbox_type == SandboxType.SUBPROCESS
        assert cfg.network_enabled is False
        assert cfg.timeout <= 60

    @pytest.mark.skipif(not HAS_SANDBOX, reason="sandbox_manager not importable")
    @patch("Jotty.core.orchestration.sandbox_manager.subprocess.run")
    def test_sandboxed_with_docker(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        mgr = SandboxManager()
        cfg = mgr.get_sandbox_config(TrustLevel.SANDBOXED)
        assert cfg.sandbox_type == SandboxType.DOCKER
        assert cfg.network_enabled is False

    @pytest.mark.skipif(not HAS_SANDBOX, reason="sandbox_manager not importable")
    @patch("Jotty.core.orchestration.sandbox_manager.subprocess.run")
    def test_dangerous_no_backends(self, mock_run):
        mock_run.side_effect = FileNotFoundError
        mgr = SandboxManager()
        cfg = mgr.get_sandbox_config(TrustLevel.DANGEROUS)
        assert cfg.sandbox_type == SandboxType.SUBPROCESS
        assert cfg.timeout == 30
        assert cfg.network_enabled is False

    @pytest.mark.skipif(not HAS_SANDBOX, reason="sandbox_manager not importable")
    @patch("Jotty.core.orchestration.sandbox_manager.subprocess.run")
    def test_dangerous_with_docker(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        mgr = SandboxManager()
        cfg = mgr.get_sandbox_config(TrustLevel.DANGEROUS)
        assert cfg.sandbox_type == SandboxType.DOCKER
        assert cfg.timeout == 60
        assert cfg.memory_limit == "256m"
        assert cfg.cpu_limit == 0.5

    @pytest.mark.skipif(not HAS_SANDBOX, reason="sandbox_manager not importable")
    @patch("Jotty.core.orchestration.sandbox_manager.subprocess.run")
    def test_sandboxed_docker_memory_limit(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        mgr = SandboxManager()
        cfg = mgr.get_sandbox_config(TrustLevel.SANDBOXED)
        assert cfg.memory_limit == "512m"

    @pytest.mark.skipif(not HAS_SANDBOX, reason="sandbox_manager not importable")
    @patch("Jotty.core.orchestration.sandbox_manager.subprocess.run")
    def test_trusted_uses_default_timeout(self, mock_run):
        mock_run.side_effect = FileNotFoundError
        mgr = SandboxManager(config={"timeout": 90})
        cfg = mgr.get_sandbox_config(TrustLevel.TRUSTED)
        assert cfg.timeout == 90

    @pytest.mark.skipif(not HAS_SANDBOX, reason="sandbox_manager not importable")
    @patch("Jotty.core.orchestration.sandbox_manager.subprocess.run")
    def test_sandboxed_no_docker_caps_timeout(self, mock_run):
        """Without docker, SANDBOXED timeout is capped at 60."""
        mock_run.side_effect = FileNotFoundError
        mgr = SandboxManager(config={"timeout": 200})
        cfg = mgr.get_sandbox_config(TrustLevel.SANDBOXED)
        assert cfg.timeout == 60

    @pytest.mark.skipif(not HAS_SANDBOX, reason="sandbox_manager not importable")
    @patch("Jotty.core.orchestration.sandbox_manager.subprocess.run")
    def test_dangerous_e2b_simulated(self, mock_run):
        """Simulate e2b available by patching attributes after init."""
        mock_run.side_effect = FileNotFoundError
        mgr = SandboxManager()
        mgr.e2b_available = True
        cfg = mgr.get_sandbox_config(TrustLevel.DANGEROUS)
        assert cfg.sandbox_type == SandboxType.E2B
        assert cfg.memory_limit == "256m"
        assert cfg.cpu_limit == 0.5

    @pytest.mark.skipif(not HAS_SANDBOX, reason="sandbox_manager not importable")
    @patch("Jotty.core.orchestration.sandbox_manager.subprocess.run")
    def test_trusted_cpu_limit(self, mock_run):
        mock_run.side_effect = FileNotFoundError
        mgr = SandboxManager()
        cfg = mgr.get_sandbox_config(TrustLevel.TRUSTED)
        assert cfg.cpu_limit == 1.0


# =============================================================================
# SandboxManager Execute Tests
# =============================================================================

@pytest.mark.unit
class TestSandboxManagerExecute:
    """Tests for SandboxManager execution methods."""

    @pytest.mark.skipif(not HAS_SANDBOX, reason="sandbox_manager not importable")
    @pytest.mark.asyncio
    @patch("Jotty.core.orchestration.sandbox_manager.subprocess.run")
    async def test_execute_trusted_direct(self, mock_run):
        """TRUSTED code runs via exec() directly."""
        mock_run.side_effect = FileNotFoundError
        mgr = SandboxManager()
        result = await mgr.execute_sandboxed(
            code="result = 42",
            trust_level=TrustLevel.TRUSTED,
        )
        assert result.success is True
        assert result.output == 42
        assert result.sandbox_type == "none"

    @pytest.mark.skipif(not HAS_SANDBOX, reason="sandbox_manager not importable")
    @pytest.mark.asyncio
    @patch("Jotty.core.orchestration.sandbox_manager.subprocess.run")
    async def test_execute_trusted_with_context(self, mock_run):
        mock_run.side_effect = FileNotFoundError
        mgr = SandboxManager()
        result = await mgr.execute_sandboxed(
            code="result = x + y",
            trust_level=TrustLevel.TRUSTED,
            context={"x": 10, "y": 20},
        )
        assert result.success is True
        assert result.output == 30

    @pytest.mark.skipif(not HAS_SANDBOX, reason="sandbox_manager not importable")
    @pytest.mark.asyncio
    @patch("Jotty.core.orchestration.sandbox_manager.subprocess.run")
    async def test_execute_trusted_error(self, mock_run):
        mock_run.side_effect = FileNotFoundError
        mgr = SandboxManager()
        result = await mgr.execute_sandboxed(
            code="raise ValueError('boom')",
            trust_level=TrustLevel.TRUSTED,
        )
        assert result.success is False
        assert "boom" in result.error

    @pytest.mark.skipif(not HAS_SANDBOX, reason="sandbox_manager not importable")
    @pytest.mark.asyncio
    @patch("Jotty.core.orchestration.sandbox_manager.subprocess.run")
    async def test_execute_trusted_non_python(self, mock_run):
        mock_run.side_effect = FileNotFoundError
        mgr = SandboxManager()
        result = await mgr.execute_sandboxed(
            code="console.log('hi')",
            trust_level=TrustLevel.TRUSTED,
            language="javascript",
        )
        assert result.success is False
        assert "Python" in result.error

    @pytest.mark.skipif(not HAS_SANDBOX, reason="sandbox_manager not importable")
    @pytest.mark.asyncio
    @patch("Jotty.core.orchestration.sandbox_manager.subprocess.run")
    async def test_execute_subprocess_simple(self, mock_run):
        """Subprocess execution for SANDBOXED code."""
        mock_run.side_effect = FileNotFoundError
        mgr = SandboxManager()
        result = await mgr.execute_sandboxed(
            code="print('hello world')",
            trust_level=TrustLevel.SANDBOXED,
        )
        assert result.success is True
        assert "hello world" in result.stdout

    @pytest.mark.skipif(not HAS_SANDBOX, reason="sandbox_manager not importable")
    @pytest.mark.asyncio
    @patch("Jotty.core.orchestration.sandbox_manager.subprocess.run")
    async def test_execute_subprocess_with_context(self, mock_run):
        """Subprocess with context injection."""
        mock_run.side_effect = FileNotFoundError
        mgr = SandboxManager()
        result = await mgr.execute_sandboxed(
            code="print(greeting)",
            trust_level=TrustLevel.SANDBOXED,
            context={"greeting": "hello from context"},
        )
        assert result.success is True
        assert "hello from context" in result.stdout

    @pytest.mark.skipif(not HAS_SANDBOX, reason="sandbox_manager not importable")
    @pytest.mark.asyncio
    @patch("Jotty.core.orchestration.sandbox_manager.subprocess.run")
    async def test_execute_subprocess_error(self, mock_run):
        mock_run.side_effect = FileNotFoundError
        mgr = SandboxManager()
        result = await mgr.execute_sandboxed(
            code="import sys; sys.exit(1)",
            trust_level=TrustLevel.SANDBOXED,
        )
        assert result.success is False
        assert result.exit_code != 0

    @pytest.mark.skipif(not HAS_SANDBOX, reason="sandbox_manager not importable")
    @pytest.mark.asyncio
    @patch("Jotty.core.orchestration.sandbox_manager.subprocess.run")
    async def test_execute_subprocess_non_python(self, mock_run):
        mock_run.side_effect = FileNotFoundError
        mgr = SandboxManager()
        result = await mgr.execute_sandboxed(
            code="console.log('hi')",
            trust_level=TrustLevel.SANDBOXED,
            language="javascript",
        )
        assert result.success is False
        assert "Python" in result.error

    @pytest.mark.skipif(not HAS_SANDBOX, reason="sandbox_manager not importable")
    @pytest.mark.asyncio
    @patch("Jotty.core.orchestration.sandbox_manager.subprocess.run")
    async def test_execution_time_tracked(self, mock_run):
        mock_run.side_effect = FileNotFoundError
        mgr = SandboxManager()
        result = await mgr.execute_sandboxed(
            code="result = 1",
            trust_level=TrustLevel.TRUSTED,
        )
        assert result.execution_time > 0

    @pytest.mark.skipif(not HAS_SANDBOX, reason="sandbox_manager not importable")
    def test_build_context_code_string(self):
        mgr = SandboxManager.__new__(SandboxManager)
        code = mgr._build_context_code({"name": "alice"})
        assert 'name = """alice"""' in code

    @pytest.mark.skipif(not HAS_SANDBOX, reason="sandbox_manager not importable")
    def test_build_context_code_int(self):
        mgr = SandboxManager.__new__(SandboxManager)
        code = mgr._build_context_code({"x": 42})
        assert "x = 42" in code

    @pytest.mark.skipif(not HAS_SANDBOX, reason="sandbox_manager not importable")
    def test_build_context_code_list(self):
        mgr = SandboxManager.__new__(SandboxManager)
        code = mgr._build_context_code({"items": [1, 2, 3]})
        assert "items = [1, 2, 3]" in code

    @pytest.mark.skipif(not HAS_SANDBOX, reason="sandbox_manager not importable")
    def test_build_context_code_bool(self):
        mgr = SandboxManager.__new__(SandboxManager)
        code = mgr._build_context_code({"flag": True})
        assert "flag = True" in code

    @pytest.mark.skipif(not HAS_SANDBOX, reason="sandbox_manager not importable")
    def test_build_context_code_none(self):
        mgr = SandboxManager.__new__(SandboxManager)
        code = mgr._build_context_code({"val": None})
        assert "val = None" in code


# =============================================================================
# SandboxManager Status Tests
# =============================================================================

@pytest.mark.unit
class TestSandboxManagerStatus:
    """Tests for SandboxManager.get_available_backends() and get_status()."""

    @pytest.mark.skipif(not HAS_SANDBOX, reason="sandbox_manager not importable")
    @patch("Jotty.core.orchestration.sandbox_manager.subprocess.run")
    def test_available_backends_subprocess_always(self, mock_run):
        mock_run.side_effect = FileNotFoundError
        mgr = SandboxManager()
        backends = mgr.get_available_backends()
        assert "subprocess" in backends

    @pytest.mark.skipif(not HAS_SANDBOX, reason="sandbox_manager not importable")
    @patch("Jotty.core.orchestration.sandbox_manager.subprocess.run")
    def test_available_backends_with_docker(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        mgr = SandboxManager()
        backends = mgr.get_available_backends()
        assert "docker" in backends
        assert "subprocess" in backends

    @pytest.mark.skipif(not HAS_SANDBOX, reason="sandbox_manager not importable")
    @patch("Jotty.core.orchestration.sandbox_manager.subprocess.run")
    def test_status_keys(self, mock_run):
        mock_run.side_effect = FileNotFoundError
        mgr = SandboxManager()
        status = mgr.get_status()
        assert "e2b_available" in status
        assert "docker_available" in status
        assert "available_backends" in status
        assert "default_timeout" in status
        assert "docker_image" in status

    @pytest.mark.skipif(not HAS_SANDBOX, reason="sandbox_manager not importable")
    @patch("Jotty.core.orchestration.sandbox_manager.subprocess.run")
    def test_status_values(self, mock_run):
        mock_run.side_effect = FileNotFoundError
        mgr = SandboxManager(config={"timeout": 90})
        status = mgr.get_status()
        assert status["default_timeout"] == 90
        assert status["e2b_available"] is False
        assert status["docker_available"] is False


# =============================================================================
# EffectivenessTracker Init Tests
# =============================================================================

@pytest.mark.unit
class TestEffectivenessTrackerInit:
    """Tests for EffectivenessTracker initialization."""

    @pytest.mark.skipif(not HAS_EFFECTIVENESS, reason="learning_pipeline not importable")
    def test_default_windows(self):
        t = EffectivenessTracker()
        assert t.recent_window == 20
        assert t.historical_window == 100

    @pytest.mark.skipif(not HAS_EFFECTIVENESS, reason="learning_pipeline not importable")
    def test_custom_windows(self):
        t = EffectivenessTracker(recent_window=5, historical_window=50)
        assert t.recent_window == 5
        assert t.historical_window == 50

    @pytest.mark.skipif(not HAS_EFFECTIVENESS, reason="learning_pipeline not importable")
    def test_empty_global_on_init(self):
        t = EffectivenessTracker()
        assert len(t._global) == 0

    @pytest.mark.skipif(not HAS_EFFECTIVENESS, reason="learning_pipeline not importable")
    def test_empty_records_on_init(self):
        t = EffectivenessTracker()
        assert len(t._records) == 0


# =============================================================================
# EffectivenessTracker Record Tests
# =============================================================================

@pytest.mark.unit
class TestEffectivenessTrackerRecord:
    """Tests for EffectivenessTracker.record()."""

    @pytest.mark.skipif(not HAS_EFFECTIVENESS, reason="learning_pipeline not importable")
    def test_single_record(self):
        t = EffectivenessTracker()
        t.record("analysis", success=True, quality=0.8)
        assert len(t._records["analysis"]) == 1
        assert len(t._global) == 1

    @pytest.mark.skipif(not HAS_EFFECTIVENESS, reason="learning_pipeline not importable")
    def test_multiple_records(self):
        t = EffectivenessTracker()
        t.record("analysis", success=True, quality=0.8)
        t.record("analysis", success=False, quality=0.2)
        t.record("coding", success=True, quality=0.9)
        assert len(t._records["analysis"]) == 2
        assert len(t._records["coding"]) == 1
        assert len(t._global) == 3

    @pytest.mark.skipif(not HAS_EFFECTIVENESS, reason="learning_pipeline not importable")
    def test_quality_clamped_high(self):
        t = EffectivenessTracker()
        t.record("analysis", success=True, quality=1.5)
        _, _, q, _ = t._records["analysis"][0]
        assert q == 1.0

    @pytest.mark.skipif(not HAS_EFFECTIVENESS, reason="learning_pipeline not importable")
    def test_quality_clamped_low(self):
        t = EffectivenessTracker()
        t.record("analysis", success=True, quality=-0.5)
        _, _, q, _ = t._records["analysis"][0]
        assert q == 0.0

    @pytest.mark.skipif(not HAS_EFFECTIVENESS, reason="learning_pipeline not importable")
    def test_agent_stored(self):
        t = EffectivenessTracker()
        t.record("coding", success=True, quality=0.7, agent="code_agent")
        _, _, _, a = t._records["coding"][0]
        assert a == "code_agent"

    @pytest.mark.skipif(not HAS_EFFECTIVENESS, reason="learning_pipeline not importable")
    def test_timestamp_stored(self):
        t = EffectivenessTracker()
        before = time.time()
        t.record("analysis", success=True)
        after = time.time()
        ts, _, _, _ = t._records["analysis"][0]
        assert before <= ts <= after

    @pytest.mark.skipif(not HAS_EFFECTIVENESS, reason="learning_pipeline not importable")
    def test_maxlen_respected(self):
        t = EffectivenessTracker(recent_window=3, historical_window=2)
        for i in range(10):
            t.record("analysis", success=True, quality=0.5)
        assert len(t._records["analysis"]) == 5  # recent + historical
        assert len(t._global) == 5

    @pytest.mark.skipif(not HAS_EFFECTIVENESS, reason="learning_pipeline not importable")
    def test_default_quality(self):
        t = EffectivenessTracker()
        t.record("analysis", success=True)
        _, _, q, _ = t._records["analysis"][0]
        assert q == 0.0


# =============================================================================
# EffectivenessTracker Windows Tests
# =============================================================================

@pytest.mark.unit
class TestEffectivenessTrackerWindows:
    """Tests for _split_windows() and _rate()."""

    @pytest.mark.skipif(not HAS_EFFECTIVENESS, reason="learning_pipeline not importable")
    def test_split_windows_small(self):
        """Fewer items than recent_window: all recent, no historical."""
        t = EffectivenessTracker(recent_window=5)
        for i in range(3):
            t.record("a", success=True)
        recent, historical = t._split_windows(t._records["a"])
        assert len(recent) == 3
        assert len(historical) == 0

    @pytest.mark.skipif(not HAS_EFFECTIVENESS, reason="learning_pipeline not importable")
    def test_split_windows_exact(self):
        """Exactly recent_window items: all recent, no historical."""
        t = EffectivenessTracker(recent_window=5)
        for i in range(5):
            t.record("a", success=True)
        recent, historical = t._split_windows(t._records["a"])
        assert len(recent) == 5
        assert len(historical) == 0

    @pytest.mark.skipif(not HAS_EFFECTIVENESS, reason="learning_pipeline not importable")
    def test_split_windows_overflow(self):
        """More items than recent_window: splits correctly."""
        t = EffectivenessTracker(recent_window=3, historical_window=10)
        for i in range(8):
            t.record("a", success=(i % 2 == 0))
        recent, historical = t._split_windows(t._records["a"])
        assert len(recent) == 3
        assert len(historical) == 5

    @pytest.mark.skipif(not HAS_EFFECTIVENESS, reason="learning_pipeline not importable")
    def test_rate_empty(self):
        t = EffectivenessTracker()
        sr, aq = t._rate([])
        assert sr == 0.0
        assert aq == 0.0

    @pytest.mark.skipif(not HAS_EFFECTIVENESS, reason="learning_pipeline not importable")
    def test_rate_all_success(self):
        t = EffectivenessTracker()
        records = [(time.time(), True, 0.8, "a") for _ in range(5)]
        sr, aq = t._rate(records)
        assert sr == 1.0
        assert abs(aq - 0.8) < 1e-6

    @pytest.mark.skipif(not HAS_EFFECTIVENESS, reason="learning_pipeline not importable")
    def test_rate_mixed(self):
        t = EffectivenessTracker()
        records = [
            (time.time(), True, 0.9, "a"),
            (time.time(), False, 0.1, "a"),
        ]
        sr, aq = t._rate(records)
        assert sr == 0.5
        assert abs(aq - 0.5) < 1e-6


# =============================================================================
# EffectivenessTracker Report Tests
# =============================================================================

@pytest.mark.unit
class TestEffectivenessTrackerReport:
    """Tests for improvement_report()."""

    @pytest.mark.skipif(not HAS_EFFECTIVENESS, reason="learning_pipeline not importable")
    def test_empty_report(self):
        t = EffectivenessTracker()
        report = t.improvement_report()
        assert "_global" in report
        assert report["_global"]["total_episodes"] == 0

    @pytest.mark.skipif(not HAS_EFFECTIVENESS, reason="learning_pipeline not importable")
    def test_single_task_report(self):
        t = EffectivenessTracker(recent_window=3, historical_window=10)
        for _ in range(5):
            t.record("analysis", success=True, quality=0.8)
        report = t.improvement_report()
        assert "analysis" in report
        assert report["analysis"]["total_episodes"] == 5

    @pytest.mark.skipif(not HAS_EFFECTIVENESS, reason="learning_pipeline not importable")
    def test_multi_task_report(self):
        t = EffectivenessTracker()
        t.record("analysis", success=True, quality=0.8)
        t.record("coding", success=False, quality=0.2)
        report = t.improvement_report()
        assert "analysis" in report
        assert "coding" in report
        assert "_global" in report

    @pytest.mark.skipif(not HAS_EFFECTIVENESS, reason="learning_pipeline not importable")
    def test_report_trend_positive(self):
        """Recent better than historical => positive trend."""
        t = EffectivenessTracker(recent_window=3, historical_window=10)
        # Historical: low success
        for _ in range(7):
            t.record("a", success=False, quality=0.1)
        # Recent: high success
        for _ in range(3):
            t.record("a", success=True, quality=0.9)
        report = t.improvement_report()
        assert report["a"]["trend"] > 0

    @pytest.mark.skipif(not HAS_EFFECTIVENESS, reason="learning_pipeline not importable")
    def test_report_trend_negative(self):
        """Recent worse than historical => negative trend."""
        t = EffectivenessTracker(recent_window=3, historical_window=10)
        for _ in range(7):
            t.record("a", success=True, quality=0.9)
        for _ in range(3):
            t.record("a", success=False, quality=0.1)
        report = t.improvement_report()
        assert report["a"]["trend"] < 0

    @pytest.mark.skipif(not HAS_EFFECTIVENESS, reason="learning_pipeline not importable")
    def test_report_quality_trend(self):
        t = EffectivenessTracker(recent_window=2, historical_window=10)
        for _ in range(5):
            t.record("a", success=True, quality=0.3)
        for _ in range(2):
            t.record("a", success=True, quality=0.9)
        report = t.improvement_report()
        assert report["a"]["quality_trend"] > 0

    @pytest.mark.skipif(not HAS_EFFECTIVENESS, reason="learning_pipeline not importable")
    def test_report_global_totals(self):
        t = EffectivenessTracker()
        t.record("a", success=True)
        t.record("b", success=True)
        t.record("c", success=False)
        report = t.improvement_report()
        assert report["_global"]["total_episodes"] == 3

    @pytest.mark.skipif(not HAS_EFFECTIVENESS, reason="learning_pipeline not importable")
    def test_report_improving_flag(self):
        """'improving' requires recent > historical AND len(historical) >= 5."""
        t = EffectivenessTracker(recent_window=3, historical_window=10)
        # Historical: low (need >= 5)
        for _ in range(6):
            t.record("a", success=False, quality=0.1)
        # Recent: high
        for _ in range(3):
            t.record("a", success=True, quality=0.9)
        report = t.improvement_report()
        assert report["a"]["improving"] is True


# =============================================================================
# EffectivenessTracker IsImproving Tests
# =============================================================================

@pytest.mark.unit
class TestEffectivenessTrackerIsImproving:
    """Tests for is_improving()."""

    @pytest.mark.skipif(not HAS_EFFECTIVENESS, reason="learning_pipeline not importable")
    def test_not_improving_empty(self):
        t = EffectivenessTracker()
        assert t.is_improving() is False

    @pytest.mark.skipif(not HAS_EFFECTIVENESS, reason="learning_pipeline not importable")
    def test_not_improving_insufficient_historical(self):
        t = EffectivenessTracker(recent_window=3, historical_window=10)
        for _ in range(3):
            t.record("a", success=True)
        assert t.is_improving("a") is False

    @pytest.mark.skipif(not HAS_EFFECTIVENESS, reason="learning_pipeline not importable")
    def test_improving_with_data(self):
        t = EffectivenessTracker(recent_window=3, historical_window=10)
        for _ in range(6):
            t.record("a", success=False)
        for _ in range(3):
            t.record("a", success=True)
        assert t.is_improving("a") is True

    @pytest.mark.skipif(not HAS_EFFECTIVENESS, reason="learning_pipeline not importable")
    def test_not_improving_declining(self):
        t = EffectivenessTracker(recent_window=3, historical_window=10)
        for _ in range(6):
            t.record("a", success=True)
        for _ in range(3):
            t.record("a", success=False)
        assert t.is_improving("a") is False

    @pytest.mark.skipif(not HAS_EFFECTIVENESS, reason="learning_pipeline not importable")
    def test_global_improving(self):
        t = EffectivenessTracker(recent_window=3, historical_window=10)
        for _ in range(6):
            t.record("a", success=False)
        for _ in range(3):
            t.record("a", success=True)
        assert t.is_improving() is True

    @pytest.mark.skipif(not HAS_EFFECTIVENESS, reason="learning_pipeline not importable")
    def test_unknown_task_type_returns_false(self):
        t = EffectivenessTracker()
        assert t.is_improving("nonexistent") is False


# =============================================================================
# EffectivenessTracker Persistence Tests
# =============================================================================

@pytest.mark.unit
class TestEffectivenessTrackerPersistence:
    """Tests for to_dict() and from_dict()."""

    @pytest.mark.skipif(not HAS_EFFECTIVENESS, reason="learning_pipeline not importable")
    def test_to_dict_empty(self):
        t = EffectivenessTracker()
        d = t.to_dict()
        assert d == {}

    @pytest.mark.skipif(not HAS_EFFECTIVENESS, reason="learning_pipeline not importable")
    def test_to_dict_with_data(self):
        t = EffectivenessTracker()
        t.record("analysis", success=True, quality=0.8, agent="agent1")
        d = t.to_dict()
        assert "analysis" in d
        assert len(d["analysis"]) == 1
        entry = d["analysis"][0]
        assert entry["s"] is True
        assert entry["q"] == 0.8
        assert entry["a"] == "agent1"

    @pytest.mark.skipif(not HAS_EFFECTIVENESS, reason="learning_pipeline not importable")
    def test_from_dict(self):
        data = {
            "analysis": [
                {"t": 1000, "s": True, "q": 0.8, "a": "agent1"},
                {"t": 1001, "s": False, "q": 0.2, "a": "agent2"},
            ]
        }
        t = EffectivenessTracker.from_dict(data)
        assert len(t._records["analysis"]) == 2
        assert len(t._global) == 2

    @pytest.mark.skipif(not HAS_EFFECTIVENESS, reason="learning_pipeline not importable")
    def test_roundtrip(self):
        t = EffectivenessTracker()
        t.record("analysis", success=True, quality=0.8, agent="a1")
        t.record("coding", success=False, quality=0.3, agent="a2")
        d = t.to_dict()
        t2 = EffectivenessTracker.from_dict(d)
        d2 = t2.to_dict()
        assert d.keys() == d2.keys()
        for key in d:
            assert len(d[key]) == len(d2[key])


# =============================================================================
# ParadigmExecutor Init Tests
# =============================================================================

@pytest.mark.unit
class TestParadigmExecutorInit:
    """Tests for ParadigmExecutor initialization."""

    @pytest.mark.skipif(not HAS_PARADIGM or not HAS_FOUNDATION, reason="paradigm_executor not importable")
    def test_stores_manager_ref(self):
        mgr = _make_mock_manager()
        pe = ParadigmExecutor(mgr)
        assert pe._manager is mgr

    @pytest.mark.skipif(not HAS_PARADIGM or not HAS_FOUNDATION, reason="paradigm_executor not importable")
    def test_different_managers(self):
        m1 = _make_mock_manager()
        m2 = _make_mock_manager()
        pe1 = ParadigmExecutor(m1)
        pe2 = ParadigmExecutor(m2)
        assert pe1._manager is not pe2._manager

    @pytest.mark.skipif(not HAS_PARADIGM or not HAS_FOUNDATION, reason="paradigm_executor not importable")
    def test_manager_accessible(self):
        mgr = _make_mock_manager(episode_count=5)
        pe = ParadigmExecutor(mgr)
        assert pe._manager.episode_count == 5


# =============================================================================
# ParadigmExecutor RunAgent Tests
# =============================================================================

@pytest.mark.unit
class TestParadigmExecutorRunAgent:
    """Tests for ParadigmExecutor.run_agent() fast path vs full path."""

    @pytest.mark.skipif(not HAS_PARADIGM or not HAS_FOUNDATION, reason="paradigm_executor not importable")
    @pytest.mark.asyncio
    async def test_full_path_with_search_keyword(self):
        """Goals containing tool keywords use the full pipeline."""
        mgr = _make_mock_manager()
        pe = ParadigmExecutor(mgr)
        runner = MagicMock()
        expected = _make_episode_result(output="searched result")
        runner.run = AsyncMock(return_value=expected)
        result = await pe.run_agent(runner, "search for information", "agent1")
        runner.run.assert_awaited_once()
        assert result.output == "searched result"

    @pytest.mark.skipif(not HAS_PARADIGM or not HAS_FOUNDATION, reason="paradigm_executor not importable")
    @pytest.mark.asyncio
    async def test_full_path_with_fetch_keyword(self):
        mgr = _make_mock_manager()
        pe = ParadigmExecutor(mgr)
        runner = MagicMock()
        expected = _make_episode_result(output="fetched")
        runner.run = AsyncMock(return_value=expected)
        result = await pe.run_agent(runner, "fetch the data from API", "agent1")
        runner.run.assert_awaited_once()

    @pytest.mark.skipif(not HAS_PARADIGM or not HAS_FOUNDATION, reason="paradigm_executor not importable")
    @pytest.mark.asyncio
    async def test_full_path_with_email_keyword(self):
        mgr = _make_mock_manager()
        pe = ParadigmExecutor(mgr)
        runner = MagicMock()
        expected = _make_episode_result(output="emailed")
        runner.run = AsyncMock(return_value=expected)
        result = await pe.run_agent(runner, "email the report to the team", "agent1")
        runner.run.assert_awaited_once()

    @pytest.mark.skipif(not HAS_PARADIGM or not HAS_FOUNDATION, reason="paradigm_executor not importable")
    @pytest.mark.asyncio
    async def test_fast_path_no_tool_keywords(self):
        """Simple goal without tool keywords attempts fast path via dspy LM."""
        mgr = _make_mock_manager()
        pe = ParadigmExecutor(mgr)
        runner = MagicMock()
        runner._run_hooks = MagicMock(return_value={"goal": "summarize this text"})

        mock_lm = MagicMock()
        mock_lm.return_value = "fast path response"

        with patch("dspy.settings") as mock_settings:
            mock_settings.lm = mock_lm
            result = await pe.run_agent(runner, "summarize this text", "agent1")
            assert result.success is True
            assert result.output == "fast path response"
            assert "agent1" in result.agent_contributions

    @pytest.mark.skipif(not HAS_PARADIGM or not HAS_FOUNDATION, reason="paradigm_executor not importable")
    @pytest.mark.asyncio
    async def test_fast_path_returns_list(self):
        """When LM returns a list, first element is used."""
        mgr = _make_mock_manager()
        pe = ParadigmExecutor(mgr)
        runner = MagicMock()
        runner._run_hooks = MagicMock(return_value={"goal": "analyze this"})

        mock_lm = MagicMock()
        mock_lm.return_value = ["response one", "response two"]

        with patch("dspy.settings") as mock_settings:
            mock_settings.lm = mock_lm
            result = await pe.run_agent(runner, "analyze this", "agent1")
            assert result.output == "response one"

    @pytest.mark.skipif(not HAS_PARADIGM or not HAS_FOUNDATION, reason="paradigm_executor not importable")
    @pytest.mark.asyncio
    async def test_fast_path_fallback_on_error(self):
        """If fast path fails, falls back to full pipeline."""
        mgr = _make_mock_manager()
        pe = ParadigmExecutor(mgr)
        runner = MagicMock()
        runner._run_hooks = MagicMock(side_effect=RuntimeError("hook failed"))
        expected = _make_episode_result(output="fallback result")
        runner.run = AsyncMock(return_value=expected)

        mock_lm = MagicMock()
        with patch("dspy.settings") as mock_settings:
            mock_settings.lm = mock_lm
            result = await pe.run_agent(runner, "analyze this", "agent1")
            runner.run.assert_awaited_once()
            assert result.output == "fallback result"

    @pytest.mark.skipif(not HAS_PARADIGM or not HAS_FOUNDATION, reason="paradigm_executor not importable")
    @pytest.mark.asyncio
    async def test_fast_path_no_lm_falls_back(self):
        """If no dspy LM configured, falls back to full pipeline."""
        mgr = _make_mock_manager()
        pe = ParadigmExecutor(mgr)
        runner = MagicMock()
        expected = _make_episode_result(output="full pipeline")
        runner.run = AsyncMock(return_value=expected)

        with patch("dspy.settings") as mock_settings:
            mock_settings.lm = None
            result = await pe.run_agent(runner, "analyze this", "agent1")
            runner.run.assert_awaited_once()

    @pytest.mark.skipif(not HAS_PARADIGM or not HAS_FOUNDATION, reason="paradigm_executor not importable")
    @pytest.mark.asyncio
    async def test_full_path_database_keyword(self):
        mgr = _make_mock_manager()
        pe = ParadigmExecutor(mgr)
        runner = MagicMock()
        expected = _make_episode_result(output="db result")
        runner.run = AsyncMock(return_value=expected)
        result = await pe.run_agent(runner, "query the database", "agent1")
        runner.run.assert_awaited_once()


# =============================================================================
# ParadigmExecutor Aggregate Tests
# =============================================================================

@pytest.mark.unit
class TestParadigmExecutorAggregate:
    """Tests for ParadigmExecutor.aggregate_results()."""

    @pytest.mark.skipif(not HAS_PARADIGM or not HAS_FOUNDATION, reason="paradigm_executor not importable")
    def test_aggregate_empty(self):
        mgr = _make_mock_manager()
        pe = ParadigmExecutor(mgr)
        result = pe.aggregate_results({}, "test goal")
        assert result.success is False
        assert result.output is None
        assert "No tasks executed" in result.alerts

    @pytest.mark.skipif(not HAS_PARADIGM or not HAS_FOUNDATION, reason="paradigm_executor not importable")
    def test_aggregate_single_result(self):
        mgr = _make_mock_manager()
        pe = ParadigmExecutor(mgr)
        r = _make_episode_result(output="only one")
        result = pe.aggregate_results({"agent1": r}, "test goal")
        assert result is r

    @pytest.mark.skipif(not HAS_PARADIGM or not HAS_FOUNDATION, reason="paradigm_executor not importable")
    def test_aggregate_multiple_all_success(self):
        mgr = _make_mock_manager()
        pe = ParadigmExecutor(mgr)
        r1 = _make_episode_result(output="out1", success=True)
        r2 = _make_episode_result(output="out2", success=True)
        result = pe.aggregate_results({"a1": r1, "a2": r2}, "goal")
        assert result.success is True

    @pytest.mark.skipif(not HAS_PARADIGM or not HAS_FOUNDATION, reason="paradigm_executor not importable")
    def test_aggregate_multiple_partial_failure(self):
        mgr = _make_mock_manager()
        pe = ParadigmExecutor(mgr)
        r1 = _make_episode_result(output="out1", success=True)
        r2 = _make_episode_result(output="out2", success=False)
        result = pe.aggregate_results({"a1": r1, "a2": r2}, "goal")
        assert result.success is False

    @pytest.mark.skipif(not HAS_PARADIGM or not HAS_FOUNDATION, reason="paradigm_executor not importable")
    def test_aggregate_merges_trajectory(self):
        mgr = _make_mock_manager()
        pe = ParadigmExecutor(mgr)
        r1 = _make_episode_result(trajectory=[{"step": 1}])
        r2 = _make_episode_result(trajectory=[{"step": 2}])
        result = pe.aggregate_results({"a1": r1, "a2": r2}, "goal")
        assert len(result.trajectory) == 2
        agents_in_traj = [s["agent"] for s in result.trajectory]
        assert "a1" in agents_in_traj
        assert "a2" in agents_in_traj

    @pytest.mark.skipif(not HAS_PARADIGM or not HAS_FOUNDATION, reason="paradigm_executor not importable")
    def test_aggregate_merges_contributions(self):
        mgr = _make_mock_manager()
        pe = ParadigmExecutor(mgr)
        r1 = _make_episode_result(agent_contributions={"a1": "contrib1"})
        r2 = _make_episode_result(agent_contributions={"a2": "contrib2"})
        result = pe.aggregate_results({"a1": r1, "a2": r2}, "goal")
        assert "a1" in result.agent_contributions
        assert "a2" in result.agent_contributions


# =============================================================================
# ParadigmExecutor Credit Assignment Tests
# =============================================================================

@pytest.mark.unit
class TestParadigmExecutorCredit:
    """Tests for assign_cooperative_credit()."""

    @pytest.mark.skipif(not HAS_PARADIGM or not HAS_FOUNDATION, reason="paradigm_executor not importable")
    def test_credit_skipped_empty(self):
        mgr = _make_mock_manager()
        pe = ParadigmExecutor(mgr)
        pe.assign_cooperative_credit({}, "goal")
        mgr.learning_manager.record_outcome.assert_not_called()

    @pytest.mark.skipif(not HAS_PARADIGM or not HAS_FOUNDATION, reason="paradigm_executor not importable")
    def test_credit_skipped_single(self):
        mgr = _make_mock_manager()
        pe = ParadigmExecutor(mgr)
        r1 = _make_episode_result()
        pe.assign_cooperative_credit({"a1": r1}, "goal")
        mgr.learning_manager.record_outcome.assert_not_called()

    @pytest.mark.skipif(not HAS_PARADIGM or not HAS_FOUNDATION, reason="paradigm_executor not importable")
    def test_credit_recorded_for_multiple(self):
        mgr = _make_mock_manager()
        pe = ParadigmExecutor(mgr)
        r1 = _make_episode_result(success=True)
        r2 = _make_episode_result(success=True)
        pe.assign_cooperative_credit({"a1": r1, "a2": r2}, "goal")
        assert mgr.learning_manager.record_outcome.call_count == 2

    @pytest.mark.skipif(not HAS_PARADIGM or not HAS_FOUNDATION, reason="paradigm_executor not importable")
    def test_credit_weights_updated_on_success(self):
        mgr = _make_mock_manager()
        pe = ParadigmExecutor(mgr)
        r1 = _make_episode_result(success=True)
        r2 = _make_episode_result(success=True)
        pe.assign_cooperative_credit({"a1": r1, "a2": r2}, "goal")
        mgr.credit_weights.update_from_feedback.assert_called()

    @pytest.mark.skipif(not HAS_PARADIGM or not HAS_FOUNDATION, reason="paradigm_executor not importable")
    def test_credit_weights_updated_on_failure(self):
        mgr = _make_mock_manager()
        pe = ParadigmExecutor(mgr)
        r1 = _make_episode_result(success=False)
        r2 = _make_episode_result(success=False)
        pe.assign_cooperative_credit({"a1": r1, "a2": r2}, "goal")
        mgr.credit_weights.update_from_feedback.assert_called()


# =============================================================================
# OptimizationConfig Tests
# =============================================================================

@pytest.mark.unit
class TestOptimizationConfig:
    """Tests for OptimizationConfig dataclass."""

    @pytest.mark.skipif(not HAS_OPTIMIZATION, reason="optimization_pipeline not importable")
    def test_defaults(self):
        cfg = OptimizationConfig()
        assert cfg.max_iterations == 5
        assert cfg.required_pass_count == 2
        assert cfg.enable_teacher_model is True
        assert cfg.enable_kb_updates is True
        assert cfg.kb_update_requires_teacher is True
        assert cfg.evaluation_function is None
        assert cfg.gold_standard_provider is None
        assert cfg.output_path is None
        assert cfg.enable_thinking_log is True
        assert cfg.save_improvements is True
        assert cfg.enable_credit_assignment is True
        assert cfg.enable_adaptive_learning is True
        assert cfg.min_credit_threshold == 0.1

    @pytest.mark.skipif(not HAS_OPTIMIZATION, reason="optimization_pipeline not importable")
    def test_custom_values(self):
        cfg = OptimizationConfig(
            max_iterations=10,
            required_pass_count=3,
            enable_teacher_model=False,
            enable_kb_updates=False,
            min_credit_threshold=0.5,
        )
        assert cfg.max_iterations == 10
        assert cfg.required_pass_count == 3
        assert cfg.enable_teacher_model is False
        assert cfg.enable_kb_updates is False
        assert cfg.min_credit_threshold == 0.5

    @pytest.mark.skipif(not HAS_OPTIMIZATION, reason="optimization_pipeline not importable")
    def test_callable_fields(self):
        eval_fn = MagicMock()
        gold_fn = MagicMock()
        cfg = OptimizationConfig(
            evaluation_function=eval_fn,
            gold_standard_provider=gold_fn,
        )
        assert cfg.evaluation_function is eval_fn
        assert cfg.gold_standard_provider is gold_fn

    @pytest.mark.skipif(not HAS_OPTIMIZATION, reason="optimization_pipeline not importable")
    def test_path_fields(self):
        cfg = OptimizationConfig(
            output_path=Path("/tmp/test"),
            thinking_log_path=Path("/tmp/thinking.log"),
            improvements_file=Path("/tmp/improvements.json"),
        )
        assert cfg.output_path == Path("/tmp/test")
        assert cfg.thinking_log_path == Path("/tmp/thinking.log")
        assert cfg.improvements_file == Path("/tmp/improvements.json")

    @pytest.mark.skipif(not HAS_OPTIMIZATION, reason="optimization_pipeline not importable")
    def test_dspy_jotty_flags(self):
        cfg = OptimizationConfig(
            update_dspy_instructions=True,
            update_jotty_instructions=True,
        )
        assert cfg.update_dspy_instructions is True
        assert cfg.update_jotty_instructions is True

    @pytest.mark.skipif(not HAS_OPTIMIZATION, reason="optimization_pipeline not importable")
    def test_incremental_learning_flag(self):
        cfg = OptimizationConfig(enable_incremental_learning=False)
        assert cfg.enable_incremental_learning is False


# =============================================================================
# IterationResult Tests
# =============================================================================

@pytest.mark.unit
class TestIterationResult:
    """Tests for IterationResult dataclass."""

    @pytest.mark.skipif(not HAS_OPTIMIZATION, reason="optimization_pipeline not importable")
    def test_creation(self):
        ir = IterationResult(
            iteration=1,
            success=True,
            evaluation_score=1.0,
            evaluation_status="CORRECT",
            output="result",
        )
        assert ir.iteration == 1
        assert ir.success is True
        assert ir.evaluation_score == 1.0
        assert ir.evaluation_status == "CORRECT"
        assert ir.output == "result"

    @pytest.mark.skipif(not HAS_OPTIMIZATION, reason="optimization_pipeline not importable")
    def test_defaults(self):
        ir = IterationResult(
            iteration=0,
            success=False,
            evaluation_score=0.0,
            evaluation_status="ERROR",
            output=None,
        )
        assert ir.metadata == {}
        assert ir.teacher_output is None
        assert ir.kb_updates is None
        assert ir.error is None

    @pytest.mark.skipif(not HAS_OPTIMIZATION, reason="optimization_pipeline not importable")
    def test_with_all_fields(self):
        ir = IterationResult(
            iteration=3,
            success=True,
            evaluation_score=0.95,
            evaluation_status="CORRECT",
            output="final",
            metadata={"key": "val"},
            teacher_output="teacher says",
            kb_updates={"update": True},
            error=None,
        )
        assert ir.metadata == {"key": "val"}
        assert ir.teacher_output == "teacher says"
        assert ir.kb_updates == {"update": True}

    @pytest.mark.skipif(not HAS_OPTIMIZATION, reason="optimization_pipeline not importable")
    def test_with_error(self):
        ir = IterationResult(
            iteration=1,
            success=False,
            evaluation_score=0.0,
            evaluation_status="ERROR",
            output=None,
            error="something broke",
        )
        assert ir.error == "something broke"

    @pytest.mark.skipif(not HAS_OPTIMIZATION, reason="optimization_pipeline not importable")
    def test_metadata_independent(self):
        a = IterationResult(iteration=1, success=True, evaluation_score=1.0, evaluation_status="OK", output="a")
        b = IterationResult(iteration=2, success=True, evaluation_score=1.0, evaluation_status="OK", output="b")
        a.metadata["x"] = 1
        assert "x" not in b.metadata

    @pytest.mark.skipif(not HAS_OPTIMIZATION, reason="optimization_pipeline not importable")
    def test_fields_count(self):
        assert len(fields(IterationResult)) == 9


# =============================================================================
# OptimizationPipeline Init Tests
# =============================================================================

@pytest.mark.unit
class TestOptimizationPipelineInit:
    """Tests for OptimizationPipeline.__init__()."""

    @pytest.mark.skipif(not HAS_OPTIMIZATION or not HAS_FOUNDATION, reason="optimization_pipeline not importable")
    def test_basic_init(self):
        agent = AgentConfig(name="test_agent", agent=MagicMock())
        cfg = OptimizationConfig(enable_thinking_log=False, save_improvements=False)
        pipeline = OptimizationPipeline(agents=[agent], config=cfg)
        assert pipeline.agents == [agent]
        assert pipeline.config is cfg
        assert pipeline.iteration_count == 0
        assert pipeline.consecutive_passes == 0

    @pytest.mark.skipif(not HAS_OPTIMIZATION or not HAS_FOUNDATION, reason="optimization_pipeline not importable")
    def test_with_conductor(self):
        agent = AgentConfig(name="test_agent", agent=MagicMock())
        cfg = OptimizationConfig(enable_thinking_log=False, save_improvements=False)
        conductor = MagicMock()
        pipeline = OptimizationPipeline(agents=[agent], config=cfg, conductor=conductor)
        assert pipeline.conductor is conductor

    @pytest.mark.skipif(not HAS_OPTIMIZATION or not HAS_FOUNDATION, reason="optimization_pipeline not importable")
    def test_jotty_config_default(self):
        agent = AgentConfig(name="test_agent", agent=MagicMock())
        cfg = OptimizationConfig(enable_thinking_log=False, save_improvements=False)
        pipeline = OptimizationPipeline(agents=[agent], config=cfg)
        assert pipeline.jotty_config is not None

    @pytest.mark.skipif(not HAS_OPTIMIZATION or not HAS_FOUNDATION, reason="optimization_pipeline not importable")
    def test_credit_assignment_enabled(self):
        agent = AgentConfig(name="test_agent", agent=MagicMock())
        cfg = OptimizationConfig(
            enable_thinking_log=False,
            save_improvements=False,
            enable_credit_assignment=True,
        )
        pipeline = OptimizationPipeline(agents=[agent], config=cfg)
        assert pipeline.credit_assignment is not None

    @pytest.mark.skipif(not HAS_OPTIMIZATION or not HAS_FOUNDATION, reason="optimization_pipeline not importable")
    def test_adaptive_learning_enabled(self):
        agent = AgentConfig(name="test_agent", agent=MagicMock())
        cfg = OptimizationConfig(
            enable_thinking_log=False,
            save_improvements=False,
            enable_adaptive_learning=True,
        )
        pipeline = OptimizationPipeline(agents=[agent], config=cfg)
        assert pipeline.adaptive_learning is not None


# =============================================================================
# OptimizationPipeline ThinkingLog Tests
# =============================================================================

@pytest.mark.unit
class TestOptimizationPipelineThinkingLog:
    """Tests for _write_thinking_log() and _clear_thinking_log()."""

    @pytest.mark.skipif(not HAS_OPTIMIZATION or not HAS_FOUNDATION, reason="optimization_pipeline not importable")
    def test_write_log_disabled(self):
        agent = AgentConfig(name="test_agent", agent=MagicMock())
        cfg = OptimizationConfig(enable_thinking_log=False, save_improvements=False)
        pipeline = OptimizationPipeline(agents=[agent], config=cfg)
        # Should not raise even without path
        pipeline._write_thinking_log("test message")

    @pytest.mark.skipif(not HAS_OPTIMIZATION or not HAS_FOUNDATION, reason="optimization_pipeline not importable")
    def test_write_log_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "thinking.log"
            agent = AgentConfig(name="test_agent", agent=MagicMock())
            cfg = OptimizationConfig(
                enable_thinking_log=True,
                thinking_log_path=log_path,
                save_improvements=False,
            )
            pipeline = OptimizationPipeline(agents=[agent], config=cfg)
            pipeline._write_thinking_log("hello log")
            content = log_path.read_text()
            assert "hello log" in content

    @pytest.mark.skipif(not HAS_OPTIMIZATION or not HAS_FOUNDATION, reason="optimization_pipeline not importable")
    def test_clear_log(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "thinking.log"
            log_path.write_text("old content")
            agent = AgentConfig(name="test_agent", agent=MagicMock())
            cfg = OptimizationConfig(
                enable_thinking_log=True,
                thinking_log_path=log_path,
                save_improvements=False,
            )
            pipeline = OptimizationPipeline(agents=[agent], config=cfg)
            # __init__ already calls _clear_thinking_log
            content = log_path.read_text()
            assert content == ""

    @pytest.mark.skipif(not HAS_OPTIMIZATION or not HAS_FOUNDATION, reason="optimization_pipeline not importable")
    def test_write_log_timestamp_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "thinking.log"
            agent = AgentConfig(name="test_agent", agent=MagicMock())
            cfg = OptimizationConfig(
                enable_thinking_log=True,
                thinking_log_path=log_path,
                save_improvements=False,
            )
            pipeline = OptimizationPipeline(agents=[agent], config=cfg)
            pipeline._write_thinking_log("timestamped")
            content = log_path.read_text()
            # Should have [YYYY-MM-DD HH:MM:SS.mmm] format
            assert "[20" in content
            assert "timestamped" in content

    @pytest.mark.skipif(not HAS_OPTIMIZATION or not HAS_FOUNDATION, reason="optimization_pipeline not importable")
    def test_write_log_no_path(self):
        agent = AgentConfig(name="test_agent", agent=MagicMock())
        cfg = OptimizationConfig(
            enable_thinking_log=True,
            save_improvements=False,
        )
        pipeline = OptimizationPipeline(agents=[agent], config=cfg)
        pipeline.thinking_log_path = None
        # Should not raise
        pipeline._write_thinking_log("ignored")


# =============================================================================
# OptimizationPipeline Optimize Tests
# =============================================================================

@pytest.mark.unit
class TestOptimizationPipelineOptimize:
    """Tests for OptimizationPipeline.optimize()."""

    @pytest.mark.skipif(not HAS_OPTIMIZATION or not HAS_FOUNDATION, reason="optimization_pipeline not importable")
    @pytest.mark.asyncio
    async def test_optimize_immediate_success(self):
        """Pipeline completes when evaluation passes required_pass_count times."""
        mock_agent = MagicMock()
        mock_agent.return_value = "correct output"
        agent_cfg = AgentConfig(name="worker", agent=mock_agent)

        async def eval_fn(output, gold_standard, task, context):
            return {"score": 1.0, "status": "CORRECT"}

        cfg = OptimizationConfig(
            max_iterations=5,
            required_pass_count=2,
            enable_thinking_log=False,
            save_improvements=False,
            enable_teacher_model=False,
            enable_kb_updates=False,
            evaluation_function=eval_fn,
        )
        pipeline = OptimizationPipeline(agents=[agent_cfg], config=cfg)
        result = await pipeline.optimize(task="test task", gold_standard="expected")
        assert result["optimization_complete"] is True
        assert result["consecutive_passes"] >= 2

    @pytest.mark.skipif(not HAS_OPTIMIZATION or not HAS_FOUNDATION, reason="optimization_pipeline not importable")
    @pytest.mark.asyncio
    async def test_optimize_max_iterations_reached(self):
        """Pipeline stops after max_iterations even without success."""
        mock_agent = MagicMock()
        mock_agent.return_value = "wrong output"
        agent_cfg = AgentConfig(name="worker", agent=mock_agent)

        async def eval_fn(output, gold_standard, task, context):
            return {"score": 0.0, "status": "INCORRECT"}

        cfg = OptimizationConfig(
            max_iterations=3,
            required_pass_count=2,
            enable_thinking_log=False,
            save_improvements=False,
            enable_teacher_model=False,
            enable_kb_updates=False,
            evaluation_function=eval_fn,
        )
        pipeline = OptimizationPipeline(agents=[agent_cfg], config=cfg)
        result = await pipeline.optimize(task="test task", gold_standard="expected")
        assert result["optimization_complete"] is False
        assert result["total_iterations"] == 3

    @pytest.mark.skipif(not HAS_OPTIMIZATION or not HAS_FOUNDATION, reason="optimization_pipeline not importable")
    @pytest.mark.asyncio
    async def test_optimize_error_handling(self):
        """Pipeline handles agent errors gracefully."""
        mock_agent = MagicMock(spec=[])  # spec=[] removes auto-generated forward
        mock_agent.side_effect = RuntimeError("agent broke")
        agent_cfg = AgentConfig(name="worker", agent=mock_agent)

        cfg = OptimizationConfig(
            max_iterations=2,
            enable_thinking_log=False,
            save_improvements=False,
            enable_teacher_model=False,
            enable_kb_updates=False,
        )
        pipeline = OptimizationPipeline(agents=[agent_cfg], config=cfg)
        result = await pipeline.optimize(task="test task")
        assert result["optimization_complete"] is False
        # Iterations should have error entries
        errors = [it for it in result["iterations"] if it["error"] is not None]
        assert len(errors) > 0

    @pytest.mark.skipif(not HAS_OPTIMIZATION or not HAS_FOUNDATION, reason="optimization_pipeline not importable")
    @pytest.mark.asyncio
    async def test_optimize_returns_best_result(self):
        """Final result should be the best iteration."""
        call_count = [0]
        mock_agent = MagicMock()
        mock_agent.return_value = "output"
        agent_cfg = AgentConfig(name="worker", agent=mock_agent)

        async def eval_fn(output, gold_standard, task, context):
            call_count[0] += 1
            if call_count[0] >= 3:
                return {"score": 1.0, "status": "CORRECT"}
            return {"score": 0.5, "status": "INCORRECT"}

        cfg = OptimizationConfig(
            max_iterations=5,
            required_pass_count=2,
            enable_thinking_log=False,
            save_improvements=False,
            enable_teacher_model=False,
            enable_kb_updates=False,
            evaluation_function=eval_fn,
        )
        pipeline = OptimizationPipeline(agents=[agent_cfg], config=cfg)
        result = await pipeline.optimize(task="test task", gold_standard="expected")
        assert result["final_result"] is not None
        assert result["final_result"]["evaluation_score"] is not None

    @pytest.mark.skipif(not HAS_OPTIMIZATION or not HAS_FOUNDATION, reason="optimization_pipeline not importable")
    @pytest.mark.asyncio
    async def test_optimize_with_conductor(self):
        """Pipeline uses conductor when available."""
        conductor = MagicMock()
        episode_result = _make_episode_result(output="conductor output")
        conductor.arun = AsyncMock(return_value=episode_result)

        agent_cfg = AgentConfig(name="worker", agent=MagicMock())

        async def eval_fn(output, gold_standard, task, context):
            return {"score": 1.0, "status": "CORRECT"}

        cfg = OptimizationConfig(
            max_iterations=5,
            required_pass_count=1,
            enable_thinking_log=False,
            save_improvements=False,
            enable_teacher_model=False,
            enable_kb_updates=False,
            evaluation_function=eval_fn,
        )
        pipeline = OptimizationPipeline(
            agents=[agent_cfg], config=cfg, conductor=conductor,
        )
        result = await pipeline.optimize(task="test task", gold_standard="expected")
        assert result["optimization_complete"] is True
        conductor.arun.assert_awaited()

    @pytest.mark.skipif(not HAS_OPTIMIZATION or not HAS_FOUNDATION, reason="optimization_pipeline not importable")
    @pytest.mark.asyncio
    async def test_optimize_default_evaluation(self):
        """Without custom eval function, uses simple string comparison."""
        mock_agent = MagicMock(spec=[])  # spec=[] prevents auto-generated forward
        mock_agent.return_value = "expected output"
        agent_cfg = AgentConfig(name="worker", agent=mock_agent)

        cfg = OptimizationConfig(
            max_iterations=3,
            required_pass_count=1,
            enable_thinking_log=False,
            save_improvements=False,
            enable_teacher_model=False,
            enable_kb_updates=False,
        )
        pipeline = OptimizationPipeline(agents=[agent_cfg], config=cfg)
        result = await pipeline.optimize(task="test", gold_standard="expected output")
        assert result["optimization_complete"] is True

    @pytest.mark.skipif(not HAS_OPTIMIZATION or not HAS_FOUNDATION, reason="optimization_pipeline not importable")
    @pytest.mark.asyncio
    async def test_optimize_iterations_structure(self):
        """Each iteration in results has expected keys."""
        mock_agent = MagicMock()
        mock_agent.return_value = "output"
        agent_cfg = AgentConfig(name="worker", agent=mock_agent)

        async def eval_fn(output, gold_standard, task, context):
            return {"score": 1.0, "status": "CORRECT"}

        cfg = OptimizationConfig(
            max_iterations=3,
            required_pass_count=1,
            enable_thinking_log=False,
            save_improvements=False,
            enable_teacher_model=False,
            enable_kb_updates=False,
            evaluation_function=eval_fn,
        )
        pipeline = OptimizationPipeline(agents=[agent_cfg], config=cfg)
        result = await pipeline.optimize(task="test", gold_standard="gold")
        for it in result["iterations"]:
            assert "iteration" in it
            assert "success" in it
            assert "evaluation_score" in it
            assert "evaluation_status" in it

    @pytest.mark.skipif(not HAS_OPTIMIZATION or not HAS_FOUNDATION, reason="optimization_pipeline not importable")
    @pytest.mark.asyncio
    async def test_optimize_consecutive_passes_reset_on_failure(self):
        """Consecutive passes reset when evaluation fails."""
        call_count = [0]
        mock_agent = MagicMock()
        mock_agent.return_value = "output"
        agent_cfg = AgentConfig(name="worker", agent=mock_agent)

        async def eval_fn(output, gold_standard, task, context):
            call_count[0] += 1
            # Pass, fail, pass, pass
            if call_count[0] in (1, 3, 4):
                return {"score": 1.0, "status": "CORRECT"}
            return {"score": 0.0, "status": "INCORRECT"}

        cfg = OptimizationConfig(
            max_iterations=5,
            required_pass_count=2,
            enable_thinking_log=False,
            save_improvements=False,
            enable_teacher_model=False,
            enable_kb_updates=False,
            evaluation_function=eval_fn,
        )
        pipeline = OptimizationPipeline(agents=[agent_cfg], config=cfg)
        result = await pipeline.optimize(task="test", gold_standard="gold")
        # Should complete after 4 iterations (pass, fail, pass, pass)
        assert result["optimization_complete"] is True
        assert result["total_iterations"] == 4

    @pytest.mark.skipif(not HAS_OPTIMIZATION or not HAS_FOUNDATION, reason="optimization_pipeline not importable")
    def test_get_best_result_empty(self):
        """_get_best_result with no iterations returns None values."""
        agent_cfg = AgentConfig(name="worker", agent=MagicMock())
        cfg = OptimizationConfig(
            enable_thinking_log=False,
            save_improvements=False,
        )
        pipeline = OptimizationPipeline(agents=[agent_cfg], config=cfg)
        best = pipeline._get_best_result()
        assert best["iteration"] is None
        assert best["output"] is None

    @pytest.mark.skipif(not HAS_OPTIMIZATION or not HAS_FOUNDATION, reason="optimization_pipeline not importable")
    def test_get_previous_outputs_empty(self):
        agent_cfg = AgentConfig(name="worker", agent=MagicMock())
        cfg = OptimizationConfig(
            enable_thinking_log=False,
            save_improvements=False,
        )
        pipeline = OptimizationPipeline(agents=[agent_cfg], config=cfg)
        outputs = pipeline._get_previous_outputs()
        assert outputs == {}


# =============================================================================
# OptimizationPipeline Improvement Recording Tests
# =============================================================================

@pytest.mark.unit
class TestOptimizationPipelineImprovements:
    """Tests for improvement recording in OptimizationPipeline."""

    @pytest.mark.skipif(not HAS_OPTIMIZATION or not HAS_FOUNDATION, reason="optimization_pipeline not importable")
    def test_record_improvement_disabled(self):
        agent_cfg = AgentConfig(name="worker", agent=MagicMock())
        cfg = OptimizationConfig(
            enable_thinking_log=False,
            save_improvements=False,
        )
        pipeline = OptimizationPipeline(agents=[agent_cfg], config=cfg)
        result = pipeline._record_improvement(
            iteration=1,
            student_output="student",
            teacher_output="teacher",
            task="task",
            evaluation_result={"score": 0.5},
        )
        assert result is None

    @pytest.mark.skipif(not HAS_OPTIMIZATION or not HAS_FOUNDATION, reason="optimization_pipeline not importable")
    def test_record_improvement_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            imp_file = Path(tmpdir) / "improvements.json"
            agent_cfg = AgentConfig(name="worker", agent=MagicMock())
            cfg = OptimizationConfig(
                enable_thinking_log=False,
                save_improvements=True,
                improvements_file=imp_file,
            )
            pipeline = OptimizationPipeline(agents=[agent_cfg], config=cfg)
            result = pipeline._record_improvement(
                iteration=1,
                student_output="student output",
                teacher_output="teacher output",
                task="my task",
                evaluation_result={"score": 0.5},
            )
            assert result is not None
            assert result["iteration"] == 1
            assert result["task"] == "my task"
            assert len(pipeline.improvements) == 1

    @pytest.mark.skipif(not HAS_OPTIMIZATION or not HAS_FOUNDATION, reason="optimization_pipeline not importable")
    def test_extract_learned_pattern(self):
        agent_cfg = AgentConfig(name="worker", agent=MagicMock())
        cfg = OptimizationConfig(
            enable_thinking_log=False,
            save_improvements=False,
        )
        pipeline = OptimizationPipeline(agents=[agent_cfg], config=cfg)
        pattern = pipeline._extract_learned_pattern(
            student_output="wrong answer",
            teacher_output="correct answer",
            task="solve math",
        )
        assert isinstance(pattern, str)
        assert len(pattern) > 0

    @pytest.mark.skipif(not HAS_OPTIMIZATION or not HAS_FOUNDATION, reason="optimization_pipeline not importable")
    def test_validate_teacher_output_empty(self):
        agent_cfg = AgentConfig(name="worker", agent=MagicMock())
        cfg = OptimizationConfig(
            enable_thinking_log=False,
            save_improvements=False,
        )
        pipeline = OptimizationPipeline(agents=[agent_cfg], config=cfg)
        result = pipeline._validate_teacher_output(
            teacher_output="",
            student_output="student",
            gold_standard="gold",
            evaluation_result={"score": 0.5},
        )
        assert result is None

    @pytest.mark.skipif(not HAS_OPTIMIZATION or not HAS_FOUNDATION, reason="optimization_pipeline not importable")
    def test_validate_teacher_output_same_as_student(self):
        agent_cfg = AgentConfig(name="worker", agent=MagicMock())
        cfg = OptimizationConfig(
            enable_thinking_log=False,
            save_improvements=False,
        )
        pipeline = OptimizationPipeline(agents=[agent_cfg], config=cfg)
        result = pipeline._validate_teacher_output(
            teacher_output="same output",
            student_output="same output",
            gold_standard="gold",
            evaluation_result={"score": 0.5},
        )
        assert result is None

    @pytest.mark.skipif(not HAS_OPTIMIZATION or not HAS_FOUNDATION, reason="optimization_pipeline not importable")
    def test_validate_teacher_output_good(self):
        agent_cfg = AgentConfig(name="worker", agent=MagicMock())
        cfg = OptimizationConfig(
            enable_thinking_log=False,
            save_improvements=False,
        )
        pipeline = OptimizationPipeline(agents=[agent_cfg], config=cfg)
        result = pipeline._validate_teacher_output(
            teacher_output="improved output here",
            student_output="original output here",
            gold_standard="improved output here is expected",
            evaluation_result={"score": 0.5},
        )
        assert result is not None
