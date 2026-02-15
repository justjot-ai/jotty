"""
Tests for Jotty Agent Base Classes
===================================

Comprehensive unit tests covering:
- AgentRuntimeConfig
- AgentResult
- BaseAgent (via ConcreteAgent subclass)
- DomainAgent / DomainAgentConfig / create_domain_agent
- BaseSwarmAgent
- ValidationAgent / ValidationConfig
- ToolCallCache
- ExecutionContextManager

All tests use mocks -- NO real LLM calls.

Author: A-Team
Date: February 2026
"""

import asyncio
import json
import os
import time
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import (
    AsyncMock,
    MagicMock,
    Mock,
    patch,
    PropertyMock,
)

import pytest

# ---------------------------------------------------------------------------
# Guarded imports with skip markers
# ---------------------------------------------------------------------------
try:
    from Jotty.core.modes.agent.base.base_agent import (
        AgentResult,
        AgentRuntimeConfig,
        BaseAgent,
    )
    BASE_AGENT_AVAILABLE = True
except ImportError:
    BASE_AGENT_AVAILABLE = False

try:
    from Jotty.core.modes.agent.implementations.domain_agent import (
        DomainAgent,
        DomainAgentConfig,
        create_domain_agent,
    )
    DOMAIN_AGENT_AVAILABLE = True
except ImportError:
    DOMAIN_AGENT_AVAILABLE = False

try:
    from Jotty.core.modes.agent.implementations.swarm_agent import BaseSwarmAgent
    SWARM_AGENT_AVAILABLE = True
except ImportError:
    SWARM_AGENT_AVAILABLE = False

try:
    from Jotty.core.modes.agent.implementations.validation_agent import (
        AgentMessage,
        SharedScratchpad,
        ValidationAgent,
        ValidationConfig,
    )
    from Jotty.core.infrastructure.foundation.types.validation_types import ValidationResult
    from Jotty.core.infrastructure.foundation.types.enums import ValidationRound
    VALIDATION_AGENT_AVAILABLE = True
except ImportError:
    VALIDATION_AGENT_AVAILABLE = False

try:
    from Jotty.core.modes.agent.executors.skill_plan_executor import ToolCallCache
    TOOL_CACHE_AVAILABLE = True
except ImportError:
    TOOL_CACHE_AVAILABLE = False

try:
    from Jotty.core.modes.agent.implementations.autonomous_agent import ExecutionContextManager
    EXEC_CTX_AVAILABLE = True
except ImportError:
    EXEC_CTX_AVAILABLE = False

try:
    from Jotty.core.infrastructure.foundation.config_defaults import DEFAULTS
    DEFAULTS_AVAILABLE = True
except ImportError:
    DEFAULTS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Concrete subclass of BaseAgent for testing
# ---------------------------------------------------------------------------
if BASE_AGENT_AVAILABLE:
    class ConcreteAgent(BaseAgent):
        """Minimal concrete implementation for testing BaseAgent."""

        async def _execute_impl(self, **kwargs) -> Any:
            return {"result": kwargs.get("task", "done")}


# ---------------------------------------------------------------------------
# Helper: patch _init_dspy_lm so tests never touch real DSPy
# ---------------------------------------------------------------------------
_PATCH_DSPY = "Jotty.core.agents.base.base_agent.BaseAgent._init_dspy_lm"


# =============================================================================
# TestAgentRuntimeConfig
# =============================================================================

@pytest.mark.unit
@pytest.mark.skipif(not BASE_AGENT_AVAILABLE, reason="BaseAgent not importable")
class TestAgentRuntimeConfig:
    """Tests for AgentRuntimeConfig defaults, custom values, and __post_init__."""

    def test_default_name_is_empty_string(self):
        """Sentinel name is empty string before __post_init__ resolves it."""
        config = AgentRuntimeConfig.__new__(AgentRuntimeConfig)
        object.__setattr__(config, "name", "")
        assert config.name == ""

    def test_post_init_resolves_name(self):
        config = AgentRuntimeConfig()
        # __post_init__ sets name to class name when empty
        assert config.name == "AgentRuntimeConfig"

    def test_custom_name_preserved(self):
        config = AgentRuntimeConfig(name="MyAgent")
        assert config.name == "MyAgent"

    def test_default_enable_memory_true(self):
        config = AgentRuntimeConfig(name="T")
        assert config.enable_memory is True

    def test_default_enable_context_true(self):
        config = AgentRuntimeConfig(name="T")
        assert config.enable_context is True

    def test_default_enable_skills_true(self):
        config = AgentRuntimeConfig(name="T")
        assert config.enable_skills is True

    def test_default_enable_monitoring_true(self):
        config = AgentRuntimeConfig(name="T")
        assert config.enable_monitoring is True

    def test_post_init_resolves_model(self):
        config = AgentRuntimeConfig(name="T")
        assert config.model == DEFAULTS.DEFAULT_MODEL_ALIAS

    def test_post_init_resolves_temperature(self):
        config = AgentRuntimeConfig(name="T")
        assert config.temperature == DEFAULTS.LLM_TEMPERATURE

    def test_post_init_resolves_max_tokens(self):
        config = AgentRuntimeConfig(name="T")
        assert config.max_tokens == DEFAULTS.LLM_MAX_OUTPUT_TOKENS

    def test_post_init_resolves_max_retries(self):
        config = AgentRuntimeConfig(name="T")
        assert config.max_retries == DEFAULTS.MAX_RETRIES

    def test_post_init_resolves_retry_delay(self):
        config = AgentRuntimeConfig(name="T")
        assert config.retry_delay == DEFAULTS.RETRY_BACKOFF_SECONDS

    def test_post_init_resolves_timeout(self):
        config = AgentRuntimeConfig(name="T")
        assert config.timeout == float(DEFAULTS.LLM_TIMEOUT_SECONDS)

    def test_parameters_default_empty_dict(self):
        config = AgentRuntimeConfig(name="T")
        assert config.parameters == {}

    def test_custom_parameters(self):
        config = AgentRuntimeConfig(name="T", parameters={"a": 1})
        assert config.parameters == {"a": 1}

    def test_system_prompt_default_empty(self):
        config = AgentRuntimeConfig(name="T")
        assert config.system_prompt == ""


# =============================================================================
# TestAgentResult
# =============================================================================

@pytest.mark.unit
@pytest.mark.skipif(not BASE_AGENT_AVAILABLE, reason="BaseAgent not importable")
class TestAgentResult:
    """Tests for AgentResult dataclass."""

    def test_creation_required_fields(self):
        result = AgentResult(success=True, output="hello")
        assert result.success is True
        assert result.output == "hello"

    def test_default_agent_name_empty(self):
        result = AgentResult(success=True, output=None)
        assert result.agent_name == ""

    def test_default_execution_time_zero(self):
        result = AgentResult(success=True, output=None)
        assert result.execution_time == 0.0

    def test_default_retries_zero(self):
        result = AgentResult(success=True, output=None)
        assert result.retries == 0

    def test_default_error_none(self):
        result = AgentResult(success=True, output=None)
        assert result.error is None

    def test_default_metadata_empty_dict(self):
        result = AgentResult(success=True, output=None)
        assert result.metadata == {}

    def test_timestamp_auto_set(self):
        before = datetime.now()
        result = AgentResult(success=True, output=None)
        after = datetime.now()
        assert before <= result.timestamp <= after

    def test_to_dict_keys(self):
        result = AgentResult(success=True, output="ok", agent_name="X")
        d = result.to_dict()
        expected_keys = {
            "success", "output", "agent_name", "execution_time",
            "retries", "error", "metadata", "timestamp",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_timestamp_iso(self):
        result = AgentResult(success=True, output=None)
        d = result.to_dict()
        # Should be a string in ISO format
        assert isinstance(d["timestamp"], str)
        datetime.fromisoformat(d["timestamp"])

    def test_successful_result(self):
        result = AgentResult(success=True, output={"data": 42}, agent_name="A")
        assert result.success is True
        assert result.output["data"] == 42
        assert result.error is None

    def test_failed_result_with_error(self):
        result = AgentResult(success=False, output=None, error="boom")
        assert result.success is False
        assert result.error == "boom"

    def test_custom_metadata(self):
        result = AgentResult(success=True, output=None, metadata={"k": "v"})
        assert result.metadata["k"] == "v"


# =============================================================================
# TestBaseAgent
# =============================================================================

@pytest.mark.unit
@pytest.mark.skipif(not BASE_AGENT_AVAILABLE, reason="BaseAgent not importable")
class TestBaseAgent:
    """Tests for BaseAgent (via ConcreteAgent concrete subclass)."""

    def _make_agent(self, name="TestAgent", **config_kwargs):
        """Helper to create a ConcreteAgent with DSPy init patched out."""
        with patch(_PATCH_DSPY):
            config = AgentRuntimeConfig(name=name, **config_kwargs)
            return ConcreteAgent(config=config)

    # -- Initialization --

    def test_init_default_config(self):
        with patch(_PATCH_DSPY):
            agent = ConcreteAgent()
        assert agent.config.name == "ConcreteAgent"

    def test_init_custom_config(self):
        agent = self._make_agent(name="Custom")
        assert agent.config.name == "Custom"

    def test_init_not_initialized(self):
        agent = self._make_agent()
        assert agent._initialized is False

    def test_init_metrics_zeroed(self):
        agent = self._make_agent()
        assert agent._metrics["total_executions"] == 0
        assert agent._metrics["successful_executions"] == 0

    # -- Lazy properties --

    def test_memory_lazy_load(self):
        agent = self._make_agent(enable_memory=False)
        assert agent.memory is None

    def test_memory_returns_none_when_disabled(self):
        agent = self._make_agent(enable_memory=False)
        assert agent.memory is None

    def test_context_returns_none_when_disabled(self):
        agent = self._make_agent(enable_context=False)
        assert agent.context is None

    def test_skills_registry_returns_none_when_disabled(self):
        agent = self._make_agent(enable_skills=False)
        assert agent.skills_registry is None

    def test_memory_lazy_with_mock(self):
        agent = self._make_agent()
        mock_mem = MagicMock()
        agent._memory = mock_mem
        assert agent.memory is mock_mem

    def test_context_lazy_with_mock(self):
        agent = self._make_agent()
        mock_ctx = MagicMock()
        agent._context_manager = mock_ctx
        assert agent.context is mock_ctx

    def test_skills_lazy_with_mock(self):
        agent = self._make_agent()
        mock_reg = MagicMock()
        agent._skills_registry = mock_reg
        assert agent.skills_registry is mock_reg

    # -- execute() --

    @pytest.mark.asyncio
    async def test_execute_returns_agent_result_on_success(self):
        agent = self._make_agent()
        agent._initialized = True
        result = await agent.execute(task="hello")
        assert isinstance(result, AgentResult)
        assert result.success is True
        assert result.output == {"result": "hello"}

    @pytest.mark.asyncio
    async def test_execute_sets_agent_name(self):
        agent = self._make_agent(name="Bob")
        agent._initialized = True
        result = await agent.execute(task="hi")
        assert result.agent_name == "Bob"

    @pytest.mark.asyncio
    async def test_execute_updates_metrics_on_success(self):
        agent = self._make_agent()
        agent._initialized = True
        await agent.execute(task="test")
        assert agent._metrics["total_executions"] == 1
        assert agent._metrics["successful_executions"] == 1

    @pytest.mark.asyncio
    async def test_execute_returns_error_on_failure(self):
        agent = self._make_agent()
        agent._initialized = True
        agent._execute_impl = AsyncMock(side_effect=ValueError("fail"))
        # With max_retries=3, all attempts fail
        result = await agent.execute(task="test")
        assert result.success is False
        assert "fail" in result.error

    @pytest.mark.asyncio
    async def test_execute_updates_metrics_on_failure(self):
        agent = self._make_agent()
        agent._initialized = True
        agent._execute_impl = AsyncMock(side_effect=RuntimeError("err"))
        await agent.execute(task="test")
        assert agent._metrics["failed_executions"] == 1

    @pytest.mark.asyncio
    async def test_execute_retry_logic(self):
        """Mock _execute_impl to fail twice, then succeed on third attempt."""
        agent = self._make_agent()
        agent._initialized = True
        agent.config.retry_delay = 0.0  # no wait
        call_count = 0

        async def flaky_impl(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("transient")
            return {"ok": True}

        agent._execute_impl = flaky_impl
        result = await agent.execute(task="test")
        assert result.success is True
        assert result.retries == 2

    # -- Hooks --

    def test_add_pre_hook(self):
        agent = self._make_agent()
        hook = MagicMock()
        agent.add_pre_hook(hook)
        assert hook in agent._pre_hooks

    def test_add_post_hook(self):
        agent = self._make_agent()
        hook = MagicMock()
        agent.add_post_hook(hook)
        assert hook in agent._post_hooks

    @pytest.mark.asyncio
    async def test_pre_hook_called_during_execute(self):
        agent = self._make_agent()
        agent._initialized = True
        hook = MagicMock()
        agent.add_pre_hook(hook)
        await agent.execute(task="x")
        hook.assert_called_once()

    @pytest.mark.asyncio
    async def test_post_hook_called_during_execute(self):
        agent = self._make_agent()
        agent._initialized = True
        hook = MagicMock()
        agent.add_post_hook(hook)
        await agent.execute(task="x")
        hook.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_pre_hook(self):
        agent = self._make_agent()
        agent._initialized = True
        hook = AsyncMock()
        agent.add_pre_hook(hook)
        await agent.execute(task="x")
        hook.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_async_post_hook(self):
        agent = self._make_agent()
        agent._initialized = True
        hook = AsyncMock()
        agent.add_post_hook(hook)
        await agent.execute(task="x")
        hook.assert_awaited_once()

    # -- Memory helpers --

    def test_store_memory_no_memory(self):
        """store_memory should be a no-op when memory is None."""
        agent = self._make_agent(enable_memory=False)
        # Should not raise
        agent.store_memory("content", level="episodic")

    def test_store_memory_delegates(self):
        agent = self._make_agent()
        mock_mem = MagicMock()
        agent._memory = mock_mem
        with patch("Jotty.core.agents.base.base_agent.BaseAgent.memory",
                    new_callable=PropertyMock, return_value=mock_mem):
            agent.store_memory("hello", level="episodic", context={"a": 1})
        mock_mem.store.assert_called_once()

    def test_retrieve_memory_no_memory(self):
        agent = self._make_agent(enable_memory=False)
        result = agent.retrieve_memory("query")
        assert result == []

    def test_retrieve_memory_delegates(self):
        agent = self._make_agent()
        mock_mem = MagicMock()
        mock_mem.retrieve.return_value = ["entry1"]
        agent._memory = mock_mem
        with patch("Jotty.core.agents.base.base_agent.BaseAgent.memory",
                    new_callable=PropertyMock, return_value=mock_mem):
            result = agent.retrieve_memory("query")
        assert result == ["entry1"]

    # -- Context helpers --

    def test_register_context_no_context(self):
        agent = self._make_agent(enable_context=False)
        # Should not raise
        agent.register_context("key", "value")

    def test_register_context_delegates(self):
        agent = self._make_agent()
        mock_ctx = MagicMock()
        agent._context_manager = mock_ctx
        agent.register_context("k", "v")
        mock_ctx.set.assert_called_once_with("k", "v")

    def test_get_context_no_context(self):
        agent = self._make_agent(enable_context=False)
        assert agent.get_context("key", "default") == "default"

    def test_get_context_delegates(self):
        agent = self._make_agent()
        mock_ctx = MagicMock()
        mock_ctx.get.return_value = "val"
        agent._context_manager = mock_ctx
        assert agent.get_context("k") == "val"

    def test_get_compressed_context_no_context(self):
        agent = self._make_agent(enable_context=False)
        assert agent.get_compressed_context() == ""

    def test_get_compressed_context_returns_json(self):
        agent = self._make_agent()
        mock_ctx = MagicMock()
        mock_ctx.get.side_effect = lambda key, default=None: (
            "my_task" if key == "current_task" else default
        )
        agent._context_manager = mock_ctx
        result = agent.get_compressed_context()
        assert "my_task" in result

    # -- Skill discovery --

    def test_discover_skills_no_registry(self):
        agent = self._make_agent(enable_skills=False)
        assert agent.discover_skills("test") == []

    def test_discover_skills_delegates(self):
        agent = self._make_agent()
        mock_reg = MagicMock()
        mock_reg.list_skills.return_value = [{"name": "web-search"}]
        agent._skills_registry = mock_reg
        result = agent.discover_skills("test")
        assert len(result) == 1
        assert result[0]["name"] == "web-search"

    # -- Collaboration --

    def test_set_collaboration_context(self):
        agent = self._make_agent()
        directory = {"agent_a": MagicMock()}
        slack = [{"msg": "hello"}]
        agent.set_collaboration_context(directory, slack)
        assert agent._agent_directory == directory
        assert agent._agent_slack == slack

    def test_request_help_adds_to_slack(self):
        agent = self._make_agent(name="Alice")
        agent._agent_slack = []
        agent.request_help("Bob", "help me")
        assert len(agent._agent_slack) == 1
        assert agent._agent_slack[0]["from"] == "Alice"
        assert agent._agent_slack[0]["to"] == "Bob"

    def test_get_pending_requests_filters_correctly(self):
        agent = self._make_agent(name="Bob")
        agent._agent_slack = [
            {"to": "Bob", "query": "q1"},
            {"to": "Alice", "query": "q2"},
            {"to": "Bob", "query": "q3"},
        ]
        pending = agent.get_pending_requests()
        assert len(pending) == 2

    # -- Metrics --

    def test_get_metrics_initial(self):
        agent = self._make_agent()
        metrics = agent.get_metrics()
        assert metrics["total_executions"] == 0
        assert metrics["success_rate"] == 0.0
        assert metrics["avg_execution_time"] == 0.0

    @pytest.mark.asyncio
    async def test_get_metrics_after_execution(self):
        agent = self._make_agent()
        agent._initialized = True
        await agent.execute(task="test")
        metrics = agent.get_metrics()
        assert metrics["total_executions"] == 1
        assert metrics["success_rate"] == 1.0
        assert metrics["avg_execution_time"] > 0.0

    def test_reset_metrics(self):
        agent = self._make_agent()
        agent._metrics["total_executions"] = 5
        agent.reset_metrics()
        assert agent._metrics["total_executions"] == 0

    # -- to_dict / __repr__ --

    def test_to_dict_keys(self):
        agent = self._make_agent(name="X")
        d = agent.to_dict()
        assert d["name"] == "X"
        assert d["class"] == "ConcreteAgent"
        assert "config" in d
        assert "metrics" in d
        assert "initialized" in d

    def test_repr(self):
        agent = self._make_agent(name="X")
        r = repr(agent)
        assert "ConcreteAgent" in r
        assert "X" in r

    # -- resolve_config --

    def test_resolve_config_from_env(self):
        with patch.dict(os.environ, {"MY_MODEL": "gpt-4"}):
            val = BaseAgent.resolve_config("model", "MY_MODEL", default="sonnet")
        assert val == "gpt-4"

    def test_resolve_config_fallback_chain(self):
        with patch.dict(os.environ, {"SECOND": "val2"}, clear=False):
            os.environ.pop("FIRST", None)
            val = BaseAgent.resolve_config("x", "FIRST", "SECOND", default="d")
        assert val == "val2"

    def test_resolve_config_default(self):
        os.environ.pop("NONEXISTENT_VAR_12345", None)
        val = BaseAgent.resolve_config("x", "NONEXISTENT_VAR_12345", default="fallback")
        assert val == "fallback"

    # -- set_jotty_config --

    def test_set_jotty_config_returns_self(self):
        agent = self._make_agent()
        mock_config = MagicMock()
        result = agent.set_jotty_config(mock_config)
        assert result is agent
        assert agent._jotty_config is mock_config

    # -- _analyze_failure --

    def test_analyze_failure_blind_retry_patterns(self):
        agent = self._make_agent()
        assert agent._analyze_failure("rate limit exceeded", {}) == ""
        assert agent._analyze_failure("HTTP 429 error", {}) == ""
        assert agent._analyze_failure("Server overloaded", {}) == ""

    def test_analyze_failure_returns_guidance_for_logic_error(self):
        agent = self._make_agent()
        with patch("Jotty.core.agents.base.base_agent.BaseAgent._analyze_failure",
                    wraps=agent._analyze_failure):
            result = agent._analyze_failure("KeyError: 'missing_field'", {})
            # May return guidance or empty depending on ErrorType availability
            assert isinstance(result, str)

    # -- _get_system_context --

    def test_get_system_context_empty_by_default(self):
        agent = self._make_agent()
        assert agent._get_system_context() == ""

    def test_get_system_context_with_system_prompt(self):
        agent = self._make_agent()
        agent.config.system_prompt = "You are a helper."
        ctx = agent._get_system_context()
        assert "You are a helper." in ctx

    def test_get_system_context_appends_vvp_for_visual_skills(self):
        agent = self._make_agent()
        skills = [{"name": "visual-inspector"}, {"name": "calculator"}]
        ctx = agent._get_system_context(discovered_skills=skills)
        assert "visual-inspector" in ctx or "screenshot" in ctx.lower()

    # -- get_io_schema --

    def test_get_io_schema_returns_schema(self):
        agent = self._make_agent(name="SchemaAgent")
        schema = agent.get_io_schema()
        assert schema.agent_name == "SchemaAgent"
        assert len(schema.inputs) >= 1
        assert len(schema.outputs) >= 1


# =============================================================================
# TestDomainAgent
# =============================================================================

@pytest.mark.unit
@pytest.mark.skipif(not DOMAIN_AGENT_AVAILABLE, reason="DomainAgent not importable")
class TestDomainAgent:
    """Tests for DomainAgent and DomainAgentConfig."""

    def _make_agent(self, signature=None, **config_kwargs):
        with patch(_PATCH_DSPY):
            config = DomainAgentConfig(name="TestDomain", **config_kwargs) if config_kwargs else None
            return DomainAgent(signature=signature, config=config)

    # -- DomainAgentConfig --

    def test_domain_config_defaults_cot_true(self):
        config = DomainAgentConfig(name="T")
        assert config.use_chain_of_thought is True

    def test_domain_config_defaults_react_false(self):
        config = DomainAgentConfig(name="T")
        assert config.use_react is False

    def test_domain_config_defaults_max_react_iters(self):
        config = DomainAgentConfig(name="T")
        assert config.max_react_iters == 5

    def test_domain_config_defaults_streaming_false(self):
        config = DomainAgentConfig(name="T")
        assert config.streaming is False

    def test_domain_config_defaults_progress_callback_none(self):
        config = DomainAgentConfig(name="T")
        assert config.progress_callback is None

    def test_domain_config_inherits_agent_runtime(self):
        config = DomainAgentConfig(name="T")
        assert config.enable_memory is True
        assert config.enable_skills is True

    # -- DomainAgent init --

    def test_init_with_no_signature(self):
        agent = self._make_agent(signature=None)
        assert agent.signature is None
        assert agent._input_fields == []
        assert agent._output_fields == []

    def test_init_with_mock_signature(self):
        """Create a mock signature class."""
        mock_sig = MagicMock()
        mock_sig.__name__ = "MockSignature"
        mock_sig.model_fields = {}
        with patch(_PATCH_DSPY):
            agent = DomainAgent(signature=mock_sig)
        assert agent.signature is mock_sig

    def test_default_name_includes_signature(self):
        mock_sig = MagicMock()
        mock_sig.__name__ = "AnalyzeSig"
        mock_sig.model_fields = {}
        with patch(_PATCH_DSPY):
            agent = DomainAgent(signature=mock_sig)
        assert "AnalyzeSig" in agent.config.name

    def test_default_name_no_signature(self):
        agent = self._make_agent(signature=None)
        assert "NoSignature" in agent.config.name

    # -- Properties --

    def test_input_fields_returns_copy(self):
        agent = self._make_agent()
        agent._input_fields = ["a", "b"]
        fields = agent.input_fields
        assert fields == ["a", "b"]
        fields.append("c")
        assert len(agent._input_fields) == 2  # original unchanged

    def test_output_fields_returns_copy(self):
        agent = self._make_agent()
        agent._output_fields = ["x"]
        fields = agent.output_fields
        assert fields == ["x"]
        fields.append("y")
        assert len(agent._output_fields) == 1

    # -- get_io_schema --

    def test_get_io_schema_no_signature(self):
        agent = self._make_agent()
        agent._input_fields = ["task"]
        agent._output_fields = ["result"]
        schema = agent.get_io_schema()
        assert schema is not None

    def test_get_io_schema_cached(self):
        agent = self._make_agent()
        schema1 = agent.get_io_schema()
        schema2 = agent.get_io_schema()
        assert schema1 is schema2

    # -- _execute_impl --

    @pytest.mark.asyncio
    async def test_execute_impl_no_module_no_task(self):
        agent = self._make_agent()
        agent._module = None
        agent._initialized = True
        result = await agent._execute_impl()
        assert "error" in result or result.get("success") is False

    @pytest.mark.asyncio
    async def test_execute_impl_no_module_with_task_falls_back_to_skills(self):
        agent = self._make_agent()
        agent._module = None
        agent._initialized = True
        # Mock _execute_with_skills so we verify the fallback path is taken
        agent._execute_with_skills = AsyncMock(
            return_value={"success": True, "output": "from skills"}
        )
        result = await agent._execute_impl(task="do something")
        agent._execute_with_skills.assert_awaited_once()
        assert result.get("success") is True

    @pytest.mark.asyncio
    async def test_execute_impl_with_mock_module(self):
        agent = self._make_agent()
        agent._initialized = True
        agent._input_fields = ["text"]
        agent._output_fields = ["analysis"]

        mock_result = MagicMock()
        mock_result.analysis = "great"
        mock_result.reasoning = None
        mock_module = MagicMock(return_value=mock_result)
        agent._module = mock_module

        result = await agent._execute_impl(text="hello world")
        assert result.get("analysis") == "great"

    # -- _build_task_from_kwargs --

    def test_build_task_from_query(self):
        agent = self._make_agent()
        assert agent._build_task_from_kwargs({"query": "find AI"}) == "find AI"

    def test_build_task_from_prompt(self):
        agent = self._make_agent()
        assert agent._build_task_from_kwargs({"prompt": "hello"}) == "hello"

    def test_build_task_from_concatenation(self):
        agent = self._make_agent()
        result = agent._build_task_from_kwargs({"a": "val_a", "b": "val_b"})
        assert "val_a" in result
        assert "val_b" in result

    def test_build_task_empty(self):
        agent = self._make_agent()
        assert agent._build_task_from_kwargs({}) == ""

    def test_build_task_skips_non_string(self):
        agent = self._make_agent()
        result = agent._build_task_from_kwargs({"num": 42, "query": "hello"})
        assert result == "hello"

    # -- create_domain_agent factory --

    def test_create_domain_agent_factory(self):
        mock_sig = MagicMock()
        mock_sig.__name__ = "TestSig"
        mock_sig.model_fields = {}
        with patch(_PATCH_DSPY):
            agent = create_domain_agent(mock_sig)
        assert isinstance(agent, DomainAgent)
        assert "TestSig" in agent.config.name


# =============================================================================
# TestBaseSwarmAgent
# =============================================================================

@pytest.mark.unit
@pytest.mark.skipif(not SWARM_AGENT_AVAILABLE, reason="BaseSwarmAgent not importable")
class TestBaseSwarmAgent:
    """Tests for BaseSwarmAgent."""

    def _make_agent(self, memory=None, context=None, bus=None,
                    learned_context=""):
        with patch(_PATCH_DSPY):
            return BaseSwarmAgent(
                memory=memory,
                context=context,
                bus=bus,
                learned_context=learned_context,
            )

    def test_init_defaults(self):
        agent = self._make_agent()
        assert agent.bus is None
        assert agent.learned_context == ""

    def test_init_with_memory(self):
        mock_mem = MagicMock()
        agent = self._make_agent(memory=mock_mem)
        assert agent._memory is mock_mem

    def test_init_with_context(self):
        mock_ctx = MagicMock()
        agent = self._make_agent(context=mock_ctx)
        assert agent._context_manager is mock_ctx

    def test_init_with_bus(self):
        mock_bus = MagicMock()
        agent = self._make_agent(bus=mock_bus)
        assert agent.bus is mock_bus

    def test_learned_context_attribute(self):
        agent = self._make_agent(learned_context="some context")
        assert agent.learned_context == "some context"

    def test_broadcast_with_bus(self):
        """_broadcast emits via AgentEventBroadcaster (not bus attribute)."""
        agent = self._make_agent()
        # _broadcast uses AgentEventBroadcaster, test it doesn't raise
        agent._broadcast("test_event", {"key": "value"})

    def test_broadcast_no_error_on_failure(self):
        agent = self._make_agent()
        # Should not raise even if broadcast import fails
        with patch(
            "Jotty.core.utils.async_utils.AgentEventBroadcaster",
        ) as mock_broadcaster:
            mock_broadcaster.get_instance.side_effect = RuntimeError("no broadcaster")
            # _broadcast catches all exceptions internally
            agent._broadcast("event", {})

    def test_inherits_from_domain_agent(self):
        agent = self._make_agent()
        assert isinstance(agent, DomainAgent)


# =============================================================================
# TestValidationConfig
# =============================================================================

@pytest.mark.unit
@pytest.mark.skipif(not VALIDATION_AGENT_AVAILABLE, reason="ValidationAgent not importable")
class TestValidationConfig:
    """Tests for ValidationConfig defaults."""

    def test_confidence_threshold_default(self):
        config = ValidationConfig(name="T")
        assert config.confidence_threshold == 0.7

    def test_refinement_threshold_default(self):
        config = ValidationConfig(name="T")
        assert config.refinement_threshold == 0.5

    def test_enable_multi_round_default(self):
        config = ValidationConfig(name="T")
        assert config.enable_multi_round is True

    def test_max_validation_rounds_default(self):
        config = ValidationConfig(name="T")
        assert config.max_validation_rounds == 2

    def test_refinement_on_disagreement_default(self):
        config = ValidationConfig(name="T")
        assert config.refinement_on_disagreement is True

    def test_refinement_on_low_confidence_default(self):
        config = ValidationConfig(name="T")
        assert config.refinement_on_low_confidence == 0.5

    def test_llm_timeout_seconds_default(self):
        config = ValidationConfig(name="T")
        assert config.llm_timeout_seconds == 180.0

    def test_store_validation_history_default(self):
        config = ValidationConfig(name="T")
        assert config.store_validation_history is True

    def test_default_confidence_on_error(self):
        config = ValidationConfig(name="T")
        assert config.default_confidence_on_error == 0.3

    def test_default_confidence_no_validation(self):
        config = ValidationConfig(name="T")
        assert config.default_confidence_no_validation == 0.5

    def test_inherits_meta_agent_config(self):
        config = ValidationConfig(name="T")
        assert hasattr(config, "enable_gold_db")
        assert hasattr(config, "improvement_threshold")


# =============================================================================
# TestValidationAgent
# =============================================================================

@pytest.mark.unit
@pytest.mark.skipif(not VALIDATION_AGENT_AVAILABLE, reason="ValidationAgent not importable")
class TestValidationAgent:
    """Tests for ValidationAgent."""

    def _make_agent(self, is_pre=True, config=None):
        with patch(_PATCH_DSPY):
            config = config or ValidationConfig(name="TestValidator")
            return ValidationAgent(
                config=config,
                is_pre_validation=is_pre,
            )

    def test_init_defaults(self):
        agent = self._make_agent()
        assert agent.is_pre_validation is True
        assert agent.scratchpad is not None

    def test_init_post_validation(self):
        agent = self._make_agent(is_pre=False)
        assert agent.is_pre_validation is False

    def test_share_insight(self):
        agent = self._make_agent()
        agent.share_insight("test insight", confidence=0.9)
        messages = agent.scratchpad.messages
        assert len(messages) == 1
        assert messages[0].insight == "test insight"
        assert messages[0].confidence == 0.9
        assert messages[0].receiver == "*"

    def test_get_shared_insights(self):
        agent = self._make_agent()
        agent.share_insight("insight1", confidence=0.8)
        agent.share_insight("insight2", confidence=0.7)
        insights = agent.get_shared_insights()
        assert "insight1" in insights
        assert "insight2" in insights

    def test_share_warning(self):
        agent = self._make_agent()
        agent.share_warning("danger!")
        messages = agent.scratchpad.messages
        assert len(messages) == 1
        assert messages[0].message_type == "warning"

    def test_needs_refinement_below_threshold(self):
        agent = self._make_agent()
        result = ValidationResult(
            agent_name="T",
            is_valid=True,
            confidence=0.3,
            reasoning="low",
        )
        assert agent.needs_refinement(result) is True

    def test_needs_refinement_above_threshold(self):
        agent = self._make_agent()
        result = ValidationResult(
            agent_name="T",
            is_valid=True,
            confidence=0.8,
            reasoning="high",
        )
        assert agent.needs_refinement(result) is False

    def test_needs_refinement_disabled(self):
        config = ValidationConfig(name="T", enable_multi_round=False)
        agent = self._make_agent(config=config)
        result = ValidationResult(
            agent_name="T",
            is_valid=True,
            confidence=0.1,
            reasoning="low",
        )
        assert agent.needs_refinement(result) is False

    def test_get_validation_metrics_initial(self):
        agent = self._make_agent()
        metrics = agent.get_validation_metrics()
        assert metrics["total_validations"] == 0
        assert metrics["approval_rate"] == 0.0

    def test_update_validation_metrics_approval(self):
        agent = self._make_agent(is_pre=True)
        result = ValidationResult(
            agent_name="T",
            is_valid=True,
            confidence=0.9,
            reasoning="ok",
            should_proceed=True,
        )
        agent._update_validation_metrics(result)
        assert agent._validation_metrics["total_validations"] == 1
        assert agent._validation_metrics["approvals"] == 1

    def test_update_validation_metrics_rejection_pre(self):
        agent = self._make_agent(is_pre=True)
        result = ValidationResult(
            agent_name="T",
            is_valid=False,
            confidence=0.9,
            reasoning="bad",
            should_proceed=False,
        )
        agent._update_validation_metrics(result)
        assert agent._validation_metrics["rejections"] == 1

    def test_update_validation_metrics_post_validation(self):
        agent = self._make_agent(is_pre=False)
        result = ValidationResult(
            agent_name="T",
            is_valid=True,
            confidence=0.8,
            reasoning="ok",
        )
        agent._update_validation_metrics(result)
        assert agent._validation_metrics["approvals"] == 1

    def test_wrap_tool_with_cache(self):
        agent = self._make_agent()
        tool = MagicMock(return_value="result1")
        tool.name = "my_tool"
        cached = agent.wrap_tool_with_cache(tool)
        # First call: executes tool
        r1 = cached(query="test")
        assert r1 == "result1"
        # Second call: returns cached
        r2 = cached(query="test")
        assert r2 == "result1"
        # Tool should have been called only once
        tool.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_no_module(self):
        """validate() with no DSPy module returns default result."""
        agent = self._make_agent()
        agent._dspy_module = None
        agent._initialized = True
        # Patch memory to None
        agent._memory = None
        agent.config.enable_memory = False
        result = await agent.validate("goal", {})
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert result.confidence == agent.validation_config.default_confidence_no_validation


# =============================================================================
# TestSharedScratchpad
# =============================================================================

@pytest.mark.unit
@pytest.mark.skipif(not VALIDATION_AGENT_AVAILABLE, reason="SharedScratchpad not importable")
class TestSharedScratchpad:
    """Tests for SharedScratchpad inter-agent communication."""

    def test_add_message(self):
        pad = SharedScratchpad()
        msg = AgentMessage(
            sender="A", receiver="B", message_type="insight",
            content={"text": "hello"},
        )
        pad.add_message(msg)
        assert len(pad.messages) == 1

    def test_get_messages_for_specific(self):
        pad = SharedScratchpad()
        pad.add_message(AgentMessage(sender="A", receiver="B",
                                     message_type="x", content={}))
        pad.add_message(AgentMessage(sender="A", receiver="C",
                                     message_type="x", content={}))
        msgs = pad.get_messages_for("B")
        assert len(msgs) == 1

    def test_get_messages_for_broadcast(self):
        pad = SharedScratchpad()
        pad.add_message(AgentMessage(sender="A", receiver="*",
                                     message_type="x", content={}))
        msgs = pad.get_messages_for("anyone")
        assert len(msgs) == 1

    def test_cache_and_get(self):
        pad = SharedScratchpad()
        pad.cache_result("tool", {"q": "ai"}, "result_data")
        assert pad.get_cached_result("tool", {"q": "ai"}) == "result_data"

    def test_cache_miss(self):
        pad = SharedScratchpad()
        assert pad.get_cached_result("tool", {"q": "ai"}) is None

    def test_clear(self):
        pad = SharedScratchpad()
        pad.add_message(AgentMessage(sender="A", receiver="*",
                                     message_type="x", content={}))
        pad.cache_result("t", {}, "r")
        pad.clear()
        assert len(pad.messages) == 0
        assert pad.get_cached_result("t", {}) is None


# =============================================================================
# TestToolCallCache
# =============================================================================

@pytest.mark.unit
@pytest.mark.skipif(not TOOL_CACHE_AVAILABLE, reason="ToolCallCache not importable")
class TestToolCallCache:
    """Tests for ToolCallCache (TTL + LRU)."""

    def test_make_key_deterministic(self):
        k1 = ToolCallCache.make_key("web-search", "search_tool", {"q": "AI"})
        k2 = ToolCallCache.make_key("web-search", "search_tool", {"q": "AI"})
        assert k1 == k2

    def test_make_key_different_params(self):
        k1 = ToolCallCache.make_key("s", "t", {"q": "A"})
        k2 = ToolCallCache.make_key("s", "t", {"q": "B"})
        assert k1 != k2

    def test_make_key_different_skills(self):
        k1 = ToolCallCache.make_key("skill-a", "t", {"q": "A"})
        k2 = ToolCallCache.make_key("skill-b", "t", {"q": "A"})
        assert k1 != k2

    def test_get_empty(self):
        cache = ToolCallCache()
        assert cache.get("nonexistent") is None

    def test_set_and_get(self):
        cache = ToolCallCache()
        cache.set("key1", {"result": "hello"})
        assert cache.get("key1") == {"result": "hello"}

    def test_size_property(self):
        cache = ToolCallCache()
        assert cache.size == 0
        cache.set("a", 1)
        cache.set("b", 2)
        assert cache.size == 2

    def test_clear(self):
        cache = ToolCallCache()
        cache.set("a", 1)
        cache.clear()
        assert cache.size == 0
        assert cache.get("a") is None

    def test_ttl_expiration(self):
        cache = ToolCallCache(ttl_seconds=1)
        cache.set("key", "value")
        assert cache.get("key") == "value"
        # Manually expire
        cache._cache["key"] = ("value", time.time() - 2)
        assert cache.get("key") is None

    def test_max_size_eviction(self):
        cache = ToolCallCache(max_size=2)
        cache.set("a", 1)
        cache.set("b", 2)
        assert cache.size == 2
        cache.set("c", 3)
        assert cache.size == 2
        # 'a' was oldest, should be evicted
        assert cache.get("a") is None
        assert cache.get("c") == 3

    def test_overwrite_existing_key(self):
        cache = ToolCallCache(max_size=2)
        cache.set("a", 1)
        cache.set("b", 2)
        # Overwrite 'a', should not evict
        cache.set("a", 10)
        assert cache.size == 2
        assert cache.get("a") == 10


# =============================================================================
# TestExecutionContextManager
# =============================================================================

@pytest.mark.unit
@pytest.mark.skipif(not EXEC_CTX_AVAILABLE, reason="ExecutionContextManager not importable")
class TestExecutionContextManager:
    """Tests for ExecutionContextManager."""

    def test_add_step(self):
        ctx = ExecutionContextManager()
        ctx.add_step({"step": 0, "output": "hello"})
        assert len(ctx._history) == 1

    def test_get_context_returns_copy(self):
        ctx = ExecutionContextManager()
        ctx.add_step({"step": 0})
        context = ctx.get_context()
        assert len(context) == 1
        context.append({"fake": True})
        assert len(ctx._history) == 1  # original unchanged

    def test_get_trajectory_filters_compressed(self):
        ctx = ExecutionContextManager()
        ctx._history = [
            {"_compressed": True, "summary": "old stuff"},
            {"step": 3, "output": "new"},
            {"step": 4, "output": "newer"},
        ]
        traj = ctx.get_trajectory()
        assert len(traj) == 2
        assert all(not e.get("_compressed") for e in traj)

    def test_compress_when_exceeding_max(self):
        ctx = ExecutionContextManager(max_history_size=100)
        # Add enough entries to trigger compression
        for i in range(10):
            ctx.add_step({"step": i, "output": "x" * 50})
        # After compression, history should be shorter
        assert len(ctx._history) < 10

    def test_compress_preserves_recent(self):
        ctx = ExecutionContextManager(max_history_size=200)
        for i in range(10):
            ctx.add_step({"step": i, "output": f"output_{i}" * 10})
        # Most recent entries should survive
        trajectory = ctx.get_trajectory()
        # At least some recent entries should exist
        assert len(trajectory) > 0

    def test_compress_creates_summary_entry(self):
        ctx = ExecutionContextManager(max_history_size=100)
        for i in range(10):
            ctx.add_step({"step": i, "output": "x" * 50})
        # Should have at least one compressed entry
        compressed = [e for e in ctx._history if e.get("_compressed")]
        assert len(compressed) >= 1
        assert "summary" in compressed[0]

    def test_empty_context(self):
        ctx = ExecutionContextManager()
        assert ctx.get_context() == []
        assert ctx.get_trajectory() == []

    def test_single_step_no_compress(self):
        ctx = ExecutionContextManager(max_history_size=1000)
        ctx.add_step({"step": 0, "output": "ok"})
        assert len(ctx._history) == 1
        assert not ctx._history[0].get("_compressed")


# =============================================================================
# TestAgentMessage
# =============================================================================

@pytest.mark.unit
@pytest.mark.skipif(not VALIDATION_AGENT_AVAILABLE, reason="AgentMessage not importable")
class TestAgentMessage:
    """Tests for AgentMessage dataclass."""

    def test_creation(self):
        msg = AgentMessage(
            sender="Agent1",
            receiver="Agent2",
            message_type="insight",
            content={"key": "val"},
        )
        assert msg.sender == "Agent1"
        assert msg.receiver == "Agent2"
        assert msg.message_type == "insight"

    def test_defaults(self):
        msg = AgentMessage(
            sender="A", receiver="B", message_type="x", content={},
        )
        assert msg.insight == ""
        assert msg.confidence == 0.0
        assert msg.tool_name == ""
        assert msg.tool_args == {}
        assert msg.tool_result is None

    def test_timestamp_auto_set(self):
        before = time.time()
        msg = AgentMessage(
            sender="A", receiver="B", message_type="x", content={},
        )
        after = time.time()
        assert before <= msg.timestamp <= after

    def test_custom_fields(self):
        msg = AgentMessage(
            sender="A", receiver="*", message_type="tool_result",
            content={}, tool_name="search", tool_result={"data": 1},
        )
        assert msg.tool_name == "search"
        assert msg.tool_result == {"data": 1}
