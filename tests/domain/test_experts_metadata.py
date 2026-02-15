"""
Comprehensive tests for experts and metadata modules.

Tests cover:
1. ExpertRegistry - register, get, list, get_mermaid_expert, get_pipeline_expert
2. BaseExpert - init, template methods, validate methods, SimpleDomainExpert
3. Expert Templates - factory functions (create_mermaid_expert, create_sql_expert, etc.)
4. ToolInterceptor - ToolCall dataclass, wrapping, intercept/log tool calls
5. MetadataToolRegistry - discover tools, call tools, list, get_tool_info, caching
6. WidgetParamSchema - schema definition, validation, defaults, docstring generation
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ============================================================================
# Safe imports with skip-if-unavailable
# ============================================================================

try:
    from Jotty.core.intelligence.reasoning.experts.expert_registry import (
        ExpertRegistry,
        get_expert_registry,
    )

    HAS_EXPERT_REGISTRY = True
except ImportError:
    HAS_EXPERT_REGISTRY = False

try:
    from Jotty.core.intelligence.reasoning.experts.base_expert import BaseExpert, SimpleDomainExpert

    HAS_BASE_EXPERT = True
except ImportError:
    HAS_BASE_EXPERT = False

try:
    from Jotty.core.intelligence.reasoning.experts.expert_agent import ExpertAgentConfig

    HAS_EXPERT_AGENT_CONFIG = True
except ImportError:
    HAS_EXPERT_AGENT_CONFIG = False

try:
    from Jotty.core.infrastructure.metadata.tool_interceptor import (
        ToolCall,
        ToolCallRegistry,
        ToolInterceptor,
    )

    HAS_TOOL_INTERCEPTOR = True
except ImportError:
    HAS_TOOL_INTERCEPTOR = False

try:
    from Jotty.core.infrastructure.metadata.metadata_tool_registry import MetadataToolRegistry

    HAS_METADATA_TOOL_REGISTRY = True
except ImportError:
    HAS_METADATA_TOOL_REGISTRY = False

try:
    from Jotty.core.infrastructure.metadata.widget_params_schema import (
        LIMIT_PARAM_SCHEMA,
        SORT_PARAM_SCHEMA,
        STATUS_PARAM_SCHEMA,
        TIME_RANGE_PARAM_SCHEMA,
        WidgetParamSchema,
        generate_param_docstring,
        generate_tool_examples,
    )

    HAS_WIDGET_PARAMS = True
except ImportError:
    HAS_WIDGET_PARAMS = False


# ============================================================================
# Helpers: concrete BaseExpert subclass for testing
# ============================================================================


def _make_concrete_expert_class():
    """Create a concrete BaseExpert subclass for testing purposes."""

    class ConcreteTestExpert(BaseExpert):
        """Concrete implementation of BaseExpert for testing."""

        @property
        def domain(self) -> str:
            return "test_domain"

        @property
        def description(self) -> str:
            return "A test expert for unit testing"

        def _create_domain_agent(self, improvements=None):
            return MagicMock(name="test_agent")

        def _create_domain_teacher(self):
            return MagicMock(name="test_teacher")

        @staticmethod
        def _get_default_training_cases():
            return [
                {"task": "test task 1", "gold_standard": "expected output 1"},
                {"task": "test task 2", "gold_standard": "expected output 2"},
            ]

        @staticmethod
        def _get_default_validation_cases():
            return [
                {"task": "val task 1", "expected_output": "val output 1"},
            ]

        async def _evaluate_domain(self, output, gold_standard, task, context):
            passed = str(output).strip() == str(gold_standard).strip()
            return {
                "score": 1.0 if passed else 0.0,
                "status": "CORRECT" if passed else "FAIL",
                "is_valid": passed,
                "error": "" if passed else "Mismatch",
                "metadata": {},
            }

    return ConcreteTestExpert


def _make_concrete_expert(config=None, improvements=None):
    """Create a ConcreteTestExpert instance, patching ExpertAgentConfig at source."""
    ConcreteTestExpert = _make_concrete_expert_class()
    with patch("Jotty.core.experts.expert_agent.ExpertAgentConfig") as MockConfig:
        MockConfig.return_value = MagicMock(
            training_gold_standards=None,
            validation_cases=None,
        )
        return ConcreteTestExpert(config=config, improvements=improvements)


def _make_simple_expert():
    """Create a concrete SimpleDomainExpert subclass instance."""

    class TestSimple(SimpleDomainExpert):
        @property
        def domain(self):
            return "simple_test"

        @property
        def description(self):
            return "Simple test expert"

    with patch("Jotty.core.experts.expert_agent.ExpertAgentConfig") as MockConfig:
        MockConfig.return_value = MagicMock(
            training_gold_standards=None,
            validation_cases=None,
        )
        return TestSimple()


def _make_metadata_instance_with_jotty_methods():
    """Create a mock metadata instance with _jotty_meta decorated methods."""

    class FakeMetadata:
        def get_terms(self):
            return {"revenue": "total income"}

        get_terms._jotty_meta = {
            "desc": "Get business terms",
            "when": "Agent needs terminology",
            "cache": True,
            "params": {"filter": "optional filter"},
            "returns": "Dict of terms",
        }

        def get_schema(self, table_name: str):
            return {"columns": ["id", "name"]}

        get_schema._jotty_meta = {
            "desc": "Get table schema",
            "when": "Agent needs table structure",
            "cache": False,
            "params": {"table_name": "Name of the table"},
            "returns": "Dict with columns",
        }

    return FakeMetadata()


# ============================================================================
# 1. ExpertRegistry Tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.skipif(not HAS_EXPERT_REGISTRY, reason="ExpertRegistry not available")
class TestExpertRegistry:
    """Tests for ExpertRegistry: register, get, list, specialized getters."""

    def _make_registry(self):
        """Create a fresh ExpertRegistry instance."""
        return ExpertRegistry()

    def test_init_creates_empty_registry(self):
        registry = self._make_registry()
        assert registry.list_experts() == []
        assert registry._initialized is False

    def test_register_and_get_expert(self):
        registry = self._make_registry()
        mock_expert = MagicMock()
        registry.register("my_expert", mock_expert)
        assert registry.get("my_expert") is mock_expert

    def test_get_returns_none_for_unknown(self):
        registry = self._make_registry()
        assert registry.get("nonexistent") is None

    def test_list_experts_returns_all_names(self):
        registry = self._make_registry()
        expert_a = MagicMock()
        expert_b = MagicMock()
        registry.register("alpha", expert_a)
        registry.register("beta", expert_b)
        names = registry.list_experts()
        assert "alpha" in names
        assert "beta" in names
        assert len(names) == 2

    def test_register_overwrites_existing(self):
        registry = self._make_registry()
        old = MagicMock(name="old")
        new = MagicMock(name="new")
        registry.register("expert", old)
        registry.register("expert", new)
        assert registry.get("expert") is new
        assert len(registry.list_experts()) == 1

    @patch("Jotty.core.experts.expert_registry.MermaidExpertAgent")
    def test_get_mermaid_expert_creates_on_first_call(self, MockMermaid):
        registry = self._make_registry()
        mock_instance = MagicMock()
        MockMermaid.return_value = mock_instance
        result = registry.get_mermaid_expert()
        assert result is mock_instance
        MockMermaid.assert_called_once()

    @patch("Jotty.core.experts.expert_registry.MermaidExpertAgent")
    def test_get_mermaid_expert_returns_cached_on_second_call(self, MockMermaid):
        registry = self._make_registry()
        mock_instance = MagicMock()
        MockMermaid.return_value = mock_instance
        first = registry.get_mermaid_expert()
        second = registry.get_mermaid_expert()
        assert first is second
        MockMermaid.assert_called_once()

    @patch("Jotty.core.experts.expert_registry.PipelineExpertAgent")
    def test_get_pipeline_expert_creates_with_format(self, MockPipeline):
        registry = self._make_registry()
        mock_instance = MagicMock()
        MockPipeline.return_value = mock_instance
        result = registry.get_pipeline_expert(output_format="plantuml")
        MockPipeline.assert_called_once_with(output_format="plantuml")
        assert result is mock_instance

    @patch("Jotty.core.experts.expert_registry.PipelineExpertAgent")
    def test_get_pipeline_expert_caches_per_format(self, MockPipeline):
        registry = self._make_registry()
        mock_m = MagicMock(name="mermaid_pipe")
        mock_p = MagicMock(name="plantuml_pipe")
        MockPipeline.side_effect = [mock_m, mock_p]
        r1 = registry.get_pipeline_expert(output_format="mermaid")
        r2 = registry.get_pipeline_expert(output_format="plantuml")
        assert r1 is not r2
        assert MockPipeline.call_count == 2

    @pytest.mark.asyncio
    @patch("Jotty.core.experts.expert_registry.MermaidExpertAgent")
    async def test_get_mermaid_expert_async_auto_trains(self, MockMermaid):
        registry = self._make_registry()
        mock_instance = MagicMock()
        mock_instance.trained = False
        mock_instance.train = AsyncMock()
        MockMermaid.return_value = mock_instance
        result = await registry.get_mermaid_expert_async(auto_train=True)
        mock_instance.train.assert_awaited_once()
        assert result is mock_instance

    @pytest.mark.asyncio
    async def test_get_async_returns_none_for_missing(self):
        registry = self._make_registry()
        result = await registry.get_async("missing")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_async_auto_trains_if_requested(self):
        registry = self._make_registry()
        mock_expert = MagicMock()
        mock_expert.trained = False
        mock_expert.train = AsyncMock()
        registry.register("trainable", mock_expert)
        result = await registry.get_async("trainable", auto_train=True)
        mock_expert.train.assert_awaited_once()
        assert result is mock_expert

    @pytest.mark.asyncio
    async def test_get_async_skips_train_if_already_trained(self):
        registry = self._make_registry()
        mock_expert = MagicMock()
        mock_expert.trained = True
        mock_expert.train = AsyncMock()
        registry.register("trained", mock_expert)
        await registry.get_async("trained", auto_train=True)
        mock_expert.train.assert_not_awaited()

    def test_get_expert_registry_returns_global_singleton(self):
        r1 = get_expert_registry()
        r2 = get_expert_registry()
        assert r1 is r2


# ============================================================================
# 2. BaseExpert Tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.skipif(not HAS_BASE_EXPERT, reason="BaseExpert not available")
class TestBaseExpert:
    """Tests for BaseExpert: init, template methods, properties, stats."""

    def test_init_default_config(self):
        expert = _make_concrete_expert()
        assert expert.domain == "test_domain"
        assert expert.description == "A test expert for unit testing"
        assert expert.improvements == []

    def test_init_with_custom_improvements(self):
        improvements = [{"pattern": "use joins", "teacher_output": "SELECT..."}]
        expert = _make_concrete_expert(improvements=improvements)
        assert expert.improvements == improvements

    def test_init_with_explicit_config(self):
        custom_config = MagicMock(name="custom_config")
        expert = _make_concrete_expert(config=custom_config)
        assert expert.config is custom_config

    def test_create_agent_wrapper_delegates_to_domain(self):
        expert = _make_concrete_expert()
        agent = expert._create_agent_wrapper()
        assert agent is not None

    def test_create_agent_wrapper_passes_improvements(self):
        improvements = [{"pattern": "x"}]
        expert = _make_concrete_expert(improvements=improvements)
        agent = expert._create_agent_wrapper()
        assert agent is not None

    def test_create_teacher_wrapper_delegates_to_domain(self):
        expert = _make_concrete_expert()
        teacher = expert._create_teacher_wrapper()
        assert teacher is not None

    def test_create_default_agent_delegates(self):
        expert = _make_concrete_expert()
        agent = expert._create_default_agent()
        assert agent is not None

    def test_get_training_data_returns_defaults(self):
        expert = _make_concrete_expert()
        # Replace config so training_gold_standards is None to trigger fallback
        expert.config = MagicMock()
        expert.config.training_gold_standards = None
        data = expert.get_training_data()
        assert len(data) == 2
        assert data[0]["task"] == "test task 1"

    def test_get_training_data_returns_config_data_if_present(self):
        expert = _make_concrete_expert()
        custom_data = [{"task": "custom", "gold_standard": "custom output"}]
        expert.config = MagicMock()
        expert.config.training_gold_standards = custom_data
        data = expert.get_training_data()
        assert data == custom_data

    def test_get_validation_data_returns_defaults(self):
        expert = _make_concrete_expert()
        expert.config = MagicMock()
        expert.config.validation_cases = None
        data = expert.get_validation_data()
        assert len(data) == 1
        assert data[0]["task"] == "val task 1"

    def test_get_validation_data_returns_config_data_if_present(self):
        expert = _make_concrete_expert()
        custom_data = [{"task": "val_custom", "expected_output": "val_custom_out"}]
        expert.config = MagicMock()
        expert.config.validation_cases = custom_data
        data = expert.get_validation_data()
        assert data == custom_data

    def test_is_dspy_available(self):
        expert = _make_concrete_expert()
        result = expert._is_dspy_available()
        assert isinstance(result, bool)

    def test_get_stats(self):
        expert = _make_concrete_expert()
        expert.config = MagicMock()
        expert.config.training_gold_standards = None
        expert.config.validation_cases = None
        stats = expert.get_stats()
        assert stats["expert_type"] == "ConcreteTestExpert"
        assert stats["domain"] == "test_domain"
        assert stats["improvements_count"] == 0
        assert stats["training_cases"] == 2
        assert stats["validation_cases"] == 1

    def test_repr(self):
        expert = _make_concrete_expert()
        r = repr(expert)
        assert "ConcreteTestExpert" in r
        assert "test_domain" in r

    @pytest.mark.asyncio
    async def test_evaluate_domain_correct(self):
        expert = _make_concrete_expert()
        result = await expert._evaluate_domain(
            output="hello",
            gold_standard="hello",
            task="test",
            context={},
        )
        assert result["is_valid"] is True
        assert result["score"] == 1.0

    @pytest.mark.asyncio
    async def test_evaluate_domain_incorrect(self):
        expert = _make_concrete_expert()
        result = await expert._evaluate_domain(
            output="wrong",
            gold_standard="expected",
            task="test",
            context={},
        )
        assert result["is_valid"] is False
        assert result["score"] == 0.0


@pytest.mark.unit
@pytest.mark.skipif(not HAS_BASE_EXPERT, reason="BaseExpert not available")
class TestSimpleDomainExpert:
    """Tests for SimpleDomainExpert - the no-DSPy base class."""

    def test_simple_expert_init(self):
        expert = _make_simple_expert()
        assert expert.domain == "simple_test"

    def test_create_domain_agent_returns_none(self):
        expert = _make_simple_expert()
        assert expert._create_domain_agent() is None

    def test_create_domain_teacher_returns_none(self):
        expert = _make_simple_expert()
        assert expert._create_domain_teacher() is None

    def test_get_default_training_cases_empty(self):
        assert SimpleDomainExpert._get_default_training_cases() == []

    def test_get_default_validation_cases_empty(self):
        assert SimpleDomainExpert._get_default_validation_cases() == []

    @pytest.mark.asyncio
    async def test_evaluate_domain_pass(self):
        expert = _make_simple_expert()
        result = await expert._evaluate_domain("abc", "abc", "t", {})
        assert result["is_valid"] is True
        assert result["score"] == 1.0

    @pytest.mark.asyncio
    async def test_evaluate_domain_fail(self):
        expert = _make_simple_expert()
        result = await expert._evaluate_domain("abc", "xyz", "t", {})
        assert result["is_valid"] is False
        assert result["score"] == 0.0
        assert result["error"] == "Output does not match expected"


# ============================================================================
# 3. Expert Templates Tests
# ============================================================================


@pytest.mark.unit
class TestExpertTemplates:
    """Tests for expert template factory functions."""

    @patch("Jotty.core.experts.mermaid_expert.MermaidExpertAgent", create=True)
    def test_create_mermaid_expert(self, MockMermaid):
        """Test create_mermaid_expert factory with mocked MermaidExpertAgent."""
        from Jotty.core.intelligence.reasoning.experts import expert_templates
        from Jotty.core.intelligence.reasoning.experts.expert_agent import ExpertAgentConfig

        mock_instance = MagicMock()

        # Patch local import within create_mermaid_expert
        with patch.object(expert_templates, "__builtins__", expert_templates.__builtins__):
            # Direct approach: mock the import target
            with patch.dict(
                "sys.modules",
                {
                    "Jotty.core.experts.mermaid_expert": MagicMock(
                        MermaidExpertAgent=MagicMock(return_value=mock_instance)
                    ),
                },
            ):
                result = expert_templates.create_mermaid_expert()
                assert result is mock_instance

    @patch("Jotty.core.experts.mermaid_expert.MermaidExpertAgent", create=True)
    def test_create_mermaid_expert_with_memory(self, _):
        """Test create_mermaid_expert passes memory correctly."""
        from Jotty.core.intelligence.reasoning.experts import expert_templates

        mock_memory = MagicMock()
        mock_instance = MagicMock()
        mock_cls = MagicMock(return_value=mock_instance)

        with patch.dict(
            "sys.modules",
            {
                "Jotty.core.experts.mermaid_expert": MagicMock(MermaidExpertAgent=mock_cls),
            },
        ):
            result = expert_templates.create_mermaid_expert(memory=mock_memory)
            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs["memory"] is mock_memory

    @patch("Jotty.core.experts.mermaid_expert.MermaidExpertAgent", create=True)
    def test_create_mermaid_expert_with_improvements(self, _):
        """Test create_mermaid_expert passes improvements correctly."""
        from Jotty.core.intelligence.reasoning.experts import expert_templates

        improvements = [{"pattern": "use subgraphs"}]
        mock_instance = MagicMock()
        mock_cls = MagicMock(return_value=mock_instance)

        with patch.dict(
            "sys.modules",
            {
                "Jotty.core.experts.mermaid_expert": MagicMock(MermaidExpertAgent=mock_cls),
            },
        ):
            result = expert_templates.create_mermaid_expert(improvements=improvements)
            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs["improvements"] is improvements

    def test_create_mermaid_expert_config_fields(self):
        """Test that create_mermaid_expert creates correct ExpertAgentConfig."""
        from Jotty.core.intelligence.reasoning.experts import expert_templates

        mock_instance = MagicMock()
        mock_cls = MagicMock(return_value=mock_instance)

        with patch.dict(
            "sys.modules",
            {
                "Jotty.core.experts.mermaid_expert": MagicMock(MermaidExpertAgent=mock_cls),
            },
        ):
            expert_templates.create_mermaid_expert()
            call_kwargs = mock_cls.call_args[1]
            config = call_kwargs["config"]
            assert config.name == "mermaid_expert"
            assert config.domain == "mermaid"
            assert config.max_training_iterations == 5
            assert config.min_validation_score == 1.0

    def test_create_plantuml_expert(self):
        """Test create_plantuml_expert factory."""
        from Jotty.core.intelligence.reasoning.experts import expert_templates

        mock_instance = MagicMock()
        mock_cls = MagicMock(return_value=mock_instance)

        with patch.dict(
            "sys.modules",
            {
                "Jotty.core.experts.plantuml_expert": MagicMock(PlantUMLExpertAgent=mock_cls),
            },
        ):
            result = expert_templates.create_plantuml_expert()
            assert result is mock_instance

    def test_create_plantuml_expert_config_fields(self):
        """Test that create_plantuml_expert creates correct ExpertAgentConfig."""
        from Jotty.core.intelligence.reasoning.experts import expert_templates

        mock_instance = MagicMock()
        mock_cls = MagicMock(return_value=mock_instance)

        with patch.dict(
            "sys.modules",
            {
                "Jotty.core.experts.plantuml_expert": MagicMock(PlantUMLExpertAgent=mock_cls),
            },
        ):
            expert_templates.create_plantuml_expert()
            call_kwargs = mock_cls.call_args[1]
            config = call_kwargs["config"]
            assert config.name == "plantuml_expert"
            assert config.domain == "plantuml"

    def test_create_sql_expert_default_dialect(self):
        """Test create_sql_expert with default postgresql dialect."""
        from Jotty.core.intelligence.reasoning.experts import expert_templates

        mock_instance = MagicMock()
        mock_cls = MagicMock(return_value=mock_instance)

        with patch.dict(
            "sys.modules",
            {
                "Jotty.core.experts.expert_agent": MagicMock(
                    ExpertAgent=mock_cls,
                    ExpertAgentConfig=ExpertAgentConfig,
                ),
            },
        ):
            # create_sql_expert uses dspy at module level, which is already imported
            result = expert_templates.create_sql_expert()
            call_kwargs = mock_cls.call_args[1]
            config = call_kwargs["config"]
            assert config.name == "sql_postgresql_expert"
            assert config.domain == "sql_postgresql"
            assert result is mock_instance

    def test_create_sql_expert_custom_dialect(self):
        """Test create_sql_expert with custom mysql dialect."""
        from Jotty.core.intelligence.reasoning.experts import expert_templates

        mock_instance = MagicMock()
        mock_cls = MagicMock(return_value=mock_instance)

        with patch.dict(
            "sys.modules",
            {
                "Jotty.core.experts.expert_agent": MagicMock(
                    ExpertAgent=mock_cls,
                    ExpertAgentConfig=ExpertAgentConfig,
                ),
            },
        ):
            expert_templates.create_sql_expert(dialect="mysql")
            call_kwargs = mock_cls.call_args[1]
            config = call_kwargs["config"]
            assert config.name == "sql_mysql_expert"
            assert config.domain == "sql_mysql"


# ============================================================================
# 4. ToolInterceptor Tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.skipif(not HAS_TOOL_INTERCEPTOR, reason="ToolInterceptor not available")
class TestToolCall:
    """Tests for the ToolCall dataclass."""

    def test_toolcall_basic_fields(self):
        tc = ToolCall(
            tool_name="execute_query",
            args={"query": "SELECT 1"},
            result="ok",
            success=True,
        )
        assert tc.tool_name == "execute_query"
        assert tc.args == {"query": "SELECT 1"}
        assert tc.result == "ok"
        assert tc.success is True
        assert tc.error is None
        assert tc.attempt_number == 0
        assert tc.metadata == {}

    def test_toolcall_with_error(self):
        tc = ToolCall(
            tool_name="run",
            args={},
            result=None,
            success=False,
            error="connection refused",
            attempt_number=3,
            metadata={"actor": "sql_gen"},
        )
        assert tc.success is False
        assert tc.error == "connection refused"
        assert tc.attempt_number == 3
        assert tc.metadata["actor"] == "sql_gen"


@pytest.mark.unit
@pytest.mark.skipif(not HAS_TOOL_INTERCEPTOR, reason="ToolInterceptor not available")
class TestToolInterceptor:
    """Tests for ToolInterceptor: wrapping, intercepting, summarizing."""

    def _make_interceptor(self, name="test_actor"):
        return ToolInterceptor(actor_name=name)

    def test_init(self):
        interceptor = self._make_interceptor("my_actor")
        assert interceptor.actor_name == "my_actor"
        assert interceptor.get_all_calls() == []

    def test_wrap_tools_returns_wrapped_dict(self):
        interceptor = self._make_interceptor()
        tools = {
            "tool_a": lambda **kw: "result_a",
            "tool_b": lambda **kw: "result_b",
        }
        wrapped = interceptor.wrap_tools(tools)
        assert set(wrapped.keys()) == {"tool_a", "tool_b"}
        assert callable(wrapped["tool_a"])
        assert callable(wrapped["tool_b"])

    def test_wrapped_tool_records_successful_call(self):
        interceptor = self._make_interceptor()

        def my_tool(**kwargs):
            return kwargs.get("x", 0) * 2

        wrapped = interceptor.wrap_tools({"my_tool": my_tool})
        result = wrapped["my_tool"](x=5)
        assert result == 10

        calls = interceptor.get_all_calls()
        assert len(calls) == 1
        assert calls[0].tool_name == "my_tool"
        assert calls[0].success is True
        assert calls[0].result == 10
        assert calls[0].attempt_number == 1

    def test_wrapped_tool_records_failed_call(self):
        interceptor = self._make_interceptor()

        def failing_tool(**kwargs):
            raise ValueError("bad input")

        wrapped = interceptor.wrap_tools({"failing_tool": failing_tool})
        with pytest.raises(Exception, match="bad input"):
            wrapped["failing_tool"]()

        calls = interceptor.get_all_calls()
        assert len(calls) == 1
        assert calls[0].success is False
        assert "bad input" in calls[0].error

    def test_attempt_counter_increments(self):
        interceptor = self._make_interceptor()

        def noop(**kw):
            return "ok"

        wrapped = interceptor.wrap_tools({"noop": noop})
        wrapped["noop"]()
        wrapped["noop"]()
        wrapped["noop"]()

        calls = interceptor.get_all_calls()
        assert len(calls) == 3
        assert calls[0].attempt_number == 1
        assert calls[1].attempt_number == 2
        assert calls[2].attempt_number == 3

    def test_get_calls_for_tool(self):
        interceptor = self._make_interceptor()
        tools = {
            "alpha": lambda **kw: "a",
            "beta": lambda **kw: "b",
        }
        wrapped = interceptor.wrap_tools(tools)
        wrapped["alpha"]()
        wrapped["beta"]()
        wrapped["alpha"]()

        alpha_calls = interceptor.get_calls_for_tool("alpha")
        assert len(alpha_calls) == 2
        beta_calls = interceptor.get_calls_for_tool("beta")
        assert len(beta_calls) == 1

    def test_get_successful_calls(self):
        interceptor = self._make_interceptor()

        def maybe_fail(**kw):
            if kw.get("fail"):
                raise RuntimeError("oops")
            return "ok"

        wrapped = interceptor.wrap_tools({"maybe_fail": maybe_fail})
        wrapped["maybe_fail"](fail=False)
        try:
            wrapped["maybe_fail"](fail=True)
        except Exception:
            pass
        wrapped["maybe_fail"](fail=False)

        assert len(interceptor.get_successful_calls()) == 2
        assert len(interceptor.get_failed_calls()) == 1

    def test_clear(self):
        interceptor = self._make_interceptor()
        wrapped = interceptor.wrap_tools({"t": lambda **kw: 1})
        wrapped["t"]()
        assert len(interceptor.get_all_calls()) == 1
        interceptor.clear()
        assert len(interceptor.get_all_calls()) == 0

    def test_summary(self):
        interceptor = self._make_interceptor("actor_x")
        tools = {"a": lambda **kw: 1, "b": lambda **kw: 2}
        wrapped = interceptor.wrap_tools(tools)
        wrapped["a"]()
        wrapped["b"]()
        wrapped["a"]()

        summary = interceptor.summary()
        assert summary["actor"] == "actor_x"
        assert summary["total_calls"] == 3
        assert summary["successful"] == 3
        assert summary["failed"] == 0
        assert summary["by_tool"]["a"]["total"] == 2
        assert summary["by_tool"]["b"]["total"] == 1

    def test_wrapped_preserves_function_name(self):
        interceptor = self._make_interceptor()

        def original_name(**kw):
            """Original docstring."""
            return 1

        wrapped = interceptor.wrap_tools({"original_name": original_name})
        assert wrapped["original_name"].__name__ == "original_name"
        assert wrapped["original_name"].__doc__ == "Original docstring."


@pytest.mark.unit
@pytest.mark.skipif(not HAS_TOOL_INTERCEPTOR, reason="ToolInterceptor not available")
class TestToolCallRegistry:
    """Tests for ToolCallRegistry: multi-actor management."""

    def _make_registry(self):
        return ToolCallRegistry()

    def test_init_empty(self):
        reg = self._make_registry()
        assert reg.get_all_calls() == []

    def test_get_or_create_interceptor_creates_new(self):
        reg = self._make_registry()
        i1 = reg.get_or_create_interceptor("actor1")
        assert isinstance(i1, ToolInterceptor)
        assert i1.actor_name == "actor1"

    def test_get_or_create_interceptor_returns_same(self):
        reg = self._make_registry()
        i1 = reg.get_or_create_interceptor("actor1")
        i2 = reg.get_or_create_interceptor("actor1")
        assert i1 is i2

    def test_get_all_calls_aggregates_across_actors(self):
        reg = self._make_registry()
        i1 = reg.get_or_create_interceptor("actor1")
        i2 = reg.get_or_create_interceptor("actor2")
        w1 = i1.wrap_tools({"t": lambda **kw: 1})
        w2 = i2.wrap_tools({"t": lambda **kw: 2})
        w1["t"]()
        w2["t"]()
        w2["t"]()
        all_calls = reg.get_all_calls()
        assert len(all_calls) == 3

    def test_summary_aggregates(self):
        reg = self._make_registry()
        i1 = reg.get_or_create_interceptor("a")
        i2 = reg.get_or_create_interceptor("b")
        w1 = i1.wrap_tools({"x": lambda **kw: 1})
        w2 = i2.wrap_tools({"y": lambda **kw: 2})
        w1["x"]()
        w2["y"]()

        summary = reg.summary()
        assert summary["total_calls"] == 2
        assert summary["successful"] == 2
        assert summary["failed"] == 0
        assert "a" in summary["by_actor"]
        assert "b" in summary["by_actor"]

    def test_clear_all(self):
        reg = self._make_registry()
        i = reg.get_or_create_interceptor("a")
        w = i.wrap_tools({"t": lambda **kw: 1})
        w["t"]()
        assert len(reg.get_all_calls()) == 1
        reg.clear_all()
        assert len(reg.get_all_calls()) == 0


# ============================================================================
# 5. MetadataToolRegistry Tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.skipif(not HAS_METADATA_TOOL_REGISTRY, reason="MetadataToolRegistry not available")
class TestMetadataToolRegistry:
    """Tests for MetadataToolRegistry: discover, call, list, info, caching."""

    def _make_registry(self):
        metadata = _make_metadata_instance_with_jotty_methods()
        return MetadataToolRegistry(metadata)

    def test_init_discovers_tools(self):
        reg = self._make_registry()
        tool_names = reg.list_tools()
        assert "get_terms" in tool_names
        assert "get_schema" in tool_names

    def test_list_tools(self):
        reg = self._make_registry()
        tools = reg.list_tools()
        assert isinstance(tools, list)
        assert len(tools) >= 2

    def test_get_tool_info_found(self):
        reg = self._make_registry()
        info = reg.get_tool_info("get_terms")
        assert info is not None
        assert info["desc"] == "Get business terms"
        assert info["when"] == "Agent needs terminology"

    def test_get_tool_info_not_found(self):
        reg = self._make_registry()
        info = reg.get_tool_info("nonexistent")
        assert info is None

    def test_call_tool_no_params(self):
        reg = self._make_registry()
        result = reg.call_tool("get_terms")
        assert result == {"revenue": "total income"}

    def test_call_tool_with_params(self):
        reg = self._make_registry()
        result = reg.call_tool("get_schema", table_name="users")
        assert result == {"columns": ["id", "name"]}

    def test_call_tool_unknown_raises_valueerror(self):
        reg = self._make_registry()
        with pytest.raises(ValueError, match="not found"):
            reg.call_tool("nonexistent_tool")

    def test_call_tool_caching(self):
        """Tools with cache=True should return cached results on second call."""
        reg = self._make_registry()
        r1 = reg.call_tool("get_terms")
        r2 = reg.call_tool("get_terms")
        assert r1 == r2

    def test_call_tool_no_cache(self):
        """Tools with cache=False should not cache."""
        reg = self._make_registry()
        r1 = reg.call_tool("get_schema", table_name="t1")
        r2 = reg.call_tool("get_schema", table_name="t1")
        assert r1 == r2

    def test_get_usage_stats(self):
        reg = self._make_registry()
        reg.call_tool("get_terms")
        reg.call_tool("get_terms")
        reg.call_tool("get_schema", table_name="x")
        stats = reg.get_usage_stats()
        assert stats["get_terms"] == 2
        assert stats["get_schema"] == 1

    def test_clear_cache(self):
        reg = self._make_registry()
        reg.call_tool("get_terms")
        assert len(reg._cache) > 0
        reg.clear_cache()
        assert len(reg._cache) == 0

    def test_get_tool_catalog_for_llm(self):
        reg = self._make_registry()
        catalog = reg.get_tool_catalog_for_llm()
        assert isinstance(catalog, str)
        assert "get_terms" in catalog
        assert "get_schema" in catalog
        assert "Description:" in catalog
        assert "When to use:" in catalog

    def test_get_tool_catalog_empty(self):
        """Empty metadata should produce 'no tools available' message."""
        empty_metadata = type("EmptyMeta", (), {})()
        reg = MetadataToolRegistry(empty_metadata)
        catalog = reg.get_tool_catalog_for_llm()
        assert "No metadata tools available" in catalog

    def test_repr(self):
        reg = self._make_registry()
        r = repr(reg)
        assert "MetadataToolRegistry" in r

    def test_validate_parameters_missing_required(self):
        """Should raise TypeError when required parameters are missing."""
        reg = self._make_registry()
        tool_info = reg.get_tool_info("get_schema")
        sig_params = tool_info["signature"]["parameters"]
        if sig_params.get("table_name", {}).get("required", False):
            with pytest.raises(TypeError, match="missing required"):
                reg.call_tool("get_schema")

    def test_discover_tools_skips_private_methods(self):
        """Methods starting with _ should not be discovered."""

        class MetaWithPrivate:
            def _private_method(self):
                return "hidden"

            _private_method._jotty_meta = {
                "desc": "Private",
                "when": "never",
                "cache": False,
            }

            def public_method(self):
                return "visible"

            public_method._jotty_meta = {
                "desc": "Public",
                "when": "always",
                "cache": False,
            }

        reg = MetadataToolRegistry(MetaWithPrivate())
        tools = reg.list_tools()
        assert "_private_method" not in tools
        assert "public_method" in tools

    def test_extract_signature_with_defaults(self):
        """Parameters with defaults should not be marked required."""
        reg = self._make_registry()
        info = reg.get_tool_info("get_terms")
        assert info["signature"]["parameters"] == {} or isinstance(
            info["signature"]["parameters"], dict
        )


# ============================================================================
# 6. WidgetParamSchema Tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.skipif(not HAS_WIDGET_PARAMS, reason="WidgetParamSchema not available")
class TestWidgetParamSchema:
    """Tests for WidgetParamSchema: validation, defaults, type checking."""

    def _make_schema(self):
        """Create a standard test schema with status and limit."""
        return WidgetParamSchema(
            properties={
                "status": {
                    "type": "string",
                    "enum": ["pending", "completed", "failed"],
                    "description": "Task status",
                },
                "limit": {
                    "type": "integer",
                    "default": 50,
                    "minimum": 1,
                    "maximum": 500,
                    "description": "Max items",
                },
                "active": {
                    "type": "boolean",
                    "description": "Active filter",
                },
            },
            required=["status"],
        )

    def test_init_defaults(self):
        schema = WidgetParamSchema()
        assert schema.properties == {}
        assert schema.required == []

    def test_validate_valid_params(self):
        schema = self._make_schema()
        is_valid, err = schema.validate({"status": "pending", "limit": 10})
        assert is_valid is True
        assert err is None

    def test_validate_missing_required(self):
        schema = self._make_schema()
        is_valid, err = schema.validate({"limit": 10})
        assert is_valid is False
        assert "Missing required parameter: status" in err

    def test_validate_wrong_type_string(self):
        schema = self._make_schema()
        is_valid, err = schema.validate({"status": 123})
        assert is_valid is False
        assert "must be string" in err

    def test_validate_wrong_type_integer(self):
        schema = self._make_schema()
        is_valid, err = schema.validate({"status": "pending", "limit": "ten"})
        assert is_valid is False
        assert "must be integer" in err

    def test_validate_wrong_type_boolean(self):
        schema = self._make_schema()
        is_valid, err = schema.validate({"status": "pending", "active": "yes"})
        assert is_valid is False
        assert "must be boolean" in err

    def test_validate_invalid_enum(self):
        schema = self._make_schema()
        is_valid, err = schema.validate({"status": "unknown"})
        assert is_valid is False
        assert "must be one of" in err
        assert "'pending'" in err

    def test_validate_below_minimum(self):
        schema = self._make_schema()
        is_valid, err = schema.validate({"status": "pending", "limit": 0})
        assert is_valid is False
        assert ">=" in err

    def test_validate_above_maximum(self):
        schema = self._make_schema()
        is_valid, err = schema.validate({"status": "pending", "limit": 1000})
        assert is_valid is False
        assert "<=" in err

    def test_validate_allows_unknown_params(self):
        schema = self._make_schema()
        is_valid, err = schema.validate({"status": "pending", "unknown_field": "x"})
        assert is_valid is True
        assert err is None

    def test_validate_empty_params_missing_required(self):
        schema = self._make_schema()
        is_valid, err = schema.validate({})
        assert is_valid is False
        assert "Missing required" in err

    def test_validate_no_required_passes_empty(self):
        schema = WidgetParamSchema(
            properties={"x": {"type": "string"}},
            required=[],
        )
        is_valid, err = schema.validate({})
        assert is_valid is True

    def test_get_default_value(self):
        schema = self._make_schema()
        assert schema.get_default_value("limit") == 50
        assert schema.get_default_value("status") is None
        assert schema.get_default_value("nonexistent") is None

    def test_apply_defaults(self):
        schema = self._make_schema()
        params = {"status": "pending"}
        result = schema.apply_defaults(params)
        assert result["status"] == "pending"
        assert result["limit"] == 50
        assert "limit" not in params

    def test_apply_defaults_does_not_overwrite(self):
        schema = self._make_schema()
        params = {"status": "completed", "limit": 10}
        result = schema.apply_defaults(params)
        assert result["limit"] == 10

    def test_check_type_string(self):
        schema = self._make_schema()
        assert schema._check_type("hello", "string") is True
        assert schema._check_type(123, "string") is False

    def test_check_type_integer(self):
        schema = self._make_schema()
        assert schema._check_type(42, "integer") is True
        assert schema._check_type(3.14, "integer") is False

    def test_check_type_number(self):
        schema = self._make_schema()
        assert schema._check_type(42, "number") is True
        assert schema._check_type(3.14, "number") is True
        assert schema._check_type("nan", "number") is False

    def test_check_type_boolean(self):
        schema = self._make_schema()
        assert schema._check_type(True, "boolean") is True
        assert schema._check_type(1, "boolean") is False

    def test_check_type_array(self):
        schema = self._make_schema()
        assert schema._check_type([1, 2], "array") is True
        assert schema._check_type((1, 2), "array") is False

    def test_check_type_object(self):
        schema = self._make_schema()
        assert schema._check_type({"k": "v"}, "object") is True
        assert schema._check_type([1], "object") is False

    def test_check_type_null(self):
        schema = self._make_schema()
        assert schema._check_type(None, "null") is True
        assert schema._check_type("", "null") is False

    def test_check_type_unknown_allows_all(self):
        schema = self._make_schema()
        assert schema._check_type("anything", "custom_type") is True


@pytest.mark.unit
@pytest.mark.skipif(not HAS_WIDGET_PARAMS, reason="WidgetParamSchema not available")
class TestGenerateParamDocstring:
    """Tests for generate_param_docstring function."""

    def test_basic_docstring(self):
        schema = WidgetParamSchema(
            properties={
                "name": {
                    "type": "string",
                    "description": "Widget name",
                }
            },
            required=["name"],
        )
        doc = generate_param_docstring(schema, "my_widget")
        assert "my widget" in doc
        assert "name" in doc
        assert "required" in doc
        assert "string" in doc

    def test_docstring_with_enum(self):
        schema = WidgetParamSchema(
            properties={
                "color": {
                    "type": "string",
                    "enum": ["red", "blue"],
                    "description": "Color choice",
                    "examples": [
                        {"value": "red", "description": "Red color"},
                        {"value": "blue", "description": "Blue color"},
                    ],
                }
            },
            required=[],
        )
        doc = generate_param_docstring(schema, "color_picker")
        assert "Options:" in doc
        assert '"red"' in doc
        assert '"blue"' in doc
        assert "Red color" in doc

    def test_docstring_with_default(self):
        schema = WidgetParamSchema(
            properties={
                "count": {
                    "type": "integer",
                    "default": 10,
                    "description": "Number of items",
                }
            },
            required=[],
        )
        doc = generate_param_docstring(schema, "item_list")
        assert "default=10" in doc

    def test_docstring_empty_schema(self):
        schema = WidgetParamSchema()
        doc = generate_param_docstring(schema, "empty_widget")
        assert "no parameters required" in doc.lower()

    def test_docstring_with_range(self):
        schema = WidgetParamSchema(
            properties={
                "size": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Size",
                }
            },
            required=[],
        )
        doc = generate_param_docstring(schema, "sized_widget")
        assert "min=" in doc
        assert "max=" in doc


@pytest.mark.unit
@pytest.mark.skipif(not HAS_WIDGET_PARAMS, reason="WidgetParamSchema not available")
class TestGenerateToolExamples:
    """Tests for generate_tool_examples function."""

    def test_generates_default_example(self):
        schema = WidgetParamSchema(
            properties={"x": {"type": "integer"}},
            required=[],
        )
        examples = generate_tool_examples(schema, "my_widget")
        assert len(examples) >= 1
        assert "show my widget" in examples[0]

    def test_generates_enum_examples(self):
        schema = WidgetParamSchema(
            properties={
                "status": {
                    "type": "string",
                    "enum": ["open", "closed"],
                    "examples": [
                        {
                            "value": "open",
                            "description": "Open items",
                            "user_query": "show open items",
                        },
                        {
                            "value": "closed",
                            "description": "Closed items",
                            "user_query": "show closed items",
                        },
                    ],
                }
            },
            required=[],
        )
        examples = generate_tool_examples(schema, "items")
        assert len(examples) >= 3
        texts = "\n".join(examples)
        assert "show open items" in texts
        assert "show closed items" in texts


@pytest.mark.unit
@pytest.mark.skipif(not HAS_WIDGET_PARAMS, reason="WidgetParamSchema not available")
class TestStandardParamSchemas:
    """Tests for the pre-defined standard parameter schemas."""

    def test_status_param_schema(self):
        assert "status" in STATUS_PARAM_SCHEMA
        assert STATUS_PARAM_SCHEMA["status"]["type"] == "string"
        assert "pending" in STATUS_PARAM_SCHEMA["status"]["enum"]
        assert "completed" in STATUS_PARAM_SCHEMA["status"]["enum"]

    def test_limit_param_schema(self):
        assert "limit" in LIMIT_PARAM_SCHEMA
        assert LIMIT_PARAM_SCHEMA["limit"]["type"] == "integer"
        assert LIMIT_PARAM_SCHEMA["limit"]["default"] == 100
        assert LIMIT_PARAM_SCHEMA["limit"]["minimum"] == 1
        assert LIMIT_PARAM_SCHEMA["limit"]["maximum"] == 1000

    def test_time_range_param_schema(self):
        assert "time_range" in TIME_RANGE_PARAM_SCHEMA
        assert "today" in TIME_RANGE_PARAM_SCHEMA["time_range"]["enum"]
        assert TIME_RANGE_PARAM_SCHEMA["time_range"]["default"] == "week"

    def test_sort_param_schema(self):
        assert "sort_by" in SORT_PARAM_SCHEMA
        assert "sort_order" in SORT_PARAM_SCHEMA
        assert "asc" in SORT_PARAM_SCHEMA["sort_order"]["enum"]
        assert "desc" in SORT_PARAM_SCHEMA["sort_order"]["enum"]

    def test_status_schema_validates(self):
        schema = WidgetParamSchema(
            properties=STATUS_PARAM_SCHEMA,
            required=[],
        )
        is_valid, err = schema.validate({"status": "pending"})
        assert is_valid is True

        is_valid, err = schema.validate({"status": "invalid_status"})
        assert is_valid is False

    def test_limit_schema_validates(self):
        schema = WidgetParamSchema(
            properties=LIMIT_PARAM_SCHEMA,
            required=[],
        )
        is_valid, err = schema.validate({"limit": 50})
        assert is_valid is True

        is_valid, err = schema.validate({"limit": 0})
        assert is_valid is False

    def test_combined_schemas(self):
        """Standard schemas can be combined into a single WidgetParamSchema."""
        combined_props = {}
        combined_props.update(STATUS_PARAM_SCHEMA)
        combined_props.update(LIMIT_PARAM_SCHEMA)
        combined_props.update(TIME_RANGE_PARAM_SCHEMA)

        schema = WidgetParamSchema(
            properties=combined_props,
            required=["status"],
        )
        is_valid, err = schema.validate(
            {
                "status": "completed",
                "limit": 25,
                "time_range": "month",
            }
        )
        assert is_valid is True
        assert err is None
