"""
Tests for Pipeline Skill Framework
=====================================

Tests the generic pipeline skill system (pipeline_skill.py) which supports
declarative Source -> Processor -> Sink patterns for composite skill execution.

Covers:
- StepType enum: values and membership
- PipelineSkill: initialization, validation, execution, template resolution
- _resolve_params: dict, string, callable, nested templates
- _resolve_template_string: single variable, multi variable, step references
- _get_template_value: simple lookup, dot-notation, nested paths, error handling
- _execute_step: skill/tool resolution, sync/async execution, error paths
- create_pipeline_skill: factory function
- Error handling: empty pipeline, missing source, failed required steps

All registry lookups and tool executions are mocked.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try importing the module under test
try:
    from Jotty.core.capabilities.registry.pipeline_skill import (
        PipelineSkill,
        StepType,
        create_pipeline_skill,
    )

    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False


# =============================================================================
# Helper: create a mock registry with configurable skills/tools
# =============================================================================


def _make_mock_registry(skills_config: Dict[str, Dict[str, Any]] = None):
    """
    Create a mock registry with skills and tools.

    Args:
        skills_config: Dict mapping skill_name -> {tool_name: callable or return_value}
    """
    registry = MagicMock()

    if skills_config is None:
        skills_config = {}

    def get_skill(name):
        if name not in skills_config:
            return None
        skill = MagicMock()
        tools = {}
        for tool_name, handler in skills_config[name].items():
            if callable(handler):
                tools[tool_name] = handler
            else:
                tools[tool_name] = MagicMock(return_value=handler)
        skill.tools = tools
        return skill

    registry.get_skill = get_skill
    return registry


def _make_basic_pipeline():
    """Create a minimal valid pipeline (source + sink)."""
    return [
        {
            "type": "source",
            "skill": "data-fetcher",
            "tool": "fetch_tool",
            "params": {"url": "{{target_url}}"},
        },
        {
            "type": "sink",
            "skill": "output-writer",
            "tool": "write_tool",
            "params": {"data": "{{source_0.result}}"},
        },
    ]


# =============================================================================
# StepType Enum Tests
# =============================================================================


@pytest.mark.skipif(not PIPELINE_AVAILABLE, reason="PipelineSkill not importable")
@pytest.mark.unit
class TestStepTypeEnum:
    """Tests for StepType enum values."""

    def test_source_value(self):
        """StepType.SOURCE has value 'source'."""
        assert StepType.SOURCE.value == "source"

    def test_processor_value(self):
        """StepType.PROCESSOR has value 'processor'."""
        assert StepType.PROCESSOR.value == "processor"

    def test_sink_value(self):
        """StepType.SINK has value 'sink'."""
        assert StepType.SINK.value == "sink"

    def test_step_type_count(self):
        """StepType has exactly three members."""
        assert len(StepType) == 3


# =============================================================================
# PipelineSkill Initialization Tests
# =============================================================================


@pytest.mark.skipif(not PIPELINE_AVAILABLE, reason="PipelineSkill not importable")
@pytest.mark.unit
class TestPipelineSkillInit:
    """Tests for PipelineSkill initialization and validation."""

    def test_basic_init(self):
        """PipelineSkill initializes with name, description, and pipeline."""
        pipeline = _make_basic_pipeline()
        skill = PipelineSkill("test-pipe", "Test pipeline", pipeline)
        assert skill.name == "test-pipe"
        assert skill.description == "Test pipeline"
        assert len(skill.pipeline) == 2

    def test_empty_pipeline_raises(self):
        """PipelineSkill raises ValueError for empty pipeline."""
        with pytest.raises(ValueError, match="at least one step"):
            PipelineSkill("empty", "Empty", [])

    def test_missing_source_raises(self):
        """PipelineSkill raises ValueError when no source step is present."""
        pipeline = [
            {
                "type": "processor",
                "skill": "transform",
                "tool": "transform_tool",
                "params": {},
            },
            {
                "type": "sink",
                "skill": "output",
                "tool": "write_tool",
                "params": {},
            },
        ]
        with pytest.raises(ValueError, match="source"):
            PipelineSkill("no-src", "No source", pipeline)

    def test_missing_sink_logs_warning(self):
        """PipelineSkill warns (but does not raise) when no sink step is present."""
        pipeline = [
            {
                "type": "source",
                "skill": "data-fetcher",
                "tool": "fetch_tool",
                "params": {},
            },
        ]
        # Should not raise, just log warning
        skill = PipelineSkill("no-sink", "No sink", pipeline)
        assert skill.name == "no-sink"

    def test_pipeline_with_all_step_types(self):
        """PipelineSkill accepts pipeline with source, processor, and sink."""
        pipeline = [
            {"type": "source", "skill": "src", "tool": "src_tool", "params": {}},
            {"type": "processor", "skill": "proc", "tool": "proc_tool", "params": {}},
            {"type": "sink", "skill": "sink", "tool": "sink_tool", "params": {}},
        ]
        skill = PipelineSkill("full", "Full pipeline", pipeline)
        assert len(skill.pipeline) == 3


# =============================================================================
# Template Resolution Tests
# =============================================================================


@pytest.mark.skipif(not PIPELINE_AVAILABLE, reason="PipelineSkill not importable")
@pytest.mark.unit
class TestTemplateResolution:
    """Tests for _resolve_params, _resolve_template_string, _get_template_value."""

    def _make_skill(self):
        """Create a minimal PipelineSkill for testing resolve methods."""
        pipeline = _make_basic_pipeline()
        return PipelineSkill("test", "test", pipeline)

    def test_resolve_params_dict_with_templates(self):
        """_resolve_params resolves {{variable}} in dict values."""
        skill = self._make_skill()
        params = {"query": "{{topic}}", "limit": 10}
        current = {"topic": "AI trends"}
        results = {}
        resolved = skill._resolve_params(params, current, results)
        assert resolved["query"] == "AI trends"
        assert resolved["limit"] == 10

    def test_resolve_params_callable(self):
        """_resolve_params calls callable params."""
        skill = self._make_skill()
        func = MagicMock(return_value={"key": "value"})
        current = {"x": 1}
        results = {}
        resolved = skill._resolve_params(func, current, results)
        func.assert_called_once_with(current, results)
        assert resolved == {"key": "value"}

    def test_resolve_params_string_template(self):
        """_resolve_params resolves string templates."""
        skill = self._make_skill()
        current = {"name": "Jotty"}
        results = {}
        resolved = skill._resolve_params("{{name}}", current, results)
        assert resolved == "Jotty"

    def test_resolve_params_nested_dict(self):
        """_resolve_params resolves nested dicts recursively."""
        skill = self._make_skill()
        params = {"outer": {"inner": "{{val}}"}}
        current = {"val": "deep"}
        results = {}
        resolved = skill._resolve_params(params, current, results)
        assert resolved["outer"]["inner"] == "deep"

    def test_resolve_params_passthrough_non_template(self):
        """_resolve_params passes through non-template values as-is."""
        skill = self._make_skill()
        params = {"static": "no_template", "num": 42}
        resolved = skill._resolve_params(params, {}, {})
        assert resolved["static"] == "no_template"
        assert resolved["num"] == 42

    def test_resolve_params_returns_non_dict_as_is(self):
        """_resolve_params returns non-dict/non-string/non-callable as-is."""
        skill = self._make_skill()
        assert skill._resolve_params(42, {}, {}) == 42
        assert skill._resolve_params(None, {}, {}) is None

    def test_resolve_template_string_single_variable(self):
        """_resolve_template_string returns raw value for single variable."""
        skill = self._make_skill()
        current = {"items": [1, 2, 3]}
        result = skill._resolve_template_string("{{items}}", current, {})
        assert result == [1, 2, 3]  # Returns the list directly, not stringified

    def test_resolve_template_string_multi_variable(self):
        """_resolve_template_string substitutes multiple variables."""
        skill = self._make_skill()
        current = {"first": "Hello", "second": "World"}
        result = skill._resolve_template_string("{{first}} {{second}}!", current, {})
        assert result == "Hello World!"

    def test_resolve_template_string_no_templates(self):
        """_resolve_template_string returns string as-is with no templates."""
        skill = self._make_skill()
        result = skill._resolve_template_string("plain text", {}, {})
        assert result == "plain text"

    def test_resolve_template_string_step_reference(self):
        """_resolve_template_string resolves step.field references."""
        skill = self._make_skill()
        results = {"source_0": {"result": "fetched data"}}
        result = skill._resolve_template_string("{{source_0.result}}", {}, results)
        assert result == "fetched data"

    def test_resolve_template_nested_step_reference(self):
        """_resolve_template_string resolves nested step.field.subfield references."""
        skill = self._make_skill()
        results = {"proc": {"output": {"summary": "done"}}}
        result = skill._resolve_template_string("{{proc.output.summary}}", {}, results)
        assert result == "done"

    def test_get_template_value_from_current_params(self):
        """_get_template_value finds simple variables in current_params."""
        skill = self._make_skill()
        value = skill._get_template_value("topic", {"topic": "AI"}, {})
        assert value == "AI"

    def test_get_template_value_from_results(self):
        """_get_template_value falls back to results dict."""
        skill = self._make_skill()
        value = skill._get_template_value("key", {}, {"key": "from_results"})
        assert value == "from_results"

    def test_get_template_value_dot_notation(self):
        """_get_template_value resolves dot-notation (step.field)."""
        skill = self._make_skill()
        results = {"source": {"data": "value"}}
        value = skill._get_template_value("source.data", {}, results)
        assert value == "value"

    def test_get_template_value_missing_raises(self):
        """_get_template_value raises ValueError for missing variables."""
        skill = self._make_skill()
        with pytest.raises(ValueError, match="not found"):
            skill._get_template_value("missing", {}, {})

    def test_get_template_value_missing_step_raises(self):
        """_get_template_value raises ValueError for missing step key."""
        skill = self._make_skill()
        with pytest.raises(ValueError, match="not found in results"):
            skill._get_template_value("nosuch.field", {}, {})

    def test_get_template_value_non_dict_step_raises(self):
        """_get_template_value raises ValueError when navigating non-dict."""
        skill = self._make_skill()
        results = {"step": "not_a_dict"}
        with pytest.raises(ValueError, match="non-dict"):
            skill._get_template_value("step.field", {}, results)

    def test_get_template_value_none_field_raises(self):
        """_get_template_value raises ValueError when field is None."""
        skill = self._make_skill()
        results = {"step": {"field": None}}
        with pytest.raises(ValueError, match="None"):
            skill._get_template_value("step.field", {}, results)


# =============================================================================
# Pipeline Execution Tests
# =============================================================================


@pytest.mark.skipif(not PIPELINE_AVAILABLE, reason="PipelineSkill not importable")
@pytest.mark.unit
class TestPipelineExecution:
    """Tests for PipelineSkill.execute method."""

    @pytest.mark.asyncio
    async def test_execute_basic_pipeline(self):
        """execute runs all steps and returns success."""
        pipeline = [
            {
                "type": "source",
                "skill": "fetcher",
                "tool": "fetch",
                "params": {"url": "http://example.com"},
            },
            {
                "type": "sink",
                "skill": "writer",
                "tool": "write",
                "params": {"data": "{{content}}"},
            },
        ]
        skill = PipelineSkill("basic", "Basic pipeline", pipeline)

        registry = _make_mock_registry(
            {
                "fetcher": {
                    "fetch": lambda p: {"success": True, "content": "hello"},
                },
                "writer": {
                    "write": lambda p: {"success": True, "written": True},
                },
            }
        )

        result = await skill.execute({"content": "initial"}, registry)
        assert result["_success"] is True

    @pytest.mark.asyncio
    async def test_execute_stores_initial_params(self):
        """execute stores initial_params under _initial key."""
        pipeline = _make_basic_pipeline()
        skill = PipelineSkill("test", "test", pipeline)

        registry = _make_mock_registry(
            {
                "data-fetcher": {"fetch_tool": lambda p: {"success": True, "result": "ok"}},
                "output-writer": {"write_tool": lambda p: {"success": True}},
            }
        )

        result = await skill.execute({"target_url": "http://x.com"}, registry)
        assert result["_initial"]["target_url"] == "http://x.com"

    @pytest.mark.asyncio
    async def test_execute_required_step_failure_stops(self):
        """execute stops when a required step fails."""
        pipeline = [
            {
                "type": "source",
                "skill": "fetcher",
                "tool": "fetch",
                "params": {},
                "required": True,
            },
            {
                "type": "sink",
                "skill": "writer",
                "tool": "write",
                "params": {},
            },
        ]
        skill = PipelineSkill("fail-test", "Failure test", pipeline)

        registry = _make_mock_registry(
            {
                "fetcher": {
                    "fetch": lambda p: {"success": False, "error": "network down"},
                },
                "writer": {
                    "write": lambda p: {"success": True},
                },
            }
        )

        result = await skill.execute({}, registry)
        assert result["_success"] is False
        assert "network down" in result["_error"]

    @pytest.mark.asyncio
    async def test_execute_optional_step_failure_continues(self):
        """execute continues when an optional (required=False) step fails."""
        pipeline = [
            {
                "type": "source",
                "skill": "fetcher",
                "tool": "fetch",
                "params": {},
            },
            {
                "type": "processor",
                "skill": "enricher",
                "tool": "enrich",
                "params": {},
                "required": False,
            },
            {
                "type": "sink",
                "skill": "writer",
                "tool": "write",
                "params": {},
            },
        ]
        skill = PipelineSkill("opt-fail", "Optional failure", pipeline)

        registry = _make_mock_registry(
            {
                "fetcher": {"fetch": lambda p: {"success": True, "data": "raw"}},
                "enricher": {"enrich": lambda p: {"success": False, "error": "optional fail"}},
                "writer": {"write": lambda p: {"success": True}},
            }
        )

        result = await skill.execute({}, registry)
        assert result["_success"] is True

    @pytest.mark.asyncio
    async def test_execute_with_async_tool(self):
        """execute handles async tool functions."""

        async def async_fetch(params):
            return {"success": True, "data": "async result"}

        pipeline = [
            {"type": "source", "skill": "async-src", "tool": "afetch", "params": {}},
            {"type": "sink", "skill": "sync-sink", "tool": "swrite", "params": {}},
        ]
        skill = PipelineSkill("async-test", "Async test", pipeline)

        registry = _make_mock_registry(
            {
                "async-src": {"afetch": async_fetch},
                "sync-sink": {"swrite": lambda p: {"success": True}},
            }
        )

        result = await skill.execute({}, registry)
        assert result["_success"] is True

    @pytest.mark.asyncio
    async def test_execute_step_params_chaining(self):
        """execute passes results from previous step to next via current_params."""
        captured_params = {}

        def sink_fn(params):
            captured_params.update(params)
            return {"success": True}

        pipeline = [
            {"type": "source", "skill": "src", "tool": "src_tool", "params": {}},
            {"type": "sink", "skill": "sink", "tool": "sink_tool", "params": {}},
        ]
        skill = PipelineSkill("chain", "Chain test", pipeline)

        registry = _make_mock_registry(
            {
                "src": {"src_tool": lambda p: {"success": True, "output": "chained_value"}},
                "sink": {"sink_tool": sink_fn},
            }
        )

        result = await skill.execute({"initial": "val"}, registry)
        assert result["_success"] is True

    @pytest.mark.asyncio
    async def test_execute_output_key_custom(self):
        """execute uses custom output_key when specified."""
        pipeline = [
            {
                "type": "source",
                "skill": "src",
                "tool": "src_tool",
                "params": {},
                "output_key": "my_source",
            },
            {"type": "sink", "skill": "sink", "tool": "sink_tool", "params": {}},
        ]
        skill = PipelineSkill("outkey", "Output key test", pipeline)

        registry = _make_mock_registry(
            {
                "src": {"src_tool": lambda p: {"success": True, "value": "keyed"}},
                "sink": {"sink_tool": lambda p: {"success": True}},
            }
        )

        result = await skill.execute({}, registry)
        assert "my_source" in result
        assert result["my_source"]["value"] == "keyed"

    @pytest.mark.asyncio
    async def test_execute_output_key_default(self):
        """execute uses default output_key (type_index) when not specified."""
        pipeline = [
            {"type": "source", "skill": "src", "tool": "st", "params": {}},
            {"type": "sink", "skill": "sink", "tool": "sk", "params": {}},
        ]
        skill = PipelineSkill("defkey", "Default key", pipeline)

        registry = _make_mock_registry(
            {
                "src": {"st": lambda p: {"success": True}},
                "sink": {"sk": lambda p: {"success": True}},
            }
        )

        result = await skill.execute({}, registry)
        assert "source_0" in result
        assert "sink_1" in result


# =============================================================================
# _execute_step Tests
# =============================================================================


@pytest.mark.skipif(not PIPELINE_AVAILABLE, reason="PipelineSkill not importable")
@pytest.mark.unit
class TestExecuteStep:
    """Tests for PipelineSkill._execute_step internal method."""

    def _make_skill(self):
        pipeline = _make_basic_pipeline()
        return PipelineSkill("step-test", "Step test", pipeline)

    @pytest.mark.asyncio
    async def test_execute_step_missing_skill_name(self):
        """_execute_step returns error when skill name is missing."""
        skill = self._make_skill()
        step = {"tool": "something"}
        registry = MagicMock()
        result = await skill._execute_step(step, {}, registry, {})
        assert result["success"] is False
        assert "skill and tool required" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_step_missing_tool_name(self):
        """_execute_step returns error when tool name is missing."""
        skill = self._make_skill()
        step = {"skill": "something"}
        registry = MagicMock()
        result = await skill._execute_step(step, {}, registry, {})
        assert result["success"] is False
        assert "skill and tool required" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_step_skill_not_found(self):
        """_execute_step returns error when skill is not in registry."""
        skill = self._make_skill()
        step = {"skill": "nonexistent", "tool": "some_tool"}
        registry = MagicMock()
        registry.get_skill.return_value = None
        result = await skill._execute_step(step, {}, registry, {})
        assert result["success"] is False
        assert "Skill not found" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_step_tool_not_found(self):
        """_execute_step returns error when tool is not in skill."""
        skill = self._make_skill()
        step = {"skill": "existing", "tool": "missing_tool"}
        mock_skill = MagicMock()
        mock_skill.tools = {}
        registry = MagicMock()
        registry.get_skill.return_value = mock_skill
        result = await skill._execute_step(step, {}, registry, {})
        assert result["success"] is False
        assert "Tool not found" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_step_tool_exception(self):
        """_execute_step catches tool exceptions and returns error."""
        skill = self._make_skill()
        step = {"skill": "buggy", "tool": "crash_tool"}

        def crash(params):
            raise RuntimeError("Tool crashed!")

        registry = _make_mock_registry(
            {
                "buggy": {"crash_tool": crash},
            }
        )

        result = await skill._execute_step(step, {}, registry, {})
        assert result["success"] is False
        assert "Tool crashed!" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_step_sync_tool(self):
        """_execute_step executes sync tool functions."""
        skill = self._make_skill()
        step = {"skill": "sync", "tool": "sync_tool"}

        registry = _make_mock_registry(
            {
                "sync": {"sync_tool": lambda p: {"success": True, "result": "sync ok"}},
            }
        )

        result = await skill._execute_step(step, {}, registry, {})
        assert result["success"] is True
        assert result["result"] == "sync ok"

    @pytest.mark.asyncio
    async def test_execute_step_async_tool(self):
        """_execute_step executes async tool functions."""
        skill = self._make_skill()
        step = {"skill": "async-skill", "tool": "async_tool"}

        async def async_handler(params):
            return {"success": True, "result": "async ok"}

        registry = _make_mock_registry(
            {
                "async-skill": {"async_tool": async_handler},
            }
        )

        result = await skill._execute_step(step, {}, registry, {})
        assert result["success"] is True
        assert result["result"] == "async ok"


# =============================================================================
# Factory Function Tests
# =============================================================================


@pytest.mark.skipif(not PIPELINE_AVAILABLE, reason="PipelineSkill not importable")
@pytest.mark.unit
class TestCreatePipelineSkill:
    """Tests for the create_pipeline_skill factory function."""

    def test_factory_returns_pipeline_skill(self):
        """create_pipeline_skill returns a PipelineSkill instance."""
        pipeline = _make_basic_pipeline()
        skill = create_pipeline_skill("factory-test", "Factory pipeline", pipeline)
        assert isinstance(skill, PipelineSkill)
        assert skill.name == "factory-test"
        assert skill.description == "Factory pipeline"

    def test_factory_validates_pipeline(self):
        """create_pipeline_skill raises ValueError for invalid pipeline."""
        with pytest.raises(ValueError):
            create_pipeline_skill("bad", "Bad pipeline", [])

    def test_factory_multi_step(self):
        """create_pipeline_skill handles multi-step pipelines."""
        pipeline = [
            {"type": "source", "skill": "s", "tool": "st", "params": {}},
            {"type": "processor", "skill": "p", "tool": "pt", "params": {}},
            {"type": "processor", "skill": "p2", "tool": "pt2", "params": {}},
            {"type": "sink", "skill": "k", "tool": "kt", "params": {}},
        ]
        skill = create_pipeline_skill("multi", "Multi step", pipeline)
        assert len(skill.pipeline) == 4


# =============================================================================
# Integration-Style Tests (Still Mocked)
# =============================================================================


@pytest.mark.skipif(not PIPELINE_AVAILABLE, reason="PipelineSkill not importable")
@pytest.mark.unit
class TestPipelineIntegration:
    """Integration-style tests for full pipeline execution with template resolution."""

    @pytest.mark.asyncio
    async def test_full_search_summarize_send_pipeline(self):
        """Full pipeline: source fetches, processor transforms, sink outputs."""
        pipeline = [
            {
                "type": "source",
                "skill": "web-search",
                "tool": "search_tool",
                "params": {"query": "{{topic}}", "max_results": 5},
            },
            {
                "type": "processor",
                "skill": "summarizer",
                "tool": "summarize_tool",
                "params": {"text": "{{results}}"},
            },
            {
                "type": "sink",
                "skill": "telegram",
                "tool": "send_tool",
                "params": {"message": "{{summary}}"},
            },
        ]
        skill = PipelineSkill("full-pipe", "Full pipeline", pipeline)

        registry = _make_mock_registry(
            {
                "web-search": {
                    "search_tool": lambda p: {
                        "success": True,
                        "results": "Search results about AI",
                    },
                },
                "summarizer": {
                    "summarize_tool": lambda p: {
                        "success": True,
                        "summary": "AI is advancing rapidly",
                    },
                },
                "telegram": {
                    "send_tool": lambda p: {"success": True, "sent": True},
                },
            }
        )

        result = await skill.execute({"topic": "AI trends"}, registry)
        assert result["_success"] is True

    @pytest.mark.asyncio
    async def test_pipeline_with_step_output_reference(self):
        """Pipeline uses step output references ({{step_key.field}})."""
        pipeline = [
            {
                "type": "source",
                "skill": "fetcher",
                "tool": "fetch",
                "params": {},
                "output_key": "fetched",
            },
            {
                "type": "sink",
                "skill": "sender",
                "tool": "send",
                "params": {"data": "{{fetched.payload}}"},
                "output_key": "sent",
            },
        ]
        skill = PipelineSkill("ref-pipe", "Reference pipeline", pipeline)

        registry = _make_mock_registry(
            {
                "fetcher": {
                    "fetch": lambda p: {"success": True, "payload": "important data"},
                },
                "sender": {
                    "send": lambda p: {"success": True, "delivered": True},
                },
            }
        )

        result = await skill.execute({}, registry)
        assert result["_success"] is True
        assert result["fetched"]["payload"] == "important data"

    @pytest.mark.asyncio
    async def test_pipeline_default_step_type_is_processor(self):
        """Steps without explicit type default to 'processor'."""
        pipeline = [
            {"type": "source", "skill": "src", "tool": "st", "params": {}},
            # No "type" key -- should default to "processor" in execute
            {"skill": "proc", "tool": "pt", "params": {}},
            {"type": "sink", "skill": "sink", "tool": "sk", "params": {}},
        ]
        skill = PipelineSkill("default-type", "Default type test", pipeline)

        registry = _make_mock_registry(
            {
                "src": {"st": lambda p: {"success": True}},
                "proc": {"pt": lambda p: {"success": True}},
                "sink": {"sk": lambda p: {"success": True}},
            }
        )

        result = await skill.execute({}, registry)
        assert result["_success"] is True
        # The step with no type should have key "processor_1"
        assert "processor_1" in result
