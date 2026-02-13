"""
Tests for ParameterResolver and SkillPlanExecutor.

Covers template resolution, dotted-path lookups, auto-wiring,
content validation, command/path sanitization, task type inference,
artifact tagging, and large output spilling.
"""

import os
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from Jotty.core.agents.base.skill_plan_executor import ParameterResolver, SkillPlanExecutor


# =============================================================================
# ParameterResolver Tests
# =============================================================================


@pytest.mark.unit
class TestParameterResolver:
    """Tests for the ParameterResolver class."""

    def test_resolve_returns_params_unchanged_when_no_templates(self):
        """Params without template variables pass through unmodified."""
        outputs = {"step_0": {"result": "hello"}}
        resolver = ParameterResolver(outputs)

        params = {"query": "plain text", "count": 5}
        resolved = resolver.resolve(params)

        assert resolved["query"] == "plain text"
        assert resolved["count"] == 5

    def test_resolve_substitutes_dollar_brace_ref(self):
        """${ref} templates are replaced with values from outputs."""
        outputs = {"step_0": {"path": "/tmp/report.pdf"}}
        resolver = ParameterResolver(outputs)

        params = {"file": "${step_0.path}"}
        resolved = resolver.resolve(params)

        assert resolved["file"] == "/tmp/report.pdf"

    def test_resolve_substitutes_bare_brace_ref(self):
        """{ref} templates (without $) are replaced from outputs."""
        outputs = {"step_0": {"title": "My Report"}}
        resolver = ParameterResolver(outputs)

        params = {"header": "{step_0.title}"}
        resolved = resolver.resolve(params)

        assert resolved["header"] == "My Report"

    def test_resolve_handles_nested_dicts_with_templates(self):
        """Template resolution recurses into nested dict values."""
        outputs = {"step_1": {"url": "https://example.com"}}
        resolver = ParameterResolver(outputs)

        params = {
            "config": {
                "endpoint": "${step_1.url}",
                "timeout": 30,
            }
        }
        resolved = resolver.resolve(params)

        assert resolved["config"]["endpoint"] == "https://example.com"
        assert resolved["config"]["timeout"] == 30

    def test_resolve_path_resolves_dotted_paths(self):
        """resolve_path navigates nested dicts via dot-separated keys."""
        outputs = {
            "step_1": {
                "data": {"name": "Alice"}
            }
        }
        resolver = ParameterResolver(outputs)

        result = resolver.resolve_path("step_1.data.name")
        assert result == "Alice"

    def test_resolve_path_returns_original_if_path_not_found(self):
        """resolve_path returns the original path string for unknown keys."""
        outputs = {}
        resolver = ParameterResolver(outputs)

        result = resolver.resolve_path("nonexistent.field")
        # With empty outputs, it should return the original path
        assert result == "nonexistent.field"

    def test_resolve_auto_wires_missing_params_from_outputs(self):
        """When tool_schema is provided, missing required params are auto-wired."""
        outputs = {"step_0": {"query": "test search", "path": "/tmp/out.txt"}}
        resolver = ParameterResolver(outputs)

        # Build a mock ToolSchema that:
        # - resolve_aliases: returns params unchanged
        # - auto_wire: injects 'query' from outputs
        # - validate: returns a valid result with no coerced params
        mock_param = Mock()
        mock_param.name = "query"
        mock_param.required = True

        mock_schema = Mock()
        mock_schema.resolve_aliases = Mock(side_effect=lambda p: p)
        mock_schema.auto_wire = Mock(
            side_effect=lambda params, outs: {**params, "query": "test search"}
        )
        mock_validation = Mock()
        mock_validation.coerced_params = {}
        mock_validation.errors = []
        mock_schema.validate = Mock(return_value=mock_validation)
        mock_schema.params = [mock_param]
        mock_schema.get_param = Mock(return_value=mock_param)
        mock_schema.name = "test_tool"

        params = {"other_param": "value"}
        resolved = resolver.resolve(params, tool_schema=mock_schema)

        # auto_wire should have been called and injected 'query'
        mock_schema.auto_wire.assert_called_once()
        assert resolved["query"] == "test search"

    def test_is_bad_content_returns_true_for_json_schema_like(self):
        """_is_bad_content flags short JSON-like success responses as bad."""
        resolver = ParameterResolver({})

        # Short success JSON response that looks like a status, not real content
        bad = '{"success": true, "bytes_written": 1234}'
        assert resolver._is_bad_content(bad) is True

    def test_sanitize_command_param_flags_suspicious_long_commands(self):
        """_sanitize_command_param rewrites long non-command strings to python invocations."""
        outputs = {
            "step_0": {"path": "/tmp/script.py", "success": True}
        }
        resolver = ParameterResolver(outputs)

        # A long string with many spaces (>15) that doesn't look like a shell command
        long_non_command = "This is a very long description of what " + \
            "should be done with the task and it contains way too many words " + \
            "to be a real shell command and should be auto-fixed by the sanitizer"
        step = Mock()
        step.description = "Run the script"

        result = resolver._sanitize_command_param("command", long_non_command, step)
        # Should be rewritten to "python /tmp/script.py" since there's a .py path in outputs
        assert result == "python /tmp/script.py"

    def test_resolve_with_empty_outputs_returns_params_as_is(self):
        """When outputs dict is empty, params pass through without modification."""
        resolver = ParameterResolver({})

        params = {"name": "test", "value": "${missing.ref}"}
        resolved = resolver.resolve(params)

        # The ${missing.ref} resolves via resolve_path which returns "missing.ref"
        # for empty outputs, so the template gets replaced with the path string itself
        assert resolved["name"] == "test"
        assert "missing" in resolved["value"]


# =============================================================================
# SkillPlanExecutor Tests
# =============================================================================


@pytest.mark.unit
class TestSkillPlanExecutor:
    """Tests for the SkillPlanExecutor class."""

    def test_constructor_with_no_planner_creates_instance(self):
        """SkillPlanExecutor can be instantiated with just a registry."""
        mock_registry = Mock()
        executor = SkillPlanExecutor(skills_registry=mock_registry)

        assert executor._skills_registry is mock_registry
        assert executor._planner is None
        assert executor._max_steps == 10
        assert executor._enable_replanning is True

    def test_resolve_params_delegates_to_parameter_resolver(self):
        """resolve_params creates a ParameterResolver and calls resolve."""
        mock_registry = Mock()
        executor = SkillPlanExecutor(skills_registry=mock_registry)

        outputs = {"step_0": {"result": "data"}}
        params = {"key": "${step_0.result}"}
        resolved = executor.resolve_params(params, outputs)

        assert resolved["key"] == "data"

    def test_resolve_path_delegates_correctly(self):
        """resolve_path creates a ParameterResolver and resolves the dotted path."""
        mock_registry = Mock()
        executor = SkillPlanExecutor(skills_registry=mock_registry)

        outputs = {"step_1": {"path": "/tmp/file.txt"}}
        result = executor.resolve_path("step_1.path", outputs)

        assert result == "/tmp/file.txt"

    def test_infer_task_type_returns_research_for_keywords(self):
        """infer_task_type detects research tasks via keyword matching."""
        mock_registry = Mock()
        executor = SkillPlanExecutor(skills_registry=mock_registry)
        # Patch planner property to return None so keyword fallback is used
        with patch.object(type(executor), 'planner', new_callable=lambda: property(lambda self: None)):
            result = executor.infer_task_type("Research the latest AI trends")
            assert result == "research"

            result2 = executor.infer_task_type("Find information about Python")
            assert result2 == "research"

    def test_infer_task_type_returns_creation_for_keywords(self):
        """infer_task_type detects creation tasks via keyword matching."""
        mock_registry = Mock()
        executor = SkillPlanExecutor(skills_registry=mock_registry)
        with patch.object(type(executor), 'planner', new_callable=lambda: property(lambda self: None)):
            result = executor.infer_task_type("Create a report about sales")
            assert result == "creation"

            result2 = executor.infer_task_type("Generate a chart showing revenue")
            assert result2 == "creation"

    def test_infer_task_type_returns_unknown_for_gibberish(self):
        """infer_task_type returns 'unknown' when no keywords match."""
        mock_registry = Mock()
        executor = SkillPlanExecutor(skills_registry=mock_registry)
        # Patch the planner property to return None so keyword fallback is used
        # (setting _planner=None triggers lazy-load of a real TaskPlanner)
        with patch.object(type(executor), 'planner', new_callable=lambda: property(lambda self: None)):
            result = executor.infer_task_type("xyzzy plugh foobar")
        assert result == "unknown"

    def test_infer_artifact_tags_returns_tags_based_on_skill_name(self):
        """_infer_artifact_tags assigns semantic tags from skill name keywords."""
        step = Mock()
        step.skill_name = "web-search"
        result_data = {"success": True, "results": []}

        tags = SkillPlanExecutor._infer_artifact_tags(step, result_data)
        assert "research" in tags

        # Test code skill
        step2 = Mock()
        step2.skill_name = "python-executor"
        tags2 = SkillPlanExecutor._infer_artifact_tags(step2, {"success": True})
        assert "code" in tags2

        # Test file output tagging
        step3 = Mock()
        step3.skill_name = "unknown-skill"
        tags3 = SkillPlanExecutor._infer_artifact_tags(step3, {"path": "/tmp/out.txt"})
        assert "file_output" in tags3

        # Test general fallback
        step4 = Mock()
        step4.skill_name = "totally-unrecognized"
        tags4 = SkillPlanExecutor._infer_artifact_tags(step4, {"success": True})
        assert "general" in tags4

    @patch("Jotty.core.agents.base.skill_plan_executor.os.makedirs")
    @patch("Jotty.core.agents.base.skill_plan_executor.Path")
    def test_spill_large_values_replaces_large_strings(self, mock_path_cls, mock_makedirs):
        """_spill_large_values replaces strings exceeding threshold with FileReference."""
        # Build a large string over the threshold
        large_value = "x" * 600_000
        result = {"content": large_value, "status": "ok"}

        # Mock Path(...).write_text to avoid disk I/O
        mock_path_instance = MagicMock()
        mock_path_cls.return_value = mock_path_instance

        spilled = SkillPlanExecutor._spill_large_values(result, threshold=500_000)

        # The large value should be replaced with a FileReference
        assert spilled["status"] == "ok"
        assert spilled["content"] is not large_value

        # Check the replacement is a FileReference (imported from _execution_types)
        from Jotty.core.agents._execution_types import FileReference
        assert isinstance(spilled["content"], FileReference)
        assert spilled["content"].size_bytes > 0
        assert "Spilled large value" in spilled["content"].description

        # Small value should be unchanged
        assert spilled["status"] == "ok"
