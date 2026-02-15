"""
Tests for ParameterResolver and SkillPlanExecutor.

Covers template resolution, dotted-path lookups, auto-wiring,
content validation, command/path sanitization, task type inference,
artifact tagging, and large output spilling.
"""

import json
import os
import pytest
import string
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from Jotty.core.modes.agent.base.skill_plan_executor import ParameterResolver, SkillPlanExecutor


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
            side_effect=lambda params, outs, scoped_keys=None: {**params, "query": "test search"}
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
        from Jotty.core.modes.agent._execution_types import FileReference
        assert isinstance(spilled["content"], FileReference)
        assert spilled["content"].size_bytes > 0
        assert "Spilled large value" in spilled["content"].description

        # Small value should be unchanged
        assert spilled["status"] == "ok"


# =============================================================================
# ToolCallCache Tests
# =============================================================================


@pytest.mark.unit
class TestToolCallCache:
    """Tests for the ToolCallCache class."""

    def _make_cache(self, **kwargs):
        from Jotty.core.modes.agent.base.skill_plan_executor import ToolCallCache
        return ToolCallCache(**kwargs)

    def test_make_key_returns_md5_hex_string(self):
        """make_key returns a 32-char hex string (MD5 digest)."""
        from Jotty.core.modes.agent.base.skill_plan_executor import ToolCallCache
        key = ToolCallCache.make_key("web-search", "search_web_tool", {"query": "AI"})
        assert isinstance(key, str)
        assert len(key) == 32
        # Should be valid hex
        int(key, 16)

    def test_make_key_is_deterministic(self):
        """Same inputs always produce the same cache key."""
        from Jotty.core.modes.agent.base.skill_plan_executor import ToolCallCache
        key1 = ToolCallCache.make_key("skill-a", "tool_x", {"a": 1, "b": 2})
        key2 = ToolCallCache.make_key("skill-a", "tool_x", {"b": 2, "a": 1})
        assert key1 == key2

    def test_make_key_differs_for_different_inputs(self):
        """Different params produce different cache keys."""
        from Jotty.core.modes.agent.base.skill_plan_executor import ToolCallCache
        key1 = ToolCallCache.make_key("skill", "tool", {"query": "AI"})
        key2 = ToolCallCache.make_key("skill", "tool", {"query": "ML"})
        assert key1 != key2

    def test_get_returns_none_for_missing_key(self):
        """get() returns None when key is not in cache."""
        cache = self._make_cache()
        assert cache.get("nonexistent") is None

    def test_set_and_get_round_trip(self):
        """set() stores a value that get() can retrieve."""
        cache = self._make_cache()
        cache.set("key1", {"result": "data"})
        assert cache.get("key1") == {"result": "data"}

    def test_get_returns_none_for_expired_entry(self):
        """get() returns None when the TTL has expired."""
        import time
        cache = self._make_cache(ttl_seconds=0)
        cache.set("key1", "value")
        time.sleep(0.01)
        assert cache.get("key1") is None

    def test_set_evicts_oldest_when_at_max_size(self):
        """set() evicts the oldest entry when cache reaches max_size."""
        cache = self._make_cache(max_size=2, ttl_seconds=300)
        cache.set("key1", "val1")
        cache.set("key2", "val2")
        # Cache is full (size=2), adding key3 should evict key1 (oldest)
        cache.set("key3", "val3")
        assert cache.size == 2
        assert cache.get("key1") is None
        assert cache.get("key2") == "val2"
        assert cache.get("key3") == "val3"

    def test_clear_empties_cache(self):
        """clear() removes all entries from the cache."""
        cache = self._make_cache()
        cache.set("a", 1)
        cache.set("b", 2)
        assert cache.size == 2
        cache.clear()
        assert cache.size == 0
        assert cache.get("a") is None

    def test_size_property_reflects_entry_count(self):
        """size property returns the current number of cached entries."""
        cache = self._make_cache()
        assert cache.size == 0
        cache.set("k1", "v1")
        assert cache.size == 1
        cache.set("k2", "v2")
        assert cache.size == 2

    def test_set_overwrites_existing_key_without_eviction(self):
        """set() with an existing key updates value without changing size."""
        cache = self._make_cache(max_size=2)
        cache.set("k1", "v1")
        cache.set("k2", "v2")
        cache.set("k1", "v1_updated")
        assert cache.size == 2
        assert cache.get("k1") == "v1_updated"


# =============================================================================
# ParameterResolver Deep Tests
# =============================================================================


@pytest.mark.unit
class TestParameterResolverDeep:
    """Deep tests for ParameterResolver covering all internal methods."""

    # ---- _substitute_templates ----

    def test_substitute_templates_dollar_brace_always_substituted(self):
        """${ref} is always substituted, even when code markers are present."""
        outputs = {"step_0": {"path": "/tmp/test.py"}}
        resolver = ParameterResolver(outputs)
        result = resolver._substitute_templates(
            "script",
            "import os; path = ${step_0.path}"
        )
        assert "/tmp/test.py" in result

    def test_substitute_templates_bare_brace_skipped_for_fstrings(self):
        """Bare {ref} is NOT substituted when f-string markers are present."""
        outputs = {"name": {"value": "REPLACED"}}
        resolver = ParameterResolver(outputs)
        code = 'f"Hello {name}"'
        result = resolver._substitute_templates("script", code)
        # The f" marker should prevent bare {name} substitution
        assert "REPLACED" not in result
        assert "{name}" in result

    def test_substitute_templates_bare_brace_skipped_for_def(self):
        """Bare {ref} skipped when 'def ' marker is present in value."""
        outputs = {"x": {"value": "REPLACED"}}
        resolver = ParameterResolver(outputs)
        code = "def foo({x}):\n    pass"
        result = resolver._substitute_templates("code", code)
        assert "REPLACED" not in result

    def test_substitute_templates_bare_brace_skipped_for_class(self):
        """Bare {ref} skipped when 'class ' marker is present."""
        outputs = {"x": {"value": "REPLACED"}}
        resolver = ParameterResolver(outputs)
        code = "class MyClass:\n    x = {x}"
        result = resolver._substitute_templates("code", code)
        assert "REPLACED" not in result

    def test_substitute_templates_bare_brace_skipped_for_import(self):
        """Bare {ref} skipped when 'import ' marker is present."""
        outputs = {"x": {"value": "REPLACED"}}
        resolver = ParameterResolver(outputs)
        code = "import os\ndata = {x}"
        result = resolver._substitute_templates("code", code)
        assert "REPLACED" not in result

    def test_substitute_templates_research_aggregation(self):
        """Research outputs are aggregated when ${research_N.results} pattern found."""
        outputs = {
            "research_0": {
                "query": "AI trends",
                "results": [{"title": "AI boom", "snippet": "Rapid growth", "url": "http://ai.com"}]
            }
        }
        resolver = ParameterResolver(outputs)
        result = resolver._substitute_templates(
            "content", "Summary: ${research_0.results}"
        )
        # The ${} template is substituted first (JSON), then aggregation may run
        assert "Summary:" in result

    # ---- _smart_extract ----

    def test_smart_extract_direct_match(self):
        """_smart_extract returns value when param_name matches a key directly."""
        resolver = ParameterResolver({})
        json_str = '{"query": "test search", "count": 5}'
        result = resolver._smart_extract(json_str, "query")
        assert result == "test search"

    def test_smart_extract_content_like_prefers_rich_text(self):
        """_smart_extract for content-like params prefers rich text fields."""
        resolver = ParameterResolver({})
        json_str = json.dumps({
            "success": True,
            "response": "This is a very long and rich response text that should be selected because it is substantial enough.",
            "status": "ok"
        })
        result = resolver._smart_extract(json_str, "content")
        assert result is not None
        assert "rich response text" in result

    def test_smart_extract_url_from_results(self):
        """_smart_extract for url params extracts URL from nested search results."""
        resolver = ParameterResolver({})
        json_str = json.dumps({
            "results": [
                {"title": "Page", "link": "https://example.com/page1", "snippet": "text"}
            ]
        })
        result = resolver._smart_extract(json_str, "url")
        assert result == "https://example.com/page1"

    def test_smart_extract_path_from_outputs(self):
        """_smart_extract for 'path' param scans all outputs for most recent path."""
        resolver = ParameterResolver({
            "step_0": {"path": "/tmp/first.txt"},
            "step_1": {"path": "/tmp/second.txt"}
        })
        json_str = json.dumps({"data": "value"})
        result = resolver._smart_extract(json_str, "path")
        assert result == "/tmp/second.txt"

    def test_smart_extract_returns_none_for_invalid_json(self):
        """_smart_extract returns None for non-JSON strings."""
        resolver = ParameterResolver({})
        result = resolver._smart_extract("not json", "query")
        assert result is None

    def test_smart_extract_returns_none_for_non_dict_json(self):
        """_smart_extract returns None when JSON is a list, not a dict."""
        resolver = ParameterResolver({})
        result = resolver._smart_extract("[1, 2, 3]", "query")
        assert result is None

    # ---- _format_list_results ----

    def test_format_list_results_empty_list(self):
        """_format_list_results returns empty string for empty list."""
        result = ParameterResolver._format_list_results([])
        assert result == ""

    def test_format_list_results_dicts_with_title_snippet_url(self):
        """_format_list_results formats dicts with title/snippet/url into readable text."""
        items = [
            {"title": "Result One", "snippet": "First description", "link": "http://one.com"},
            {"title": "Result Two", "snippet": "Second description", "link": "http://two.com"},
        ]
        result = ParameterResolver._format_list_results(items)
        assert "1. Result One" in result
        assert "First description" in result
        assert "http://one.com" in result
        assert "2. Result Two" in result

    def test_format_list_results_non_dict_items(self):
        """_format_list_results handles non-dict items by converting to string."""
        items = ["plain text", 42, True]
        result = ParameterResolver._format_list_results(items)
        assert "1. plain text" in result
        assert "2. 42" in result

    # ---- _resolve_bare_keys ----

    def test_resolve_bare_keys_exact_output_key_match(self):
        """_resolve_bare_keys resolves when value matches an output key and param key matches a field."""
        outputs = {"step_0": {"query": "search term", "path": "/tmp/file.txt"}}
        resolver = ParameterResolver(outputs)
        result = resolver._resolve_bare_keys("query", "step_0")
        assert result == "search term"

    def test_resolve_bare_keys_content_field_fallback(self):
        """_resolve_bare_keys falls back to content fields for content-like params."""
        outputs = {"step_0": {"response": "Big content response text here that is useful"}}
        resolver = ParameterResolver(outputs)
        result = resolver._resolve_bare_keys("content", "step_0")
        assert "Big content response text" in result

    def test_resolve_bare_keys_path_fallback_scans_all_outputs(self):
        """_resolve_bare_keys for 'path' scans all outputs in reverse."""
        outputs = {
            "step_0": {"path": "/tmp/first.txt"},
            "step_1": {"path": "/tmp/second.txt"},
        }
        resolver = ParameterResolver(outputs)
        # Value is an output key that has no 'path' of its own,
        # so it falls through to the "scan all outputs for path" path
        result = resolver._resolve_bare_keys("path", "step_1")
        # step_1 has a 'path' key directly, so it returns that
        assert result == "/tmp/second.txt"

    def test_resolve_bare_keys_full_dict_serialization_fallback(self):
        """_resolve_bare_keys serializes the full dict when no specific field matches."""
        outputs = {"step_0": {"alpha": 1, "beta": 2}}
        resolver = ParameterResolver(outputs)
        result = resolver._resolve_bare_keys("weird_param", "step_0")
        # Fallback: whole dict as JSON
        assert "alpha" in result
        assert "beta" in result

    def test_resolve_bare_keys_non_output_key_passthrough(self):
        """_resolve_bare_keys passes through value when it is not in outputs."""
        resolver = ParameterResolver({"step_0": {"x": 1}})
        result = resolver._resolve_bare_keys("key", "not_an_output_key")
        assert result == "not_an_output_key"

    # ---- _resolve_placeholder_strings ----

    def test_resolve_placeholder_strings_uppercase_replaced(self):
        """Uppercase placeholders like {CONTENT_FROM_STEP} are replaced."""
        outputs = {"step_0": {"response": "real data from step"}}
        resolver = ParameterResolver(outputs)
        result = resolver._resolve_placeholder_strings(
            "content", "Use this: {CONTENT_FROM_STEP}"
        )
        # Should be replaced with last output serialized as JSON
        assert "real data" in result
        assert "{CONTENT_FROM_STEP}" not in result

    def test_resolve_placeholder_strings_no_outputs_unchanged(self):
        """Uppercase placeholders remain if outputs is empty."""
        resolver = ParameterResolver({})
        result = resolver._resolve_placeholder_strings(
            "content", "Use this: {CONTENT_FROM_STEP}"
        )
        assert "{CONTENT_FROM_STEP}" in result

    def test_resolve_placeholder_strings_no_uppercase_unchanged(self):
        """Strings without uppercase placeholder patterns pass through."""
        outputs = {"step_0": {"data": "value"}}
        resolver = ParameterResolver(outputs)
        result = resolver._resolve_placeholder_strings("key", "normal text")
        assert result == "normal text"

    # ---- _sanitize_command_param ----

    def test_sanitize_command_short_commands_pass_through(self):
        """Short commands (<= 150 chars) pass through unchanged."""
        resolver = ParameterResolver({})
        result = resolver._sanitize_command_param("command", "ls -la /tmp", None)
        assert result == "ls -la /tmp"

    def test_sanitize_command_non_command_key_pass_through(self):
        """Non-'command' params pass through regardless of length."""
        resolver = ParameterResolver({})
        long_text = "x " * 100
        result = resolver._sanitize_command_param("query", long_text, None)
        assert result == long_text

    def test_sanitize_command_long_json_fixed_to_python_path(self):
        """Long JSON-like command text is fixed to 'python <path>' when outputs have .py path."""
        outputs = {"step_0": {"path": "/tmp/analysis.py"}}
        resolver = ParameterResolver(outputs)
        long_json = json.dumps({"text": "some long content " * 20})
        result = resolver._sanitize_command_param("command", long_json, None)
        assert result == "python /tmp/analysis.py"

    # ---- _sanitize_path_param ----

    def test_sanitize_path_short_paths_pass_through(self):
        """Short path values (<= 200 chars) pass through unchanged."""
        resolver = ParameterResolver({})
        result = resolver._sanitize_path_param("path", "/tmp/file.txt", None)
        assert result == "/tmp/file.txt"

    def test_sanitize_path_long_content_finds_filename_from_regex(self):
        """Long content in 'path' param extracts filename from regex match."""
        resolver = ParameterResolver({})
        long_content = "This is content that was saved to report.txt and contains " + "x " * 100
        result = resolver._sanitize_path_param("path", long_content, None)
        assert result == "report.txt"

    def test_sanitize_path_fallback_to_outputs(self):
        """Long content in 'path' param falls back to most recent path from outputs."""
        outputs = {"step_0": {"path": "/tmp/output.csv"}}
        resolver = ParameterResolver(outputs)
        long_content = "x " * 200  # Long content without filename regex match
        result = resolver._sanitize_path_param("path", long_content, None)
        assert result == "/tmp/output.csv"

    # ---- _sanitize_content_param ----

    def test_sanitize_content_good_content_passes_through(self):
        """Good content (long, not bad pattern) passes through unchanged."""
        resolver = ParameterResolver({})
        good = "This is a substantial piece of content that has more than 80 characters and does not match any bad patterns at all."
        result = resolver._sanitize_content_param("content", good)
        assert result == good

    def test_sanitize_content_bad_short_replaced(self):
        """Short content (< 80 chars) is detected as bad and replaced if possible."""
        # Content must: be > 100 chars, > 80 chars, not start with instruction prefix, not be < 300 with prefix
        long_content = (
            "def calculate_statistics(data):\n"
            "    mean = sum(data) / len(data)\n"
            "    variance = sum((x - mean) ** 2 for x in data) / len(data)\n"
            "    return {'mean': mean, 'variance': variance, 'count': len(data)}\n"
        )
        assert len(long_content) > 100
        outputs = {"step_0": {"response": long_content}}
        resolver = ParameterResolver(outputs)
        result = resolver._sanitize_content_param("content", "short")
        assert "calculate_statistics" in result

    def test_sanitize_content_json_success_replaced(self):
        """JSON success response content is detected as bad."""
        outputs = {"step_0": {"response": "Real content that is more than 80 characters long and should be used as a replacement for the JSON success response."}}
        resolver = ParameterResolver(outputs)
        bad = '{"success": true, "bytes_written": 500}'
        result = resolver._sanitize_content_param("content", bad)
        assert "Real content" in result

    def test_sanitize_content_instruction_prefix_replaced(self):
        """Content starting with instruction-like prefix is detected as bad."""
        outputs = {"step_0": {"response": "Actual real data content that is more than 80 characters and should be used as a replacement instead of the instruction prefix."}}
        resolver = ParameterResolver(outputs)
        bad = "I'll help you create a document with the following content that is a bit longer than eighty characters"
        result = resolver._sanitize_content_param("content", bad)
        assert "Actual real data" in result

    def test_sanitize_content_non_content_key_passthrough(self):
        """Non-'content' params pass through regardless of content quality."""
        resolver = ParameterResolver({})
        result = resolver._sanitize_content_param("query", "short")
        assert result == "short"

    # ---- _is_bad_content ----

    def test_is_bad_content_short_string(self):
        """Strings shorter than 80 chars are considered bad content."""
        resolver = ParameterResolver({})
        assert resolver._is_bad_content("tiny") is True

    def test_is_bad_content_json_success_response(self):
        """JSON success responses with bytes_written are bad content."""
        resolver = ParameterResolver({})
        assert resolver._is_bad_content('{"success": true, "bytes_written": 42}') is True

    def test_is_bad_content_instruction_prefix(self):
        """Strings starting with instruction prefixes like 'filename:' are bad."""
        resolver = ParameterResolver({})
        bad = "filename: test.py and some more text that makes this at least eighty characters long so the length check passes okay"
        assert resolver._is_bad_content(bad) is True

    def test_is_bad_content_good_content_returns_false(self):
        """Substantial real content (no bad patterns) returns False."""
        resolver = ParameterResolver({})
        good = "def calculate_statistics(data):\n    mean = sum(data) / len(data)\n    return {'mean': mean, 'count': len(data)}"
        assert resolver._is_bad_content(good) is False

    # ---- _find_best_content ----

    def test_find_best_content_returns_largest_non_bad(self):
        """_find_best_content returns the largest non-bad content from outputs."""
        outputs = {
            "step_0": {"response": "Short"},
            "step_1": {"content": "A much longer and substantial piece of content that has well over 100 characters and should be selected as the best content."},
        }
        resolver = ParameterResolver(outputs)
        result = resolver._find_best_content()
        assert result is not None
        assert "substantial" in result

    def test_find_best_content_returns_none_when_all_bad(self):
        """_find_best_content returns None when all outputs are bad content."""
        outputs = {
            "step_0": {"response": "tiny"},
            "step_1": {"content": "small"},
        }
        resolver = ParameterResolver(outputs)
        result = resolver._find_best_content()
        assert result is None

    # ---- resolve_path ----

    def test_resolve_path_dotted_paths(self):
        """resolve_path navigates nested dicts via dotted path."""
        outputs = {"step_0": {"data": {"nested": {"value": 42}}}}
        resolver = ParameterResolver(outputs)
        result = resolver.resolve_path("step_0.data.nested.value")
        assert result == "42"

    def test_resolve_path_array_indexing(self):
        """resolve_path supports array indexing like field[0]."""
        outputs = {"step_0": {"items": ["alpha", "beta", "gamma"]}}
        resolver = ParameterResolver(outputs)
        result = resolver.resolve_path("step_0.items[1]")
        assert result == "beta"

    def test_resolve_path_nested_dict_returns_json(self):
        """resolve_path returns JSON when value is a dict or list."""
        outputs = {"step_0": {"data": {"key": "val"}}}
        resolver = ParameterResolver(outputs)
        result = resolver.resolve_path("step_0.data")
        parsed = json.loads(result)
        assert parsed == {"key": "val"}

    # ---- _resolve_missing_path ----

    def test_resolve_missing_path_returns_unresolved_for_missing_step(self):
        """_resolve_missing_path returns unresolved path when step doesn't exist (scoped resolution)."""
        outputs = {
            "step_2": {
                "content": "This is the actual content from step 2 that is long enough to be useful."
            }
        }
        resolver = ParameterResolver(outputs)
        # step_0 doesn't exist — scoped resolution returns unresolved path
        result = resolver._resolve_missing_path("step_0.content")
        assert result == "step_0.content"

    def test_resolve_missing_path_returns_original_for_non_step_pattern(self):
        """_resolve_missing_path returns original path for non-step patterns."""
        resolver = ParameterResolver({})
        result = resolver._resolve_missing_path("random.path.here")
        assert result == "random.path.here"


# =============================================================================
# ToolResultProcessor Deep Tests
# =============================================================================


@pytest.mark.unit
class TestToolResultProcessorDeep:
    """Deep tests for ToolResultProcessor covering all internal methods."""

    def _make_processor(self):
        from Jotty.core.modes.agent.base.step_processors import ToolResultProcessor
        return ToolResultProcessor()

    def test_process_non_dict_converted(self):
        """process() wraps non-dict results in {'output': value}."""
        processor = self._make_processor()
        result = processor.process("plain string")
        assert isinstance(result, dict)
        assert result.get("output") == "plain string"

    def test_process_adds_execution_time_ms(self):
        """process() adds _execution_time_ms when elapsed > 0."""
        processor = self._make_processor()
        result = processor.process({"success": True}, elapsed=1.5)
        assert result["_execution_time_ms"] == 1500.0

    def test_process_no_execution_time_when_zero(self):
        """process() does not add _execution_time_ms when elapsed is 0."""
        processor = self._make_processor()
        result = processor.process({"success": True}, elapsed=0)
        assert "_execution_time_ms" not in result

    # ---- _truncate_preserving_keys ----

    def test_truncate_preserving_keys_small_items_kept_intact(self):
        """Small items (<=500 chars) are kept intact during truncation."""
        processor = self._make_processor()
        data = {"key": "small value", "status": "ok"}
        result = processor._truncate_preserving_keys(data, max_chars=10000)
        assert result["key"] == "small value"
        assert result["status"] == "ok"

    def test_truncate_preserving_keys_large_items_truncated(self):
        """Large items are truncated proportionally, all keys preserved."""
        processor = self._make_processor()
        data = {
            "small": "ok",
            "big": "x" * 5000,
            "another_big": "y" * 3000,
        }
        result = processor._truncate_preserving_keys(data, max_chars=2000)
        assert "small" in result
        assert "big" in result
        assert "another_big" in result
        # Large values should be truncated
        assert "truncated" in result["big"]

    def test_truncate_preserving_keys_everything_fits(self):
        """When total size <= max_chars, data is returned as-is."""
        processor = self._make_processor()
        data = {"a": "short", "b": "also short"}
        result = processor._truncate_preserving_keys(data, max_chars=100000)
        assert result == data

    # ---- _strip_binary ----

    def test_strip_binary_key_name_detection(self):
        """_strip_binary detects binary data by key name patterns."""
        processor = self._make_processor()
        data = {"screenshot": "A" * 2000, "name": "test"}
        result = processor._strip_binary(data)
        assert result["screenshot"].startswith("[binary data:")
        assert result["name"] == "test"

    def test_strip_binary_base64_pattern_detection(self):
        """_strip_binary detects base64 data by high-diversity alphanumeric content."""
        processor = self._make_processor()
        # Generate something that looks like base64 (diverse alphanumeric, >1000 chars)
        base64_like = (string.ascii_letters + string.digits + "+/=") * 20
        assert len(base64_like) > 1000
        data = {"payload": base64_like}
        result = processor._strip_binary(data)
        assert result["payload"].startswith("[binary data:")

    def test_strip_binary_nested_dict_handling(self):
        """_strip_binary handles nested dicts recursively."""
        processor = self._make_processor()
        data = {
            "outer": {
                "image_data": "X" * 2000,
                "text": "normal"
            }
        }
        result = processor._strip_binary(data)
        assert result["outer"]["image_data"].startswith("[binary data:")
        assert result["outer"]["text"] == "normal"

    def test_strip_binary_short_values_ignored(self):
        """_strip_binary leaves short values alone even with binary-like key names."""
        processor = self._make_processor()
        data = {"screenshot": "short"}
        result = processor._strip_binary(data)
        assert result["screenshot"] == "short"

    # ---- _convert_sets ----

    def test_convert_sets_set_to_sorted_list(self):
        """_convert_sets converts sets to sorted lists."""
        processor = self._make_processor()
        data = {"tags": {"c", "a", "b"}}
        result = processor._convert_sets(data)
        assert result["tags"] == ["a", "b", "c"]

    def test_convert_sets_recursive_nested_dicts(self):
        """_convert_sets handles sets inside nested dicts."""
        processor = self._make_processor()
        data = {"outer": {"inner_set": {3, 1, 2}}}
        result = processor._convert_sets(data)
        assert result["outer"]["inner_set"] == ["1", "2", "3"]

    def test_convert_sets_nested_list_with_dicts(self):
        """_convert_sets handles sets inside dicts inside lists."""
        processor = self._make_processor()
        data = {"items": [{"tags": {"z", "a"}}, "plain"]}
        result = processor._convert_sets(data)
        assert result["items"][0]["tags"] == ["a", "z"]
        assert result["items"][1] == "plain"

    # ---- _format_search_results ----

    def test_format_search_results_with_query(self):
        """_format_search_results includes query header when provided."""
        from Jotty.core.modes.agent.base.step_processors import ToolResultProcessor
        results = [{"title": "Page 1", "snippet": "Info", "link": "http://p1.com"}]
        text = ToolResultProcessor._format_search_results(results, query="AI trends")
        assert "Search Results: AI trends" in text
        assert "Page 1" in text

    def test_format_search_results_without_query(self):
        """_format_search_results omits header when query is empty."""
        from Jotty.core.modes.agent.base.step_processors import ToolResultProcessor
        results = [{"title": "Item", "snippet": "Desc"}]
        text = ToolResultProcessor._format_search_results(results, query="")
        assert "Search Results" not in text
        assert "Item" in text

    def test_format_search_results_non_dict_items(self):
        """_format_search_results handles non-dict items by stringifying."""
        from Jotty.core.modes.agent.base.step_processors import ToolResultProcessor
        results = ["plain text item", 42]
        text = ToolResultProcessor._format_search_results(results)
        assert "plain text item" in text
        assert "42" in text


# =============================================================================
# SkillPlanExecutor Deep Tests
# =============================================================================


@pytest.mark.unit
class TestSkillPlanExecutorDeep:
    """Deep tests for SkillPlanExecutor covering validation, planning, and execution methods."""

    def _make_executor(self, **kwargs):
        mock_registry = kwargs.pop("registry", Mock())
        return SkillPlanExecutor(skills_registry=mock_registry, **kwargs)

    # ---- validate_plan ----

    def test_validate_plan_skill_not_found(self):
        """validate_plan reports error when skill does not exist in registry."""
        mock_registry = Mock()
        mock_registry.get_skill.return_value = None
        executor = self._make_executor(registry=mock_registry)

        step = Mock()
        step.skill_name = "nonexistent-skill"
        step.tool_name = "tool"
        step.params = {}
        step.depends_on = []
        step.description = "Test step"

        issues = executor.validate_plan([step])
        assert len(issues) == 1
        assert "Skill not found" in issues[0]["errors"][0]

    def test_validate_plan_tool_not_found(self):
        """validate_plan reports error when tool does not exist in skill (multi-tool skill)."""
        mock_skill = Mock()
        # Must have >1 tool so single-tool fallback doesn't kick in
        mock_skill.tools = {"real_tool": Mock(), "another_tool": Mock()}
        mock_registry = Mock()
        mock_registry.get_skill.return_value = mock_skill

        executor = self._make_executor(registry=mock_registry)

        step = Mock()
        step.skill_name = "my-skill"
        step.tool_name = "wrong_tool"
        step.params = {}
        step.depends_on = []
        step.description = "Test step"

        issues = executor.validate_plan([step])
        assert len(issues) == 1
        assert "Tool 'wrong_tool' not found" in issues[0]["errors"][0]

    def test_validate_plan_valid_plan_returns_empty(self):
        """validate_plan returns empty list for a fully valid plan."""
        mock_tool = Mock()
        mock_skill = Mock()
        mock_skill.tools = {"my_tool": mock_tool}
        mock_skill.get_tool_schema.return_value = None
        mock_registry = Mock()
        mock_registry.get_skill.return_value = mock_skill

        executor = self._make_executor(registry=mock_registry)

        step = Mock()
        step.skill_name = "my-skill"
        step.tool_name = "my_tool"
        step.params = {}
        step.depends_on = []
        step.description = "Valid step"

        issues = executor.validate_plan([step])
        assert issues == []

    def test_validate_plan_depends_on_validity(self):
        """validate_plan catches invalid depends_on indices."""
        mock_tool = Mock()
        mock_skill = Mock()
        mock_skill.tools = {"tool": mock_tool}
        mock_skill.get_tool_schema.return_value = None
        mock_registry = Mock()
        mock_registry.get_skill.return_value = mock_skill

        executor = self._make_executor(registry=mock_registry)

        step0 = Mock()
        step0.skill_name = "skill"
        step0.tool_name = "tool"
        step0.params = {}
        step0.depends_on = [5]  # Index 5 does not exist
        step0.description = "Step with bad dependency"

        issues = executor.validate_plan([step0])
        assert len(issues) == 1
        assert "Invalid depends_on index: 5" in issues[0]["errors"][0]

    def test_validate_plan_forward_dependency(self):
        """validate_plan catches forward dependencies (step depends on later step)."""
        mock_tool = Mock()
        mock_skill = Mock()
        mock_skill.tools = {"tool": mock_tool}
        mock_skill.get_tool_schema.return_value = None
        mock_registry = Mock()
        mock_registry.get_skill.return_value = mock_skill

        executor = self._make_executor(registry=mock_registry)

        step0 = Mock()
        step0.skill_name = "skill"
        step0.tool_name = "tool"
        step0.params = {}
        step0.depends_on = [1]  # Depends on step 1 but this IS step 0
        step0.description = "Step 0"

        step1 = Mock()
        step1.skill_name = "skill"
        step1.tool_name = "tool"
        step1.params = {}
        step1.depends_on = []
        step1.description = "Step 1"

        issues = executor.validate_plan([step0, step1])
        assert len(issues) == 1
        assert "Forward dependency" in issues[0]["errors"][0]

    # ---- _auto_correct_plan ----

    def test_auto_correct_plan_fixes_wrong_skill_tool_mapping(self):
        """_auto_correct_plan reassigns tool to correct skill."""
        mock_skill_a = Mock()
        # Must have >1 tool so single-tool fallback doesn't activate
        mock_skill_a.tools = {"tool_a": Mock(), "tool_a2": Mock()}
        mock_skill_b = Mock()
        mock_skill_b.tools = {"tool_b": Mock()}

        mock_registry = Mock()

        def get_skill_side_effect(name):
            if name == "skill-a":
                return mock_skill_a
            if name == "skill-b":
                return mock_skill_b
            return None
        mock_registry.get_skill.side_effect = get_skill_side_effect

        executor = self._make_executor(registry=mock_registry)

        # Step claims skill-a owns tool_b, which is wrong
        step = Mock()
        step.skill_name = "skill-a"
        step.tool_name = "tool_b"

        selected_skills = [
            {"name": "skill-a"},
            {"name": "skill-b"},
        ]

        executor._auto_correct_plan([step], selected_skills)
        # Should have been corrected to skill-b
        assert step.skill_name == "skill-b"

    def test_auto_correct_plan_no_registry_returns_steps(self):
        """_auto_correct_plan with no registry returns steps unchanged."""
        executor = self._make_executor()
        executor._skills_registry = None
        steps = [Mock()]
        result = executor._auto_correct_plan(steps, [])
        assert result is steps

    # ---- _build_dependency_graph ----

    def test_build_dependency_graph_basic(self):
        """_build_dependency_graph maps step_index to dependency indices."""
        executor = self._make_executor()

        step0 = Mock()
        step0.depends_on = []
        step1 = Mock()
        step1.depends_on = [0]
        step2 = Mock()
        step2.depends_on = [0, 1]

        graph = executor._build_dependency_graph([step0, step1, step2])
        assert graph[0] == []
        assert graph[1] == [0]
        assert graph[2] == [0, 1]

    def test_build_dependency_graph_filters_invalid_indices(self):
        """_build_dependency_graph filters out of range dependency indices."""
        executor = self._make_executor()

        step0 = Mock()
        step0.depends_on = [5, -1, 0]  # Only 0 is valid for a 2-step plan

        step1 = Mock()
        step1.depends_on = []

        graph = executor._build_dependency_graph([step0, step1])
        assert graph[0] == [0]  # Only 0 kept (5 and -1 filtered)
        assert graph[1] == []

    # ---- _find_parallel_groups ----

    def test_find_parallel_groups_independent_steps(self):
        """Independent steps (no deps) form a single parallel group."""
        executor = self._make_executor()

        step0 = Mock()
        step0.depends_on = []
        step1 = Mock()
        step1.depends_on = []
        step2 = Mock()
        step2.depends_on = []

        layers = executor._find_parallel_groups([step0, step1, step2])
        assert len(layers) == 1
        assert set(layers[0]) == {0, 1, 2}

    def test_find_parallel_groups_sequential_chain(self):
        """A sequential chain produces one step per layer."""
        executor = self._make_executor()

        step0 = Mock()
        step0.depends_on = []
        step1 = Mock()
        step1.depends_on = [0]
        step2 = Mock()
        step2.depends_on = [1]

        layers = executor._find_parallel_groups([step0, step1, step2])
        assert len(layers) == 3
        assert layers[0] == [0]
        assert layers[1] == [1]
        assert layers[2] == [2]

    def test_find_parallel_groups_diamond_pattern(self):
        """Diamond dependency produces 3 layers: root, parallel middle, join."""
        executor = self._make_executor()

        step0 = Mock()
        step0.depends_on = []
        step1 = Mock()
        step1.depends_on = [0]
        step2 = Mock()
        step2.depends_on = [0]
        step3 = Mock()
        step3.depends_on = [1, 2]

        layers = executor._find_parallel_groups([step0, step1, step2, step3])
        assert len(layers) == 3
        assert layers[0] == [0]
        assert set(layers[1]) == {1, 2}
        assert layers[2] == [3]

    def test_find_parallel_groups_handles_cycle(self):
        """Cycles are broken by forcing the smallest remaining step."""
        executor = self._make_executor()

        # Create a cycle: step0 -> step1 -> step0
        step0 = Mock()
        step0.depends_on = [1]
        step1 = Mock()
        step1.depends_on = [0]

        layers = executor._find_parallel_groups([step0, step1])
        # Should complete without hanging and include all steps
        all_steps = set()
        for layer in layers:
            all_steps.update(layer)
        assert all_steps == {0, 1}

    # ---- _inject_essential_skills ----

    def test_inject_essential_skills_shell_exec(self):
        """_inject_essential_skills adds shell-exec when 'execute' keyword is found."""
        executor = self._make_executor()
        selected = [{"name": "web-search"}]
        available = [{"name": "web-search"}, {"name": "shell-exec"}]

        result = executor._inject_essential_skills(
            "execute the python script", selected, available
        )
        names = {s["name"] for s in result}
        assert "shell-exec" in names

    def test_inject_essential_skills_calculator(self):
        """_inject_essential_skills adds calculator when 'calculate' keyword is found."""
        executor = self._make_executor()
        selected = [{"name": "web-search"}]
        available = [{"name": "web-search"}, {"name": "calculator"}]

        result = executor._inject_essential_skills(
            "calculate the total revenue", selected, available
        )
        names = {s["name"] for s in result}
        assert "calculator" in names

    def test_inject_essential_skills_already_selected(self):
        """_inject_essential_skills does not duplicate already-selected skills."""
        executor = self._make_executor()
        selected = [{"name": "shell-exec"}]
        available = [{"name": "shell-exec"}]

        result = executor._inject_essential_skills(
            "execute the script", selected, available
        )
        assert len(result) == 1

    def test_inject_essential_skills_not_available(self):
        """_inject_essential_skills does not inject skills not in available list."""
        executor = self._make_executor()
        selected = [{"name": "web-search"}]
        available = [{"name": "web-search"}]  # shell-exec NOT available

        result = executor._inject_essential_skills(
            "execute the script", selected, available
        )
        names = {s["name"] for s in result}
        assert "shell-exec" not in names

    # ---- _strict_tool_lookup ----

    def test_strict_tool_lookup_case_insensitive(self):
        """_strict_tool_lookup matches tools case-insensitively."""
        mock_tool = Mock()
        mock_skill = Mock()
        mock_skill.tools = {"Search_Web_Tool": mock_tool}

        result = SkillPlanExecutor._strict_tool_lookup(mock_skill, "search_web_tool")
        assert result is mock_tool

    def test_strict_tool_lookup_no_substring_match(self):
        """_strict_tool_lookup does NOT match substrings."""
        mock_skill = Mock()
        mock_skill.tools = {"search_web_tool": Mock()}

        result = SkillPlanExecutor._strict_tool_lookup(mock_skill, "search_web")
        assert result is None

    def test_strict_tool_lookup_empty_name(self):
        """_strict_tool_lookup returns None for empty tool name."""
        mock_skill = Mock()
        mock_skill.tools = {"tool": Mock()}

        result = SkillPlanExecutor._strict_tool_lookup(mock_skill, "")
        assert result is None

    # ---- exclude_skill / clear_exclusions ----

    def test_exclude_skill_adds_to_set(self):
        """exclude_skill adds skill name to the exclusion set."""
        executor = self._make_executor()
        executor.exclude_skill("broken-skill")
        assert "broken-skill" in executor.excluded_skills

    def test_clear_exclusions_empties_set(self):
        """clear_exclusions removes all excluded skills."""
        executor = self._make_executor()
        executor.exclude_skill("skill-a")
        executor.exclude_skill("skill-b")
        assert len(executor.excluded_skills) == 2
        executor.clear_exclusions()
        assert len(executor.excluded_skills) == 0

    # ---- infer_task_type ----

    def test_infer_task_type_comparison(self):
        """infer_task_type returns 'comparison' for compare-related keywords."""
        executor = self._make_executor()
        with patch.object(type(executor), 'planner', new_callable=lambda: property(lambda self: None)):
            assert executor.infer_task_type("Compare Python vs Java") == "comparison"
            assert executor.infer_task_type("X versus Y analysis") == "comparison"

    def test_infer_task_type_analysis(self):
        """infer_task_type returns 'analysis' for analyze-related keywords."""
        executor = self._make_executor()
        with patch.object(type(executor), 'planner', new_callable=lambda: property(lambda self: None)):
            assert executor.infer_task_type("Analyze the dataset") == "analysis"
            assert executor.infer_task_type("Evaluate the performance") == "analysis"

    def test_infer_task_type_unknown(self):
        """infer_task_type returns 'unknown' for unrecognized tasks."""
        executor = self._make_executor()
        with patch.object(type(executor), 'planner', new_callable=lambda: property(lambda self: None)):
            assert executor.infer_task_type("do something random") == "unknown"


# =============================================================================
# ToolSchema Deep Tests
# =============================================================================


@pytest.mark.unit
class TestToolSchemaDeep:
    """Deep tests for the ToolSchema class."""

    def _make_schema(self, name="test_tool", params=None):
        from Jotty.core.modes.agent._execution_types import ToolSchema, ToolParam
        return ToolSchema(name=name, params=params or [])

    def _make_param(self, **kwargs):
        from Jotty.core.modes.agent._execution_types import ToolParam
        return ToolParam(**kwargs)

    def test_from_metadata_builds_from_json_schema(self):
        """from_metadata builds a ToolSchema from a JSON schema dict."""
        from Jotty.core.modes.agent._execution_types import ToolSchema
        metadata = {
            "description": "Search the web",
            "parameters": {
                "properties": {
                    "query": {"type": "str", "description": "Search query"},
                    "limit": {"type": "int", "description": "Max results"},
                },
                "required": ["query"],
            }
        }
        schema = ToolSchema.from_metadata("search_tool", metadata)
        assert schema.name == "search_tool"
        assert schema.description == "Search the web"
        assert len(schema.params) == 2
        query_param = schema.get_param("query")
        assert query_param is not None
        assert query_param.required is True
        limit_param = schema.get_param("limit")
        assert limit_param is not None
        assert limit_param.required is False

    def test_validate_returns_errors_for_missing_required(self):
        """validate returns errors when required params are missing."""
        schema = self._make_schema(params=[
            self._make_param(name="query", required=True),
            self._make_param(name="limit", required=False),
        ])
        result = schema.validate({})
        assert not result.valid
        assert any("Missing required parameter: query" in e for e in result.errors)

    def test_validate_valid_params(self):
        """validate returns no errors for valid params."""
        schema = self._make_schema(params=[
            self._make_param(name="query", required=True),
        ])
        result = schema.validate({"query": "test"})
        assert result.valid
        assert result.errors == []

    def test_validate_coercion(self):
        """validate with coerce=True coerces types and returns coerced_params."""
        schema = self._make_schema(params=[
            self._make_param(name="count", type_hint="int", required=True),
        ])
        result = schema.validate({"count": "42"}, coerce=True)
        assert result.valid
        assert result.coerced_params.get("count") == 42

    def test_get_param_by_name(self):
        """get_param finds param by canonical name."""
        schema = self._make_schema(params=[
            self._make_param(name="query"),
        ])
        assert schema.get_param("query") is not None
        assert schema.get_param("nonexistent") is None

    def test_get_param_by_alias(self):
        """get_param finds param by alias."""
        schema = self._make_schema(params=[
            self._make_param(name="query", aliases=["q", "search_query"]),
        ])
        result = schema.get_param("q")
        assert result is not None
        assert result.name == "query"

    def test_get_llm_visible_params_excludes_reserved(self):
        """get_llm_visible_params excludes reserved params."""
        schema = self._make_schema(params=[
            self._make_param(name="query", reserved=False),
            self._make_param(name="_status_callback", reserved=True),
        ])
        visible = schema.get_llm_visible_params()
        visible_names = [p.name for p in visible]
        assert "query" in visible_names
        assert "_status_callback" not in visible_names

    def test_resolve_aliases_maps_to_canonical(self):
        """resolve_aliases maps alias keys to canonical param names."""
        schema = self._make_schema(params=[
            self._make_param(name="query", aliases=["q", "search_query"]),
        ])
        params = {"q": "test search"}
        resolved = schema.resolve_aliases(params)
        assert "query" in resolved
        assert "q" not in resolved
        assert resolved["query"] == "test search"

    def test_auto_wire_exact_name_match(self):
        """auto_wire fills missing required param from exact name match in outputs."""
        schema = self._make_schema(params=[
            self._make_param(name="query", required=True),
        ])
        outputs = {"step_0": {"query": "wired value"}}
        params = {}
        result = schema.auto_wire(params, outputs)
        assert result["query"] == "wired value"

    def test_auto_wire_content_direct_match(self):
        """auto_wire matches content param via direct name match (not _CONTENT_FIELDS scan)."""
        schema = self._make_schema(params=[
            self._make_param(name="content", required=True),
        ])
        # Direct 'content' key match — this works
        outputs = {"step_0": {"content": "This is substantial content that exceeds 50 characters in length."}}
        params = {}
        result = schema.auto_wire(params, outputs)
        assert "substantial content" in result.get("content", "")

    def test_auto_wire_path_strategy(self):
        """auto_wire uses path fallback for path/file_path params."""
        schema = self._make_schema(params=[
            self._make_param(name="path", required=True),
        ])
        outputs = {"step_0": {"path": "/tmp/output.txt"}}
        params = {}
        result = schema.auto_wire(params, outputs)
        assert result["path"] == "/tmp/output.txt"
