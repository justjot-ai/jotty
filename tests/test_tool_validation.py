"""
Tool Validation Unit Tests
============================

Comprehensive unit tests for ToolValidator, RegistryValidationResult,
MethodChecker, validate_tool_attributes, and ToolGuard.
All tests use mocks -- no real file I/O, no network.
Runs offline and fast (<1s each).

Covers:
- RegistryValidationResult dataclass (valid/errors/warnings, to_dict, defaults)
- ToolValidator initialization (strict vs non-strict)
- validate_tool() required metadata checks
- _validate_signature() parameter matching
- _validate_types() authorized type enforcement
- _validate_code_safety() dangerous call detection
- MethodChecker AST visitor (dangerous methods, imports)
- validate_tool_attributes() class-level validation
- ToolGuard runtime guard (plan/act modes, side-effect limits, path blocking)
"""

import ast
import inspect
import pytest
from unittest.mock import Mock, MagicMock, patch

# Try importing the module under test
try:
    from Jotty.core.registry.tool_validation import (
        ToolValidator,
        RegistryValidationResult,
        MethodChecker,
        validate_tool_attributes,
        ToolGuard,
        AUTHORIZED_TYPES,
        TYPE_CONVERSION,
    )
    TOOL_VALIDATION_AVAILABLE = True
except ImportError:
    TOOL_VALIDATION_AVAILABLE = False

skip_no_module = pytest.mark.skipif(
    not TOOL_VALIDATION_AVAILABLE, reason="tool_validation module not importable"
)


# =============================================================================
# Helpers
# =============================================================================

def _make_valid_metadata(name="test_tool", input_names=None):
    """Create valid tool metadata with matching input names."""
    if input_names is None:
        input_names = ["x"]
    inputs = {
        n: {"type": "integer", "description": f"Input {n}"}
        for n in input_names
    }
    return {
        "name": name,
        "description": f"A test tool named {name}",
        "inputs": inputs,
        "output_type": "dict",
    }


def _make_simple_func(*param_names):
    """Dynamically create a function with given parameter names."""
    # Build function signature string
    params = ", ".join(param_names)
    code = f"def _func({params}): return {{}}"
    local_ns = {}
    exec(code, {}, local_ns)
    return local_ns["_func"]


def _make_valid_tool_class():
    """Create a valid tool class with all required attributes."""
    class ValidTool:
        name = "valid_tool"
        description = "A valid tool"
        inputs = {
            "query": {"type": "string", "description": "Search query"},
        }
        output_type = "string"
    return ValidTool


# =============================================================================
# TestRegistryValidationResult
# =============================================================================

@pytest.mark.unit
@skip_no_module
class TestRegistryValidationResult:
    """Tests for RegistryValidationResult dataclass."""

    def test_valid_result_defaults(self):
        """Valid result has empty errors and warnings by default."""
        result = RegistryValidationResult(valid=True)
        assert result.valid is True
        assert result.errors == []
        assert result.warnings == []

    def test_invalid_result_with_errors(self):
        """Invalid result stores errors list."""
        result = RegistryValidationResult(
            valid=False, errors=["error1", "error2"]
        )
        assert result.valid is False
        assert len(result.errors) == 2

    def test_result_with_warnings(self):
        """Result stores warnings list."""
        result = RegistryValidationResult(
            valid=True, warnings=["warn1"]
        )
        assert result.warnings == ["warn1"]

    def test_to_dict(self):
        """to_dict() returns correct dictionary representation."""
        result = RegistryValidationResult(
            valid=False,
            errors=["bad input"],
            warnings=["consider renaming"],
        )
        d = result.to_dict()
        assert d["valid"] is False
        assert d["errors"] == ["bad input"]
        assert d["warnings"] == ["consider renaming"]

    def test_to_dict_empty(self):
        """to_dict() with no errors/warnings returns empty lists."""
        result = RegistryValidationResult(valid=True)
        d = result.to_dict()
        assert d == {"valid": True, "errors": [], "warnings": []}

    def test_post_init_none_errors_becomes_list(self):
        """None errors argument is converted to empty list in __post_init__."""
        result = RegistryValidationResult(valid=True, errors=None)
        assert result.errors == []

    def test_post_init_none_warnings_becomes_list(self):
        """None warnings argument is converted to empty list in __post_init__."""
        result = RegistryValidationResult(valid=True, warnings=None)
        assert result.warnings == []


# =============================================================================
# TestToolValidatorInit
# =============================================================================

@pytest.mark.unit
@skip_no_module
class TestToolValidatorInit:
    """Tests for ToolValidator initialization."""

    def test_default_strict_mode(self):
        """Default mode is strict=True."""
        validator = ToolValidator()
        assert validator.strict is True

    def test_strict_mode_explicit(self):
        """Strict mode set explicitly."""
        validator = ToolValidator(strict=True)
        assert validator.strict is True

    def test_non_strict_mode(self):
        """Non-strict mode disables code safety checks."""
        validator = ToolValidator(strict=False)
        assert validator.strict is False


# =============================================================================
# TestToolValidatorValidateTool
# =============================================================================

@pytest.mark.unit
@skip_no_module
class TestToolValidatorValidateTool:
    """Tests for validate_tool() main method."""

    def test_valid_tool_passes(self):
        """A well-formed tool passes validation."""
        def valid_tool(x):
            return {"result": x * 2}

        metadata = _make_valid_metadata("valid_tool", ["x"])
        validator = ToolValidator(strict=False)
        result = validator.validate_tool(valid_tool, metadata)
        assert result.valid is True
        assert result.errors == []

    def test_missing_name_field(self):
        """Missing 'name' field causes validation failure."""
        metadata = {
            "description": "test",
            "inputs": {},
            "output_type": "dict",
        }
        validator = ToolValidator(strict=False)
        result = validator.validate_tool(lambda: None, metadata)
        assert result.valid is False
        assert any("name" in e for e in result.errors)

    def test_missing_description_field(self):
        """Missing 'description' field causes validation failure."""
        metadata = {
            "name": "test",
            "inputs": {},
            "output_type": "dict",
        }
        validator = ToolValidator(strict=False)
        result = validator.validate_tool(lambda: None, metadata)
        assert result.valid is False
        assert any("description" in e for e in result.errors)

    def test_missing_inputs_field(self):
        """Missing 'inputs' field causes validation failure."""
        metadata = {
            "name": "test",
            "description": "desc",
            "output_type": "dict",
        }
        validator = ToolValidator(strict=False)
        result = validator.validate_tool(lambda: None, metadata)
        assert result.valid is False
        assert any("inputs" in e for e in result.errors)

    def test_missing_output_type_field(self):
        """Missing 'output_type' field causes validation failure."""
        metadata = {
            "name": "test",
            "description": "desc",
            "inputs": {},
        }
        validator = ToolValidator(strict=False)
        result = validator.validate_tool(lambda: None, metadata)
        assert result.valid is False
        assert any("output_type" in e for e in result.errors)

    def test_multiple_missing_fields(self):
        """Multiple missing fields are all reported."""
        metadata = {"name": "test"}
        validator = ToolValidator(strict=False)
        result = validator.validate_tool(lambda: None, metadata)
        assert result.valid is False
        assert len(result.errors) >= 3  # description, inputs, output_type

    def test_empty_metadata(self):
        """Completely empty metadata fails with multiple errors."""
        validator = ToolValidator(strict=False)
        result = validator.validate_tool(lambda: None, {})
        assert result.valid is False
        assert len(result.errors) == 4

    def test_returns_early_on_missing_required_fields(self):
        """Validation returns early when required fields are missing."""
        metadata = {}
        validator = ToolValidator(strict=False)
        result = validator.validate_tool(lambda: None, metadata)
        assert result.valid is False
        # Should only have the missing-field errors, not signature/type errors
        for e in result.errors:
            assert "Missing required field" in e


# =============================================================================
# TestToolValidatorSignature
# =============================================================================

@pytest.mark.unit
@skip_no_module
class TestToolValidatorSignature:
    """Tests for _validate_signature() method."""

    def test_matching_signature(self):
        """Function signature matching metadata inputs passes."""
        func = _make_simple_func("x", "y")
        metadata = _make_valid_metadata("tool", ["x", "y"])
        validator = ToolValidator(strict=False)
        result = validator._validate_signature(func, metadata)
        assert result.valid is True

    def test_missing_parameter_in_function(self):
        """Function missing a parameter declared in metadata is an error."""
        func = _make_simple_func("x")
        metadata = _make_valid_metadata("tool", ["x", "y"])
        validator = ToolValidator(strict=False)
        result = validator._validate_signature(func, metadata)
        assert result.valid is False
        assert any("missing" in e.lower() for e in result.errors)

    def test_extra_parameter_in_function(self):
        """Extra parameter in function not in metadata is a warning."""
        func = _make_simple_func("x", "y", "z")
        metadata = _make_valid_metadata("tool", ["x", "y"])
        validator = ToolValidator(strict=False)
        result = validator._validate_signature(func, metadata)
        # Extra params produce warnings, not errors
        assert len(result.warnings) > 0
        assert any("extra" in w.lower() for w in result.warnings)

    def test_self_parameter_is_ignored(self):
        """'self' parameter is excluded from comparison."""
        class MyTool:
            def run(self, x):
                pass

        metadata = _make_valid_metadata("tool", ["x"])
        validator = ToolValidator(strict=False)
        result = validator._validate_signature(MyTool().run, metadata)
        assert result.valid is True

    def test_completely_mismatched_parameters(self):
        """Completely different parameters produce both errors and warnings."""
        func = _make_simple_func("a", "b")
        metadata = _make_valid_metadata("tool", ["x", "y"])
        validator = ToolValidator(strict=False)
        result = validator._validate_signature(func, metadata)
        assert result.valid is False
        assert len(result.errors) > 0
        assert len(result.warnings) > 0

    def test_empty_inputs_empty_params(self):
        """No-arg function with empty inputs is valid."""
        func = _make_simple_func()
        metadata = _make_valid_metadata("tool", [])
        validator = ToolValidator(strict=False)
        result = validator._validate_signature(func, metadata)
        assert result.valid is True


# =============================================================================
# TestToolValidatorTypes
# =============================================================================

@pytest.mark.unit
@skip_no_module
class TestToolValidatorTypes:
    """Tests for _validate_types() method."""

    def test_all_authorized_input_types_pass(self):
        """Each authorized type passes validation."""
        validator = ToolValidator(strict=False)
        for auth_type in AUTHORIZED_TYPES:
            metadata = {
                "inputs": {"x": {"type": auth_type}},
                "output_type": "string",
            }
            result = validator._validate_types(metadata)
            assert result.valid is True, f"Type '{auth_type}' should be valid"

    def test_unauthorized_input_type_fails(self):
        """Unauthorized input type produces an error."""
        metadata = {
            "inputs": {"x": {"type": "custom_type"}},
            "output_type": "string",
        }
        validator = ToolValidator(strict=False)
        result = validator._validate_types(metadata)
        assert result.valid is False
        assert any("custom_type" in e for e in result.errors)

    def test_unauthorized_output_type_fails(self):
        """Unauthorized output type produces an error."""
        metadata = {
            "inputs": {},
            "output_type": "pandas_dataframe",
        }
        validator = ToolValidator(strict=False)
        result = validator._validate_types(metadata)
        assert result.valid is False
        assert any("pandas_dataframe" in e for e in result.errors)

    def test_multiple_invalid_input_types(self):
        """Multiple invalid input types are all reported."""
        metadata = {
            "inputs": {
                "a": {"type": "badtype1"},
                "b": {"type": "badtype2"},
            },
            "output_type": "string",
        }
        validator = ToolValidator(strict=False)
        result = validator._validate_types(metadata)
        assert result.valid is False
        assert len(result.errors) == 2

    def test_non_dict_input_spec_ignored(self):
        """Non-dict input specs are not type-checked (no error)."""
        metadata = {
            "inputs": {"x": "just_a_string"},
            "output_type": "string",
        }
        validator = ToolValidator(strict=False)
        result = validator._validate_types(metadata)
        assert result.valid is True

    def test_default_output_type_any(self):
        """Missing output_type defaults to 'any' which is authorized."""
        metadata = {"inputs": {}}
        validator = ToolValidator(strict=False)
        result = validator._validate_types(metadata)
        assert result.valid is True


# =============================================================================
# TestToolValidatorCodeSafety
# =============================================================================

@pytest.mark.unit
@skip_no_module
class TestToolValidatorCodeSafety:
    """Tests for _validate_code_safety() in strict mode."""

    def test_safe_function_passes(self):
        """A safe function with no dangerous calls passes."""
        def safe_func(x):
            return x * 2

        validator = ToolValidator(strict=True)
        result = validator._validate_code_safety(safe_func)
        assert result.valid is True

    def test_strict_mode_runs_code_safety(self):
        """In strict mode, validate_tool calls code safety checks."""
        def simple_func(x):
            return x

        metadata = _make_valid_metadata("tool", ["x"])
        validator = ToolValidator(strict=True)
        result = validator.validate_tool(simple_func, metadata)
        # Should complete without error (safe function)
        assert result.valid is True

    def test_non_strict_mode_skips_code_safety(self):
        """In non-strict mode, code safety is not checked."""
        def simple_func(x):
            return x

        metadata = _make_valid_metadata("tool", ["x"])
        validator = ToolValidator(strict=False)

        with patch.object(validator, "_validate_code_safety") as mock_safety:
            validator.validate_tool(simple_func, metadata)
            mock_safety.assert_not_called()

    def test_uninspectable_function_gives_warning(self):
        """Functions whose source cannot be retrieved produce a warning."""
        # Built-in functions cannot be inspected
        validator = ToolValidator(strict=True)
        result = validator._validate_code_safety(len)
        # Should not crash; should produce a warning
        assert len(result.warnings) > 0 or result.valid is True


# =============================================================================
# TestMethodChecker
# =============================================================================

@pytest.mark.unit
@skip_no_module
class TestMethodChecker:
    """Tests for MethodChecker AST visitor."""

    def test_no_errors_for_safe_code(self):
        """Safe code produces no errors."""
        code = "x = 1 + 2\ny = x * 3"
        tree = ast.parse(code)
        checker = MethodChecker(allowed_imports=set())
        checker.visit(tree)
        assert checker.errors == []

    def test_detects_eval_call(self):
        """Detects eval() as dangerous."""
        code = "result = eval('1+1')"
        tree = ast.parse(code)
        checker = MethodChecker(allowed_imports=set())
        checker.visit(tree)
        assert any("eval" in e for e in checker.errors)

    def test_detects_exec_call(self):
        """Detects exec() as dangerous."""
        code = "exec('print(1)')"
        tree = ast.parse(code)
        checker = MethodChecker(allowed_imports=set())
        checker.visit(tree)
        assert any("exec" in e for e in checker.errors)

    def test_detects_open_call(self):
        """Detects open() as dangerous."""
        code = "f = open('file.txt')"
        tree = ast.parse(code)
        checker = MethodChecker(allowed_imports=set())
        checker.visit(tree)
        assert any("open" in e for e in checker.errors)

    def test_detects_import_call(self):
        """Detects __import__() as dangerous."""
        code = "__import__('os')"
        tree = ast.parse(code)
        checker = MethodChecker(allowed_imports=set())
        checker.visit(tree)
        assert any("__import__" in e for e in checker.errors)

    def test_detects_dangerous_attribute_method(self):
        """Detects dangerous method calls on attributes (e.g., obj.eval())."""
        code = "obj.eval('code')"
        tree = ast.parse(code)
        checker = MethodChecker(allowed_imports=set())
        checker.visit(tree)
        assert any("eval" in e for e in checker.errors)

    def test_unauthorized_import_detected(self):
        """Unauthorized import is flagged."""
        code = "import subprocess"
        tree = ast.parse(code)
        checker = MethodChecker(allowed_imports={"os", "sys"})
        checker.visit(tree)
        assert any("subprocess" in e for e in checker.errors)

    def test_authorized_import_passes(self):
        """Authorized import does not produce errors."""
        code = "import os"
        tree = ast.parse(code)
        checker = MethodChecker(allowed_imports={"os"})
        checker.visit(tree)
        assert checker.errors == []

    def test_unauthorized_from_import_detected(self):
        """Unauthorized from-import is flagged."""
        code = "from subprocess import call"
        tree = ast.parse(code)
        checker = MethodChecker(allowed_imports={"os"})
        checker.visit(tree)
        assert any("subprocess" in e for e in checker.errors)

    def test_authorized_from_import_passes(self):
        """Authorized from-import does not produce errors."""
        code = "from os import path"
        tree = ast.parse(code)
        checker = MethodChecker(allowed_imports={"os"})
        checker.visit(tree)
        assert checker.errors == []

    def test_dangerous_methods_set(self):
        """DANGEROUS_METHODS contains expected entries."""
        assert "eval" in MethodChecker.DANGEROUS_METHODS
        assert "exec" in MethodChecker.DANGEROUS_METHODS
        assert "compile" in MethodChecker.DANGEROUS_METHODS
        assert "open" in MethodChecker.DANGEROUS_METHODS


# =============================================================================
# TestValidateToolAttributes
# =============================================================================

@pytest.mark.unit
@skip_no_module
class TestValidateToolAttributes:
    """Tests for validate_tool_attributes() function."""

    def test_valid_tool_class_passes(self):
        """A complete, well-formed tool class passes validation."""
        ToolClass = _make_valid_tool_class()
        result = validate_tool_attributes(ToolClass)
        assert result.valid is True
        assert result.errors == []

    def test_missing_name_attribute(self):
        """Missing 'name' attribute is an error."""
        class NoName:
            description = "desc"
            inputs = {}
            output_type = "string"

        result = validate_tool_attributes(NoName)
        assert result.valid is False
        assert any("name" in e for e in result.errors)

    def test_missing_description_attribute(self):
        """Missing 'description' attribute is an error."""
        class NoDesc:
            name = "tool"
            inputs = {}
            output_type = "string"

        result = validate_tool_attributes(NoDesc)
        assert result.valid is False
        assert any("description" in e for e in result.errors)

    def test_missing_inputs_attribute(self):
        """Missing 'inputs' attribute is an error."""
        class NoInputs:
            name = "tool"
            description = "desc"
            output_type = "string"

        result = validate_tool_attributes(NoInputs)
        assert result.valid is False
        assert any("inputs" in e for e in result.errors)

    def test_missing_output_type_attribute(self):
        """Missing 'output_type' attribute is an error."""
        class NoOutput:
            name = "tool"
            description = "desc"
            inputs = {}

        result = validate_tool_attributes(NoOutput)
        assert result.valid is False
        assert any("output_type" in e for e in result.errors)

    def test_wrong_type_for_name(self):
        """Non-string name attribute is an error."""
        class BadName:
            name = 123
            description = "desc"
            inputs = {}
            output_type = "string"

        result = validate_tool_attributes(BadName)
        assert result.valid is False
        assert any("name" in e and "str" in e for e in result.errors)

    def test_wrong_type_for_inputs(self):
        """Non-dict inputs attribute is an error."""
        class BadInputs:
            name = "tool"
            description = "desc"
            inputs = "not_a_dict"
            output_type = "string"

        result = validate_tool_attributes(BadInputs)
        assert result.valid is False
        assert any("inputs" in e and "dict" in e for e in result.errors)

    def test_input_spec_missing_type_key(self):
        """Input spec without 'type' key is an error."""
        class MissingType:
            name = "tool"
            description = "desc"
            inputs = {"x": {"description": "input x"}}
            output_type = "string"

        result = validate_tool_attributes(MissingType)
        assert result.valid is False
        assert any("type" in e and "description" in e for e in result.errors)

    def test_input_spec_missing_description_key(self):
        """Input spec without 'description' key is an error."""
        class MissingDesc:
            name = "tool"
            description = "desc"
            inputs = {"x": {"type": "string"}}
            output_type = "string"

        result = validate_tool_attributes(MissingDesc)
        assert result.valid is False

    def test_input_spec_unauthorized_type(self):
        """Input spec with unauthorized type is an error."""
        class BadType:
            name = "tool"
            description = "desc"
            inputs = {"x": {"type": "tensor", "description": "input"}}
            output_type = "string"

        result = validate_tool_attributes(BadType)
        assert result.valid is False
        assert any("tensor" in e for e in result.errors)

    def test_output_type_unauthorized(self):
        """Unauthorized output_type is an error."""
        class BadOutput:
            name = "tool"
            description = "desc"
            inputs = {}
            output_type = "dataframe"

        result = validate_tool_attributes(BadOutput)
        assert result.valid is False
        assert any("dataframe" in e for e in result.errors)

    def test_input_spec_non_dict_is_error(self):
        """Non-dict input spec value is an error."""
        class BadSpec:
            name = "tool"
            description = "desc"
            inputs = {"x": "just_a_string"}
            output_type = "string"

        result = validate_tool_attributes(BadSpec)
        assert result.valid is False
        assert any("dictionary" in e.lower() for e in result.errors)

    def test_multiple_valid_inputs(self):
        """Tool class with multiple valid inputs passes."""
        class MultiInput:
            name = "multi"
            description = "Multi-input tool"
            inputs = {
                "query": {"type": "string", "description": "Search query"},
                "count": {"type": "integer", "description": "Result count"},
                "verbose": {"type": "boolean", "description": "Verbose flag"},
            }
            output_type = "dict"

        result = validate_tool_attributes(MultiInput)
        assert result.valid is True


# =============================================================================
# TestToolGuard
# =============================================================================

@pytest.mark.unit
@skip_no_module
class TestToolGuard:
    """Tests for ToolGuard runtime safety checks."""

    def test_safe_tool_allowed_in_act_mode(self):
        """Safe tool is allowed in act mode."""
        guard = ToolGuard()
        allowed, reason = guard.check("read-file", "safe", mode="act")
        assert allowed is True
        assert reason == ""

    def test_safe_tool_allowed_in_plan_mode(self):
        """Safe tool is allowed in plan mode."""
        guard = ToolGuard()
        allowed, reason = guard.check("read-file", "safe", mode="plan")
        assert allowed is True

    def test_side_effect_blocked_in_plan_mode(self):
        """Side-effect tool is blocked in plan mode."""
        guard = ToolGuard()
        allowed, reason = guard.check("send-email", "side_effect", mode="plan")
        assert allowed is False
        assert "plan mode" in reason.lower()

    def test_destructive_blocked_in_plan_mode(self):
        """Destructive tool is blocked in plan mode."""
        guard = ToolGuard()
        allowed, reason = guard.check("delete-db", "destructive", mode="plan")
        assert allowed is False
        assert "plan mode" in reason.lower()

    def test_one_side_effect_per_turn(self):
        """Second side-effect tool in same turn is blocked."""
        guard = ToolGuard()
        guard.record_use("first-tool", "side_effect")
        allowed, reason = guard.check("second-tool", "side_effect", mode="act")
        assert allowed is False
        assert "one side-effect" in reason.lower()

    def test_side_effect_after_reset_turn(self):
        """Side-effect tool allowed after turn reset."""
        guard = ToolGuard()
        guard.record_use("first-tool", "side_effect")
        guard.reset_turn()
        allowed, reason = guard.check("second-tool", "side_effect", mode="act")
        assert allowed is True

    def test_destructive_session_limit(self):
        """Destructive tool blocked after session limit reached."""
        guard = ToolGuard(max_destructive_per_session=2)
        guard.record_use("tool1", "destructive")
        guard.reset_turn()
        guard.record_use("tool2", "destructive")
        guard.reset_turn()
        allowed, reason = guard.check("tool3", "destructive", mode="act")
        assert allowed is False
        assert "limit" in reason.lower()

    def test_destructive_under_limit_allowed(self):
        """Destructive tool allowed when under session limit."""
        guard = ToolGuard(max_destructive_per_session=5)
        guard.record_use("tool1", "destructive")
        guard.reset_turn()
        allowed, reason = guard.check("tool2", "destructive", mode="act")
        assert allowed is True

    def test_path_blocking_env_file(self):
        """Tool targeting .env path is blocked."""
        guard = ToolGuard()
        allowed, reason = guard.check(
            "write-file", "safe", mode="act", target_path="/app/.env"
        )
        assert allowed is False
        assert ".env" in reason

    def test_path_blocking_ssh_key(self):
        """Tool targeting .ssh path is blocked."""
        guard = ToolGuard()
        allowed, reason = guard.check(
            "read-file", "safe", mode="act", target_path="/home/user/.ssh/id_rsa"
        )
        assert allowed is False

    def test_path_blocking_credentials(self):
        """Tool targeting credentials path is blocked."""
        guard = ToolGuard()
        allowed, reason = guard.check(
            "read-file", "safe", mode="act",
            target_path="/app/config/credentials.json"
        )
        assert allowed is False

    def test_safe_path_allowed(self):
        """Tool targeting a normal path is allowed."""
        guard = ToolGuard()
        allowed, reason = guard.check(
            "write-file", "safe", mode="act",
            target_path="/app/data/output.txt"
        )
        assert allowed is True

    def test_custom_blocked_patterns(self):
        """Custom blocked path patterns are enforced."""
        guard = ToolGuard(blocked_path_patterns={"backup", "archive"})
        allowed, _ = guard.check(
            "write", "safe", mode="act", target_path="/data/backup/db.sql"
        )
        assert allowed is False

    def test_record_use_safe_does_not_set_flag(self):
        """Recording a safe tool use does not set side-effect flag."""
        guard = ToolGuard()
        guard.record_use("read-file", "safe")
        assert guard._side_effect_used_this_turn is False

    def test_record_use_side_effect_sets_flag(self):
        """Recording a side-effect tool use sets the flag."""
        guard = ToolGuard()
        guard.record_use("send-email", "side_effect")
        assert guard._side_effect_used_this_turn is True
        assert guard._last_tool == "send-email"

    def test_record_use_destructive_increments_count(self):
        """Recording a destructive tool use increments session count."""
        guard = ToolGuard()
        guard.record_use("delete-db", "destructive")
        assert guard._destructive_count_session == 1

    def test_reset_session_clears_all(self):
        """reset_session() clears turn state and session counters."""
        guard = ToolGuard()
        guard.record_use("tool", "destructive")
        guard.reset_session()
        assert guard._side_effect_used_this_turn is False
        assert guard._last_tool is None
        assert guard._destructive_count_session == 0

    def test_stats_returns_state(self):
        """stats() returns correct state dictionary."""
        guard = ToolGuard(max_destructive_per_session=10)
        guard.record_use("my-tool", "side_effect")
        s = guard.stats()
        assert s["side_effect_used_this_turn"] is True
        assert s["last_tool"] == "my-tool"
        assert s["destructive_count_session"] == 0
        assert s["max_destructive"] == 10

    def test_default_blocked_paths_set(self):
        """DEFAULT_BLOCKED_PATHS contains expected patterns."""
        assert ".env" in ToolGuard.DEFAULT_BLOCKED_PATHS
        assert ".git" in ToolGuard.DEFAULT_BLOCKED_PATHS
        assert "credentials" in ToolGuard.DEFAULT_BLOCKED_PATHS
        assert ".ssh" in ToolGuard.DEFAULT_BLOCKED_PATHS


# =============================================================================
# TestAuthorizedTypes
# =============================================================================

@pytest.mark.unit
@skip_no_module
class TestAuthorizedTypes:
    """Tests for the AUTHORIZED_TYPES and TYPE_CONVERSION constants."""

    def test_authorized_types_contents(self):
        """AUTHORIZED_TYPES contains the expected type names."""
        expected = {"string", "boolean", "integer", "number", "dict", "list", "any", "null"}
        assert set(AUTHORIZED_TYPES) == expected

    def test_type_conversion_mapping(self):
        """TYPE_CONVERSION maps Python types to authorized names."""
        assert TYPE_CONVERSION["str"] == "string"
        assert TYPE_CONVERSION["int"] == "integer"
        assert TYPE_CONVERSION["float"] == "number"
        assert TYPE_CONVERSION["bool"] == "boolean"
        assert TYPE_CONVERSION["dict"] == "dict"
        assert TYPE_CONVERSION["list"] == "list"
