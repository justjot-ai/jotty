"""
Tests for core/validation.py
==============================
Covers: ParamValidator, validate_params, ValidationError re-export.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from Jotty.core.validation import ParamValidator, ValidationError, validate_params

# ===========================================================================
# ParamValidator — Initialization
# ===========================================================================


@pytest.mark.unit
class TestParamValidatorInit:
    """ParamValidator initialization and basic state."""

    def test_empty_init(self):
        """ParamValidator with no params defaults to empty dict."""
        v = ParamValidator()
        assert v.params == {}
        assert v.validated == {}
        assert v.errors == []

    def test_init_with_params(self):
        """ParamValidator stores provided params."""
        v = ParamValidator({"key": "value", "num": 42})
        assert v.params == {"key": "value", "num": 42}

    def test_none_params_defaults_to_empty(self):
        """ParamValidator(None) uses empty dict."""
        v = ParamValidator(None)
        assert v.params == {}

    def test_context_manager_returns_self(self):
        """Context manager __enter__ returns the validator."""
        v = ParamValidator({"a": 1})
        with v as ctx:
            assert ctx is v

    def test_context_manager_does_not_suppress(self):
        """Context manager __exit__ does not suppress exceptions."""
        with pytest.raises(ValidationError):
            with ParamValidator({}) as v:
                v.require("missing")


# ===========================================================================
# ParamValidator — require()
# ===========================================================================


@pytest.mark.unit
class TestParamValidatorRequire:
    """Tests for ParamValidator.require()."""

    def test_require_present_value(self):
        """require() returns value when present."""
        v = ParamValidator({"name": "Alice"})
        result = v.require("name")
        assert result == "Alice"
        assert v.validated["name"] == "Alice"

    def test_require_missing_raises(self):
        """require() raises ValidationError for missing param."""
        v = ParamValidator({})
        with pytest.raises(ValidationError, match="'name' is required"):
            v.require("name")
        assert len(v.errors) == 1

    def test_require_missing_custom_message(self):
        """require() uses custom message when provided."""
        v = ParamValidator({})
        with pytest.raises(ValidationError, match="Please provide name"):
            v.require("name", message="Please provide name")

    def test_require_correct_type(self):
        """require() passes when type matches."""
        v = ParamValidator({"count": 5})
        result = v.require("count", int)
        assert result == 5

    def test_require_wrong_type_raises(self):
        """require() raises when type doesn't match."""
        v = ParamValidator({"count": "five"})
        with pytest.raises(ValidationError, match="must be int"):
            v.require("count", int)

    def test_require_none_value_raises(self):
        """require() raises for None value."""
        v = ParamValidator({"key": None})
        with pytest.raises(ValidationError):
            v.require("key")


# ===========================================================================
# ParamValidator — optional()
# ===========================================================================


@pytest.mark.unit
class TestParamValidatorOptional:
    """Tests for ParamValidator.optional()."""

    def test_optional_present_value(self):
        """optional() returns value when present."""
        v = ParamValidator({"format": "pdf"})
        result = v.optional("format", str, default="html")
        assert result == "pdf"
        assert v.validated["format"] == "pdf"

    def test_optional_missing_returns_default(self):
        """optional() returns default for missing param."""
        v = ParamValidator({})
        result = v.optional("format", str, default="html")
        assert result == "html"
        assert v.validated["format"] == "html"

    def test_optional_none_value_returns_default(self):
        """optional() returns default when value is None."""
        v = ParamValidator({"fmt": None})
        result = v.optional("fmt", str, default="json")
        assert result == "json"

    def test_optional_type_coercion_bool(self):
        """optional() coerces string to bool."""
        v = ParamValidator({"flag": "true"})
        result = v.optional("flag", bool)
        assert result is True

    def test_optional_type_coercion_int(self):
        """optional() coerces string to int."""
        v = ParamValidator({"count": "42"})
        result = v.optional("count", int)
        assert result == 42

    def test_optional_type_coercion_float(self):
        """optional() coerces string to float."""
        v = ParamValidator({"rate": "3.14"})
        result = v.optional("rate", float)
        assert result == 3.14

    def test_optional_type_coercion_str(self):
        """optional() coerces int to str."""
        v = ParamValidator({"id": 123})
        result = v.optional("id", str)
        assert result == "123"

    def test_optional_invalid_coercion_raises(self):
        """optional() raises when coercion to unsupported type fails."""
        v = ParamValidator({"data": "abc"})
        with pytest.raises(ValidationError, match="must be list"):
            v.optional("data", list)

    def test_optional_choices_valid(self):
        """optional() accepts value in choices."""
        v = ParamValidator({"format": "pdf"})
        result = v.optional("format", str, choices=["pdf", "html", "markdown"])
        assert result == "pdf"

    def test_optional_choices_invalid_raises(self):
        """optional() raises for value not in choices."""
        v = ParamValidator({"format": "docx"})
        with pytest.raises(ValidationError, match="must be one of"):
            v.optional("format", str, choices=["pdf", "html"])


# ===========================================================================
# ParamValidator — require_one_of()
# ===========================================================================


@pytest.mark.unit
class TestParamValidatorRequireOneOf:
    """Tests for ParamValidator.require_one_of()."""

    def test_first_present(self):
        """require_one_of() returns first present param name."""
        v = ParamValidator({"url": "http://x.com", "file": "/tmp/f"})
        result = v.require_one_of("url", "file")
        assert result == "url"
        assert v.validated["url"] == "http://x.com"

    def test_second_present(self):
        """require_one_of() returns second if first missing."""
        v = ParamValidator({"file": "/tmp/f"})
        result = v.require_one_of("url", "file")
        assert result == "file"

    def test_none_present_raises(self):
        """require_one_of() raises when none present."""
        v = ParamValidator({})
        with pytest.raises(ValidationError, match="At least one of"):
            v.require_one_of("url", "file", "path")

    def test_none_present_custom_message(self):
        """require_one_of() uses custom message."""
        v = ParamValidator({})
        with pytest.raises(ValidationError, match="Need input"):
            v.require_one_of("url", "file", message="Need input")


# ===========================================================================
# ParamValidator — validate_pattern()
# ===========================================================================


@pytest.mark.unit
class TestParamValidatorPattern:
    """Tests for ParamValidator.validate_pattern()."""

    def test_builtin_pattern_url(self):
        """validate_pattern with 'url' key matches URLs."""
        v = ParamValidator({"link": "https://example.com/path"})
        result = v.validate_pattern("link", "url")
        assert result == "https://example.com/path"

    def test_builtin_pattern_email(self):
        """validate_pattern with 'email' key matches emails."""
        v = ParamValidator({"email": "user@example.com"})
        result = v.validate_pattern("email", "email")
        assert result == "user@example.com"

    def test_builtin_pattern_arxiv_id(self):
        """validate_pattern with 'arxiv_id' key matches arxiv IDs."""
        v = ParamValidator({"id": "2301.12345"})
        result = v.validate_pattern("id", "arxiv_id")
        assert result == "2301.12345"

    def test_custom_regex_pattern(self):
        """validate_pattern with custom regex works."""
        v = ParamValidator({"code": "ABC-123"})
        result = v.validate_pattern("code", r"^[A-Z]+-\d+$")
        assert result == "ABC-123"

    def test_invalid_pattern_raises(self):
        """validate_pattern raises on mismatch."""
        v = ParamValidator({"email": "not-an-email"})
        with pytest.raises(ValidationError, match="invalid format"):
            v.validate_pattern("email", "email")

    def test_missing_required_raises(self):
        """validate_pattern raises for missing required param."""
        v = ParamValidator({})
        with pytest.raises(ValidationError, match="is required"):
            v.validate_pattern("url", "url", required=True)

    def test_missing_optional_returns_none(self):
        """validate_pattern returns None for missing optional param."""
        v = ParamValidator({})
        result = v.validate_pattern("url", "url", required=False)
        assert result is None

    def test_custom_error_message(self):
        """validate_pattern uses custom message."""
        v = ParamValidator({"link": "bad"})
        with pytest.raises(ValidationError, match="Bad link"):
            v.validate_pattern("link", "url", message="Bad link")


# ===========================================================================
# ParamValidator — validate_url() / validate_email()
# ===========================================================================


@pytest.mark.unit
class TestParamValidatorUrlEmail:
    """Tests for validate_url and validate_email convenience methods."""

    def test_validate_url_valid(self):
        """validate_url accepts valid URL."""
        v = ParamValidator({"site": "https://example.com"})
        assert v.validate_url("site") == "https://example.com"

    def test_validate_url_invalid(self):
        """validate_url raises on invalid URL."""
        v = ParamValidator({"site": "not-a-url"})
        with pytest.raises(ValidationError, match="valid URL"):
            v.validate_url("site")

    def test_validate_email_valid(self):
        """validate_email accepts valid email."""
        v = ParamValidator({"mail": "user@test.com"})
        assert v.validate_email("mail") == "user@test.com"

    def test_validate_email_invalid(self):
        """validate_email raises on invalid email."""
        v = ParamValidator({"mail": "bad"})
        with pytest.raises(ValidationError, match="valid email"):
            v.validate_email("mail")


# ===========================================================================
# ParamValidator — validate_file_exists()
# ===========================================================================


@pytest.mark.unit
class TestParamValidatorFileExists:
    """Tests for ParamValidator.validate_file_exists()."""

    def test_existing_file(self, tmp_path):
        """validate_file_exists returns Path for existing file."""
        f = tmp_path / "test.txt"
        f.write_text("hello")
        v = ParamValidator({"path": str(f)})
        result = v.validate_file_exists("path")
        assert result == f
        assert v.validated["path"] == f

    def test_nonexistent_file_raises(self):
        """validate_file_exists raises for missing file."""
        v = ParamValidator({"path": "/nonexistent/file.txt"})
        with pytest.raises(ValidationError, match="File not found"):
            v.validate_file_exists("path")

    def test_missing_required_raises(self):
        """validate_file_exists raises for missing required param."""
        v = ParamValidator({})
        with pytest.raises(ValidationError, match="is required"):
            v.validate_file_exists("path", required=True)

    def test_missing_optional_returns_none(self):
        """validate_file_exists returns None for missing optional param."""
        v = ParamValidator({})
        result = v.validate_file_exists("path", required=False)
        assert result is None


# ===========================================================================
# ParamValidator — validate_dir_exists()
# ===========================================================================


@pytest.mark.unit
class TestParamValidatorDirExists:
    """Tests for ParamValidator.validate_dir_exists()."""

    def test_existing_dir(self, tmp_path):
        """validate_dir_exists returns Path for existing directory."""
        v = ParamValidator({"dir": str(tmp_path)})
        result = v.validate_dir_exists("dir")
        assert result == tmp_path

    def test_nonexistent_dir_raises(self):
        """validate_dir_exists raises for missing directory."""
        v = ParamValidator({"dir": "/nonexistent/dir"})
        with pytest.raises(ValidationError, match="Directory not found"):
            v.validate_dir_exists("dir")

    def test_create_dir(self, tmp_path):
        """validate_dir_exists creates directory when create=True."""
        new_dir = tmp_path / "new_sub"
        v = ParamValidator({"dir": str(new_dir)})
        result = v.validate_dir_exists("dir", create=True)
        assert result == new_dir
        assert new_dir.exists()

    def test_file_not_dir_raises(self, tmp_path):
        """validate_dir_exists raises when path is a file, not dir."""
        f = tmp_path / "file.txt"
        f.write_text("data")
        v = ParamValidator({"dir": str(f)})
        with pytest.raises(ValidationError, match="not a directory"):
            v.validate_dir_exists("dir")

    def test_missing_required_raises(self):
        """validate_dir_exists raises for missing required param."""
        v = ParamValidator({})
        with pytest.raises(ValidationError, match="is required"):
            v.validate_dir_exists("dir", required=True)

    def test_missing_optional_returns_none(self):
        """validate_dir_exists returns None for missing optional."""
        v = ParamValidator({})
        result = v.validate_dir_exists("dir", required=False)
        assert result is None


# ===========================================================================
# ParamValidator — validate_range()
# ===========================================================================


@pytest.mark.unit
class TestParamValidatorRange:
    """Tests for ParamValidator.validate_range()."""

    def test_in_range(self):
        """validate_range passes for value within range."""
        v = ParamValidator({"temp": 0.7})
        result = v.validate_range("temp", min_val=0.0, max_val=1.0)
        assert result == 0.7

    def test_below_min_raises(self):
        """validate_range raises for value below min."""
        v = ParamValidator({"temp": -1})
        with pytest.raises(ValidationError, match="at least"):
            v.validate_range("temp", min_val=0)

    def test_above_max_raises(self):
        """validate_range raises for value above max."""
        v = ParamValidator({"temp": 2.0})
        with pytest.raises(ValidationError, match="at most"):
            v.validate_range("temp", max_val=1.0)

    def test_non_numeric_raises(self):
        """validate_range raises for non-numeric value."""
        v = ParamValidator({"temp": "abc"})
        with pytest.raises(ValidationError, match="must be a number"):
            v.validate_range("temp")

    def test_missing_with_default(self):
        """validate_range returns default for missing param."""
        v = ParamValidator({})
        result = v.validate_range("temp", default=0.5)
        assert result == 0.5

    def test_missing_required_raises(self):
        """validate_range raises for missing required param."""
        v = ParamValidator({})
        with pytest.raises(ValidationError, match="is required"):
            v.validate_range("temp", required=True)

    def test_string_int_coercion(self):
        """validate_range coerces string integers."""
        v = ParamValidator({"count": "42"})
        result = v.validate_range("count", min_val=0, max_val=100)
        assert result == 42

    def test_string_float_coercion(self):
        """validate_range coerces string floats."""
        v = ParamValidator({"rate": "3.14"})
        result = v.validate_range("rate", min_val=0.0)
        assert result == 3.14

    def test_boundary_values(self):
        """validate_range accepts boundary values."""
        v = ParamValidator({"val": 0})
        result = v.validate_range("val", min_val=0, max_val=10)
        assert result == 0

        v2 = ParamValidator({"val": 10})
        result2 = v2.validate_range("val", min_val=0, max_val=10)
        assert result2 == 10


# ===========================================================================
# ParamValidator — is_valid() / get_errors()
# ===========================================================================


@pytest.mark.unit
class TestParamValidatorState:
    """Tests for is_valid and get_errors."""

    def test_is_valid_no_errors(self):
        """is_valid returns True when no errors."""
        v = ParamValidator({"name": "test"})
        v.require("name")
        assert v.is_valid() is True

    def test_is_valid_with_errors(self):
        """is_valid returns False after validation error."""
        v = ParamValidator({})
        try:
            v.require("name")
        except ValidationError:
            pass
        assert v.is_valid() is False

    def test_get_errors_empty(self):
        """get_errors returns empty list when no errors."""
        v = ParamValidator({"name": "test"})
        v.require("name")
        assert v.get_errors() == []

    def test_get_errors_multiple(self):
        """get_errors accumulates multiple errors."""
        v = ParamValidator({})
        for name in ["a", "b", "c"]:
            try:
                v.require(name)
            except ValidationError:
                pass
        assert len(v.get_errors()) == 3


# ===========================================================================
# validate_params() — Schema-based validation
# ===========================================================================


@pytest.mark.unit
class TestValidateParams:
    """Tests for the validate_params() function."""

    def test_required_field(self):
        """validate_params with required field."""
        result = validate_params({"name": "test"}, {"name": {"required": True, "type": str}})
        assert result["name"] == "test"

    def test_missing_required_raises(self):
        """validate_params raises for missing required."""
        with pytest.raises(ValidationError):
            validate_params({}, {"name": {"required": True, "type": str}})

    def test_optional_with_default(self):
        """validate_params uses default for missing optional."""
        result = validate_params({}, {"format": {"type": str, "default": "pdf"}})
        assert result["format"] == "pdf"

    def test_choices_validation(self):
        """validate_params validates choices."""
        result = validate_params(
            {"format": "pdf"}, {"format": {"type": str, "choices": ["pdf", "html"]}}
        )
        assert result["format"] == "pdf"

    def test_choices_invalid_raises(self):
        """validate_params raises for invalid choice."""
        with pytest.raises(ValidationError):
            validate_params(
                {"format": "docx"}, {"format": {"type": str, "choices": ["pdf", "html"]}}
            )

    def test_range_validation(self):
        """validate_params validates numeric ranges."""
        result = validate_params({"count": 5}, {"count": {"min": 1, "max": 10}})
        assert result["count"] == 5

    def test_url_validation(self):
        """validate_params validates URLs."""
        result = validate_params(
            {"link": "https://example.com"}, {"link": {"url": True, "required": True}}
        )
        assert result["link"] == "https://example.com"

    def test_email_validation(self):
        """validate_params validates emails."""
        result = validate_params(
            {"mail": "u@test.com"}, {"mail": {"email": True, "required": True}}
        )
        assert result["mail"] == "u@test.com"

    def test_pattern_validation(self):
        """validate_params validates patterns."""
        result = validate_params(
            {"code": "ABC-123"}, {"code": {"pattern": r"^[A-Z]+-\d+$", "required": True}}
        )
        assert result["code"] == "ABC-123"

    def test_file_exists_validation(self, tmp_path):
        """validate_params validates file existence."""
        f = tmp_path / "test.txt"
        f.write_text("data")
        result = validate_params(
            {"path": str(f)}, {"path": {"file_exists": True, "required": True}}
        )
        assert result["path"] == f

    def test_dir_exists_with_create(self, tmp_path):
        """validate_params creates directory when specified."""
        new_dir = tmp_path / "output"
        result = validate_params(
            {"dir": str(new_dir)}, {"dir": {"dir_exists": True, "create": True}}
        )
        assert result["dir"] == new_dir
        assert new_dir.exists()

    def test_multiple_fields(self):
        """validate_params handles multiple fields."""
        result = validate_params(
            {"name": "test", "count": 5, "format": "pdf"},
            {
                "name": {"required": True, "type": str},
                "count": {"min": 1, "max": 100},
                "format": {"type": str, "choices": ["pdf", "html"], "default": "html"},
            },
        )
        assert result["name"] == "test"
        assert result["count"] == 5
        assert result["format"] == "pdf"


# ===========================================================================
# PATTERNS constant
# ===========================================================================


@pytest.mark.unit
class TestParamValidatorPatterns:
    """Tests for the PATTERNS class constant."""

    def test_all_patterns_exist(self):
        """All expected pattern keys are present."""
        expected = {"email", "url", "arxiv_id", "youtube_url", "kindle_email"}
        assert expected.issubset(set(ParamValidator.PATTERNS.keys()))

    def test_youtube_url_pattern(self):
        """YouTube URL pattern matches valid URLs."""
        v = ParamValidator({"url": "https://www.youtube.com/watch?v=abc123"})
        result = v.validate_pattern("url", "youtube_url")
        assert result is not None

    def test_kindle_email_pattern(self):
        """Kindle email pattern matches @kindle.com."""
        v = ParamValidator({"email": "user@kindle.com"})
        result = v.validate_pattern("email", "kindle_email")
        assert result is not None

    def test_arxiv_legacy_format(self):
        """ArXiv pattern matches legacy format."""
        v = ParamValidator({"id": "hep-th/0601001"})
        result = v.validate_pattern("id", "arxiv_id")
        assert result is not None
