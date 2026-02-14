"""
Tests for OpenAPI Specification Generator and SwarmCodeGenerator.

Covers:
- _python_type_to_openapi: Python type → OpenAPI schema conversion
- _dataclass_to_schema: Dataclass introspection → OpenAPI schema
- generate_openapi_spec: Full OpenAPI 3.0 spec generation
- save_openapi_spec: File persistence
- SwarmCodeGenerator: Code generation utilities
"""

import json
import pytest
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from unittest.mock import MagicMock, patch

# ── Import openapi module with fallback ──────────────────────────────────────

try:
    from Jotty.core.api.openapi import (
        _python_type_to_openapi,
        _dataclass_to_schema,
        generate_openapi_spec,
        save_openapi_spec,
    )
    OPENAPI_AVAILABLE = True
except ImportError:
    OPENAPI_AVAILABLE = False

# ── Import SwarmCodeGenerator with fallback ──────────────────────────────────

try:
    from Jotty.core.orchestration.swarm_code_generator import (
        SwarmCodeGenerator,
        GeneratedCode,
    )
    CODEGEN_AVAILABLE = True
except ImportError:
    CODEGEN_AVAILABLE = False


# ── Test helpers: Enum and dataclass definitions ─────────────────────────────

class Color(Enum):
    """Colour choices."""
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


class Priority(Enum):
    LOW = "low"
    HIGH = "high"


@dataclass
class SimpleRecord:
    """A simple record for testing."""
    name: str
    age: int


@dataclass
class RecordWithDefaults:
    """Record with default values."""
    title: str
    count: int = 0
    active: bool = True
    label: str = "default"


@dataclass
class RecordWithCallback:
    """Record that has callback and private fields."""
    name: str
    on_callback: Any = None
    _private: str = "hidden"
    event_callback: Any = None


@dataclass
class RecordWithEnum:
    """Record with an enum default."""
    priority: Priority = Priority.LOW


@dataclass
class RecordWithComplexTypes:
    """Complex type record."""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    color: Optional[Color] = None
    timestamp: Optional[datetime] = None


# =============================================================================
# Section 1: _python_type_to_openapi
# =============================================================================

@pytest.mark.unit
class TestPythonTypeToOpenapi:
    """Tests for _python_type_to_openapi conversion."""

    pytestmark = pytest.mark.skipif(
        not OPENAPI_AVAILABLE, reason="Jotty openapi module not importable"
    )

    # ── Primitive types ──────────────────────────────────────────────────

    def test_str_maps_to_string(self):
        assert _python_type_to_openapi(str) == {"type": "string"}

    def test_int_maps_to_integer(self):
        assert _python_type_to_openapi(int) == {"type": "integer"}

    def test_float_maps_to_number_float(self):
        assert _python_type_to_openapi(float) == {"type": "number", "format": "float"}

    def test_bool_maps_to_boolean(self):
        assert _python_type_to_openapi(bool) == {"type": "boolean"}

    def test_datetime_maps_to_string_datetime(self):
        assert _python_type_to_openapi(datetime) == {
            "type": "string",
            "format": "date-time",
        }

    def test_any_maps_to_empty_schema(self):
        assert _python_type_to_openapi(Any) == {}

    # ── NoneType ─────────────────────────────────────────────────────────

    def test_nonetype_maps_to_nullable(self):
        assert _python_type_to_openapi(type(None)) == {"nullable": True}

    # ── Optional[X] ─────────────────────────────────────────────────────

    def test_optional_str_is_nullable_string(self):
        result = _python_type_to_openapi(Optional[str])
        assert result == {"type": "string", "nullable": True}

    def test_optional_int_is_nullable_integer(self):
        result = _python_type_to_openapi(Optional[int])
        assert result == {"type": "integer", "nullable": True}

    # ── List[X] ──────────────────────────────────────────────────────────

    def test_list_str_is_array_of_strings(self):
        result = _python_type_to_openapi(List[str])
        assert result == {"type": "array", "items": {"type": "string"}}

    def test_list_int_is_array_of_integers(self):
        result = _python_type_to_openapi(List[int])
        assert result == {"type": "array", "items": {"type": "integer"}}

    # ── Dict[K, V] ──────────────────────────────────────────────────────

    def test_dict_str_any_is_object(self):
        result = _python_type_to_openapi(Dict[str, Any])
        assert result == {"type": "object", "additionalProperties": True}

    # ── Enum ─────────────────────────────────────────────────────────────

    def test_enum_maps_to_string_enum(self):
        result = _python_type_to_openapi(Color)
        assert result == {"type": "string", "enum": ["red", "green", "blue"]}

    def test_enum_populates_enums_collected(self):
        collected = {}
        _python_type_to_openapi(Color, enums_collected=collected)
        assert "Color" in collected
        assert collected["Color"] == ["red", "green", "blue"]

    def test_enum_without_enums_collected_does_not_error(self):
        result = _python_type_to_openapi(Priority)
        assert result["enum"] == ["low", "high"]

    # ── Union types ──────────────────────────────────────────────────────

    def test_union_str_int_maps_to_oneof(self):
        result = _python_type_to_openapi(Union[str, int])
        assert "oneOf" in result
        assert {"type": "string"} in result["oneOf"]
        assert {"type": "integer"} in result["oneOf"]

    def test_union_with_none_treated_as_optional(self):
        """Union[str, None] should behave like Optional[str]."""
        result = _python_type_to_openapi(Union[str, None])
        assert result == {"type": "string", "nullable": True}


# =============================================================================
# Section 2: _dataclass_to_schema
# =============================================================================

@pytest.mark.unit
class TestDataclassToSchema:
    """Tests for _dataclass_to_schema conversion."""

    pytestmark = pytest.mark.skipif(
        not OPENAPI_AVAILABLE, reason="Jotty openapi module not importable"
    )

    def test_simple_dataclass_properties(self):
        schema = _dataclass_to_schema(SimpleRecord)
        props = schema["properties"]
        assert props["name"] == {"type": "string"}
        assert props["age"] == {"type": "integer"}

    def test_required_fields_no_default(self):
        schema = _dataclass_to_schema(SimpleRecord)
        assert "required" in schema
        assert "name" in schema["required"]
        assert "age" in schema["required"]

    def test_fields_with_defaults_have_default_key(self):
        schema = _dataclass_to_schema(RecordWithDefaults)
        props = schema["properties"]
        assert props["count"]["default"] == 0
        assert props["active"]["default"] is True
        assert props["label"]["default"] == "default"

    def test_required_excludes_fields_with_defaults(self):
        schema = _dataclass_to_schema(RecordWithDefaults)
        required = schema.get("required", [])
        assert "title" in required
        assert "count" not in required
        assert "active" not in required

    def test_callback_fields_skipped(self):
        schema = _dataclass_to_schema(RecordWithCallback)
        props = schema["properties"]
        assert "on_callback" not in props
        assert "event_callback" not in props

    def test_private_fields_skipped(self):
        schema = _dataclass_to_schema(RecordWithCallback)
        props = schema["properties"]
        assert "_private" not in props

    def test_docstring_becomes_description(self):
        schema = _dataclass_to_schema(SimpleRecord)
        assert "description" in schema
        assert schema["description"] == "A simple record for testing."

    def test_enum_default_serialized_to_value(self):
        schema = _dataclass_to_schema(RecordWithEnum)
        props = schema["properties"]
        assert props["priority"]["default"] == "low"

    def test_type_is_object(self):
        schema = _dataclass_to_schema(SimpleRecord)
        assert schema["type"] == "object"

    def test_complex_types_in_schema(self):
        schema = _dataclass_to_schema(RecordWithComplexTypes)
        props = schema["properties"]
        assert props["tags"]["type"] == "array"
        assert props["metadata"]["type"] == "object"


# =============================================================================
# Section 3: generate_openapi_spec
# =============================================================================

@pytest.mark.unit
class TestGenerateOpenapiSpec:
    """Tests for generate_openapi_spec function."""

    pytestmark = pytest.mark.skipif(
        not OPENAPI_AVAILABLE, reason="Jotty openapi module not importable"
    )

    def test_spec_has_top_level_keys(self):
        spec = generate_openapi_spec(base_url="http://localhost:9999")
        assert "openapi" in spec
        assert "info" in spec
        assert "servers" in spec
        assert "paths" in spec
        assert "components" in spec

    def test_openapi_version_is_3(self):
        spec = generate_openapi_spec(base_url="http://localhost:9999")
        assert spec["openapi"].startswith("3.")

    def test_info_contains_title_version_description(self):
        spec = generate_openapi_spec(base_url="http://localhost:9999")
        info = spec["info"]
        assert "title" in info
        assert "version" in info
        assert "description" in info

    def test_custom_title_version_description(self):
        spec = generate_openapi_spec(
            title="My API",
            version="5.0.0",
            description="Custom description",
            base_url="http://localhost:9999",
        )
        assert spec["info"]["title"] == "My API"
        assert spec["info"]["version"] == "5.0.0"
        assert spec["info"]["description"] == "Custom description"

    def test_paths_has_health(self):
        spec = generate_openapi_spec(base_url="http://localhost:9999")
        assert "/health" in spec["paths"]

    def test_paths_has_api_chat(self):
        spec = generate_openapi_spec(base_url="http://localhost:9999")
        assert "/api/chat" in spec["paths"]

    def test_paths_has_api_workflow(self):
        spec = generate_openapi_spec(base_url="http://localhost:9999")
        assert "/api/workflow" in spec["paths"]

    def test_paths_has_api_skills(self):
        spec = generate_openapi_spec(base_url="http://localhost:9999")
        assert "/api/skills" in spec["paths"]

    def test_components_schemas_exist(self):
        spec = generate_openapi_spec(base_url="http://localhost:9999")
        schemas = spec["components"]["schemas"]
        assert isinstance(schemas, dict)
        assert len(schemas) > 0

    def test_components_has_security_schemes(self):
        spec = generate_openapi_spec(base_url="http://localhost:9999")
        assert "securitySchemes" in spec["components"]
        assert "BearerAuth" in spec["components"]["securitySchemes"]

    def test_servers_populated(self):
        spec = generate_openapi_spec(base_url="http://test:8000")
        assert len(spec["servers"]) >= 1
        assert spec["servers"][0]["url"] == "http://test:8000"

    def test_spec_is_json_serializable(self):
        spec = generate_openapi_spec(base_url="http://localhost:9999")
        # Must not raise
        json.dumps(spec)


# =============================================================================
# Section 4: save_openapi_spec
# =============================================================================

@pytest.mark.unit
class TestSaveOpenapiSpec:
    """Tests for save_openapi_spec file writing."""

    pytestmark = pytest.mark.skipif(
        not OPENAPI_AVAILABLE, reason="Jotty openapi module not importable"
    )

    def test_saves_valid_json(self, tmp_path):
        spec = {"openapi": "3.0.3", "info": {"title": "Test"}}
        output = tmp_path / "openapi.json"
        save_openapi_spec(spec, output)

        loaded = json.loads(output.read_text())
        assert loaded["openapi"] == "3.0.3"
        assert loaded["info"]["title"] == "Test"

    def test_creates_parent_directories(self, tmp_path):
        spec = {"openapi": "3.0.3"}
        nested = tmp_path / "a" / "b" / "c" / "spec.json"
        save_openapi_spec(spec, nested)

        assert nested.exists()
        loaded = json.loads(nested.read_text())
        assert loaded["openapi"] == "3.0.3"

    def test_overwrites_existing_file(self, tmp_path):
        output = tmp_path / "openapi.json"
        save_openapi_spec({"v": 1}, output)
        save_openapi_spec({"v": 2}, output)
        loaded = json.loads(output.read_text())
        assert loaded["v"] == 2


# =============================================================================
# Section 5: SwarmCodeGenerator
# =============================================================================

@pytest.mark.unit
class TestGeneratedCodeDataclass:
    """Tests for the GeneratedCode dataclass."""

    pytestmark = pytest.mark.skipif(
        not CODEGEN_AVAILABLE, reason="SwarmCodeGenerator not importable"
    )

    def test_required_fields(self):
        gc = GeneratedCode(
            code="print('hi')",
            language="python",
            dependencies=["requests"],
            description="Hello script",
        )
        assert gc.code == "print('hi')"
        assert gc.language == "python"
        assert gc.dependencies == ["requests"]
        assert gc.description == "Hello script"

    def test_optional_fields_default(self):
        gc = GeneratedCode(
            code="x", language="py", dependencies=[], description="d"
        )
        assert gc.usage_example is None
        assert gc.file_path is None
        assert gc.metadata == {}

    def test_optional_fields_set(self):
        gc = GeneratedCode(
            code="x",
            language="python",
            dependencies=[],
            description="d",
            usage_example="example()",
            file_path="/tmp/gen.py",
            metadata={"key": "val"},
        )
        assert gc.usage_example == "example()"
        assert gc.file_path == "/tmp/gen.py"
        assert gc.metadata == {"key": "val"}


@pytest.mark.unit
class TestSwarmCodeGenerator:
    """Tests for SwarmCodeGenerator methods."""

    pytestmark = pytest.mark.skipif(
        not CODEGEN_AVAILABLE, reason="SwarmCodeGenerator not importable"
    )

    # ── generate_glue_code (template fallback) ───────────────────────────

    def test_generate_glue_code_returns_generated_code(self):
        """Without an LLM, should fall back to template-based generation."""
        gen = SwarmCodeGenerator()
        # Patch _init_dependencies to avoid importing TaskPlanner
        gen._planner = MagicMock()

        result = gen.generate_glue_code("source_tool", "dest_tool")

        assert isinstance(result, GeneratedCode)
        assert result.language == "python"
        assert "source_tool" in result.code
        assert "dest_tool" in result.code

    def test_generate_glue_code_with_transformation(self):
        gen = SwarmCodeGenerator()
        gen._planner = MagicMock()

        result = gen.generate_glue_code(
            "redis_cache", "postgres_db", transformation="flatten JSON"
        )

        assert isinstance(result, GeneratedCode)
        assert "redis_cache" in result.code
        assert "postgres_db" in result.code

    # ── generate_provider_adapter (template fallback) ────────────────────

    def test_generate_provider_adapter_returns_generated_code(self):
        gen = SwarmCodeGenerator()
        gen._planner = MagicMock()

        result = gen.generate_provider_adapter(
            package_name="pytesseract",
            package_info={"description": "OCR tool", "version": "0.3.10"},
            categories=["document"],
        )

        assert isinstance(result, GeneratedCode)
        assert result.language == "python"
        assert "pytesseract" in result.code.lower() or "Pytesseract" in result.code
        assert result.dependencies == ["pytesseract"]

    def test_generate_provider_adapter_file_path(self):
        gen = SwarmCodeGenerator()
        gen._planner = MagicMock()

        result = gen.generate_provider_adapter(
            "my-package", {"description": "A pkg"}, ["data"]
        )
        assert result.file_path == "providers/my_package_provider.py"

    def test_generate_provider_adapter_metadata(self):
        gen = SwarmCodeGenerator()
        gen._planner = MagicMock()

        result = gen.generate_provider_adapter(
            "cool-lib", {"description": "Cool"}, ["ai", "nlp"]
        )
        assert result.metadata["package_name"] == "cool-lib"
        assert result.metadata["categories"] == ["ai", "nlp"]
        assert result.metadata["generated_by"] == "template"

    # ── _extract_code_from_response ──────────────────────────────────────

    def test_extract_code_from_python_block(self):
        gen = SwarmCodeGenerator()
        response = "Here is the code:\n```python\nprint('hello')\n```\nDone."
        assert gen._extract_code_from_response(response) == "print('hello')"

    def test_extract_code_from_generic_block(self):
        gen = SwarmCodeGenerator()
        response = "Code:\n```\nfoo()\n```"
        assert gen._extract_code_from_response(response) == "foo()"

    def test_extract_code_no_block_returns_stripped(self):
        gen = SwarmCodeGenerator()
        response = "  just plain text  "
        assert gen._extract_code_from_response(response) == "just plain text"

    def test_extract_code_multiple_blocks_returns_first(self):
        gen = SwarmCodeGenerator()
        response = "```python\nfirst()\n```\n```python\nsecond()\n```"
        assert gen._extract_code_from_response(response) == "first()"

    # ── _extract_dependencies ────────────────────────────────────────────

    def test_extract_dependencies_from_imports(self):
        gen = SwarmCodeGenerator()
        code = "import requests\nimport numpy\nfrom pandas import DataFrame"
        deps = gen._extract_dependencies(code)
        assert "requests" in deps
        assert "numpy" in deps
        assert "pandas" in deps

    def test_extract_dependencies_excludes_stdlib(self):
        gen = SwarmCodeGenerator()
        code = "import os\nimport sys\nimport json\nimport logging\nimport requests"
        deps = gen._extract_dependencies(code)
        assert "os" not in deps
        assert "sys" not in deps
        assert "json" not in deps
        assert "logging" not in deps
        assert "requests" in deps

    def test_extract_dependencies_deduplicates(self):
        gen = SwarmCodeGenerator()
        code = "import requests\nimport requests\nfrom requests import get"
        deps = gen._extract_dependencies(code)
        assert deps.count("requests") == 1

    def test_extract_dependencies_empty_code(self):
        gen = SwarmCodeGenerator()
        deps = gen._extract_dependencies("x = 1 + 2")
        assert deps == []
