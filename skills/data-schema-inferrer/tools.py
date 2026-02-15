"""Data Schema Inferrer Skill — infer JSON Schema from data (pure Python)."""
import json
from typing import Dict, Any, List, Optional

from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("data-schema-inferrer")


def _infer_type(value: Any) -> Dict[str, Any]:
    """Infer JSON Schema for a single value."""
    if value is None:
        return {"type": "null"}
    elif isinstance(value, bool):
        return {"type": "boolean"}
    elif isinstance(value, int):
        return {"type": "integer"}
    elif isinstance(value, float):
        return {"type": "number"}
    elif isinstance(value, str):
        schema = {"type": "string"}
        if len(value) > 0:
            # Detect common formats
            import re
            if re.match(r"^\d{4}-\d{2}-\d{2}$", value):
                schema["format"] = "date"
            elif re.match(r"^\d{4}-\d{2}-\d{2}T", value):
                schema["format"] = "date-time"
            elif re.match(r"^[^@]+@[^@]+\.[^@]+$", value):
                schema["format"] = "email"
            elif re.match(r"^https?://", value):
                schema["format"] = "uri"
            elif re.match(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", value, re.I):
                schema["format"] = "uuid"
        return schema
    elif isinstance(value, list):
        if not value:
            return {"type": "array", "items": {}}
        item_schemas = [_infer_type(item) for item in value]
        # If all items same type, use single schema
        types = set(json.dumps(s, sort_keys=True) for s in item_schemas)
        if len(types) == 1:
            return {"type": "array", "items": item_schemas[0]}
        else:
            return {"type": "array", "items": {"oneOf": item_schemas}}
    elif isinstance(value, dict):
        properties = {}
        required = []
        for k, v in value.items():
            properties[k] = _infer_type(v)
            if v is not None:
                required.append(k)
        schema = {"type": "object", "properties": properties}
        if required:
            schema["required"] = sorted(required)
        return schema
    else:
        return {"type": "string"}


def _merge_schemas(samples: List[Any]) -> Dict[str, Any]:
    """Merge schemas from multiple samples."""
    if not samples:
        return {}
    if len(samples) == 1:
        return _infer_type(samples[0])

    # For array of objects, merge property sets
    all_props = {}
    required_counts = {}
    total = len(samples)

    for sample in samples:
        schema = _infer_type(sample)
        if schema.get("type") == "object":
            for prop, prop_schema in schema.get("properties", {}).items():
                if prop not in all_props:
                    all_props[prop] = prop_schema
                required_counts[prop] = required_counts.get(prop, 0) + 1

    required = [k for k, count in required_counts.items() if count == total]
    result = {"type": "object", "properties": all_props}
    if required:
        result["required"] = sorted(required)
    return result


@tool_wrapper(required_params=["data"])
def infer_schema_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Infer JSON Schema from sample data."""
    status.set_callback(params.pop("_status_callback", None))
    data = params["data"]
    title = params.get("title", "Inferred Schema")

    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError as e:
            return tool_error(f"Invalid JSON: {e}")

    if isinstance(data, list) and data and isinstance(data[0], dict):
        schema = _merge_schemas(data)
    else:
        schema = _infer_type(data)

    schema["$schema"] = "http://json-schema.org/draft-07/schema#"
    schema["title"] = title

    return tool_response(schema=schema)


@tool_wrapper(required_params=["data", "schema"])
def validate_against_schema_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate data against a JSON Schema (basic validation)."""
    status.set_callback(params.pop("_status_callback", None))
    data = params["data"]
    schema = params["schema"]
    errors = []

    def _validate(value, sch, path="$"):
        expected_type = sch.get("type")
        type_map = {"string": str, "integer": int, "number": (int, float),
                    "boolean": bool, "array": list, "object": dict, "null": type(None)}
        if expected_type and expected_type in type_map:
            if not isinstance(value, type_map[expected_type]):
                errors.append(f"{path}: expected {expected_type}, got {type(value).__name__}")
                return
        if expected_type == "object" and isinstance(value, dict):
            for req in sch.get("required", []):
                if req not in value:
                    errors.append(f"{path}: missing required property '{req}'")
            for prop, prop_schema in sch.get("properties", {}).items():
                if prop in value:
                    _validate(value[prop], prop_schema, f"{path}.{prop}")
        elif expected_type == "array" and isinstance(value, list):
            item_schema = sch.get("items", {})
            for i, item in enumerate(value):
                _validate(item, item_schema, f"{path}[{i}]")

    _validate(data, schema)
    return tool_response(valid=len(errors) == 0, errors=errors, error_count=len(errors))


__all__ = ["infer_schema_tool", "validate_against_schema_tool"]
