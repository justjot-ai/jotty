"""
Data Transformer Skill

SOTA Agentic data transformation with ReAct + Format Tools.

The transformer is a ReAct agent with TOOLS to test formats:
1. Agent decides: "Try json.loads"
2. Tool executes: Returns success/error
3. Agent sees error: "Missing quotes"
4. Agent decides: "Fix and retry"
5. Tool executes: Returns success + result

Transforms between: JSON ↔ CSV ↔ Dict ↔ List ↔ String
"""

import logging
from typing import Any, Dict, get_type_hints

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_error, tool_response, tool_wrapper

logger = logging.getLogger(__name__)
status = SkillStatus("data-transformer")


@tool_wrapper(required_params=["source", "target_format"])
def transform_data_format(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform data from one format to another using ReAct agent.

    Iteratively tries formats, gets errors, fixes, and retries until success.

    Params:
        source: Source data (any format: string, dict, list, etc.)
        target_format: Target format ('dict', 'list', 'str', 'json', 'csv')
        context: Optional context for intelligent transformation
        param_name: Optional parameter name for context

    Returns:
        {
            "success": True,
            "result": ...,
            "source_type": "str",
            "target_type": "dict",
            "transformations_tried": 3
        }
    """
    status.set_callback(params.pop("_status_callback", None))

    try:
        from .transformer import SmartDataTransformer

        source = params["source"]
        target_format = params["target_format"].lower()
        context = params.get("context", "")
        param_name = params.get("param_name", "data")

        # Map string format to Python type
        format_type_map = {
            "dict": dict,
            "list": list,
            "str": str,
            "string": str,
            "json": dict,  # JSON typically becomes dict
            "csv": list,  # CSV becomes list of dicts
        }

        if target_format not in format_type_map:
            return tool_error(
                f"Unsupported target_format: {target_format}. "
                f"Supported: {list(format_type_map.keys())}"
            )

        target_type = format_type_map[target_format]
        source_type = type(source).__name__

        status.update(f"Transforming {source_type} → {target_format}...")

        transformer = SmartDataTransformer()
        result = transformer.transform(
            source=source, target_type=target_type, context=context, param_name=param_name
        )

        transformations_tried = len(transformer.transformation_history)

        status.complete(f"Transformed successfully ({transformations_tried} attempts)")

        return tool_response(
            result=result,
            source_type=source_type,
            target_type=target_format,
            transformations_tried=transformations_tried,
            transformation_history=(
                transformer.transformation_history[-3:] if transformations_tried > 0 else []
            ),
        )

    except Exception as e:
        logger.error(f"Data transformation failed: {e}")
        status.error(f"Failed: {e}")
        return tool_error(str(e))


@tool_wrapper(required_params=["source"])
def parse_json_string(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse JSON string with automatic fixing.

    Uses ReAct agent to iteratively fix JSON errors.

    Params:
        source: JSON string (may have errors)

    Returns:
        {
            "success": True,
            "result": {...},
            "fixes_applied": ["added_quotes", "fixed_trailing_comma"]
        }
    """
    status.set_callback(params.pop("_status_callback", None))

    try:
        from .transformer import SmartDataTransformer

        source = params["source"]

        if not isinstance(source, str):
            return tool_error(f"Expected string, got {type(source).__name__}")

        status.update("Parsing JSON with auto-fix...")

        transformer = SmartDataTransformer()
        result = transformer.transform(source, dict, context="JSON data")

        status.complete("JSON parsed successfully")

        return tool_response(result=result, fixes_applied=len(transformer.transformation_history))

    except Exception as e:
        logger.error(f"JSON parsing failed: {e}")
        status.error(f"Failed: {e}")
        return tool_error(str(e))


@tool_wrapper(required_params=["source"])
def parse_csv_string(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse CSV string to list of dicts.

    Params:
        source: CSV string
        has_header: Whether CSV has header row (default: True)

    Returns:
        {
            "success": True,
            "result": [{"col1": "val1", ...}, ...],
            "row_count": 10
        }
    """
    status.set_callback(params.pop("_status_callback", None))

    try:
        import csv
        import io

        source = params["source"]
        has_header = params.get("has_header", True)

        if not isinstance(source, str):
            return tool_error(f"Expected string, got {type(source).__name__}")

        status.update("Parsing CSV...")

        if has_header:
            reader = csv.DictReader(io.StringIO(source))
            result = list(reader)
        else:
            reader = csv.reader(io.StringIO(source))
            result = [list(row) for row in reader]

        status.complete(f"Parsed {len(result)} rows")

        return tool_response(result=result, row_count=len(result), has_header=has_header)

    except Exception as e:
        logger.error(f"CSV parsing failed: {e}")
        status.error(f"Failed: {e}")
        return tool_error(str(e))


@tool_wrapper(required_params=["data"])
def convert_to_json(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert any data structure to JSON string.

    Params:
        data: Data to convert (dict, list, etc.)
        pretty: Pretty-print JSON (default: False)
        indent: Indentation spaces if pretty=True (default: 2)

    Returns:
        {
            "success": True,
            "result": "...",
            "size_bytes": 1234
        }
    """
    status.set_callback(params.pop("_status_callback", None))

    try:
        import json

        data = params["data"]
        pretty = params.get("pretty", False)
        indent = params.get("indent", 2) if pretty else None

        status.update("Converting to JSON...")

        result = json.dumps(data, indent=indent, ensure_ascii=False)

        status.complete(f"Converted to JSON ({len(result)} bytes)")

        return tool_response(result=result, size_bytes=len(result), pretty=pretty)

    except TypeError as e:
        return tool_error(f"Data not JSON serializable: {e}")
    except Exception as e:
        logger.error(f"JSON conversion failed: {e}")
        status.error(f"Failed: {e}")
        return tool_error(str(e))


__all__ = [
    "transform_data_format",
    "parse_json_string",
    "parse_csv_string",
    "convert_to_json",
]
