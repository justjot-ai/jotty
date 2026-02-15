"""
Data Extractor Skill

Intelligently extracts data from ANY format:
- Direct strings
- Objects with methods (introspection)
- Callables
- File paths
- Dicts
- JSON strings
- Mixed formats

NO configuration needed - just works!
"""

import logging
from typing import Dict, Any, Optional
from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, async_tool_wrapper

logger = logging.getLogger(__name__)
status = SkillStatus("data-extractor")


@async_tool_wrapper()
async def extract_data(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract data from any format using multiple strategies.

    The extractor tries different strategies until one works:
    - Direct string
    - Callable function
    - Dict key lookup
    - Object method/attribute
    - File path reading
    - JSON parsing
    - String fallback

    Params:
        data_source: Data in ANY format (string, dict, object, file path, callable, etc.)
        param_name: What to extract (e.g., 'table_metadata', 'user_info')
        context_key: Optional key in context providers

    Returns:
        {
            "success": True,
            "extracted_data": ...,
            "strategy_used": "json_string",
            "source_type": "str"
        }
    """
    status.set_callback(params.pop("_status_callback", None))

    try:
        from .extractor import SmartDataExtractor

        data_source = params.get("data_source")
        param_name = params.get("param_name", "data")
        context_key = params.get("context_key")

        if data_source is None:
            return tool_error("data_source is required")

        status.update(f"Extracting '{param_name}' from {type(data_source).__name__}...")

        extractor = SmartDataExtractor()
        result = await extractor.extract(data_source, param_name, context_key)

        if result is None:
            status.error("All extraction strategies failed")
            return tool_error(
                f"Could not extract '{param_name}' from {type(data_source).__name__}",
                metadata={
                    "source_type": type(data_source).__name__,
                    "strategies_tried": [s["strategy"] for s in extractor.extraction_stats],
                }
            )

        # Get successful strategy
        successful_strategy = next(
            (s["strategy"] for s in extractor.extraction_stats if s["success"]),
            "unknown"
        )

        status.complete(f"Extracted using strategy: {successful_strategy}")

        return tool_response(
            extracted_data=result,
            strategy_used=successful_strategy,
            source_type=type(data_source).__name__,
            strategies_tried=len(extractor.extraction_stats)
        )

    except Exception as e:
        logger.error(f"Data extraction failed: {e}")
        status.error(f"Failed: {e}")
        return tool_error(str(e))


@async_tool_wrapper()
async def extract_from_file(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract data from a file (JSON, CSV, text, etc.).

    Params:
        file_path: Path to file
        format: Optional format hint (json, csv, text)

    Returns:
        {
            "success": True,
            "extracted_data": ...,
            "format_detected": "json"
        }
    """
    status.set_callback(params.pop("_status_callback", None))

    try:
        import json
        from pathlib import Path

        file_path = params.get("file_path")
        format_hint = params.get("format")

        if not file_path:
            return tool_error("file_path is required")

        path = Path(file_path)
        if not path.exists():
            return tool_error(f"File not found: {file_path}")

        status.update(f"Reading file: {file_path}")

        content = path.read_text()

        # Try to detect format
        if format_hint == "json" or file_path.endswith(".json"):
            data = json.loads(content)
            detected_format = "json"
        elif format_hint == "csv" or file_path.endswith(".csv"):
            import csv
            import io
            reader = csv.DictReader(io.StringIO(content))
            data = list(reader)
            detected_format = "csv"
        else:
            data = content
            detected_format = "text"

        status.complete(f"File read successfully as {detected_format}")

        return tool_response(
            extracted_data=data,
            format_detected=detected_format,
            file_size=len(content)
        )

    except json.JSONDecodeError as e:
        return tool_error(f"Invalid JSON: {e}")
    except Exception as e:
        logger.error(f"File extraction failed: {e}")
        status.error(f"Failed: {e}")
        return tool_error(str(e))


@async_tool_wrapper()
async def extract_from_object(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract data from Python object using introspection.

    Tries methods, attributes, dict access, etc.

    Params:
        obj: Python object
        key: Key/method/attribute name to extract

    Returns:
        {
            "success": True,
            "extracted_data": ...,
            "extraction_method": "attribute"
        }
    """
    status.set_callback(params.pop("_status_callback", None))

    try:
        from .extractor import SmartDataExtractor

        obj = params.get("obj")
        key = params.get("key")

        if obj is None:
            return tool_error("obj is required")
        if not key:
            return tool_error("key is required")

        status.update(f"Extracting '{key}' from {type(obj).__name__}...")

        extractor = SmartDataExtractor()
        result = await extractor.extract(obj, key)

        if result is None:
            return tool_error(f"Could not extract '{key}' from object")

        successful_strategy = next(
            (s["strategy"] for s in extractor.extraction_stats if s["success"]),
            "unknown"
        )

        status.complete(f"Extracted using: {successful_strategy}")

        return tool_response(
            extracted_data=result,
            extraction_method=successful_strategy
        )

    except Exception as e:
        logger.error(f"Object extraction failed: {e}")
        status.error(f"Failed: {e}")
        return tool_error(str(e))


__all__ = [
    "extract_data",
    "extract_from_file",
    "extract_from_object",
]
