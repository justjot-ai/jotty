"""JSON Transformer Skill — flatten, merge, query JSON."""
import json
import copy
from typing import Dict, Any, List

from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.utils.skill_status import SkillStatus

status = SkillStatus("json-transformer")


def _flatten(obj: Any, prefix: str = "", sep: str = ".") -> Dict[str, Any]:
    items = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{prefix}{sep}{k}" if prefix else k
            items.update(_flatten(v, new_key, sep))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            new_key = f"{prefix}{sep}{i}" if prefix else str(i)
            items.update(_flatten(v, new_key, sep))
    else:
        items[prefix] = obj
    return items


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def _query_path(data: Any, path: str) -> Any:
    parts = path.split(".")
    current = data
    for part in parts:
        if isinstance(current, dict):
            if part not in current:
                raise KeyError(f"Key not found: {part}")
            current = current[part]
        elif isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                raise KeyError(f"Invalid index: {part}")
        else:
            raise KeyError(f"Cannot traverse into {type(current).__name__}")
    return current


@tool_wrapper(required_params=["data"])
def flatten_json_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten nested JSON into dot-notation keys."""
    status.set_callback(params.pop("_status_callback", None))
    data = params["data"]
    sep = params.get("separator", ".")
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError as e:
            return tool_error(f"Invalid JSON: {e}")
    result = _flatten(data, sep=sep)
    return tool_response(result=result, keys_count=len(result))


@tool_wrapper(required_params=["objects"])
def merge_json_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two or more JSON objects."""
    status.set_callback(params.pop("_status_callback", None))
    objects = params["objects"]
    if not isinstance(objects, list) or len(objects) < 2:
        return tool_error("Provide a list of at least 2 objects")
    result = {}
    for obj in objects:
        if isinstance(obj, str):
            obj = json.loads(obj)
        if not isinstance(obj, dict):
            return tool_error("All items must be JSON objects")
        result = _deep_merge(result, obj)
    return tool_response(result=result)


@tool_wrapper(required_params=["data", "path"])
def query_json_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Query JSON with dot-notation path."""
    status.set_callback(params.pop("_status_callback", None))
    data = params["data"]
    if isinstance(data, str):
        data = json.loads(data)
    try:
        result = _query_path(data, params["path"])
        return tool_response(result=result, path=params["path"])
    except KeyError as e:
        return tool_error(str(e))


__all__ = ["flatten_json_tool", "merge_json_tool", "query_json_tool"]
