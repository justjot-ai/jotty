"""Validate YAML syntax and convert YAML<->JSON."""
import json
from typing import Dict, Any
from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.utils.skill_status import SkillStatus

status = SkillStatus("yaml-validator")

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


@tool_wrapper(required_params=["yaml_text"])
def validate_yaml(params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate YAML text and optionally convert to JSON."""
    status.set_callback(params.pop("_status_callback", None))
    if not HAS_YAML:
        return tool_error("PyYAML not installed. Run: pip install pyyaml")
    text = params["yaml_text"]
    to_json = params.get("to_json", False)
    try:
        docs = list(yaml.safe_load_all(text))
        parsed = docs[0] if len(docs) == 1 else docs
    except yaml.YAMLError as e:
        return tool_error(f"Invalid YAML: {e}")
    result = dict(valid=True, document_count=len(docs), type=type(parsed).__name__)
    if to_json:
        result["json"] = json.dumps(parsed, indent=2, default=str)
    return tool_response(data=result)


@tool_wrapper(required_params=["json_text"])
def json_to_yaml(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert JSON text to YAML."""
    status.set_callback(params.pop("_status_callback", None))
    if not HAS_YAML:
        return tool_error("PyYAML not installed. Run: pip install pyyaml")
    try:
        data = json.loads(params["json_text"])
    except json.JSONDecodeError as e:
        return tool_error(f"Invalid JSON: {e}")
    yaml_text = yaml.dump(data, default_flow_style=False, sort_keys=False)
    return tool_response(yaml=yaml_text)


__all__ = ["validate_yaml", "json_to_yaml"]
