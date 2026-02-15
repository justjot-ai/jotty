"""Parse and generate TOML content."""
import json
from typing import Dict, Any
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("toml-parser")

# Python 3.11+ has tomllib; fallback to tomli
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:
        tomllib = None  # type: ignore[assignment]


def _to_toml(data: dict, prefix: str = "") -> str:
    """Simple TOML serializer for basic types."""
    lines = []
    tables = []
    for k, v in data.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            tables.append((full_key, v))
        elif isinstance(v, str):
            lines.append(f'{k} = "{v}"')
        elif isinstance(v, bool):
            lines.append(f"{k} = {'true' if v else 'false'}")
        elif isinstance(v, (int, float)):
            lines.append(f"{k} = {v}")
        elif isinstance(v, list):
            items = ", ".join(json.dumps(i) for i in v)
            lines.append(f"{k} = [{items}]")
        else:
            lines.append(f'{k} = "{v}"')
    result = "\n".join(lines)
    for tkey, tval in tables:
        result += f"\n\n[{tkey}]\n" + _to_toml(tval)
    return result


@tool_wrapper(required_params=["toml_text"])
def parse_toml(params: Dict[str, Any]) -> Dict[str, Any]:
    """Parse TOML text and return as dict."""
    status.set_callback(params.pop("_status_callback", None))
    if tomllib is None:
        return tool_error("No TOML parser available. Use Python 3.11+ or pip install tomli")
    try:
        parsed = tomllib.loads(params["toml_text"])
    except Exception as e:
        return tool_error(f"Invalid TOML: {e}")
    return tool_response(data=parsed, sections=list(parsed.keys()))


@tool_wrapper(required_params=["data"])
def generate_toml(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate TOML text from a dict."""
    status.set_callback(params.pop("_status_callback", None))
    data = params["data"]
    if not isinstance(data, dict):
        return tool_error("data must be a dict")
    toml_text = _to_toml(data)
    return tool_response(toml=toml_text)


__all__ = ["parse_toml", "generate_toml"]
