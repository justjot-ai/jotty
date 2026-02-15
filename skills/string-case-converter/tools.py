"""String Case Converter Skill."""
import re
from typing import Dict, Any

from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("string-case-converter")


def _split_words(text: str) -> list:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
    return re.split(r"[\s_\-]+", text.strip())


@tool_wrapper(required_params=["text", "to_case"])
def convert_case_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert string between naming conventions."""
    status.set_callback(params.pop("_status_callback", None))
    words = _split_words(params["text"])
    target = params["to_case"].lower().replace(" ", "").replace("_", "")

    if target == "camelcase":
        result = words[0].lower() + "".join(w.capitalize() for w in words[1:])
    elif target == "pascalcase":
        result = "".join(w.capitalize() for w in words)
    elif target in ("snakecase", "snake"):
        result = "_".join(w.lower() for w in words)
    elif target in ("kebabcase", "kebab"):
        result = "-".join(w.lower() for w in words)
    elif target in ("uppercase", "upper", "screamingsnake"):
        result = "_".join(w.upper() for w in words)
    elif target in ("titlecase", "title"):
        result = " ".join(w.capitalize() for w in words)
    elif target in ("lowercase", "lower"):
        result = " ".join(w.lower() for w in words)
    elif target in ("dotcase", "dot"):
        result = ".".join(w.lower() for w in words)
    else:
        return tool_error(f"Unknown case: {params['to_case']}. Use: camelCase, snake_case, kebab-case, PascalCase, UPPER_CASE, Title Case")

    return tool_response(result=result, from_words=words, to_case=params["to_case"])


__all__ = ["convert_case_tool"]
