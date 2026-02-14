"""NATO Phonetic Alphabet Skill — convert text to NATO spelling."""
from typing import Dict, Any

from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.utils.skill_status import SkillStatus

status = SkillStatus("nato-phonetic-alphabet")

_NATO = {
    "A": "Alfa", "B": "Bravo", "C": "Charlie", "D": "Delta", "E": "Echo",
    "F": "Foxtrot", "G": "Golf", "H": "Hotel", "I": "India", "J": "Juliett",
    "K": "Kilo", "L": "Lima", "M": "Mike", "N": "November", "O": "Oscar",
    "P": "Papa", "Q": "Quebec", "R": "Romeo", "S": "Sierra", "T": "Tango",
    "U": "Uniform", "V": "Victor", "W": "Whiskey", "X": "X-ray", "Y": "Yankee",
    "Z": "Zulu",
    "0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
    "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Niner",
}
_REVERSE = {v.upper(): k for k, v in _NATO.items()}


@tool_wrapper(required_params=["action"])
def nato_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert text to NATO phonetic alphabet or decode back."""
    status.set_callback(params.pop("_status_callback", None))
    action = params["action"]

    if action == "encode":
        text = params.get("text", "")
        if not text:
            return tool_error("text required")
        result = []
        for ch in text.upper():
            if ch == " ":
                result.append("[space]")
            elif ch in _NATO:
                result.append(_NATO[ch])
            else:
                result.append(ch)
        return tool_response(
            text=text, nato=" ".join(result),
            words=result, count=len(result),
        )

    if action == "decode":
        words_str = params.get("words", "")
        if not words_str:
            return tool_error("words required")
        words = words_str.split()
        chars = []
        for w in words:
            if w == "[space]":
                chars.append(" ")
            else:
                chars.append(_REVERSE.get(w.upper(), w))
        text = "".join(chars)
        return tool_response(words=words_str, text=text)

    return tool_error(f"Unknown action: {action}. Use: encode, decode")


__all__ = ["nato_tool"]
