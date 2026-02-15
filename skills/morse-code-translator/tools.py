"""Morse Code Translator Skill — encode/decode Morse code."""

from typing import Any, Dict

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_error, tool_response, tool_wrapper

status = SkillStatus("morse-code-translator")

_ENCODE = {
    "A": ".-",
    "B": "-...",
    "C": "-.-.",
    "D": "-..",
    "E": ".",
    "F": "..-.",
    "G": "--.",
    "H": "....",
    "I": "..",
    "J": ".---",
    "K": "-.-",
    "L": ".-..",
    "M": "--",
    "N": "-.",
    "O": "---",
    "P": ".--.",
    "Q": "--.-",
    "R": ".-.",
    "S": "...",
    "T": "-",
    "U": "..-",
    "V": "...-",
    "W": ".--",
    "X": "-..-",
    "Y": "-.--",
    "Z": "--..",
    "0": "-----",
    "1": ".----",
    "2": "..---",
    "3": "...--",
    "4": "....-",
    "5": ".....",
    "6": "-....",
    "7": "--...",
    "8": "---..",
    "9": "----.",
    ".": ".-.-.-",
    ",": "--..--",
    "?": "..--..",
    "'": ".----.",
    "!": "-.-.--",
    "/": "-..-.",
    "(": "-.--.",
    ")": "-.--.-",
    "&": ".-...",
    ":": "---...",
    ";": "-.-.-.",
    "=": "-...-",
    "+": ".-.-.",
    "-": "-....-",
    "_": "..--.-",
    '"': ".-..-.",
    "$": "...-..-",
    "@": ".--.-.",
}
_DECODE = {v: k for k, v in _ENCODE.items()}


@tool_wrapper(required_params=["action"])
def morse_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Encode text to Morse code or decode Morse to text."""
    status.set_callback(params.pop("_status_callback", None))
    action = params["action"]

    if action == "encode":
        text = params.get("text", "")
        if not text:
            return tool_error("text required")
        words = text.upper().split()
        coded_words = []
        for word in words:
            letters = []
            for ch in word:
                code = _ENCODE.get(ch)
                if code:
                    letters.append(code)
            coded_words.append(" ".join(letters))
        morse = " / ".join(coded_words)
        return tool_response(text=text, morse=morse)

    if action == "decode":
        morse = params.get("morse", "")
        if not morse:
            return tool_error("morse required")
        words = morse.strip().split(" / ")
        decoded = []
        for word in words:
            chars = []
            for code in word.strip().split():
                ch = _DECODE.get(code, "?")
                chars.append(ch)
            decoded.append("".join(chars))
        text = " ".join(decoded)
        return tool_response(morse=morse, text=text)

    return tool_error(f"Unknown action: {action}. Use: encode, decode")


__all__ = ["morse_tool"]
