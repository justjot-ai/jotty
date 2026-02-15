"""Encryption tool — educational ciphers (Caesar, Vigenere, ROT13, XOR)."""
from typing import Dict, Any
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("encryption-tool")

def _caesar(text: str, shift: int, decrypt: bool = False) -> str:
    if decrypt:
        shift = -shift
    result = []
    for ch in text:
        if ch.isalpha():
            base = ord("A") if ch.isupper() else ord("a")
            result.append(chr((ord(ch) - base + shift) % 26 + base))
        else:
            result.append(ch)
    return "".join(result)

def _vigenere(text: str, key: str, decrypt: bool = False) -> str:
    if not key.isalpha():
        raise ValueError("Key must be alphabetic")
    key = key.lower()
    result, ki = [], 0
    for ch in text:
        if ch.isalpha():
            base = ord("A") if ch.isupper() else ord("a")
            shift = ord(key[ki % len(key)]) - ord("a")
            if decrypt:
                shift = -shift
            result.append(chr((ord(ch) - base + shift) % 26 + base))
            ki += 1
        else:
            result.append(ch)
    return "".join(result)

def _xor(text: str, key: str) -> str:
    return "".join(chr(ord(c) ^ ord(key[i % len(key)])) for i, c in enumerate(text))

@tool_wrapper(required_params=["operation", "text"])
def encryption_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Educational ciphers: caesar, vigenere, rot13, xor."""
    status.set_callback(params.pop("_status_callback", None))
    op = params["operation"].lower()
    text = params["text"]
    decrypt = params.get("decrypt", False)
    try:
        if op == "caesar":
            shift = int(params.get("shift", 3))
            result = _caesar(text, shift, decrypt)
            return tool_response(result=result, cipher="caesar", shift=shift)
        if op == "rot13":
            result = _caesar(text, 13)
            return tool_response(result=result, cipher="rot13")
        if op == "vigenere":
            key = params.get("key", "")
            if not key:
                return tool_error("key required for Vigenere cipher")
            result = _vigenere(text, key, decrypt)
            return tool_response(result=result, cipher="vigenere", key=key)
        if op == "xor":
            key = params.get("key", "")
            if not key:
                return tool_error("key required for XOR")
            result = _xor(text, key)
            hex_result = result.encode("utf-8", errors="replace").hex()
            return tool_response(result=hex_result, cipher="xor", note="Hex-encoded output")
        return tool_error(f"Unknown op: {op}. Use caesar/rot13/vigenere/xor")
    except Exception as e:
        return tool_error(str(e))

__all__ = ["encryption_tool"]
