"""Base64 Encoder Skill — encode/decode Base64 and hex."""
import base64
import binascii
from typing import Dict, Any

from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("base64-encoder")


@tool_wrapper()
def base64_encode_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Encode text to Base64, URL-safe Base64, or hex."""
    status.set_callback(params.pop("_status_callback", None))
    text = params.get("text", "")
    if not text:
        return tool_error("text parameter required")
    encoding = params.get("encoding", "base64").lower()
    data = text.encode("utf-8")

    if encoding == "base64":
        encoded = base64.b64encode(data).decode("ascii")
    elif encoding in ("base64url", "urlsafe"):
        encoded = base64.urlsafe_b64encode(data).decode("ascii")
    elif encoding == "hex":
        encoded = data.hex()
    else:
        return tool_error(f"Unsupported encoding: {encoding}. Use: base64, base64url, hex")

    return tool_response(encoded=encoded, encoding=encoding, original_length=len(text))


@tool_wrapper()
def base64_decode_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Decode Base64, URL-safe Base64, or hex string."""
    status.set_callback(params.pop("_status_callback", None))
    encoded = params.get("encoded", "") or params.get("text", "")
    if not encoded:
        return tool_error("encoded parameter required")
    encoding = params.get("encoding", "base64").lower()

    try:
        if encoding == "base64":
            decoded = base64.b64decode(encoded).decode("utf-8")
        elif encoding in ("base64url", "urlsafe"):
            decoded = base64.urlsafe_b64decode(encoded).decode("utf-8")
        elif encoding == "hex":
            decoded = bytes.fromhex(encoded).decode("utf-8")
        else:
            return tool_error(f"Unsupported encoding: {encoding}")
        return tool_response(decoded=decoded, encoding=encoding)
    except (binascii.Error, ValueError, UnicodeDecodeError) as e:
        return tool_error(f"Decode failed: {e}")


__all__ = ["base64_encode_tool", "base64_decode_tool"]
