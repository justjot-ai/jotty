"""JWT Decoder Skill — decode and inspect JWT tokens."""
import json
import base64
from datetime import datetime, timezone
from typing import Dict, Any

from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("jwt-decoder")


def _decode_segment(segment: str) -> dict:
    padding = 4 - len(segment) % 4
    segment += "=" * padding
    decoded = base64.urlsafe_b64decode(segment)
    return json.loads(decoded)


@tool_wrapper(required_params=["token"])
def decode_jwt_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Decode a JWT token without verification."""
    status.set_callback(params.pop("_status_callback", None))
    token = params["token"].strip()
    parts = token.split(".")

    if len(parts) != 3:
        return tool_error(f"Invalid JWT: expected 3 parts, got {len(parts)}")

    try:
        header = _decode_segment(parts[0])
        payload = _decode_segment(parts[1])
    except (json.JSONDecodeError, Exception) as e:
        return tool_error(f"Failed to decode JWT: {e}")

    expired = None
    expires_at = None
    if "exp" in payload:
        exp_dt = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
        expired = exp_dt < datetime.now(timezone.utc)
        expires_at = exp_dt.isoformat()

    issued_at = None
    if "iat" in payload:
        issued_at = datetime.fromtimestamp(payload["iat"], tz=timezone.utc).isoformat()

    return tool_response(header=header, payload=payload, expired=expired,
                         expires_at=expires_at, issued_at=issued_at,
                         algorithm=header.get("alg", "unknown"))


__all__ = ["decode_jwt_tool"]
