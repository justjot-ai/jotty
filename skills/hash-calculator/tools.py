"""Hash Calculator Skill — compute and verify hashes."""
import hashlib
from pathlib import Path
from typing import Dict, Any

from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.utils.skill_status import SkillStatus

status = SkillStatus("hash-calculator")

ALGORITHMS = {"md5", "sha1", "sha256", "sha512", "sha384", "sha224"}


@tool_wrapper()
def hash_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Compute hash of text or file."""
    status.set_callback(params.pop("_status_callback", None))
    text = params.get("text")
    file_path = params.get("file_path")
    algo = params.get("algorithm", "sha256").lower()

    if algo not in ALGORITHMS:
        return tool_error(f"Unsupported algorithm: {algo}. Use one of: {sorted(ALGORITHMS)}")
    if not text and not file_path:
        return tool_error("Provide either text or file_path")

    h = hashlib.new(algo)
    if file_path:
        p = Path(file_path)
        if not p.exists():
            return tool_error(f"File not found: {file_path}")
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
    else:
        h.update(text.encode("utf-8"))

    return tool_response(hash=h.hexdigest(), algorithm=algo,
                         input_type="file" if file_path else "text")


@tool_wrapper(required_params=["expected_hash"])
def verify_hash_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Verify a hash matches expected value."""
    status.set_callback(params.pop("_status_callback", None))
    result = hash_tool(params)
    if not result.get("success"):
        return result
    match = result["hash"].lower() == params["expected_hash"].strip().lower()
    return tool_response(match=match, computed_hash=result["hash"],
                         expected_hash=params["expected_hash"], algorithm=result["algorithm"])


__all__ = ["hash_tool", "verify_hash_tool"]
