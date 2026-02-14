"""Calculate and verify checksums using hashlib."""
import hashlib
from pathlib import Path
from typing import Dict, Any
from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.utils.skill_status import SkillStatus

status = SkillStatus("checksum-verifier")

_ALGOS = {"md5": hashlib.md5, "sha1": hashlib.sha1, "sha256": hashlib.sha256, "sha512": hashlib.sha512}


def _hash_bytes(data: bytes, algo: str) -> str:
    fn = _ALGOS.get(algo.lower())
    if not fn:
        raise ValueError(f"Unsupported algorithm: {algo}. Use: {', '.join(_ALGOS)}")
    return fn(data).hexdigest()


@tool_wrapper()
def calculate_checksum(params: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate a checksum for text or a file."""
    status.set_callback(params.pop("_status_callback", None))
    algo = params.get("algorithm", "sha256").lower()
    text = params.get("text")
    file_path = params.get("file_path")
    if not text and not file_path:
        return tool_error("Provide either text or file_path")
    if text:
        digest = _hash_bytes(text.encode("utf-8"), algo)
        return tool_response(checksum=digest, algorithm=algo, input_type="text",
                             length=len(text))
    p = Path(file_path)
    if not p.exists():
        return tool_error(f"File not found: {file_path}")
    h = _ALGOS.get(algo)
    if not h:
        return tool_error(f"Unsupported algorithm: {algo}")
    hasher = h()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return tool_response(checksum=hasher.hexdigest(), algorithm=algo,
                         input_type="file", file=str(p), size=p.stat().st_size)


@tool_wrapper(required_params=["expected"])
def verify_checksum(params: Dict[str, Any]) -> Dict[str, Any]:
    """Verify a checksum matches an expected value."""
    status.set_callback(params.pop("_status_callback", None))
    expected = params["expected"].strip().lower()
    result = calculate_checksum(params)
    if not result.get("success"):
        return result
    actual = result["checksum"]
    match = actual == expected
    return tool_response(match=match, expected=expected, actual=actual,
                         algorithm=result["algorithm"])


__all__ = ["calculate_checksum", "verify_checksum"]
