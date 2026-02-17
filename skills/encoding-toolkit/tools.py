"""
Encoding Toolkit — Unified encoding, hashing, security, and identity skill.

Consolidates: base64-encoder, hash-calculator, jwt-decoder, password-generator,
uuid-generator, checksum-verifier, encryption-tool.
"""

import base64
import binascii
import hashlib
import json
import math
import secrets
import string
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_error, tool_response, tool_wrapper

status = SkillStatus("encoding-toolkit")

# =============================================================================
# CONSTANTS
# =============================================================================

HASH_ALGORITHMS = {"md5", "sha1", "sha256", "sha512", "sha384", "sha224"}

ULID_CHARS = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"

WORD_LIST = [
    "abandon",
    "ability",
    "able",
    "about",
    "above",
    "absent",
    "absorb",
    "abstract",
    "access",
    "accident",
    "account",
    "achieve",
    "acquire",
    "across",
    "action",
    "adapt",
    "address",
    "adjust",
    "admit",
    "advance",
    "advice",
    "afford",
    "agent",
    "agree",
    "ahead",
    "alarm",
    "album",
    "alert",
    "alien",
    "allow",
    "almost",
    "alpha",
    "already",
    "alter",
    "always",
    "amazing",
    "among",
    "amount",
    "anchor",
    "ancient",
    "anger",
    "angle",
    "animal",
    "annual",
    "answer",
    "antenna",
    "apple",
    "armor",
    "army",
    "arrive",
    "arrow",
    "basket",
    "battle",
    "beach",
    "beauty",
    "become",
    "before",
    "begin",
    "behind",
    "believe",
    "below",
    "bench",
    "benefit",
    "beyond",
    "bicycle",
    "blanket",
    "blast",
    "blossom",
    "board",
    "bonus",
    "border",
    "bottle",
    "bounce",
    "brave",
    "breeze",
    "bridge",
    "bright",
    "broken",
    "bronze",
    "bubble",
    "budget",
    "buffalo",
    "burden",
    "cabin",
    "cable",
    "camera",
    "canal",
    "canyon",
    "carbon",
    "cargo",
    "carpet",
    "castle",
    "catalog",
    "cattle",
    "ceiling",
    "cement",
    "census",
    "certain",
    "chair",
    "change",
    "chapter",
    "charge",
    "cherry",
    "chicken",
    "choice",
    "circle",
    "citizen",
    "civil",
    "claim",
    "clarify",
    "click",
    "climb",
    "clinic",
    "clock",
    "cluster",
    "coach",
    "coconut",
    "coffee",
    "collect",
    "column",
    "combine",
    "comfort",
    "common",
    "company",
    "concert",
    "connect",
    "control",
    "copper",
    "coral",
    "correct",
    "cotton",
    "country",
    "couple",
    "course",
    "cousin",
    "cover",
    "craft",
    "cream",
    "credit",
    "cricket",
    "cross",
    "crowd",
    "crystal",
    "custom",
    "cycle",
    "damage",
    "dance",
    "danger",
    "daring",
    "dawn",
    "debate",
    "decade",
    "define",
    "demand",
    "depart",
    "deposit",
    "depth",
    "derive",
    "desert",
    "design",
    "detail",
    "detect",
    "develop",
    "device",
    "diamond",
    "diary",
    "digital",
    "dinner",
    "dinosaur",
    "direct",
    "dismiss",
    "display",
    "distance",
    "divide",
    "dolphin",
    "domain",
    "dragon",
    "drama",
    "dream",
    "drift",
    "driver",
    "dynamic",
    "eager",
    "eagle",
    "earth",
    "ecology",
    "economy",
    "educate",
    "effort",
    "either",
    "elbow",
    "elder",
    "electric",
    "elegant",
    "element",
    "elephant",
    "elevator",
    "elite",
    "embark",
    "embrace",
    "emerge",
    "emotion",
    "employ",
    "empower",
    "enable",
    "endorse",
    "energy",
    "enforce",
    "engage",
    "engine",
    "enhance",
    "enjoy",
    "ensure",
    "enter",
    "entire",
    "entry",
    "envelope",
    "episode",
    "equal",
    "equip",
    "escape",
    "essence",
    "estate",
    "eternal",
    "evidence",
    "evolve",
    "example",
    "exchange",
    "excite",
    "exercise",
    "exhibit",
    "expand",
    "expect",
    "explain",
    "expose",
    "extend",
    "fabric",
    "faculty",
    "falcon",
    "family",
    "famous",
    "fantasy",
    "fashion",
    "father",
    "fault",
    "favorite",
    "feature",
    "federal",
    "fence",
    "festival",
    "fiction",
    "field",
    "figure",
    "filter",
    "finger",
    "finish",
    "fitness",
    "flame",
    "flash",
    "flavor",
    "flight",
    "float",
    "flower",
    "fluid",
    "focus",
    "follow",
    "force",
    "forest",
    "forget",
    "fortune",
    "forward",
    "fossil",
    "foster",
    "fragile",
    "frame",
    "freedom",
    "frequent",
    "fresh",
    "friend",
    "frost",
    "frozen",
    "fruit",
    "fuel",
    "future",
    "gadget",
    "galaxy",
    "gallery",
    "garden",
    "gather",
    "general",
    "gentle",
    "genius",
    "gesture",
    "giant",
    "ginger",
    "giraffe",
    "glacier",
    "glance",
    "glimpse",
    "globe",
    "glory",
    "glove",
    "golden",
    "gospel",
    "govern",
    "grace",
    "grain",
    "grant",
    "gravity",
    "grocery",
    "group",
    "growth",
    "guard",
    "guitar",
    "habit",
    "hammer",
    "hamster",
    "harbor",
    "harvest",
    "hazard",
    "health",
    "heart",
    "heaven",
    "heavy",
    "height",
    "helmet",
    "hidden",
    "highway",
    "history",
    "hobby",
    "hockey",
    "holiday",
    "hollow",
    "honey",
    "horizon",
    "horror",
    "hospital",
    "hotel",
    "humble",
    "humor",
    "hundred",
    "hungry",
    "hybrid",
]


# =============================================================================
# INTERNAL HELPERS
# =============================================================================


def _generate_ulid() -> str:
    t = int(time.time() * 1000)
    time_part = ""
    for _ in range(10):
        time_part = ULID_CHARS[t & 0x1F] + time_part
        t >>= 5
    rand_part = "".join(secrets.choice(ULID_CHARS) for _ in range(16))
    return time_part + rand_part


def _estimate_strength(length: int, charset_size: int) -> str:
    bits = length * math.log2(charset_size) if charset_size > 0 else 0
    if bits >= 128:
        return "very_strong"
    elif bits >= 80:
        return "strong"
    elif bits >= 60:
        return "moderate"
    return "weak"


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


# =============================================================================
# ENCODING TOOLS
# =============================================================================


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


# =============================================================================
# HASHING TOOLS
# =============================================================================


@tool_wrapper()
def hash_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Compute hash of text or file (md5, sha1, sha256, sha512, sha384, sha224)."""
    status.set_callback(params.pop("_status_callback", None))
    text = params.get("text")
    file_path = params.get("file_path")
    algo = params.get("algorithm", "sha256").lower()
    if algo not in HASH_ALGORITHMS:
        return tool_error(f"Unsupported algorithm: {algo}. Use one of: {sorted(HASH_ALGORITHMS)}")
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
    return tool_response(
        hash=h.hexdigest(), algorithm=algo, input_type="file" if file_path else "text"
    )


@tool_wrapper(required_params=["expected_hash"])
def verify_hash_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Verify a hash matches expected value."""
    status.set_callback(params.pop("_status_callback", None))
    result = hash_tool(params)
    if not result.get("success"):
        return result
    match = result["hash"].lower() == params["expected_hash"].strip().lower()
    return tool_response(
        match=match,
        computed_hash=result["hash"],
        expected_hash=params["expected_hash"],
        algorithm=result["algorithm"],
    )


# =============================================================================
# JWT TOOL
# =============================================================================


@tool_wrapper(required_params=["token"])
def decode_jwt_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Decode a JWT token without verification, showing header, payload, expiry."""
    status.set_callback(params.pop("_status_callback", None))
    token = params["token"].strip()
    parts = token.split(".")
    if len(parts) != 3:
        return tool_error(f"Invalid JWT: expected 3 parts, got {len(parts)}")
    try:

        def _decode_segment(segment: str) -> dict:
            padding = 4 - len(segment) % 4
            segment += "=" * padding
            decoded = base64.urlsafe_b64decode(segment)
            return json.loads(decoded)

        header = _decode_segment(parts[0])
        payload = _decode_segment(parts[1])
    except (json.JSONDecodeError, Exception) as e:
        return tool_error(f"Failed to decode JWT: {e}")
    expired = None
    expires_at = issued_at = None
    if "exp" in payload:
        exp_dt = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
        expired = exp_dt < datetime.now(timezone.utc)
        expires_at = exp_dt.isoformat()
    if "iat" in payload:
        issued_at = datetime.fromtimestamp(payload["iat"], tz=timezone.utc).isoformat()
    return tool_response(
        header=header,
        payload=payload,
        expired=expired,
        expires_at=expires_at,
        issued_at=issued_at,
        algorithm=header.get("alg", "unknown"),
    )


# =============================================================================
# PASSWORD / PASSPHRASE TOOLS
# =============================================================================


@tool_wrapper()
def generate_password_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate cryptographically secure random passwords."""
    status.set_callback(params.pop("_status_callback", None))
    length = min(max(int(params.get("length", 16)), 4), 128)
    count = min(max(int(params.get("count", 1)), 1), 20)
    charset = ""
    if params.get("lowercase", True):
        charset += string.ascii_lowercase
    if params.get("uppercase", True):
        charset += string.ascii_uppercase
    if params.get("digits", True):
        charset += string.digits
    if params.get("symbols", True):
        charset += "!@#$%^&*()-_=+[]{}|;:,.<>?"
    if not charset:
        return tool_error("At least one character class must be enabled")
    passwords = ["".join(secrets.choice(charset) for _ in range(length)) for _ in range(count)]
    return tool_response(
        passwords=passwords,
        strength=_estimate_strength(length, len(charset)),
        length=length,
        charset_size=len(charset),
    )


@tool_wrapper()
def generate_passphrase_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a passphrase from random dictionary words."""
    status.set_callback(params.pop("_status_callback", None))
    word_count = min(max(int(params.get("words", 5)), 3), 12)
    separator = params.get("separator", "-")
    capitalize = params.get("capitalize", True)
    words = [secrets.choice(WORD_LIST) for _ in range(word_count)]
    if capitalize:
        words = [w.capitalize() for w in words]
    passphrase = separator.join(words)
    bits = word_count * math.log2(len(WORD_LIST))
    return tool_response(
        passphrase=passphrase,
        word_count=word_count,
        entropy_bits=round(bits, 1),
        strength=_estimate_strength(word_count, len(WORD_LIST)),
    )


# =============================================================================
# UUID / ULID TOOLS
# =============================================================================


@tool_wrapper()
def generate_uuid_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate UUID identifiers (v1, v4, v5)."""
    status.set_callback(params.pop("_status_callback", None))
    version = int(params.get("version", 4))
    count = min(max(int(params.get("count", 1)), 1), 100)
    upper = params.get("uppercase", False)
    uuids = []
    for _ in range(count):
        if version == 1:
            u = str(uuid.uuid1())
        elif version == 4:
            u = str(uuid.uuid4())
        elif version == 5:
            namespace = params.get("namespace", "dns")
            name = params.get("name", "example.com")
            ns = {
                "dns": uuid.NAMESPACE_DNS,
                "url": uuid.NAMESPACE_URL,
                "oid": uuid.NAMESPACE_OID,
                "x500": uuid.NAMESPACE_X500,
            }.get(namespace, uuid.NAMESPACE_DNS)
            u = str(uuid.uuid5(ns, name))
        else:
            return tool_error(f"Unsupported version: {version}. Use 1, 4, or 5")
        uuids.append(u.upper() if upper else u)
    return tool_response(uuids=uuids, version=version, count=len(uuids))


@tool_wrapper()
def generate_ulid_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate ULID identifiers (sortable, 128-bit, Crockford Base32)."""
    status.set_callback(params.pop("_status_callback", None))
    count = min(max(int(params.get("count", 1)), 1), 100)
    ulids = [_generate_ulid() for _ in range(count)]
    return tool_response(ulids=ulids, count=len(ulids))


# =============================================================================
# CIPHER TOOL
# =============================================================================


@tool_wrapper(required_params=["operation", "text"])
def encryption_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Apply educational ciphers: Caesar, Vigenere, ROT13, XOR."""
    status.set_callback(params.pop("_status_callback", None))
    op = params["operation"].lower()
    text = params["text"]
    decrypt = params.get("decrypt", False)
    try:
        if op == "caesar":
            shift = int(params.get("shift", 3))
            return tool_response(result=_caesar(text, shift, decrypt), cipher="caesar", shift=shift)
        if op == "rot13":
            return tool_response(result=_caesar(text, 13), cipher="rot13")
        if op == "vigenere":
            key = params.get("key", "")
            if not key:
                return tool_error("key required for Vigenere cipher")
            return tool_response(result=_vigenere(text, key, decrypt), cipher="vigenere", key=key)
        if op == "xor":
            key = params.get("key", "")
            if not key:
                return tool_error("key required for XOR")
            hex_result = _xor(text, key).encode("utf-8", errors="replace").hex()
            return tool_response(result=hex_result, cipher="xor", note="Hex-encoded output")
        return tool_error(f"Unknown operation: {op}. Use: caesar, rot13, vigenere, xor")
    except Exception as e:
        return tool_error(str(e))


__all__ = [
    "base64_encode_tool",
    "base64_decode_tool",
    "hash_tool",
    "verify_hash_tool",
    "decode_jwt_tool",
    "generate_password_tool",
    "generate_passphrase_tool",
    "generate_uuid_tool",
    "generate_ulid_tool",
    "encryption_tool",
]
