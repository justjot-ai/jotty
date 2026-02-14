"""UUID Generator Skill — generate UUIDs and ULIDs."""
import uuid
import time
import secrets
from typing import Dict, Any

from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.utils.skill_status import SkillStatus

status = SkillStatus("uuid-generator")

ULID_CHARS = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def _generate_ulid() -> str:
    t = int(time.time() * 1000)
    time_part = ""
    for _ in range(10):
        time_part = ULID_CHARS[t & 0x1F] + time_part
        t >>= 5
    rand_part = "".join(secrets.choice(ULID_CHARS) for _ in range(16))
    return time_part + rand_part


@tool_wrapper()
def generate_uuid_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate UUID identifiers."""
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
            ns = {"dns": uuid.NAMESPACE_DNS, "url": uuid.NAMESPACE_URL,
                  "oid": uuid.NAMESPACE_OID, "x500": uuid.NAMESPACE_X500}.get(
                namespace, uuid.NAMESPACE_DNS)
            u = str(uuid.uuid5(ns, name))
        else:
            return tool_error(f"Unsupported version: {version}. Use 1, 4, or 5")
        uuids.append(u.upper() if upper else u)

    return tool_response(uuids=uuids, version=version, count=len(uuids))


@tool_wrapper()
def generate_ulid_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate ULID identifiers (sortable, 128-bit)."""
    status.set_callback(params.pop("_status_callback", None))
    count = min(max(int(params.get("count", 1)), 1), 100)
    ulids = [_generate_ulid() for _ in range(count)]
    return tool_response(ulids=ulids, count=len(ulids))


__all__ = ["generate_uuid_tool", "generate_ulid_tool"]
