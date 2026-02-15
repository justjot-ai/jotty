"""DNS Lookup Skill — resolve domain records."""
import socket
import requests
from typing import Dict, Any, List

from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("dns-lookup")


@tool_wrapper(required_params=["domain"])
def dns_lookup_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Perform DNS lookup using DNS-over-HTTPS (Cloudflare)."""
    status.set_callback(params.pop("_status_callback", None))
    domain = params["domain"].strip().rstrip(".")
    record_type = params.get("record_type", "A").upper()
    valid_types = {"A", "AAAA", "MX", "CNAME", "TXT", "NS", "SOA", "SRV", "PTR"}

    if record_type not in valid_types:
        return tool_error(f"Invalid record type. Use one of: {sorted(valid_types)}")

    try:
        resp = requests.get(
            "https://cloudflare-dns.com/dns-query",
            params={"name": domain, "type": record_type},
            headers={"Accept": "application/dns-json"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        records = []
        for answer in data.get("Answer", []):
            records.append({
                "name": answer.get("name", ""),
                "type": answer.get("type", 0),
                "ttl": answer.get("TTL", 0),
                "data": answer.get("data", ""),
            })

        return tool_response(domain=domain, record_type=record_type,
                             records=records, count=len(records),
                             status_code=data.get("Status", -1))
    except requests.RequestException as e:
        return tool_error(f"DNS lookup failed: {e}")


@tool_wrapper(required_params=["domain"])
def dns_all_records_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Get all common DNS record types for a domain."""
    status.set_callback(params.pop("_status_callback", None))
    domain = params["domain"]
    all_records = {}
    for rtype in ["A", "AAAA", "MX", "NS", "TXT", "CNAME"]:
        result = dns_lookup_tool({"domain": domain, "record_type": rtype})
        if result.get("success") and result.get("records"):
            all_records[rtype] = result["records"]
    return tool_response(domain=domain, records=all_records)


__all__ = ["dns_lookup_tool", "dns_all_records_tool"]
