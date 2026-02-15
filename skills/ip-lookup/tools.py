"""IP Lookup Skill — geolocation and network info."""

from typing import Any, Dict

import requests

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_error, tool_response, tool_wrapper

status = SkillStatus("ip-lookup")


@tool_wrapper()
def ip_lookup_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Look up IP geolocation and network info."""
    status.set_callback(params.pop("_status_callback", None))
    ip = params.get("ip", "")

    try:
        url = f"http://ip-api.com/json/{ip}" if ip else "http://ip-api.com/json/"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") == "fail":
            return tool_error(data.get("message", "Lookup failed"))

        return tool_response(
            ip=data.get("query", ip),
            country=data.get("country", ""),
            country_code=data.get("countryCode", ""),
            region=data.get("regionName", ""),
            city=data.get("city", ""),
            zip=data.get("zip", ""),
            lat=data.get("lat"),
            lon=data.get("lon"),
            timezone=data.get("timezone", ""),
            isp=data.get("isp", ""),
            org=data.get("org", ""),
            as_number=data.get("as", ""),
        )
    except requests.RequestException as e:
        return tool_error(f"Lookup failed: {e}")


__all__ = ["ip_lookup_tool"]
