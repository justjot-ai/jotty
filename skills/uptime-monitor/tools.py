"""Uptime Monitor Skill — check HTTP endpoint availability."""

import time
from typing import Any, Dict

import requests

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_error, tool_response, tool_wrapper

status = SkillStatus("uptime-monitor")


@tool_wrapper(required_params=["url"])
def check_endpoint_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Check HTTP endpoint availability and response time."""
    status.set_callback(params.pop("_status_callback", None))
    url = params["url"].strip()
    method = params.get("method", "GET").upper()
    timeout = min(int(params.get("timeout", 10)), 30)
    expected_status = int(params.get("expected_status", 200))

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    start = time.monotonic()
    try:
        resp = requests.request(method, url, timeout=timeout, allow_redirects=True)
        elapsed_ms = round((time.monotonic() - start) * 1000, 1)

        available = resp.status_code == expected_status
        headers_dict = {k: v for k, v in list(resp.headers.items())[:20]}

        return tool_response(
            url=url,
            status_code=resp.status_code,
            response_time_ms=elapsed_ms,
            available=available,
            content_length=len(resp.content),
            headers=headers_dict,
            redirect_url=resp.url if resp.url != url else None,
        )
    except requests.ConnectionError:
        elapsed_ms = round((time.monotonic() - start) * 1000, 1)
        return tool_response(
            url=url,
            status_code=0,
            response_time_ms=elapsed_ms,
            available=False,
            error="Connection refused",
        )
    except requests.Timeout:
        elapsed_ms = round((time.monotonic() - start) * 1000, 1)
        return tool_response(
            url=url, status_code=0, response_time_ms=elapsed_ms, available=False, error="Timeout"
        )
    except requests.RequestException as e:
        return tool_error(f"Request failed: {e}")


@tool_wrapper(required_params=["urls"])
def check_multiple_endpoints_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Check multiple endpoints at once."""
    status.set_callback(params.pop("_status_callback", None))
    urls = params["urls"]
    if not isinstance(urls, list):
        return tool_error("urls must be a list")

    results = []
    for url in urls[:20]:
        result = check_endpoint_tool({"url": url, "timeout": params.get("timeout", 10)})
        results.append(result)

    available_count = sum(1 for r in results if r.get("available"))
    return tool_response(
        results=results,
        total=len(results),
        available=available_count,
        unavailable=len(results) - available_count,
    )


__all__ = ["check_endpoint_tool", "check_multiple_endpoints_tool"]
