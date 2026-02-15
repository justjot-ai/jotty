"""Webhook Dispatcher Skill — send webhooks with retry logic."""
import time
import json
import hashlib
import hmac
import requests
from typing import Dict, Any

from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus

status = SkillStatus("webhook-dispatcher")


@tool_wrapper(required_params=["url", "payload"])
def send_webhook_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Send HTTP webhook payload with retry logic."""
    status.set_callback(params.pop("_status_callback", None))
    url = params["url"]
    payload = params["payload"]
    method = params.get("method", "POST").upper()
    custom_headers = params.get("headers", {})
    max_retries = min(int(params.get("retries", 3)), 5)
    timeout = min(int(params.get("timeout", 10)), 30)
    secret = params.get("secret")

    if not url.startswith(("http://", "https://")):
        return tool_error("URL must start with http:// or https://")

    headers = {"Content-Type": "application/json", "User-Agent": "Jotty-Webhook/1.0"}
    headers.update(custom_headers)

    body = json.dumps(payload)

    if secret:
        sig = hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()
        headers["X-Webhook-Signature"] = f"sha256={sig}"

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.request(method, url, data=body, headers=headers, timeout=timeout)
            if resp.status_code < 500:
                return tool_response(
                    status_code=resp.status_code,
                    attempts=attempt,
                    response_body=resp.text[:1000],
                    url=url,
                    delivered=200 <= resp.status_code < 300,
                )
            last_error = f"HTTP {resp.status_code}"
        except requests.RequestException as e:
            last_error = str(e)

        if attempt < max_retries:
            time.sleep(min(2 ** (attempt - 1), 8))

    return tool_response(
        status_code=0,
        attempts=max_retries,
        delivered=False,
        error=f"All {max_retries} attempts failed. Last error: {last_error}",
        url=url,
    )


__all__ = ["send_webhook_tool"]
