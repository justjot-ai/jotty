"""URL Parser Skill — parse and manipulate URLs."""
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse, quote, unquote
from typing import Dict, Any

from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.utils.skill_status import SkillStatus

status = SkillStatus("url-parser")


@tool_wrapper(required_params=["url"])
def parse_url_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a URL into its components."""
    status.set_callback(params.pop("_status_callback", None))
    try:
        parsed = urlparse(params["url"])
        query_params = {k: v[0] if len(v) == 1 else v for k, v in parse_qs(parsed.query).items()}
        return tool_response(
            scheme=parsed.scheme, host=parsed.hostname or "",
            port=parsed.port, path=parsed.path,
            query=query_params, fragment=parsed.fragment,
            username=parsed.username, password=parsed.password,
        )
    except Exception as e:
        return tool_error(f"Failed to parse URL: {e}")


@tool_wrapper()
def build_url_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """Build a URL from components."""
    status.set_callback(params.pop("_status_callback", None))
    scheme = params.get("scheme", "https")
    host = params.get("host", "")
    port = params.get("port")
    path = params.get("path", "/")
    query = params.get("query", {})
    fragment = params.get("fragment", "")

    if not host:
        return tool_error("host parameter required")

    netloc = host
    if port:
        netloc = f"{host}:{port}"
    qs = urlencode(query) if query else ""
    url = urlunparse((scheme, netloc, path, "", qs, fragment))
    return tool_response(url=url)


@tool_wrapper(required_params=["text"])
def url_encode_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """URL-encode or decode text."""
    status.set_callback(params.pop("_status_callback", None))
    text = params["text"]
    mode = params.get("mode", "encode")
    if mode == "decode":
        return tool_response(result=unquote(text))
    return tool_response(result=quote(text, safe=params.get("safe", "")))


__all__ = ["parse_url_tool", "build_url_tool", "url_encode_tool"]
