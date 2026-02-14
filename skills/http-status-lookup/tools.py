"""Look up HTTP status code meanings and categories."""
from typing import Dict, Any
from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.utils.skill_status import SkillStatus

status = SkillStatus("http-status-lookup")

_CODES: Dict[int, tuple] = {
    100: ("Continue", "Server received headers, client should proceed"),
    101: ("Switching Protocols", "Server switching to requested protocol"),
    200: ("OK", "Request succeeded"),
    201: ("Created", "Resource created successfully"),
    202: ("Accepted", "Request accepted for processing"),
    204: ("No Content", "Success with no response body"),
    301: ("Moved Permanently", "Resource permanently moved to new URL"),
    302: ("Found", "Resource temporarily at different URL"),
    304: ("Not Modified", "Resource not modified since last request"),
    307: ("Temporary Redirect", "Temporary redirect preserving method"),
    308: ("Permanent Redirect", "Permanent redirect preserving method"),
    400: ("Bad Request", "Server cannot process due to client error"),
    401: ("Unauthorized", "Authentication required"),
    403: ("Forbidden", "Server refuses to authorize request"),
    404: ("Not Found", "Resource not found"),
    405: ("Method Not Allowed", "HTTP method not supported for resource"),
    408: ("Request Timeout", "Server timed out waiting for request"),
    409: ("Conflict", "Request conflicts with current resource state"),
    410: ("Gone", "Resource permanently removed"),
    413: ("Payload Too Large", "Request body exceeds server limit"),
    415: ("Unsupported Media Type", "Media type not supported"),
    418: ("I'm a Teapot", "RFC 2324 - server is a teapot"),
    422: ("Unprocessable Entity", "Request well-formed but has semantic errors"),
    429: ("Too Many Requests", "Rate limit exceeded"),
    500: ("Internal Server Error", "Unexpected server error"),
    501: ("Not Implemented", "Server does not support the functionality"),
    502: ("Bad Gateway", "Invalid response from upstream server"),
    503: ("Service Unavailable", "Server temporarily unavailable"),
    504: ("Gateway Timeout", "Upstream server timed out"),
}
_CATEGORIES = {1: "Informational", 2: "Success", 3: "Redirection", 4: "Client Error", 5: "Server Error"}


@tool_wrapper()
def lookup_http_status(params: Dict[str, Any]) -> Dict[str, Any]:
    """Look up an HTTP status code or list codes by category."""
    status.set_callback(params.pop("_status_callback", None))
    code = params.get("code")
    category = params.get("category")

    if code is not None:
        code = int(code)
        info = _CODES.get(code)
        if not info:
            cat = _CATEGORIES.get(code // 100, "Unknown")
            return tool_response(code=code, name="Unknown", category=cat,
                                 description=f"Non-standard {cat.lower()} status code")
        return tool_response(code=code, name=info[0], category=_CATEGORIES.get(code // 100, "Unknown"),
                             description=info[1])

    if category is not None:
        cat_num = int(category[0]) if isinstance(category, str) and category[0].isdigit() else category
        codes = {c: v[0] for c, v in _CODES.items() if c // 100 == int(cat_num)}
        return tool_response(category=_CATEGORIES.get(int(cat_num), "Unknown"), codes=codes)

    return tool_response(categories=_CATEGORIES, total_codes=len(_CODES))


__all__ = ["lookup_http_status"]
