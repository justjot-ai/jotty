"""Generate OpenAPI/Swagger docs from endpoint definitions."""
import json
from typing import Dict, Any, List
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, tool_wrapper
from Jotty.core.infrastructure.utils.skill_status import SkillStatus
status = SkillStatus("api-docs-generator")


@tool_wrapper(required_params=["title", "endpoints"])
def generate_api_docs(params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate OpenAPI 3.0 spec from endpoint definitions.

    Params:
        title: API title
        version: API version (default "1.0.0")
        description: API description
        base_url: server base URL
        endpoints: list of endpoint dicts:
            - path: URL path (e.g. /users/{id})
            - method: HTTP method
            - summary: endpoint summary
            - description: detailed description
            - parameters: list of {name, in, type, required, description}
            - request_body: {type, properties} for POST/PUT
            - responses: dict of {status_code: {description, schema}}
            - tags: list of tag strings
    """
    status.set_callback(params.pop("_status_callback", None))
    title = params["title"]
    version = params.get("version", "1.0.0")
    desc = params.get("description", "")
    base_url = params.get("base_url", "http://localhost:8000")
    endpoints = params["endpoints"]

    spec: Dict[str, Any] = {
        "openapi": "3.0.3",
        "info": {"title": title, "version": version, "description": desc},
        "servers": [{"url": base_url}],
        "paths": {},
    }

    for ep in endpoints:
        path = ep.get("path", "/")
        method = ep.get("method", "get").lower()
        operation: Dict[str, Any] = {
            "summary": ep.get("summary", ""),
            "description": ep.get("description", ""),
        }
        if ep.get("tags"):
            operation["tags"] = ep["tags"]

        # Parameters
        if ep.get("parameters"):
            operation["parameters"] = []
            for p in ep["parameters"]:
                operation["parameters"].append({
                    "name": p["name"],
                    "in": p.get("in", "query"),
                    "required": p.get("required", False),
                    "description": p.get("description", ""),
                    "schema": {"type": p.get("type", "string")},
                })

        # Request body
        if ep.get("request_body") and method in ("post", "put", "patch"):
            rb = ep["request_body"]
            props = {}
            for k, v in rb.get("properties", {}).items():
                props[k] = {"type": v} if isinstance(v, str) else v
            operation["requestBody"] = {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"type": "object", "properties": props}
                    }
                },
            }

        # Responses
        responses = ep.get("responses", {"200": {"description": "Success"}})
        operation["responses"] = {}
        for code, info in responses.items():
            resp: Dict[str, Any] = {"description": info.get("description", "")}
            if info.get("schema"):
                resp["content"] = {
                    "application/json": {"schema": info["schema"]}
                }
            operation["responses"][str(code)] = resp

        spec["paths"].setdefault(path, {})[method] = operation

    spec_json = json.dumps(spec, indent=2)
    return tool_response(openapi_spec=spec, spec_json=spec_json, endpoint_count=len(endpoints))


__all__ = ["generate_api_docs"]
