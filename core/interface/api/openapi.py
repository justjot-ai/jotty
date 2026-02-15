"""
OpenAPI Specification Generator - Auto-generated from SDK types
===============================================================

Generates OpenAPI 3.0 spec by introspecting sdk_types.py dataclasses.
Single source of truth: the Python types. OpenAPI spec is always in sync.

Usage:
    from Jotty.core.interface.api.openapi import generate_openapi_spec, save_openapi_spec
    spec = generate_openapi_spec()
    save_openapi_spec(spec, Path("sdk/openapi.json"))
"""

import inspect
import json
import logging
from dataclasses import MISSING
from dataclasses import fields as dc_fields
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, get_type_hints

logger = logging.getLogger(__name__)


def _python_type_to_openapi(
    py_type: Any, enums_collected: Dict[str, list] = None
) -> Dict[str, Any]:
    """Convert a Python type annotation to OpenAPI schema."""
    origin = getattr(py_type, "__origin__", None)

    # Handle Optional[X] -> nullable X
    if origin is Union:
        args = py_type.__args__
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            schema = _python_type_to_openapi(non_none[0], enums_collected)
            schema["nullable"] = True
            return schema
        return {"oneOf": [_python_type_to_openapi(a, enums_collected) for a in non_none]}

    # Handle List[X]
    if origin is list:
        item_type = py_type.__args__[0] if py_type.__args__ else Any
        return {"type": "array", "items": _python_type_to_openapi(item_type, enums_collected)}

    # Handle Dict[K, V]
    if origin is dict:
        return {"type": "object", "additionalProperties": True}

    # Handle Enum
    if isinstance(py_type, type) and issubclass(py_type, Enum):
        values = [e.value for e in py_type]
        if enums_collected is not None:
            enums_collected[py_type.__name__] = values
        return {"type": "string", "enum": values}

    # Handle Callable (skip in schema)
    if py_type is type(None):
        return {"nullable": True}

    # Primitives
    type_map = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number", "format": "float"},
        bool: {"type": "boolean"},
        datetime: {"type": "string", "format": "date-time"},
        Any: {},
    }

    return type_map.get(py_type, {"type": "object", "additionalProperties": True})


def _dataclass_to_schema(cls, enums_collected: Dict[str, list] = None) -> Dict[str, Any]:
    """Convert a dataclass to OpenAPI schema, reading field types."""
    try:
        hints = get_type_hints(cls)
    except Exception:
        hints = {}

    properties = {}
    required = []

    for f in dc_fields(cls):
        # Skip callback fields (not serializable)
        if "callback" in f.name.lower() or f.name.startswith("_"):
            continue

        py_type = hints.get(f.name, Any)
        schema = _python_type_to_openapi(py_type, enums_collected)

        # Add description from default if available
        if f.default is not MISSING:
            if isinstance(f.default, Enum):
                schema["default"] = f.default.value
            elif isinstance(f.default, (str, int, float, bool)):
                schema["default"] = f.default

        properties[f.name] = schema

        # Required if no default and no default_factory
        if f.default is MISSING and f.default_factory is MISSING:
            required.append(f.name)

    result = {
        "type": "object",
        "properties": properties,
    }
    if required:
        result["required"] = required

    # Add description from docstring
    if cls.__doc__:
        first_line = cls.__doc__.strip().split("\n")[0].strip()
        if first_line:
            result["description"] = first_line

    return result


def generate_openapi_spec(
    title: str = "Jotty API",
    version: str = "2.0.0",
    description: str = "AI Agent Framework - Multi-agent orchestration API",
    base_url: str = None,
) -> Dict[str, Any]:
    """
    Generate OpenAPI 3.0 spec from sdk_types.py dataclasses.

    Introspects ExecutionContext, SDKEvent, SDKSession, SDKResponse, SDKRequest
    and generates schemas automatically. Endpoints defined declaratively.
    """
    import os

    if base_url is None:
        try:
            from Jotty.core.infrastructure.foundation.config_defaults import DEFAULTS as _DEFAULTS

            base_url = os.getenv("JOTTY_GATEWAY_URL", _DEFAULTS.JOTTY_GATEWAY_URL)
        except ImportError:
            base_url = os.getenv("JOTTY_GATEWAY_URL", "http://localhost:8766")

    from Jotty.core.infrastructure.foundation.types.sdk_types import (
        ChannelType,
        ExecutionContext,
        ExecutionMode,
        ResponseFormat,
        SDKEvent,
        SDKEventType,
        SDKRequest,
        SDKResponse,
        SDKSession,
    )

    enums_collected = {}

    # Auto-generate schemas from dataclasses
    schemas = {}
    for cls in [ExecutionContext, SDKEvent, SDKSession, SDKResponse, SDKRequest]:
        schemas[cls.__name__] = _dataclass_to_schema(cls, enums_collected)

    # Add request/response schemas for endpoints
    schemas["ChatExecuteRequest"] = {
        "type": "object",
        "properties": {
            "message": {"type": "string", "description": "User message"},
            "messages": {
                "type": "array",
                "description": "Chat messages array (useChat format)",
                "items": {"$ref": "#/components/schemas/ChatMessage"},
            },
            "history": {
                "type": "array",
                "description": "Conversation history",
                "items": {"$ref": "#/components/schemas/ChatMessage"},
            },
            "session_id": {"type": "string", "nullable": True},
            "context": {"$ref": "#/components/schemas/ExecutionContext"},
        },
    }

    schemas["ChatMessage"] = {
        "type": "object",
        "properties": {
            "role": {"type": "string", "enum": ["user", "assistant", "system"]},
            "content": {"type": "string"},
            "timestamp": {"type": "string", "format": "date-time", "nullable": True},
        },
        "required": ["role", "content"],
    }

    schemas["WorkflowExecuteRequest"] = {
        "type": "object",
        "properties": {
            "goal": {"type": "string", "description": "Workflow goal"},
            "context": {"type": "object", "additionalProperties": True},
            "session_id": {"type": "string", "nullable": True},
        },
        "required": ["goal"],
    }

    schemas["HealthResponse"] = {
        "type": "object",
        "properties": {
            "status": {"type": "string", "example": "healthy"},
            "service": {"type": "string", "example": "jotty"},
            "version": {"type": "string"},
            "skills_loaded": {"type": "integer"},
            "lm_configured": {"type": "boolean"},
        },
        "required": ["status", "service"],
    }

    schemas["ErrorResponse"] = {
        "type": "object",
        "properties": {
            "success": {"type": "boolean", "example": False},
            "error": {"type": "string"},
            "error_code": {"type": "string", "nullable": True},
        },
        "required": ["success", "error"],
    }

    schemas["SkillInfo"] = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "description": {"type": "string"},
            "skill_type": {"type": "string", "enum": ["base", "derived", "composite"]},
            "capabilities": {"type": "array", "items": {"type": "string"}},
            "base_skills": {"type": "array", "items": {"type": "string"}},
            "tools": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["name"],
    }

    # Define endpoints declaratively
    def _ref(name: Any) -> Dict:
        return {"$ref": f"#/components/schemas/{name}"}

    def _json(schema: Any) -> Dict:
        return {"application/json": {"schema": schema}}

    def _sse() -> Dict:
        return {"text/event-stream": {"schema": {"type": "string", "format": "binary"}}}

    def _responses(ok_schema: Any, ok_desc: Any = "Success") -> Any:
        r = {"200": {"description": ok_desc, "content": _json(ok_schema)}}
        r["400"] = {"description": "Bad request", "content": _json(_ref("ErrorResponse"))}
        r["500"] = {"description": "Internal server error", "content": _json(_ref("ErrorResponse"))}
        return r

    def _auth() -> List:
        return [{"BearerAuth": []}]

    paths = {
        "/health": {
            "get": {
                "tags": ["Health"],
                "summary": "Health check",
                "operationId": "healthCheck",
                "responses": {
                    "200": {"description": "Healthy", "content": _json(_ref("HealthResponse"))}
                },
            }
        },
        "/api/chat": {
            "post": {
                "tags": ["Chat"],
                "summary": "Execute chat",
                "operationId": "chatExecute",
                "requestBody": {"required": True, "content": _json(_ref("ChatExecuteRequest"))},
                "responses": _responses(_ref("SDKResponse"), "Chat response"),
                "security": _auth(),
            }
        },
        "/api/chat/stream": {
            "post": {
                "tags": ["Chat"],
                "summary": "Stream chat response (SSE)",
                "operationId": "chatStream",
                "requestBody": {"required": True, "content": _json(_ref("ChatExecuteRequest"))},
                "responses": {"200": {"description": "SSE stream", "content": _sse()}},
                "security": _auth(),
            }
        },
        "/api/workflow": {
            "post": {
                "tags": ["Workflow"],
                "summary": "Execute workflow",
                "operationId": "workflowExecute",
                "requestBody": {"required": True, "content": _json(_ref("WorkflowExecuteRequest"))},
                "responses": _responses(_ref("SDKResponse"), "Workflow result"),
                "security": _auth(),
            }
        },
        "/api/workflow/stream": {
            "post": {
                "tags": ["Workflow"],
                "summary": "Stream workflow (SSE)",
                "operationId": "workflowStream",
                "requestBody": {"required": True, "content": _json(_ref("WorkflowExecuteRequest"))},
                "responses": {"200": {"description": "SSE stream", "content": _sse()}},
                "security": _auth(),
            }
        },
        "/api/skills": {
            "get": {
                "tags": ["Skills"],
                "summary": "List skills",
                "operationId": "listSkills",
                "responses": {
                    "200": {
                        "description": "Skills list",
                        "content": _json(
                            {
                                "type": "object",
                                "properties": {
                                    "success": {"type": "boolean"},
                                    "skills": {"type": "array", "items": _ref("SkillInfo")},
                                    "count": {"type": "integer"},
                                },
                            }
                        ),
                    }
                },
                "security": _auth(),
            }
        },
        "/api/skill/{name}": {
            "post": {
                "tags": ["Skills"],
                "summary": "Execute skill",
                "operationId": "executeSkill",
                "parameters": [
                    {"name": "name", "in": "path", "required": True, "schema": {"type": "string"}}
                ],
                "requestBody": {
                    "required": True,
                    "content": _json({"type": "object", "additionalProperties": True}),
                },
                "responses": _responses(_ref("SDKResponse"), "Skill result"),
                "security": _auth(),
            },
            "get": {
                "tags": ["Skills"],
                "summary": "Get skill info",
                "operationId": "getSkillInfo",
                "parameters": [
                    {"name": "name", "in": "path", "required": True, "schema": {"type": "string"}}
                ],
                "responses": {
                    "200": {"description": "Skill info", "content": _json(_ref("SkillInfo"))}
                },
                "security": _auth(),
            },
        },
        "/api/agent/{name}": {
            "post": {
                "tags": ["Agents"],
                "summary": "Execute with agent",
                "operationId": "executeAgent",
                "parameters": [
                    {"name": "name", "in": "path", "required": True, "schema": {"type": "string"}}
                ],
                "requestBody": {
                    "required": True,
                    "content": _json(
                        {
                            "type": "object",
                            "properties": {
                                "task": {"type": "string"},
                                "context": {"type": "object", "additionalProperties": True},
                            },
                            "required": ["task"],
                        }
                    ),
                },
                "responses": _responses(_ref("SDKResponse"), "Agent result"),
                "security": _auth(),
            }
        },
        "/api/session/{user_id}": {
            "get": {
                "tags": ["Sessions"],
                "summary": "Get session",
                "operationId": "getSession",
                "parameters": [
                    {
                        "name": "user_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                    }
                ],
                "responses": {
                    "200": {"description": "Session data", "content": _json(_ref("SDKSession"))}
                },
                "security": _auth(),
            },
            "put": {
                "tags": ["Sessions"],
                "summary": "Update session",
                "operationId": "updateSession",
                "parameters": [
                    {
                        "name": "user_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                    }
                ],
                "requestBody": {"required": True, "content": _json(_ref("SDKSession"))},
                "responses": {
                    "200": {"description": "Updated", "content": _json(_ref("SDKSession"))}
                },
                "security": _auth(),
            },
            "delete": {
                "tags": ["Sessions"],
                "summary": "Delete session",
                "operationId": "deleteSession",
                "parameters": [
                    {
                        "name": "user_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                    }
                ],
                "responses": {"204": {"description": "Deleted"}},
                "security": _auth(),
            },
        },
    }

    spec = {
        "openapi": "3.0.3",
        "info": {
            "title": title,
            "version": version,
            "description": description,
            "license": {"name": "MIT", "url": "https://opensource.org/licenses/MIT"},
        },
        "servers": [{"url": base_url, "description": "Default server"}],
        "tags": [
            {"name": "Health", "description": "Health check"},
            {"name": "Chat", "description": "Chat interactions"},
            {"name": "Workflow", "description": "Multi-step workflow execution"},
            {"name": "Skills", "description": "Skill discovery and execution"},
            {"name": "Agents", "description": "Agent management"},
            {"name": "Sessions", "description": "Session management"},
        ],
        "paths": paths,
        "components": {
            "securitySchemes": {
                "BearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "bearerFormat": "JWT",
                }
            },
            "schemas": schemas,
        },
    }

    return spec


def save_openapi_spec(spec: Dict[str, Any], output_path: Path) -> None:
    """Save OpenAPI specification to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(spec, f, indent=2)


if __name__ == "__main__":
    import sys

    output_file = Path("sdk/openapi.json")
    if len(sys.argv) > 1:
        output_file = Path(sys.argv[1])

    spec = generate_openapi_spec()
    save_openapi_spec(spec, output_file)

    logger.info(f"OpenAPI spec generated: {output_file}")
    logger.info(f"  Version: {spec['info']['version']}")
    logger.info(f"  Endpoints: {len(spec['paths'])}")
    logger.info(f"  Schemas: {len(spec['components']['schemas'])}")
