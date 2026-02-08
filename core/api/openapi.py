"""
OpenAPI Specification Generator for Jotty

Generates OpenAPI 3.0 specification from Flask server routes.
This spec is then used to generate SDKs for multiple languages.
"""

import json
import inspect
from typing import Dict, Any, List, Optional
from pathlib import Path


def generate_openapi_spec(
    title: str = "Jotty API",
    version: str = "1.0.0",
    description: str = "Multi-agent orchestration framework API",
    base_url: str = "http://localhost:8080"
) -> Dict[str, Any]:
    """
    Generate OpenAPI 3.0 specification for Jotty API.
    
    This spec defines all endpoints, request/response schemas, and authentication.
    SDK generators use this spec to create client libraries.
    """
    
    spec = {
        "openapi": "3.0.3",
        "info": {
            "title": title,
            "version": version,
            "description": description,
            "contact": {
                "name": "Jotty Support",
                "url": "https://github.com/your-org/jotty"
            },
            "license": {
                "name": "MIT",
                "url": "https://opensource.org/licenses/MIT"
            }
        },
        "servers": [
            {
                "url": base_url,
                "description": "Default server"
            }
        ],
        "tags": [
            {
                "name": "Chat",
                "description": "Chat interaction endpoints"
            },
            {
                "name": "Workflow",
                "description": "Workflow execution endpoints"
            },
            {
                "name": "Agents",
                "description": "Agent management endpoints"
            },
            {
                "name": "Skills",
                "description": "Skill discovery and execution endpoints"
            },
            {
                "name": "Sessions",
                "description": "Session management endpoints"
            },
            {
                "name": "Health",
                "description": "Health check endpoints"
            }
        ],
        "paths": {
            "/api/health": {
                "get": {
                    "tags": ["Health"],
                    "summary": "Health check",
                    "description": "Check if the server is running and healthy",
                    "operationId": "healthCheck",
                    "responses": {
                        "200": {
                            "description": "Server is healthy",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/HealthResponse"
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/api/chat/execute": {
                "post": {
                    "tags": ["Chat"],
                    "summary": "Execute chat synchronously",
                    "description": "Execute a chat interaction and wait for the complete response",
                    "operationId": "chatExecute",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/ChatExecuteRequest"
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Chat response",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/ChatExecuteResponse"
                                    }
                                }
                            }
                        },
                        "400": {
                            "description": "Bad request",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/ErrorResponse"
                                    }
                                }
                            }
                        },
                        "500": {
                            "description": "Internal server error",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/ErrorResponse"
                                    }
                                }
                            }
                        }
                    },
                    "security": [
                        {
                            "BearerAuth": []
                        }
                    ]
                }
            },
            "/api/chat/stream": {
                "post": {
                    "tags": ["Chat"],
                    "summary": "Stream chat response",
                    "description": "Execute a chat interaction with Server-Sent Events (SSE) streaming",
                    "operationId": "chatStream",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/ChatStreamRequest"
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Streaming chat response",
                            "content": {
                                "text/event-stream": {
                                    "schema": {
                                        "type": "string",
                                        "format": "binary"
                                    }
                                }
                            }
                        },
                        "400": {
                            "description": "Bad request",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/ErrorResponse"
                                    }
                                }
                            }
                        },
                        "500": {
                            "description": "Internal server error",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/ErrorResponse"
                                    }
                                }
                            }
                        }
                    },
                    "security": [
                        {
                            "BearerAuth": []
                        }
                    ]
                }
            },
            "/api/workflow/execute": {
                "post": {
                    "tags": ["Workflow"],
                    "summary": "Execute workflow synchronously",
                    "description": "Execute a workflow and wait for completion",
                    "operationId": "workflowExecute",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/WorkflowExecuteRequest"
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Workflow result",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/WorkflowExecuteResponse"
                                    }
                                }
                            }
                        },
                        "400": {
                            "description": "Bad request",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/ErrorResponse"
                                    }
                                }
                            }
                        },
                        "500": {
                            "description": "Internal server error",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/ErrorResponse"
                                    }
                                }
                            }
                        }
                    },
                    "security": [
                        {
                            "BearerAuth": []
                        }
                    ]
                }
            },
            "/api/workflow/stream": {
                "post": {
                    "tags": ["Workflow"],
                    "summary": "Stream workflow execution",
                    "description": "Execute a workflow with Server-Sent Events (SSE) streaming",
                    "operationId": "workflowStream",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/WorkflowStreamRequest"
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Streaming workflow events",
                            "content": {
                                "text/event-stream": {
                                    "schema": {
                                        "type": "string",
                                        "format": "binary"
                                    }
                                }
                            }
                        },
                        "400": {
                            "description": "Bad request",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/ErrorResponse"
                                    }
                                }
                            }
                        },
                        "500": {
                            "description": "Internal server error",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/ErrorResponse"
                                    }
                                }
                            }
                        }
                    },
                    "security": [
                        {
                            "BearerAuth": []
                        }
                    ]
                }
            },
            "/api/agents": {
                "get": {
                    "tags": ["Agents"],
                    "summary": "List available agents",
                    "description": "Get a list of all available agents",
                    "operationId": "listAgents",
                    "responses": {
                        "200": {
                            "description": "List of agents",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/AgentsResponse"
                                    }
                                }
                            }
                        }
                    },
                    "security": [
                        {
                            "BearerAuth": []
                        }
                    ]
                }
            },
            "/api/agent/{name}": {
                "post": {
                    "tags": ["Agents"],
                    "summary": "Execute task with specific agent",
                    "description": "Execute a task using a specific agent",
                    "operationId": "executeAgent",
                    "parameters": [
                        {
                            "name": "name",
                            "in": "path",
                            "required": True,
                            "schema": {
                                "type": "string"
                            },
                            "description": "Agent name"
                        }
                    ],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/AgentExecuteRequest"
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Agent execution result",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/SDKResponse"
                                    }
                                }
                            }
                        }
                    },
                    "security": [{"BearerAuth": []}]
                }
            },
            "/api/skills": {
                "get": {
                    "tags": ["Skills"],
                    "summary": "List available skills",
                    "description": "Get a list of all available skills",
                    "operationId": "listSkills",
                    "responses": {
                        "200": {
                            "description": "List of skills",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/SkillsResponse"
                                    }
                                }
                            }
                        }
                    },
                    "security": [{"BearerAuth": []}]
                }
            },
            "/api/skill/{name}": {
                "post": {
                    "tags": ["Skills"],
                    "summary": "Execute a skill",
                    "description": "Execute a specific skill with parameters",
                    "operationId": "executeSkill",
                    "parameters": [
                        {
                            "name": "name",
                            "in": "path",
                            "required": True,
                            "schema": {
                                "type": "string"
                            },
                            "description": "Skill name"
                        }
                    ],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "additionalProperties": True
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Skill execution result",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/SDKResponse"
                                    }
                                }
                            }
                        }
                    },
                    "security": [{"BearerAuth": []}]
                },
                "get": {
                    "tags": ["Skills"],
                    "summary": "Get skill info",
                    "description": "Get information about a specific skill",
                    "operationId": "getSkillInfo",
                    "parameters": [
                        {
                            "name": "name",
                            "in": "path",
                            "required": True,
                            "schema": {
                                "type": "string"
                            },
                            "description": "Skill name"
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Skill information",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/SkillInfo"
                                    }
                                }
                            }
                        }
                    },
                    "security": [{"BearerAuth": []}]
                }
            },
            "/api/session/{user_id}": {
                "get": {
                    "tags": ["Sessions"],
                    "summary": "Get user session",
                    "description": "Get session data for a user",
                    "operationId": "getSession",
                    "parameters": [
                        {
                            "name": "user_id",
                            "in": "path",
                            "required": True,
                            "schema": {
                                "type": "string"
                            },
                            "description": "User identifier"
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Session data",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/SDKSession"
                                    }
                                }
                            }
                        },
                        "404": {
                            "description": "Session not found",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/ErrorResponse"
                                    }
                                }
                            }
                        }
                    },
                    "security": [{"BearerAuth": []}]
                },
                "put": {
                    "tags": ["Sessions"],
                    "summary": "Update user session",
                    "description": "Update or create session data for a user",
                    "operationId": "updateSession",
                    "parameters": [
                        {
                            "name": "user_id",
                            "in": "path",
                            "required": True,
                            "schema": {
                                "type": "string"
                            },
                            "description": "User identifier"
                        }
                    ],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": "#/components/schemas/SDKSession"
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Session updated",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "$ref": "#/components/schemas/SDKSession"
                                    }
                                }
                            }
                        }
                    },
                    "security": [{"BearerAuth": []}]
                },
                "delete": {
                    "tags": ["Sessions"],
                    "summary": "Delete user session",
                    "description": "Delete session data for a user",
                    "operationId": "deleteSession",
                    "parameters": [
                        {
                            "name": "user_id",
                            "in": "path",
                            "required": True,
                            "schema": {
                                "type": "string"
                            },
                            "description": "User identifier"
                        }
                    ],
                    "responses": {
                        "204": {
                            "description": "Session deleted"
                        }
                    },
                    "security": [{"BearerAuth": []}]
                }
            }
        },
        "components": {
            "securitySchemes": {
                "BearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "bearerFormat": "JWT",
                    "description": "Bearer token authentication. Include 'Bearer {token}' in Authorization header."
                }
            },
            "schemas": {
                "HealthResponse": {
                    "type": "object",
                    "properties": {
                        "status": {
                            "type": "string",
                            "example": "ok"
                        },
                        "service": {
                            "type": "string",
                            "example": "jotty"
                        },
                        "version": {
                            "type": "string",
                            "example": "1.0.0"
                        },
                        "agents": {
                            "type": "integer",
                            "example": 5
                        }
                    },
                    "required": ["status", "service", "version", "agents"]
                },
                "ChatMessage": {
                    "type": "object",
                    "properties": {
                        "role": {
                            "type": "string",
                            "enum": ["user", "assistant", "system"],
                            "example": "user"
                        },
                        "content": {
                            "oneOf": [
                                {
                                    "type": "string"
                                },
                                {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "type": {
                                                "type": "string"
                                            },
                                            "text": {
                                                "type": "string"
                                            }
                                        }
                                    }
                                }
                            ],
                            "example": "Hello, how can you help me?"
                        },
                        "timestamp": {
                            "type": "string",
                            "format": "date-time",
                            "nullable": True
                        }
                    },
                    "required": ["role", "content"]
                },
                "ChatExecuteRequest": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "User message (legacy format)",
                            "example": "What is the weather today?"
                        },
                        "messages": {
                            "type": "array",
                            "description": "Chat messages array (useChat format)",
                            "items": {
                                "$ref": "#/components/schemas/ChatMessage"
                            }
                        },
                        "history": {
                            "type": "array",
                            "description": "Conversation history (legacy format)",
                            "items": {
                                "$ref": "#/components/schemas/ChatMessage"
                            }
                        },
                        "agentId": {
                            "type": "string",
                            "description": "Specific agent ID to use",
                            "nullable": True
                        },
                        "provider": {
                            "type": "string",
                            "description": "LLM provider (opencode, claude-cli, etc.)",
                            "nullable": True
                        },
                        "model": {
                            "type": "string",
                            "description": "Model name",
                            "nullable": True
                        }
                    }
                },
                "ChatExecuteResponse": {
                    "type": "object",
                    "properties": {
                        "success": {
                            "type": "boolean",
                            "example": True
                        },
                        "final_output": {
                            "type": "string",
                            "description": "Final response text"
                        },
                        "agent_id": {
                            "type": "string",
                            "nullable": True
                        },
                        "metadata": {
                            "type": "object",
                            "additionalProperties": True
                        }
                    },
                    "required": ["success"]
                },
                "ChatStreamRequest": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "User message (legacy format)",
                            "example": "What is the weather today?"
                        },
                        "messages": {
                            "type": "array",
                            "description": "Chat messages array (useChat format)",
                            "items": {
                                "$ref": "#/components/schemas/ChatMessage"
                            }
                        },
                        "history": {
                            "type": "array",
                            "description": "Conversation history (legacy format)",
                            "items": {
                                "$ref": "#/components/schemas/ChatMessage"
                            }
                        },
                        "agentId": {
                            "type": "string",
                            "description": "Specific agent ID to use",
                            "nullable": True
                        },
                        "provider": {
                            "type": "string",
                            "description": "LLM provider",
                            "nullable": True
                        },
                        "model": {
                            "type": "string",
                            "description": "Model name",
                            "nullable": True
                        }
                    }
                },
                "WorkflowExecuteRequest": {
                    "type": "object",
                    "properties": {
                        "goal": {
                            "type": "string",
                            "description": "Workflow goal/objective",
                            "example": "Analyze sales data and generate report"
                        },
                        "context": {
                            "type": "object",
                            "description": "Additional context for workflow execution",
                            "additionalProperties": True,
                            "example": {
                                "date_range": "2024-01-01 to 2024-12-31",
                                "department": "sales"
                            }
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["dynamic", "static"],
                            "default": "dynamic",
                            "description": "Orchestration mode"
                        },
                        "agent_order": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "Required for static mode - ordered list of agent IDs",
                            "nullable": True
                        }
                    },
                    "required": ["goal"]
                },
                "WorkflowExecuteResponse": {
                    "type": "object",
                    "properties": {
                        "success": {
                            "type": "boolean",
                            "example": True
                        },
                        "final_output": {
                            "type": "string",
                            "description": "Final workflow result"
                        },
                        "steps": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": True
                            },
                            "description": "Individual workflow steps"
                        },
                        "metadata": {
                            "type": "object",
                            "additionalProperties": True
                        }
                    },
                    "required": ["success"]
                },
                "WorkflowStreamRequest": {
                    "type": "object",
                    "properties": {
                        "goal": {
                            "type": "string",
                            "description": "Workflow goal/objective",
                            "example": "Analyze sales data and generate report"
                        },
                        "context": {
                            "type": "object",
                            "description": "Additional context for workflow execution",
                            "additionalProperties": True
                        }
                    },
                    "required": ["goal"]
                },
                "AgentsResponse": {
                    "type": "object",
                    "properties": {
                        "success": {
                            "type": "boolean",
                            "example": True
                        },
                        "agents": {
                            "type": "array",
                            "items": {
                                "$ref": "#/components/schemas/AgentInfo"
                            }
                        }
                    },
                    "required": ["success", "agents"]
                },
                "AgentInfo": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "example": "research-assistant"
                        },
                        "name": {
                            "type": "string",
                            "example": "Research Assistant"
                        },
                        "description": {
                            "type": "string",
                            "example": "Helps with research tasks"
                        }
                    },
                    "required": ["id", "name"]
                },
                "ErrorResponse": {
                    "type": "object",
                    "properties": {
                        "success": {
                            "type": "boolean",
                            "example": False
                        },
                        "error": {
                            "type": "string",
                            "example": "Invalid request parameters"
                        }
                    },
                    "required": ["success", "error"]
                },
                "SDKResponse": {
                    "type": "object",
                    "description": "Standardized SDK response format",
                    "properties": {
                        "success": {
                            "type": "boolean",
                            "example": True
                        },
                        "content": {
                            "description": "Response content (any type)",
                            "nullable": True
                        },
                        "response_format": {
                            "type": "string",
                            "enum": ["text", "markdown", "json", "html", "a2ui"],
                            "default": "text"
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["chat", "workflow", "agent", "skill", "stream"],
                            "nullable": True
                        },
                        "request_id": {
                            "type": "string",
                            "nullable": True
                        },
                        "execution_time": {
                            "type": "number",
                            "format": "float",
                            "example": 1.25
                        },
                        "timestamp": {
                            "type": "string",
                            "format": "date-time"
                        },
                        "skills_used": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "agents_used": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "steps_executed": {
                            "type": "integer",
                            "example": 3
                        },
                        "error": {
                            "type": "string",
                            "nullable": True
                        },
                        "error_code": {
                            "type": "string",
                            "nullable": True
                        },
                        "metadata": {
                            "type": "object",
                            "additionalProperties": True
                        }
                    },
                    "required": ["success"]
                },
                "SDKSession": {
                    "type": "object",
                    "description": "Persistent session for cross-channel user tracking",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "example": "sess_abc123"
                        },
                        "user_id": {
                            "type": "string",
                            "example": "user_123"
                        },
                        "channels": {
                            "type": "object",
                            "description": "Channel type to channel ID mapping",
                            "additionalProperties": {"type": "string"}
                        },
                        "primary_channel": {
                            "type": "string",
                            "enum": ["cli", "web", "sdk", "telegram", "slack", "discord", "whatsapp", "websocket", "http"],
                            "nullable": True
                        },
                        "messages": {
                            "type": "array",
                            "items": {
                                "$ref": "#/components/schemas/ChatMessage"
                            }
                        },
                        "max_history": {
                            "type": "integer",
                            "default": 50
                        },
                        "user_name": {
                            "type": "string",
                            "nullable": True
                        },
                        "preferences": {
                            "type": "object",
                            "additionalProperties": True
                        },
                        "created_at": {
                            "type": "string",
                            "format": "date-time"
                        },
                        "updated_at": {
                            "type": "string",
                            "format": "date-time"
                        },
                        "last_active": {
                            "type": "string",
                            "format": "date-time"
                        },
                        "metadata": {
                            "type": "object",
                            "additionalProperties": True
                        }
                    },
                    "required": ["session_id", "user_id"]
                },
                "SDKEvent": {
                    "type": "object",
                    "description": "Event emitted during SDK execution",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["start", "complete", "error", "thinking", "planning", "skill_start", "skill_progress", "skill_complete", "stream", "delta", "agent_start", "agent_complete", "memory_recall", "memory_store"]
                        },
                        "data": {
                            "description": "Event-specific data",
                            "nullable": True
                        },
                        "context_id": {
                            "type": "string",
                            "nullable": True
                        },
                        "timestamp": {
                            "type": "string",
                            "format": "date-time"
                        }
                    },
                    "required": ["type"]
                },
                "ExecutionContext": {
                    "type": "object",
                    "description": "Unified context for request execution",
                    "properties": {
                        "mode": {
                            "type": "string",
                            "enum": ["chat", "workflow", "agent", "skill", "stream"]
                        },
                        "channel": {
                            "type": "string",
                            "enum": ["cli", "web", "sdk", "telegram", "slack", "discord", "whatsapp", "websocket", "http", "custom"]
                        },
                        "session_id": {
                            "type": "string"
                        },
                        "user_id": {
                            "type": "string",
                            "nullable": True
                        },
                        "user_name": {
                            "type": "string",
                            "nullable": True
                        },
                        "request_id": {
                            "type": "string"
                        },
                        "timestamp": {
                            "type": "string",
                            "format": "date-time"
                        },
                        "channel_id": {
                            "type": "string",
                            "nullable": True
                        },
                        "message_id": {
                            "type": "string",
                            "nullable": True
                        },
                        "timeout": {
                            "type": "number",
                            "default": 300.0
                        },
                        "max_steps": {
                            "type": "integer",
                            "default": 10
                        },
                        "streaming": {
                            "type": "boolean",
                            "default": False
                        },
                        "response_format": {
                            "type": "string",
                            "enum": ["text", "markdown", "json", "html", "a2ui"],
                            "default": "markdown"
                        },
                        "metadata": {
                            "type": "object",
                            "additionalProperties": True
                        }
                    },
                    "required": ["mode", "channel"]
                },
                "SkillsResponse": {
                    "type": "object",
                    "properties": {
                        "success": {
                            "type": "boolean",
                            "example": True
                        },
                        "skills": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "count": {
                            "type": "integer",
                            "example": 126
                        }
                    },
                    "required": ["success", "skills"]
                },
                "SkillInfo": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "example": "web-search"
                        },
                        "description": {
                            "type": "string",
                            "example": "Search the web for information"
                        },
                        "category": {
                            "type": "string",
                            "example": "input",
                            "nullable": True
                        },
                        "tools": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "parameters": {
                            "type": "object",
                            "additionalProperties": True
                        }
                    },
                    "required": ["name"]
                },
                "AgentExecuteRequest": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "Task for the agent to execute"
                        },
                        "context": {
                            "type": "object",
                            "additionalProperties": True
                        }
                    },
                    "required": ["task"]
                }
            }
        }
    }
    
    return spec


def save_openapi_spec(spec: Dict[str, Any], output_path: Path) -> None:
    """Save OpenAPI specification to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(spec, f, indent=2)
    print(f"✅ OpenAPI spec saved to {output_path}")


if __name__ == "__main__":
    import sys
    
    output_file = Path("sdk/openapi.json")
    if len(sys.argv) > 1:
        output_file = Path(sys.argv[1])
    
    spec = generate_openapi_spec()
    save_openapi_spec(spec, output_file)
    
    print(f"\n📋 OpenAPI specification generated:")
    print(f"   - File: {output_file}")
    print(f"   - Version: {spec['info']['version']}")
    print(f"   - Endpoints: {len(spec['paths'])}")
    print(f"\n💡 Next steps:")
    print(f"   1. Review the spec: cat {output_file}")
    print(f"   2. Generate SDKs: python sdk/generate_sdks.py")
