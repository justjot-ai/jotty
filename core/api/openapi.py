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
