"""
OpenAPI 3.0 Specification Generator

Auto-generates OpenAPI spec from Jotty's API endpoints.

Usage:
    from Jotty.core.api.openapi_generator import generate_openapi_spec
    
    spec = generate_openapi_spec()
    with open('openapi.yaml', 'w') as f:
        f.write(spec)
"""
from typing import Dict, Any, List
import yaml


def generate_openapi_spec() -> str:
    """
    Generate OpenAPI 3.0 specification for Jotty API.
    
    Returns:
        OpenAPI spec as YAML string
    """
    spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "Jotty AI Agent Framework API",
            "version": "1.0.0",
            "description": "REST API for Jotty - AI agent framework with 164 skills, multi-level memory, and swarm intelligence.",
            "contact": {
                "name": "Jotty Support",
                "url": "https://github.com/your-repo/jotty"
            }
        },
        "servers": [
            {"url": "http://localhost:8766", "description": "Local development"},
            {"url": "https://api.jotty.ai", "description": "Production"}
        ],
        "paths": {
            "/health": {
                "get": {
                    "summary": "Health Check",
                    "description": "Get system health status",
                    "responses": {
                        "200": {
                            "description": "System health status",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/HealthStatus"}
                                }
                            }
                        }
                    }
                }
            },
            "/metrics": {
                "get": {
                    "summary": "Prometheus Metrics",
                    "description": "Get Prometheus metrics",
                    "responses": {
                        "200": {
                            "description": "Metrics in Prometheus format",
                            "content": {"text/plain": {}}
                        }
                    }
                }
            },
            "/skills": {
                "get": {
                    "summary": "List Skills",
                    "description": "List all available skills",
                    "responses": {
                        "200": {
                            "description": "List of skills",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/SkillList"}
                                }
                            }
                        }
                    }
                }
            },
            "/skills/{skill_name}/execute": {
                "post": {
                    "summary": "Execute Skill",
                    "description": "Execute a skill with parameters",
                    "parameters": [
                        {
                            "name": "skill_name",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"}
                        }
                    ],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/SkillExecutionRequest"}
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Skill execution result",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/SkillExecutionResponse"}
                                }
                            }
                        },
                        "404": {"description": "Skill not found"},
                        "429": {"description": "Rate limit exceeded"}
                    }
                }
            },
            "/agents/{agent_name}/execute": {
                "post": {
                    "summary": "Execute Agent",
                    "description": "Execute an agent with a task",
                    "parameters": [
                        {
                            "name": "agent_name",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"}
                        }
                    ],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/AgentExecutionRequest"}
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Agent execution result",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/AgentExecutionResponse"}
                                }
                            }
                        }
                    }
                }
            },
            "/memory/store": {
                "post": {
                    "summary": "Store Memory",
                    "description": "Store content in memory system",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/MemoryStoreRequest"}
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Memory stored",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/MemoryStoreResponse"}
                                }
                            }
                        }
                    }
                }
            },
            "/memory/retrieve": {
                "post": {
                    "summary": "Retrieve Memories",
                    "description": "Retrieve memories by query",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/MemoryRetrieveRequest"}
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Retrieved memories",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/MemoryRetrieveResponse"}
                                }
                            }
                        }
                    }
                }
            }
        },
        "components": {
            "schemas": {
                "HealthStatus": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "enum": ["healthy", "degraded", "unhealthy"]},
                        "checks": {"type": "array", "items": {"$ref": "#/components/schemas/HealthCheck"}},
                        "timestamp": {"type": "number"}
                    }
                },
                "HealthCheck": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "status": {"type": "string"},
                        "message": {"type": "string"},
                        "duration_ms": {"type": "number"}
                    }
                },
                "SkillList": {
                    "type": "object",
                    "properties": {
                        "skills": {"type": "array", "items": {"$ref": "#/components/schemas/Skill"}},
                        "count": {"type": "integer"}
                    }
                },
                "Skill": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "category": {"type": "string"},
                        "version": {"type": "string"}
                    }
                },
                "SkillExecutionRequest": {
                    "type": "object",
                    "properties": {
                        "params": {"type": "object"}
                    },
                    "required": ["params"]
                },
                "SkillExecutionResponse": {
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean"},
                        "result": {"type": "object"},
                        "error": {"type": "string"}
                    }
                },
                "AgentExecutionRequest": {
                    "type": "object",
                    "properties": {
                        "task": {"type": "string"},
                        "context": {"type": "object"}
                    },
                    "required": ["task"]
                },
                "AgentExecutionResponse": {
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean"},
                        "result": {"type": "object"}
                    }
                },
                "MemoryStoreRequest": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "level": {"type": "string", "enum": ["episodic", "semantic", "procedural", "meta", "causal"]},
                        "goal": {"type": "string"},
                        "metadata": {"type": "object"}
                    },
                    "required": ["content", "level", "goal"]
                },
                "MemoryStoreResponse": {
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "string"}
                    }
                },
                "MemoryRetrieveRequest": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "goal": {"type": "string"},
                        "top_k": {"type": "integer", "default": 5}
                    },
                    "required": ["query", "goal"]
                },
                "MemoryRetrieveResponse": {
                    "type": "object",
                    "properties": {
                        "memories": {"type": "array", "items": {"$ref": "#/components/schemas/Memory"}}
                    }
                },
                "Memory": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "level": {"type": "string"},
                        "relevance": {"type": "number"}
                    }
                }
            },
            "securitySchemes": {
                "ApiKeyAuth": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "X-API-Key"
                }
            }
        },
        "security": [{"ApiKeyAuth": []}]
    }

    return yaml.dump(spec, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    spec = generate_openapi_spec()
    print(spec)
    
    # Save to file
    with open('openapi.yaml', 'w') as f:
        f.write(spec)
    
    print("\n✅ OpenAPI spec generated: openapi.yaml")
