"""
System routes - health, capabilities, models, agents, providers, commands, features, static.
"""

import asyncio
import json
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)



def register_system_routes(app, api):
    from fastapi import HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File, Form
    from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, HTMLResponse
    from pydantic import BaseModel

    static_dir = Path(__file__).parent / "static"

    @app.get("/")
    async def root():
        """Serve chat UI."""
        index_file = static_dir / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return {"message": "Jotty API", "docs": "/docs"}

    @app.get("/health")
    async def health():
        """Health check."""
        return {"status": "ok", "timestamp": datetime.now().isoformat()}

    # ==========================================================================
    # DRY: Unified Registry APIs - Uses existing registries, no duplication
    # ==========================================================================

    @app.get("/api/widgets")
    async def get_widgets():
        """Get all widgets from unified registry."""
        try:
            from Jotty.core.capabilities.registry.unified_registry import get_unified_registry
            registry = get_unified_registry()
            return registry.get_widgets()
        except ImportError:
            # Fallback if registry not available
            return {"widgets": [], "categories": []}

    @app.get("/api/tools")
    async def get_tools():
        """Get all tools from unified registry."""
        try:
            from Jotty.core.capabilities.registry.unified_registry import get_unified_registry
            registry = get_unified_registry()
            return registry.get_tools()
        except ImportError:
            return {"tools": [], "categories": []}

    @app.get("/api/capabilities")
    async def get_capabilities():
        """Get unified tools + widgets + defaults (DRY single source of truth)."""
        # Fallback widgets and tools when registry is empty
        fallback_widgets = [
            {"value": "markdown", "label": "Markdown", "icon": "📝", "category": "Content"},
            {"value": "code", "label": "Code Block", "icon": "💻", "category": "Content"},
            {"value": "chart", "label": "Chart", "icon": "📊", "category": "Visualization"},
            {"value": "table", "label": "Table", "icon": "📋", "category": "Data"},
            {"value": "image", "label": "Image", "icon": "🖼️", "category": "Media"},
            {"value": "pdf", "label": "PDF Export", "icon": "📄", "category": "Export"},
            {"value": "slides", "label": "Slides", "icon": "🎯", "category": "Export"},
            {"value": "mermaid", "label": "Mermaid Diagram", "icon": "🔀", "category": "Visualization"},
            {"value": "latex", "label": "LaTeX Math", "icon": "∑", "category": "Content"},
            {"value": "json", "label": "JSON Viewer", "icon": "{ }", "category": "Data"},
        ]
        fallback_tools = [
            {"name": "web_search", "description": "Search the web", "category": "Research"},
            {"name": "web_browse", "description": "Browse and read web pages", "category": "Research"},
            {"name": "file_read", "description": "Read local files", "category": "Files"},
            {"name": "file_write", "description": "Write to local files", "category": "Files"},
            {"name": "code_execute", "description": "Execute code snippets", "category": "Development"},
            {"name": "shell", "description": "Run shell commands", "category": "System"},
            {"name": "image_generate", "description": "Generate images with AI", "category": "Media"},
            {"name": "pdf_generate", "description": "Generate PDF documents", "category": "Export"},
        ]

        try:
            from Jotty.core.capabilities.registry.unified_registry import get_unified_registry
            registry = get_unified_registry()
            all_data = registry.get_all()
            defaults = registry.get_enabled_defaults()

            # Use fallbacks if registry returns empty
            widgets = all_data.get("widgets", {}).get("available", [])
            tools = all_data.get("tools", {}).get("available", [])

            if not widgets:
                all_data["widgets"] = {"available": fallback_widgets, "categories": ["Content", "Visualization", "Data", "Media", "Export"]}
            if not tools:
                all_data["tools"] = {"available": fallback_tools, "categories": ["Research", "Files", "Development", "System", "Media", "Export"]}

            return {
                **all_data,
                "defaults": defaults if defaults.get("widgets") or defaults.get("tools") else {
                    "widgets": ["markdown", "code", "chart", "table"],
                    "tools": ["web_search", "web_browse"]
                }
            }
        except (ImportError, Exception):
            return {
                "widgets": {"available": fallback_widgets, "categories": ["Content", "Visualization", "Data", "Media", "Export"]},
                "tools": {"available": fallback_tools, "categories": ["Research", "Files", "Development", "System", "Media", "Export"]},
                "defaults": {"widgets": ["markdown", "code", "chart", "table"], "tools": ["web_search", "web_browse"]}
            }

    @app.get("/api/agents")
    async def list_agents():
        """Get agents from skills registry."""
        try:
            from Jotty.core.capabilities.registry.skills_registry import get_skills_registry
            registry = get_skills_registry()
            if hasattr(registry, 'list_agents_from_skills'):
                agents = registry.list_agents_from_skills()
                # Handle if it's a coroutine
                if hasattr(agents, '__await__'):
                    import asyncio
                    agents = await agents
                return {"agents": agents if agents else [], "count": len(agents) if agents else 0}
            return {"agents": [], "count": 0}
        except (ImportError, Exception) as e:
            # Fallback: Return basic agent list from CLI commands
            return {
                "agents": [
                    {"id": "research", "name": "Research Agent", "description": "Web research and synthesis", "category": "research"},
                    {"id": "code", "name": "Code Agent", "description": "Code analysis and generation", "category": "development"},
                    {"id": "ml", "name": "ML Agent", "description": "Machine learning pipeline", "category": "ml"},
                ],
                "count": 3
            }

    @app.get("/api/providers")
    async def list_providers():
        """Get LM providers status with detailed model info."""
        import os
        import shutil

        providers = {}

        # All available providers with their models
        provider_configs = {
            "anthropic": {
                "name": "Anthropic",
                "icon": "🅰️",
                "env_key": "ANTHROPIC_API_KEY",
                "models": [
                    {"id": "claude-sonnet-4-20250514", "name": "Claude 4 Sonnet", "context": "200K", "vision": True, "recommended": True},
                    {"id": "claude-opus-4-20250514", "name": "Claude 4 Opus", "context": "200K", "vision": True},
                    {"id": "claude-3-5-haiku-20241022", "name": "Claude 3.5 Haiku", "context": "200K", "vision": True, "fast": True},
                ]
            },
            "openai": {
                "name": "OpenAI",
                "icon": "🤖",
                "env_key": "OPENAI_API_KEY",
                "models": [
                    {"id": "gpt-4o", "name": "GPT-4o", "context": "128K", "vision": True, "recommended": True},
                    {"id": "gpt-4-turbo", "name": "GPT-4 Turbo", "context": "128K", "vision": True},
                    {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo", "context": "16K", "fast": True},
                ]
            },
            "google": {
                "name": "Google",
                "icon": "🔷",
                "env_key": "GOOGLE_API_KEY",
                "models": [
                    {"id": "gemini-2.0-flash-exp", "name": "Gemini 2.0 Flash", "context": "1M", "vision": True, "recommended": True},
                    {"id": "gemini-pro", "name": "Gemini Pro", "context": "32K"},
                    {"id": "gemini-pro-vision", "name": "Gemini Pro Vision", "context": "32K", "vision": True},
                ]
            },
            "groq": {
                "name": "Groq",
                "icon": "⚡",
                "env_key": "GROQ_API_KEY",
                "models": [
                    {"id": "llama-3.1-70b-versatile", "name": "Llama 3.1 70B", "context": "128K", "fast": True, "recommended": True},
                    {"id": "llama-3.1-8b-instant", "name": "Llama 3.1 8B", "context": "128K", "fast": True},
                    {"id": "mixtral-8x7b-32768", "name": "Mixtral 8x7B", "context": "32K", "fast": True},
                ]
            },
            "openrouter": {
                "name": "OpenRouter",
                "icon": "🌐",
                "env_key": "OPENROUTER_API_KEY",
                "models": [
                    {"id": "anthropic/claude-3-opus", "name": "Claude 3 Opus (OR)", "context": "200K", "vision": True},
                    {"id": "meta-llama/llama-3.3-70b-instruct:free", "name": "Llama 3.3 70B (Free)", "context": "128K", "free": True, "recommended": True},
                    {"id": "openai/gpt-4", "name": "GPT-4 (OR)", "context": "128K"},
                ]
            },
            "claude-cli": {
                "name": "Claude CLI",
                "icon": "💻",
                "env_key": None,  # Check for binary
                "models": [
                    {"id": "sonnet", "name": "Sonnet (via CLI)", "context": "200K", "vision": True, "local": True, "recommended": True},
                    {"id": "opus", "name": "Opus (via CLI)", "context": "200K", "vision": True, "local": True},
                    {"id": "haiku", "name": "Haiku (via CLI)", "context": "200K", "vision": True, "local": True, "fast": True},
                ]
            },
        }

        for provider_id, config in provider_configs.items():
            if config["env_key"]:
                configured = bool(os.environ.get(config["env_key"]))
            elif provider_id == "claude-cli":
                configured = shutil.which("claude") is not None
            else:
                configured = False

            providers[provider_id] = {
                "name": config["name"],
                "icon": config["icon"],
                "configured": configured,
                "models": config["models"] if configured else []
            }

        # Get current active model
        current_model = None
        try:
            import dspy
            if hasattr(dspy.settings, 'lm') and dspy.settings.lm:
                lm = dspy.settings.lm
                # Get model name from wrapped LM if ContextAwareLM
                if hasattr(lm, '_wrapped'):
                    lm = lm._wrapped
                current_model = getattr(lm, 'model', None) or getattr(lm, 'model_name', None)
        except Exception:
            pass

        return {
            "providers": providers,
            "current_model": current_model,
            "current_provider": None  # Will be inferred from model name
        }

    @app.get("/api/models")
    async def list_models():
        """Get flat list of all available models for model selector."""
        providers_response = await list_providers()
        providers = providers_response.get("providers", {})

        models = []
        for provider_id, provider in providers.items():
            if provider.get("configured"):
                for model in provider.get("models", []):
                    models.append({
                        "id": f"{provider_id}/{model['id']}",
                        "provider": provider_id,
                        "provider_name": provider["name"],
                        "provider_icon": provider["icon"],
                        "model_id": model["id"],
                        "name": model["name"],
                        "context": model.get("context", "N/A"),
                        "vision": model.get("vision", False),
                        "fast": model.get("fast", False),
                        "free": model.get("free", False),
                        "local": model.get("local", False),
                        "recommended": model.get("recommended", False),
                    })

        # Sort: recommended first, then by provider
        models.sort(key=lambda m: (not m["recommended"], m["provider"]))

        return {
            "models": models,
            "current": providers_response.get("current_model"),
            "count": len(models)
        }

    class SetModelRequest(BaseModel):
        provider: str
        model: str

    @app.post("/api/models/set")
    async def set_model(request: SetModelRequest):
        """Set the active LLM model by storing preference (avoids DSPy threading issues)."""
        try:
            # Store model preference - will be used when creating LLM instances
            api._current_provider = request.provider
            api._current_model = request.model

            # Also set environment variable so UnifiedLMProvider picks it up
            import os
            os.environ["JOTTY_LLM_PROVIDER"] = request.provider
            os.environ["JOTTY_LLM_MODEL"] = request.model

            # For vision calls, we use Anthropic SDK directly anyway
            # This preference is for text-only LLM calls

            return {
                "success": True,
                "provider": request.provider,
                "model": request.model,
                "message": f"Model preference set to {request.provider}/{request.model}"
            }
        except Exception as e:
            logger.error(f"Failed to set model: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }

    class SwarmRequest(BaseModel):
        task: str
        mode: str = "auto"  # auto | manual | workflow
        agents: Optional[List[str]] = None
        workflow: Optional[dict] = None
        session_id: Optional[str] = None

    @app.post("/api/agents/swarm")
    async def execute_swarm(request: SwarmRequest):
        """Execute multi-agent swarm."""
        try:
            from Jotty.core.intelligence.orchestration import Orchestrator
            manager = Orchestrator()
            result = await manager.execute(
                task=request.task,
                mode=request.mode,
                agents=request.agents,
                workflow=request.workflow
            )
            return {"success": True, "result": result}
        except ImportError:
            # Fallback: Execute through regular chat with swarm hint
            session_id = request.session_id or str(uuid.uuid4())[:8]
            enhanced_task = f"[Swarm Mode: {request.mode}] {request.task}"
            result = await api.process_message(
                message=enhanced_task,
                session_id=session_id
            )
            return {"success": result.get("success", False), "result": result}
        except Exception as e:
            logger.error(f"Swarm execution error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    # CLI Commands endpoints
    @app.get("/api/features")
    async def get_feature_flags():
        """Get feature flags for UI capabilities."""
        return {
            "features": {
                "mcp_tools": True,
                "artifacts": True,
                "code_interpreter": True,
                "web_search": True,
                "branching": True,
                "voice": True,
                "documents": True,
                "shareable_links": True,
                "temporary_chat": True
            }
        }

    # ===== VOICE ENDPOINTS =====
    # Speech-to-Text, Text-to-Speech, and Voice-to-Voice pipelines

    @app.get("/static/style.css")
    async def get_css():
        css_file = static_dir / "style.css"
        if css_file.exists():
            return FileResponse(css_file, media_type="text/css")
        raise HTTPException(status_code=404, detail="CSS not found")

    @app.get("/static/app.js")
    async def get_js():
        js_file = static_dir / "app.js"
        if js_file.exists():
            return FileResponse(js_file, media_type="application/javascript")
        raise HTTPException(status_code=404, detail="JS not found")

    # Mount remaining static files
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

