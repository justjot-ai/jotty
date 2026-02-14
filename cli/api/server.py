"""
Jotty REST API Server
=====================

HTTP API for n8n integration and external automation.
Delegates to ModeRouter for actual execution (same path as SDK/Gateway).

Endpoints:
    /api/health    - Health check
    /api/run       - Execute natural language tasks (chat/workflow)
    /api/command   - Execute slash commands (CLI-specific)
    /api/skills    - List available skills
    /api/skills/{name} - Execute a specific skill
    /api/tasks/{id}    - Get async task result
    /api/commands      - List available CLI commands
    /docs          - OpenAPI interactive docs (Swagger UI)
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import FastAPI
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None
    BaseModel = object


class TaskRequest(BaseModel if FASTAPI_AVAILABLE else object):
    """Request model for task execution."""
    task: str
    options: Optional[Dict[str, Any]] = None
    async_mode: bool = False


class CommandRequest(BaseModel if FASTAPI_AVAILABLE else object):
    """Request model for command execution."""
    command: str
    args: Optional[str] = ""


class JottyAPIServer:
    """
    REST API Server for Jotty.

    Delegates to ModeRouter for all execution (same path as SDK and Gateway).
    Keeps a JottyCLI instance only for slash-command execution.

    For n8n integration, use webhook nodes to call these endpoints.
    """

    def __init__(self, host: str = '0.0.0.0', port: int = 8765) -> None:
        self.host = host
        self.port = port
        self._cli = None
        self._router = None
        self._app = None
        self._task_results: Dict[str, Any] = {}

    def _get_cli(self) -> Any:
        """Lazy-load JottyCLI (only needed for /api/command)."""
        if self._cli is None:
            from ..app import JottyCLI
            self._cli = JottyCLI()
        return self._cli

    def _get_mode_router(self) -> Any:
        """Lazy-load ModeRouter (used for all task/skill execution)."""
        if self._router is None:
            # Use absolute imports — works both when run as Jotty.cli.api.server
            # and when run standalone (background process with sys.path insert)
            from Jotty.core.api.mode_router import get_mode_router
            self._router = get_mode_router()
        return self._router

    def _make_context(self, mode: str = 'chat', **kwargs: Any) -> Any:
        """Create an ExecutionContext for ModeRouter calls."""
        from Jotty.core.foundation.types.sdk_types import (
            ExecutionMode, ChannelType, ExecutionContext, ResponseFormat,
        )
        mode_map = {
            "chat": ExecutionMode.CHAT,
            "workflow": ExecutionMode.WORKFLOW,
            "skill": ExecutionMode.SKILL,
            "agent": ExecutionMode.AGENT,
        }
        return ExecutionContext(
            mode=mode_map.get(mode, ExecutionMode.CHAT),
            channel=ChannelType.HTTP,
            response_format=ResponseFormat.MARKDOWN,
            **kwargs,
        )

    def create_app(self) -> "FastAPI":
        """Create FastAPI application."""
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI not installed. Run: pip install fastapi uvicorn")

        app = FastAPI(
            title="Jotty API",
            description="REST API for Jotty Multi-Agent AI Assistant",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc",
        )

        # CORS for n8n and external integrations
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # =====================================================================
        # Health
        # =====================================================================

        @app.get("/api/health")
        async def health() -> Any:
            return {"status": "healthy", "service": "jotty-api"}

        # =====================================================================
        # Task execution (via ModeRouter)
        # =====================================================================

        @app.post("/api/run")
        async def run_task(request: TaskRequest, background_tasks: BackgroundTasks) -> Any:
            """Execute a natural language task via ModeRouter."""
            if request.async_mode:
                task_id = f"task_{len(self._task_results)}"
                background_tasks.add_task(
                    self._execute_task_async, task_id, request.task, request.options
                )
                return {"task_id": task_id, "status": "processing"}

            router = self._get_mode_router()
            context = self._make_context("workflow")
            result = await router.route(request.task, context)

            return {
                "success": result.success,
                "result": result.content,
                "skills_used": result.skills_used,
                "steps_executed": result.steps_executed,
                "execution_time": result.execution_time,
                "error": result.error,
            }

        # =====================================================================
        # Slash commands (CLI-specific, kept for backward compat)
        # =====================================================================

        @app.post("/api/command")
        async def run_command(request: CommandRequest) -> Any:
            """Execute a slash command (delegates to CLI)."""
            cli = self._get_cli()
            full_cmd = f"/{request.command} {request.args}".strip()
            result = await cli.run_once(full_cmd)
            return {"success": True, "command": request.command, "result": str(result)}

        # =====================================================================
        # Skills (via ModeRouter)
        # =====================================================================

        @app.get("/api/skills")
        async def list_skills() -> Any:
            """List available skills via registry."""
            router = self._get_mode_router()
            router._ensure_initialized()
            registry = router._registry

            if registry is None:
                return {"skills": [], "count": 0}

            skills = []
            for skill_info in registry.list_skills():
                skills.append({
                    "name": skill_info.get("name", ""),
                    "category": skill_info.get("category", "general"),
                    "description": (skill_info.get("description", "") or "")[:100],
                })
            return {"skills": skills, "count": len(skills)}

        @app.post("/api/skills/{skill_name}")
        async def execute_skill(skill_name: str, params: Dict[str, Any] = {}) -> Any:
            """Execute a specific skill via ModeRouter."""
            from Jotty.core.foundation.types.sdk_types import SDKRequest, ExecutionMode

            router = self._get_mode_router()
            context = self._make_context("skill")
            request = SDKRequest(
                content=json.dumps(params),
                mode=ExecutionMode.SKILL,
                skill_name=skill_name,
            )
            result = await router.route(request, context)

            return {
                "success": result.success,
                "skill": skill_name,
                "result": result.content,
                "error": result.error,
            }

        # =====================================================================
        # Async task results
        # =====================================================================

        @app.get("/api/tasks/{task_id}")
        async def get_task_result(task_id: str) -> Any:
            """Get result of async task."""
            if task_id not in self._task_results:
                raise HTTPException(status_code=404, detail="Task not found")
            return self._task_results[task_id]

        # =====================================================================
        # Commands listing (CLI metadata)
        # =====================================================================

        @app.get("/api/commands")
        async def list_commands() -> Any:
            """List available CLI commands."""
            cli = self._get_cli()
            commands = []
            for name, cmd in cli.command_registry._commands.items():
                if name == cmd.name:  # Skip aliases
                    commands.append({
                        "name": cmd.name,
                        "aliases": cmd.aliases,
                        "description": cmd.description,
                        "usage": cmd.usage,
                        "category": cmd.category,
                    })
            return {"commands": commands, "count": len(commands)}

        self._app = app
        return app

    async def _execute_task_async(self, task_id: str, task: str, options: dict) -> Any:
        """Execute task asynchronously via ModeRouter."""
        try:
            router = self._get_mode_router()
            context = self._make_context("workflow")
            result = await router.route(task, context)

            self._task_results[task_id] = {
                "status": "completed",
                "success": result.success,
                "result": result.content,
                "skills_used": result.skills_used,
                "execution_time": result.execution_time,
                "error": result.error,
            }
        except Exception as e:
            self._task_results[task_id] = {
                "status": "failed",
                "error": str(e),
            }

    def run(self) -> Any:
        """Start the API server."""
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI not installed. Run: pip install fastapi uvicorn")

        app = self.create_app()
        logger.info(f"Starting Jotty API server on {self.host}:{self.port}")
        uvicorn.run(app, host=self.host, port=self.port)

    async def run_async(self) -> Any:
        """Start the API server asynchronously."""
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI not installed")

        app = self.create_app()
        config = uvicorn.Config(app, host=self.host, port=self.port)
        server = uvicorn.Server(config)
        await server.serve()


def start_api_server(host: str = '0.0.0.0', port: int = 8765) -> Any:
    """Start the Jotty API server."""
    server = JottyAPIServer(host, port)
    server.run()


if __name__ == "__main__":
    start_api_server()
