"""
Jotty REST API Server
=====================

HTTP API for n8n integration and external automation.
Exposes all Jotty CLI commands as REST endpoints.
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

    Exposes endpoints for:
    - /api/run - Execute natural language tasks
    - /api/command - Execute slash commands
    - /api/research - Research topics
    - /api/ml - Run ML pipelines
    - /api/skills - List/execute skills
    - /api/health - Health check

    For n8n integration, use webhook nodes to call these endpoints.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self.host = host
        self.port = port
        self._cli = None
        self._app = None
        self._task_results: Dict[str, Any] = {}

    def _get_cli(self):
        """Lazy-load JottyCLI."""
        if self._cli is None:
            from ..app import JottyCLI
            self._cli = JottyCLI()
        return self._cli

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

        # CORS for n8n
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Health check
        @app.get("/api/health")
        async def health():
            return {"status": "healthy", "service": "jotty-api"}

        # Execute task (natural language)
        @app.post("/api/run")
        async def run_task(request: TaskRequest, background_tasks: BackgroundTasks):
            """Execute a natural language task."""
            cli = self._get_cli()

            if request.async_mode:
                # Run in background
                task_id = f"task_{len(self._task_results)}"
                background_tasks.add_task(
                    self._execute_task_async, task_id, request.task, request.options
                )
                return {"task_id": task_id, "status": "processing"}

            # Run synchronously
            result = await cli.run_once(request.task)
            return {"success": True, "result": str(result)}

        # Execute command
        @app.post("/api/command")
        async def run_command(request: CommandRequest):
            """Execute a slash command."""
            cli = self._get_cli()
            full_cmd = f"/{request.command} {request.args}".strip()
            result = await cli.run_once(full_cmd)
            return {"success": True, "command": request.command, "result": str(result)}

        # Research endpoint
        @app.post("/api/research")
        async def research(topic: str, deep: bool = False):
            """Research a topic."""
            cli = self._get_cli()
            flags = "--deep" if deep else ""
            result = await cli.run_once(f"/research {topic} {flags}")
            return {"success": True, "topic": topic, "result": str(result)}

        # ML endpoint
        @app.post("/api/ml")
        async def run_ml(dataset: str, target: Optional[str] = None, iterations: int = 2):
            """Run ML pipeline."""
            cli = self._get_cli()
            cmd = f"/ml {dataset}"
            if target:
                cmd += f" --target {target}"
            cmd += f" --iterations {iterations}"
            result = await cli.run_once(cmd)
            return {"success": True, "dataset": dataset, "result": str(result)}

        # List skills
        @app.get("/api/skills")
        async def list_skills():
            """List available skills."""
            cli = self._get_cli()
            registry = cli.get_skills_registry()
            skills = []
            for name, skill in registry._skills.items():
                skills.append({
                    "name": name,
                    "category": getattr(skill, 'category', 'general'),
                    "description": getattr(skill, 'description', '')[:100]
                })
            return {"skills": skills, "count": len(skills)}

        # Execute skill
        @app.post("/api/skills/{skill_name}")
        async def execute_skill(skill_name: str, params: Dict[str, Any] = {}):
            """Execute a specific skill."""
            cli = self._get_cli()
            result = await cli.run_once(f"/tools {skill_name} {json.dumps(params)}")
            return {"success": True, "skill": skill_name, "result": str(result)}

        # Get task result (for async tasks)
        @app.get("/api/tasks/{task_id}")
        async def get_task_result(task_id: str):
            """Get result of async task."""
            if task_id not in self._task_results:
                raise HTTPException(status_code=404, detail="Task not found")
            return self._task_results[task_id]

        # List commands
        @app.get("/api/commands")
        async def list_commands():
            """List available commands."""
            cli = self._get_cli()
            commands = []
            for name, cmd in cli.command_registry._commands.items():
                if name == cmd.name:  # Skip aliases
                    commands.append({
                        "name": cmd.name,
                        "aliases": cmd.aliases,
                        "description": cmd.description,
                        "usage": cmd.usage,
                        "category": cmd.category
                    })
            return {"commands": commands, "count": len(commands)}

        # JustJot integration
        @app.post("/api/justjot")
        async def create_idea(title: str, content: str, tags: list = []):
            """Create idea on JustJot.ai."""
            cli = self._get_cli()
            # Store content temporarily
            cli._output_history = [content]
            result = await cli.run_once(f"/J --tags {','.join(tags)}")
            return {"success": True, "result": str(result)}

        self._app = app
        return app

    async def _execute_task_async(self, task_id: str, task: str, options: dict):
        """Execute task asynchronously."""
        try:
            cli = self._get_cli()
            result = await cli.run_once(task)
            self._task_results[task_id] = {
                "status": "completed",
                "result": str(result)
            }
        except Exception as e:
            self._task_results[task_id] = {
                "status": "failed",
                "error": str(e)
            }

    def run(self):
        """Start the API server."""
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI not installed. Run: pip install fastapi uvicorn")

        app = self.create_app()
        logger.info(f"Starting Jotty API server on {self.host}:{self.port}")
        uvicorn.run(app, host=self.host, port=self.port)

    async def run_async(self):
        """Start the API server asynchronously."""
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI not installed")

        app = self.create_app()
        config = uvicorn.Config(app, host=self.host, port=self.port)
        server = uvicorn.Server(config)
        await server.serve()


def start_api_server(host: str = "0.0.0.0", port: int = 8765):
    """Start the Jotty API server."""
    server = JottyAPIServer(host, port)
    server.run()


if __name__ == "__main__":
    start_api_server()
