"""
Workflow Command
================

/workflow - Manage n8n workflows and schedules
"""

import json
import logging
import asyncio
import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Any, Optional

from .base import BaseCommand, CommandResult, ParsedArgs

if TYPE_CHECKING:
    from ..app import JottyCLI

logger = logging.getLogger(__name__)

# n8n API configuration
N8N_BASE_URL = "http://localhost:5678"
N8N_API_KEY = None  # Set via config or environment

# Local workflow registry file
WORKFLOW_REGISTRY_FILE = Path.home() / ".jotty" / "workflow_registry.json"


def _load_workflow_registry() -> Dict[str, Any]:
    """Load local workflow name-to-ID registry."""
    if WORKFLOW_REGISTRY_FILE.exists():
        return json.loads(WORKFLOW_REGISTRY_FILE.read_text())
    return {"workflows": {}, "aliases": {}}


def _save_workflow_registry(registry: Dict[str, Any]) -> Any:
    """Save workflow registry."""
    WORKFLOW_REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    WORKFLOW_REGISTRY_FILE.write_text(json.dumps(registry, indent=2))


class WorkflowCommand(BaseCommand):
    """
    /workflow - Manage n8n workflows and scheduled tasks.

    Create, list, and manage automated workflows via n8n.
    """

    name = "workflow"
    aliases = ["n8n", "schedule", "automate"]
    description = "Manage n8n workflows and schedules"
    usage = "/workflow [list|create|run|start|stop|status] [args]"
    category = "automation"

    # Workflow templates for common tasks
    WORKFLOW_TEMPLATES = {
        "daily-research": {
            "name": "Daily Research Report",
            "description": "Research a topic daily and save to JustJot",
            "schedule": "0 9 * * *",  # 9 AM daily
            "nodes": ["schedule", "jotty-research", "justjot-save"]
        },
        "ml-monitor": {
            "name": "ML Model Monitor",
            "description": "Run ML pipeline and track metrics",
            "schedule": "0 */6 * * *",  # Every 6 hours
            "nodes": ["schedule", "jotty-ml", "mlflow-log"]
        },
        "news-digest": {
            "name": "News Digest",
            "description": "Research trending topics and create digest",
            "schedule": "0 8,18 * * *",  # 8 AM and 6 PM
            "nodes": ["schedule", "jotty-research", "telegram-send"]
        },
        "backup-sessions": {
            "name": "Session Backup",
            "description": "Export and backup Jotty sessions",
            "schedule": "0 0 * * *",  # Midnight daily
            "nodes": ["schedule", "jotty-export", "file-save"]
        }
    }

    async def execute(self, args: ParsedArgs, cli: "JottyCLI") -> CommandResult:
        """Execute workflow command."""

        # Handle --help
        if args.flags.get("help") or (args.positional and args.positional[0] == "--help"):
            return await self._show_help(cli)

        subcommand = args.positional[0] if args.positional else "list"

        if subcommand == "list":
            return await self._list_workflows(cli)
        elif subcommand == "templates":
            return await self._list_templates(cli)
        elif subcommand == "create":
            template = args.positional[1] if len(args.positional) > 1 else None
            nickname = args.flags.get("name", args.flags.get("as"))
            return await self._create_workflow(cli, template, args.flags, nickname)
        elif subcommand == "run":
            name_or_id = args.positional[1] if len(args.positional) > 1 else None
            return await self._run_workflow(cli, name_or_id)
        elif subcommand == "alias":
            # /workflow alias <nickname> <workflow-id>
            nickname = args.positional[1] if len(args.positional) > 1 else None
            workflow_id = args.positional[2] if len(args.positional) > 2 else None
            return await self._add_alias(cli, nickname, workflow_id)
        elif subcommand == "export":
            return await self._export_workflow(cli, args.flags)
        elif subcommand in ("server", "start"):
            # Handle /workflow server stop
            if len(args.positional) > 1 and args.positional[1] == "stop":
                return await self._stop_api_server(cli)
            return await self._start_api_server(cli, args.flags)
        elif subcommand == "stop":
            return await self._stop_api_server(cli)
        elif subcommand == "status":
            return await self._check_n8n_status(cli)
        else:
            # Try to run as workflow name directly
            # /workflow daily-research  -> runs the workflow named "daily-research"
            return await self._run_workflow(cli, subcommand)

    async def _list_workflows(self, cli: "JottyCLI") -> CommandResult:
        """List n8n workflows."""
        cli.renderer.header("Workflows")

        # Show local registered workflows first
        registry = _load_workflow_registry()
        if registry["workflows"]:
            cli.renderer.print("\n[bold]Your Workflows:[/bold]")
            for name, info in registry["workflows"].items():
                cli.renderer.print(f"  • [cyan]{name}[/cyan] [dim](ID: {info['id']})[/dim]")
            cli.renderer.print("")

        if registry.get("aliases"):
            cli.renderer.print("[bold]Aliases:[/bold]")
            for alias, wf_id in registry["aliases"].items():
                cli.renderer.print(f"  • [cyan]{alias}[/cyan] -> {wf_id}")
            cli.renderer.print("")

        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                headers = {}
                if N8N_API_KEY:
                    headers["X-N8N-API-KEY"] = N8N_API_KEY

                async with session.get(
                    f"{N8N_BASE_URL}/api/v1/workflows",
                    headers=headers
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        workflows = data.get("data", [])

                        if not workflows:
                            cli.renderer.info("No workflows found.")
                            cli.renderer.info("Use /workflow templates to see available templates")
                            return CommandResult.ok()

                        for wf in workflows[:20]:
                            active = "" if wf.get("active") else "○"
                            cli.renderer.print(
                                f"  {active} [cyan]{wf.get('name', 'Unnamed')}[/cyan] "
                                f"[dim](ID: {wf.get('id')})[/dim]"
                            )

                        return CommandResult.ok(data=workflows)
                    else:
                        cli.renderer.warning(f"n8n API returned {resp.status}")
                        cli.renderer.info("Make sure n8n is running and accessible")
                        return await self._list_templates(cli)

        except ImportError:
            cli.renderer.warning("aiohttp not installed. Install with: pip install aiohttp")
            return await self._list_templates(cli)
        except Exception as e:
            cli.renderer.warning(f"Could not connect to n8n: {e}")
            cli.renderer.info("Showing local templates instead:")
            return await self._list_templates(cli)

    async def _list_templates(self, cli: "JottyCLI") -> CommandResult:
        """List available workflow templates."""
        cli.renderer.print("\n[bold]Available Workflow Templates:[/bold]")
        cli.renderer.print("[dim]" + "─" * 50 + "[/dim]")

        for name, template in self.WORKFLOW_TEMPLATES.items():
            cli.renderer.print(f"\n  [cyan]{name}[/cyan]")
            cli.renderer.print(f"    {template['description']}")
            cli.renderer.print(f"    [dim]Schedule: {template['schedule']}[/dim]")

        cli.renderer.print("\n[dim]" + "─" * 50 + "[/dim]")
        cli.renderer.info("Create with: /workflow create <template-name>")
        cli.renderer.info("Or start API: /workflow server")

        return CommandResult.ok()

    async def _create_workflow(
        self,
        cli: "JottyCLI",
        template: Optional[str],
        flags: dict,
        nickname: Optional[str] = None
    ) -> CommandResult:
        """Create a new workflow from template."""

        if not template:
            cli.renderer.error("Template name required")
            cli.renderer.info("Available templates:")
            for name in self.WORKFLOW_TEMPLATES.keys():
                cli.renderer.print(f"  - {name}")
            cli.renderer.info("\nUsage: /workflow create <template> --name <nickname>")
            return CommandResult.fail("Template required")

        if template not in self.WORKFLOW_TEMPLATES:
            cli.renderer.error(f"Unknown template: {template}")
            return CommandResult.fail("Unknown template")

        tmpl = self.WORKFLOW_TEMPLATES[template]
        # Use nickname or template name
        workflow_name = nickname or template
        cli.renderer.info(f"Creating workflow: {workflow_name}")

        # Generate n8n workflow JSON
        workflow_json = self._generate_n8n_workflow(template, tmpl, flags)

        # Try to create in n8n
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                headers = {"Content-Type": "application/json"}
                if N8N_API_KEY:
                    headers["X-N8N-API-KEY"] = N8N_API_KEY

                async with session.post(
                    f"{N8N_BASE_URL}/api/v1/workflows",
                    headers=headers,
                    json=workflow_json
                ) as resp:
                    if resp.status in [200, 201]:
                        data = await resp.json()
                        workflow_id = data.get('id')

                        # Register nickname
                        registry = _load_workflow_registry()
                        registry["workflows"][workflow_name] = {
                            "id": workflow_id,
                            "template": template,
                            "created": str(asyncio.get_running_loop().time())
                        }
                        _save_workflow_registry(registry)

                        cli.renderer.success(f"Workflow '{workflow_name}' created!")
                        cli.renderer.info(f"Run with: /workflow run {workflow_name}")
                        cli.renderer.info(f"View in n8n: {N8N_BASE_URL}/workflow/{workflow_id}")
                        return CommandResult.ok(data=data)
                    else:
                        error = await resp.text()
                        cli.renderer.warning(f"n8n API error: {error}")

        except Exception as e:
            cli.renderer.warning(f"Could not create in n8n: {e}")

        # Save locally as fallback
        output_path = Path.home() / ".jotty" / "workflows" / f"{template}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(workflow_json, indent=2))

        cli.renderer.success(f"Workflow saved to: {output_path}")
        cli.renderer.info("Import manually into n8n via Settings > Import")

        return CommandResult.ok(output=str(output_path))

    def _generate_n8n_workflow(self, name: str, template: dict, flags: dict) -> dict:
        """Generate n8n workflow JSON."""

        # Base workflow structure
        workflow = {
            "name": template["name"],
            "nodes": [],
            "connections": {},
            "active": False,
            "settings": {
                "executionOrder": "v1"
            }
        }

        # Add schedule trigger
        schedule_node = {
            "parameters": {
                "rule": {
                    "interval": [{"field": "cronExpression", "expression": template["schedule"]}]
                }
            },
            "name": "Schedule Trigger",
            "type": "n8n-nodes-base.scheduleTrigger",
            "position": [250, 300],
            "typeVersion": 1
        }
        workflow["nodes"].append(schedule_node)

        # Add HTTP Request node to call Jotty API
        api_host = flags.get("api_host", "localhost")
        api_port = flags.get("api_port", "8765")

        http_node = {
            "parameters": {
                "method": "POST",
                "url": f"http://{api_host}:{api_port}/api/run",
                "sendBody": True,
                "bodyParameters": {
                    "parameters": [
                        {"name": "task", "value": self._get_task_for_template(name)}
                    ]
                },
                "options": {}
            },
            "name": "Jotty API",
            "type": "n8n-nodes-base.httpRequest",
            "position": [450, 300],
            "typeVersion": 3
        }
        workflow["nodes"].append(http_node)

        # Connect nodes
        workflow["connections"] = {
            "Schedule Trigger": {
                "main": [[{"node": "Jotty API", "type": "main", "index": 0}]]
            }
        }

        return workflow

    def _resolve_workflow_id(self, name_or_id: str) -> str:
        """Resolve workflow name/alias to ID."""
        registry = _load_workflow_registry()

        # Check if it's a registered workflow name
        if name_or_id in registry["workflows"]:
            return registry["workflows"][name_or_id]["id"]

        # Check if it's an alias
        if name_or_id in registry.get("aliases", {}):
            return registry["aliases"][name_or_id]

        # Assume it's already an ID
        return name_or_id

    async def _add_alias(self, cli: "JottyCLI", nickname: str, workflow_id: str) -> CommandResult:
        """Add an alias for a workflow."""
        if not nickname or not workflow_id:
            cli.renderer.error("Usage: /workflow alias <nickname> <workflow-id>")
            cli.renderer.info("Example: /workflow alias morning-news abc123")
            return CommandResult.fail("Missing arguments")

        registry = _load_workflow_registry()
        if "aliases" not in registry:
            registry["aliases"] = {}

        registry["aliases"][nickname] = workflow_id
        _save_workflow_registry(registry)

        cli.renderer.success(f"Alias '{nickname}' -> '{workflow_id}' created")
        cli.renderer.info(f"Run with: /workflow run {nickname}")
        cli.renderer.info(f"      or: /workflow {nickname}")

        return CommandResult.ok()

    def _get_task_for_template(self, template_name: str) -> str:
        """Get the task string for a template."""
        tasks = {
            "daily-research": "/research AI trends --deep",
            "ml-monitor": "/ml titanic --iterations 1",
            "news-digest": "/research technology news",
            "backup-sessions": "/export history"
        }
        return tasks.get(template_name, "/help")

    async def _run_workflow(self, cli: "JottyCLI", name_or_id: str) -> CommandResult:
        """Run a workflow by name or ID."""
        if not name_or_id:
            cli.renderer.error("Workflow name or ID required")
            cli.renderer.info("Usage: /workflow run <name>")
            cli.renderer.info("   or: /workflow <name>")

            # Show registered workflows and templates
            cli.renderer.print("\n[bold]Available templates:[/bold]")
            for name in self.WORKFLOW_TEMPLATES.keys():
                cli.renderer.print(f"  • {name}")

            registry = _load_workflow_registry()
            if registry["workflows"]:
                cli.renderer.print("\n[bold]Registered workflows:[/bold]")
                for name, info in registry["workflows"].items():
                    cli.renderer.print(f"  • {name}")
            return CommandResult.fail("Workflow name required")

        # Check if it's a template - run directly without n8n
        if name_or_id in self.WORKFLOW_TEMPLATES:
            return await self._run_template_directly(cli, name_or_id)

        # Resolve name to ID
        workflow_id = self._resolve_workflow_id(name_or_id)
        display_name = name_or_id if workflow_id != name_or_id else workflow_id

        cli.renderer.info(f"Running workflow: {display_name}")

        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                headers = {}
                if N8N_API_KEY:
                    headers["X-N8N-API-KEY"] = N8N_API_KEY

                async with session.post(
                    f"{N8N_BASE_URL}/api/v1/workflows/{workflow_id}/run",
                    headers=headers
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        cli.renderer.success("Workflow started")
                        return CommandResult.ok(data=data)
                    else:
                        error = await resp.text()
                        cli.renderer.error(f"Failed to run: {error}")
                        return CommandResult.fail(error)

        except Exception as e:
            cli.renderer.warning(f"n8n not available: {e}")
            cli.renderer.info("Running template directly instead...")
            # Try to run as template
            if name_or_id in self.WORKFLOW_TEMPLATES:
                return await self._run_template_directly(cli, name_or_id)
            cli.renderer.error("Workflow not found as template")
            return CommandResult.fail(str(e))

    async def _run_template_directly(self, cli: "JottyCLI", template_name: str) -> CommandResult:
        """Run a workflow template directly without n8n."""
        if template_name not in self.WORKFLOW_TEMPLATES:
            cli.renderer.error(f"Template not found: {template_name}")
            return CommandResult.fail("Template not found")

        template = self.WORKFLOW_TEMPLATES[template_name]
        task = self._get_task_for_template(template_name)

        cli.renderer.info(f"Running: {template['name']}")
        cli.renderer.print(f"[dim]Task: {task}[/dim]")
        cli.renderer.newline()

        # Execute the task directly
        result = await cli.run_once(task)
        return result

    async def _show_help(self, cli: "JottyCLI") -> CommandResult:
        """Show workflow help."""
        cli.renderer.header("Workflow Command")
        cli.renderer.print("""
[bold]Usage:[/bold]
  /workflow [subcommand] [args]

[bold]Subcommands:[/bold]
  list          List workflows and templates
  templates     Show available templates
  run <name>    Run a workflow/template
  create <tpl>  Create workflow from template
  start         Start API server (background)
  stop          Stop API server
  status        Check n8n status
  alias <n> <id> Create nickname for workflow

[bold]Examples:[/bold]
  /workflow                     List all workflows
  /workflow run daily-research  Run the daily-research template
  /workflow start               Start API server for n8n
  /workflow stop                Stop API server

[bold]Templates:[/bold]""")
        for name, tpl in self.WORKFLOW_TEMPLATES.items():
            cli.renderer.print(f"  {name}: {tpl['description']}")

        cli.renderer.print("""
[bold]Note:[/bold]
  Templates can run directly without n8n.
  For scheduling, install n8n: docker run -p 5678:5678 n8nio/n8n
""")
        return CommandResult.ok()

    async def _export_workflow(self, cli: "JottyCLI", flags: dict) -> CommandResult:
        """Export all workflow templates."""
        output_dir = Path.home() / ".jotty" / "workflows"
        output_dir.mkdir(parents=True, exist_ok=True)

        cli.renderer.info("Exporting workflow templates...")

        for name, template in self.WORKFLOW_TEMPLATES.items():
            workflow = self._generate_n8n_workflow(name, template, flags)
            path = output_dir / f"{name}.json"
            path.write_text(json.dumps(workflow, indent=2))
            cli.renderer.print(f" {name}.json")

        cli.renderer.success(f"Exported to: {output_dir}")
        cli.renderer.info("Import into n8n via Settings > Import")

        return CommandResult.ok(output=str(output_dir))

    async def _start_api_server(self, cli: "JottyCLI", flags: dict) -> CommandResult:
        """Start the Jotty API server for n8n integration (runs in background)."""
        import socket
        import subprocess
        import os

        host = flags.get("host", "0.0.0.0")
        port = int(flags.get("port", 8765))
        foreground = flags.get("foreground", flags.get("fg", False))

        # Check if port is available
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind((host, port))
            sock.close()
        except OSError:
            cli.renderer.error(f"Port {port} is already in use!")
            cli.renderer.info("Options:")
            cli.renderer.info(f"  1. Kill existing: /workflow server stop")
            cli.renderer.info(f"  2. Use different port: /workflow server --port 8766")
            return CommandResult.fail(f"Port {port} in use")

        if foreground:
            # Run in foreground (blocking)
            return await self._run_server_foreground(cli, host, port)
        else:
            # Run in background (non-blocking)
            return await self._run_server_background(cli, host, port)

    async def _run_server_background(self, cli: "JottyCLI", host: str, port: int) -> CommandResult:
        """Start server in background process."""
        import subprocess
        import sys

        cli.renderer.info(f"Starting API server on port {port}...")

        # Get the Jotty module path
        jotty_path = str(Path(__file__).parent.parent.parent.resolve())

        # Create a simple server script
        server_script = f'''
import sys
sys.path.insert(0, "{jotty_path}")
from Jotty.apps.cli.api import JottyAPIServer
import uvicorn

server = JottyAPIServer("{host}", {port})
app = server.create_app()
uvicorn.run(app, host="{host}", port={port}, log_level="warning")
'''

        # Write to temp file
        script_path = Path.home() / ".jotty" / "api_server.py"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(server_script)

        # Save PID file path
        pid_file = Path.home() / ".jotty" / "api_server.pid"

        # Start in background
        process = subprocess.Popen(
            [sys.executable, str(script_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )

        # Save PID
        pid_file.write_text(str(process.pid))

        # Wait a moment and check if it started
        await asyncio.sleep(1)

        if process.poll() is None:
            cli.renderer.success(f"API Server running in background (PID: {process.pid})")
            cli.renderer.info(f"  URL: http://localhost:{port}")
            cli.renderer.info(f"  Docs: http://localhost:{port}/docs")
            cli.renderer.info(f"  n8n webhook: http://localhost:{port}/api/run")
            cli.renderer.newline()
            cli.renderer.info("Stop with: /workflow server stop")
        else:
            cli.renderer.error("Server failed to start")
            return CommandResult.fail("Server failed")

        return CommandResult.ok()

    async def _run_server_foreground(self, cli: "JottyCLI", host: str, port: int) -> CommandResult:
        """Run server in foreground (blocking)."""
        cli.renderer.header("Starting Jotty API Server (foreground)")
        cli.renderer.info(f"Host: {host}")
        cli.renderer.info(f"Port: {port}")
        cli.renderer.info(f"Docs: http://localhost:{port}/docs")
        cli.renderer.newline()
        cli.renderer.info("Press Ctrl+C to stop")

        try:
            from ..api import JottyAPIServer
            import uvicorn

            server = JottyAPIServer(host, port)
            app = server.create_app()

            config = uvicorn.Config(app, host=host, port=port, log_level="info")
            uvi_server = uvicorn.Server(config)
            await uvi_server.serve()

        except ImportError as e:
            cli.renderer.error(f"Missing dependency: {e}")
            cli.renderer.info("Install with: pip install fastapi uvicorn")
            return CommandResult.fail(str(e))
        except KeyboardInterrupt:
            cli.renderer.info("Server stopped.")

        return CommandResult.ok()

    async def _stop_api_server(self, cli: "JottyCLI", port: int = 8765) -> CommandResult:
        """Stop the background API server."""
        import signal
        import subprocess

        pid_file = Path.home() / ".jotty" / "api_server.pid"
        stopped = False

        # Try PID file first
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                os.kill(pid, signal.SIGTERM)
                pid_file.unlink()
                cli.renderer.success(f"Server stopped (PID: {pid})")
                stopped = True
            except ProcessLookupError:
                pid_file.unlink()
                cli.renderer.info("PID file was stale, cleaning up...")
            except Exception as e:
                cli.renderer.warning(f"Could not stop via PID: {e}")

        # Also try to find process using the port
        if not stopped:
            try:
                # Find process using port 8765
                result = subprocess.run(
                    ["lsof", "-ti", f":{port}"],
                    capture_output=True, text=True
                )
                if result.stdout.strip():
                    pids = result.stdout.strip().split('\n')
                    for pid in pids:
                        try:
                            os.kill(int(pid), signal.SIGTERM)
                            cli.renderer.success(f"Killed process on port {port} (PID: {pid})")
                            stopped = True
                        except Exception:
                            pass
            except FileNotFoundError:
                # lsof not available, try fuser
                try:
                    result = subprocess.run(
                        ["fuser", f"{port}/tcp"],
                        capture_output=True, text=True
                    )
                    if result.stdout.strip():
                        for pid in result.stdout.strip().split():
                            try:
                                os.kill(int(pid), signal.SIGTERM)
                                cli.renderer.success(f"Killed process on port {port} (PID: {pid})")
                                stopped = True
                            except Exception:
                                pass
                except Exception:
                    pass

        if not stopped:
            cli.renderer.warning(f"No server found on port {port}")
            cli.renderer.info("If port is still busy, manually check with: lsof -i :8765")

        return CommandResult.ok()

    async def _check_n8n_status(self, cli: "JottyCLI") -> CommandResult:
        """Check n8n connection status."""
        cli.renderer.info("Checking n8n status...")

        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{N8N_BASE_URL}/healthz", timeout=5) as resp:
                    if resp.status == 200:
                        cli.renderer.success("n8n is running")
                        cli.renderer.info(f"URL: {N8N_BASE_URL}")
                        return CommandResult.ok()
                    else:
                        cli.renderer.warning(f"n8n returned status {resp.status}")
                        return CommandResult.fail(f"Status: {resp.status}")

        except Exception as e:
            cli.renderer.error(f"n8n not accessible: {e}")
            cli.renderer.info("Make sure n8n is running:")
            cli.renderer.info("  docker-compose up -d n8n")
            return CommandResult.fail(str(e))

    def get_completions(self, partial: str) -> list:
        """Get completions."""
        subcommands = ["list", "templates", "create", "run", "export", "start", "server", "stop", "status", "alias"]
        templates = list(self.WORKFLOW_TEMPLATES.keys())
        flags = ["--host", "--port", "--name", "--as"]

        # Add registered workflow names
        try:
            registry = _load_workflow_registry()
            workflow_names = list(registry.get("workflows", {}).keys())
            aliases = list(registry.get("aliases", {}).keys())
        except (OSError, json.JSONDecodeError, KeyError):
            # Registry loading failed, use empty lists
            workflow_names = []
            aliases = []

        all_options = subcommands + templates + flags + workflow_names + aliases
        return [s for s in all_options if s.startswith(partial)]
