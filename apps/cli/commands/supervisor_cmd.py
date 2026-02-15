"""
Supervisor Command
==================

Supervisor control and status.
"""

from typing import TYPE_CHECKING

import aiohttp

from .base import BaseCommand, CommandResult, ParsedArgs

if TYPE_CHECKING:
    from ..app import JottyCLI


class SupervisorCommand(BaseCommand):
    """Supervisor control and monitoring."""

    name = "supervisor"
    aliases = ["sv"]
    description = "Control and monitor Supervisor Coder orchestrator"
    usage = "/supervisor [status|credentials]"
    category = "supervisor"

    SUPERVISOR_URL = "http://localhost:8080"

    async def execute(self, args: ParsedArgs, cli: "JottyCLI") -> CommandResult:
        """Execute supervisor command."""
        subcommand = args.positional[0] if args.positional else "status"

        if subcommand == "status":
            return await self._show_status(cli)
        elif subcommand == "credentials":
            return await self._check_credentials(cli)
        else:
            return await self._show_status(cli)

    async def _api_request(self, method: str, endpoint: str) -> dict:
        """Make HTTP request to Supervisor API."""
        url = f"{self.SUPERVISOR_URL}{endpoint}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                return await resp.json()

    async def _show_status(self, cli: "JottyCLI") -> CommandResult:
        """Show orchestrator status."""
        try:
            # Get orchestrator status
            status = await self._api_request("GET", "/api/orchestrator/status")

            # Get task state for running tasks
            state = await self._api_request("GET", "/api/state")

            running = status.get("running", False)
            orchestrator_type = status.get("type", "unknown")

            # Count running tasks
            in_progress = state.get("in_progress_tasks", [])
            pending = state.get("pending_tasks", [])

            # Build status info
            info = {
                "Orchestrator": orchestrator_type,
                "Running": " Yes" if running else " No",
                "Tasks In Progress": len(in_progress),
                "Tasks Pending": len(pending),
            }

            # Show running task details
            if in_progress:
                task_details = state.get("task_details", {})
                running_info = []
                for task_id in in_progress[:5]:
                    task = task_details.get(task_id, {})
                    title = task.get("title", "Unknown")[:30]
                    running_info.append(f"  • {task_id}: {title}")
                info["Running Tasks"] = "\n" + "\n".join(running_info)

            cli.renderer.tree(info, title="Supervisor Status")
            return CommandResult.ok(data={"status": status, "state": state})

        except aiohttp.ClientConnectorError:
            cli.renderer.error("Cannot connect to Supervisor at localhost:8080")
            cli.renderer.info("Make sure the Supervisor container is running")
            return CommandResult.fail("Connection failed")
        except Exception as e:
            cli.renderer.error(f"API error: {e}")
            return CommandResult.fail(str(e))

    async def _check_credentials(self, cli: "JottyCLI") -> CommandResult:
        """Check agent credentials."""
        try:
            result = await self._api_request("GET", "/api/credentials/check")

            # Display credentials status
            lines = []
            for agent, available in result.items():
                status = " Available" if available else " Missing"
                lines.append(f"{agent.capitalize():<12}: {status}")

            cli.renderer.panel("\n".join(lines), title="Agent Credentials", style="cyan")
            return CommandResult.ok(data=result)

        except aiohttp.ClientConnectorError:
            cli.renderer.error("Cannot connect to Supervisor at localhost:8080")
            return CommandResult.fail("Connection failed")
        except Exception as e:
            cli.renderer.error(f"API error: {e}")
            return CommandResult.fail(str(e))

    def get_completions(self, partial: str) -> list:
        """Get subcommand completions."""
        subcommands = ["status", "credentials"]
        return [s for s in subcommands if s.startswith(partial)]
