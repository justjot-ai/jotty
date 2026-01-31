"""
Swimlane Command
================

View tasks organized by swimlane/status.
"""

import aiohttp
from typing import TYPE_CHECKING, Dict, List

from .base import BaseCommand, CommandResult, ParsedArgs

if TYPE_CHECKING:
    from ..app import JottyCLI


class SwimlaneCommand(BaseCommand):
    """View tasks organized by swimlane."""

    name = "swimlane"
    aliases = ["lane", "lanes"]
    description = "View tasks organized by swimlane (backlog, pending, in_progress, completed, failed)"
    usage = "/swimlane [lane_name]"
    category = "supervisor"

    SUPERVISOR_URL = "http://localhost:8080"

    LANE_KEYS = {
        "suggested": "suggested_tasks",
        "backlog": "backlog_tasks",
        "pending": "pending_tasks",
        "in_progress": "in_progress_tasks",
        "completed": "completed_task_files",
        "failed": "failed_task_files"
    }

    LANE_STYLES = {
        "suggested": "dim",
        "backlog": "white",
        "pending": "cyan",
        "in_progress": "yellow",
        "completed": "green",
        "failed": "red"
    }

    async def execute(self, args: ParsedArgs, cli: "JottyCLI") -> CommandResult:
        """Execute swimlane command."""
        lane_name = args.positional[0] if args.positional else None

        if lane_name:
            return await self._show_lane(lane_name, cli)
        else:
            return await self._show_all_lanes(cli)

    async def _api_request(self, endpoint: str) -> dict:
        """Make HTTP request to Supervisor API."""
        url = f"{self.SUPERVISOR_URL}{endpoint}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                return await resp.json()

    async def _show_all_lanes(self, cli: "JottyCLI") -> CommandResult:
        """Show all swimlanes with task counts."""
        try:
            state = await self._api_request("/api/state")

            lines = []
            total = 0

            for lane_name, lane_key in self.LANE_KEYS.items():
                tasks = state.get(lane_key, [])
                count = len(tasks)
                total += count

                style = self.LANE_STYLES.get(lane_name, "white")
                bar = "█" * min(count, 30)
                lines.append(f"[{style}]{lane_name:<12}[/{style}]: {count:>3} {bar}")

            lines.append(f"\n{'Total':<12}: {total:>3}")

            cli.renderer.panel("\n".join(lines), title="Swimlanes", style="magenta")

            # Show hint
            cli.renderer.info("Tip: /swimlane <name> to view tasks in a specific lane")

            return CommandResult.ok(data=state)

        except aiohttp.ClientConnectorError:
            cli.renderer.error("Cannot connect to Supervisor at localhost:8080")
            return CommandResult.fail("Connection failed")
        except Exception as e:
            cli.renderer.error(f"API error: {e}")
            return CommandResult.fail(str(e))

    async def _show_lane(self, lane_name: str, cli: "JottyCLI") -> CommandResult:
        """Show tasks in a specific swimlane."""
        # Normalize lane name
        lane_name = lane_name.lower().replace("-", "_")

        if lane_name not in self.LANE_KEYS:
            cli.renderer.error(f"Unknown lane: {lane_name}")
            cli.renderer.info(f"Available lanes: {', '.join(self.LANE_KEYS.keys())}")
            return CommandResult.fail("Unknown lane")

        try:
            state = await self._api_request("/api/state")

            lane_key = self.LANE_KEYS[lane_name]
            task_ids = state.get(lane_key, [])
            task_details = state.get("task_details", {})

            if not task_ids:
                cli.renderer.info(f"No tasks in '{lane_name}' lane")
                return CommandResult.ok(data=[])

            # Build task list
            tasks = []
            lines = []

            for task_id in task_ids[:20]:  # Limit to 20 tasks
                task = task_details.get(task_id, {})
                tasks.append(task)

                title = task.get("title", "Untitled")[:50]
                priority = task.get("priority", 3)
                agent = task.get("agent_type", "claude")

                lines.append(f"• [{task_id}] {title}")
                lines.append(f"    Priority: {priority}, Agent: {agent}")

            if len(task_ids) > 20:
                lines.append(f"\n... and {len(task_ids) - 20} more")

            style = self.LANE_STYLES.get(lane_name, "white")
            cli.renderer.panel(
                "\n".join(lines),
                title=f"Swimlane: {lane_name.replace('_', ' ').title()} ({len(task_ids)} tasks)",
                style=style
            )

            return CommandResult.ok(data=tasks)

        except aiohttp.ClientConnectorError:
            cli.renderer.error("Cannot connect to Supervisor at localhost:8080")
            return CommandResult.fail("Connection failed")
        except Exception as e:
            cli.renderer.error(f"API error: {e}")
            return CommandResult.fail(str(e))

    def get_completions(self, partial: str) -> list:
        """Get lane name completions."""
        lanes = list(self.LANE_KEYS.keys())
        return [l for l in lanes if l.startswith(partial)]
