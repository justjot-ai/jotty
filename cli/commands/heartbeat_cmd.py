"""
Heartbeat Command
=================

/heartbeat - Proactive AI that acts without being asked
/remind - Set reminders

Like OpenClaw's "Heartbeat" feature.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from .base import BaseCommand, CommandResult, ParsedArgs

if TYPE_CHECKING:
    from ..app import JottyCLI

logger = logging.getLogger(__name__)


class HeartbeatCommand(BaseCommand):
    """
    /heartbeat - Proactive background monitoring.

    Jotty can act WITHOUT being asked:
    - Monitor your inbox
    - Send morning briefings
    - Watch files for changes
    - Alert you to important events
    """

    name = "heartbeat"
    aliases = ["hb", "proactive"]
    description = "Proactive AI that acts without being asked"
    usage = "/heartbeat [start|stop|status|tasks]"
    category = "automation"

    _engine = None

    async def execute(self, args: ParsedArgs, cli: "JottyCLI") -> CommandResult:
        """Execute heartbeat command."""
        subcommand = args.positional[0] if args.positional else "status"

        if subcommand == "start":
            return await self._start_heartbeat(cli)
        elif subcommand == "stop":
            return await self._stop_heartbeat(cli)
        elif subcommand == "status":
            return await self._show_status(cli)
        elif subcommand == "tasks":
            return await self._list_tasks(cli)
        elif subcommand == "enable":
            task_name = args.positional[1] if len(args.positional) > 1 else None
            return await self._toggle_task(cli, task_name, True)
        elif subcommand == "disable":
            task_name = args.positional[1] if len(args.positional) > 1 else None
            return await self._toggle_task(cli, task_name, False)
        else:
            return await self._show_status(cli)

    async def _get_engine(self, cli: "JottyCLI"):
        """Get or create heartbeat engine."""
        if HeartbeatCommand._engine is None:
            from ..heartbeat import HeartbeatEngine, create_default_tasks
            HeartbeatCommand._engine = HeartbeatEngine(cli)

            # Add default tasks
            for task in create_default_tasks():
                HeartbeatCommand._engine.add_task(task)

            # Add notification handler
            async def notify_handler(title, message, priority):
                # Send to connected channels
                if priority == "urgent":
                    icon = "🔴"
                elif priority == "high":
                    icon = "🟡"
                else:
                    icon = "💡"

                cli.renderer.print(f"\n{icon} [bold]{title}[/bold]: {message}")

                # Also send to Telegram if configured
                try:
                    from skills.telegram_sender.tools import send_telegram_message_tool
                    await send_telegram_message_tool({
                        "message": f"{icon} {title}\n\n{message}"
                    })
                except Exception:
                    pass  # Telegram not configured

            HeartbeatCommand._engine.add_notifier(notify_handler)

        return HeartbeatCommand._engine

    async def _start_heartbeat(self, cli: "JottyCLI") -> CommandResult:
        """Start the heartbeat engine."""
        engine = await self._get_engine(cli)

        if engine._running:
            cli.renderer.warning("Heartbeat already running")
            return CommandResult.ok()

        await engine.start()
        cli.renderer.success("Heartbeat started")
        cli.renderer.info("Jotty is now proactively monitoring in the background")
        cli.renderer.newline()
        cli.renderer.print("[bold]Active tasks:[/bold]")

        for name, task in engine._tasks.items():
            status = "✓" if task.enabled else "○"
            cli.renderer.print(f"  {status} {name}: {task.description}")

        cli.renderer.newline()
        cli.renderer.info("Stop with: /heartbeat stop")

        return CommandResult.ok()

    async def _stop_heartbeat(self, cli: "JottyCLI") -> CommandResult:
        """Stop the heartbeat engine."""
        engine = await self._get_engine(cli)
        await engine.stop()
        cli.renderer.success("Heartbeat stopped")
        return CommandResult.ok()

    async def _show_status(self, cli: "JottyCLI") -> CommandResult:
        """Show heartbeat status."""
        engine = await self._get_engine(cli)
        status = engine.status

        if status["running"]:
            cli.renderer.success("Heartbeat: Running")
        else:
            cli.renderer.warning("Heartbeat: Stopped")
            cli.renderer.info("Start with: /heartbeat start")

        cli.renderer.newline()
        cli.renderer.print(f"[bold]Tasks: {len(status['tasks'])}[/bold]")

        for name, task_status in status["tasks"].items():
            enabled = "✓" if task_status["enabled"] else "○"
            last_run = task_status["last_run"] or "never"
            cli.renderer.print(
                f"  {enabled} {name} ({task_status['frequency']}) - "
                f"runs: {task_status['run_count']}, last: {last_run}"
            )

        return CommandResult.ok(data=status)

    async def _list_tasks(self, cli: "JottyCLI") -> CommandResult:
        """List all heartbeat tasks."""
        engine = await self._get_engine(cli)

        cli.renderer.header("Heartbeat Tasks")

        for name, task in engine._tasks.items():
            status = "[green]enabled[/green]" if task.enabled else "[dim]disabled[/dim]"
            cli.renderer.print(f"\n[cyan]{name}[/cyan] ({status})")
            cli.renderer.print(f"  {task.description}")
            cli.renderer.print(f"  Frequency: {task.frequency.name}")
            if task.last_run:
                cli.renderer.print(f"  Last run: {task.last_run.strftime('%Y-%m-%d %H:%M')}")
            cli.renderer.print(f"  Run count: {task.run_count}")

        cli.renderer.newline()
        cli.renderer.info("Enable/disable: /heartbeat enable <task_name>")

        return CommandResult.ok()

    async def _toggle_task(self, cli: "JottyCLI", task_name: str, enable: bool) -> CommandResult:
        """Enable/disable a task."""
        if not task_name:
            cli.renderer.error("Task name required")
            return CommandResult.fail("Task name required")

        engine = await self._get_engine(cli)
        engine.enable_task(task_name, enable)

        action = "enabled" if enable else "disabled"
        cli.renderer.success(f"Task '{task_name}' {action}")

        return CommandResult.ok()

    def get_completions(self, partial: str) -> list:
        subcommands = ["start", "stop", "status", "tasks", "enable", "disable"]
        return [s for s in subcommands if s.startswith(partial)]


class RemindCommand(BaseCommand):
    """
    /remind - Set a reminder.

    Quick way to set reminders that trigger via heartbeat.
    """

    name = "remind"
    aliases = ["reminder", "alert"]
    description = "Set a reminder"
    usage = "/remind <message> --in 30m | --at 14:00"
    category = "automation"

    async def execute(self, args: ParsedArgs, cli: "JottyCLI") -> CommandResult:
        """Execute remind command."""
        if not args.positional:
            return await self._list_reminders(cli)

        message = " ".join(args.positional)

        # Parse time
        in_time = args.flags.get("in")
        at_time = args.flags.get("at")

        if in_time:
            remind_at = self._parse_relative_time(in_time)
        elif at_time:
            remind_at = self._parse_absolute_time(at_time)
        else:
            # Default to 30 minutes
            remind_at = datetime.now() + timedelta(minutes=30)

        if not remind_at:
            cli.renderer.error("Could not parse time")
            cli.renderer.info("Usage: /remind message --in 30m")
            cli.renderer.info("       /remind message --at 14:00")
            return CommandResult.fail("Invalid time")

        # Save reminder
        reminder = {
            "message": message,
            "time": remind_at.isoformat(),
            "created": datetime.now().isoformat(),
            "sent": False,
            "priority": args.flags.get("priority", "normal")
        }

        reminders_file = Path.home() / ".jotty" / "reminders.json"
        reminders_file.parent.mkdir(parents=True, exist_ok=True)

        if reminders_file.exists():
            reminders = json.loads(reminders_file.read_text())
        else:
            reminders = []

        reminders.append(reminder)
        reminders_file.write_text(json.dumps(reminders, indent=2))

        cli.renderer.success(f"Reminder set for {remind_at.strftime('%Y-%m-%d %H:%M')}")
        cli.renderer.print(f"  Message: {message}")

        # Make sure heartbeat is running
        cli.renderer.info("Make sure heartbeat is running: /heartbeat start")

        return CommandResult.ok(data=reminder)

    async def _list_reminders(self, cli: "JottyCLI") -> CommandResult:
        """List upcoming reminders."""
        reminders_file = Path.home() / ".jotty" / "reminders.json"

        if not reminders_file.exists():
            cli.renderer.info("No reminders set")
            cli.renderer.info("Set one: /remind message --in 30m")
            return CommandResult.ok()

        reminders = json.loads(reminders_file.read_text())
        pending = [r for r in reminders if not r.get("sent")]

        if not pending:
            cli.renderer.info("No pending reminders")
            return CommandResult.ok()

        cli.renderer.header("Reminders")
        for r in sorted(pending, key=lambda x: x["time"]):
            time = datetime.fromisoformat(r["time"]).strftime("%Y-%m-%d %H:%M")
            cli.renderer.print(f"  • {time}: {r['message']}")

        return CommandResult.ok(data={"reminders": pending})

    def _parse_relative_time(self, time_str: str) -> datetime:
        """Parse relative time like '30m', '2h', '1d'."""
        try:
            value = int(time_str[:-1])
            unit = time_str[-1].lower()

            if unit == "m":
                return datetime.now() + timedelta(minutes=value)
            elif unit == "h":
                return datetime.now() + timedelta(hours=value)
            elif unit == "d":
                return datetime.now() + timedelta(days=value)
            elif unit == "w":
                return datetime.now() + timedelta(weeks=value)
        except (ValueError, IndexError):
            pass
        return None

    def _parse_absolute_time(self, time_str: str) -> datetime:
        """Parse absolute time like '14:00' or '2024-01-15 14:00'."""
        try:
            if " " in time_str:
                return datetime.strptime(time_str, "%Y-%m-%d %H:%M")
            else:
                today = datetime.now().date()
                time = datetime.strptime(time_str, "%H:%M").time()
                result = datetime.combine(today, time)
                # If time already passed today, set for tomorrow
                if result < datetime.now():
                    result += timedelta(days=1)
                return result
        except ValueError:
            pass
        return None

    def get_completions(self, partial: str) -> list:
        return ["--in", "--at", "--priority"]
