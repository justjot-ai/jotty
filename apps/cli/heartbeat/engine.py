"""
Heartbeat Engine
================

Core engine for proactive background tasks.
Runs continuously, executing monitors and tasks on schedule.

Like OpenClaw's heartbeat - Jotty can act WITHOUT being asked.
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)


class TaskFrequency(Enum):
    """How often a task runs."""
    EVERY_MINUTE = 60
    EVERY_5_MINUTES = 300
    EVERY_15_MINUTES = 900
    EVERY_HOUR = 3600
    EVERY_6_HOURS = 21600
    DAILY = 86400
    WEEKLY = 604800


@dataclass
class HeartbeatTask:
    """A task that runs on the heartbeat."""
    name: str
    description: str
    frequency: TaskFrequency
    handler: Callable
    enabled: bool = True
    last_run: Optional[datetime] = None
    run_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None

    # Conditions
    run_on_start: bool = False
    quiet_hours: Optional[tuple] = None  # (start_hour, end_hour)

    def should_run(self) -> bool:
        """Check if task should run now."""
        if not self.enabled:
            return False

        # Check quiet hours
        if self.quiet_hours:
            current_hour = datetime.now().hour
            start, end = self.quiet_hours
            if start <= current_hour < end:
                return False

        # Check if enough time has passed
        if self.last_run is None:
            return self.run_on_start

        elapsed = (datetime.now() - self.last_run).total_seconds()
        return elapsed >= self.frequency.value


class HeartbeatEngine:
    """
    The Heartbeat - makes Jotty proactive.

    Continuously runs background monitors and tasks:
    - Email monitoring for urgent messages
    - Calendar reminders
    - File change detection
    - Custom automation tasks
    - Scheduled notifications

    Usage:
        engine = HeartbeatEngine(cli)
        engine.add_task(HeartbeatTask(
            name="email_check",
            description="Check for urgent emails",
            frequency=TaskFrequency.EVERY_5_MINUTES,
            handler=check_email_handler
        ))
        await engine.start()
    """

    def __init__(self, cli: Any = None) -> None:
        self._cli = cli
        self._tasks: Dict[str, HeartbeatTask] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._state_file = Path.home() / ".jotty" / "heartbeat_state.json"
        self._notifiers: List[Callable] = []

        # Load saved state
        self._load_state()

    def set_cli(self, cli: Any) -> Any:
        """Set CLI instance for executing tasks."""
        self._cli = cli

    def add_notifier(self, notifier: Callable) -> Any:
        """Add notification handler (e.g., send to Telegram, WhatsApp)."""
        self._notifiers.append(notifier)

    def add_task(self, task: HeartbeatTask) -> Any:
        """Add a heartbeat task."""
        self._tasks[task.name] = task
        logger.info(f"Heartbeat task added: {task.name} ({task.frequency.name})")

    def remove_task(self, name: str) -> Any:
        """Remove a task."""
        if name in self._tasks:
            del self._tasks[name]

    def enable_task(self, name: str, enabled: bool = True) -> Any:
        """Enable/disable a task."""
        if name in self._tasks:
            self._tasks[name].enabled = enabled

    async def start(self) -> Any:
        """Start the heartbeat engine."""
        if self._running:
            logger.warning("Heartbeat already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._heartbeat_loop())
        logger.info("Heartbeat engine started")

    async def stop(self) -> Any:
        """Stop the heartbeat engine."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._save_state()
        logger.info("Heartbeat engine stopped")

    async def _heartbeat_loop(self) -> Any:
        """Main heartbeat loop."""
        logger.info("Heartbeat loop started")

        while self._running:
            try:
                # Check each task
                for task in self._tasks.values():
                    if task.should_run():
                        await self._run_task(task)

                # Sleep until next check (every 30 seconds)
                await asyncio.sleep(30)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat loop error: {e}", exc_info=True)
                await asyncio.sleep(60)

    async def _run_task(self, task: HeartbeatTask) -> Any:
        """Run a heartbeat task."""
        try:
            logger.debug(f"Running heartbeat task: {task.name}")
            task.last_run = datetime.now()
            task.run_count += 1

            # Run the handler
            if asyncio.iscoroutinefunction(task.handler):
                result = await task.handler(self._cli)
            else:
                result = task.handler(self._cli)

            # Handle notifications from task
            if result and isinstance(result, dict):
                if result.get("notify"):
                    await self._send_notification(
                        title=result.get("title", task.name),
                        message=result.get("message", ""),
                        priority=result.get("priority", "normal")
                    )

            logger.debug(f"Task {task.name} completed")

        except Exception as e:
            task.error_count += 1
            task.last_error = str(e)
            logger.error(f"Heartbeat task {task.name} failed: {e}")

    async def _send_notification(self, title: str, message: str, priority: str = 'normal') -> Any:
        """Send notification through all registered notifiers."""
        for notifier in self._notifiers:
            try:
                if asyncio.iscoroutinefunction(notifier):
                    await notifier(title, message, priority)
                else:
                    notifier(title, message, priority)
            except Exception as e:
                logger.error(f"Notifier error: {e}")

        # Also print to CLI if available
        if self._cli and hasattr(self._cli, 'renderer'):
            icon = "" if priority == "urgent" else ""
            self._cli.renderer.print(f"\n{icon} [bold]{title}[/bold]: {message}")

    def _load_state(self) -> Any:
        """Load saved state."""
        try:
            if self._state_file.exists():
                data = json.loads(self._state_file.read_text())
                for name, state in data.get("tasks", {}).items():
                    if name in self._tasks:
                        self._tasks[name].last_run = datetime.fromisoformat(state["last_run"]) if state.get("last_run") else None
                        self._tasks[name].run_count = state.get("run_count", 0)
                        self._tasks[name].error_count = state.get("error_count", 0)
        except Exception as e:
            logger.warning(f"Could not load heartbeat state: {e}")

    def _save_state(self) -> Any:
        """Save state to disk."""
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "tasks": {
                    name: {
                        "last_run": task.last_run.isoformat() if task.last_run else None,
                        "run_count": task.run_count,
                        "error_count": task.error_count
                    }
                    for name, task in self._tasks.items()
                },
                "saved_at": datetime.now().isoformat()
            }
            self._state_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Could not save heartbeat state: {e}")

    @property
    def status(self) -> Dict[str, Any]:
        """Get heartbeat status."""
        return {
            "running": self._running,
            "tasks": {
                name: {
                    "enabled": task.enabled,
                    "frequency": task.frequency.name,
                    "last_run": task.last_run.isoformat() if task.last_run else None,
                    "run_count": task.run_count,
                    "error_count": task.error_count
                }
                for name, task in self._tasks.items()
            }
        }


# ============ BUILT-IN HEARTBEAT TASKS ============

async def morning_briefing_handler(cli: Any) -> Dict:
    """Generate and send morning briefing."""
    current_hour = datetime.now().hour

    # Only run between 7-9 AM
    if not (7 <= current_hour <= 9):
        return {}

    try:
        # Generate briefing using Jotty
        result = await cli.run_once(
            "Create a brief morning summary: today's date, day of week, "
            "and a motivational quote. Keep it under 3 sentences."
        )

        return {
            "notify": True,
            "title": "Good Morning!",
            "message": str(result),
            "priority": "normal"
        }
    except Exception as e:
        logger.error(f"Morning briefing error: {e}")
        return {}


async def email_monitor_handler(cli: Any) -> Dict:
    """Check for urgent emails (placeholder - needs email integration)."""
    # This would integrate with Gmail API
    # For now, it's a placeholder
    return {}


async def reminder_handler(cli: Any) -> Dict:
    """Check for upcoming reminders."""
    # Load reminders from ~/.jotty/reminders.json
    reminders_file = Path.home() / ".jotty" / "reminders.json"

    if not reminders_file.exists():
        return {}

    try:
        reminders = json.loads(reminders_file.read_text())
        now = datetime.now()

        for reminder in reminders:
            remind_at = datetime.fromisoformat(reminder["time"])
            if now >= remind_at and not reminder.get("sent"):
                reminder["sent"] = True
                reminders_file.write_text(json.dumps(reminders, indent=2))

                return {
                    "notify": True,
                    "title": "Reminder",
                    "message": reminder["message"],
                    "priority": reminder.get("priority", "normal")
                }
    except Exception as e:
        logger.error(f"Reminder check error: {e}")

    return {}


def create_default_tasks() -> List[HeartbeatTask]:
    """Create default heartbeat tasks."""
    return [
        HeartbeatTask(
            name="morning_briefing",
            description="Daily morning briefing at 8 AM",
            frequency=TaskFrequency.DAILY,
            handler=morning_briefing_handler,
            enabled=False,  # Disabled by default
            run_on_start=False,
            quiet_hours=(22, 7)  # No notifications 10 PM - 7 AM
        ),
        HeartbeatTask(
            name="reminders",
            description="Check for upcoming reminders",
            frequency=TaskFrequency.EVERY_MINUTE,
            handler=reminder_handler,
            enabled=True,
            run_on_start=True
        ),
    ]
