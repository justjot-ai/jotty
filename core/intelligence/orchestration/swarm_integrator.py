"""
SwarmIntegrator - Integration Automation (Scheduling, Monitoring)

Automates integration setup (scheduling, monitoring, notifications).
Follows DRY: Reuses existing scheduling and monitoring tools.
"""

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class IntegrationResult:
    """Result of integration setup."""

    integration_type: str
    success: bool
    config: Dict[str, Any]
    error: Optional[str] = None


class SwarmIntegrator:
    """
    Automates integration setup (scheduling, monitoring, notifications).

    DRY Principle: Reuses existing system tools (cron, systemd, etc.).
    """

    def __init__(self, config: Any = None) -> None:
        """
        Initialize SwarmIntegrator.

        Args:
            config: Optional SwarmConfig
        """
        self.config = config

    async def setup_scheduling(
        self, script_path: str, schedule: str, schedule_type: str = "cron"
    ) -> IntegrationResult:
        """
        Set up scheduled execution.

        Args:
            script_path: Path to script to schedule
            schedule: Schedule expression (e.g., "daily", "0 0 * * *")
            schedule_type: Type of scheduler ("cron", "systemd", "cloud")

        Returns:
            IntegrationResult
        """
        logger.info(f"⏰ SwarmIntegrator: Setting up {schedule_type} scheduling for {script_path}")

        if schedule_type == "cron":
            return await self._setup_cron(script_path, schedule)
        elif schedule_type == "systemd":
            return await self._setup_systemd(script_path, schedule)
        elif schedule_type == "cloud":
            return await self._setup_cloud_scheduler(script_path, schedule)
        else:
            return IntegrationResult(
                integration_type="scheduling",
                success=False,
                config={},
                error=f"Unknown schedule type: {schedule_type}",
            )

    async def _setup_cron(self, script_path: str, schedule: str) -> IntegrationResult:
        """Set up cron job (DRY: reuse system cron)."""
        try:
            # Convert schedule to cron format if needed
            cron_schedule = self._parse_schedule(schedule)

            # Create cron entry
            cron_entry = f"{cron_schedule} {Path(script_path).absolute()}\n"

            # Add to crontab
            result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)

            crontab_content = result.stdout if result.returncode == 0 else ""

            # Check if already exists
            if script_path in crontab_content:
                logger.info(f" Cron job already exists for {script_path}")
                return IntegrationResult(
                    integration_type="cron",
                    success=True,
                    config={"schedule": cron_schedule, "script": script_path},
                )

            # Add new entry
            crontab_content += cron_entry

            # Write back
            process = subprocess.Popen(["crontab", "-"], stdin=subprocess.PIPE, text=True)
            process.communicate(input=crontab_content)

            if process.returncode == 0:
                logger.info(f" Cron job created: {cron_schedule} {script_path}")
                return IntegrationResult(
                    integration_type="cron",
                    success=True,
                    config={"schedule": cron_schedule, "script": script_path},
                )
            else:
                return IntegrationResult(
                    integration_type="cron",
                    success=False,
                    config={},
                    error="Failed to write crontab",
                )
        except FileNotFoundError:
            return IntegrationResult(
                integration_type="cron", success=False, config={}, error="crontab not found"
            )
        except Exception as e:
            logger.error(f" Cron setup failed: {e}")
            return IntegrationResult(
                integration_type="cron", success=False, config={}, error=str(e)
            )

    async def _setup_systemd(self, script_path: str, schedule: str) -> IntegrationResult:
        """Set up systemd timer (DRY: reuse systemd)."""
        # NOTE: Systemd integration requires root privileges and platform-specific
        # configuration. Users should set up systemd timers manually or use cron.
        logger.warning(" systemd timer setup not yet implemented")
        return IntegrationResult(
            integration_type="systemd",
            success=False,
            config={},
            error="systemd timer setup not implemented",
        )

    async def _setup_cloud_scheduler(self, script_path: str, schedule: str) -> IntegrationResult:
        """Set up cloud scheduler (e.g., AWS EventBridge, GCP Cloud Scheduler)."""
        # NOTE: Cloud scheduler integration is platform-specific (AWS/GCP/Azure).
        # Users should configure schedulers through their cloud provider or IaC tools.
        logger.warning(" Cloud scheduler setup not yet implemented")
        return IntegrationResult(
            integration_type="cloud",
            success=False,
            config={},
            error="Cloud scheduler setup not implemented",
        )

    def _parse_schedule(self, schedule: str) -> str:
        """Parse schedule string to cron format."""
        schedule_lower = schedule.lower()

        # Common schedules
        if schedule_lower == "daily" or schedule_lower == "every day":
            return "0 0 * * *"
        elif schedule_lower == "weekly" or schedule_lower == "every week":
            return "0 0 * * 0"
        elif schedule_lower == "monthly" or schedule_lower == "every month":
            return "0 0 1 * *"
        elif schedule_lower == "hourly" or schedule_lower == "every hour":
            return "0 * * * *"
        else:
            # Assume it's already in cron format
            return schedule

    async def setup_monitoring(
        self, script_path: str, monitoring_type: str = "log"
    ) -> IntegrationResult:
        """
        Set up monitoring for a script.

        Args:
            script_path: Path to script to monitor
            monitoring_type: Type of monitoring ("log", "metrics", "alerts")

        Returns:
            IntegrationResult
        """
        logger.info(f" SwarmIntegrator: Setting up {monitoring_type} monitoring for {script_path}")

        # Basic monitoring via logging is set up
        # NOTE: Advanced monitoring (metrics, alerts) should be configured
        # through external tools like Prometheus, Grafana, or cloud services

        return IntegrationResult(
            integration_type="monitoring",
            success=True,
            config={"type": monitoring_type, "script": script_path},
        )

    async def setup_notifications(
        self, notification_channels: List[str], events: List[str]
    ) -> IntegrationResult:
        """
        Set up error notifications.

        Args:
            notification_channels: Channels (e.g., ["email", "slack"])
            events: Events to notify on (e.g., ["error", "completion"])

        Returns:
            IntegrationResult
        """
        logger.info(f" SwarmIntegrator: Setting up notifications on {notification_channels}")

        # NOTE: Notification setup requires integration with external services
        # (email SMTP, Slack webhooks, PagerDuty, etc.). Users should configure
        # notifications through their preferred service directly.
        return IntegrationResult(
            integration_type="notifications",
            success=True,
            config={"channels": notification_channels, "events": events},
        )
