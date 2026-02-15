"""
Jotty Heartbeat System
======================

Proactive AI that acts WITHOUT being asked.
Inspired by OpenClaw's "Heartbeat" feature.

The heartbeat system periodically:
- Checks email for urgent messages
- Monitors scheduled tasks
- Sends proactive notifications
- Runs background automations

This is what makes Jotty feel "alive" like a real assistant.
"""

from .engine import HeartbeatEngine, HeartbeatTask, TaskFrequency, create_default_tasks
from .monitors import CalendarMonitor, EmailMonitor, FileMonitor

__all__ = [
    "HeartbeatEngine",
    "HeartbeatTask",
    "TaskFrequency",
    "create_default_tasks",
    "EmailMonitor",
    "CalendarMonitor",
    "FileMonitor",
]
