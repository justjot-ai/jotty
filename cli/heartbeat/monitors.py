"""
Heartbeat Monitors
==================

Background monitors for proactive features:
- Email monitoring for urgent messages
- Calendar reminders
- File change detection
- Custom watchers
"""

import asyncio
import logging
import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MonitorResult:
    """Result from a monitor check."""
    has_update: bool = False
    title: str = ""
    message: str = ""
    priority: str = "normal"  # normal, high, urgent
    data: Dict[str, Any] = None


class EmailMonitor:
    """
    Monitor email for urgent messages.

    Uses Gmail API to check for:
    - Unread emails from VIP contacts
    - Emails with urgent keywords
    - Calendar invites
    """

    def __init__(self, cli=None):
        self._cli = cli
        self._vip_contacts: List[str] = []
        self._urgent_keywords = ["urgent", "asap", "emergency", "important", "deadline"]
        self._last_check: Optional[datetime] = None
        self._seen_ids: set = set()

    def add_vip(self, email: str):
        """Add a VIP contact."""
        self._vip_contacts.append(email.lower())

    async def check(self) -> MonitorResult:
        """Check for urgent emails."""
        # This would use Gmail API
        # Placeholder for now
        return MonitorResult(has_update=False)


class CalendarMonitor:
    """
    Monitor calendar for upcoming events.

    Sends reminders before meetings.
    """

    def __init__(self, cli=None):
        self._cli = cli
        self._reminder_minutes = [15, 5]  # Remind 15 min and 5 min before
        self._reminded_events: set = set()

    async def check(self) -> MonitorResult:
        """Check for upcoming calendar events."""
        # This would use Google Calendar API
        # Placeholder for now
        return MonitorResult(has_update=False)


class FileMonitor:
    """
    Monitor files/directories for changes.

    Can watch:
    - Specific files
    - Directories for new files
    - File content changes
    """

    def __init__(self, cli=None):
        self._cli = cli
        self._watched_paths: Dict[str, str] = {}  # path -> hash
        self._watch_dirs: List[Path] = []

    def watch_file(self, path: str):
        """Add a file to watch."""
        path_obj = Path(path).expanduser()
        if path_obj.exists():
            self._watched_paths[str(path_obj)] = self._hash_file(path_obj)

    def watch_directory(self, path: str):
        """Add a directory to watch for new files."""
        path_obj = Path(path).expanduser()
        if path_obj.is_dir():
            self._watch_dirs.append(path_obj)

    def _hash_file(self, path: Path) -> str:
        """Get hash of file contents."""
        try:
            return hashlib.md5(path.read_bytes()).hexdigest()
        except Exception:
            return ""

    async def check(self) -> MonitorResult:
        """Check for file changes."""
        changes = []

        # Check watched files
        for path_str, old_hash in list(self._watched_paths.items()):
            path = Path(path_str)
            if path.exists():
                new_hash = self._hash_file(path)
                if new_hash != old_hash:
                    changes.append(f"Modified: {path.name}")
                    self._watched_paths[path_str] = new_hash
            else:
                changes.append(f"Deleted: {path.name}")
                del self._watched_paths[path_str]

        # Check watched directories for new files
        for dir_path in self._watch_dirs:
            for file_path in dir_path.iterdir():
                if str(file_path) not in self._watched_paths:
                    changes.append(f"New file: {file_path.name}")
                    self._watched_paths[str(file_path)] = self._hash_file(file_path)

        if changes:
            return MonitorResult(
                has_update=True,
                title="File Changes Detected",
                message="\n".join(changes[:5]),
                priority="normal",
                data={"changes": changes}
            )

        return MonitorResult(has_update=False)


class InboxMonitor:
    """
    Unified inbox monitor.

    Checks all connected channels for new messages:
    - WhatsApp
    - Telegram
    - Slack
    - Discord
    """

    def __init__(self, cli=None):
        self._cli = cli
        self._last_check: Dict[str, datetime] = {}
        self._unread_counts: Dict[str, int] = {}

    async def check(self) -> MonitorResult:
        """Check all inboxes for new messages."""
        new_messages = []

        # This would check each connected channel
        # For now, placeholder

        if new_messages:
            return MonitorResult(
                has_update=True,
                title="New Messages",
                message=f"You have {len(new_messages)} new messages",
                priority="normal",
                data={"messages": new_messages}
            )

        return MonitorResult(has_update=False)


class WebMonitor:
    """
    Monitor web pages for changes.

    Useful for:
    - Price tracking
    - News alerts
    - Stock updates
    """

    def __init__(self, cli=None):
        self._cli = cli
        self._watched_urls: Dict[str, Dict] = {}  # url -> {selector, last_value}

    def watch_url(self, url: str, selector: str = None, description: str = ""):
        """Add URL to watch."""
        self._watched_urls[url] = {
            "selector": selector,
            "description": description,
            "last_value": None,
            "last_check": None
        }

    async def check(self) -> MonitorResult:
        """Check watched URLs for changes."""
        changes = []

        for url, config in self._watched_urls.items():
            try:
                # Use web scraper skill
                if self._cli:
                    result = await self._cli.run_once(
                        f'/tools scrape_website {{"url": "{url}"}}'
                    )

                    # Compare with last value
                    if config["last_value"] and result != config["last_value"]:
                        changes.append(config.get("description", url))

                    config["last_value"] = result
                    config["last_check"] = datetime.now().isoformat()

            except Exception as e:
                logger.error(f"Web monitor error for {url}: {e}")

        if changes:
            return MonitorResult(
                has_update=True,
                title="Web Page Changes",
                message="\n".join(changes[:3]),
                priority="normal",
                data={"changes": changes}
            )

        return MonitorResult(has_update=False)
