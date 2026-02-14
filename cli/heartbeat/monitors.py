"""
Heartbeat Monitors
==================

Background monitors for proactive features:
- Email monitoring for urgent messages
- Calendar reminders
- File change detection
- Clipboard watching
- AI marker detection
- Custom watchers
"""

import asyncio
import logging
import json
import hashlib
import re
import subprocess
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass, field

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

    def __init__(self, cli: Any = None) -> None:
        self._cli = cli
        self._vip_contacts: List[str] = []
        self._urgent_keywords = ["urgent", "asap", "emergency", "important", "deadline"]
        self._last_check: Optional[datetime] = None
        self._seen_ids: set = set()

    def add_vip(self, email: str) -> Any:
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

    def __init__(self, cli: Any = None) -> None:
        self._cli = cli
        self._reminder_minutes = [15, 5]  # Remind 15 min and 5 min before
        self._reminded_events: set = set()

    async def check(self) -> MonitorResult:
        """Check for upcoming calendar events."""
        # This would use Google Calendar API
        # Placeholder for now
        return MonitorResult(has_update=False)


@dataclass
class AIMarker:
    """An AI directive marker found in a file."""
    path: str
    line_number: int
    marker_type: str  # "execute" or "query"
    content: str


class FileMonitor:
    """
    Monitor files/directories for changes.

    Can watch:
    - Specific files
    - Directories for new files
    - File content changes
    - AI marker directives (# AI! and # AI?)
    """

    # Regex for AI markers: # AI! (execute) and # AI? (query)
    AI_MARKER_PATTERN = re.compile(
        r'#\s*AI([!?])\s*(.*)', re.IGNORECASE
    )

    def __init__(self, cli: Any = None) -> None:
        self._cli = cli
        self._watched_paths: Dict[str, str] = {}  # path -> hash
        self._watch_dirs: List[Path] = []

    def watch_file(self, path: str) -> Any:
        """Add a file to watch."""
        path_obj = Path(path).expanduser()
        if path_obj.exists():
            self._watched_paths[str(path_obj)] = self._hash_file(path_obj)

    def watch_directory(self, path: str) -> Any:
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

    def check_ai_markers(self, path: str) -> List[AIMarker]:
        """
        Scan a file for AI directive markers.

        Looks for:
        - # AI! <instruction>  — execute directive
        - # AI? <question>     — query directive

        Args:
            path: File path to scan

        Returns:
            List of AIMarker instances found
        """
        markers = []
        try:
            path_obj = Path(path)
            if not path_obj.exists() or not path_obj.is_file():
                return markers

            with open(path_obj, 'r', errors='replace') as f:
                for line_num, line in enumerate(f, start=1):
                    match = self.AI_MARKER_PATTERN.search(line)
                    if match:
                        marker_char = match.group(1)
                        content = match.group(2).strip()
                        marker_type = "execute" if marker_char == "!" else "query"
                        markers.append(AIMarker(
                            path=str(path_obj),
                            line_number=line_num,
                            marker_type=marker_type,
                            content=content,
                        ))
        except Exception as e:
            logger.debug(f"AI marker scan failed for {path}: {e}")
        return markers

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

    async def check_with_markers(self) -> MonitorResult:
        """
        Enhanced check that returns file changes alongside AI markers.

        Returns markers found in modified files for routing to the CLI handler.
        """
        result = await self.check()
        markers = []

        # Scan modified files for AI markers
        for path_str in self._watched_paths:
            path = Path(path_str)
            if path.exists():
                file_markers = self.check_ai_markers(path_str)
                markers.extend(file_markers)

        if markers:
            marker_data = result.data or {}
            marker_data["ai_markers"] = [
                {
                    "path": m.path,
                    "line": m.line_number,
                    "type": m.marker_type,
                    "content": m.content,
                }
                for m in markers
            ]
            return MonitorResult(
                has_update=True,
                title=result.title or "AI Markers Detected",
                message=result.message or f"{len(markers)} AI marker(s) found",
                priority="high" if any(m.marker_type == "execute" for m in markers) else "normal",
                data=marker_data,
            )

        return result


class InboxMonitor:
    """
    Unified inbox monitor.

    Checks all connected channels for new messages:
    - WhatsApp
    - Telegram
    - Slack
    - Discord
    """

    def __init__(self, cli: Any = None) -> None:
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

    def __init__(self, cli: Any = None) -> None:
        self._cli = cli
        self._watched_urls: Dict[str, Dict] = {}  # url -> {selector, last_value}

    def watch_url(self, url: str, selector: str = None, description: str = '') -> Any:
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


class ClipboardWatcher:
    """
    Daemon thread that polls the system clipboard for changes.

    Tries pyperclip, then falls back to xclip/xsel/pbcopy.
    Graceful no-op on headless/CI environments.
    """

    POLL_INTERVAL = 0.5  # seconds

    def __init__(self) -> None:
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable[[str], None]] = []
        self.last_clipboard_content: str = ""
        self._read_fn: Optional[Callable[[], str]] = None
        self._setup_reader()

    def _setup_reader(self) -> Any:
        """Detect available clipboard reader."""
        # Try pyperclip
        try:
            import pyperclip
            pyperclip.paste()  # Test access
            self._read_fn = pyperclip.paste
            return
        except Exception:
            pass

        # Try xclip
        try:
            result = subprocess.run(
                ["xclip", "-selection", "clipboard", "-o"],
                capture_output=True, text=True, timeout=2,
            )
            if result.returncode == 0:
                self._read_fn = self._read_xclip
                return
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Try xsel
        try:
            result = subprocess.run(
                ["xsel", "--clipboard", "--output"],
                capture_output=True, text=True, timeout=2,
            )
            if result.returncode == 0:
                self._read_fn = self._read_xsel
                return
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Try pbpaste (macOS)
        try:
            result = subprocess.run(
                ["pbpaste"],
                capture_output=True, text=True, timeout=2,
            )
            if result.returncode == 0:
                self._read_fn = self._read_pbpaste
                return
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        logger.debug("No clipboard reader available")

    def _read_xclip(self) -> str:
        result = subprocess.run(
            ["xclip", "-selection", "clipboard", "-o"],
            capture_output=True, text=True, timeout=2,
        )
        return result.stdout if result.returncode == 0 else ""

    def _read_xsel(self) -> str:
        result = subprocess.run(
            ["xsel", "--clipboard", "--output"],
            capture_output=True, text=True, timeout=2,
        )
        return result.stdout if result.returncode == 0 else ""

    def _read_pbpaste(self) -> str:
        result = subprocess.run(
            ["pbpaste"],
            capture_output=True, text=True, timeout=2,
        )
        return result.stdout if result.returncode == 0 else ""

    def on_change(self, callback: Callable[[str], None]) -> Any:
        """Register a callback for clipboard changes."""
        self._callbacks.append(callback)

    def start(self) -> Any:
        """Start the clipboard watcher daemon thread."""
        if not self._read_fn:
            return  # No clipboard reader available

        # Seed with current content
        try:
            self.last_clipboard_content = self._read_fn()
        except Exception:
            self.last_clipboard_content = ""

        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> Any:
        """Stop the clipboard watcher."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _poll_loop(self) -> Any:
        """Background polling loop."""
        while self._running:
            try:
                current = self._read_fn()
                if current and current != self.last_clipboard_content:
                    self.last_clipboard_content = current
                    for cb in self._callbacks:
                        try:
                            cb(current)
                        except Exception as e:
                            logger.debug(f"Clipboard callback error: {e}")
            except Exception:
                pass  # Silently handle clipboard read failures
            time.sleep(self.POLL_INTERVAL)
