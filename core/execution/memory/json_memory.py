"""
JSON File Memory Backend
========================

Simple file-based memory with TTL.
Stores memories as JSON in ~/jotty/memory/
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class JSONMemory:
    """
    Simple JSON-based memory with TTL.

    Structure:
    ~/jotty/memory/
        {hash}.json  # One file per unique goal pattern

    Each file contains:
    {
        "goal_pattern": "...",
        "entries": [
            {
                "goal": "...",
                "result": "...",
                "success": true,
                "confidence": 0.9,
                "created_at": "2024-...",
                "expires_at": "2024-...",
            },
            ...
        ]
    }
    """

    def __init__(self, base_path: Path = None):
        """Initialize JSON memory backend."""
        if base_path is None:
            base_path = Path.home() / "jotty" / "memory"

        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"JSONMemory initialized at {self.base_path}")

    async def store(
        self,
        goal: str,
        result: str,
        success: bool = True,
        confidence: float = 1.0,
        ttl_hours: int = 24,
        **kwargs
    ):
        """
        Store memory entry.

        Args:
            goal: Task goal
            result: Execution result
            success: Whether execution succeeded
            confidence: Confidence score (0-1)
            ttl_hours: Time-to-live in hours
        """
        try:
            # Get or create file for this goal pattern
            file_path = self._get_file_path(goal)
            data = self._load_file(file_path)

            # Create entry
            entry = {
                'goal': goal,
                'result': result,
                'success': success,
                'confidence': confidence,
                'created_at': datetime.now().isoformat(),
                'expires_at': (datetime.now() + timedelta(hours=ttl_hours)).isoformat(),
            }

            # Add to entries
            data['entries'].append(entry)

            # Limit entries per file (keep last 10)
            data['entries'] = data['entries'][-10:]

            # Save
            self._save_file(file_path, data)

            logger.debug(f"Stored memory for: {goal[:50]}...")

        except Exception as e:
            logger.error(f"Failed to store memory: {e}", exc_info=True)

    async def retrieve(
        self,
        goal: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memory entries.

        Args:
            goal: Task goal to match
            limit: Maximum entries to return

        Returns:
            List of matching entries, newest first
        """
        try:
            file_path = self._get_file_path(goal)

            if not file_path.exists():
                return []

            data = self._load_file(file_path)

            # Filter expired entries
            now = datetime.now()
            valid_entries = []

            for entry in data['entries']:
                expires_at = datetime.fromisoformat(entry['expires_at'])
                if expires_at > now:
                    valid_entries.append(entry)

            # Return newest first
            valid_entries.reverse()

            # Add relevance scores (simple: exact match = 1.0, else 0.8)
            for entry in valid_entries[:limit]:
                entry['score'] = 1.0 if entry['goal'] == goal else 0.8
                entry['summary'] = f"Previous result: {entry['result'][:100]}..."

            logger.debug(f"Retrieved {len(valid_entries[:limit])} memories for: {goal[:50]}...")

            return valid_entries[:limit]

        except Exception as e:
            logger.error(f"Failed to retrieve memory: {e}", exc_info=True)
            return []

    async def clear(self):
        """Clear all memory files."""
        try:
            for file_path in self.base_path.glob("*.json"):
                file_path.unlink()
            logger.info("Cleared all memory files")
        except Exception as e:
            logger.error(f"Failed to clear memory: {e}", exc_info=True)

    def _get_file_path(self, goal: str) -> Path:
        """Get file path for goal pattern."""
        # Hash goal to create filename (groups similar goals)
        # Use first 20 chars for pattern matching
        pattern = goal[:20].lower().strip()
        file_hash = hashlib.md5(pattern.encode()).hexdigest()[:8]
        return self.base_path / f"{file_hash}.json"

    def _load_file(self, file_path: Path) -> Dict:
        """Load or create file."""
        if file_path.exists():
            with open(file_path, 'r') as f:
                return json.load(f)
        else:
            return {'entries': []}

    def _save_file(self, file_path: Path, data: Dict):
        """Save file atomically."""
        temp_path = file_path.with_suffix('.tmp')

        with open(temp_path, 'w') as f:
            json.dump(data, f, indent=2)

        temp_path.replace(file_path)
