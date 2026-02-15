"""
No-Op Memory Backend
====================

For testing or when memory is disabled.
"""

from typing import Any, Dict, List


class NoOpMemory:
    """Memory backend that does nothing."""

    async def store(self, **kwargs: Any) -> Any:
        """Store nothing."""
        pass

    async def retrieve(self, goal: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve nothing."""
        return []

    async def clear(self) -> Any:
        """Clear nothing."""
        pass
