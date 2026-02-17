"""
Persistence Layer - Data Persistence & Session Management
========================================================

Session management, shared context, and persistence.

Modules:
--------
- persistence: Data persistence
- session_manager: Session management
- shared_context: Shared context across sessions
"""

from .session_manager import SessionManager
from .shared_context import SharedContext
from .vault import Vault

__all__ = [
    "Vault",
    "SessionManager",
    "SharedContext",
]
