"""
DEPRECATED: jotty_core.py - Backward Compatibility Wrapper
===========================================================

This module provides backward compatibility for code using the old
JottyCore class name.

🔄 Phase 7 Refactoring: JottyCore → SingleAgentOrchestrator

New code should use:
    from Jotty.core.orchestration import SingleAgentOrchestrator

Old code will still work with deprecation warnings:
    from Jotty.core.orchestration.jotty_core import JottyCore  # Deprecated

Deprecation Timeline:
- Version 6.0 (Current): Old imports work with warnings
- Version 7.0 (Future): Old imports will be removed
"""

# Re-export SingleAgentOrchestrator as JottyCore (deprecated alias)
from .single_agent_orchestrator import (
    SingleAgentOrchestrator as JottyCore,
    create_jotty,
    create_reval,
    PersistenceManager,
)

# Re-export convenience function
create_jotty_core = create_jotty

__all__ = [
    'JottyCore',  # Deprecated alias
    'create_jotty',
    'create_jotty_core',
    'create_reval',
    'PersistenceManager',
]
