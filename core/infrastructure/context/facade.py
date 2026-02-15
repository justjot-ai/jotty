"""
Context Subsystem Facade
=========================

Clean, discoverable API for context management components.
No new business logic — just imports + convenience accessors.

UNIFIED ARCHITECTURE:
- SmartContextManager now includes ALL features:
  - Priority-based budgeting (from context_manager.py)
  - Overflow detection (from global_context_guard.py)
  - Function wrapping (from global_context_guard.py)
  - DSPy patching (from global_context_guard.py)

Usage:
    from Jotty.core.infrastructure.context.facade import get_context_manager

    ctx = get_context_manager()
"""

from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from Jotty.core.infrastructure.context.context_manager import SmartContextManager
    from Jotty.core.infrastructure.context.content_gate import ContentGate


def get_context_manager() -> 'SmartContextManager':
    """
    Return a SmartContextManager (unified context management).

    Features:
    - Priority-based context budgeting
    - Overflow detection and recovery
    - Function wrapping with auto-retry
    - DSPy integration support

    Returns:
        SmartContextManager instance with all features.
    """
    from Jotty.core.infrastructure.context.context_manager import SmartContextManager
    return SmartContextManager()


def get_context_guard() -> 'SmartContextManager':
    """
    Return a context guard (alias for get_context_manager).

    DEPRECATED: Use get_context_manager() instead.
    Kept for backwards compatibility - returns same unified manager.

    Returns:
        SmartContextManager instance (same as get_context_manager).
    """
    return get_context_manager()


def get_content_gate() -> 'ContentGate':
    """
    Return a ContentGate for relevance-based content filtering.

    Returns:
        ContentGate instance.
    """
    from Jotty.core.infrastructure.context.content_gate import ContentGate
    return ContentGate()


def list_components() -> Dict[str, str]:
    """
    List all context subsystem components with descriptions.

    Returns:
        Dict mapping component name to description.
    """
    return {
        "SmartContextManager": "UNIFIED: Priority budgeting + overflow detection + function wrapping + DSPy patching",
        "ContentGate": "Relevance filtering before context injection",
        "AgenticCompressor": "LLM-based compression with Shapley impact prioritization",
        "ContextChunker": "LLM-based semantic chunking for large inputs",
        "ContextGradient": "Context-as-gradient learning for cooperation",
        "ContextApplier": "Applies context gradients to agent prompts",
    }
