"""
Context Subsystem Facade
=========================

Clean, discoverable API for context management components.
No new business logic — just imports + convenience accessors.

Usage:
    from Jotty.core.context.facade import get_context_manager, list_components

    ctx = get_context_manager()
"""

from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from Jotty.core.context.context_manager import SmartContextManager
    from Jotty.core.context.global_context_guard import GlobalContextGuard
    from Jotty.core.context.content_gate import ContentGate


def get_context_manager() -> 'SmartContextManager':
    """
    Return a SmartContextManager for auto-chunking and compression.

    Returns:
        SmartContextManager instance.
    """
    from Jotty.core.context.context_manager import SmartContextManager
    return SmartContextManager()


def get_context_guard() -> 'GlobalContextGuard':
    """
    Return a GlobalContextGuard for context overflow protection.

    Returns:
        GlobalContextGuard instance.
    """
    from Jotty.core.context.global_context_guard import GlobalContextGuard
    return GlobalContextGuard()


def get_content_gate() -> 'ContentGate':
    """
    Return a ContentGate for relevance-based content filtering.

    Returns:
        ContentGate instance.
    """
    from Jotty.core.context.content_gate import ContentGate
    return ContentGate()


def list_components() -> Dict[str, str]:
    """
    List all context subsystem components with descriptions.

    Returns:
        Dict mapping component name to description.
    """
    return {
        "SmartContextManager": "Auto-chunking and compression coordinator",
        "GlobalContextGuard": "Global context overflow prevention with budget enforcement",
        "ContentGate": "Relevance filtering before context injection",
        "LLMContextManager": "Context overflow prevention for LLM calls",
        "AgenticCompressor": "LLM-based context compression",
        "ContextChunker": "LLM-based chunking for large inputs",
        "ContextGradient": "Context-as-gradient learning for cooperation",
        "ContextApplier": "Applies context gradients to agent prompts",
    }
