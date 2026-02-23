"""Backward-compatibility shim — all facade functions moved to memory_system.py."""

from .memory_system import (  # noqa: F401
    _lock,
    _resolve_memory_config,
    _singletons,
    get_brain_manager,
    get_consolidator,
    get_memory_system,
    get_rag_retriever,
    list_components,
)
