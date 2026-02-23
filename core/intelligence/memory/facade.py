"""Backward-compatibility shim — all facade functions moved to memory_system.py."""

from .document_ingestion import (  # noqa: F401
    DocumentIngestionPipeline,
    get_ingestion_pipeline,
)
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
from .vector_store import (  # noqa: F401
    DocumentChunk,
    RetrievalResult,
    VectorStore,
    get_vector_store,
)
