"""Memory configuration — capacities, RAG retrieval, chunking."""

from dataclasses import dataclass


@dataclass
class MemoryConfig:
    """Memory capacities and RAG retrieval settings."""
    episodic_capacity: int = 1000
    semantic_capacity: int = 500
    procedural_capacity: int = 200
    meta_capacity: int = 100
    causal_capacity: int = 150
    max_entry_tokens: int = 2000
    enable_llm_rag: bool = True
    rag_window_size: int = 5
    rag_max_candidates: int = 50
    rag_relevance_threshold: float = 0.6
    rag_use_cot: bool = True
    retrieval_mode: str = "synthesize"
    synthesis_fetch_size: int = 200
    synthesis_max_tokens: int = 800
    chunk_size: int = 500
    chunk_overlap: int = 50
