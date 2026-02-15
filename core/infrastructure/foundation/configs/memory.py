"""Memory configuration — capacities, RAG retrieval, chunking."""

from dataclasses import dataclass
from typing import Any


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

    def __post_init__(self) -> None:
        # Capacity fields: allow zero (disabled) or positive
        _capacity_fields = {
            'episodic_capacity': self.episodic_capacity,
            'semantic_capacity': self.semantic_capacity,
            'procedural_capacity': self.procedural_capacity,
            'meta_capacity': self.meta_capacity,
            'causal_capacity': self.causal_capacity,
        }
        for name, val in _capacity_fields.items():
            if val < 0:
                raise ValueError(f"{name} must be >= 0, got {val}")

        # Strictly positive fields (non-capacity settings)
        _pos_fields = {
            'max_entry_tokens': self.max_entry_tokens,
            'rag_window_size': self.rag_window_size,
            'rag_max_candidates': self.rag_max_candidates,
            'synthesis_fetch_size': self.synthesis_fetch_size,
            'synthesis_max_tokens': self.synthesis_max_tokens,
            'chunk_size': self.chunk_size,
        }
        for name, val in _pos_fields.items():
            if val < 1:
                raise ValueError(f"{name} must be >= 1, got {val}")

        # Unit interval [0, 1]
        if not (0.0 <= self.rag_relevance_threshold <= 1.0):
            raise ValueError(
                f"rag_relevance_threshold must be in [0, 1], got {self.rag_relevance_threshold}"
            )

        # chunk_overlap must be non-negative and less than chunk_size
        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be >= 0, got {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be < chunk_size ({self.chunk_size})"
            )

        # retrieval_mode validation
        valid_modes = {"synthesize", "raw", "ranked", "discrete"}
        if self.retrieval_mode not in valid_modes:
            raise ValueError(
                f"retrieval_mode must be one of {valid_modes}, got '{self.retrieval_mode}'"
            )
