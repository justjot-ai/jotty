"""
Jotty v6.0 - LLM-Based RAG System
=================================

NO EMBEDDING MODELS - Uses PURE LLM semantic matching.
NO KEYWORD MATCHING - Works with any language, code, or unique text.

Why no embeddings:
1. Embedding models fail on tricky/nuanced text
2. They miss context-dependent meaning
3. Can't reason about relevance
4. No interpretability

Why no keywords:
1. Multilingual text breaks keyword extraction
2. Code and technical terms don't match
3. Unique phrasing gets zero matches
4. Synonyms and paraphrases missed

Our approach:
1. Recency + Value pre-ranking (no content analysis)
2. Sliding window chunking (handles large content)
3. PURE LLM semantic scoring (works for ANY text)
4. Budget-aware selection (no truncation)
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import re
import hashlib
import json

logger = logging.getLogger(__name__)

_dspy_module = None
def _get_dspy():
    global _dspy_module
    if _dspy_module is None:
        import dspy
        _dspy_module = dspy
    return _dspy_module

from ..foundation.data_structures import (
    MemoryEntry, MemoryLevel, JottyConfig,
    GoalHierarchy, GoalNode
)


# =============================================================================
# SLIDING WINDOW CHUNKER
# =============================================================================

@dataclass
class ContentChunk:
    """A chunk of content with metadata."""
    content: str
    chunk_index: int
    total_chunks: int
    source_key: str
    token_count: int
    
    # Position info
    start_char: int
    end_char: int
    
    # For reconstruction
    has_overlap_before: bool = False
    has_overlap_after: bool = False


class SlidingWindowChunker:
    """
    Chunks content using sliding window with overlap.
    
    Why sliding window:
    - Preserves context at boundaries
    - Handles variable-length content
    - Enables parallel processing
    """
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        """
        Parameters:
            chunk_size: Target tokens per chunk (approximate)
            overlap: Overlap tokens between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chars_per_token = 4  # Approximate
    
    def chunk_content(self, content: str, source_key: str) -> List[ContentChunk]:
        """
        Split content into overlapping chunks.
        """
        if not content:
            return []
        
        chunk_chars = self.chunk_size * self.chars_per_token
        overlap_chars = self.overlap * self.chars_per_token
        
        # If content fits in one chunk, return as-is
        if len(content) <= chunk_chars:
            return [ContentChunk(
                content=content,
                chunk_index=0,
                total_chunks=1,
                source_key=source_key,
                token_count=len(content) // self.chars_per_token + 1,
                start_char=0,
                end_char=len(content)
            )]
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(content):
            end = min(start + chunk_chars, len(content))
            
            # Try to break at sentence boundary
            if end < len(content):
                search_start = end - int(chunk_chars * 0.2)
                search_region = content[search_start:end]
                
                for pattern in ['. ', '.\n', '? ', '!\n', '\n\n']:
                    last_boundary = search_region.rfind(pattern)
                    if last_boundary != -1:
                        end = search_start + last_boundary + len(pattern)
                        break
            
            chunk_content = content[start:end]
            
            chunks.append(ContentChunk(
                content=chunk_content,
                chunk_index=chunk_index,
                total_chunks=-1,
                source_key=source_key,
                token_count=len(chunk_content) // self.chars_per_token + 1,
                start_char=start,
                end_char=end,
                has_overlap_before=start > 0,
                has_overlap_after=end < len(content)
            ))
            
            start = end - overlap_chars if end < len(content) else end
            chunk_index += 1
        
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return chunks


# =============================================================================
# RECENCY + VALUE PRE-RANKER (NO KEYWORDS!)
# =============================================================================

class RecencyValueRanker:
    """
    Pre-ranks memories by recency and value WITHOUT any content analysis.
    
    Why no keywords:
    - Multilingual: "日付リテラル" (date literal in Japanese) won't match "DATE"
    - Code: "def parse_date():" won't match "date parsing"
    - Unique text: Domain-specific jargon breaks keyword extraction
    - Synonyms: "revenue" vs "sales" vs "income" all mean similar things
    
    Our approach:
    - Use recency (recent memories more likely relevant)
    - Use learned value (high-value memories proved useful)
    - Use access patterns (frequently accessed = important)
    - Let LLM do ALL semantic matching
    """
    
    def __init__(self, config: JottyConfig):
        self.config = config
    
    def prerank(self, 
                memories: List[MemoryEntry],
                goal: str,
                max_candidates: int = 100) -> List[Tuple[MemoryEntry, float]]:
        """
        Pre-rank memories by recency + value + access patterns.
        
        NO content analysis - purely metadata-based ranking.
        Returns candidates for LLM scoring.
        """
        import math
        
        now = datetime.now()
        scored = []
        
        for memory in memories:
            # Recency score (exponential decay, half-life = 24 hours)
            age_hours = (now - memory.last_accessed).total_seconds() / 3600
            recency_score = math.exp(-0.029 * age_hours)  # ~0.5 at 24h
            
            # Value score (goal-conditioned if available)
            value_score = memory.get_value(goal)
            
            # Access frequency score (log scale)
            access_score = math.log(memory.access_count + 1) / 10  # Normalize
            access_score = min(1.0, access_score)
            
            # Level priority (higher levels = more refined knowledge)
            level_priority = {
                MemoryLevel.META: 1.0,
                MemoryLevel.CAUSAL: 0.95,
                MemoryLevel.SEMANTIC: 0.9,
                MemoryLevel.PROCEDURAL: 0.85,
                MemoryLevel.EPISODIC: 0.7
            }
            level_score = level_priority.get(memory.level, 0.5)
            
            # Combined score (weighted)
            # Value most important, then level, then recency, then access
            combined = (
                0.4 * value_score +
                0.25 * level_score +
                0.2 * recency_score +
                0.15 * access_score
            )
            
            scored.append((memory, combined))
        
        # Sort by score and return top candidates
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:max_candidates]


# =============================================================================
# OPTIONAL EMBEDDING PRE-FILTER (Fast ~2ms first pass)
# =============================================================================

class EmbeddingPreFilter:
    """
    Optional fast embedding pre-filter before expensive LLM scoring.

    Uses local sentence-transformers model (~2ms per query) to narrow
    down candidates before the LLM scorer runs (~500ms per batch).

    This is a performance optimization, not a replacement for LLM scoring.
    The LLM scorer still handles all the nuanced semantic matching.

    Architecture:
        1. EmbeddingPreFilter narrows N memories -> top-K candidates (~2ms)
        2. LLMRelevanceScorer scores K candidates semantically (~500ms)
        3. Combined: O(2ms + 500ms) vs O(N * 500ms) for pure LLM

    Falls back gracefully to pure LLM if embeddings unavailable.
    """

    _model = None
    _model_name = None

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", top_k: int = 50):
        """
        Args:
            model_name: Sentence-transformers model name
            top_k: Number of candidates to pass to LLM scorer
        """
        self.model_name = model_name
        self.top_k = top_k
        self._available = None  # Lazy check

    @classmethod
    def _get_model(cls, model_name: str):
        """Lazy-load sentence-transformers model (shared across instances)."""
        if cls._model is None or cls._model_name != model_name:
            try:
                from sentence_transformers import SentenceTransformer
                cls._model = SentenceTransformer(model_name)
                cls._model_name = model_name
                logger.info(f"Embedding pre-filter loaded: {model_name}")
            except ImportError:
                logger.debug(
                    "sentence-transformers not installed, embedding pre-filter disabled. "
                    "Install with: pip install sentence-transformers"
                )
                cls._model = False  # Mark as unavailable
        return cls._model if cls._model is not False else None

    @property
    def available(self) -> bool:
        """Check if embedding model is available."""
        if self._available is None:
            self._available = self._get_model(self.model_name) is not None
        return self._available

    def prefilter(self, query: str, memories: List['MemoryEntry'],
                  goal: str = "") -> List['MemoryEntry']:
        """
        Fast embedding-based pre-filter.

        Args:
            query: Search query
            memories: All candidate memories
            goal: Optional goal for context

        Returns:
            Top-K memories by embedding similarity (for LLM to refine)
        """
        if not self.available or len(memories) <= self.top_k:
            return memories  # No filtering needed

        model = self._get_model(self.model_name)
        if model is None:
            return memories

        try:
            import numpy as np

            # Encode query
            query_text = f"{query} {goal}".strip()
            query_embedding = model.encode([query_text], show_progress_bar=False)[0]

            # Encode memory contents (batch for efficiency)
            memory_texts = [m.content[:500] for m in memories]
            memory_embeddings = model.encode(memory_texts, show_progress_bar=False)

            # Cosine similarity
            query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
            memory_norms = memory_embeddings / (
                np.linalg.norm(memory_embeddings, axis=1, keepdims=True) + 1e-10
            )
            similarities = memory_norms @ query_norm

            # Get top-K indices
            top_indices = np.argsort(similarities)[-self.top_k:][::-1]
            filtered = [memories[i] for i in top_indices]

            logger.debug(
                f"Embedding pre-filter: {len(memories)} -> {len(filtered)} candidates "
                f"(top similarity: {similarities[top_indices[0]]:.3f})"
            )
            return filtered

        except Exception as e:
            logger.debug(f"Embedding pre-filter failed, using all memories: {e}")
            return memories


# =============================================================================
# LLM RELEVANCE SCORER (Pure Semantic - No Keywords!)
# =============================================================================

_RelevanceSignature = None
def _get_relevance_signature():
    global _RelevanceSignature
    if _RelevanceSignature is None:
        dspy = _get_dspy()
        class RelevanceSignature(dspy.Signature):
            """Score relevance of memories to a query using pure semantic understanding."""
            query: str = dspy.InputField(desc="The current query/goal - can be ANY language or format")
            memory_batch: str = dspy.InputField(desc="Batch of memories to score (JSON)")
            context_hints: str = dspy.InputField(desc="Additional context about the task domain")
            reasoning: str = dspy.OutputField(desc="Step-by-step semantic analysis of each memory's relevance")
            scores: str = dspy.OutputField(desc="JSON dict mapping memory_key to relevance score 0.0-1.0")
        _RelevanceSignature = RelevanceSignature
    return _RelevanceSignature


_MemorySynthesisSignature = None
def _get_memory_synthesis_signature():
    global _MemorySynthesisSignature
    if _MemorySynthesisSignature is None:
        dspy = _get_dspy()
        class MemorySynthesisSignature(dspy.Signature):
            """Brain-Inspired Memory Synthesis (Neuroscience-Aligned)."""
            query: str = dspy.InputField(desc="Current task/question")
            goal: str = dspy.InputField(desc="Overall goal context")
            memories: str = dspy.InputField(desc="All relevant memories (JSON list with level, content, value)")
            context_hints: str = dspy.InputField(desc="Additional context about task domain")
            analysis: str = dspy.OutputField(desc="Deep analysis: patterns across memories, contradictions, causal relationships")
            core_principles: str = dspy.OutputField(desc="Key principles synthesized from experiences (what we learned)")
            anti_patterns: str = dspy.OutputField(desc="What to avoid based on failures and mistakes")
            recommendations: str = dspy.OutputField(desc="Specific actionable recommendations for current task")
            wisdom: str = dspy.OutputField(desc="Integrated wisdom - coherent schema combining all insights")
        _MemorySynthesisSignature = MemorySynthesisSignature
    return _MemorySynthesisSignature


class LLMRelevanceScorer:
    """
    PURE LLM semantic scoring - NO keywords, NO embeddings.
    
    Works with:
    - Any language (English, Japanese, Chinese, mixed)
    - Code snippets (Python, SQL, any syntax)
    - Technical jargon and domain-specific terms
    - Unique phrasing and synonyms
    - Metaphors and indirect references
    
    Examples of what keywords CANNOT handle but LLM CAN:
    
    1. Multilingual:
       Query: "日付の処理方法" (How to handle dates in Japanese)
       Memory: "Use DATE literal for Trino"
       Keywords: ZERO match
       LLM: 0.95 (understands both are about date handling)
    
    2. Code:
       Query: "def process_timestamp():"
       Memory: "DateTime columns need explicit casting"
       Keywords: ZERO match
       LLM: 0.85 (understands timestamp relates to DateTime)
    
    3. Synonyms:
       Query: "revenue by territory"
       Memory: "sales aggregation by region"
       Keywords: ZERO match
       LLM: 0.90 (revenue≈sales, territory≈region)
    
    4. Indirect:
       Query: "why is my query slow?"
       Memory: "Large tables need partition filters"
       Keywords: ZERO match
       LLM: 0.80 (understands missing filters cause slowness)
    """
    
    def __init__(self, config: JottyConfig):
        self.config = config
        self.window_size = config.rag_window_size
        self.use_cot = config.rag_use_cot
        
        # Create the scorer agent (DSPy loaded on first use)
        dspy = _get_dspy()
        sig = _get_relevance_signature()
        if self.use_cot:
            self.scorer = dspy.ChainOfThought(sig)
        else:
            self.scorer = dspy.Predict(sig)
    
    def score_batch(self, query: str, memories: List[MemoryEntry],
                    context_hints: str = "") -> Dict[str, float]:
        """
        Score memories using PURE LLM semantic understanding.
        
        No keyword matching - the LLM understands meaning across:
        - Languages
        - Coding styles
        - Domain jargon
        - Synonyms and paraphrases
        """
        import json
        
        all_scores = {}
        
        # Process in windows
        for i in range(0, len(memories), self.window_size):
            batch = memories[i:i + self.window_size]
            
            # Format batch as JSON for LLM
            batch_data = []
            for mem in batch:
                batch_data.append({
                    "key": mem.key,
                    "content": mem.content,
                    "level": mem.level.value,
                    "value": round(mem.default_value, 2)
                })
            
            # Build context hints for semantic matching
            semantic_hints = context_hints or ""
            semantic_hints += """

IMPORTANT: Score based on SEMANTIC meaning, not keyword overlap.
- Consider synonyms (revenue = sales = income)
- Consider translations (日付 = date = fecha)
- Consider indirect relevance (slow query → missing index)
- Consider code meaning (def parse_date → date handling)

Score 0.9-1.0: Directly answers or highly relevant
Score 0.7-0.8: Related and useful context
Score 0.4-0.6: Tangentially related
Score 0.1-0.3: Weak connection
Score 0.0: Completely unrelated
"""
            
            try:
                result = self.scorer(
                    query=query,
                    memory_batch=json.dumps(batch_data, indent=2, ensure_ascii=False),
                    context_hints=semantic_hints
                )
                
                # Parse scores from response
                scores_text = result.scores
                
                try:
                    if "```" in scores_text:
                        match = re.search(r'```(?:json)?\s*(.*?)\s*```', scores_text, re.S)
                        if match:
                            scores_text = match.group(1)
                    
                    batch_scores = json.loads(scores_text)
                    
                    # Normalize scores
                    for key, score in batch_scores.items():
                        if isinstance(score, (int, float)):
                            all_scores[key] = max(0.0, min(1.0, float(score)))
                        else:
                            all_scores[key] = 0.5
                            
                except json.JSONDecodeError:
                    # Fallback: extract key-value pairs
                    for mem in batch:
                        match = re.search(rf'{mem.key}["\s:]+(\d+\.?\d*)', scores_text)
                        if match:
                            all_scores[mem.key] = float(match.group(1))
                        else:
                            all_scores[mem.key] = 0.5
                
            except Exception as e:
                # On error, use neutral score
                for mem in batch:
                    all_scores[mem.key] = 0.5
        
        return all_scores


# =============================================================================
# MEMORY SYNTHESIZER (Brain-Inspired!)
# =============================================================================

class MemorySynthesizer:
    """
    🧠 Neuroscience-Aligned Memory Synthesis

    How human memory ACTUALLY works:
    - Hippocampus: Pattern completion across related memories
    - Neocortex: Consolidates into coherent schemas
    - Working memory: Receives integrated wisdom, not raw episodes

    What this does:
    1. Fetches broadly (like hippocampal pattern completion)
    2. Synthesizes into coherent wisdom (like neocortical consolidation)
    3. Returns integrated schema (like semantic memory)

    vs Old Approach:
    - Old: Fetch 10 discrete memories → agent synthesizes
    - New: Fetch 200 memories → LLM synthesizes → agent gets wisdom

    The synthesis creates NEW knowledge that doesn't exist in any single memory!
    """

    def __init__(self, config: JottyConfig):
        self.config = config
        self.synthesizer = _get_dspy().ChainOfThought(_get_memory_synthesis_signature())

    def synthesize_wisdom(self,
                          query: str,
                          goal: str,
                          memories: List[MemoryEntry],
                          context_hints: str = "") -> str:
        """
        Synthesize wisdom from multiple memories.

        This is MORE intelligent than discrete retrieval because:
        1. Finds emergent patterns across memories
        2. Resolves contradictions
        3. Creates causal insights
        4. Integrates knowledge into coherent schema

        Returns:
            Synthesized wisdom as a coherent text (NOT discrete memories!)
        """
        if not memories:
            return ""

        # Format memories for LLM synthesis
        memory_data = []
        for mem in memories:
            memory_data.append({
                "level": mem.level.value,
                "content": mem.content[:500],  # Limit per-memory content
                "value": round(mem.default_value, 2),
                "access_count": mem.access_count
            })

        memories_json = json.dumps(memory_data, indent=2, ensure_ascii=False)

        try:
            result = self.synthesizer(
                query=query,
                goal=goal,
                memories=memories_json,
                context_hints=context_hints
            )

            # Build synthesized wisdom text
            wisdom_parts = []

            if hasattr(result, 'core_principles') and result.core_principles:
                wisdom_parts.append(f"**Core Principles:**\n{result.core_principles}")

            if hasattr(result, 'anti_patterns') and result.anti_patterns:
                wisdom_parts.append(f"\n**Anti-Patterns (What to Avoid):**\n{result.anti_patterns}")

            if hasattr(result, 'recommendations') and result.recommendations:
                wisdom_parts.append(f"\n**Recommendations:**\n{result.recommendations}")

            if hasattr(result, 'wisdom') and result.wisdom:
                wisdom_parts.append(f"\n**Integrated Wisdom:**\n{result.wisdom}")

            synthesized = "\n".join(wisdom_parts)

            # Trim to max tokens (rough estimate: 4 chars/token)
            max_chars = self.config.synthesis_max_tokens * 4
            if len(synthesized) > max_chars:
                synthesized = synthesized[:max_chars] + "\n...(truncated for context limits)"

            logger.debug(f"🧠 Synthesized {len(memories)} memories into {len(synthesized)} char wisdom")

            return synthesized

        except Exception as e:
            logger.warning(f"⚠️  Synthesis failed: {e}, falling back to discrete concatenation")

            # Fallback: Simple concatenation of top memories
            fallback = []
            for mem in memories[:5]:  # Top 5 only
                fallback.append(f"- [{mem.level.value}] {mem.content[:200]}")

            return "\n".join(fallback)


# =============================================================================
# DEDUPLICATION ENGINE (Shannon Enhancement)
# =============================================================================

_DeduplicationSignature = None
def _get_dedup_signature():
    global _DeduplicationSignature
    if _DeduplicationSignature is None:
        dspy = _get_dspy()
        class DeduplicationSignature(dspy.Signature):
            """Check if two memories are semantically similar enough to merge."""
            memory_a: str = dspy.InputField(desc="First memory content")
            memory_b: str = dspy.InputField(desc="Second memory content")
            context: str = dspy.InputField(desc="Context about memory purpose")
            reasoning: str = dspy.OutputField(desc="Analysis of similarity")
            is_duplicate: bool = dspy.OutputField(desc="True if memories convey same information")
            similarity_score: float = dspy.OutputField(desc="Similarity 0.0-1.0")
            merged_content: str = dspy.OutputField(desc="If duplicate, the merged content preserving all info")
        _DeduplicationSignature = DeduplicationSignature
    return _DeduplicationSignature


class DeduplicationEngine:
    """
    LLM-based deduplication to reduce memory redundancy.

    Why LLM over string similarity:
    1. Understands semantic equivalence
    2. Can merge information intelligently
    3. Preserves unique details from both
    """

    # Set to True to skip expensive LLM deduplication calls (use hash-only)
    SKIP_LLM_DEDUP = True

    def __init__(self, config: JottyConfig):
        self.config = config
        self.threshold = config.similarity_threshold
        self.checker = _get_dspy().ChainOfThought(_get_dedup_signature()) if not self.SKIP_LLM_DEDUP else None
    
    def check_duplicate(self, mem_a: MemoryEntry, mem_b: MemoryEntry) -> Tuple[bool, float, str]:
        """
        Check if two memories are duplicates.

        Returns:
            (is_duplicate, similarity, merged_content)
        """
        # Quick hash check first
        if mem_a.content_hash == mem_b.content_hash:
            return True, 1.0, mem_a.content

        # Quick length check (very different lengths unlikely duplicates)
        len_ratio = min(len(mem_a.content), len(mem_b.content)) / max(len(mem_a.content), len(mem_b.content))
        if len_ratio < 0.3:
            return False, len_ratio, ""

        # Skip expensive LLM calls if configured
        if self.SKIP_LLM_DEDUP or self.checker is None:
            return False, len_ratio, ""

        try:
            result = self.checker(
                memory_a=mem_a.content,
                memory_b=mem_b.content,
                context=f"Memory level: {mem_a.level.value}, Agent: {mem_a.source_agent}"
            )

            is_dup = result.is_duplicate
            sim = float(result.similarity_score) if result.similarity_score else 0.0
            merged = result.merged_content if is_dup else ""

            return is_dup and sim >= self.threshold, sim, merged

        except Exception:
            return False, 0.0, ""
    
    def deduplicate_batch(self, memories: List[MemoryEntry]) -> List[MemoryEntry]:
        """
        Remove duplicates from a batch of memories.
        
        Returns deduplicated list with merged content.
        
        🔥 A-TEAM FIX: Skip deduplication for very large memories (>10KB)
        to prevent LLM context overflow and 400 Bad Request errors!
        """
        if len(memories) <= 1:
            return memories
        
        # 🔥 A-TEAM: Filter out huge memories that would cause LLM timeout
        MAX_MEMORY_SIZE = 10000  # 10KB limit
        normal_memories = []
        large_memories = []
        
        for mem in memories:
            if len(mem.content) > MAX_MEMORY_SIZE:
                large_memories.append(mem)
                logger.info(f"📦 Memory too large ({len(mem.content)} chars), skipping dedup for: {mem.key[:50]}...")
            else:
                normal_memories.append(mem)
        
        # Only deduplicate normal-sized memories
        if len(normal_memories) <= 1:
            return normal_memories + large_memories
        
        # Group by content hash first (exact duplicates)
        hash_groups: Dict[str, List[MemoryEntry]] = {}
        for mem in normal_memories:
            if mem.content_hash not in hash_groups:
                hash_groups[mem.content_hash] = []
            hash_groups[mem.content_hash].append(mem)
        
        # Keep one from each hash group (highest value)
        unique = []
        for group in hash_groups.values():
            best = max(group, key=lambda m: m.default_value)
            unique.append(best)
        
        # Now check semantic duplicates (more expensive)
        if len(unique) <= 1:
            return unique
        
        # Compare pairs (O(n²) but n is small after hash dedup)
        to_remove = set()
        for i, mem_a in enumerate(unique):
            if mem_a.key in to_remove:
                continue
            for j, mem_b in enumerate(unique[i+1:], i+1):
                if mem_b.key in to_remove:
                    continue
                
                is_dup, sim, merged = self.check_duplicate(mem_a, mem_b)
                
                if is_dup:
                    # Merge into the one with higher value
                    if mem_a.default_value >= mem_b.default_value:
                        mem_a.content = merged
                        mem_a.similar_entries.append(mem_b.key)
                        to_remove.add(mem_b.key)
                    else:
                        mem_b.content = merged
                        mem_b.similar_entries.append(mem_a.key)
                        to_remove.add(mem_a.key)
        
        # Return deduplicated normal memories + all large memories
        return [m for m in unique if m.key not in to_remove] + large_memories


# =============================================================================
# CAUSAL EXTRACTOR (Aristotle Enhancement)
# =============================================================================

_CausalExtractionSignature = None
def _get_causal_signature():
    global _CausalExtractionSignature
    if _CausalExtractionSignature is None:
        dspy = _get_dspy()
        class CausalExtractionSignature(dspy.Signature):
            """Extract causal relationships from episode experiences."""
            success_episodes: str = dspy.InputField(desc="Summary of successful episodes")
            failure_episodes: str = dspy.InputField(desc="Summary of failed episodes")
            domain_context: str = dspy.InputField(desc="Domain information")
            reasoning: str = dspy.OutputField(desc="Analysis of what caused success vs failure")
            causal_links: str = dspy.OutputField(desc="JSON list of {cause, effect, conditions, confidence}")
        _CausalExtractionSignature = CausalExtractionSignature
    return _CausalExtractionSignature


class CausalExtractor:
    """
    Extracts causal knowledge (WHY things work) from experiences.
    
    Moves beyond correlation to causation:
    - "DATE literal needed" (correlation)
    - "Trino parser requires type annotation for date columns" (causation)
    """
    
    def __init__(self, config: JottyConfig):
        self.config = config
        self.extractor = _get_dspy().ChainOfThought(_get_causal_signature())
        self.min_evidence = config.causal_min_support
    
    def extract_from_episodes(self, 
                               success_episodes: List[Dict],
                               failure_episodes: List[Dict],
                               domain: str = "sql") -> List[Dict]:
        """
        Extract causal links from contrasting episodes.
        """
        import json
        
        # Format episodes
        success_summary = self._summarize_episodes(success_episodes)
        failure_summary = self._summarize_episodes(failure_episodes)
        
        try:
            result = self.extractor(
                success_episodes=success_summary,
                failure_episodes=failure_summary,
                domain_context=f"Domain: {domain}. Extract root causes, not just correlations."
            )
            
            # Parse causal links
            links_text = result.causal_links
            
            # Handle markdown
            if "```" in links_text:
                match = re.search(r'```(?:json)?\s*(.*?)\s*```', links_text, re.S)
                if match:
                    links_text = match.group(1)
            
            links = json.loads(links_text)
            
            # Filter by confidence
            return [
                link for link in links 
                if link.get('confidence', 0) >= self.config.causal_confidence_threshold
            ]
            
        except Exception as e:
            return []
    
    def _summarize_episodes(self, episodes: List[Dict]) -> str:
        """Create summary of episodes for causal analysis."""
        if not episodes:
            return "No episodes"
        
        summaries = []
        for ep in episodes:  # Limit for context
            summary = f"Episode {ep.get('id', '?')}: "
            summary += f"Query: {ep.get('query', '?')}. "
            summary += f"Result: {ep.get('result', '?')}"
            summaries.append(summary)
        
        return "\n".join(summaries)


# =============================================================================
# MAIN LLM RAG RETRIEVER (PURE SEMANTIC)
# =============================================================================

class LLMRAGRetriever:
    """
    PURE LLM-based RAG - NO keywords, NO embeddings.
    
    Pipeline:
    1. Recency + Value pre-ranking (metadata only, no content analysis)
    2. Sliding window chunking (if needed)
    3. PURE LLM semantic scoring (works for ANY text/language)
    4. Deduplication (Shannon optimization)
    5. Budget-aware selection (no truncation)
    6. Goal-conditioned value blending (RL integration)
    
    Works with:
    - English, Japanese, Chinese, Spanish, any language
    - Mixed language text
    - Code (Python, SQL, JavaScript, etc.)
    - Technical documentation
    - Unique phrasing
    - Domain-specific jargon
    """
    
    def __init__(self, config: JottyConfig, use_embedding_prefilter: bool = True):
        self.config = config
        
        self.chunker = SlidingWindowChunker(
            chunk_size=config.chunk_size,
            overlap=config.chunk_overlap
        )
        
        # NO keyword filter - use recency/value ranking instead
        self.preranker = RecencyValueRanker(config)

        # Optional embedding pre-filter (fast ~2ms first pass)
        # Narrows candidates before expensive LLM scoring
        self.embedding_prefilter = None
        if use_embedding_prefilter:
            self.embedding_prefilter = EmbeddingPreFilter(
                top_k=min(50, config.rag_max_candidates)
            )
        
        # Pure LLM semantic scoring (second pass on pre-filtered candidates)
        self.scorer = LLMRelevanceScorer(config)

        # Brain-Inspired Memory Synthesis
        self.synthesizer = MemorySynthesizer(config)

        # Deduplication
        self.deduplicator = DeduplicationEngine(config)
    
    def retrieve(self,
                 query: str,
                 goal: str,
                 memories: List[MemoryEntry],
                 budget_tokens: int,
                 goal_hierarchy: Optional[GoalHierarchy] = None,
                 context_hints: str = "") -> List[MemoryEntry]:
        """
        Retrieve relevant memories using PURE LLM semantic matching.
        
        NO keyword matching - works for any language or content type.
        
        Parameters:
            query: The current query text (ANY language)
            goal: The goal for value lookup
            memories: All available memories
            budget_tokens: Token budget for memories
            goal_hierarchy: Optional hierarchy for knowledge transfer
            context_hints: Additional context for LLM scorer
        
        Returns:
            List of relevant memories that fit within budget
        """
        if not memories:
            return []
        
        # Step 1: Pre-rank by recency + value (NO content analysis)
        candidates = self.preranker.prerank(
            memories=memories,
            goal=goal,
            max_candidates=self.config.rag_max_candidates
        )
        
        if not candidates:
            return []
        
        # Step 1.5: Optional embedding pre-filter (fast ~2ms)
        # Narrows candidates before expensive LLM scoring
        candidate_memories = [m for m, _ in candidates]
        if (self.embedding_prefilter and self.embedding_prefilter.available
                and len(candidate_memories) > self.embedding_prefilter.top_k):
            candidate_memories = self.embedding_prefilter.prefilter(
                query=query,
                memories=candidate_memories,
                goal=goal
            )
        
        # Step 2: PURE LLM semantic scoring (on pre-filtered candidates)
        relevance_scores = self.scorer.score_batch(
            query=query,
            memories=candidate_memories,
            context_hints=context_hints
        )
        
        # Step 3: Combine scores (relevance + value)
        scored_memories = []
        for memory in candidate_memories:
            relevance = relevance_scores.get(memory.key, 0.5)
            
            # Get value with optional transfer
            if goal_hierarchy and self.config.enable_goal_hierarchy:
                value = memory.get_value_with_transfer(
                    goal=goal,
                    goal_hierarchy=goal_hierarchy,
                    transfer_weight=self.config.goal_transfer_weight
                )
            else:
                value = memory.get_value(goal)
            
            # Combined score: relevance primary, value secondary
            combined = 0.7 * relevance + 0.3 * value
            
            if combined >= self.config.rag_relevance_threshold:
                scored_memories.append((memory, combined))
        
        # Sort by combined score
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        
        # Step 4: Deduplication
        if self.config.enable_deduplication:
            unique_memories = self.deduplicator.deduplicate_batch(
                [m for m, _ in scored_memories]
            )
        else:
            unique_memories = [m for m, _ in scored_memories]
        
        # Step 5: Budget-aware selection (NO TRUNCATION)
        selected = []
        tokens_used = 0
        
        for memory in unique_memories:
            if tokens_used + memory.token_count <= budget_tokens:
                selected.append(memory)
                tokens_used += memory.token_count
        
        return selected

    def retrieve_and_synthesize(self,
                                 query: str,
                                 goal: str,
                                 memories: List[MemoryEntry],
                                 context_hints: str = "") -> str:
        """
        🧠 Brain-Inspired Retrieval: Fetch broadly + Synthesize wisdom

        How this is MORE intelligent than discrete retrieval:
        1. Fetches comprehensively (200 memories vs 10)
        2. Finds emergent patterns across memories
        3. Resolves contradictions
        4. Creates causal insights
        5. Returns integrated schema (not discrete facts)

        This is neuroscience-aligned:
        - Hippocampus: Pattern completion (broad fetch)
        - Neocortex: Consolidation (synthesis)
        - Working memory: Receives wisdom (not raw episodes)

        Returns:
            Synthesized wisdom as coherent text (NOT list of memories!)
        """
        if not memories:
            return ""

        # Step 1: Broad fetch (like hippocampal pattern completion)
        # Use MORE candidates than discrete mode
        candidates = self.preranker.prerank(
            memories=memories,
            goal=goal,
            max_candidates=self.config.synthesis_fetch_size  # 200 vs 50!
        )

        if not candidates:
            return ""

        candidate_memories = [m for m, _ in candidates]

        logger.info(f"🧠 Synthesis mode: Fetched {len(candidate_memories)} memories for synthesis")

        # Step 2: Synthesize (like neocortical consolidation)
        synthesized_wisdom = self.synthesizer.synthesize_wisdom(
            query=query,
            goal=goal,
            memories=candidate_memories,
            context_hints=context_hints
        )

        return synthesized_wisdom

    def retrieve_with_chunks(self,
                             query: str,
                             goal: str,
                             large_content: str,
                             source_key: str,
                             budget_tokens: int) -> List[ContentChunk]:
        """
        Retrieve relevant chunks from large content using LLM scoring.
        """
        chunks = self.chunker.chunk_content(large_content, source_key)
        
        if not chunks:
            return []
        
        # Create temporary MemoryEntry objects for scoring
        temp_memories = []
        for chunk in chunks:
            temp_mem = MemoryEntry(
                key=f"{source_key}_chunk_{chunk.chunk_index}",
                content=chunk.content,
                level=MemoryLevel.EPISODIC,
                context={},
                token_count=chunk.token_count
            )
            temp_memories.append(temp_mem)
        
        # Score with LLM
        scores = self.scorer.score_batch(query, temp_memories)
        
        # Sort chunks by score
        chunk_scores = [(chunk, scores.get(f"{source_key}_chunk_{chunk.chunk_index}", 0.5)) 
                        for chunk in chunks]
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select within budget
        selected = []
        tokens_used = 0
        
        for chunk, score in chunk_scores:
            if tokens_used + chunk.token_count <= budget_tokens:
                selected.append(chunk)
                tokens_used += chunk.token_count
        
        # Sort selected by original order for coherence
        selected.sort(key=lambda c: c.chunk_index)
        
        return selected
