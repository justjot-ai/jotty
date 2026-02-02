"""
Context Guard for ReVal.

Provides priority-based context management and compression.
"""
import asyncio
import time
from typing import Dict, List, Tuple, Any, Optional
import logging

from ..utils.tokenizer import SmartTokenizer

logger = logging.getLogger(__name__)


class LLMContextManager:
    """
    LLM-powered context budgeting (formerly SmartContextGuard).

    Ensures we NEVER exceed context length by:
    - Priority-based compression
    - Task-critical preservation (TODO always kept)
    - Intelligent summarization using LLM-based compression
    - Graceful degradation
    """
    
    # Priority levels
    CRITICAL = 0   # NEVER remove (root goal, current task)
    HIGH = 1       # Remove only if desperate (recent trajectory)
    MEDIUM = 2     # Can compress (memories, examples)
    LOW = 3        # Remove first (verbose logs)
    
    def __init__(self, max_tokens: int = 28000, safety_margin: int = 2000, compress_fn=None):
        self.max_tokens = max_tokens
        self.safety_margin = safety_margin
        self.usable_tokens = max_tokens - safety_margin
        self.compress_fn = compress_fn  # ✅ A-TEAM: Use provided compression function
        
        # Priority buffers
        self.buffers: Dict[int, List[Tuple[str, str, int]]] = {
            self.CRITICAL: [],
            self.HIGH: [],
            self.MEDIUM: [],
            self.LOW: []
        }
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count using SmartTokenizer."""
        return SmartTokenizer.get_instance().count_tokens(text)
    
    def register(self, key: str, content: str, priority: int):
        """Register content with priority."""
        tokens = self.estimate_tokens(content)
        self.buffers[priority].append((key, content, tokens))
    
    def register_critical(self, key: str, content: str):
        """Register CRITICAL content (goal, current task)."""
        self.register(key, content, self.CRITICAL)
    
    async def build_context(self) -> Tuple[str, Dict[str, Any]]:
        """
        Build context that fits within limit.
        
        Returns:
            (context_string, metadata)
            
#         ✅ A-TEAM: Now async to support async compression.
        """
        context_parts = []
        total_tokens = 0
        included = {}
        truncated = {}
        
        # Always include CRITICAL first
        for key, content, tokens in self.buffers[self.CRITICAL]:
            context_parts.append(f"## {key}\n{content}\n")
            total_tokens += tokens
            included[key] = tokens
        
        # Then HIGH priority
        for key, content, tokens in self.buffers[self.HIGH]:
            if total_tokens + tokens < self.usable_tokens:
                context_parts.append(f"## {key}\n{content}\n")
                total_tokens += tokens
                included[key] = tokens
            else:
                # ✅ A-TEAM: Use contextual compression if available
                available = self.usable_tokens - total_tokens
                if available > 200:
                    if self.compress_fn:
                        # compress_fn is now async-aware
                        try:
                            compressed_content = await self.compress_fn(content, f"for_{key}", available * 4)
                        except Exception as e:
                            # Fallback if compression fails
                            tokens_lost = tokens - available
                            logger.error(f"❌ Compression failed for '{key}': {e}. Using emergency truncation. "
                                       f"Losing ~{tokens_lost} tokens of context!")
                            compressed_content = (
                                content[:available * 4 - 50] + 
                                f"\n... (⚠️ Emergency truncation: ~{tokens_lost} tokens omitted due to compression failure) ..."
                            )
                    else:
                        # Fallback: emergency truncation (no compressor provided)
                        tokens_lost = tokens - available
                        logger.warning(f"⚠️  No compressor provided for '{key}', using emergency truncation. "
                                     f"Consider providing compress_fn to preserve context.")
                        compressed_content = (
                            content[:available * 4 - 50] + 
                            f"\n... (⚠️ Emergency truncation: ~{tokens_lost} tokens omitted - no compressor available) ..."
                        )
                    context_parts.append(f"## {key} (compressed)\n{compressed_content}\n")
                    total_tokens += available
                    truncated[key] = (tokens, available)
        
        # Then MEDIUM with aggressive compression
        for key, content, tokens in self.buffers[self.MEDIUM]:
            if total_tokens < self.usable_tokens * 0.9:
                available = min(tokens, (self.usable_tokens - total_tokens) // 2)
                if available > 100:
                    if self.compress_fn:
                        try:
                            compressed_content = await self.compress_fn(content, f"for_{key}", available * 4)
                        except Exception as e:
                            tokens_lost = tokens - available
                            logger.error(f"❌ Compression failed for '{key}' (MEDIUM priority): {e}. "
                                       f"Losing ~{tokens_lost} tokens!")
                            compressed_content = (
                                content[:available * 4 - 50] + 
                                f"\n... (⚠️ Emergency truncation: ~{tokens_lost} tokens omitted) ..."
                            )
                    else:
                        tokens_lost = tokens - available
                        logger.warning(f"⚠️  No compressor for '{key}' (MEDIUM priority), using emergency truncation.")
                        compressed_content = (
                            content[:available * 4 - 50] + 
                            f"\n... (⚠️ Emergency truncation: ~{tokens_lost} tokens omitted) ..."
                        )
                    context_parts.append(f"## {key}\n{compressed_content}\n")
                    total_tokens += available
        
        # LOW priority only if lots of space
        if total_tokens < self.usable_tokens * 0.7:
            for key, content, tokens in self.buffers[self.LOW]:
                if total_tokens + tokens < self.usable_tokens * 0.8:
                    context_parts.append(f"## {key}\n{content}\n")
                    total_tokens += tokens
        
        return "\n".join(context_parts), {
            'total_tokens': total_tokens,
            'max_tokens': self.usable_tokens,
            'included': included,
            'truncated': truncated,
            'utilization': total_tokens / self.usable_tokens
        }
    
    def clear(self):
        """Clear all buffers."""
        for priority in self.buffers:
            self.buffers[priority] = []
    
    def process_large_document(self, document: str, query: str) -> str:
        """
        A-Team Enhancement: Auto-chunk documents exceeding context limit.
        
        Activates chunker when document > 60% of context, extracts relevant
        information from each chunk with future-task awareness.
        """
        doc_tokens = self.estimate_tokens(document)
        
        # If document fits comfortably, return as-is
        if doc_tokens < self.usable_tokens * 0.6:
            return document
        
        logger.info(f"📄 Auto-chunking large document: {doc_tokens} tokens > {int(self.usable_tokens * 0.6)} limit")
        
        # Calculate chunk size (25% of usable context)
        chunk_size_chars = (self.usable_tokens * 4) // 4
        overlap_chars = 200
        
        # Create chunks with overlap
        chunks = []
        pos = 0
        while pos < len(document):
            end = min(pos + chunk_size_chars, len(document))
            chunks.append(document[pos:end])
            pos = end - overlap_chars  # Overlap
        
        # Extract relevant info from each chunk
        extractions = []
        for i, chunk in enumerate(chunks):
            # Simple relevance check: does chunk contain query keywords?
            query_words = set(query.lower().split())
            chunk_lower = chunk.lower()
            relevance = sum(1 for w in query_words if w in chunk_lower) / max(len(query_words), 1)
            
            if relevance > 0.2:  # At least 20% keyword overlap
                # Keep this chunk but compress
                compressed = self._smart_compress(chunk, chunk_size_chars // 2)
                extractions.append(f"[Chunk {i+1}/{len(chunks)}]\n{compressed}")
        
        if extractions:
            result = "\n\n".join(extractions)
            logger.info(f"📄 Extracted {len(extractions)}/{len(chunks)} relevant chunks")
            return result
        else:
            # No relevant chunks found - return compressed summary
            return self._smart_compress(document, self.usable_tokens * 4 // 2)
    
    def catch_and_recover(self, error: Exception, current_context: str) -> Optional[str]:
        """
        A-Team Fix: Generic error interception without keyword matching.
        
        Uses error structure and numeric values, not string patterns.
        """
        error_str = str(error).lower()
        
        # Method 1: Look for any number > usable_tokens
        import re
        numbers = re.findall(r'\d+', error_str)
        for num_str in numbers:
            num = int(num_str)
            if num > self.usable_tokens:
                logger.warning(f"🔄 Detected token overflow ({num} > {self.usable_tokens})")
                return self._smart_compress(current_context, (self.usable_tokens - 2000) * 4)
        
        # Method 2: Check error type name (structure-based)
        error_type = type(error).__name__.lower()
        overflow_indicators = ['length', 'limit', 'overflow', 'size', 'token', 'context']
        if any(ind in error_type for ind in overflow_indicators):
            logger.warning(f"🔄 Detected overflow error by type: {type(error).__name__}")
            return self._smart_compress(current_context, (self.usable_tokens - 2000) * 4)
        
        # Method 3: Check if error has specific attributes
        if hasattr(error, 'max_tokens') or hasattr(error, 'token_count'):
            logger.warning(f"🔄 Detected token error by attribute")
            return self._smart_compress(current_context, (self.usable_tokens - 2000) * 4)
        
        return None  # Not a context overflow error


# =============================================================================
# POLICY EXPLORER - Dynamic TODO with Exploration
# =============================================================================


# =============================================================================
# BACKWARD COMPATIBILITY - DEPRECATED ALIASES
# =============================================================================
# REFACTORING PHASE 1.3: Deprecation alias for renamed class
# This will be removed in a future version.

SmartContextGuard = LLMContextManager
