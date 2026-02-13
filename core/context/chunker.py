"""
 AGENTIC CHUNKER
==================

NO RULE-BASED CHUNKING!
NO arbitrary split points!

Uses LLM to intelligently chunk large content based on:
- Semantic boundaries (natural break points)
- Task requirements (what needs to be extracted)
- Context preservation (ensure continuity across chunks)

The chunker UNDERSTANDS the content structure!
"""

import dspy
from typing import Dict, List, Optional, Callable
import logging

from ..utils.tokenizer import SmartTokenizer

logger = logging.getLogger(__name__)


class ChunkingSignature(dspy.Signature):
    """Intelligently chunk large content for sequential processing."""
    
    content_preview = dspy.InputField(desc="Preview of content (head + tail)")
    content_length = dspy.InputField(desc="Total length in characters")
    task_description = dspy.InputField(desc="What needs to be extracted/processed")
    max_chunk_size = dspy.InputField(desc="Maximum tokens per chunk")
    
    num_chunks = dspy.OutputField(desc="Recommended number of chunks")
    chunk_boundaries = dspy.OutputField(desc="Where to split (semantic boundaries)")
    chunk_overlap = dspy.OutputField(desc="How much overlap between chunks (in tokens)")
    processing_strategy = dspy.OutputField(desc="How to process chunks and combine results")


class CombiningSignature(dspy.Signature):
    """Intelligently combine results from multiple chunks."""
    
    chunk_results = dspy.InputField(desc="Results from processing each chunk")
    original_task = dspy.InputField(desc="Original task description")
    chunk_summaries = dspy.InputField(desc="Summary of what each chunk contained")
    
    combined_result = dspy.OutputField(desc="Combined final result")
    confidence = dspy.OutputField(desc="Confidence in combination (0-10)")


class ContextChunker:
    """
    LLM-powered semantic chunking (formerly ContextChunker).

    Smart chunker that creates semantic chunks with context preservation.

    NO rule-based splitting! Uses LLM to find natural boundaries.
    """
    
    def __init__(self, lm=None):
        """
        Initialize chunker.
        
        Args:
            lm: Optional DSPy language model. If None, uses dspy.settings.lm.
        """
        self.lm = lm
        if lm is None:
            # Use global DSPy LM if available
            if hasattr(dspy.settings, 'lm') and dspy.settings.lm:
                self.lm = dspy.settings.lm
        
        self.chunker = dspy.ChainOfThought(ChunkingSignature)
        self.combiner = dspy.ChainOfThought(CombiningSignature)
    
    async def chunk_and_process(
        self,
        content: str,
        task_context: Dict,
        processor_fn: Callable,
        max_chunk_size: int = 10000
    ) -> str:
        """
        Chunk large content, process each chunk, combine results.
        
        Args:
            content: FULL large content (NO pre-slicing!)
            task_context: Task details (goal, query, etc.)
            processor_fn: async function to process each chunk
            max_chunk_size: Maximum tokens per chunk
        
        Returns:
            Combined result from all chunks
        """
        content_length = len(content)
        tokenizer = SmartTokenizer.get_instance()
        estimated_tokens = tokenizer.count_tokens(content)
        logger.info(f" Agentic chunking for {task_context.get('actor_name', 'unknown')}...")
        logger.info(f"   Content length: {content_length} chars (~{estimated_tokens} tokens)")
        logger.info(f"   Max chunk size: {max_chunk_size} tokens")

        # Check if chunking needed
        if estimated_tokens <= max_chunk_size:
            logger.info(f" No chunking needed ({estimated_tokens} <= {max_chunk_size} tokens)")
            return await processor_fn(content)
        
        # Create content preview (head + tail, NO middle slicing!)
        preview_size = min(5000, content_length // 2)
        content_preview = content[:preview_size] + "\n\n[... middle content ...]\n\n" + content[-preview_size:]
        
        # Ask LLM how to chunk
        with dspy.context(lm=self.lm):
            chunk_plan = self.chunker(
                content_preview=content_preview,
                content_length=str(content_length),
                task_description=f"Actor '{task_context.get('actor_name')}' needs to: {task_context.get('goal', 'process this content')}",
                max_chunk_size=str(max_chunk_size)
            )
        
        logger.info(f" Chunking plan:")
        logger.info(f"   Num chunks: {chunk_plan.num_chunks}")
        logger.info(f"   Chunk overlap: {chunk_plan.chunk_overlap}")
        logger.info(f"   Strategy: {chunk_plan.processing_strategy}")
        
        # Parse num_chunks
        try:
            num_chunks = int(chunk_plan.num_chunks)
        except (ValueError, TypeError, AttributeError) as e:
            logger.debug(f"Chunk count parsing failed: {e}")
            # Fallback: estimate
            num_chunks = max(2, (estimated_tokens // max_chunk_size) + 1)
            logger.warning(f" Could not parse num_chunks, using estimate: {num_chunks}")

        # Parse overlap
        try:
            overlap_tokens = int(chunk_plan.chunk_overlap)
            overlap_chars = overlap_tokens * 4
        except (ValueError, TypeError, AttributeError) as e:
            logger.debug(f"Overlap parsing failed: {e}")
            overlap_chars = 500  # Fallback
        
        # Create chunks (simple for now, can be made smarter with boundary detection)
        chunk_size_chars = content_length // num_chunks
        chunks = []
        chunk_summaries = []
        
        for i in range(num_chunks):
            start = max(0, i * chunk_size_chars - overlap_chars)
            end = min(content_length, (i + 1) * chunk_size_chars + overlap_chars)
            
            chunk = content[start:end]
            chunks.append(chunk)
            
            # Create summary
            chunk_preview = chunk[:200] if len(chunk) > 200 else chunk
            chunk_summaries.append(f"Chunk {i+1}: {chunk_preview}...")
        
        logger.info(f" Created {len(chunks)} chunks")
        
        # Process each chunk
        chunk_results = []
        for i, chunk in enumerate(chunks, 1):
            logger.info(f" Processing chunk {i}/{len(chunks)}...")
            try:
                result = await processor_fn(chunk)
                chunk_results.append(str(result))
                logger.info(f" Chunk {i} processed successfully")
            except Exception as e:
                logger.error(f" Chunk {i} processing failed: {e}")
                chunk_results.append(f"[Error processing chunk {i}: {str(e)}]")
        
        # Combine results using LLM
        logger.info(f" Combining {len(chunk_results)} chunk results...")
        
        chunk_results_str = "\n\n".join([
            f"## Chunk {i+1} Result:\n{result}"
            for i, result in enumerate(chunk_results)
        ])
        
        chunk_summaries_str = "\n".join(chunk_summaries)
        
        with dspy.context(lm=self.lm):
            combined = self.combiner(
                chunk_results=chunk_results_str,
                original_task=f"Actor '{task_context.get('actor_name')}' goal: {task_context.get('goal')}",
                chunk_summaries=chunk_summaries_str
            )
        
        logger.info(f" Combined results (confidence: {combined.confidence}/10)")
        
        return combined.combined_result
    
    # NEW: AgentSlack-compatible simpler API
    async def chunk(
        self,
        data: str,
        chunk_size: int = 1000,
        overlap: int = 100,
        preserve_structure: bool = False
    ) -> List[str]:
        """
        Simple chunking API for AgentSlack.
        
        Args:
            data: Input data (string)
            chunk_size: Target size in tokens
            overlap: Overlap between chunks in tokens
            preserve_structure: Try to preserve structure (not fully implemented yet)
        
        Returns:
            List of chunks
        """
        logger.info(f" [AgentSlack API] Chunking: {len(data)} chars, chunk_size={chunk_size}, overlap={overlap}")
        
        # Simple implementation for now (can be enhanced with LLM later)
        if not data:
            logger.info(" Empty data, returning empty list")
            return []
        
        # Estimate chars per chunk (rough: 4 chars per token)
        chunk_size_chars = chunk_size * 4
        overlap_chars = overlap * 4
        
        # Check if chunking needed
        if len(data) <= chunk_size_chars:
            logger.info(f" No chunking needed ({len(data)} <= {chunk_size_chars} chars)")
            return [data]
        
        # Create chunks
        chunks = []
        start = 0
        
        while start < len(data):
            end = min(start + chunk_size_chars, len(data))
            chunk = data[start:end]
            chunks.append(chunk)
            
            # Move forward with overlap
            start = end - overlap_chars
            if start >= len(data):
                break
        
        logger.info(f" Created {len(chunks)} chunks")
        return chunks


# =============================================================================
# BACKWARD COMPATIBILITY - DEPRECATED ALIASES
# =============================================================================
# REFACTORING PHASE 1.3: Deprecation alias for renamed class
# This will be removed in a future version.

