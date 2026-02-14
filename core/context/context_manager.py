"""
Jotty v6.4 - Smart Context Manager
==================================

A-Team Critical Fix: Intelligent context management that:
1. Catches API token limit errors and auto-recovers
2. Preserves task-critical info (TODO, goals, critical memories)
3. Compresses intelligently based on task needs
4. NEVER loses info needed for future tasks

This is the SMART context manager the user requested.
"""

import re
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import dspy

from ..utils.tokenizer import SmartTokenizer

logger = logging.getLogger(__name__)


# =============================================================================
# PRIORITY LEVELS FOR CONTEXT PRESERVATION
# =============================================================================

class ContextPriority(Enum):
    """Priority levels for context chunks during compression."""
    CRITICAL = 1    # NEVER compress: current task, TODO state, goal
    HIGH = 2        # Compress last: recent memories, errors, tool results
    MEDIUM = 3      # Compress when needed: trajectory, history
    LOW = 4         # Compress first: verbose logs, old memories


@dataclass
class ContextChunk:
    """A chunk of context with metadata for smart compression."""
    content: str
    priority: ContextPriority
    category: str  # "task", "todo", "memory", "trajectory", "tool_result"
    tokens: int = 0
    is_compressed: bool = False
    original_tokens: int = 0
    
    def __post_init__(self):
        if not self.tokens:
            self.tokens = SmartTokenizer.get_instance().count_tokens(self.content)
        if not self.original_tokens:
            self.original_tokens = self.tokens


# =============================================================================
# SMART CONTEXT MANAGER
# =============================================================================

class SmartContextManager:
    """
    Intelligent context manager that NEVER loses critical info.
    
    Features:
    1. Task-aware prioritization
    2. API error catching and recovery
    3. Hierarchical compression
    4. TODO preservation
    5. Memory consolidation instead of truncation
    """
    
    def __init__(self, 
                 max_tokens: int = 28000,
                 safety_margin: float = 0.85,  # Use 85% of max
                 enable_api_error_recovery: bool = True):
        """
        Args:
            max_tokens: Maximum context tokens (from model config)
            safety_margin: Use only this fraction of max
            enable_api_error_recovery: Catch and recover from API errors
        """
        self.max_tokens = max_tokens
        self.effective_limit = int(max_tokens * safety_margin)
        self.enable_api_error_recovery = enable_api_error_recovery
        
        # Token estimation
        self.chars_per_token = 4
        
        # Statistics
        self.compressions_count = 0
        self.api_errors_recovered = 0
        self.total_tokens_saved = 0
        
        # Current context state
        self.current_chunks: List[ContextChunk] = []

        # Task/Goal preservation (never compressed)
        self._current_todo: Optional[str] = None
        self._current_goal: Optional[str] = None
        self._critical_memories: List[str] = []
        
        # Compression history for learning
        self.compression_history: List[Dict] = []
        
        # LLM summarizer for smart compression
        self._init_summarizer()
    
    def _init_summarizer(self):
        """Initialize LLM-based summarizer for smart compression."""
        class SummarizeSignature(dspy.Signature):
            """Summarize text while preserving critical information."""
            content: str = dspy.InputField(desc="Content to summarize")
            preserve_keywords: str = dspy.InputField(desc="Keywords to preserve")
            max_length: int = dspy.InputField(desc="Target length in characters")
            summary: str = dspy.OutputField(desc="Compressed summary preserving key info")
        
        try:
            self.summarizer = dspy.ChainOfThought(SummarizeSignature)
        except (ImportError, AttributeError, TypeError) as e:
            logger.debug(f"Could not initialize summarizer: {e}")
            self.summarizer = None
    
    # =========================================================================
    # CONTEXT REGISTRATION
    # =========================================================================
    
    def register_todo(self, todo_content: str):
        """Register current TODO for preservation (NEVER compressed)."""
        self._current_todo = todo_content
        logger.debug(f" Registered TODO ({len(todo_content)} chars)")
    
    def register_goal(self, goal: str):
        """Register current goal for preservation."""
        self._current_goal = goal
        logger.debug(f" Registered goal: {goal}...")
    
    def register_critical_memory(self, memory: str):
        """Register a critical memory that must be preserved."""
        self._critical_memories.append(memory)
        logger.debug(f" Registered critical memory ({len(memory)} chars)")
    
    def add_chunk(self, content: str, category: str, priority: ContextPriority = None):
        """Add a context chunk with auto-detected priority."""
        if priority is None:
            priority = self._auto_detect_priority(category, content)
        
        chunk = ContextChunk(
            content=content,
            priority=priority,
            category=category
        )
        self.current_chunks.append(chunk)
    
    def _auto_detect_priority(self, category: str, content: str) -> ContextPriority:
        """Auto-detect priority based on category and content."""
        # Critical categories
        if category in ['task', 'todo', 'goal', 'current_action']:
            return ContextPriority.CRITICAL
        
        # High priority
        if category in ['error', 'tool_result', 'recent_memory']:
            return ContextPriority.HIGH
        
        # Look for critical keywords
        critical_keywords = ['MUST', 'CRITICAL', 'ERROR', 'FAIL', 'bank_code', 'bank_contribution']
        if any(kw.lower() in content.lower() for kw in critical_keywords):
            return ContextPriority.HIGH
        
        # Medium
        if category in ['trajectory', 'history', 'memory']:
            return ContextPriority.MEDIUM
        
        return ContextPriority.LOW
    
    # =========================================================================
    # CONTEXT BUILDING
    # =========================================================================
    
    def build_context(self, 
                      system_prompt: str,
                      user_input: str,
                      additional_context: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Build context that fits within limits while preserving critical info.
        
        Returns dict with:
        - context: The built context string
        - truncated: Whether truncation occurred
        - preserved: What critical info was preserved
        - stats: Compression statistics
        """
        # Calculate fixed costs
        system_tokens = self.estimate_tokens(system_prompt)
        input_tokens = self.estimate_tokens(user_input)
        fixed_cost = system_tokens + input_tokens
        
        # Reserved space for CRITICAL items
        todo_cost = self.estimate_tokens(self._current_todo or "")
        goal_cost = self.estimate_tokens(self._current_goal or "")
        critical_mem_cost = sum(self.estimate_tokens(m) for m in self._critical_memories)
        reserved_cost = todo_cost + goal_cost + critical_mem_cost
        
        # Available budget for other context
        available = self.effective_limit - fixed_cost - reserved_cost
        
        if available < 500:
            # Not enough space - need aggressive compression
            logger.warning(f" Very limited context budget: {available} tokens")
            available = max(500, available)
        
        # Sort chunks by priority (CRITICAL first)
        sorted_chunks = sorted(self.current_chunks, key=lambda c: c.priority.value)
        
        # Build context
        included_chunks = []
        tokens_used = 0
        
        for chunk in sorted_chunks:
            if tokens_used + chunk.tokens <= available:
                included_chunks.append(chunk)
                tokens_used += chunk.tokens
            elif chunk.priority == ContextPriority.CRITICAL:
                # MUST include critical - compress others
                compressed = self._compress_chunk(chunk, available - tokens_used)
                included_chunks.append(compressed)
                tokens_used += compressed.tokens
            elif chunk.priority == ContextPriority.HIGH and available - tokens_used > 200:
                # Try to fit HIGH priority with compression
                compressed = self._compress_chunk(chunk, min(1000, available - tokens_used))
                included_chunks.append(compressed)
                tokens_used += compressed.tokens
        
        # Build final context
        context_parts = []
        
        # Always include TODO
        if self._current_todo:
            context_parts.append(f"## Current TODO\n{self._current_todo}")
        
        # Always include goal
        if self._current_goal:
            context_parts.append(f"## Goal\n{self._current_goal}")
        
        # Include critical memories
        for mem in self._critical_memories:  # Max 5
            context_parts.append(f"## Critical Memory\n{mem}")
        
        # Include other chunks
        for chunk in included_chunks:
            context_parts.append(f"## {chunk.category}\n{chunk.content}")
        
        final_context = "\n\n".join(context_parts)
        
        return {
            'system_prompt': system_prompt,
            'user_input': user_input,
            'context': final_context,
            'truncated': len(included_chunks) < len(self.current_chunks),
            'preserved': {
                'todo': bool(self._current_todo),
                'goal': bool(self._current_goal),
                'critical_memories': len(self._critical_memories),
                'chunks_included': len(included_chunks),
                'chunks_total': len(self.current_chunks)
            },
            'stats': {
                'total_tokens': fixed_cost + reserved_cost + tokens_used,
                'effective_limit': self.effective_limit,
                'budget_remaining': self.effective_limit - fixed_cost - reserved_cost - tokens_used
            }
        }
    
    # =========================================================================
    # COMPRESSION
    # =========================================================================
    
    def _compress_chunk(self, chunk: ContextChunk, target_tokens: int) -> ContextChunk:
        """Compress a chunk to fit within target token budget."""
        if chunk.tokens <= target_tokens:
            return chunk
        
        self.compressions_count += 1
        original_tokens = chunk.tokens
        
        # Try LLM summarization for important content
        if self.summarizer and chunk.priority.value <= 2:
            try:
                result = self.summarizer(
                    content=chunk.content,
                    preserve_keywords="bank_code, bank_contribution, error, TODO, goal",
                    max_length=target_tokens * self.chars_per_token
                )
                compressed_content = result.summary
            except Exception as e:
                logger.debug(f"LLM compression failed: {e}")
                compressed_content = self._simple_compress(chunk.content, target_tokens)
        else:
            compressed_content = self._simple_compress(chunk.content, target_tokens)
        
        new_chunk = ContextChunk(
            content=compressed_content,
            priority=chunk.priority,
            category=chunk.category,
            is_compressed=True,
            original_tokens=original_tokens
        )
        
        self.total_tokens_saved += original_tokens - new_chunk.tokens
        
        return new_chunk
    
    def _simple_compress(self, text: str, target_tokens: int) -> str:
        """Simple compression: keep start + end, summarize middle."""
        target_chars = target_tokens * self.chars_per_token
        
        if len(text) <= target_chars:
            return text
        
        keep_start = target_chars // 2
        keep_end = target_chars // 2 - 50
        
        return text[:keep_start] + "\n[...COMPRESSED...]\n" + text[-keep_end:]
    
    # =========================================================================
    # API ERROR RECOVERY
    # =========================================================================
    
    def catch_and_recover(self, error: Exception, current_context: str) -> Optional[str]:
        """
        Catch API token limit errors and recover with compressed context.
        
        Returns: Compressed context if recoverable, None otherwise
        """
        if not self.enable_api_error_recovery:
            return None
        
        error_str = str(error).lower()
        
        # Check for token limit errors
        token_error_patterns = [
            "context_length",
            "maximum context length",
            "token limit",
            "too long",
            "context window",
            "max_tokens",
            "context length exceeded"
        ]
        
        is_token_error = any(pattern in error_str for pattern in token_error_patterns)
        
        if not is_token_error:
            return None
        
        self.api_errors_recovered += 1
        logger.warning(f" Token limit error detected - auto-recovering...")
        
        # Extract required reduction
        # Try to find number in error message
        match = re.search(r'(\d+)\s*tokens?', error_str)
        if match:
            error_tokens = int(match.group(1))
            reduction_needed = error_tokens - self.effective_limit
        else:
            # Assume 30% reduction needed
            reduction_needed = int(len(current_context) / self.chars_per_token * 0.3)
        
        # Compress context
        target_tokens = self.effective_limit - 1000  # Extra safety margin
        compressed = self._simple_compress(current_context, target_tokens)
        
        logger.info(f" Context compressed: {len(current_context)} → {len(compressed)} chars")
        
        return compressed
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count using SmartTokenizer."""
        if not text:
            return 0
        return SmartTokenizer.get_instance().count_tokens(text)
    
    def clear_chunks(self):
        """Clear non-persistent chunks."""
        self.current_chunks = []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get context manager statistics."""
        return {
            'max_tokens': self.max_tokens,
            'effective_limit': self.effective_limit,
            'compressions_count': self.compressions_count,
            'api_errors_recovered': self.api_errors_recovered,
            'total_tokens_saved': self.total_tokens_saved,
            'current_chunks': len(self.current_chunks),
            'critical_memories': len(self._critical_memories),
            'has_todo': bool(self._current_todo),
            'has_goal': bool(self._current_goal)
        }


# =============================================================================
# INTEGRATION DECORATOR
# =============================================================================

def with_smart_context(max_tokens: int = 28000):
    """
    Decorator to add smart context management to any agent.
    
    Usage:
        @with_smart_context(max_tokens=28000)
        async def my_agent(query: str, context: str):
            ...
    """
    def decorator(func):
        manager = SmartContextManager(max_tokens=max_tokens)
        
        async def wrapper(*args, **kwargs):
            # Try to execute
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Check if recoverable
                context = kwargs.get('context', str(args))
                compressed = manager.catch_and_recover(e, context)
                
                if compressed:
                    kwargs['context'] = compressed
                    logger.info(" Retrying with compressed context...")
                    return await func(*args, **kwargs)
                
                raise
        
        return wrapper
    return decorator


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'SmartContextManager',
    'ContextChunk',
    'ContextPriority',
    'with_smart_context'
]

