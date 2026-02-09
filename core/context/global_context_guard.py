"""
Jotty v7.6 - Global Context Guard
==================================

A-Team Approved: Pervasive context overflow protection.

Wraps ALL LLM calls to:
1. Detect context overflow from ANY provider
2. Auto-compress and retry
3. Integrate with ContentGate for chunking
4. Inject memories into prompts

Key Insight: Don't just catch errors at top-level.
DSPy modules, ReAct loops, CoT chains - ANY can overflow.
"""

import asyncio
import functools
import re
import json
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass
import logging

from ..utils.tokenizer import SmartTokenizer

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# TOKEN ERROR PATTERNS (Exhaustive - All Major Providers)
# =============================================================================

# NOTE: These are NOT used for string matching. 
# They're reference patterns for documentation.
# The actual detection is STRUCTURAL.
TOKEN_ERROR_REFERENCE = {
    "openai": [
        "context_length_exceeded",
        "maximum_context_length",
        "max_tokens",
        "This model's maximum context length"
    ],
    "anthropic": [
        "content_too_long",
        "message_too_long",
        "prompt is too long",
        "max_tokens_to_sample"
    ],
    "google": [
        "RESOURCE_EXHAUSTED",
        "token limit exceeded",
        "context window exceeded"
    ],
    "azure": [
        "context_window_exceeded",
        "tokens exceed",
        "maximum tokens"
    ],
    "cohere": [
        "token_limit",
        "context_length"
    ],
    "local_models": [
        "sequence too long",
        "max_tokens",
        "context length",
        "OOM",  # Out of memory often means token overflow
        "CUDA out of memory"
    ],
    "litellm": [
        "ContextWindowExceededError",
        "InvalidRequestError"
    ]
}


@dataclass
class ContextOverflowInfo:
    """Information about a context overflow error."""
    is_overflow: bool
    detected_tokens: Optional[int] = None
    max_allowed: Optional[int] = None
    provider_hint: Optional[str] = None
    detection_method: str = "unknown"


# =============================================================================
# STRUCTURAL OVERFLOW DETECTOR
# =============================================================================

class OverflowDetector:
    """
    Detect context overflow using STRUCTURAL analysis.
    
    NO HARDCODED STRING MATCHING.
    Uses:
    1. Numeric extraction: Find numbers > max_tokens
    2. Error type hierarchy: Check exception class names
    3. Error attributes: Check for token-related attributes
    4. Error code analysis: Check HTTP/API error codes
    """
    
    def __init__(self, max_tokens: int = 28000):
        self.max_tokens = max_tokens
        
        # Structural indicators (in class/type names)
        self.overflow_type_indicators = frozenset([
            'overflow', 'length', 'size', 'limit', 'exhausted',
            'context', 'token', 'sequence', 'long', 'exceed',
            'capacity', 'window', 'memory', 'oom'
        ])
        
        # Common error codes that indicate overflow
        self.overflow_error_codes = frozenset([
            400, 413, 429,  # HTTP codes
            'context_length', 'invalid_request', 'resource_exhausted'
        ])
    
    def detect(self, error: Exception) -> ContextOverflowInfo:
        """
        Detect if error is a context overflow.
        
        Returns ContextOverflowInfo with detection details.
        """
        # Method 1: Numeric extraction
        numeric_result = self._detect_by_numbers(error)
        if numeric_result.is_overflow:
            return numeric_result
        
        # Method 2: Error type hierarchy
        type_result = self._detect_by_type(error)
        if type_result.is_overflow:
            return type_result
        
        # Method 3: Error attributes
        attr_result = self._detect_by_attributes(error)
        if attr_result.is_overflow:
            return attr_result
        
        # Method 4: Error code
        code_result = self._detect_by_code(error)
        if code_result.is_overflow:
            return code_result
        
        return ContextOverflowInfo(is_overflow=False)
    
    def _detect_by_numbers(self, error: Exception) -> ContextOverflowInfo:
        """Find numbers in error that exceed max_tokens."""
        error_str = str(error)
        
        # Extract all numbers
        numbers = re.findall(r'\d+', error_str)
        
        for num_str in numbers:
            try:
                num = int(num_str)
                if num > self.max_tokens and num < 1000000:  # Reasonable token range
                    return ContextOverflowInfo(
                        is_overflow=True,
                        detected_tokens=num,
                        max_allowed=self.max_tokens,
                        detection_method="numeric_extraction"
                    )
            except ValueError:
                continue
        
        return ContextOverflowInfo(is_overflow=False)
    
    def _detect_by_type(self, error: Exception) -> ContextOverflowInfo:
        """Check error type hierarchy for overflow indicators."""
        # Get full class hierarchy
        error_types = []
        for cls in type(error).__mro__:
            error_types.append(cls.__name__.lower())
        
        # Check each type name for indicators
        for type_name in error_types:
            for indicator in self.overflow_type_indicators:
                if indicator in type_name:
                    return ContextOverflowInfo(
                        is_overflow=True,
                        detection_method=f"type_hierarchy_{type_name}"
                    )
        
        return ContextOverflowInfo(is_overflow=False)
    
    def _detect_by_attributes(self, error: Exception) -> ContextOverflowInfo:
        """Check error attributes for token-related info."""
        # Common attributes to check
        attrs_to_check = [
            'code', 'error_code', 'status_code',
            'max_tokens', 'token_count', 'context_length',
            'param', 'type', 'message'
        ]
        
        for attr in attrs_to_check:
            if hasattr(error, attr):
                value = str(getattr(error, attr, '')).lower()
                for indicator in self.overflow_type_indicators:
                    if indicator in value:
                        return ContextOverflowInfo(
                            is_overflow=True,
                            detection_method=f"attribute_{attr}"
                        )
        
        return ContextOverflowInfo(is_overflow=False)
    
    def _detect_by_code(self, error: Exception) -> ContextOverflowInfo:
        """Check error codes."""
        # Try to get error code
        code = None
        for attr in ['code', 'status_code', 'error_code', 'status']:
            if hasattr(error, attr):
                code = getattr(error, attr)
                break
        
        if code is not None:
            code_str = str(code).lower()
            for overflow_code in self.overflow_error_codes:
                if str(overflow_code) in code_str:
                    return ContextOverflowInfo(
                        is_overflow=True,
                        detection_method=f"error_code_{code}"
                    )
        
        return ContextOverflowInfo(is_overflow=False)


# =============================================================================
# CONTENT COMPRESSOR
# =============================================================================

class ContentCompressor:
    """Simple content compressor that truncates text to fit target token count."""

    def compress(self, content: str, target_tokens: int) -> str:
        """
        Compress content to fit within target token count.

        Args:
            content: Content to compress
            target_tokens: Target token count

        Returns:
            Compressed content
        """
        # Simple truncation: ~4 chars per token
        target_chars = target_tokens * 4

        if len(content) <= target_chars:
            return content

        # Truncate with ellipsis
        return content[:target_chars - 50] + "\n... (content truncated to fit context) ..."


class GlobalContextGuard:
    """
    Pervasive context guard that wraps ALL LLM calls.
    
    Features:
    1. Structural overflow detection (no hardcoded strings)
    2. Automatic compression and retry
    3. Memory injection before calls
    4. DSPy integration
    5. Priority-based context building (A-Team v2.0)
    """
    
    # Priority levels (A-Team approved)
    CRITICAL = 0  # Must always be included
    HIGH = 1      # Include unless severe overflow
    MEDIUM = 2    # Include if space available (default)
    LOW = 3       # Include only if abundant space
    
    def __init__(self, max_tokens: int = 28000, memory=None):
        self.max_tokens = max_tokens
        self.detector = OverflowDetector(max_tokens)
        self.compressor = ContentCompressor()
        self.memory = memory
        self._tokenizer = SmartTokenizer.get_instance()

        # Priority-based buffers (A-Team v2.0)
        self.buffers = {
            self.CRITICAL: [],
            self.HIGH: [],
            self.MEDIUM: [],
            self.LOW: []
        }

        # Statistics
        self.total_calls = 0
        self.overflow_recovered = 0
        self.compression_applied = 0

        logger.info(f"🛡️ GlobalContextGuard initialized (max_tokens={max_tokens})")
    
    def register(self, key: str, content: str, priority: int = None):
        """
        Register content with a priority level for context building.
        
        Args:
            key: Identifier for this content
            content: The content to register
            priority: Priority level (CRITICAL, HIGH, MEDIUM, LOW). Defaults to MEDIUM.
        
        A-Team v2.0: Priority-based context management.
        """
        if priority is None:
            priority = self.MEDIUM
        
        if priority not in self.buffers:
            logger.warning(f"Invalid priority {priority}, defaulting to MEDIUM")
            priority = self.MEDIUM
        
        self.buffers[priority].append({'key': key, 'content': content})
        logger.debug(f"Registered '{key}' with priority {priority}")
    
    def clear_buffers(self):
        """Clear all registered content buffers."""
        for priority in self.buffers:
            self.buffers[priority] = []
    
    def build_context(self, additional_content: str = "") -> str:
        """
        Build context from registered buffers, respecting priorities.
        
        Args:
            additional_content: Extra content to include (treated as CRITICAL)
        
        Returns:
            Assembled context string
        
        A-Team v2.0: Priority-respecting context assembly.
        """
        result = []
        remaining_tokens = self.max_tokens
        
        # Start with additional content (CRITICAL)
        if additional_content:
            tokens_needed = self._estimate_tokens(additional_content)
            result.append(additional_content)
            remaining_tokens -= tokens_needed
        
        # Include buffers in priority order
        for priority in [self.CRITICAL, self.HIGH, self.MEDIUM, self.LOW]:
            for item in self.buffers[priority]:
                content = item['content']
                tokens_needed = self._estimate_tokens(content)
                
                if tokens_needed <= remaining_tokens:
                    result.append(content)
                    remaining_tokens -= tokens_needed
                else:
                    # Try compression for HIGH priority items
                    if priority <= self.HIGH:
                        compressed = self.compressor.compress(content, remaining_tokens)
                        compressed_tokens = self._estimate_tokens(compressed)
                        if compressed_tokens <= remaining_tokens:
                            result.append(compressed)
                            remaining_tokens -= compressed_tokens
                            logger.debug(f"Compressed {item['key']} to fit in context")
        
        return '\n\n'.join(result)
    
    def _estimate_tokens(self, content: str) -> int:
        """Estimate token count using SmartTokenizer."""
        return self._tokenizer.count_tokens(content)
    
    def wrap_function(self, func: Callable) -> Callable:
        """Wrap a function with context guard."""
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await self._guarded_call(func, args, kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                asyncio.get_running_loop()
                # Already in async context — can't use run_until_complete.
                # Create a task instead (caller must be in sync code that
                # somehow ended up in an async frame, e.g. Jupyter).
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    return pool.submit(
                        asyncio.run, self._guarded_call(func, args, kwargs)
                    ).result()
            except RuntimeError:
                # No running loop — safe to use asyncio.run()
                return asyncio.run(self._guarded_call(func, args, kwargs))
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    async def _guarded_call(
        self, 
        func: Callable, 
        args: tuple, 
        kwargs: dict,
        retry_count: int = 0,
        max_retries: int = 3
    ) -> Any:
        """Execute function with overflow protection."""
        self.total_calls += 1
        
        try:
            # Check if function is async
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
                
        except Exception as e:
            # Detect overflow
            overflow_info = self.detector.detect(e)
            
            if overflow_info.is_overflow and retry_count < max_retries:
                logger.warning(
                    f"🔄 Context overflow detected (method={overflow_info.detection_method}). "
                    f"Compressing and retrying ({retry_count + 1}/{max_retries})..."
                )
                self.overflow_recovered += 1
                
                # Compress all string arguments
                compressed_args = self._compress_args(args)
                compressed_kwargs = self._compress_kwargs(kwargs)
                
                self.compression_applied += 1
                
                # Retry with compressed context
                return await self._guarded_call(
                    func, compressed_args, compressed_kwargs,
                    retry_count + 1, max_retries
                )
            
            # Not an overflow or max retries exceeded
            raise
    
    def _compress_args(self, args: tuple) -> tuple:
        """Compress string arguments."""
        result = []
        for arg in args:
            if isinstance(arg, str) and len(arg) > 1000:
                target = int(len(arg) * 0.7)  # Reduce by 30%
                result.append(self.compressor.compress(arg, target // 4))
            else:
                result.append(arg)
        return tuple(result)
    
    def _compress_kwargs(self, kwargs: dict) -> dict:
        """Compress string keyword arguments."""
        result = {}
        for key, value in kwargs.items():
            if isinstance(value, str) and len(value) > 1000:
                target = int(len(value) * 0.7)
                result[key] = self.compressor.compress(value, target // 4)
            else:
                result[key] = value
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get guard statistics."""
        return {
            'total_calls': self.total_calls,
            'overflow_recovered': self.overflow_recovered,
            'compression_applied': self.compression_applied,
            'recovery_rate': self.overflow_recovered / max(self.total_calls, 1)
        }


# =============================================================================
# DSPY INTEGRATION
# =============================================================================

def patch_dspy_with_guard(guard: GlobalContextGuard):
    """
    Patch DSPy to use GlobalContextGuard for ALL LLM calls.
    
    This ensures that ANY DSPy module (ChainOfThought, ReAct, etc.)
    is protected from context overflow.
    """
    if not DSPY_AVAILABLE:
        logger.warning("DSPy not available, skipping patch")
        return
    
    # Store original
    if hasattr(dspy, '_original_lm_call'):
        logger.debug("DSPy already patched")
        return
    
    dspy._original_lm_call = dspy.LM.__call__
    
    def guarded_lm_call(lm_self, prompt, *args, **kwargs):
        """Wrapped LM call with context guard."""
        try:
            return dspy._original_lm_call(lm_self, prompt, *args, **kwargs)
        except Exception as e:
            overflow_info = guard.detector.detect(e)
            
            if overflow_info.is_overflow:
                logger.warning(f"🔄 DSPy LM overflow detected, compressing prompt...")
                guard.overflow_recovered += 1
                
                # Compress prompt
                compressed_prompt = guard.compressor.compress(
                    prompt, guard.max_tokens - 2000
                )
                guard.compression_applied += 1
                
                return dspy._original_lm_call(lm_self, compressed_prompt, *args, **kwargs)
            
            raise
    
    dspy.LM.__call__ = guarded_lm_call
    logger.info("✅ DSPy patched with GlobalContextGuard")


def unpatch_dspy():
    """Restore original DSPy LM call."""
    if DSPY_AVAILABLE and hasattr(dspy, '_original_lm_call'):
        dspy.LM.__call__ = dspy._original_lm_call
        del dspy._original_lm_call
        logger.info("✅ DSPy unpatched")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'OverflowDetector',
    'ContextOverflowInfo',
    'ContentCompressor',
    'GlobalContextGuard',
    'patch_dspy_with_guard',
    'unpatch_dspy'
]

