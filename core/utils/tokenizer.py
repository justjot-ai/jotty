"""
Smart Tokenizer Utility
=======================

Provides accurate token estimation with tiktoken fallback to improved heuristics.

A-Team Critical Fix: Replace naive len(text)//4 with proper tokenization.

Features:
- Uses tiktoken for cl100k_base (GPT-4/Claude compatible)
- Fallback to improved heuristics if tiktoken unavailable
- Cache tokenizer instance (singleton)
- Handle CJK, code, JSON specially
"""

import re
import logging
from typing import Optional, Dict, Any
from functools import lru_cache

logger = logging.getLogger(__name__)


class SmartTokenizer:
    """
    Accurate token estimation with tiktoken fallback to heuristics.

    Singleton pattern ensures only one tokenizer instance per encoding.

    Usage:
        tokenizer = SmartTokenizer.get_instance()
        tokens = tokenizer.count_tokens("Hello world")
        tokens = tokenizer.estimate_tokens("Hello world")  # Alias
    """

    _instances: Dict[str, 'SmartTokenizer'] = {}

    # Default encoding for GPT-4/Claude models
    DEFAULT_ENCODING = "cl100k_base"

    # Heuristic multipliers for different content types
    HEURISTICS = {
        'english': 4.0,      # ~4 chars per token for English
        'code': 3.0,         # Code tends to have more tokens per char
        'json': 3.5,         # JSON has structure characters
        'cjk': 1.5,          # CJK characters are often 1 token each
        'mixed': 3.5,        # Mixed content
        'whitespace_heavy': 5.0,  # Content with lots of whitespace
    }

    def __init__(self, encoding_name: str = None):
        """
        Initialize tokenizer with specified encoding.

        Args:
            encoding_name: tiktoken encoding name (default: cl100k_base)
        """
        self.encoding_name = encoding_name or self.DEFAULT_ENCODING
        self._tiktoken_encoder = None
        self._tiktoken_available = False

        # Try to load tiktoken
        self._init_tiktoken()

        # Statistics
        self._total_calls = 0
        self._tiktoken_calls = 0
        self._heuristic_calls = 0

    def _init_tiktoken(self):
        """Initialize tiktoken if available."""
        try:
            import tiktoken
            self._tiktoken_encoder = tiktoken.get_encoding(self.encoding_name)
            self._tiktoken_available = True
            logger.debug(f"SmartTokenizer: Using tiktoken with encoding '{self.encoding_name}'")
        except ImportError:
            logger.info("SmartTokenizer: tiktoken not available, using heuristics")
            self._tiktoken_available = False
        except Exception as e:
            logger.warning(f"SmartTokenizer: tiktoken error ({e}), using heuristics")
            self._tiktoken_available = False

    @classmethod
    def get_instance(cls, encoding_name: str = None) -> 'SmartTokenizer':
        """
        Get singleton instance for the specified encoding.

        Args:
            encoding_name: tiktoken encoding name (default: cl100k_base)

        Returns:
            SmartTokenizer instance
        """
        encoding = encoding_name or cls.DEFAULT_ENCODING
        if encoding not in cls._instances:
            cls._instances[encoding] = cls(encoding)
        return cls._instances[encoding]

    @classmethod
    def reset_instances(cls):
        """Reset all cached instances (for testing)."""
        cls._instances.clear()

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text accurately.

        Uses tiktoken if available, falls back to smart heuristics.

        Args:
            text: Text to count tokens for

        Returns:
            Token count (int)
        """
        if not text:
            return 0

        self._total_calls += 1

        if self._tiktoken_available and self._tiktoken_encoder:
            self._tiktoken_calls += 1
            try:
                return len(self._tiktoken_encoder.encode(text))
            except Exception as e:
                logger.debug(f"tiktoken encoding failed: {e}, using heuristics")

        # Fallback to heuristics
        self._heuristic_calls += 1
        return self._estimate_with_heuristics(text)

    def estimate_tokens(self, text: str) -> int:
        """
        Alias for count_tokens for backward compatibility.

        Args:
            text: Text to estimate tokens for

        Returns:
            Token count (int)
        """
        return self.count_tokens(text)

    def _estimate_with_heuristics(self, text: str) -> int:
        """
        Estimate tokens using improved heuristics.

        Analyzes content type and applies appropriate multiplier.

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        if not text:
            return 0

        content_type = self._detect_content_type(text)
        chars_per_token = self.HEURISTICS.get(content_type, self.HEURISTICS['mixed'])

        # Base estimate
        base_estimate = len(text) / chars_per_token

        # Adjust for special characters and patterns
        adjustments = self._calculate_adjustments(text)

        final_estimate = base_estimate + adjustments

        # Ensure minimum of 1 token for non-empty text
        return max(1, int(final_estimate))

    def _detect_content_type(self, text: str) -> str:
        """
        Detect the primary content type of text.

        Args:
            text: Text to analyze

        Returns:
            Content type string
        """
        # Sample for performance (first 5000 chars)
        sample = text[:5000]

        # Check for CJK characters
        cjk_pattern = re.compile(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]')
        cjk_count = len(cjk_pattern.findall(sample))
        if cjk_count > len(sample) * 0.2:
            return 'cjk'

        # Check for code patterns
        code_indicators = [
            r'def\s+\w+\s*\(',   # Python function
            r'function\s+\w+\s*\(',  # JavaScript function
            r'class\s+\w+',     # Class definition
            r'\{\s*\n',         # Code blocks
            r'=>',              # Arrow functions
            r'import\s+',       # Imports
            r'from\s+\w+\s+import',  # Python imports
            r';\s*$',           # Statement endings
        ]
        code_count = sum(len(re.findall(p, sample, re.MULTILINE)) for p in code_indicators)
        if code_count > 5:
            return 'code'

        # Check for JSON
        if sample.strip().startswith(('{', '[')) and sample.strip().endswith(('}', ']')):
            try:
                import json
                json.loads(sample[:1000] if len(sample) > 1000 else sample)
                return 'json'
            except (json.JSONDecodeError, ValueError):
                # Might still be partial JSON
                if sample.count('{') > 3 or sample.count('"') > 10:
                    return 'json'

        # Check for whitespace-heavy content
        whitespace_ratio = len(re.findall(r'\s', sample)) / max(len(sample), 1)
        if whitespace_ratio > 0.4:
            return 'whitespace_heavy'

        # Check for mixed CJK
        if cjk_count > 0:
            return 'mixed'

        return 'english'

    def _calculate_adjustments(self, text: str) -> int:
        """
        Calculate token adjustments for special patterns.

        Args:
            text: Text to analyze

        Returns:
            Token adjustment count
        """
        adjustments = 0

        # Numbers often become multiple tokens
        numbers = re.findall(r'\d+', text)
        for num in numbers:
            if len(num) > 4:
                # Long numbers get split
                adjustments += len(num) // 3

        # URLs become many tokens
        urls = re.findall(r'https?://\S+', text)
        adjustments += len(urls) * 5

        # Special tokens (newlines, tabs)
        adjustments += text.count('\n') * 0.5
        adjustments += text.count('\t') * 0.5

        # Punctuation clusters
        punct_clusters = re.findall(r'[^\w\s]{3,}', text)
        adjustments += len(punct_clusters) * 2

        return int(adjustments)

    @property
    def is_tiktoken_available(self) -> bool:
        """Check if tiktoken is being used."""
        return self._tiktoken_available

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get tokenizer usage statistics.

        Returns:
            Dict with statistics
        """
        return {
            'encoding': self.encoding_name,
            'tiktoken_available': self._tiktoken_available,
            'total_calls': self._total_calls,
            'tiktoken_calls': self._tiktoken_calls,
            'heuristic_calls': self._heuristic_calls,
            'tiktoken_ratio': self._tiktoken_calls / max(self._total_calls, 1),
        }


# =============================================================================
# CONVENIENCE FUNCTIONS (Module-level API)
# =============================================================================

@lru_cache(maxsize=1)
def get_tokenizer(encoding: str = None) -> SmartTokenizer:
    """
    Get the default SmartTokenizer instance.

    Cached for performance.

    Args:
        encoding: Optional encoding name

    Returns:
        SmartTokenizer instance
    """
    return SmartTokenizer.get_instance(encoding)


def count_tokens(text: str, encoding: str = None) -> int:
    """
    Count tokens in text.

    Convenience function using default tokenizer.

    Args:
        text: Text to count tokens for
        encoding: Optional encoding name

    Returns:
        Token count
    """
    return get_tokenizer(encoding).count_tokens(text)


def estimate_tokens(text: str, encoding: str = None) -> int:
    """
    Alias for count_tokens for backward compatibility.

    Args:
        text: Text to estimate tokens for
        encoding: Optional encoding name

    Returns:
        Token count
    """
    return count_tokens(text, encoding)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'SmartTokenizer',
    'get_tokenizer',
    'count_tokens',
    'estimate_tokens',
]
