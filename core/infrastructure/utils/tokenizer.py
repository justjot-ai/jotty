"""
Smart Tokenizer Utility
=======================

Provides accurate token estimation with tiktoken fallback to improved heuristics.
Also includes model-specific token counting via tokencost (merged from token_counter.py).

Features:
- Uses tiktoken for cl100k_base (GPT-4/Claude compatible)
- Fallback to improved heuristics if tiktoken unavailable
- Cache tokenizer instance (singleton)
- Handle CJK, code, JSON specially
- Model-specific counting via tokencost (TokenCounter)
- Overflow detection and remaining token calculation
- Message-aware token counting for chat models
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional

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

    _instances: Dict[str, "SmartTokenizer"] = {}

    # Default encoding for GPT-4/Claude models
    DEFAULT_ENCODING = "cl100k_base"

    # Heuristic multipliers for different content types
    HEURISTICS = {
        "english": 4.0,  # ~4 chars per token for English
        "code": 3.0,  # Code tends to have more tokens per char
        "json": 3.5,  # JSON has structure characters
        "cjk": 1.5,  # CJK characters are often 1 token each
        "mixed": 3.5,  # Mixed content
        "whitespace_heavy": 5.0,  # Content with lots of whitespace
    }

    def __init__(self, encoding_name: str | None = None) -> None:
        """
        Initialize tokenizer with specified encoding.

        Args:
            encoding_name: tiktoken encoding name (default: cl100k_base)
        """
        self.encoding_name = encoding_name or self.DEFAULT_ENCODING
        self._tiktoken_encoder: Optional[Encoding] = None  # type: ignore[name-defined]
        self._tiktoken_available = False

        # Try to load tiktoken
        self._init_tiktoken()

        # Statistics
        self._total_calls = 0
        self._tiktoken_calls = 0
        self._heuristic_calls = 0

    def _init_tiktoken(self) -> Any:
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
    def get_instance(cls, encoding_name: str | None = None) -> "SmartTokenizer":
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
    def reset_instances(cls) -> None:
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
        chars_per_token = self.HEURISTICS.get(content_type, self.HEURISTICS["mixed"])

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
        cjk_pattern = re.compile(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]")
        cjk_count = len(cjk_pattern.findall(sample))
        if cjk_count > len(sample) * 0.2:
            return "cjk"

        # Check for code patterns
        code_indicators = [
            r"def\s+\w+\s*\(",  # Python function
            r"function\s+\w+\s*\(",  # JavaScript function
            r"class\s+\w+",  # Class definition
            r"\{\s*\n",  # Code blocks
            r"=>",  # Arrow functions
            r"import\s+",  # Imports
            r"from\s+\w+\s+import",  # Python imports
            r";\s*$",  # Statement endings
        ]
        code_count = sum(len(re.findall(p, sample, re.MULTILINE)) for p in code_indicators)
        if code_count > 5:
            return "code"

        # Check for JSON
        if sample.strip().startswith(("{", "[")) and sample.strip().endswith(("}", "]")):
            try:
                import json

                json.loads(sample[:1000] if len(sample) > 1000 else sample)
                return "json"
            except (json.JSONDecodeError, ValueError):
                # Might still be partial JSON
                if sample.count("{") > 3 or sample.count('"') > 10:
                    return "json"

        # Check for whitespace-heavy content
        whitespace_ratio = len(re.findall(r"\s", sample)) / max(len(sample), 1)
        if whitespace_ratio > 0.4:
            return "whitespace_heavy"

        # Check for mixed CJK
        if cjk_count > 0:
            return "mixed"

        return "english"

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
        numbers = re.findall(r"\d+", text)
        for num in numbers:
            if len(num) > 4:
                # Long numbers get split
                adjustments += len(num) // 3

        # URLs become many tokens
        urls = re.findall(r"https?://\S+", text)
        adjustments += len(urls) * 5

        # Special tokens (newlines, tabs)
        adjustments += text.count("\n") * 0.5  # type: ignore[assignment]
        adjustments += text.count("\t") * 0.5  # type: ignore[assignment]

        # Punctuation clusters
        punct_clusters = re.findall(r"[^\w\s]{3,}", text)
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
            "encoding": self.encoding_name,
            "tiktoken_available": self._tiktoken_available,
            "total_calls": self._total_calls,
            "tiktoken_calls": self._tiktoken_calls,
            "heuristic_calls": self._heuristic_calls,
            "tiktoken_ratio": self._tiktoken_calls / max(self._total_calls, 1),
        }


# =============================================================================
# CONVENIENCE FUNCTIONS (Module-level API)
# =============================================================================


@lru_cache(maxsize=1)
def get_tokenizer(encoding: str | None = None) -> SmartTokenizer:
    """
    Get the default SmartTokenizer instance.

    Cached for performance.

    Args:
        encoding: Optional encoding name

    Returns:
        SmartTokenizer instance
    """
    return SmartTokenizer.get_instance(encoding)


def count_tokens(text: str, encoding: str | None = None) -> int:
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


def estimate_tokens(text: str, encoding: str | None = None) -> int:
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
# MODEL-SPECIFIC TOKEN COUNTING (merged from foundation/token_counter.py)
# =============================================================================

# Lazy import: model_limits_catalog (avoids circular import with foundation/__init__.py)
_limits_catalog_loaded = False
get_limits_from_catalog = None


def _ensure_limits_catalog() -> None:
    global _limits_catalog_loaded, get_limits_from_catalog
    if _limits_catalog_loaded:
        return
    _limits_catalog_loaded = True
    try:
        from Jotty.core.infrastructure.foundation.model_limits_catalog import (
            get_model_limits as _glfc,
        )

        get_limits_from_catalog = _glfc
    except ImportError:
        pass


# Try to import tokencost (for accurate counting when network works)
try:
    from tokencost import count_message_tokens as _tc_count_message_tokens
    from tokencost import count_string_tokens as _tc_count_string_tokens

    TOKENCOST_AVAILABLE = True
except ImportError:
    TOKENCOST_AVAILABLE = False


class TokenCounter:
    """
    Model-specific token counting with tokencost integration.

    Features:
    - Model-specific tokenization (GPT-4, Claude, Llama, etc.)
    - Accurate token counts via tokencost
    - Model limit lookup (max_prompt, max_output)
    - Overflow detection
    """

    MODEL_MAPPING = {
        "gpt-4": "gpt-4",
        "gpt-4.1": "gpt-4.1",
        "gpt-4-turbo": "gpt-4-turbo",
        "gpt-4o": "gpt-4o",
        "gpt-4o-mini": "gpt-4o-mini",
        "gpt-3.5-turbo": "gpt-3.5-turbo",
        "o1-mini": "o1-mini",
        "o1-preview": "o1-preview",
        "claude-3-opus": "claude-3-opus-20240229",
        "claude-3-sonnet": "claude-3-sonnet-20240229",
        "claude-3-haiku": "claude-3-haiku-20240307",
        "claude-3.5-sonnet": "claude-3-5-sonnet-20240620",
        "claude-3.7-sonnet": "claude-3-7-sonnet-20250219",
        "llama-3-70b": "meta-llama/llama-3-70b-instruct",
        "llama-3.3-70b": "meta-llama/llama-3.3-70b-instruct",
        "gemini-pro": "gemini-1.5-pro",
        "gemini-1.5-pro": "gemini-1.5-pro",
        "gemini-2.0-flash": "gemini-2.0-flash",
        "mistral-large": "mistral-large-latest",
        "mistral-medium": "mistral-medium-latest",
    }

    USE_CONSERVATIVE_MODE = False

    def __init__(self, model: Optional[str] = None) -> None:
        if model is None:
            try:
                import dspy

                if hasattr(dspy.settings, "lm") and dspy.settings.lm:
                    lm = dspy.settings.lm
                    if hasattr(lm, "model"):
                        model = lm.model
                    elif hasattr(lm, "kwargs") and "model" in lm.kwargs:
                        model = lm.kwargs["model"]
            except (ImportError, AttributeError, TypeError):
                pass

        self.model = model or "gpt-4.1"
        self.tokencost_model = self._map_model_name(self.model)

    def _map_model_name(self, model: str) -> str:
        """Map DSPy/LiteLLM model name to TokenCost format."""
        if model in self.MODEL_MAPPING:
            return self.MODEL_MAPPING[model]

        model_lower = model.lower()
        for dspy_name, tokencost_name in self.MODEL_MAPPING.items():
            if dspy_name in model_lower or model_lower in dspy_name:
                return tokencost_name

        if "gpt-4o" in model_lower:
            return "gpt-4o"
        elif "gpt-4" in model_lower:
            return "gpt-4"
        elif "gpt-3.5" in model_lower or "gpt-35" in model_lower:
            return "gpt-3.5-turbo"
        elif "claude" in model_lower:
            if "3.5" in model or "3-5" in model:
                return "claude-3-5-sonnet-20240620"
            elif "3.7" in model or "3-7" in model:
                return "claude-3-7-sonnet-20250219"
            return "claude-3-opus-20240229"

        return model

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """Count tokens in text using tokencost (falls back to approximation)."""
        if not text:
            return 0

        model_to_use = model or self.model
        tokencost_model = self._map_model_name(model_to_use) if model else self.tokencost_model

        if TOKENCOST_AVAILABLE:
            try:
                return _tc_count_string_tokens(str(text), model=tokencost_model)  # type: ignore[no-any-return]
            except Exception:
                pass

        return len(str(text)) // 4 + 1

    def count_messages(self, messages: List[Dict[str, Any]], model: Optional[str] = None) -> int:
        """Count tokens in message list (for chat models)."""
        if not messages:
            return 0

        model_to_use = model or self.model
        tokencost_model = self._map_model_name(model_to_use) if model else self.tokencost_model

        if TOKENCOST_AVAILABLE:
            try:
                return _tc_count_message_tokens(messages, model=tokencost_model)  # type: ignore[no-any-return]
            except Exception:
                pass

        total = 0
        for msg in messages:
            content = msg.get("content", "")
            total += len(str(content)) // 4 + 10
        return total

    def get_model_limits(self, model: Optional[str] = None) -> Dict[str, int]:
        """Get model token limits from LOCAL CATALOG (no network required)."""
        _ensure_limits_catalog()
        model_to_use = model or self.model
        tokencost_model = self._map_model_name(model_to_use) if model else self.tokencost_model

        if get_limits_from_catalog is not None:
            return get_limits_from_catalog(tokencost_model, conservative=self.USE_CONSERVATIVE_MODE)  # type: ignore[no-any-return]

        return {"max_prompt": 100000, "max_output": 4096}

    def will_overflow(
        self,
        current_tokens: int,
        additional_tokens: int,
        model: Optional[str] = None,
        safety_margin: float = 0.9,
    ) -> bool:
        """Check if adding tokens will cause context overflow."""
        limits = self.get_model_limits(model)
        max_allowed = int(limits["max_prompt"] * safety_margin)
        return (current_tokens + additional_tokens) > max_allowed

    def get_remaining_tokens(
        self, current_tokens: int, model: Optional[str] = None, safety_margin: float = 0.9
    ) -> int:
        """Get remaining tokens before hitting limit."""
        limits = self.get_model_limits(model)
        max_allowed = int(limits["max_prompt"] * safety_margin)
        return max(0, max_allowed - current_tokens)


# Global TokenCounter instance (lazy initialization)
_default_counter: Optional[TokenCounter] = None


def get_token_counter(model: Optional[str] = None) -> TokenCounter:
    """Get or create default token counter."""
    global _default_counter
    if _default_counter is None:
        _default_counter = TokenCounter(model)
    elif model is not None and _default_counter.model != model:
        _default_counter = TokenCounter(model)
    return _default_counter


def count_tokens_accurate(text: str, model: Optional[str] = None) -> int:
    """Count tokens accurately using tokencost library."""
    if not text:
        return 0
    return get_token_counter(model).count_tokens(text, model)


def count_message_tokens_safe(messages: List[Dict], model: Optional[str] = None) -> int:
    """Count tokens in messages (accurate)."""
    return get_token_counter(model).count_messages(messages, model)


def get_model_limits(model: str) -> Dict[str, int]:
    """Get model token limits."""
    return get_token_counter(model).get_model_limits(model)


def will_overflow(current: int, additional: int, model: str, margin: float = 0.9) -> bool:
    """Check if will overflow."""
    return get_token_counter(model).will_overflow(current, additional, model, margin)


def get_tokenizer_info(model: str) -> Dict[str, Any]:
    """Get information about token counting for a model."""
    try:
        from tokencost import count_string_tokens  # noqa: F401

        limits = get_model_limits(model)
        return {
            "available": True,
            "type": "tokencost",
            "model": model,
            "limits": limits,
            "accurate": True,
            "supported_models": "400+",
        }
    except ImportError:
        return {
            "available": False,
            "type": "tokencost (not installed)",
            "model": model,
            "limits": {"max_prompt": 100000, "max_output": 4096},
            "accurate": False,
            "install": "pip install tokencost>=0.1.26",
        }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "SmartTokenizer",
    "get_tokenizer",
    "count_tokens",
    "estimate_tokens",
    # Model-specific (merged from token_counter)
    "TokenCounter",
    "get_token_counter",
    "count_tokens_accurate",
    "count_message_tokens_safe",
    "get_model_limits",
    "will_overflow",
    "get_tokenizer_info",
]
