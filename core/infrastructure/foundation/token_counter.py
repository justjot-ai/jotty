"""
Token Counter — Re-export Shim
================================

All token counting functionality now lives in ``utils.tokenizer``.
This module re-exports for backward compatibility.
"""

from Jotty.core.infrastructure.utils.tokenizer import (
    TokenCounter,
    count_message_tokens_safe,
    count_tokens_accurate,
    estimate_tokens,
    get_model_limits,
    get_token_counter,
    get_tokenizer_info,
    will_overflow,
)

# Re-export count_tokens from TokenCounter (model-specific version)
count_tokens = count_tokens_accurate

__all__ = [
    "TokenCounter",
    "get_token_counter",
    "count_tokens",
    "count_tokens_accurate",
    "count_message_tokens_safe",
    "get_model_limits",
    "will_overflow",
    "estimate_tokens",
    "get_tokenizer_info",
]
