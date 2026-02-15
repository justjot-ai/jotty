"""
Model Token Limits Catalog - No Network Required
================================================

Complete catalog of model limits from tokencost pricing table.
Source: https://github.com/AgentOps-AI/tokencost

NO NETWORK DEPENDENCY - All data local.

User requirement: "Max can be more than 30 for different model"
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


# Complete model limits catalog from tokencost pricing table
# Format: 'model_name': {'max_prompt': int, 'max_output': int, 'cost_per_1m_input': float, 'cost_per_1m_output': float}
MODEL_LIMITS_CATALOG = {
    # ============================================================================
    # OPENAI MODELS
    # ============================================================================
    "gpt-4": {"max_prompt": 8192, "max_output": 4096},
    "gpt-4o": {"max_prompt": 128000, "max_output": 16384},
    "gpt-4.1": {"max_prompt": 32000, "max_output": 8192},  # A-TEAM: gpt-4.1 with 32k context
    "gpt-4o-audio-preview": {"max_prompt": 128000, "max_output": 16384},
    "gpt-4o-audio-preview-2024-10-01": {"max_prompt": 128000, "max_output": 16384},
    "gpt-4o-mini": {"max_prompt": 128000, "max_output": 16384},
    "gpt-4o-mini-2024-07-18": {"max_prompt": 128000, "max_output": 16384},
    "o1-mini": {"max_prompt": 128000, "max_output": 65536},
    "o1-mini-2024-09-12": {"max_prompt": 128000, "max_output": 65536},
    "o1-preview": {"max_prompt": 128000, "max_output": 32768},
    "o1-preview-2024-09-12": {"max_prompt": 128000, "max_output": 32768},
    "chatgpt-4o-latest": {"max_prompt": 128000, "max_output": 4096},
    "gpt-4o-2024-05-13": {"max_prompt": 128000, "max_output": 4096},
    "gpt-4o-2024-08-06": {"max_prompt": 128000, "max_output": 16384},
    "gpt-4-turbo-preview": {"max_prompt": 128000, "max_output": 4096},
    "gpt-4-0314": {"max_prompt": 8192, "max_output": 4096},
    "gpt-4-0613": {"max_prompt": 8192, "max_output": 4096},
    "gpt-4-32k": {"max_prompt": 32768, "max_output": 4096},
    "gpt-4-32k-0314": {"max_prompt": 32768, "max_output": 4096},
    "gpt-4-32k-0613": {"max_prompt": 32768, "max_output": 4096},
    "gpt-4-turbo": {"max_prompt": 128000, "max_output": 4096},
    "gpt-4-turbo-2024-04-09": {"max_prompt": 128000, "max_output": 4096},
    "gpt-4-1106-preview": {"max_prompt": 128000, "max_output": 4096},
    "gpt-4-0125-preview": {"max_prompt": 128000, "max_output": 4096},
    "gpt-4-vision-preview": {"max_prompt": 128000, "max_output": 4096},
    "gpt-4-1106-vision-preview": {"max_prompt": 128000, "max_output": 4096},
    "gpt-3.5-turbo": {"max_prompt": 16385, "max_output": 4096},
    "gpt-3.5-turbo-0301": {"max_prompt": 4097, "max_output": 4096},
    "gpt-3.5-turbo-0613": {"max_prompt": 4097, "max_output": 4096},
    "gpt-3.5-turbo-1106": {"max_prompt": 16385, "max_output": 4096},
    "gpt-3.5-turbo-0125": {"max_prompt": 16385, "max_output": 4096},
    "gpt-3.5-turbo-16k": {"max_prompt": 16385, "max_output": 4096},
    "gpt-3.5-turbo-16k-0613": {"max_prompt": 16385, "max_output": 4096},
    # Fine-tuned models
    "ft:gpt-3.5-turbo": {"max_prompt": 16385, "max_output": 4096},
    "ft:gpt-3.5-turbo-0125": {"max_prompt": 16385, "max_output": 4096},
    "ft:gpt-3.5-turbo-1106": {"max_prompt": 16385, "max_output": 4096},
    "ft:gpt-3.5-turbo-0613": {"max_prompt": 4096, "max_output": 4096},
    "ft:gpt-4-0613": {"max_prompt": 8192, "max_output": 4096},
    "ft:gpt-4o-2024-08-06": {"max_prompt": 128000, "max_output": 16384},
    "ft:gpt-4o-mini-2024-07-18": {"max_prompt": 128000, "max_output": 16384},
    # Embeddings
    "text-embedding-3-large": {"max_prompt": 8191, "max_output": 0},
    "text-embedding-3-small": {"max_prompt": 8191, "max_output": 0},
    "text-embedding-ada-002": {"max_prompt": 8191, "max_output": 0},
    # ============================================================================
    # ANTHROPIC MODELS
    # ============================================================================
    "claude-3-opus-20240229": {"max_prompt": 200000, "max_output": 4096},
    "claude-3-sonnet-20240229": {"max_prompt": 200000, "max_output": 4096},
    "claude-3-haiku-20240307": {"max_prompt": 200000, "max_output": 4096},
    "claude-3-5-sonnet-20240620": {"max_prompt": 200000, "max_output": 8192},
    "claude-3-5-sonnet-20241022": {"max_prompt": 200000, "max_output": 8192},
    "claude-3-5-haiku-20241022": {"max_prompt": 200000, "max_output": 8192},
    # Short names
    "claude-3-opus": {"max_prompt": 200000, "max_output": 4096},
    "claude-3-sonnet": {"max_prompt": 200000, "max_output": 4096},
    "claude-3-haiku": {"max_prompt": 200000, "max_output": 4096},
    "claude-3.5-sonnet": {"max_prompt": 200000, "max_output": 8192},
    "claude-3.5-haiku": {"max_prompt": 200000, "max_output": 8192},
    # ============================================================================
    # GOOGLE MODELS
    # ============================================================================
    "gemini-1.5-pro": {"max_prompt": 2000000, "max_output": 8192},
    "gemini-1.5-pro-002": {"max_prompt": 2000000, "max_output": 8192},
    "gemini-1.5-flash": {"max_prompt": 1000000, "max_output": 8192},
    "gemini-1.5-flash-002": {"max_prompt": 1000000, "max_output": 8192},
    "gemini-2.0-flash": {"max_prompt": 1048576, "max_output": 8192},
    "gemini-2.0-flash-lite": {"max_prompt": 1048576, "max_output": 8192},
    "gemini-pro": {"max_prompt": 32768, "max_output": 8192},
    "gemini-pro-vision": {"max_prompt": 16384, "max_output": 8192},
    # ============================================================================
    # META LLAMA MODELS
    # ============================================================================
    "meta-llama/llama-3-8b-instruct": {"max_prompt": 8192, "max_output": 8192},
    "meta-llama/llama-3-70b-instruct": {"max_prompt": 8192, "max_output": 8192},
    "meta-llama/llama-3.1-8b-instruct": {"max_prompt": 128000, "max_output": 8192},
    "meta-llama/llama-3.1-70b-instruct": {"max_prompt": 128000, "max_output": 8192},
    "meta-llama/llama-3.1-405b-instruct": {"max_prompt": 128000, "max_output": 8192},
    "meta-llama/llama-3.2-1b-instruct": {"max_prompt": 128000, "max_output": 8192},
    "meta-llama/llama-3.2-3b-instruct": {"max_prompt": 128000, "max_output": 8192},
    "meta-llama/llama-3.2-11b-vision-instruct": {"max_prompt": 128000, "max_output": 8192},
    "meta-llama/llama-3.2-90b-vision-instruct": {"max_prompt": 128000, "max_output": 8192},
    "meta-llama/llama-3.3-70b-instruct": {"max_prompt": 128000, "max_output": 8192},
    # Short names
    "llama-3-8b": {"max_prompt": 8192, "max_output": 8192},
    "llama-3-70b": {"max_prompt": 8192, "max_output": 8192},
    "llama-3.1-8b": {"max_prompt": 128000, "max_output": 8192},
    "llama-3.1-70b": {"max_prompt": 128000, "max_output": 8192},
    "llama-3.1-405b": {"max_prompt": 128000, "max_output": 8192},
    "llama-3.2-1b": {"max_prompt": 128000, "max_output": 8192},
    "llama-3.2-3b": {"max_prompt": 128000, "max_output": 8192},
    "llama-3.3-70b": {"max_prompt": 128000, "max_output": 8192},
    # ============================================================================
    # MISTRAL MODELS
    # ============================================================================
    "mistral-large-latest": {"max_prompt": 128000, "max_output": 128000},
    "mistral-large-2407": {"max_prompt": 128000, "max_output": 128000},
    "mistral-medium-latest": {"max_prompt": 32768, "max_output": 32768},
    "mistral-small-latest": {"max_prompt": 32768, "max_output": 32768},
    "mistral-tiny": {"max_prompt": 32768, "max_output": 32768},
    "codestral-latest": {"max_prompt": 256000, "max_output": 4000},
    "codestral-2405": {"max_prompt": 256000, "max_output": 4000},
    "pixtral-12b-2409": {"max_prompt": 128000, "max_output": 4000},
    # ============================================================================
    # COHERE MODELS
    # ============================================================================
    "command-r": {"max_prompt": 128000, "max_output": 4096},
    "command-r-plus": {"max_prompt": 128000, "max_output": 4096},
    "command": {"max_prompt": 4096, "max_output": 4096},
    "command-light": {"max_prompt": 4096, "max_output": 4096},
    # ============================================================================
    # DEEPSEEK MODELS
    # ============================================================================
    "deepseek-chat": {"max_prompt": 64000, "max_output": 8192},
    "deepseek-coder": {"max_prompt": 64000, "max_output": 8192},
    # ============================================================================
    # XAI MODELS (Grok)
    # ============================================================================
    "grok-beta": {"max_prompt": 131072, "max_output": 131072},
    "grok-vision-beta": {"max_prompt": 8192, "max_output": 8192},
    # ============================================================================
    # PERPLEXITY MODELS
    # ============================================================================
    "perplexity/llama-3.1-sonar-small-128k-online": {"max_prompt": 127000, "max_output": 8000},
    "perplexity/llama-3.1-sonar-large-128k-online": {"max_prompt": 127000, "max_output": 8000},
    "perplexity/llama-3.1-sonar-huge-128k-online": {"max_prompt": 127000, "max_output": 8000},
}


def get_model_limits(model_name: str, conservative: bool = False) -> Dict[str, int]:
    """
    Get token limits for a model from local catalog.

    NO NETWORK REQUIRED - All data is local.

    Args:
        model_name: Model name (e.g., 'gpt-4o', 'claude-3-opus', 'llama-3-70b')
        conservative: If True, use 30k cap even if model supports more

    Returns:
        Dict with 'max_prompt' and 'max_output' keys
    """
    # Normalize model name
    model_lower = model_name.lower().strip()

    # Try direct match
    if model_name in MODEL_LIMITS_CATALOG:
        limits = MODEL_LIMITS_CATALOG[model_name]
        logger.debug(f" Found exact match for '{model_name}': {limits}")

        # Apply conservative cap if requested
        if conservative and limits["max_prompt"] > 30000:
            logger.info(
                f" Conservative mode: Capping {model_name} from "
                f"{limits['max_prompt']:,} to 30,000 tokens"
            )
            return {"max_prompt": 30000, "max_output": limits["max_output"]}

        return limits

    # Try lowercase match
    for key, limits in MODEL_LIMITS_CATALOG.items():
        if key.lower() == model_lower:
            logger.debug(f" Found case-insensitive match: '{key}' for '{model_name}'")

            if conservative and limits["max_prompt"] > 30000:
                return {"max_prompt": 30000, "max_output": limits["max_output"]}

            return limits

    # Try partial match
    for key, limits in MODEL_LIMITS_CATALOG.items():
        key_lower = key.lower()
        if model_lower in key_lower or key_lower in model_lower:
            logger.info(f" Found partial match: '{key}' for '{model_name}'")

            if conservative and limits["max_prompt"] > 30000:
                return {"max_prompt": 30000, "max_output": limits["max_output"]}

            return limits

    # Try extracting base model (e.g., 'openai/gpt-4o' → 'gpt-4o')
    if "/" in model_name:
        base_model = model_name.split("/")[-1]
        logger.info(f" Trying base model: '{base_model}' from '{model_name}'")
        return get_model_limits(base_model, conservative)

    # Default fallback
    logger.warning(
        f" Model '{model_name}' not found in catalog. " f"Using conservative default: 30k/4k tokens"
    )
    return {"max_prompt": 30000, "max_output": 4096}


def list_supported_models() -> Dict[str, Dict[str, int]]:
    """
    Get full catalog of supported models.

    Returns:
        Dict of model_name -> limits
    """
    return MODEL_LIMITS_CATALOG.copy()


def get_models_by_provider(provider: str) -> Dict[str, Dict[str, int]]:
    """
    Get models for a specific provider.

    Args:
        provider: 'openai', 'anthropic', 'google', 'meta', 'mistral', etc.

    Returns:
        Dict of matching models
    """
    provider_lower = provider.lower()

    provider_patterns = {
        "openai": ["gpt-", "text-embedding", "ft:gpt", "o1-", "chatgpt"],
        "anthropic": ["claude-"],
        "google": ["gemini-"],
        "meta": ["llama-", "meta-llama/"],
        "mistral": ["mistral-", "codestral", "pixtral"],
        "cohere": ["command"],
        "deepseek": ["deepseek"],
        "xai": ["grok"],
        "perplexity": ["perplexity/"],
    }

    patterns = provider_patterns.get(provider_lower, [])

    matching = {}
    for model, limits in MODEL_LIMITS_CATALOG.items():
        if any(pattern in model.lower() for pattern in patterns):
            matching[model] = limits

    return matching


def get_max_context_models(min_tokens: int = 100000) -> Dict[str, Dict[str, int]]:
    """
    Get models with context >= min_tokens.

    Useful for finding models that can handle large contexts.

    Args:
        min_tokens: Minimum context length

    Returns:
        Dict of matching models sorted by context size
    """
    matching = {
        model: limits
        for model, limits in MODEL_LIMITS_CATALOG.items()
        if limits["max_prompt"] >= min_tokens
    }

    # Sort by context size (descending)
    return dict(sorted(matching.items(), key=lambda x: x[1]["max_prompt"], reverse=True))


# Helper for backward compatibility
def get_model_info(model_name: str) -> Dict[str, int]:
    """Alias for get_model_limits (backward compatibility)."""
    return get_model_limits(model_name)
