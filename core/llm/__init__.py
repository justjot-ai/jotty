"""
Core LLM Module

Unified LLM interface for all Jotty components.

Usage:
    # Quick generation
    from core.llm import generate, generate_text

    response = generate("What is Python?")
    text = generate_text("What is Python?")

    # With specific provider
    response = generate("What is Python?", provider="anthropic")

    # With fallback
    response = generate("What is Python?", fallback=True)

    # Full control
    from core.llm import UnifiedLLM

    llm = UnifiedLLM(default_provider="claude-cli", default_model="sonnet")
    response = llm.generate("What is Python?")

    # Direct provider access
    from core.llm.providers import ClaudeCLIProvider

    response = ClaudeCLIProvider.generate("What is Python?")
"""

from .providers import (
    LLMResponse,
    ProviderType,
    ClaudeCLIProvider,
    AnthropicAPIProvider,
    GeminiProvider,
    OpenAIProvider,
    PROVIDERS,
    get_provider,
    list_providers,
    ANTHROPIC_MODELS,
    GEMINI_MODELS,
    OPENAI_MODELS,
)

from .unified import (
    UnifiedLLM,
    get_llm,
    generate,
    generate_text,
)

from .local_provider import (
    LocalLLMProvider,
    LocalLLMResponse,
)

from .provider_factory import (
    ProviderFactory,
    get_provider as get_provider_auto,
)

__all__ = [
    # Response type
    "LLMResponse",
    "ProviderType",

    # Providers
    "ClaudeCLIProvider",
    "AnthropicAPIProvider",
    "GeminiProvider",
    "OpenAIProvider",
    "PROVIDERS",
    "get_provider",
    "list_providers",

    # Model mappings
    "ANTHROPIC_MODELS",
    "GEMINI_MODELS",
    "OPENAI_MODELS",

    # Unified interface
    "UnifiedLLM",
    "get_llm",
    "generate",
    "generate_text",

    # Local-first (OpenClaw-inspired)
    "LocalLLMProvider",
    "LocalLLMResponse",
    "ProviderFactory",
    "get_provider_auto",
]
