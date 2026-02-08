"""
LLM Provider Factory
====================

Factory for creating LLM providers based on configuration.
Routes to local or cloud providers based on local_mode setting.
"""

import os
import logging
from typing import Optional, Union

from .providers import (
    ClaudeCLIProvider,
    AnthropicAPIProvider,
    GeminiProvider,
    OpenAIProvider,
    LLMResponse
)
from .local_provider import LocalLLMProvider, LocalLLMResponse

logger = logging.getLogger(__name__)


class ProviderFactory:
    """
    Factory for creating LLM providers.

    Automatically routes to local or cloud providers based on:
    1. local_mode in config
    2. JOTTY_LOCAL_MODE environment variable
    3. Explicit provider selection
    """

    @staticmethod
    def get_provider(
        provider: str = "auto",
        config=None,
        local_mode: Optional[bool] = None
    ):
        """
        Get an LLM provider instance.

        Args:
            provider: Provider name or "auto" for automatic selection
                      Options: "auto", "claude-cli", "anthropic", "gemini",
                               "openai", "ollama", "local"
            config: Optional SwarmConfig with local_mode setting
            local_mode: Override local_mode (if not using config)

        Returns:
            Provider instance with generate() method
        """
        # Determine if we should use local mode
        use_local = False

        if local_mode is not None:
            use_local = local_mode
        elif config is not None and hasattr(config, 'local_mode'):
            use_local = config.local_mode
        else:
            use_local = os.getenv("JOTTY_LOCAL_MODE", "").lower() == "true"

        # If local mode requested, return local provider
        if use_local or provider in ("local", "ollama", "llamacpp"):
            model = "ollama/llama3"
            if config and hasattr(config, 'local_model'):
                model = config.local_model
            elif provider == "ollama":
                model = os.getenv("OLLAMA_MODEL", "llama3")
                model = f"ollama/{model}"
            elif provider == "llamacpp":
                model = "llamacpp/model"

            return LocalLLMProvider(model)

        # Auto-select cloud provider based on available API keys
        if provider == "auto":
            if os.getenv("ANTHROPIC_API_KEY"):
                return AnthropicAPIProvider
            elif os.getenv("OPENAI_API_KEY"):
                return OpenAIProvider
            elif os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
                return GeminiProvider
            else:
                # Fall back to Claude CLI if installed
                return ClaudeCLIProvider

        # Explicit provider selection
        providers = {
            "claude-cli": ClaudeCLIProvider,
            "anthropic": AnthropicAPIProvider,
            "gemini": GeminiProvider,
            "openai": OpenAIProvider,
        }

        if provider in providers:
            return providers[provider]

        raise ValueError(f"Unknown provider: {provider}")

    @staticmethod
    def generate(
        prompt: str,
        provider: str = "auto",
        config=None,
        **kwargs
    ) -> Union[LLMResponse, LocalLLMResponse]:
        """
        Generate text using the appropriate provider.

        This is a convenience method that creates a provider and generates in one call.

        Args:
            prompt: The prompt text
            provider: Provider name or "auto"
            config: Optional SwarmConfig
            **kwargs: Additional arguments passed to generate()

        Returns:
            LLMResponse or LocalLLMResponse
        """
        prov = ProviderFactory.get_provider(provider, config)

        # Handle both class-based and instance-based providers
        if isinstance(prov, LocalLLMProvider):
            return prov.generate_sync(prompt, **kwargs)
        elif hasattr(prov, 'generate'):
            # Class with static generate method
            return prov.generate(prompt, **kwargs)
        else:
            raise RuntimeError(f"Provider {prov} does not have generate method")

    @staticmethod
    def list_available() -> dict:
        """List all available providers and their status."""
        result = {
            "cloud": {},
            "local": {}
        }

        # Check cloud providers
        result["cloud"]["anthropic"] = bool(os.getenv("ANTHROPIC_API_KEY"))
        result["cloud"]["openai"] = bool(os.getenv("OPENAI_API_KEY"))
        result["cloud"]["gemini"] = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))

        # Check Claude CLI
        import shutil
        result["cloud"]["claude-cli"] = shutil.which("claude") is not None

        # Check local providers
        result["local"] = LocalLLMProvider.is_available()

        return result


# Convenience function
def get_provider(provider: str = "auto", config=None, local_mode: Optional[bool] = None):
    """Get an LLM provider. See ProviderFactory.get_provider for details."""
    return ProviderFactory.get_provider(provider, config, local_mode)
