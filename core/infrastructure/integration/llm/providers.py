"""
LLM Provider Implementations

Individual provider implementations for different LLM backends.
All providers use stdin for prompts to handle long inputs properly.
"""
import subprocess
import os
import logging
from typing import Dict, Any, Optional

from Jotty.core.infrastructure.foundation.config_defaults import (
    LLM_MAX_OUTPUT_TOKENS, LLM_TIMEOUT_SECONDS, DEFAULT_MODEL_ALIAS,
)
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ProviderType(Enum):
    """Supported LLM provider types."""
    CLAUDE_CLI = "claude-cli"
    ANTHROPIC_API = "anthropic"
    GEMINI = "gemini"
    OPENAI = "openai"
    GROQ = "groq"


@dataclass
class LLMResponse:
    """Standardized LLM response."""
    success: bool
    text: str = ""
    error: str = ""
    model: str = ""
    provider: str = ""
    usage: Optional[Dict[str, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "success": self.success,
            "text": self.text,
            "model": self.model,
            "provider": self.provider,
        }
        if self.error:
            result["error"] = self.error
        if self.usage:
            result["usage"] = self.usage
        return result


# Model mappings — centralized in config_defaults
from Jotty.core.infrastructure.foundation.config_defaults import MODEL_SONNET, MODEL_OPUS, MODEL_HAIKU
ANTHROPIC_MODELS = {
    "haiku": MODEL_HAIKU,
    "sonnet": MODEL_SONNET,
    "opus": MODEL_OPUS,
}

GEMINI_MODELS = {
    "flash": "gemini-2.0-flash",
    "pro": "gemini-1.5-pro",
}

OPENAI_MODELS = {
    "gpt4": "gpt-4-turbo",
    "gpt4o": "gpt-4o",
    "gpt35": "gpt-3.5-turbo",
}


class ClaudeCLIProvider:
    """
    Claude CLI provider using subprocess with stdin.

    Uses stdin for prompts to handle long inputs properly.
    """

    @staticmethod
    def generate(prompt: str, model: str = DEFAULT_MODEL_ALIAS, timeout: int = LLM_TIMEOUT_SECONDS, **kwargs: Any) -> LLMResponse:
        """
        Generate text using Claude CLI.

        Args:
            prompt: The prompt text
            model: Model name (haiku, sonnet, opus)
            timeout: Timeout in seconds

        Returns:
            LLMResponse with result or error
        """
        try:
            # Build command - use stdin for prompt (handles long inputs)
            # Add --system-prompt to ignore workspace context (isolate from git/codebase)
            system_prompt = (
                "You are a helpful AI assistant. Only respond to the user's prompt. "
                "Ignore any workspace, git, codebase, or file system context. "
                "Focus solely on answering the user's question directly."
            )
            cmd = ["claude", "--model", model, "-p", "--system-prompt", system_prompt]

            # Handle OAuth tokens that don't work with --print mode
            env = os.environ.copy()
            api_key = env.get("ANTHROPIC_API_KEY", "")
            if api_key.startswith("sk-ant-oat"):
                env.pop("ANTHROPIC_API_KEY", None)

            # Execute with prompt via stdin
            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                env=env,
                timeout=timeout
            )

            if result.returncode != 0:
                return LLMResponse(
                    success=False,
                    error=f"Claude CLI error: {result.stderr}",
                    provider="claude-cli",
                    model=model
                )

            return LLMResponse(
                success=True,
                text=result.stdout.strip(),
                provider="claude-cli",
                model=model
            )

        except subprocess.TimeoutExpired:
            return LLMResponse(
                success=False,
                error=f"Claude CLI timeout after {timeout} seconds",
                provider="claude-cli",
                model=model
            )
        except FileNotFoundError:
            return LLMResponse(
                success=False,
                error="Claude CLI not found. Install with: npm install -g @anthropic-ai/claude-code",
                provider="claude-cli",
                model=model
            )
        except Exception as e:
            logger.error(f"Claude CLI error: {e}", exc_info=True)
            return LLMResponse(
                success=False,
                error=str(e),
                provider="claude-cli",
                model=model
            )


class AnthropicAPIProvider:
    """
    Anthropic API provider using the official SDK.
    """

    @staticmethod
    def generate(prompt: str, model: str = DEFAULT_MODEL_ALIAS, max_tokens: int = LLM_MAX_OUTPUT_TOKENS, **kwargs: Any) -> LLMResponse:
        """
        Generate text using Anthropic API.

        Args:
            prompt: The prompt text
            model: Model name (haiku, sonnet, opus) or full model ID
            max_tokens: Maximum tokens to generate

        Returns:
            LLMResponse with result or error
        """
        try:
            import anthropic
        except ImportError:
            return LLMResponse(
                success=False,
                error="anthropic package not installed. Install with: pip install anthropic",
                provider="anthropic",
                model=model
            )

        from Jotty.core.infrastructure.foundation.anthropic_client_kwargs import get_anthropic_client_kwargs
        client_kwargs = get_anthropic_client_kwargs()
        if not client_kwargs.get("api_key"):
            return LLMResponse(
                success=False,
                error="ANTHROPIC_API_KEY not set",
                provider="anthropic",
                model=model
            )

        try:
            client = anthropic.Anthropic(**client_kwargs)

            # Map short names to full model IDs
            model_id = ANTHROPIC_MODELS.get(model, model)

            message = client.messages.create(
                model=model_id,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )

            text = message.content[0].text if message.content else ""

            return LLMResponse(
                success=True,
                text=text,
                provider="anthropic",
                model=model_id,
                usage={
                    "input_tokens": message.usage.input_tokens,
                    "output_tokens": message.usage.output_tokens
                }
            )

        except Exception as e:
            logger.error(f"Anthropic API error: {e}", exc_info=True)
            return LLMResponse(
                success=False,
                error=str(e),
                provider="anthropic",
                model=model
            )


class GeminiProvider:
    """
    Google Gemini provider using the official SDK.
    """

    @staticmethod
    def generate(prompt: str, model: str = 'flash', **kwargs: Any) -> LLMResponse:
        """
        Generate text using Gemini API.

        Args:
            prompt: The prompt text
            model: Model name (flash, pro) or full model ID

        Returns:
            LLMResponse with result or error
        """
        try:
            import google.generativeai as genai
        except ImportError:
            return LLMResponse(
                success=False,
                error="google-generativeai package not installed. Install with: pip install google-generativeai",
                provider="gemini",
                model=model
            )

        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return LLMResponse(
                success=False,
                error="GOOGLE_API_KEY or GEMINI_API_KEY not set",
                provider="gemini",
                model=model
            )

        try:
            genai.configure(api_key=api_key)

            # Map short names to full model IDs
            model_id = GEMINI_MODELS.get(model, model)

            gen_model = genai.GenerativeModel(model_id)
            response = gen_model.generate_content(prompt)

            return LLMResponse(
                success=True,
                text=response.text,
                provider="gemini",
                model=model_id
            )

        except Exception as e:
            logger.error(f"Gemini API error: {e}", exc_info=True)
            return LLMResponse(
                success=False,
                error=str(e),
                provider="gemini",
                model=model
            )


class OpenAIProvider:
    """
    OpenAI API provider using the official SDK.
    """

    @staticmethod
    def generate(prompt: str, model: str = 'gpt4o', max_tokens: int = LLM_MAX_OUTPUT_TOKENS, **kwargs: Any) -> LLMResponse:
        """
        Generate text using OpenAI API.

        Args:
            prompt: The prompt text
            model: Model name (gpt4, gpt4o, gpt35) or full model ID
            max_tokens: Maximum tokens to generate

        Returns:
            LLMResponse with result or error
        """
        try:
            import openai
        except ImportError:
            return LLMResponse(
                success=False,
                error="openai package not installed. Install with: pip install openai",
                provider="openai",
                model=model
            )

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return LLMResponse(
                success=False,
                error="OPENAI_API_KEY not set",
                provider="openai",
                model=model
            )

        try:
            client = openai.OpenAI(api_key=api_key)

            # Map short names to full model IDs
            model_id = OPENAI_MODELS.get(model, model)

            response = client.chat.completions.create(
                model=model_id,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )

            text = response.choices[0].message.content if response.choices else ""

            return LLMResponse(
                success=True,
                text=text,
                provider="openai",
                model=model_id,
                usage={
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens
                } if response.usage else None
            )

        except Exception as e:
            logger.error(f"OpenAI API error: {e}", exc_info=True)
            return LLMResponse(
                success=False,
                error=str(e),
                provider="openai",
                model=model
            )


# Provider registry
PROVIDERS = {
    "claude-cli": ClaudeCLIProvider,
    "anthropic": AnthropicAPIProvider,
    "gemini": GeminiProvider,
    "openai": OpenAIProvider,
}


def get_provider(provider_name: str) -> Any:
    """Get a provider class by name."""
    return PROVIDERS.get(provider_name)


def list_providers() -> list:
    """List all available provider names."""
    return list(PROVIDERS.keys())
