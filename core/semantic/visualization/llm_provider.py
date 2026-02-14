"""
Claude LLM Provider for LIDA

Integrates Jotty's core.llm module with LIDA's TextGenerator interface.
Uses Claude CLI by default with fallback support.

This allows LIDA to use the same LLM infrastructure as other Jotty skills.
"""
import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass

from Jotty.core.foundation.config_defaults import DEFAULT_MODEL_ALIAS, LLM_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Message in a conversation (compatible with llmx.datamodel.Message)."""
    role: str
    content: str

    def __getitem__(self, key: Any) -> Any:
        """Support dict-like access for compatibility."""
        if key == 'role':
            return self.role
        elif key == 'content':
            return self.content
        raise KeyError(key)

    def get(self, key: Any, default: Any = None) -> Any:
        """Support dict-like get method."""
        try:
            return self[key]
        except KeyError:
            return default


@dataclass
class TextGenerationConfig:
    """Configuration for text generation."""
    n: int = 1
    temperature: float = 0.1
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    top_k: int = 50
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    provider: Optional[str] = None
    model: Optional[str] = None
    stop: Optional[List[str]] = None
    use_cache: bool = True


@dataclass
class TextGenerationResponse:
    """Response from text generation."""
    text: List[Message]
    config: TextGenerationConfig
    logprobs: Optional[Any] = None
    usage: Optional[Dict[str, int]] = None
    response: Optional[Any] = None


class ClaudeLLMTextGenerator:
    """
    LIDA-compatible text generator using Jotty's core.llm module.

    Uses Claude CLI by default (no API keys required).

    Usage:
        from core.semantic.visualization.llm_provider import ClaudeLLMTextGenerator
        from lida import Manager

        text_gen = ClaudeLLMTextGenerator(model="sonnet")
        manager = Manager(text_gen=text_gen)
    """

    def __init__(self, provider: str = 'claude-cli', model: str = DEFAULT_MODEL_ALIAS, timeout: int = LLM_TIMEOUT_SECONDS, fallback: bool = True, **kwargs: Any) -> None:
        """
        Initialize Claude LLM text generator.

        Args:
            provider: LLM provider (claude-cli, anthropic, gemini, openai)
            model: Model name (sonnet, opus, haiku for Claude)
            timeout: Timeout in seconds
            fallback: Enable fallback to other providers on failure
            **kwargs: Additional arguments passed to core.llm
        """
        self.provider = provider
        self.model = model
        self.timeout = timeout
        self.fallback = fallback
        self.kwargs = kwargs

        # Import core.llm lazily to avoid circular imports
        self._llm_generate = None

    @property
    def llm_generate(self) -> Any:
        """Lazy import of core.llm.generate."""
        if self._llm_generate is None:
            from core.llm import generate
            self._llm_generate = generate
        return self._llm_generate

    def generate(self, messages: Union[List[Dict], str], config: TextGenerationConfig = None, **kwargs: Any) -> TextGenerationResponse:
        """
        Generate text using Claude LLM.

        Args:
            messages: List of message dicts or string prompt
            config: Generation configuration
            **kwargs: Additional arguments

        Returns:
            TextGenerationResponse with generated text
        """
        config = config or TextGenerationConfig()

        # Convert messages to prompt
        if isinstance(messages, str):
            prompt = messages
        elif isinstance(messages, list):
            # Convert message list to prompt string
            prompt_parts = []
            for msg in messages:
                if isinstance(msg, dict):
                    role = msg.get('role', 'user')
                    content = msg.get('content', '')
                elif hasattr(msg, 'role') and hasattr(msg, 'content'):
                    role = msg.role
                    content = msg.content
                else:
                    content = str(msg)
                    role = 'user'

                if role == 'system':
                    prompt_parts.append(f"System: {content}")
                elif role == 'user':
                    prompt_parts.append(f"User: {content}")
                elif role == 'assistant':
                    prompt_parts.append(f"Assistant: {content}")
                else:
                    prompt_parts.append(content)

            prompt = "\n\n".join(prompt_parts)
        else:
            prompt = str(messages)

        # Use provider/model from config if specified
        provider = config.provider or self.provider
        model = config.model or self.model
        max_tokens = config.max_tokens or 4096

        try:
            # Call core.llm.generate
            response = self.llm_generate(
                prompt=prompt,
                provider=provider,
                model=model,
                timeout=self.timeout,
                fallback=self.fallback,
                max_tokens=max_tokens,
                **self.kwargs
            )

            if response.success:
                return TextGenerationResponse(
                    text=[Message(role="assistant", content=response.text)],
                    config=config,
                    usage=response.usage or {"total_tokens": len(response.text.split())}
                )
            else:
                logger.error(f"LLM generation failed: {response.error}")
                return TextGenerationResponse(
                    text=[Message(role="assistant", content=f"Error: {response.error}")],
                    config=config,
                    usage={"total_tokens": 0}
                )

        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return TextGenerationResponse(
                text=[Message(role="assistant", content=f"Error: {str(e)}")],
                config=config,
                usage={"total_tokens": 0}
            )

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text (approximate).

        Args:
            text: Text to count tokens for

        Returns:
            Approximate token count
        """
        # Simple approximation: ~4 chars per token for English
        # More accurate would use tiktoken or anthropic's tokenizer
        return len(text) // 4


def get_lida_text_generator(provider: str = 'claude-cli', model: str = DEFAULT_MODEL_ALIAS, **kwargs: Any) -> ClaudeLLMTextGenerator:
    """
    Get a LIDA-compatible text generator using Jotty's LLM infrastructure.

    Args:
        provider: LLM provider (claude-cli, anthropic, gemini, openai)
        model: Model name
        **kwargs: Additional arguments

    Returns:
        ClaudeLLMTextGenerator instance
    """
    return ClaudeLLMTextGenerator(provider=provider, model=model, **kwargs)


__all__ = [
    'ClaudeLLMTextGenerator',
    'get_lida_text_generator',
    'Message',
    'TextGenerationConfig',
    'TextGenerationResponse',
]
