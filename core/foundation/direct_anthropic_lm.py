#!/usr/bin/env python3
"""
Direct Anthropic API DSPy LM Provider
=====================================

Uses the Anthropic Python SDK directly - no subprocess overhead.
~10x faster than Claude CLI subprocess calls.

Requires: ANTHROPIC_API_KEY environment variable

Latency comparison:
- Claude CLI subprocess: ~3s (0.5s subprocess + 2.5s inference)
- Direct API: ~0.5s (just inference, no subprocess)
"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional
import dspy

logger = logging.getLogger(__name__)

# Model name mapping — centralized in config_defaults
from Jotty.core.foundation.config_defaults import (
    MODEL_SONNET, MODEL_OPUS, MODEL_HAIKU,
    DEFAULT_MODEL_ALIAS, LLM_TEMPERATURE, LLM_TIMEOUT_SECONDS,
)
MODEL_MAP = {
    "haiku": MODEL_HAIKU,
    "sonnet": MODEL_SONNET,
    "opus": MODEL_OPUS,
}


class DirectAnthropicLM(dspy.BaseLM):
    """
    Direct Anthropic API LM - fastest option when API key is available.

    No subprocess overhead, direct HTTP calls to Anthropic API.
    """

    def __init__(
        self,
        model: str = "haiku",
        max_tokens: Optional[int] = None,
        temperature: float = LLM_TEMPERATURE,
        timeout: int = LLM_TIMEOUT_SECONDS,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Direct Anthropic LM.

        Args:
            model: Model alias (haiku, sonnet, opus) or full model name
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            timeout: Request timeout in seconds
            api_key: API key (defaults to ANTHROPIC_API_KEY env var)
        """
        # Resolve model name
        self.model_id = MODEL_MAP.get(model, model)

        # Centralized default for max output tokens
        if max_tokens is None:
            from Jotty.core.foundation.config_defaults import LLM_MAX_OUTPUT_TOKENS
            max_tokens = LLM_MAX_OUTPUT_TOKENS

        super().__init__(model=f"anthropic-direct/{model}", **kwargs)
        self.model_alias = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.provider = "anthropic-direct"
        self.history: List[Dict[str, Any]] = []

        # Get API key
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Either:\n"
                "1. Set ANTHROPIC_API_KEY environment variable\n"
                "2. Pass api_key parameter\n"
                "3. Use AsyncClaudeCLILM instead (uses CLI auth)"
            )

        # Initialize client
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
            self._async_client = anthropic.AsyncAnthropic(api_key=self.api_key)
            logger.info(f"DirectAnthropicLM initialized (model={self.model_id})")
        except ImportError:
            raise RuntimeError("anthropic package not installed. Install with: pip install anthropic")

    def __call__(
        self,
        prompt: str = None,
        messages: List[Dict] = None,
        **kwargs
    ) -> List[str]:
        """Synchronous call interface (required by DSPy)."""
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self._async_call(prompt, messages, **kwargs)
                )
                return future.result(timeout=self.timeout + 10)
        except RuntimeError:
            # Not in async context - use sync client directly
            return self._sync_call(prompt, messages, **kwargs)

    def _sync_call(
        self,
        prompt: str = None,
        messages: List[Dict] = None,
        **kwargs
    ) -> List[str]:
        """Synchronous API call using sync client."""
        # Build messages and extract system prompt
        system_prompt = None
        if prompt:
            api_messages = [{"role": "user", "content": prompt}]
        elif messages:
            api_messages, system_prompt = self._convert_messages(messages)
        else:
            raise ValueError("Either prompt or messages must be provided")

        try:
            # Build request kwargs
            request_kwargs = {
                "model": self.model_id,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": api_messages,
            }
            if system_prompt:
                request_kwargs["system"] = system_prompt

            response = self._client.messages.create(**request_kwargs)

            # Extract text from response
            response_text = ""
            for block in response.content:
                if block.type == "text":
                    response_text += block.text

            self.history.append({
                'prompt': str(api_messages)[:500],
                'response': response_text[:500],
                'model': self.model_id,
                'usage': {
                    'input_tokens': response.usage.input_tokens,
                    'output_tokens': response.usage.output_tokens,
                }
            })

            return [response_text]

        except Exception as e:
            logger.error(f"DirectAnthropicLM error: {e}")
            raise

    async def _async_call(
        self,
        prompt: str = None,
        messages: List[Dict] = None,
        **kwargs
    ) -> List[str]:
        """Async API call."""
        # Build messages and extract system prompt
        system_prompt = None
        if prompt:
            api_messages = [{"role": "user", "content": prompt}]
        elif messages:
            api_messages, system_prompt = self._convert_messages(messages)
        else:
            raise ValueError("Either prompt or messages must be provided")

        try:
            # Build request kwargs
            request_kwargs = {
                "model": self.model_id,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": api_messages,
            }
            if system_prompt:
                request_kwargs["system"] = system_prompt

            response = await asyncio.wait_for(
                self._async_client.messages.create(**request_kwargs),
                timeout=self.timeout
            )

            # Extract text from response
            response_text = ""
            for block in response.content:
                if block.type == "text":
                    response_text += block.text

            self.history.append({
                'prompt': str(api_messages)[:500],
                'response': response_text[:500],
                'model': self.model_id,
            })

            return [response_text]

        except asyncio.TimeoutError:
            raise RuntimeError(f"API call timed out after {self.timeout}s")
        except Exception as e:
            logger.error(f"DirectAnthropicLM async error: {e}")
            raise

    def _convert_messages(self, messages: List[Dict]) -> tuple:
        """Convert DSPy messages to Anthropic format.

        Returns:
            Tuple of (api_messages, system_prompt)
            - api_messages: List of user/assistant messages
            - system_prompt: Extracted system message content or None
        """
        api_messages = []
        system_parts = []

        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                if not content:
                    continue

                # Extract system messages for top-level system parameter
                if role == 'system':
                    system_parts.append(content)
                else:
                    api_messages.append({"role": role, "content": content})
            elif isinstance(msg, str):
                api_messages.append({"role": "user", "content": msg})

        system_prompt = "\n\n".join(system_parts) if system_parts else None
        return api_messages, system_prompt

    def inspect_history(self, n: int = 1) -> List[Dict[str, Any]]:
        """DSPy-compatible history inspection."""
        return self.history[-n:] if self.history else []


def is_api_key_available() -> bool:
    """Check if ANTHROPIC_API_KEY is set."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def configure_direct_anthropic(model: str = "haiku", **kwargs) -> DirectAnthropicLM:
    """
    Configure DSPy with DirectAnthropicLM.

    Usage:
        export ANTHROPIC_API_KEY=sk-ant-...

        from Jotty.core.foundation.direct_anthropic_lm import configure_direct_anthropic
        lm = configure_direct_anthropic()
    """
    lm = DirectAnthropicLM(model=model, **kwargs)
    dspy.configure(lm=lm)
    return lm
