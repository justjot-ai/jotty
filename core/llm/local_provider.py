"""
Local LLM Provider
==================

Routes to Ollama/llama.cpp when local_mode=True.
Provides fully local inference with no external API calls.
"""

import os
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

from Jotty.core.foundation.config_defaults import LLM_TEMPERATURE, LLM_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)


@dataclass
class LocalLLMResponse:
    """Response from local LLM."""
    success: bool
    text: str = ""
    error: str = ""
    model: str = ""
    provider: str = "local"


class LocalLLMProvider:
    """
    Local LLM provider using Ollama or llama.cpp.

    Supports:
    - Ollama API (default, easiest setup)
    - llama.cpp server (alternative)

    Usage:
        provider = LocalLLMProvider("ollama/llama3")
        response = await provider.generate("What is 2+2?")
    """

    def __init__(self, model: str = 'ollama/llama3') -> None:
        """
        Initialize local LLM provider.

        Args:
            model: Model specification in format "provider/model"
                   e.g., "ollama/llama3", "ollama/mistral", "llamacpp/model.gguf"
        """
        self.model_spec = model
        self._parse_model_spec()

        # Ollama settings (centralized defaults)
        try:
            from ..foundation.config_defaults import DEFAULTS as _DEFAULTS
            _ollama_default = _DEFAULTS.OLLAMA_URL
        except ImportError:
            _ollama_default = "http://localhost:11434"
        self.ollama_host = os.getenv("OLLAMA_HOST", _ollama_default)

        # llama.cpp settings
        self.llamacpp_host = os.getenv("LLAMACPP_HOST", "http://localhost:8080")

    def _parse_model_spec(self) -> Any:
        """Parse model specification into provider and model name."""
        if "/" in self.model_spec:
            parts = self.model_spec.split("/", 1)
            self.provider_type = parts[0].lower()
            self.model_name = parts[1]
        else:
            # Default to Ollama
            self.provider_type = "ollama"
            self.model_name = self.model_spec

    async def generate(self, prompt: str, max_tokens: int = 2048, temperature: float = LLM_TEMPERATURE, **kwargs: Any) -> LocalLLMResponse:
        """
        Generate text using local LLM.

        Args:
            prompt: The prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            LocalLLMResponse with result or error
        """
        if self.provider_type == "ollama":
            return await self._generate_ollama(prompt, max_tokens, temperature)
        elif self.provider_type == "llamacpp":
            return await self._generate_llamacpp(prompt, max_tokens, temperature)
        else:
            return LocalLLMResponse(
                success=False,
                error=f"Unknown local provider: {self.provider_type}",
                model=self.model_spec
            )

    async def _generate_ollama(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> LocalLLMResponse:
        """Generate using Ollama API."""
        try:
            import httpx
        except ImportError:
            return LocalLLMResponse(
                success=False,
                error="httpx not installed. Run: pip install httpx",
                model=self.model_spec
            )

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.ollama_host}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "num_predict": max_tokens,
                            "temperature": temperature
                        }
                    },
                    timeout=float(LLM_TIMEOUT_SECONDS)
                )

                if response.status_code == 200:
                    data = response.json()
                    return LocalLLMResponse(
                        success=True,
                        text=data.get("response", ""),
                        model=self.model_spec,
                        provider="ollama"
                    )
                else:
                    return LocalLLMResponse(
                        success=False,
                        error=f"Ollama error: {response.status_code} - {response.text}",
                        model=self.model_spec
                    )

        except httpx.ConnectError:
            return LocalLLMResponse(
                success=False,
                error=f"Cannot connect to Ollama at {self.ollama_host}. Is Ollama running?",
                model=self.model_spec
            )
        except Exception as e:
            logger.error(f"Ollama error: {e}", exc_info=True)
            return LocalLLMResponse(
                success=False,
                error=str(e),
                model=self.model_spec
            )

    async def _generate_llamacpp(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> LocalLLMResponse:
        """Generate using llama.cpp server API."""
        try:
            import httpx
        except ImportError:
            return LocalLLMResponse(
                success=False,
                error="httpx not installed",
                model=self.model_spec
            )

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.llamacpp_host}/completion",
                    json={
                        "prompt": prompt,
                        "n_predict": max_tokens,
                        "temperature": temperature,
                        "stream": False
                    },
                    timeout=float(LLM_TIMEOUT_SECONDS)
                )

                if response.status_code == 200:
                    data = response.json()
                    return LocalLLMResponse(
                        success=True,
                        text=data.get("content", ""),
                        model=self.model_spec,
                        provider="llamacpp"
                    )
                else:
                    return LocalLLMResponse(
                        success=False,
                        error=f"llama.cpp error: {response.status_code}",
                        model=self.model_spec
                    )

        except httpx.ConnectError:
            return LocalLLMResponse(
                success=False,
                error=f"Cannot connect to llama.cpp at {self.llamacpp_host}",
                model=self.model_spec
            )
        except Exception as e:
            logger.error(f"llama.cpp error: {e}", exc_info=True)
            return LocalLLMResponse(
                success=False,
                error=str(e),
                model=self.model_spec
            )

    def generate_sync(self, prompt: str, max_tokens: int = 2048, temperature: float = LLM_TEMPERATURE, **kwargs: Any) -> LocalLLMResponse:
        """Synchronous wrapper for generate()."""
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.generate(prompt, max_tokens, temperature, **kwargs)
            )
        finally:
            loop.close()

    @staticmethod
    def is_available() -> Dict[str, bool]:
        """Check which local providers are available."""
        import httpx

        result = {"ollama": False, "llamacpp": False}

        try:
            from ..foundation.config_defaults import DEFAULTS as _DEFAULTS
            _ollama_default = _DEFAULTS.OLLAMA_URL
        except ImportError:
            _ollama_default = "http://localhost:11434"
        ollama_host = os.getenv("OLLAMA_HOST", _ollama_default)
        llamacpp_host = os.getenv("LLAMACPP_HOST", "http://localhost:8080")

        try:
            with httpx.Client(timeout=2.0) as client:
                try:
                    r = client.get(f"{ollama_host}/api/tags")
                    result["ollama"] = r.status_code == 200
                except Exception:
                    pass

                try:
                    r = client.get(f"{llamacpp_host}/health")
                    result["llamacpp"] = r.status_code == 200
                except Exception:
                    pass
        except Exception:
            pass

        return result
