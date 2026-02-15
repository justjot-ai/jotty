"""
Base Skill Architecture
========================

Unified base classes for all Jotty skills to eliminate code duplication.

Architecture:
    BaseSkill           - Core skill with status, config, error handling
    ├── BaseToolSkill   - Simple utility skills (no LLM)
    └── BaseLLMSkill    - LLM-powered skills (uses UnifiedLMProvider)

Benefits:
- Eliminates 100+ lines of boilerplate per skill
- Standardizes LLM access across all skills
- Consistent error handling and status reporting
- Easy testing and mocking
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

from Jotty.core.infrastructure.foundation.unified_lm_provider import UnifiedLMProvider
from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_error, tool_response

logger = logging.getLogger(__name__)


class BaseSkill(ABC):
    """
    Base class for all Jotty skills.

    Provides:
    - Status callback handling
    - Configuration management
    - Consistent error handling
    - Tool response formatting

    Subclasses should implement:
    - execute() - main skill logic
    """

    def __init__(self, skill_name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize base skill.

        Args:
            skill_name: Unique name for this skill (for logging/status)
            config: Optional configuration dict
        """
        self.skill_name = skill_name
        self.config = config or {}
        self.status = SkillStatus(skill_name)
        self.logger = logging.getLogger(f"skill.{skill_name}")

    def set_status_callback(self, callback: Optional[Callable] = None) -> None:
        """Set status callback for progress reporting."""
        self.status.set_callback(callback)

    def update_status(self, message: str, **kwargs: Any) -> None:
        """Update status with message."""
        self.status(message, **kwargs)

    @abstractmethod
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the skill logic.

        Args:
            params: Input parameters for the skill

        Returns:
            Dict with success, result, or error
        """
        pass

    def __call__(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute skill with error handling.

        Args:
            params: Input parameters

        Returns:
            Tool response dict with success/error
        """
        # Extract and set status callback if provided
        callback = params.pop("_status_callback", None)
        self.set_status_callback(callback)

        try:
            self.logger.debug(f"Executing {self.skill_name} with params: {list(params.keys())}")
            result = self.execute(params)

            # Ensure result is a proper tool response
            if not isinstance(result, dict):
                result = tool_response(result=result)
            elif "success" not in result:
                result = tool_response(**result)

            self.logger.debug(f"{self.skill_name} completed successfully")
            return result

        except KeyError as e:
            error_msg = f"Missing required parameter: {str(e)}"
            self.logger.error(f"{self.skill_name} error: {error_msg}")
            return tool_error(error_msg)

        except ValueError as e:
            error_msg = f"Invalid parameter value: {str(e)}"
            self.logger.error(f"{self.skill_name} error: {error_msg}")
            return tool_error(error_msg)

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            self.logger.exception(f"{self.skill_name} unexpected error: {error_msg}")
            return tool_error(error_msg)

    def success(self, **kwargs: Any) -> Dict[str, Any]:
        """Create success response."""
        return tool_response(**kwargs)

    def error(self, message: str, **kwargs: Any) -> Dict[str, Any]:
        """Create error response."""
        return tool_error(message, **kwargs)


class BaseToolSkill(BaseSkill):
    """
    Base class for utility skills that don't need LLM.

    Examples: calculator, file operations, web scraping, etc.

    Just implement execute() with your transformation logic.
    """

    pass


class BaseLLMSkill(BaseSkill):
    """
    Base class for skills that need LLM access.

    Provides unified LLM access via UnifiedLMProvider.
    No more creating separate Anthropic/OpenAI clients!

    Features:
    - self.lm - DSPy LM instance (auto-configured)
    - self.call_lm() - Simple LLM call helper
    - self.generate_text() - Text generation
    - self.generate_structured() - Structured JSON output

    Example:
        class MySummarizerSkill(BaseLLMSkill):
            def execute(self, params):
                text = params["text"]
                summary = self.call_lm(f"Summarize: {text}")
                return self.success(summary=summary)
    """

    def __init__(
        self,
        skill_name: str,
        config: Optional[Dict[str, Any]] = None,
        provider: str = "anthropic",
        model: Optional[str] = None,
        **lm_kwargs: Any,
    ):
        """
        Initialize LLM-powered skill.

        Args:
            skill_name: Unique skill name
            config: Optional config dict
            provider: LLM provider (anthropic, openai, google, groq, etc.)
            model: Model name (provider-specific, uses defaults if None)
            **lm_kwargs: Additional LM config (temperature, max_tokens, etc.)
        """
        super().__init__(skill_name, config)

        self.provider = provider
        self.model = model
        self.lm_kwargs = lm_kwargs

        # Lazy-init LM (don't create until first use)
        self._lm = None

    @property
    def lm(self):
        """Get unified LM instance (lazy-initialized)."""
        if self._lm is None:
            self.logger.debug(f"Initializing {self.provider} LM for {self.skill_name}")
            self._lm = UnifiedLMProvider.create_lm(
                provider=self.provider,
                model=self.model,
                inject_context=True,  # Auto-inject current date/time
                **self.lm_kwargs,
            )
        return self._lm

    def call_lm(
        self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7, **kwargs: Any
    ) -> str:
        """
        Simple LLM call helper.

        Args:
            prompt: Input prompt
            max_tokens: Max response tokens
            temperature: Sampling temperature
            **kwargs: Additional LM params

        Returns:
            LLM response text
        """
        try:
            response = self.lm(
                prompt=prompt, max_tokens=max_tokens, temperature=temperature, **kwargs
            )

            # Extract text from DSPy response
            if hasattr(response, "completions"):
                return response.completions[0].text
            elif isinstance(response, list) and len(response) > 0:
                return response[0]
            else:
                return str(response)

        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            raise

    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        """
        Generate text with optional system prompt.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Max response tokens
            temperature: Sampling temperature
            **kwargs: Additional params

        Returns:
            Generated text
        """
        if system_prompt:
            full_prompt = f"{system_prompt}\n\nUser request: {prompt}"
        else:
            full_prompt = prompt

        return self.call_lm(full_prompt, max_tokens=max_tokens, temperature=temperature, **kwargs)

    def generate_structured(
        self, prompt: str, schema: Dict[str, Any], **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Generate structured JSON output matching schema.

        Args:
            prompt: Input prompt
            schema: JSON schema for output
            **kwargs: Additional LM params

        Returns:
            Parsed JSON dict matching schema
        """
        import json

        schema_str = json.dumps(schema, indent=2)
        structured_prompt = f"""{prompt}

Respond with valid JSON matching this schema:
{schema_str}

JSON response:"""

        response = self.call_lm(structured_prompt, **kwargs)

        # Extract JSON from response (handle markdown fences)
        import re

        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response.strip()

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON: {e}")
            raise ValueError(f"LLM returned invalid JSON: {response[:200]}")


__all__ = ["BaseSkill", "BaseToolSkill", "BaseLLMSkill"]
