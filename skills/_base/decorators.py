"""
Skill Decorators
================

Decorators for creating skills with minimal boilerplate.

Example:
    @skill_tool(required_params=["text", "language"])
    def translate_text(skill, params):
        text = params["text"]
        lang = params["language"]
        translation = skill.call_lm(f"Translate to {lang}: {text}")
        return {"translation": translation}
"""

import functools
from typing import Any, Callable, Dict, List, Optional

from .base_skill import BaseLLMSkill, BaseToolSkill


def validate_params(required: Optional[List[str]] = None, optional: Optional[List[str]] = None):
    """
    Decorator to validate skill parameters.

    Args:
        required: List of required parameter names
        optional: List of optional parameter names (with defaults)

    Example:
        @validate_params(required=["text"], optional=["max_length"])
        def my_skill_execute(self, params):
            text = params["text"]
            max_length = params.get("max_length", 1000)
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, params: Dict[str, Any]) -> Dict[str, Any]:
            # Validate required params
            if required:
                missing = [p for p in required if p not in params]
                if missing:
                    return self.error(
                        f"Missing required parameters: {', '.join(missing)}. "
                        f"Required: {required}, Optional: {optional or []}"
                    )

            # Validate no extra params (unless explicitly allowed)
            if optional is not None:  # Only check if optional is specified
                allowed = set(required or []) | set(optional or [])
                extra = set(params.keys()) - allowed - {"_status_callback"}
                if extra:
                    return self.error(
                        f"Unknown parameters: {', '.join(extra)}. " f"Allowed: {sorted(allowed)}"
                    )

            return func(self, params)

        return wrapper

    return decorator


def skill_tool(
    name: str,
    required_params: Optional[List[str]] = None,
    optional_params: Optional[List[str]] = None,
    use_llm: bool = False,
    provider: str = "anthropic",
    **lm_kwargs: Any,
):
    """
    Decorator to turn a function into a skill tool.

    Automatically creates BaseToolSkill or BaseLLMSkill wrapper.

    Args:
        name: Skill name
        required_params: Required parameter names
        optional_params: Optional parameter names
        use_llm: Whether this skill needs LLM access
        provider: LLM provider (if use_llm=True)
        **lm_kwargs: Additional LM config

    Example:
        @skill_tool("summarizer", required_params=["text"], use_llm=True)
        def summarize(skill, params):
            text = params["text"]
            return {"summary": skill.call_lm(f"Summarize: {text}")}

        # Creates a callable skill:
        result = summarize({"text": "Long text here..."})
    """

    def decorator(func: Callable) -> Callable:
        # Create skill class dynamically
        if use_llm:

            class DynamicLLMSkill(BaseLLMSkill):
                def __init__(self):
                    super().__init__(name, provider=provider, **lm_kwargs)

                @validate_params(required=required_params, optional=optional_params)
                def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
                    result = func(self, params)
                    if isinstance(result, dict) and "success" not in result:
                        return self.success(**result)
                    return result

            skill_instance = DynamicLLMSkill()
        else:

            class DynamicToolSkill(BaseToolSkill):
                def __init__(self):
                    super().__init__(name)

                @validate_params(required=required_params, optional=optional_params)
                def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
                    result = func(self, params)
                    if isinstance(result, dict) and "success" not in result:
                        return self.success(**result)
                    return result

            skill_instance = DynamicToolSkill()

        # Return callable skill instance
        @functools.wraps(func)
        def wrapper(params: Dict[str, Any]) -> Dict[str, Any]:
            return skill_instance(params)

        # Attach skill instance for direct access if needed
        wrapper.skill = skill_instance
        wrapper.__doc__ = func.__doc__ or f"{name} skill"

        return wrapper

    return decorator


__all__ = ["validate_params", "skill_tool"]
