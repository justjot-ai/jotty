"""
Skill Helper Functions
=======================

Helper functions to create skills programmatically without subclassing.

Example:
    # Create a simple tool skill
    calculator = create_tool_skill(
        name="calculator",
        execute_fn=lambda params: {"result": eval(params["expression"])}
    )

    # Create an LLM skill
    summarizer = create_llm_skill(
        name="summarizer",
        execute_fn=lambda skill, params: {
            "summary": skill.call_lm(f"Summarize: {params['text']}")
        }
    )
"""

from typing import Any, Callable, Dict, Optional

from .base_skill import BaseLLMSkill, BaseToolSkill


def create_tool_skill(
    name: str,
    execute_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None,
) -> BaseToolSkill:
    """
    Create a tool skill from a function.

    Args:
        name: Skill name
        execute_fn: Function(params) -> result_dict
        config: Optional configuration

    Returns:
        BaseToolSkill instance

    Example:
        calc = create_tool_skill(
            "calc",
            lambda params: {"result": eval(params["expr"])}
        )
        result = calc({"expr": "2 + 2"})
    """

    class DynamicToolSkill(BaseToolSkill):
        def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
            result = execute_fn(params)
            if isinstance(result, dict) and "success" not in result:
                return self.success(**result)
            return result

    return DynamicToolSkill(name, config)


def create_llm_skill(
    name: str,
    execute_fn: Callable[[BaseLLMSkill, Dict[str, Any]], Dict[str, Any]],
    provider: str = "anthropic",
    model: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    **lm_kwargs: Any,
) -> BaseLLMSkill:
    """
    Create an LLM skill from a function.

    Args:
        name: Skill name
        execute_fn: Function(skill, params) -> result_dict
        provider: LLM provider
        model: Model name
        config: Optional configuration
        **lm_kwargs: Additional LM config

    Returns:
        BaseLLMSkill instance

    Example:
        summarizer = create_llm_skill(
            "summarizer",
            lambda skill, params: {
                "summary": skill.call_lm(f"Summarize: {params['text']}")
            }
        )
        result = summarizer({"text": "Long text..."})
    """

    class DynamicLLMSkill(BaseLLMSkill):
        def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
            result = execute_fn(self, params)
            if isinstance(result, dict) and "success" not in result:
                return self.success(**result)
            return result

    return DynamicLLMSkill(name, config, provider, model, **lm_kwargs)


__all__ = ["create_tool_skill", "create_llm_skill"]
