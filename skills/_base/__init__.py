"""
Base Skill Package
==================

Unified base classes and utilities for all Jotty skills.

Quick Start:
    from skills._base import BaseLLMSkill

    class MySummarizerSkill(BaseLLMSkill):
        def execute(self, params):
            text = params["text"]
            summary = self.call_lm(f"Summarize this: {text}")
            return self.success(summary=summary)

    summarize = MySummarizerSkill("summarizer")
"""

from .base_skill import BaseLLMSkill, BaseSkill, BaseToolSkill
from .decorators import skill_tool, validate_params
from .helpers import create_llm_skill, create_tool_skill

__all__ = [
    # Base classes
    "BaseSkill",
    "BaseLLMSkill",
    "BaseToolSkill",
    # Decorators
    "skill_tool",
    "validate_params",
    # Helpers
    "create_llm_skill",
    "create_tool_skill",
]
