"""
Composable Prompt System (Cline PromptRegistry pattern, KISS version)
=====================================================================

Builds agent system prompts from reusable sections, with model-family
awareness. Instead of static strings, prompts are composed from ordered
components that adapt to the LLM provider.

Usage:
    composer = PromptComposer(model="claude-sonnet-4-20250514")
    prompt = composer.compose(
        identity="You are a research agent specializing in finance.",
        tools=["web-search", "calculator"],
        learning_context=["Past success: used web-search for stock data"],
        constraints=["Never invent data. Always cite sources."],
        task="Research current S&P 500 trends",
    )

Model-family adaptations:
    - Claude: XML-style structure, tool_use native
    - GPT: Markdown structure, function_calling
    - Groq/fast: Minimal prompt, skip verbose sections
    - Generic: Safe middle ground
"""

from .composer import PromptComposer, ModelFamily
from .rules import load_project_rules, clear_rules_cache

__all__ = ['PromptComposer', 'ModelFamily', 'load_project_rules', 'clear_rules_cache']
