#!/usr/bin/env python3
"""
Context Utilities for Jotty
===========================

Central utilities for providing context to LLMs.
Ensures all LLM calls have access to accurate current date/time.
"""

from datetime import datetime
from typing import Optional


def get_current_datetime_context() -> str:
    """
    Get current date/time context for LLM.
    This ensures the LLM knows the actual current date.

    Returns:
        Formatted context string with current date and time.
    """
    now = datetime.now()
    return (
        f"Current date: {now.strftime('%Y-%m-%d')} "
        f"({now.strftime('%A, %B %d, %Y')}). "
        f"Current time: {now.strftime('%H:%M:%S')}."
    )


def get_system_context(include_time: bool = True) -> str:
    """
    Get full system context for LLM prompts.

    Args:
        include_time: Whether to include current time (default True)

    Returns:
        System context string to prepend to prompts.
    """
    now = datetime.now()

    if include_time:
        return (
            f"[System Context]\n"
            f"Current date: {now.strftime('%Y-%m-%d')} ({now.strftime('%A, %B %d, %Y')})\n"
            f"Current time: {now.strftime('%H:%M:%S')}\n"
            f"Timezone: {now.astimezone().tzname()}"
        )
    else:
        return (
            f"[System Context]\n"
            f"Current date: {now.strftime('%Y-%m-%d')} ({now.strftime('%A, %B %d, %Y')})"
        )


def prepend_context_to_prompt(prompt: str, context: Optional[str] = None) -> str:
    """
    Prepend system context to a prompt.

    Args:
        prompt: Original prompt
        context: Optional custom context (uses default if None)

    Returns:
        Prompt with context prepended.
    """
    if context is None:
        context = get_current_datetime_context()

    return f"[System Context: {context}]\n\n{prompt}"
