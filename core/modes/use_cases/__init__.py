"""
Use Case Layer for Jotty

This layer contains business logic specific to different use cases (chat, workflow).
Each use case is self-contained and can be extended independently.
"""

from .base import BaseUseCase, UseCaseConfig, UseCaseResult
from .chat import ChatUseCase
from .workflow import WorkflowUseCase

__all__ = [
    "BaseUseCase",
    "UseCaseResult",
    "UseCaseConfig",
    "ChatUseCase",
    "WorkflowUseCase",
]
