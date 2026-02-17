"""
Capability Mixins - Reusable capabilities for any execution pattern

These mixins can be added to agents, swarms, or workflows to enhance them:
- LearningCapability: Gold standard learning with optimization
- ValidationCapability: Domain-specific validation
- MemoryCapability: Enhanced memory integration

Usage:
    from Jotty.core.intelligence.reasoning.base import BaseAgent
    from Jotty.core.intelligence.reasoning.capabilities import (
        LearningCapability,
        ValidationCapability,
        MemoryCapability
    )

    class MermaidAgent(BaseAgent, LearningCapability, ValidationCapability):
        def __init__(self, config=None, enable_learning=True):
            BaseAgent.__init__(self, config)

            if enable_learning:
                LearningCapability.__init__(self, domain="mermaid")

            ValidationCapability.__init__(self, domain="mermaid")

        async def _execute_impl(self, task: str, **kwargs):
            result = await self._generate(task)
            result = await self.validate(result)
            if hasattr(self, 'learn_from_gold_standards'):
                result = await self.learn_from_gold_standards(task, result)
            return result
"""

from .learning_capability import LearningCapability
from .memory_capability import MemoryCapability
from .validation_capability import SyntaxValidator, ValidationCapability

__all__ = [
    "LearningCapability",
    "ValidationCapability",
    "SyntaxValidator",
    "MemoryCapability",
]
