"""
Skill-Based Agent - DSPy agent that wraps AutoAgent execution

Creates DSPy agents from skills that use AutoAgent for execution.
This bridges Conductor (needs AgentConfig) and AutoAgent (executes skills).
"""

import logging
from typing import Any, Optional

try:
    import dspy

    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class SkillExecutionSignature(dspy.Signature):
    """Signature for skill-based agent execution."""

    task: str = dspy.InputField(desc="Task to execute using this skill")
    result: str = dspy.OutputField(desc="Execution result")


class SkillBasedAgent(dspy.Module):
    """
    DSPy agent that wraps AutoAgent execution for a specific skill.

    This allows Conductor to use skills via AutoAgent execution.
    """

    def __init__(self, skill_name: str, tool_name: Optional[str] = None) -> None:
        """
        Initialize skill-based agent.

        Args:
            skill_name: Name of skill to execute
            tool_name: Optional specific tool name (uses first tool if None)
        """
        super().__init__()
        self.skill_name = skill_name
        self.tool_name = tool_name

        # Create DSPy predictor
        if DSPY_AVAILABLE:
            self.predictor = dspy.ChainOfThought(SkillExecutionSignature)
        else:
            self.predictor = None

        # Lazy load AutoAgent
        self._auto_agent = None

        logger.info(f" SkillBasedAgent created for skill: {skill_name}")

    def forward(self, task: str, **kwargs: Any) -> dspy.Prediction:
        """
        Execute task using AutoAgent.

        Args:
            task: Task description
            **kwargs: Additional parameters

        Returns:
            DSPy Prediction with result
        """
        # Get AutoAgent instance
        if self._auto_agent is None:
            from .auto_agent import AutoAgent

            self._auto_agent = AutoAgent()

        # Execute using AutoAgent
        # AutoAgent will use the skill internally
        try:
            import asyncio

            # Check if we're in async context
            try:
                asyncio.get_running_loop()
                # Already in async context — schedule as task
                result = asyncio.create_task(self._auto_agent.execute(task))
            except RuntimeError:
                # No running loop — safe to use asyncio.run()
                result = asyncio.run(self._auto_agent.execute(task))

            # Extract result
            if hasattr(result, "final_output"):
                result_str = str(result.final_output) or "Task completed"
            else:
                result_str = str(result)

            return dspy.Prediction(result=result_str)

        except Exception as e:
            logger.error(f"Skill execution failed: {e}", exc_info=True)
            return dspy.Prediction(result=f"Error: {str(e)}")


def create_skill_agent(skill_name: str, tool_name: Optional[str] = None) -> SkillBasedAgent:
    """Create a skill-based agent."""
    return SkillBasedAgent(skill_name, tool_name)
