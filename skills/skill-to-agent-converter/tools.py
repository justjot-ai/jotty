"""
Skill to Agent Converter - Convert skills into autonomous agents.
"""

import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def convert_skill_to_agent(
    skill_name: str,
    agent_name: Optional[str] = None,
    model: str = "sonnet",
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert a skill into an autonomous agent."""
    from .skill_to_agent_converter import SkillToAgentConverter

    try:
        converter = SkillToAgentConverter()

        # Use skill name as agent name if not provided
        if not agent_name:
            agent_name = f"{skill_name}-agent"

        # Convert the skill
        agent_config = converter.convert(
            skill_name=skill_name,
            agent_name=agent_name,
            model=model,
            system_prompt=system_prompt,
        )

        return {
            "success": True,
            "agent_name": agent_name,
            "agent_config": agent_config,
        }
    except Exception as e:
        logger.error(f"Failed to convert skill {skill_name} to agent: {e}")
        return {
            "success": False,
            "agent_name": "",
            "agent_config": {},
            "error": str(e),
        }


def test_skill_agent(agent_name: str, test_task: str) -> Dict[str, Any]:
    """Test a converted skill agent with a sample task."""
    try:
        from Jotty.core.intelligence.reasoning.agent.agents.skill_based_agent import SkillBasedAgent

        # Load the agent
        agent = SkillBasedAgent(name=agent_name)

        # Run the test task
        start_time = time.time()
        result = agent.run(test_task)
        execution_time = time.time() - start_time

        return {
            "success": True,
            "result": str(result),
            "execution_time": execution_time,
        }
    except Exception as e:
        logger.error(f"Failed to test agent {agent_name}: {e}")
        return {
            "success": False,
            "result": "",
            "execution_time": 0.0,
            "error": str(e),
        }
