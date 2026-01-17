"""
Swarm Mode - Self-Organizing Agents
====================================

Pattern:
    - Task announced
    - Agents self-select based on capabilities
    - Agents claim subtasks dynamically
    - Agents help each other when blocked

REUSES:
- p2p_discovery_phase (for task decomposition and execution)
- Agent self-selection based on tools/capabilities
"""

import asyncio
import logging
from typing import Dict, Any, List
from pathlib import Path

# REUSE existing functions (NO DUPLICATION!)
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from templates.hybrid_team_template import p2p_discovery_phase

logger = logging.getLogger(__name__)


async def run_swarm_mode(
    goal: str,
    tools: List[Any],
    shared_context,
    scratchpad,
    persistence,
    num_agents: int = 5,
    **kwargs
) -> Dict[str, Any]:
    """
    Swarm: Agents self-organize and claim tasks dynamically.

    Steps:
    1. Announce task (all agents see it)
    2. Agents self-select based on capabilities
    3. Agents work in parallel on claimed subtasks
    4. Agents help each other when blocked

    REUSES: p2p_discovery_phase (parallel self-organizing work)
    """

    session_name = f"swarm_{Path.cwd().name}"
    session_file = persistence.create_session(session_name)

    logger.info(f"🐝 SWARM MODE: {num_agents} self-organizing agents")

    # All agents work in parallel (P2P) and self-select subtasks
    agent_configs = [
        {
            'name': f'Swarm Agent {i+1}',
            'agent': None,
            'expert': None,
            'tools': tools,
            'role': f'Self-select and execute subtask based on capabilities'
        }
        for i in range(num_agents)
    ]

    # Run swarm (P2P discovery where agents self-organize)
    result = await p2p_discovery_phase(
        agents_config=agent_configs,
        task=f"SWARM TASK: {goal}\n\nSelf-select subtask based on your capabilities. Coordinate via scratchpad.",
        shared_context=shared_context,
        scratchpad=scratchpad,
        persistence=persistence,
        session_file=session_file
    )

    return {
        'swarm_results': result,
        'num_agents': num_agents,
        'session_file': session_file
    }
