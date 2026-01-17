"""
Hierarchical Mode - Lead Agent + Sub-Agents
============================================

Pattern:
    Lead Agent (coordinates)
    ├─ Sub-Agent 1 → reports results
    ├─ Sub-Agent 2 → reports results
    └─ Sub-Agent 3 → reports results

REUSES:
- p2p_discovery_phase (for task decomposition by lead)
- sequential_delivery_phase (for sub-agent execution)
"""

import asyncio
import logging
from typing import Dict, Any, List
from pathlib import Path

# REUSE existing functions (NO DUPLICATION!)
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from templates.hybrid_team_template import p2p_discovery_phase, sequential_delivery_phase

logger = logging.getLogger(__name__)


async def run_hierarchical_mode(
    goal: str,
    tools: List[Any],
    shared_context,
    scratchpad,
    persistence,
    num_sub_agents: int = 3,
    **kwargs
) -> Dict[str, Any]:
    """
    Hierarchical: Lead decomposes task, sub-agents execute.

    Steps:
    1. Lead agent analyzes goal and decomposes into subtasks
    2. Sub-agents execute subtasks in parallel
    3. Lead agent aggregates results and makes final decision

    REUSES: p2p_discovery_phase (subtask decomposition)
    """

    session_name = f"hierarchical_{Path.cwd().name}"
    session_file = persistence.create_session(session_name)

    logger.info("🏗️ HIERARCHICAL MODE: Lead + Sub-Agents")

    # Step 1: Lead agent decomposes task (using P2P with 1 agent = lead)
    lead_config = [{
        'name': 'Lead Agent',
        'agent': None,  # Will be created by p2p_discovery_phase
        'expert': None,
        'tools': tools,
        'role': f'Analyze "{goal}" and decompose into {num_sub_agents} subtasks'
    }]

    decomposition = await p2p_discovery_phase(
        agents_config=lead_config,
        task=f"Decompose '{goal}' into {num_sub_agents} independent subtasks",
        shared_context=shared_context,
        scratchpad=scratchpad,
        persistence=persistence,
        session_file=session_file
    )

    # Step 2: Sub-agents execute subtasks in parallel (using P2P)
    sub_agent_configs = [
        {
            'name': f'Sub-Agent {i+1}',
            'agent': None,
            'expert': None,
            'tools': tools,
            'role': f'Execute subtask {i+1}'
        }
        for i in range(num_sub_agents)
    ]

    sub_results = await p2p_discovery_phase(
        agents_config=sub_agent_configs,
        task=f"Execute subtasks for: {goal}",
        shared_context=shared_context,
        scratchpad=scratchpad,
        persistence=persistence,
        session_file=session_file
    )

    # Step 3: Lead aggregates (using sequential with 1 agent)
    aggregation_config = [{
        'name': 'Lead Agent (Aggregation)',
        'agent': None,
        'expert': None,
        'tools': tools,
        'role': 'Aggregate sub-agent results and make final decision'
    }]

    final_result = await sequential_delivery_phase(
        agents_config=aggregation_config,
        discoveries=sub_results,
        goal=goal,
        shared_context=shared_context,
        scratchpad=scratchpad,
        persistence=persistence,
        session_file=session_file
    )

    return {
        'decomposition': decomposition,
        'sub_results': sub_results,
        'final_result': final_result,
        'session_file': session_file
    }
