"""
Round-Robin Mode - Iterative Refinement
========================================

Pattern:
    Round 1: A adds → B refines → C extends
    Round 2: A reviews → B improves → C polishes
    Round 3: Final pass

REUSES:
- sequential_delivery_phase (each round is sequential)
"""

import asyncio
import logging
from typing import Dict, Any, List
from pathlib import Path

# REUSE existing functions (NO DUPLICATION!)
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from templates.hybrid_team_template import sequential_delivery_phase

logger = logging.getLogger(__name__)


async def run_round_robin_mode(
    goal: str,
    tools: List[Any],
    shared_context,
    scratchpad,
    persistence,
    num_agents: int = 3,
    num_rounds: int = 3,
    **kwargs
) -> Dict[str, Any]:
    """
    Round-Robin: Iterative refinement over multiple rounds.

    Each round, agents sequentially refine the work.

    REUSES: sequential_delivery_phase (each round)
    """

    session_name = f"round_robin_{Path.cwd().name}"
    session_file = persistence.create_session(session_name)

    logger.info(f"🔄 ROUND-ROBIN MODE: {num_rounds} rounds x {num_agents} agents")

    all_rounds = []

    for round_num in range(1, num_rounds + 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"ROUND {round_num}/{num_rounds}")
        logger.info(f"{'='*80}\n")

        # Sequential delivery (each agent refines previous work)
        agent_configs = [
            {
                'name': f'Agent {i+1} (Round {round_num})',
                'agent': None,
                'expert': None,
                'tools': tools,
                'role': f'Round {round_num} refinement by agent {i+1}'
            }
            for i in range(num_agents)
        ]

        # Previous round's output becomes input for this round
        discoveries = all_rounds[-1] if all_rounds else None

        round_result = await sequential_delivery_phase(
            agents_config=agent_configs,
            discoveries=discoveries,
            goal=goal,
            shared_context=shared_context,
            scratchpad=scratchpad,
            persistence=persistence,
            session_file=session_file
        )

        all_rounds.append(round_result)

    return {
        'rounds': all_rounds,
        'final_result': all_rounds[-1],
        'num_rounds': num_rounds,
        'session_file': session_file
    }
