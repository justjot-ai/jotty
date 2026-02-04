"""
Debate Mode - Competing Solutions → Critique → Vote
===================================================

Pattern:
    Phase 1: Propose (agents propose different solutions)
    Phase 2: Critique (agents review each other's proposals)
    Phase 3: Vote/Combine (select best or synthesize)

REUSES:
- p2p_discovery_phase (for proposals and critiques)
- sequential_delivery_phase (for final synthesis)
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


async def run_debate_mode(
    goal: str,
    tools: List[Any],
    shared_context,
    scratchpad,
    persistence,
    num_debaters: int = 3,
    **kwargs
) -> Dict[str, Any]:
    """
    Debate: Multiple solutions → critique → select best.

    Steps:
    1. Debaters propose different solutions in parallel
    2. Debaters critique each other's proposals in parallel
    3. Judge selects best solution or synthesizes combined solution

    REUSES: p2p_discovery_phase (proposals and critiques)
    """

    session_name = f"debate_{Path.cwd().name}"
    session_file = persistence.create_session(session_name)

    logger.info("💬 DEBATE MODE: Propose → Critique → Vote")

    # Phase 1: Proposals (P2P - each agent proposes different approach)
    proposal_configs = [
        {
            'name': f'Debater {i+1}',
            'agent': None,
            'expert': None,
            'tools': tools,
            'role': f'Propose solution {i+1} for: {goal}'
        }
        for i in range(num_debaters)
    ]

    proposals = await p2p_discovery_phase(
        agents_config=proposal_configs,
        task=f"Propose diverse solutions for: {goal}",
        shared_context=shared_context,
        scratchpad=scratchpad,
        persistence=persistence,
        session_file=session_file
    )

    # Phase 2: Critiques (P2P - agents review each other's proposals)
    critique_configs = [
        {
            'name': f'Critic {i+1}',
            'agent': None,
            'expert': None,
            'tools': tools,
            'role': f'Critique all proposals from perspective {i+1}'
        }
        for i in range(num_debaters)
    ]

    critiques = await p2p_discovery_phase(
        agents_config=critique_configs,
        task=f"Critique all proposals and score each",
        shared_context=shared_context,
        scratchpad=scratchpad,
        persistence=persistence,
        session_file=session_file
    )

    # Phase 3: Synthesis (Sequential - judge selects or combines best)
    judge_config = [{
        'name': 'Judge Agent',
        'agent': None,
        'expert': None,
        'tools': tools,
        'role': 'Select best proposal or synthesize combined solution'
    }]

    final_decision = await sequential_delivery_phase(
        agents_config=judge_config,
        discoveries={'proposals': proposals, 'critiques': critiques},
        goal=goal,
        shared_context=shared_context,
        scratchpad=scratchpad,
        persistence=persistence,
        session_file=session_file
    )

    return {
        'proposals': proposals,
        'critiques': critiques,
        'final_decision': final_decision,
        'session_file': session_file
    }
