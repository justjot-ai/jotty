"""
Pipeline Mode - Data Flow Through Stages
=========================================

Pattern:
    Data → Stage 1 → Intermediate → Stage 2 → Intermediate → Stage 3 → Output

Each stage:
- Receives specific input type
- Produces specific output type
- Doesn't need to know about other stages

REUSES:
- sequential_delivery_phase (pipeline = sequential with data emphasis)
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


async def run_pipeline_mode(
    goal: str,
    tools: List[Any],
    shared_context,
    scratchpad,
    persistence,
    stages: List[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Pipeline: Data flows through stages sequentially.

    Example stages:
    - ['Load Data', 'Clean Data', 'Analyze', 'Visualize', 'Report']
    - ['Parse Input', 'Validate', 'Process', 'Format Output']

    REUSES: sequential_delivery_phase (pipeline is sequential with data flow)
    """

    session_name = f"pipeline_{Path.cwd().name}"
    session_file = persistence.create_session(session_name)

    # Default stages if not provided
    if stages is None:
        stages = [
            'Load and validate input',
            'Process and transform',
            'Generate output'
        ]

    logger.info(f"📊 PIPELINE MODE: {len(stages)} stages")
    for i, stage in enumerate(stages, 1):
        logger.info(f"   Stage {i}: {stage}")

    # Configure agents for each stage
    agent_configs = [
        {
            'name': f'Stage {i+1}: {stage}',
            'agent': None,
            'expert': None,
            'tools': tools,
            'role': stage
        }
        for i, stage in enumerate(stages)
    ]

    # Run pipeline (sequential delivery with data flow)
    result = await sequential_delivery_phase(
        agents_config=agent_configs,
        discoveries=None,  # Start with no discoveries
        goal=goal,
        shared_context=shared_context,
        scratchpad=scratchpad,
        persistence=persistence,
        session_file=session_file
    )

    return {
        'stages': stages,
        'result': result,
        'session_file': session_file
    }
