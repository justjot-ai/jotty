"""
Workflow Modes - NEW Multi-Agent Patterns
==========================================

Specialized workflow modes for UniversalWorkflow.

All modes REUSE existing infrastructure:
- p2p_discovery_phase (from hybrid_team_template)
- sequential_delivery_phase (from hybrid_team_template)
- Conductor._execute_actor
- Conductor tools, learning, validation

NO DUPLICATION!

Available Modes:
----------------
- hierarchical: Lead agent + sub-agents
- debate: Competing solutions → critique → vote
- round_robin: Iterative refinement over multiple rounds
- pipeline: Data flow through stages
- swarm: Self-organizing agents claim tasks dynamically
"""

from .hierarchical import run_hierarchical_mode
from .debate import run_debate_mode
from .round_robin import run_round_robin_mode
from .pipeline import run_pipeline_mode
from .swarm import run_swarm_mode

__all__ = [
    'run_hierarchical_mode',
    'run_debate_mode',
    'run_round_robin_mode',
    'run_pipeline_mode',
    'run_swarm_mode',
]
