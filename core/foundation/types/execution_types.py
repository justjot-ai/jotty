"""
Execution Types - Shared Coordination and Merge Enums
=====================================================

These types are used by both agents/ and swarms/ subsystems.
Extracted to foundation/ to break the agents → swarms circular dependency.

Previously defined in core/swarms/base/agent_team.py.
Both agent_team.py and composite_agent.py now import from here.
"""

from enum import Enum


class CoordinationPattern(Enum):
    """
    Patterns for how agents coordinate within a team.

    NONE: No pattern - swarm handles coordination manually (backward compatible)
    PIPELINE: Sequential A -> B -> C, each passes output to next
    PARALLEL: Concurrent A | B | C, results merged
    CONSENSUS: All vote, majority wins (Byzantine fault tolerant)
    HIERARCHICAL: Manager delegates to workers, aggregates results
    BLACKBOARD: Shared workspace, agents contribute incrementally
    ROUND_ROBIN: Agents take turns on subtasks
    """
    NONE = "none"
    PIPELINE = "pipeline"
    PARALLEL = "parallel"
    CONSENSUS = "consensus"
    HIERARCHICAL = "hierarchical"
    BLACKBOARD = "blackboard"
    ROUND_ROBIN = "round_robin"


class MergeStrategy(Enum):
    """How to merge results from parallel execution."""
    COMBINE = "combine"      # Combine all outputs into list
    FIRST = "first"          # Take first successful result
    BEST = "best"            # Use scoring to pick best
    VOTE = "vote"            # Majority voting
    CONCAT = "concat"        # Concatenate string outputs
