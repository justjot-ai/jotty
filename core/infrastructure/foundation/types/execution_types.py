"""
Execution Types - Shared Coordination and Merge Enums
=====================================================

These types are used by both agents/ and swarms/ subsystems.
Extracted to foundation/ to break the agents → swarms circular dependency.

Previously defined in core/swarms/base/agent_team.py.
Both agent_team.py and composite_agent.py now import from here.

Updated: 2026-02-15 - Enhanced with AUTO, CUSTOM, ITERATIVE, DEBATE patterns
"""

from enum import Enum


class CoordinationPattern(Enum):
    """
    Agent coordination patterns - defines HOW agents work together.

    Patterns by category:
    - Adaptive: AUTO, CUSTOM
    - Linear: SEQUENTIAL (PIPELINE for backward compat)
    - Concurrent: PARALLEL
    - Collaborative: CONSENSUS, DEBATE, ITERATIVE
    - Hierarchical: HIERARCHICAL
    - Shared: BLACKBOARD
    """

    # ========== Adaptive ==========
    AUTO = "auto"
    """🧠 Swarm intelligently selects best pattern based on task analysis and learning"""

    CUSTOM = "custom"
    """📝 User-defined multi-stage workflow using STAGES configuration"""

    # ========== Linear ==========
    SEQUENTIAL = "sequential"
    """→ Sequential execution: A → B → C (each step depends on previous)"""

    PIPELINE = "pipeline"
    """DEPRECATED: Use SEQUENTIAL instead. Kept for backward compatibility."""

    NONE = "none"
    """LEGACY: No pattern - swarm handles coordination manually. Use CUSTOM instead."""

    # ========== Concurrent ==========
    PARALLEL = "parallel"
    """⚡ Parallel execution: A | B | C (independent tasks run concurrently)"""

    # ========== Collaborative ==========
    CONSENSUS = "consensus"
    """🤝 All agents vote, best result selected (Byzantine fault tolerant)"""

    DEBATE = "debate"
    """💬 Multi-round deliberation with synthesis:
    Round 1: Propose solutions
    Round 2-N: Critique & refine
    Final: Synthesize best-of-all solution"""

    ITERATIVE = "iterative"
    """🔄 Feedback loop until quality threshold met:
    Generate → Evaluate → Improve → Repeat"""

    # ========== Hierarchical ==========
    HIERARCHICAL = "hierarchical"
    """👔 Manager delegates to workers, aggregates results"""

    # ========== Shared Workspace ==========
    BLACKBOARD = "blackboard"
    """📋 Shared state where agents incrementally contribute"""

    ROUND_ROBIN = "round_robin"
    """🔁 Agents take turns on subtasks"""


class MergeStrategy(Enum):
    """
    Mechanical result merging strategies (no intelligence required).

    For intelligent synthesis, use SynthesisStrategy instead.
    """

    COMBINE = "combine"  # Combine all outputs into list
    FIRST = "first"  # Take first successful result
    BEST = "best"  # Use scoring to pick best
    VOTE = "vote"  # Majority voting
    CONCAT = "concat"  # Concatenate string outputs


class SynthesisStrategy(Enum):
    """
    Intelligent result synthesis strategies (requires LLM/reasoning).

    Unlike MergeStrategy (mechanical), these create new insights.
    """

    SYNTHESIZE = "synthesize"
    """🧠 LLM creates new integrated solution from all inputs"""

    CONSOLIDATE = "consolidate"
    """📊 Merge + deduplicate + organize"""

    REFINE = "refine"
    """✨ Take best result and improve it further"""

    BLEND = "blend"
    """🎨 Weighted combination of multiple results"""
