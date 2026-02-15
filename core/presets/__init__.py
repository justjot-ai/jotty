"""
A-TEAM ROSTER Integration for Jotty
====================================

Expert-driven consensus task generation system.

Based on: /var/www/sites/personal/stock_market/SYNAPSE A-TEAM ROSTER.md

Three usage modes:
1. Template mode: Use preset teams (rl_review, architecture_design, etc.)
2. Custom mode: User selects specific experts
3. Automatic mode (default): Jotty analyzes task and selects optimal team

Quick Start:
    # Template mode (preset)
    from core.presets import ATeamConductor
    conductor = ATeamConductor.from_preset("rl_review")
    result = await conductor.generate_task(description, goal)

    # Automatic mode (default)
    conductor = ATeamConductor.from_auto_selection()
    result = await conductor.generate_task(description, goal)
"""

from .ateam_conductor import (
    ATeamConductor,
    TaskGenerationResult,
    generate_task_auto,
    generate_task_with_preset,
)
from .ateam_presets import (
    ATeamPreset,
    build_custom_team,
    build_team_by_domains,
    get_all_preset_names,
    get_preset_config,
    get_preset_config_by_name,
    get_preset_experts,
)
from .ateam_roster import (
    ALL_EXPERTS,
    Expert,
    ExpertDomain,
    get_all_experts,
    get_expert,
    get_experts_by_domain,
    get_experts_by_names,
)
from .debate_manager import (
    ConsensusResult,
    DebateManager,
    DebateRound,
    DebateState,
    ExpertResponse,
    ExpertVote,
)

__all__ = [
    # Roster
    "Expert",
    "ExpertDomain",
    "get_expert",
    "get_all_experts",
    "get_experts_by_domain",
    "get_experts_by_names",
    "ALL_EXPERTS",
    # Presets
    "ATeamPreset",
    "get_preset_config",
    "get_preset_config_by_name",
    "get_preset_experts",
    "get_all_preset_names",
    "build_custom_team",
    "build_team_by_domains",
    # Debate Manager
    "DebateManager",
    "DebateRound",
    "ExpertResponse",
    "ConsensusResult",
    "DebateState",
    "ExpertVote",
    # Conductor (Main API)
    "ATeamConductor",
    "TaskGenerationResult",
    "generate_task_with_preset",
    "generate_task_auto",
]
