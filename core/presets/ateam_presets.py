"""
A-TEAM PRESETS - Pre-configured Expert Teams
=============================================

Preset configurations for A-Team reviews following the BrainPreset pattern.

Users pick ONE preset name, everything else is auto-configured.

Based on: /var/www/sites/personal/stock_market/SYNAPSE A-TEAM ROSTER.md

Usage:
    from core.presets.ateam_presets import ATeamPreset, get_preset_config

    # Get preset configuration
    config = get_preset_config(ATeamPreset.RL_REVIEW)

    # Or by string name
    config = get_preset_config_by_name("rl_review")
"""

from enum import Enum
from typing import List, Dict, Any, Optional
from .ateam_roster import (
    get_experts_by_names,
    get_all_experts,
    Expert,
    ExpertDomain,
    get_expert_names_by_domain
)


# =============================================================================
# PRESET ENUM (A-Team Approved)
# =============================================================================

class ATeamPreset(Enum):
    """
    Simple presets for A-Team configurations.
    Users pick ONE word, everything else is auto-configured.

    Presets organized by use case:
    - RL_REVIEW: For RL/MARL system design
    - ARCHITECTURE_DESIGN: For software architecture decisions
    - PRODUCT_DESIGN: For naming, UX, and product decisions
    - INFORMATION_FLOW: For memory, RAG, and data pipeline design
    - LOGIC_VERIFICATION: For algorithmic correctness
    - FULL_CONSENSUS: All 22+ experts (comprehensive review)
    - MINIMAL: Fastest (3-5 key experts)
    """
    # Specialized presets
    RL_REVIEW = "rl_review"
    ARCHITECTURE_DESIGN = "architecture_design"
    PRODUCT_DESIGN = "product_design"
    INFORMATION_FLOW = "information_flow"
    LOGIC_VERIFICATION = "logic_verification"

    # Comprehensive presets
    FULL_CONSENSUS = "full_consensus"

    # Minimal preset
    MINIMAL = "minimal"


# =============================================================================
# PRESET CONFIGURATIONS (Internal, users don't see this)
# =============================================================================

PRESET_CONFIGS: Dict[ATeamPreset, Dict[str, Any]] = {
    # RL & MARL System Design
    ATeamPreset.RL_REVIEW: {
        'experts': [
            "Richard Sutton",          # RL Architect
            "David Silver",            # Multi-Agent RL Lead
            "John von Neumann",        # Game Theory Founder
            "John Nash",               # Equilibrium Specialist
            "Jim Simons",              # Quantitative Strategy Lead
            "Alan Turing",             # Chief Logician
            "Omar Khattab",            # DSPy Framework Lead
            "Alex Chen",               # MIT GenZ Tech Lead
        ],
        'max_rounds': 5,
        'consensus_threshold': 1.0,  # 100% consensus required
        'description': 'RL/MARL system design with game-theoretic validation'
    },

    # Software Architecture & Framework Design
    ATeamPreset.ARCHITECTURE_DESIGN: {
        'experts': [
            "Cursor Staff Engineer",              # IDE Integration Lead
            "Pandas Core Contributor",            # Data Structures Lead
            "Apache Foundation Senior Engineer",  # Distributed Systems Lead
            "Anthropic Agent Systems Engineer",   # AI Safety Lead
            "OpenAI GPT Agents Core Team",        # LLM Integration Lead
            "Omar Khattab",                       # DSPy Framework Lead
            "Alan Turing",                        # Chief Logician
            "Kurt Gödel",                         # Formal Systems Architect
            "Claude Shannon",                     # Information Theorist
            "Alex Chen",                          # MIT GenZ Tech Lead
        ],
        'max_rounds': 5,
        'consensus_threshold': 1.0,
        'description': 'Software architecture and framework design decisions'
    },

    # Product, Naming & UX Design
    ATeamPreset.PRODUCT_DESIGN: {
        'experts': [
            "Alex Chen",                        # MIT GenZ Tech Lead
            "Stanford CS/Berkeley MBA Duo",     # Documentation Lead
            "Richard Thaler",                   # Behavioral Economist
            "Cursor Staff Engineer",            # IDE Integration Lead
            "Aristotle",                        # Logic & Rhetoric Master
        ],
        'max_rounds': 3,
        'consensus_threshold': 1.0,
        'description': 'Product design, naming, UX, and documentation'
    },

    # Memory, RAG, and Information Flow
    ATeamPreset.INFORMATION_FLOW: {
        'experts': [
            "Claude Shannon",                      # Information Theorist
            "Vannevar Bush",                       # Systems Architect
            "Sigmund Freud",                       # Cognitive Process Analyst
            "Omar Khattab",                        # DSPy Framework Lead
            "Pandas Core Contributor",             # Data Structures Lead
            "Apache Foundation Senior Engineer",   # Distributed Systems Lead
            "Alan Turing",                         # Chief Logician
        ],
        'max_rounds': 5,
        'consensus_threshold': 1.0,
        'description': 'Memory systems, RAG, and information flow design'
    },

    # Logic & Algorithmic Verification
    ATeamPreset.LOGIC_VERIFICATION: {
        'experts': [
            "Alan Turing",             # Chief Logician
            "Kurt Gödel",              # Formal Systems Architect
            "Aristotle",               # Logic & Rhetoric Master
            "Claude Shannon",          # Information Theorist
            "Richard Sutton",          # RL Architect (for convergence)
        ],
        'max_rounds': 4,
        'consensus_threshold': 1.0,
        'description': 'Algorithmic correctness and logical verification'
    },

    # Full Consensus (All 22+ Experts)
    ATeamPreset.FULL_CONSENSUS: {
        'experts': [expert.name for expert in get_all_experts()],
        'max_rounds': 7,
        'consensus_threshold': 1.0,
        'description': 'Comprehensive review with all 22+ experts'
    },

    # Minimal (Fastest)
    ATeamPreset.MINIMAL: {
        'experts': [
            "Alan Turing",             # Chief Logician
            "Omar Khattab",            # DSPy Framework Lead
            "Alex Chen",               # MIT GenZ Tech Lead
        ],
        'max_rounds': 3,
        'consensus_threshold': 1.0,
        'description': 'Minimal expert set for quick reviews'
    },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_preset_config(preset: ATeamPreset) -> Dict[str, Any]:
    """
    Get configuration for a preset.

    Returns dict with:
        - experts: List[str] - expert names
        - max_rounds: int - maximum debate rounds
        - consensus_threshold: float - required consensus (1.0 = 100%)
        - description: str - preset description
    """
    return PRESET_CONFIGS[preset].copy()


def get_preset_config_by_name(preset_name: str) -> Dict[str, Any]:
    """
    Get configuration by preset name string.

    Example:
        config = get_preset_config_by_name("rl_review")
    """
    try:
        preset = ATeamPreset(preset_name.lower())
        return get_preset_config(preset)
    except ValueError:
        # Default to MINIMAL if unknown
        return get_preset_config(ATeamPreset.MINIMAL)


def get_preset_experts(preset: ATeamPreset) -> List[Expert]:
    """Get Expert objects for a preset."""
    config = get_preset_config(preset)
    expert_names = config['experts']
    return get_experts_by_names(expert_names)


def get_all_preset_names() -> List[str]:
    """Get all available preset names."""
    return [preset.value for preset in ATeamPreset]


def get_preset_description(preset: ATeamPreset) -> str:
    """Get description of a preset."""
    config = get_preset_config(preset)
    return config['description']


# =============================================================================
# CUSTOM TEAM BUILDER
# =============================================================================

def build_custom_team(
    expert_names: List[str],
    max_rounds: int = 5,
    consensus_threshold: float = 1.0,
    description: str = "Custom expert team"
) -> Dict[str, Any]:
    """
    Build a custom expert team configuration.

    Args:
        expert_names: List of expert names
        max_rounds: Maximum debate rounds (default: 5)
        consensus_threshold: Required consensus (default: 1.0 = 100%)
        description: Team description

    Returns:
        Configuration dict compatible with preset configs

    Example:
        team = build_custom_team(
            expert_names=["Alan Turing", "Omar Khattab", "Alex Chen"],
            max_rounds=3,
            description="Quick architecture review"
        )
    """
    # Validate expert names
    valid_experts = [name for name in expert_names if get_experts_by_names([name])]

    if not valid_experts:
        raise ValueError(f"No valid experts in list: {expert_names}")

    return {
        'experts': valid_experts,
        'max_rounds': max_rounds,
        'consensus_threshold': consensus_threshold,
        'description': description
    }


def build_team_by_domains(
    domains: List[ExpertDomain],
    max_rounds: int = 5,
    consensus_threshold: float = 1.0
) -> Dict[str, Any]:
    """
    Build a team from specific domains.

    Args:
        domains: List of ExpertDomain enums
        max_rounds: Maximum debate rounds
        consensus_threshold: Required consensus

    Returns:
        Configuration dict

    Example:
        team = build_team_by_domains(
            domains=[ExpertDomain.RL_MARL, ExpertDomain.GAME_THEORY],
            max_rounds=4
        )
    """
    expert_names = []
    for domain in domains:
        expert_names.extend(get_expert_names_by_domain(domain))

    domain_str = ", ".join([d.value for d in domains])
    description = f"Team from domains: {domain_str}"

    return build_custom_team(
        expert_names=expert_names,
        max_rounds=max_rounds,
        consensus_threshold=consensus_threshold,
        description=description
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'ATeamPreset',
    'get_preset_config',
    'get_preset_config_by_name',
    'get_preset_experts',
    'get_all_preset_names',
    'get_preset_description',
    'build_custom_team',
    'build_team_by_domains',
    'PRESET_CONFIGS',
]
