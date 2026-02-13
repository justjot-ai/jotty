"""
A-TEAM Conductor - Main Orchestration Layer
============================================

Orchestrates A-TEAM expert debates following the BrainPreset pattern.

Factory method pattern:
    SimpleBrain.from_preset('thorough') → ATeamConductor.from_preset('rl_review')

Three usage modes:
1. Template mode: Use preset teams (rl_review, architecture_design, etc.)
2. Custom mode: User selects specific experts
3. Automatic mode (default): Jotty analyzes task and selects optimal team

Based on: /var/www/sites/personal/stock_market/SYNAPSE A-TEAM ROSTER.md

Usage:
    # Template mode (preset)
    conductor = ATeamConductor.from_preset("rl_review")
    result = await conductor.generate_task(description, goal)

    # Custom mode (user selects experts)
    conductor = ATeamConductor.from_custom_team(
        expert_names=["Alan Turing", "Omar Khattab", "Alex Chen"]
    )
    result = await conductor.generate_task(description, goal)

    # Automatic mode (Jotty selects)
    conductor = ATeamConductor.from_auto_selection()
    result = await conductor.generate_task(description, goal)
"""

import logging
import dspy
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .ateam_roster import Expert, get_experts_by_names, get_all_experts, ExpertDomain
from .ateam_presets import (
    ATeamPreset,
    get_preset_config,
    get_preset_config_by_name,
    get_preset_experts,
    build_custom_team
)
from .debate_manager import DebateManager, ConsensusResult

logger = logging.getLogger(__name__)


# =============================================================================
# AUTO SELECTION (Jotty analyzes task and selects experts)
# =============================================================================

class AutoTeamSelectionSignature(dspy.Signature):
    """Automatically select optimal expert team for a task."""
    task_description: str = dspy.InputField(desc="Task/problem description")
    goal_context: str = dspy.InputField(desc="Overall goal and context")
    available_experts: str = dspy.InputField(desc="All available experts (JSON)")

    selected_experts: str = dspy.OutputField(desc="Selected expert names (comma-separated)")
    reasoning: str = dspy.OutputField(desc="Why these experts were chosen")
    estimated_complexity: str = dspy.OutputField(desc="Task complexity: simple, moderate, complex")


# =============================================================================
# A-TEAM CONDUCTOR
# =============================================================================

@dataclass
class TaskGenerationResult:
    """Result from A-TEAM task generation."""
    consensus_decision: str
    comprehensive_task: str
    reasoning: str
    expert_votes: Dict[str, str]
    total_rounds: int
    consensus_reached: bool
    mode: str  # "template", "custom", or "automatic"


class ATeamConductor:
    """
    A-TEAM Conductor - Orchestrates expert debates.

    Following the Jotty pattern:
        SimpleBrain.from_preset('thorough')
        ATeamConductor.from_preset('rl_review')

    Design Principles:
    1. FACTORY METHODS for easy instantiation
    2. THREE MODES: template, custom, automatic
    3. PRESET PATTERN like BrainPreset
    4. 100% CONSENSUS required
    """

    def __init__(
        self,
        experts: List[Expert],
        max_rounds: int = 5,
        consensus_threshold: float = 1.0,
        mode: str = "template"
    ):
        """
        Initialize A-TEAM Conductor.

        Args:
            experts: List of Expert objects to participate
            max_rounds: Maximum debate rounds (default: 5)
            consensus_threshold: Required consensus (default: 1.0 = 100%)
            mode: Usage mode ("template", "custom", "automatic")
        """
        self.experts = experts
        self.max_rounds = max_rounds
        self.consensus_threshold = consensus_threshold
        self.mode = mode

        # Initialize debate manager
        self.debate_manager = DebateManager(
            experts=self.experts,
            max_rounds=self.max_rounds,
            consensus_threshold=self.consensus_threshold
        )

        # Auto-selection module (for automatic mode)
        self.auto_selector = dspy.ChainOfThought(AutoTeamSelectionSignature)

        logger.info(
            f" ATeamConductor initialized: mode={mode}, "
            f"experts={len(experts)}, max_rounds={max_rounds}"
        )

    # =========================================================================
    # FACTORY METHODS (Following BrainPreset Pattern)
    # =========================================================================

    @classmethod
    def from_preset(cls, preset_name: str) -> 'ATeamConductor':
        """
        Create conductor from preset name (TEMPLATE MODE).

        Options:
            - "rl_review": RL/MARL system design
            - "architecture_design": Software architecture
            - "product_design": Product, naming, UX
            - "information_flow": Memory, RAG, data pipelines
            - "logic_verification": Algorithmic correctness
            - "full_consensus": All 22+ experts
            - "minimal": Fastest (3 experts)

        Example:
            conductor = ATeamConductor.from_preset("rl_review")
            result = await conductor.generate_task(description, goal)
        """
        try:
            preset = ATeamPreset(preset_name.lower())
        except ValueError:
            logger.warning(f"Unknown preset '{preset_name}', using 'minimal'")
            preset = ATeamPreset.MINIMAL

        config = get_preset_config(preset)
        experts = get_preset_experts(preset)

        logger.info(f" Creating conductor from preset: {preset.value}")
        logger.info(f"   Experts: {len(experts)}, Max rounds: {config['max_rounds']}")

        return cls(
            experts=experts,
            max_rounds=config['max_rounds'],
            consensus_threshold=config['consensus_threshold'],
            mode="template"
        )

    @classmethod
    def from_custom_team(
        cls,
        expert_names: List[str],
        max_rounds: int = 5,
        consensus_threshold: float = 1.0
    ) -> 'ATeamConductor':
        """
        Create conductor from custom expert selection (CUSTOM MODE).

        User specifies exact experts to include.

        Example:
            conductor = ATeamConductor.from_custom_team(
                expert_names=["Alan Turing", "Omar Khattab", "Alex Chen"],
                max_rounds=3
            )
            result = await conductor.generate_task(description, goal)
        """
        experts = get_experts_by_names(expert_names)

        if not experts:
            raise ValueError(f"No valid experts found in: {expert_names}")

        logger.info(f" Creating conductor from custom team: {len(experts)} experts")
        logger.info(f"   Experts: {', '.join([e.name for e in experts])}")

        return cls(
            experts=experts,
            max_rounds=max_rounds,
            consensus_threshold=consensus_threshold,
            mode="custom"
        )

    @classmethod
    def from_auto_selection(
        cls,
        max_rounds: int = 5,
        consensus_threshold: float = 1.0
    ) -> 'ATeamConductor':
        """
        Create conductor with automatic expert selection (AUTOMATIC MODE - DEFAULT).

        Jotty analyzes the task and automatically selects optimal experts.
        This is the default mode for JustJot.ai integration.

        Example:
            conductor = ATeamConductor.from_auto_selection()
            result = await conductor.generate_task(description, goal)
            # Jotty will analyze and select experts automatically
        """
        # Start with all experts available
        # Actual selection happens in generate_task()
        experts = get_all_experts()

        logger.info(f" Creating conductor with auto-selection")
        logger.info(f"   Available: {len(experts)} experts")

        return cls(
            experts=experts,
            max_rounds=max_rounds,
            consensus_threshold=consensus_threshold,
            mode="automatic"
        )

    # =========================================================================
    # TASK GENERATION (Main API)
    # =========================================================================

    async def generate_task(
        self,
        task_description: str,
        goal_context: str = ""
    ) -> TaskGenerationResult:
        """
        Generate comprehensive task via A-TEAM consensus.

        In automatic mode, Jotty will first analyze the task and select
        optimal experts before running the debate.

        Args:
            task_description: Task/problem description
            goal_context: Overall goal and context (optional)

        Returns:
            TaskGenerationResult with consensus decision and comprehensive task

        Example:
            result = await conductor.generate_task(
                task_description="Refactor the RL reward system to support multi-objective optimization",
                goal_context="Building a production-ready MARL framework"
            )

            print(result.consensus_decision)
            print(result.comprehensive_task)
        """
        logger.info(f" Generating task via A-TEAM ({self.mode} mode)")
        logger.info(f"   Task: {task_description[:100]}...")

        # Automatic mode: Select experts first
        if self.mode == "automatic":
            selected_experts = await self._auto_select_experts(
                task_description=task_description,
                goal_context=goal_context
            )
            # Update debate manager with selected experts
            self.debate_manager.experts = selected_experts

        # Run debate
        consensus_result: ConsensusResult = await self.debate_manager.run_debate(
            task_description=task_description,
            goal_context=goal_context
        )

        # Package result
        result = TaskGenerationResult(
            consensus_decision=consensus_result.final_decision,
            comprehensive_task=consensus_result.comprehensive_task,
            reasoning=consensus_result.reasoning,
            expert_votes=consensus_result.expert_votes,
            total_rounds=consensus_result.total_rounds,
            consensus_reached=consensus_result.consensus_reached,
            mode=self.mode
        )

        logger.info(f" Task generation complete!")
        logger.info(f"   Rounds: {result.total_rounds}")
        logger.info(f"   Consensus: {result.consensus_reached}")

        return result

    async def _auto_select_experts(
        self,
        task_description: str,
        goal_context: str
    ) -> List[Expert]:
        """
        Automatically select optimal experts for the task (AUTOMATIC MODE).

        Uses DSPy to analyze task and choose relevant experts.
        """
        logger.info(f" Auto-selecting experts for task...")

        # Format available experts for LLM
        import json
        available_experts_data = [
            {
                "name": expert.name,
                "title": expert.title,
                "domain": expert.domain.value,
                "expertise": expert.expertise
            }
            for expert in self.experts
        ]
        available_experts_str = json.dumps(available_experts_data, indent=2)

        try:
            # Ask LLM to select experts
            result = self.auto_selector(
                task_description=task_description,
                goal_context=goal_context,
                available_experts=available_experts_str
            )

            # Parse selected expert names
            selected_names = [name.strip() for name in result.selected_experts.split(",")]
            selected_experts = get_experts_by_names(selected_names)

            # Ensure we have at least 3 experts
            if len(selected_experts) < 3:
                logger.warning(f"Auto-selection returned only {len(selected_experts)} experts, using minimal preset")
                selected_experts = get_preset_experts(ATeamPreset.MINIMAL)

            logger.info(f"   Selected {len(selected_experts)} experts: {', '.join([e.name for e in selected_experts])}")
            logger.info(f"   Reasoning: {result.reasoning[:200]}...")

            return selected_experts

        except Exception as e:
            logger.error(f"Auto-selection failed: {e}, falling back to minimal preset")
            return get_preset_experts(ATeamPreset.MINIMAL)

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_team_info(self) -> Dict[str, Any]:
        """Get information about current expert team."""
        return {
            'mode': self.mode,
            'expert_count': len(self.experts),
            'expert_names': [e.name for e in self.experts],
            'max_rounds': self.max_rounds,
            'consensus_threshold': self.consensus_threshold
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def generate_task_with_preset(
    preset_name: str,
    task_description: str,
    goal_context: str = ""
) -> TaskGenerationResult:
    """
    Convenience function for one-shot task generation with preset.

    Example:
        result = await generate_task_with_preset(
            preset_name="rl_review",
            task_description="Implement multi-objective RL",
            goal_context="Production MARL framework"
        )
    """
    conductor = ATeamConductor.from_preset(preset_name)
    return await conductor.generate_task(task_description, goal_context)


async def generate_task_auto(
    task_description: str,
    goal_context: str = ""
) -> TaskGenerationResult:
    """
    Convenience function for automatic expert selection (DEFAULT for JustJot.ai).

    Example:
        result = await generate_task_auto(
            task_description="Implement multi-objective RL",
            goal_context="Production MARL framework"
        )
    """
    conductor = ATeamConductor.from_auto_selection()
    return await conductor.generate_task(task_description, goal_context)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'ATeamConductor',
    'TaskGenerationResult',
    'generate_task_with_preset',
    'generate_task_auto',
]
