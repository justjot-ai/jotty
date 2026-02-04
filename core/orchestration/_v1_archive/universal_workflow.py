"""
UniversalWorkflow - Adaptive Multi-Agent Orchestration
======================================================

Thin wrapper around Conductor adding:
- Goal analysis (auto-select workflow mode)
- New workflow patterns (hierarchical, debate, round-robin, pipeline, swarm)
- Flexible context handling

ZERO DUPLICATION - Delegates to:
- Conductor for tool management, learning, validation, memory
- hybrid_team_template for P2P and sequential phases
- Existing infrastructure for everything else

Usage:
    workflow = UniversalWorkflow(actors, config)
    result = await workflow.run(
        goal="Build stock screener",
        context={'data_folder': '/path/to/data'},
        mode='auto'  # Auto-selects best workflow
    )
"""

import asyncio
import dspy
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path

# REUSE existing infrastructure (NO DUPLICATION!)
from .conductor import Conductor
from ..persistence.shared_context import SharedContext
from ..foundation.types.agent_types import SharedScratchpad, AgentMessage, CommunicationType
from ..persistence.scratchpad_persistence import ScratchpadPersistence

# REUSE existing workflow phases (NO DUPLICATION!)
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from templates.hybrid_team_template import p2p_discovery_phase, sequential_delivery_phase

logger = logging.getLogger(__name__)


# =============================================================================
# GOAL ANALYZER (NEW - Not in Conductor)
# =============================================================================

class GoalAnalysisSignature(dspy.Signature):
    """Analyze goal and recommend workflow mode."""
    goal: str = dspy.InputField(desc="User's goal")
    context: str = dspy.InputField(desc="Available context (data, code, etc.)")

    complexity: str = dspy.OutputField(desc="simple/medium/complex")
    uncertainty: str = dspy.OutputField(desc="clear/ambiguous/exploratory")
    recommended_mode: str = dspy.OutputField(desc="Workflow mode: sequential/parallel/p2p/hierarchical/debate/round-robin/pipeline/swarm")
    reasoning: str = dspy.OutputField(desc="Why this mode was recommended")
    num_agents: int = dspy.OutputField(desc="Recommended number of agents")


class GoalAnalyzer:
    """
    NEW: Analyzes goals and recommends workflow mode.

    Not in Conductor - this is the only new component!
    """

    def __init__(self):
        try:
            self.analyzer = dspy.ChainOfThought(GoalAnalysisSignature)
        except:
            self.analyzer = None
        logger.info("🎯 GoalAnalyzer initialized")

    async def analyze(self, goal: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze goal and recommend workflow."""
        if not self.analyzer:
            # Fallback: default to sequential
            return {
                'complexity': 'medium',
                'uncertainty': 'clear',
                'recommended_mode': 'sequential',
                'reasoning': 'Default mode (DSPy not available)',
                'num_agents': 3
            }

        # Format context
        context_str = ", ".join([f"{k}={v}" for k, v in (context or {}).items()])

        # Analyze
        result = self.analyzer(
            goal=goal,
            context=context_str or "None provided"
        )

        return {
            'complexity': result.complexity,
            'uncertainty': result.uncertainty,
            'recommended_mode': result.recommended_mode,
            'reasoning': result.reasoning,
            'num_agents': int(result.num_agents) if hasattr(result, 'num_agents') else 3
        }


# =============================================================================
# CONTEXT HANDLER (NEW - Flexible context parsing)
# =============================================================================

@dataclass
class StructuredContext:
    """Structured context for workflows."""
    goal: str

    # File/Folder paths
    data_folder: Optional[str] = None
    codebase: Optional[str] = None
    requirements_doc: Optional[str] = None

    # URLs
    github_repo: Optional[str] = None
    api_docs: Optional[str] = None

    # Database/API
    database: Optional[str] = None
    api_endpoint: Optional[str] = None
    api_key: Optional[str] = None

    # Previous work
    session_id: Optional[str] = None
    previous_output: Optional[str] = None

    # Constraints
    time_limit: Optional[str] = None
    quality_threshold: float = 0.85

    # User preferences
    coding_style: Optional[str] = None
    frameworks: List[str] = None

    # Other
    raw_context: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for agents."""
        result = {'goal': self.goal}
        for key, value in self.__dict__.items():
            if value is not None and key != 'goal':
                result[key] = value
        return result


class ContextHandler:
    """Parse and validate flexible context."""

    @staticmethod
    def parse(goal: str, context: Optional[Dict[str, Any]] = None) -> StructuredContext:
        """Parse flexible context into structured format."""
        ctx = context or {}
        return StructuredContext(
            goal=goal,
            data_folder=ctx.get('data_folder') or ctx.get('data') or ctx.get('data_location'),
            codebase=ctx.get('codebase') or ctx.get('code') or ctx.get('repo'),
            requirements_doc=ctx.get('requirements_doc') or ctx.get('requirements'),
            github_repo=ctx.get('github_repo') or ctx.get('github'),
            api_docs=ctx.get('api_docs'),
            database=ctx.get('database') or ctx.get('db'),
            api_endpoint=ctx.get('api_endpoint') or ctx.get('api'),
            api_key=ctx.get('api_key'),
            session_id=ctx.get('session_id'),
            previous_output=ctx.get('previous_output'),
            time_limit=ctx.get('time_limit'),
            quality_threshold=ctx.get('quality_threshold', 0.85),
            coding_style=ctx.get('coding_style'),
            frameworks=ctx.get('frameworks', []),
            raw_context=ctx
        )


# =============================================================================
# UNIVERSAL WORKFLOW (Thin Wrapper)
# =============================================================================

class UniversalWorkflow:
    """
    Universal multi-agent workflow with adaptive mode selection.

    THIN WRAPPER - Delegates to:
    - Conductor for existing modes (sequential, parallel, p2p)
    - hybrid_team_template for P2P/Sequential phases
    - Existing infrastructure for tools, learning, validation

    ADDS ONLY:
    - Goal analysis (auto-select mode)
    - New workflow modes (hierarchical, debate, round-robin, pipeline, swarm)
    - Flexible context handling

    NO DUPLICATION!
    """

    def __init__(self, actors, config):
        """
        Initialize workflow.

        Creates Conductor internally (gets ALL infrastructure).
        """
        # DELEGATE to Conductor for ALL heavy lifting
        self.conductor = Conductor(actors, config)

        # REUSE Conductor's infrastructure (NO DUPLICATION!)
        self.tool_registry = self.conductor.metadata_tool_registry
        self.tool_manager = self.conductor.tool_manager
        self.shared_context = self.conductor.shared_context
        self.state_manager = self.conductor.state_manager
        self.memory = self.conductor.memory if hasattr(self.conductor, 'memory') else None

        # Create proper SharedScratchpad object (not dict!)
        # Conductor uses dict, but workflow modes expect SharedScratchpad
        from ..foundation.types.agent_types import SharedScratchpad
        self.scratchpad = SharedScratchpad()

        # ONLY NEW components
        self.goal_analyzer = GoalAnalyzer()
        self.context_handler = ContextHandler()
        self.persistence = ScratchpadPersistence()

        logger.info("🚀 UniversalWorkflow initialized (thin wrapper around Conductor)")

    async def run(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        mode: Optional[str] = None,  # 'auto', 'sequential', 'parallel', 'p2p', 'hierarchical', 'debate', 'round-robin', 'pipeline', 'swarm'
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run workflow with adaptive mode selection.

        Args:
            goal: What to accomplish
            context: Flexible context (data_folder, codebase, urls, etc.)
            mode: Workflow mode ('auto' for auto-selection)
            **kwargs: Additional parameters (max_iterations, target_reward, etc.)

        Returns:
            {
                'status': 'success',
                'results': ...,
                'session_id': ...,
                'mode_used': ...,
                'analysis': ...
            }
        """

        # Parse context
        structured_ctx = self.context_handler.parse(goal, context)

        # Phase 0: Analyze goal (if mode is auto)
        if mode is None or mode == 'auto':
            analysis = await self.goal_analyzer.analyze(goal, context)
            mode = analysis['recommended_mode']
            logger.info(f"🎯 Goal analysis complete: {mode} mode recommended")
            logger.info(f"   Reasoning: {analysis['reasoning']}")
        else:
            analysis = None

        # Route to appropriate workflow
        logger.info(f"🎬 Running {mode} workflow")

        # Existing modes → DELEGATE to Conductor (NO DUPLICATION!)
        if mode in ['sequential', 'parallel']:
            result = await self.conductor.run(goal, **kwargs)
            return {
                'status': 'success',
                'results': result,
                'mode_used': mode,
                'analysis': analysis
            }

        # P2P hybrid → REUSE hybrid_team_template (NO DUPLICATION!)
        elif mode == 'p2p' or mode == 'hybrid':
            result = await self._run_p2p_hybrid(goal, structured_ctx, **kwargs)
            return {
                'status': 'success',
                'results': result,
                'mode_used': 'p2p',
                'analysis': analysis
            }

        # NEW modes (implemented here)
        elif mode == 'hierarchical':
            result = await self._run_hierarchical(goal, structured_ctx, **kwargs)
        elif mode == 'debate':
            result = await self._run_debate(goal, structured_ctx, **kwargs)
        elif mode == 'round-robin':
            result = await self._run_round_robin(goal, structured_ctx, **kwargs)
        elif mode == 'pipeline':
            result = await self._run_pipeline(goal, structured_ctx, **kwargs)
        elif mode == 'swarm':
            result = await self._run_swarm(goal, structured_ctx, **kwargs)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return {
            'status': 'success',
            'results': result,
            'mode_used': mode,
            'analysis': analysis
        }

    # =========================================================================
    # EXISTING MODE: P2P Hybrid (REUSE hybrid_team_template)
    # =========================================================================

    async def _run_p2p_hybrid(
        self,
        goal: str,
        context: StructuredContext,
        num_discovery_agents: int = 3,
        num_delivery_agents: int = 3,
        **kwargs
    ) -> Dict[str, Any]:
        """
        P2P Discovery + Sequential Delivery.

        REUSES: hybrid_team_template.p2p_discovery_phase() and sequential_delivery_phase()
        NO DUPLICATION!
        """

        session_name = f"p2p_hybrid_{Path.cwd().name}"
        session_file = self.persistence.create_session(session_name)

        # Get tools from Conductor (REUSE!)
        tools = self.conductor._get_auto_discovered_dspy_tools()

        # Phase 1: P2P Discovery (REUSE existing function!)
        discovery_configs = [
            {
                'name': f'Discovery Agent {i+1}',
                'agent': dspy.ChainOfThought(dspy.Signature),  # Placeholder
                'expert': None,
                'tools': tools
            }
            for i in range(num_discovery_agents)
        ]

        discoveries = await p2p_discovery_phase(
            agents_config=discovery_configs,
            task=goal,
            shared_context=self.shared_context,
            scratchpad=self.scratchpad,
            persistence=self.persistence,
            session_file=session_file
        )

        # Phase 2: Sequential Delivery (REUSE existing function!)
        delivery_configs = [
            {
                'name': f'Delivery Agent {i+1}',
                'agent': dspy.ChainOfThought(dspy.Signature),  # Placeholder
                'expert': None,
                'tools': tools
            }
            for i in range(num_delivery_agents)
        ]

        deliverables = await sequential_delivery_phase(
            agents_config=delivery_configs,
            discoveries=discoveries,
            goal=goal,
            shared_context=self.shared_context,
            scratchpad=self.scratchpad,
            persistence=self.persistence,
            session_file=session_file
        )

        return {
            'discoveries': discoveries,
            'deliverables': deliverables,
            'session_file': session_file
        }

    # =========================================================================
    # NEW MODES (Not in Conductor - Implemented here)
    # =========================================================================

    async def _run_hierarchical(
        self,
        goal: str,
        context: StructuredContext,
        **kwargs
    ) -> Dict[str, Any]:
        """Hierarchical: Lead agent + sub-agents."""
        from .workflow_modes.hierarchical import run_hierarchical_mode

        tools = self.conductor._get_auto_discovered_dspy_tools()

        return await run_hierarchical_mode(
            goal=goal,
            tools=tools,
            shared_context=self.shared_context,
            scratchpad=self.scratchpad,
            persistence=self.persistence,
            **kwargs
        )

    async def _run_debate(
        self,
        goal: str,
        context: StructuredContext,
        **kwargs
    ) -> Dict[str, Any]:
        """Debate: Competing solutions → critique → vote."""
        from .workflow_modes.debate import run_debate_mode

        tools = self.conductor._get_auto_discovered_dspy_tools()

        return await run_debate_mode(
            goal=goal,
            tools=tools,
            shared_context=self.shared_context,
            scratchpad=self.scratchpad,
            persistence=self.persistence,
            **kwargs
        )

    async def _run_round_robin(
        self,
        goal: str,
        context: StructuredContext,
        num_rounds: int = 3,
        **kwargs
    ) -> Dict[str, Any]:
        """Round-Robin: Iterative refinement."""
        from .workflow_modes.round_robin import run_round_robin_mode

        tools = self.conductor._get_auto_discovered_dspy_tools()

        return await run_round_robin_mode(
            goal=goal,
            tools=tools,
            shared_context=self.shared_context,
            scratchpad=self.scratchpad,
            persistence=self.persistence,
            num_rounds=num_rounds,
            **kwargs
        )

    async def _run_pipeline(
        self,
        goal: str,
        context: StructuredContext,
        **kwargs
    ) -> Dict[str, Any]:
        """Pipeline: Data flow through stages."""
        from .workflow_modes.pipeline import run_pipeline_mode

        tools = self.conductor._get_auto_discovered_dspy_tools()

        return await run_pipeline_mode(
            goal=goal,
            tools=tools,
            shared_context=self.shared_context,
            scratchpad=self.scratchpad,
            persistence=self.persistence,
            **kwargs
        )

    async def _run_swarm(
        self,
        goal: str,
        context: StructuredContext,
        **kwargs
    ) -> Dict[str, Any]:
        """Swarm: Self-organizing agents."""
        from .workflow_modes.swarm import run_swarm_mode

        tools = self.conductor._get_auto_discovered_dspy_tools()

        return await run_swarm_mode(
            goal=goal,
            tools=tools,
            shared_context=self.shared_context,
            scratchpad=self.scratchpad,
            persistence=self.persistence,
            **kwargs
        )
