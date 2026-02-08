"""
DomainSwarm - Template Base Class for Domain-Specific Swarms
=============================================================

Architecture:
    Agent (skills, LLM) → Team (coordination) → Swarm (learning)

Provides:
- Declarative agent team initialization via AGENT_TEAM
- Automatic agent lifecycle management
- Team coordination patterns (pipeline, parallel, consensus, etc.)
- Template execute() with pre/post learning hooks

Subclasses define:
- AGENT_TEAM: AgentTeam class attribute with optional coordination pattern
- _execute_domain(): Domain-specific logic (or use team coordination)

Usage:
    # Manual coordination (swarm handles agent orchestration)
    class CodingSwarm(DomainSwarm):
        AGENT_TEAM = AgentTeam.define(
            (ArchitectAgent, "Architect"),
            (DeveloperAgent, "Developer"),
        )

        async def _execute_domain(self, requirements: str, **kwargs):
            arch = await self._architect.design(requirements)
            code = await self._developer.develop(arch)
            return CodingResult(code=code)

    # Team coordination (team handles agent orchestration)
    class ReviewSwarm(DomainSwarm):
        AGENT_TEAM = AgentTeam.define(
            (SecurityReviewer, "Security"),
            (PerformanceReviewer, "Performance"),
            pattern=CoordinationPattern.PARALLEL,
            merge_strategy=MergeStrategy.CONCAT,
        )

        async def _execute_domain(self, code: str, **kwargs):
            # Team handles parallel execution and merging
            team_result = await self.execute_team(task=code, **kwargs)
            return ReviewResult(findings=team_result.merged_output)

Author: Jotty Team
Date: February 2026
"""

import logging
from abc import abstractmethod
from typing import Any, ClassVar, Dict, Optional

from ..base_swarm import BaseSwarm, SwarmConfig, SwarmResult
from .agent_team import AgentTeam, CoordinationPattern, TeamResult

logger = logging.getLogger(__name__)


class DomainSwarm(BaseSwarm):
    """
    Base class for domain-specific swarms.

    Inherits self-improving loop from BaseSwarm and adds:
    - Declarative agent team via AGENT_TEAM class attribute
    - Automatic agent initialization
    - Team coordination patterns
    - Template execute() pattern

    Class Attributes:
        AGENT_TEAM: Optional AgentTeam defining the swarm's agents.
                    If None, subclass must override _init_agents().

    Team Coordination:
        If AGENT_TEAM has a coordination pattern (PIPELINE, PARALLEL, etc.),
        use execute_team() to leverage automatic coordination.
    """

    # Subclasses override this with their agent team
    AGENT_TEAM: ClassVar[Optional[AgentTeam]] = None

    def __init__(self, config: SwarmConfig):
        """
        Initialize DomainSwarm.

        Args:
            config: Swarm configuration (subclass-specific)
        """
        super().__init__(config)
        self._agents_initialized = False

    def _init_agents(self):
        """
        Initialize agents from AGENT_TEAM definition.

        Called automatically by execute() before domain logic.
        Can be overridden for custom initialization.
        """
        if self._agents_initialized:
            return

        self._init_shared_resources()

        # Auto-initialize agents from team definition
        if self.AGENT_TEAM:
            agent_instances = {}

            for attr_name, spec in self.AGENT_TEAM:
                try:
                    agent = self._create_agent(spec)
                    setattr(self, attr_name, agent)
                    agent_instances[attr_name] = agent
                    logger.debug(f"Initialized {spec.display_name} -> {attr_name}")
                except Exception as e:
                    logger.error(f"Failed to initialize {spec.display_name}: {e}")
                    raise

            # Pass instances to team for coordination
            self.AGENT_TEAM.set_instances(agent_instances)

        self._agents_initialized = True
        logger.info(f"{self.__class__.__name__} agents initialized")

    def _create_agent(self, spec: 'AgentSpec'):
        """
        Create an agent instance with dynamic parameter binding.

        Inspects the agent's __init__ signature and only passes
        parameters that it accepts. This allows old agents to work
        without modification while new agents can opt-in to features
        like learned_context.
        """
        import inspect

        # Available parameters to pass
        available_params = {
            'memory': self._memory,
            'context': self._context,
            'bus': self._bus,
            'learned_context': self._agent_context(spec.display_name),
        }

        # Inspect agent's __init__ signature
        try:
            sig = inspect.signature(spec.agent_class.__init__)
            param_names = set(sig.parameters.keys()) - {'self'}
        except (ValueError, TypeError):
            # Fallback: try positional args (memory, context, bus)
            param_names = {'memory', 'context', 'bus'}

        # Check if agent accepts **kwargs
        accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in sig.parameters.values()
        ) if 'sig' in dir() else False

        # Build kwargs with only accepted parameters
        kwargs = {}
        for name, value in available_params.items():
            if name in param_names or accepts_kwargs:
                kwargs[name] = value

        # Create agent with matched parameters
        return spec.agent_class(**kwargs)

    def get_agents(self) -> Dict[str, Any]:
        """
        Get all initialized agents.

        Returns:
            Dict mapping attribute names to agent instances
        """
        if not self.AGENT_TEAM:
            return {}

        return {
            attr_name: getattr(self, attr_name, None)
            for attr_name, _ in self.AGENT_TEAM
            if hasattr(self, attr_name)
        }

    # =========================================================================
    # TEAM COORDINATION
    # =========================================================================

    async def execute_team(
        self,
        task: Any,
        context: Dict[str, Any] = None,
        **kwargs
    ) -> TeamResult:
        """
        Execute the agent team with its configured coordination pattern.

        This method leverages the team's coordination pattern (PIPELINE,
        PARALLEL, CONSENSUS, etc.) to orchestrate agent execution.

        Args:
            task: The task/input for the team
            context: Additional context for agents
            **kwargs: Additional arguments passed to agents

        Returns:
            TeamResult with outputs from all agents and merged result

        Example:
            class ReviewSwarm(DomainSwarm):
                AGENT_TEAM = AgentTeam.define(
                    (SecurityReviewer, "Security"),
                    (PerformanceReviewer, "Performance"),
                    pattern=CoordinationPattern.PARALLEL,
                )

                async def _execute_domain(self, code: str, **kwargs):
                    result = await self.execute_team(task=code)
                    return ReviewResult(findings=result.merged_output)
        """
        if not self.AGENT_TEAM:
            raise RuntimeError("No AGENT_TEAM defined for this swarm")

        if not self._agents_initialized:
            self._init_agents()

        # Build context with swarm's shared context
        full_context = context or {}
        if self._context:
            full_context["shared_context"] = self._context

        return await self.AGENT_TEAM.execute(task, full_context, **kwargs)

    def has_team_coordination(self) -> bool:
        """Check if team has a coordination pattern configured."""
        if not self.AGENT_TEAM:
            return False
        return self.AGENT_TEAM.pattern != CoordinationPattern.NONE

    # =========================================================================
    # EXECUTION TEMPLATE
    # =========================================================================

    async def execute(self, *args, **kwargs) -> SwarmResult:
        """
        Execute the swarm with pre/post learning hooks.

        This is a template method that:
        1. Initializes agents
        2. Runs pre-execute learning
        3. Calls _execute_domain() for domain logic
        4. Runs post-execute learning
        5. Returns result

        Subclasses implement _execute_domain() for their logic.
        They can use execute_team() within _execute_domain() to
        leverage team coordination patterns.
        """
        self._init_agents()

        # Pre-execute learning (loads context, warmup, etc.)
        try:
            await self._pre_execute_learning()
        except Exception as e:
            logger.warning(f"Pre-execute learning failed: {e}")

        result = None
        start_time = __import__('time').time()
        try:
            result = await self._execute_domain(*args, **kwargs)
        finally:
            # Post-execute learning (evaluation, improvement)
            # Note: This is optional - some swarms may not have this configured
            try:
                if result is not None and hasattr(self, '_post_execute_learning'):
                    execution_time = __import__('time').time() - start_time
                    success = result.success if hasattr(result, 'success') else True
                    # Try to call with the expected signature
                    await self._post_execute_learning(
                        success=success,
                        execution_time=execution_time,
                        tools_used=[],
                        task_type=self.__class__.__name__,
                        output_data={'result': str(result)[:500]} if result else None,
                        input_data={'args': str(args)[:500], 'kwargs': str(kwargs)[:500]}
                    )
            except TypeError:
                # Signature mismatch - skip silently
                pass
            except Exception as e:
                logger.debug(f"Post-execute learning skipped: {e}")

        return result

    @abstractmethod
    async def _execute_domain(self, *args, **kwargs) -> SwarmResult:
        """
        Implement domain-specific execution logic.

        This method contains the swarm's core functionality.
        Called by execute() after agent initialization and pre-learning.

        Options:
        1. Manual coordination: Call individual agents directly
           result = await self._architect.design(requirements)

        2. Team coordination: Use execute_team() for pattern-based orchestration
           team_result = await self.execute_team(task=requirements)

        Returns:
            SwarmResult with domain-specific output
        """
        pass

    def __repr__(self) -> str:
        agent_count = len(self.AGENT_TEAM) if self.AGENT_TEAM else 0
        pattern = self.AGENT_TEAM.pattern.value if self.AGENT_TEAM else "none"
        return f"{self.__class__.__name__}(agents={agent_count}, pattern={pattern}, initialized={self._agents_initialized})"


__all__ = ['DomainSwarm']
