from typing import Any
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

import asyncio
import logging
import traceback
from abc import abstractmethod
from datetime import datetime
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Type

from ..base_swarm import BaseSwarm, SwarmBaseConfig, SwarmResult, AgentRole
from ..swarm_types import _split_field, _safe_join, _safe_num
from .agent_team import AgentTeam, CoordinationPattern, TeamResult

logger = logging.getLogger(__name__)


class PhaseExecutor:
    """Manages phase execution with consistent tracing, timing, and error handling.

    Eliminates boilerplate in concrete swarms by encapsulating:
    - Phase logging (start/end with emoji)
    - Automatic _trace_phase() calls
    - Exception-safe parallel execution via asyncio.gather
    - Standard error result building

    Usage inside a DomainSwarm subclass::

        executor = self._phase_executor()
        result = await executor.run_phase(
            1, "Data Profiling", "DataProfiler", AgentRole.ACTOR,
            self._profiler.profile(summary, sample, columns),
            tools_used=['data_profile'],
        )
    """

    def __init__(self, swarm: 'DomainSwarm') -> None:
        self.swarm = swarm
        self._start_time = datetime.now()

    def elapsed(self) -> float:
        """Seconds since executor was created."""
        return (datetime.now() - self._start_time).total_seconds()

    async def run_phase(self, phase_num: int, phase_name: str, agent_name: str, agent_role: AgentRole, coro: Any, input_data: Dict[str, Any] = None, tools_used: List[str] = None) -> Any:
        """Execute a single agent phase with automatic tracing.

        Args:
            phase_num: Phase number for logging (e.g. 1, 2, 3)
            phase_name: Human-readable phase name
            agent_name: Agent class/display name for tracing
            agent_role: AgentRole enum value
            coro: Awaitable coroutine to execute
            input_data: Optional dict logged as phase input
            tools_used: Optional tool names for tracing

        Returns:
            The raw result from the coroutine.

        Raises:
            Re-raises any exception from the coroutine after logging.
        """
        logger.info(f"Phase {phase_num}: {phase_name}...")
        phase_start = datetime.now()

        result = await coro

        # Determine success heuristic
        success = True
        if isinstance(result, dict) and 'error' in result:
            success = False

        output_data = {}
        if isinstance(result, dict):
            output_data = {k: str(v)[:200] for k, v in list(result.items())[:5]}

        self.swarm._trace_phase(
            agent_name, agent_role,
            input_data or {},
            output_data,
            success=success,
            phase_start=phase_start,
            tools_used=tools_used,
        )

        return result

    async def run_parallel(
        self,
        phase_num: int,
        phase_name: str,
        tasks: List[Tuple[str, AgentRole, Any, List[str]]],
    ) -> List[Any]:
        """Execute parallel agents with asyncio.gather + per-agent tracing.

        Args:
            phase_num: Phase number for logging
            phase_name: Human-readable phase name
            tasks: List of (agent_name, agent_role, coro, tools_used) tuples

        Returns:
            List of results (exceptions converted to {'error': str(e)} dicts).
        """
        logger.info(f"Phase {phase_num}: {phase_name} ({len(tasks)} agents parallel)...")
        phase_start = datetime.now()

        coros = [t[2] for t in tasks]
        raw_results = await asyncio.gather(*coros, return_exceptions=True)

        results = []
        for i, raw in enumerate(raw_results):
            agent_name, agent_role, _, tools_used = tasks[i]
            if isinstance(raw, Exception):
                result = {'error': str(raw)}
                success = False
            else:
                result = raw
                success = not (isinstance(result, dict) and 'error' in result)

            output_data = {}
            if isinstance(result, dict):
                output_data = {'has_error': 'error' in result}

            self.swarm._trace_phase(
                agent_name, agent_role,
                {}, output_data,
                success=success,
                phase_start=phase_start,
                tools_used=tools_used or [],
            )
            results.append(result)

        return results

    def build_error_result(
        self,
        result_class: Type[SwarmResult],
        error: Exception,
        config_name: str,
        config_domain: str,
    ) -> SwarmResult:
        """Build a standard error result for except blocks."""
        return result_class(
            success=False,
            swarm_name=config_name,
            domain=config_domain,
            output={},
            execution_time=self.elapsed(),
            error=str(error),
        )


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
    # Subclasses set this to a DSPy Signature for typed I/O contracts
    SWARM_SIGNATURE: ClassVar[Optional[Type]] = None

    # Defensive utilities available as static methods on all swarms
    _split_field = staticmethod(_split_field)
    _safe_join = staticmethod(_safe_join)
    _safe_num = staticmethod(_safe_num)

    def __init__(self, config: SwarmBaseConfig) -> None:
        """
        Initialize DomainSwarm.

        Args:
            config: Swarm configuration (subclass-specific)
        """
        super().__init__(config)
        self._agents_initialized = False
        self._learning_recorded = False

    def _init_agents(self) -> None:
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

    def _create_agent(self, spec: 'AgentSpec') -> Any:
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
    # SIGNATURE / I/O SCHEMA
    # =========================================================================

    def get_io_schema(self) -> Any:
        """Get typed I/O schema from SWARM_SIGNATURE.

        Returns AgentIOSchema if SWARM_SIGNATURE is set, else None.
        Result is cached on first call.
        """
        if hasattr(self, '_io_schema') and self._io_schema is not None:
            return self._io_schema
        from Jotty.core.agents._execution_types import AgentIOSchema
        if self.SWARM_SIGNATURE is not None:
            self._io_schema = AgentIOSchema.from_dspy_signature(
                self.config.name, self.SWARM_SIGNATURE
            )
        else:
            self._io_schema = None
        return self._io_schema

    def _validate_output_fields(self, result: SwarmResult) -> None:
        """Validate and auto-populate output fields from SWARM_SIGNATURE."""
        schema = self.get_io_schema()
        if schema is None:
            return
        if not hasattr(result, 'output') or not isinstance(result.output, dict):
            return

        expected = {p.name: p for p in schema.outputs}
        actual = set(result.output.keys())
        missing = set(expected.keys()) - actual
        name = self.config.name

        # Auto-populate missing fields with type-appropriate defaults
        if missing:
            logger.info("%s: auto-populating %d missing output fields: %s", name, len(missing), missing)
            for field_name in missing:
                result.output[field_name] = ''

        # Coerce non-string values using TypeCoercer where type_hint != 'str'
        from Jotty.core.agents._execution_types import TypeCoercer
        for field_name, value in list(result.output.items()):
            if field_name in expected:
                param = expected[field_name]
                hint = (param.type_hint or 'str').lower()
                if hint not in ('str', 'string', ''):
                    coerced, error = TypeCoercer.coerce(value, param.type_hint)
                    if not error and coerced is not value:
                        result.output[field_name] = coerced

    # =========================================================================
    # TEAM COORDINATION
    # =========================================================================

    async def execute_team(self, task: Any, context: Dict[str, Any] = None, **kwargs: Any) -> TeamResult:
        """
        Execute the agent team with its configured coordination pattern.

        This method leverages the team's coordination pattern (PIPELINE,
        PARALLEL, CONSENSUS, etc.) to orchestrate agent execution.
        Also wires in coalition formation and smart routing from
        SwarmIntelligence when available.

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

        # Wire in coordination protocols from SwarmIntelligence
        si = self._swarm_intelligence
        if si and si.agent_profiles:
            task_str = str(task)[:200] if task else ""
            task_type = self.__class__.__name__

            # Coalition formation: for PARALLEL teams with 2+ agents,
            # form a coalition so agents are tracked as a coordinated unit
            if (self.AGENT_TEAM.pattern == CoordinationPattern.PARALLEL
                    and len(self.AGENT_TEAM) >= 2):
                try:
                    agent_names = [
                        getattr(self, attr, None).__class__.__name__
                        for attr, _ in self.AGENT_TEAM
                        if hasattr(self, attr)
                    ]
                    coalition = si.form_coalition(
                        task_type=task_type,
                        required_roles=[],
                        min_agents=2,
                        max_agents=len(agent_names)
                    )
                    if coalition:
                        full_context["coalition_id"] = coalition.coalition_id
                        full_context["coalition_leader"] = coalition.leader
                except Exception:
                    pass  # Non-blocking

            # Smart routing: use SwarmIntelligence to determine optimal
            # agent ordering or leader selection
            try:
                route = si.smart_route(
                    task_id=f"{task_type}_{id(task)}",
                    task_type=task_type,
                    task_description=task_str,
                    prefer_coalition=False,  # Already handled above
                    use_auction=(self.AGENT_TEAM.pattern == CoordinationPattern.NONE),
                    use_hierarchy=True
                )
                if route.get("assigned_agent"):
                    full_context["routed_agent"] = route["assigned_agent"]
                    full_context["routing_method"] = route.get("method", "unknown")
                    full_context["routing_confidence"] = route.get("confidence", 0.5)
            except Exception:
                pass  # Non-blocking

        return await self.AGENT_TEAM.execute(task, full_context, **kwargs)

    def has_team_coordination(self) -> bool:
        """Check if team has a coordination pattern configured."""
        if not self.AGENT_TEAM:
            return False
        return self.AGENT_TEAM.pattern != CoordinationPattern.NONE

    # =========================================================================
    # PHASE EXECUTOR HELPERS
    # =========================================================================

    def _phase_executor(self) -> PhaseExecutor:
        """Create a PhaseExecutor bound to this swarm."""
        return PhaseExecutor(self)

    async def _safe_execute_domain(
        self,
        task_type: str,
        default_tools: List[str],
        result_class: Type[SwarmResult],
        execute_fn: Callable,
        output_data_fn: Callable = None,
        input_data_fn: Callable = None,
    ) -> SwarmResult:
        """Wrap phase execution with standard try/except + _post_execute_learning.

        Concrete swarms call this from their main method (e.g. analyze())
        to eliminate repeated try/except/timing/learning boilerplate.

        Args:
            task_type: Task type string for learning (e.g. 'data_analysis')
            default_tools: Default tool names for _get_active_tools()
            result_class: SwarmResult subclass for error result building
            execute_fn: Async callable that performs domain phases, returns result
            output_data_fn: Optional callable(result) -> dict for learning output
            input_data_fn: Optional callable() -> dict for learning input

        Returns:
            SwarmResult from execute_fn, or error result on failure.
        """
        executor = self._phase_executor()
        try:
            result = await execute_fn(executor)

            # Record post-execution learning (success path)
            exec_time = executor.elapsed()
            output_data = output_data_fn(result) if output_data_fn else None
            input_data = input_data_fn() if input_data_fn else None
            await self._post_execute_learning(
                success=result.success if hasattr(result, 'success') else True,
                execution_time=exec_time,
                tools_used=self._get_active_tools(default_tools),
                task_type=task_type,
                output_data=output_data,
                input_data=input_data,
            )
            self._learning_recorded = True
            return result

        except Exception as e:
            logger.error(f" {self.__class__.__name__} error: {e}")
            traceback.print_exc()
            exec_time = executor.elapsed()
            await self._post_execute_learning(
                success=False,
                execution_time=exec_time,
                tools_used=self._get_active_tools(default_tools),
                task_type=task_type,
            )
            self._learning_recorded = True
            return executor.build_error_result(
                result_class, e, self.config.name, self.config.domain,
            )

    # =========================================================================
    # EXECUTION TEMPLATE
    # =========================================================================

    async def execute(self, *args: Any, **kwargs: Any) -> SwarmResult:
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
        self._learning_recorded = False  # Reset per-execution

        # Pre-execute learning (loads context, warmup, etc.)
        try:
            await self._pre_execute_learning()
        except Exception as e:
            logger.warning(f"Pre-execute learning failed: {e}")

        # Auto-remap 'task' kwarg to the swarm's first signature input field.
        # Pipeline passes 'task' generically; swarms expect domain-specific names
        # (e.g. 'requirements', 'code', 'data'). SWARM_SIGNATURE enables auto-wiring.
        if self.SWARM_SIGNATURE is not None and 'task' in kwargs:
            schema = self.get_io_schema()
            if schema and schema.inputs:
                first_input = schema.inputs[0].name
                if first_input != 'task' and first_input not in kwargs:
                    kwargs[first_input] = kwargs.pop('task')

        result = None
        start_time = __import__('time').time()
        try:
            result = await self._execute_domain(*args, **kwargs)
        finally:
            # Post-execute learning (evaluation, improvement)
            # Skip if _safe_execute_domain already recorded learning
            try:
                if (
                    not self._learning_recorded
                    and result is not None
                    and hasattr(self, '_post_execute_learning')
                ):
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

        # Validate output fields against signature contract
        if result is not None:
            self._validate_output_fields(result)

        # Attach collected traces to result (swarms record traces via
        # _trace_phase but subclasses rarely copy them into SwarmResult)
        if result is not None and hasattr(result, 'agent_traces') and hasattr(self, '_traces'):
            if not result.agent_traces and self._traces:
                result.agent_traces = list(self._traces)

        return result

    @abstractmethod
    async def _execute_domain(self, *args: Any, **kwargs: Any) -> SwarmResult:
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

    # =========================================================================
    # COMPOSITE AGENT BRIDGE
    # =========================================================================

    def to_composite(self, signature: Any = None) -> Any:
        """Convert this swarm to a CompositeAgent for composition.

        Returns a CompositeAgent that delegates execute() to this swarm,
        preserving all learning hooks and agent lifecycle.
        Auto-uses SWARM_SIGNATURE if no explicit signature provided.

        Args:
            signature: Optional DSPy signature for typed I/O

        Returns:
            CompositeAgent wrapping this swarm
        """
        from Jotty.core.agents.base.composite_agent import CompositeAgent
        return CompositeAgent.from_swarm(self, signature=signature or self.SWARM_SIGNATURE)

    def __repr__(self) -> str:
        agent_count = len(self.AGENT_TEAM) if self.AGENT_TEAM else 0
        pattern = self.AGENT_TEAM.pattern.value if self.AGENT_TEAM else "none"
        return f"{self.__class__.__name__}(agents={agent_count}, pattern={pattern}, initialized={self._agents_initialized})"


__all__ = ['DomainSwarm', 'PhaseExecutor', '_split_field', '_safe_join', '_safe_num']
