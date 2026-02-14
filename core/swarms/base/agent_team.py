"""
AgentTeam - Declarative Agent Composition with Coordination Patterns
=====================================================================

Provides:
1. Declarative agent composition via AgentTeam.define()
2. Coordination patterns (pipeline, parallel, consensus, etc.)
3. Automatic agent initialization and execution

Architecture:
    Agent (skills, LLM) → Team (coordination) → Swarm (learning)

Usage:
    # Simple team (manual coordination by swarm)
    class CodingSwarm(DomainSwarm):
        AGENT_TEAM = AgentTeam.define(
            (ArchitectAgent, "Architect"),
            (DeveloperAgent, "Developer"),
        )

    # Team with coordination pattern
    class ReviewSwarm(DomainSwarm):
        AGENT_TEAM = AgentTeam.define(
            (SecurityReviewer, "Security"),
            (PerformanceReviewer, "Performance"),
            (StyleReviewer, "Style"),
            pattern=CoordinationPattern.PARALLEL,
            merge_strategy="combine",
        )

Author: Jotty Team
Date: February 2026
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type, Union
import re

# Canonical definitions live in foundation — single source of truth
from Jotty.core.foundation.types.execution_types import CoordinationPattern, MergeStrategy

logger = logging.getLogger(__name__)


# =============================================================================
# AGENT SPECIFICATION
# =============================================================================

@dataclass
class AgentSpec:
    """Specification for a single agent in a team."""
    agent_class: Type
    display_name: str
    attr_name: Optional[str] = None
    role: Optional[str] = None  # "manager", "worker", etc.
    priority: int = 0  # Higher = executes first in pipeline

    def __post_init__(self):
        if self.attr_name is None:
            # Convert display name to attribute: "Architect" -> "_architect"
            # "TestWriter" -> "_test_writer"
            name = self.display_name
            # Insert underscore before capitals (except first)
            name = re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
            # Replace spaces with underscores
            name = name.replace(' ', '_')
            self.attr_name = f'_{name}'


# =============================================================================
# TEAM RESULT
# =============================================================================

@dataclass
class TeamResult:
    """Result from team execution."""
    success: bool
    outputs: Dict[str, Any]  # agent_name -> output
    merged_output: Any = None  # Combined result (if merge strategy used)
    pattern: CoordinationPattern = CoordinationPattern.NONE
    execution_order: List[str] = field(default_factory=list)
    errors: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# AGENT TEAM
# =============================================================================

@dataclass
class AgentTeam:
    """
    Declarative agent team with optional coordination pattern.

    Supports two modes:
    1. Declarative only (pattern=NONE): Swarm handles coordination
    2. Coordinated (pattern=PIPELINE/PARALLEL/etc.): Team handles execution

    Examples:
        # Simple declarative (backward compatible)
        team = AgentTeam.define(
            (ArchitectAgent, "Architect"),
            (DeveloperAgent, "Developer"),
        )

        # With coordination pattern
        team = AgentTeam.define(
            (Agent1, "Agent1"),
            (Agent2, "Agent2"),
            pattern=CoordinationPattern.PARALLEL,
        )
    """
    agents: Dict[str, AgentSpec] = field(default_factory=dict)
    pattern: CoordinationPattern = CoordinationPattern.NONE
    merge_strategy: MergeStrategy = MergeStrategy.COMBINE
    timeout: float = 0.0   # 0.0 → resolved in __post_init__
    max_retries: int = 0   # 0 → resolved in __post_init__

    # For hierarchical pattern
    manager_attr: Optional[str] = None

    # Initialized agent instances (set by swarm)
    _instances: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        from Jotty.core.foundation.config_defaults import DEFAULTS
        if self.timeout == 0.0:
            self.timeout = float(DEFAULTS.LLM_TIMEOUT_SECONDS)
        if self.max_retries <= 0:
            self.max_retries = DEFAULTS.MAX_RETRIES

    def add(
        self,
        agent_class: Type,
        display_name: str,
        attr_name: str = None,
        role: str = None,
        priority: int = 0
    ) -> 'AgentTeam':
        """
        Add an agent to the team.

        Args:
            agent_class: The agent class to instantiate
            display_name: Human-readable name (used for context)
            attr_name: Optional custom attribute name
            role: Optional role ("manager", "worker")
            priority: Execution priority (higher first)

        Returns:
            Self for fluent chaining
        """
        spec = AgentSpec(agent_class, display_name, attr_name, role, priority)
        self.agents[spec.attr_name] = spec
        return self

    def set_instances(self, instances: Dict[str, Any]) -> None:
        """Set initialized agent instances (called by swarm)."""
        self._instances = instances

    def get_agent_names(self) -> list:
        """Get list of display names for all agents."""
        return [spec.display_name for spec in self.agents.values()]

    def get_agents_by_role(self, role: str) -> List[AgentSpec]:
        """Get agents with specific role."""
        return [spec for spec in self.agents.values() if spec.role == role]

    def get_ordered_agents(self) -> List[tuple]:
        """Get agents ordered by priority (highest first)."""
        items = list(self.agents.items())
        items.sort(key=lambda x: x[1].priority, reverse=True)
        return items

    def __len__(self) -> int:
        return len(self.agents)

    def __iter__(self):
        return iter(self.agents.items())

    # =========================================================================
    # COORDINATION EXECUTION
    # =========================================================================

    async def execute(
        self,
        task: Any,
        context: Dict[str, Any] = None,
        **kwargs
    ) -> TeamResult:
        """
        Execute the team with the configured coordination pattern.

        Args:
            task: The task/input for the team
            context: Shared context dict
            **kwargs: Additional arguments passed to agents

        Returns:
            TeamResult with outputs from all agents
        """
        if not self._instances:
            raise RuntimeError("Team instances not set. Call set_instances() first.")

        context = context or {}

        if self.pattern == CoordinationPattern.NONE:
            # No coordination - return empty, let swarm handle
            return TeamResult(
                success=True,
                outputs={},
                pattern=self.pattern,
                metadata={"note": "No coordination pattern - swarm handles execution"}
            )

        elif self.pattern == CoordinationPattern.PIPELINE:
            return await self._execute_pipeline(task, context, **kwargs)

        elif self.pattern == CoordinationPattern.PARALLEL:
            return await self._execute_parallel(task, context, **kwargs)

        elif self.pattern == CoordinationPattern.CONSENSUS:
            return await self._execute_consensus(task, context, **kwargs)

        elif self.pattern == CoordinationPattern.HIERARCHICAL:
            return await self._execute_hierarchical(task, context, **kwargs)

        elif self.pattern == CoordinationPattern.BLACKBOARD:
            return await self._execute_blackboard(task, context, **kwargs)

        elif self.pattern == CoordinationPattern.ROUND_ROBIN:
            return await self._execute_round_robin(task, context, **kwargs)

        else:
            raise ValueError(f"Unknown coordination pattern: {self.pattern}")

    async def _execute_pipeline(
        self,
        task: Any,
        context: Dict[str, Any],
        **kwargs
    ) -> TeamResult:
        """Execute agents sequentially, passing output to next.

        When agents expose ``get_io_schema()``, output fields are auto-wired
        to the next agent's input fields by name/type match.  Falls back to
        raw output chaining when schemas are unavailable.
        """
        outputs = {}
        errors = {}
        execution_order = []
        current_input = task
        prev_schema = None  # AgentIOSchema of previous agent
        prev_output_dict = None  # Dict output for schema mapping

        for attr_name, spec in self.get_ordered_agents():
            agent = self._instances.get(attr_name)
            if not agent:
                continue

            execution_order.append(spec.display_name)

            try:
                # Schema-aware wiring: map previous output fields to this agent's inputs
                extra_kwargs = {}
                if prev_schema is not None and prev_output_dict is not None:
                    try:
                        if hasattr(agent, 'get_io_schema'):
                            cur_schema = agent.get_io_schema()
                            wired = prev_schema.map_outputs(prev_output_dict, cur_schema)
                            if wired:
                                extra_kwargs.update(wired)
                                logger.debug(
                                    f"Pipeline schema wiring: {prev_schema.agent_name} → "
                                    f"{cur_schema.agent_name}: {list(wired.keys())}"
                                )
                    except Exception as e:
                        logger.debug(f"Pipeline schema wiring skipped: {e}")

                # Pass previous output as input
                if hasattr(agent, 'execute'):
                    result = await asyncio.wait_for(
                        agent.execute(input=current_input, context=context, **extra_kwargs, **kwargs),
                        timeout=self.timeout
                    )
                    output = result.output if hasattr(result, 'output') else result
                else:
                    # Fallback for agents with different interface
                    output = current_input

                outputs[spec.display_name] = output
                current_input = output  # Chain to next

                # Track schema + output for next iteration's auto-wiring
                try:
                    if hasattr(agent, 'get_io_schema'):
                        prev_schema = agent.get_io_schema()
                        prev_output_dict = output if isinstance(output, dict) else {'output': str(output)}
                    else:
                        prev_schema = None
                        prev_output_dict = None
                except Exception:
                    prev_schema = None
                    prev_output_dict = None

            except Exception as e:
                logger.error(f"Pipeline agent {spec.display_name} failed: {e}")
                errors[spec.display_name] = str(e)
                prev_schema = None
                prev_output_dict = None
                # Continue with previous input or break based on config
                break

        return TeamResult(
            success=len(errors) == 0,
            outputs=outputs,
            merged_output=current_input,  # Final pipeline output
            pattern=self.pattern,
            execution_order=execution_order,
            errors=errors
        )

    async def _execute_parallel(
        self,
        task: Any,
        context: Dict[str, Any],
        **kwargs
    ) -> TeamResult:
        """Execute all agents in parallel, merge results."""
        outputs = {}
        errors = {}
        execution_order = []

        async def run_agent(attr_name: str, spec: AgentSpec):
            agent = self._instances.get(attr_name)
            if not agent:
                return None

            try:
                if hasattr(agent, 'execute'):
                    result = await asyncio.wait_for(
                        agent.execute(input=task, context=context, **kwargs),
                        timeout=self.timeout
                    )
                    return (spec.display_name, result.output if hasattr(result, 'output') else result)
                return None
            except Exception as e:
                logger.error(f"Parallel agent {spec.display_name} failed: {e}")
                errors[spec.display_name] = str(e)
                return None

        # Run all agents concurrently
        tasks = [run_agent(attr, spec) for attr, spec in self.agents.items()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if result and not isinstance(result, Exception):
                name, output = result
                outputs[name] = output
                execution_order.append(name)

        # Merge results based on strategy
        merged = self._merge_outputs(outputs)

        return TeamResult(
            success=len(outputs) > 0,
            outputs=outputs,
            merged_output=merged,
            pattern=self.pattern,
            execution_order=execution_order,
            errors=errors
        )

    async def _execute_consensus(
        self,
        task: Any,
        context: Dict[str, Any],
        **kwargs
    ) -> TeamResult:
        """Execute all agents, use voting for final result."""
        # First run parallel
        parallel_result = await self._execute_parallel(task, context, **kwargs)

        # Then vote on results
        votes = {}
        for name, output in parallel_result.outputs.items():
            # Convert output to voteable string
            vote_key = str(output)[:200]  # Truncate for comparison
            if vote_key not in votes:
                votes[vote_key] = {"count": 0, "output": output, "voters": []}
            votes[vote_key]["count"] += 1
            votes[vote_key]["voters"].append(name)

        # Find majority
        if votes:
            winner = max(votes.values(), key=lambda x: x["count"])
            merged = winner["output"]
        else:
            merged = None

        parallel_result.merged_output = merged
        parallel_result.metadata["votes"] = {k: v["count"] for k, v in votes.items()}
        return parallel_result

    async def _execute_hierarchical(
        self,
        task: Any,
        context: Dict[str, Any],
        **kwargs
    ) -> TeamResult:
        """Manager delegates to workers, aggregates results."""
        outputs = {}
        errors = {}
        execution_order = []

        # Get manager
        manager = None
        if self.manager_attr:
            manager = self._instances.get(self.manager_attr)
        else:
            # Find agent with "manager" role
            for attr, spec in self.agents.items():
                if spec.role == "manager":
                    manager = self._instances.get(attr)
                    break

        if not manager:
            # Fallback to parallel
            return await self._execute_parallel(task, context, **kwargs)

        # Manager creates subtasks
        try:
            if hasattr(manager, 'plan') or hasattr(manager, 'delegate'):
                method = getattr(manager, 'plan', None) or getattr(manager, 'delegate')
                subtasks = await method(task=task, context=context)
            else:
                # Simple delegation - all workers get same task
                subtasks = {spec.display_name: task
                           for attr, spec in self.agents.items()
                           if spec.role == "worker"}
        except Exception as e:
            logger.error(f"Manager planning failed: {e}")
            subtasks = {}

        # Workers execute subtasks
        workers = [(attr, spec) for attr, spec in self.agents.items()
                   if spec.role == "worker" or spec.role is None]

        for attr, spec in workers:
            agent = self._instances.get(attr)
            if not agent:
                continue

            subtask = subtasks.get(spec.display_name, task)
            execution_order.append(spec.display_name)

            try:
                if hasattr(agent, 'execute'):
                    result = await agent.execute(input=subtask, context=context, **kwargs)
                    outputs[spec.display_name] = result.output if hasattr(result, 'output') else result
            except Exception as e:
                errors[spec.display_name] = str(e)

        # Manager aggregates
        try:
            if hasattr(manager, 'aggregate'):
                merged = await manager.aggregate(outputs, context=context)
            else:
                merged = outputs
        except Exception as e:
            logger.error(f"Manager aggregation failed: {e}")
            merged = outputs

        return TeamResult(
            success=len(errors) == 0,
            outputs=outputs,
            merged_output=merged,
            pattern=self.pattern,
            execution_order=execution_order,
            errors=errors
        )

    async def _execute_blackboard(
        self,
        task: Any,
        context: Dict[str, Any],
        **kwargs
    ) -> TeamResult:
        """Agents contribute to shared blackboard until done."""
        blackboard = {"task": task, "contributions": {}, "done": False}
        outputs = {}
        errors = {}
        execution_order = []
        max_rounds = kwargs.get("max_rounds", 5)

        for round_num in range(max_rounds):
            made_contribution = False

            for attr, spec in self.get_ordered_agents():
                agent = self._instances.get(attr)
                if not agent:
                    continue

                try:
                    if hasattr(agent, 'contribute'):
                        contribution = await agent.contribute(
                            blackboard=blackboard,
                            context=context,
                            **kwargs
                        )
                        if contribution:
                            blackboard["contributions"][f"{spec.display_name}_{round_num}"] = contribution
                            outputs[spec.display_name] = contribution
                            execution_order.append(f"{spec.display_name}(round={round_num})")
                            made_contribution = True
                except Exception as e:
                    errors[spec.display_name] = str(e)

            if not made_contribution or blackboard.get("done"):
                break

        return TeamResult(
            success=len(errors) == 0,
            outputs=outputs,
            merged_output=blackboard,
            pattern=self.pattern,
            execution_order=execution_order,
            errors=errors,
            metadata={"rounds": round_num + 1}
        )

    async def _execute_round_robin(
        self,
        task: Any,
        context: Dict[str, Any],
        **kwargs
    ) -> TeamResult:
        """Agents take turns processing subtasks."""
        outputs = {}
        errors = {}
        execution_order = []

        # Get subtasks (if task is list) or create single-item list
        if isinstance(task, list):
            subtasks = task
        else:
            subtasks = [task]

        agents = list(self._instances.items())
        if not agents:
            return TeamResult(success=False, outputs={}, errors={"team": "No agents"}, pattern=self.pattern)

        for i, subtask in enumerate(subtasks):
            # Round robin selection
            attr, agent = agents[i % len(agents)]
            spec = self.agents.get(attr)
            if not spec:
                continue

            execution_order.append(f"{spec.display_name}(task={i})")

            try:
                if hasattr(agent, 'execute'):
                    result = await agent.execute(input=subtask, context=context, **kwargs)
                    outputs[f"{spec.display_name}_{i}"] = result.output if hasattr(result, 'output') else result
            except Exception as e:
                errors[f"{spec.display_name}_{i}"] = str(e)

        return TeamResult(
            success=len(errors) == 0,
            outputs=outputs,
            merged_output=list(outputs.values()),
            pattern=self.pattern,
            execution_order=execution_order,
            errors=errors
        )

    def _merge_outputs(self, outputs: Dict[str, Any]) -> Any:
        """Merge outputs based on strategy."""
        if not outputs:
            return None

        values = list(outputs.values())

        if self.merge_strategy == MergeStrategy.COMBINE:
            return outputs  # Return dict as-is

        elif self.merge_strategy == MergeStrategy.FIRST:
            return values[0] if values else None

        elif self.merge_strategy == MergeStrategy.CONCAT:
            # Concatenate string outputs
            return "\n\n".join(str(v) for v in values)

        elif self.merge_strategy == MergeStrategy.VOTE:
            # Simple majority voting
            from collections import Counter
            str_values = [str(v)[:200] for v in values]
            most_common = Counter(str_values).most_common(1)
            if most_common:
                winner_str = most_common[0][0]
                # Return original value that matches
                for v in values:
                    if str(v)[:200] == winner_str:
                        return v
            return values[0]

        elif self.merge_strategy == MergeStrategy.BEST:
            # Would need scoring function - default to first
            return values[0] if values else None

        return outputs

    # =========================================================================
    # CLASS METHODS
    # =========================================================================

    @classmethod
    def define(
        cls,
        *specs: tuple,
        pattern: CoordinationPattern = CoordinationPattern.NONE,
        merge_strategy: MergeStrategy = MergeStrategy.COMBINE,
        timeout: float = 0.0,
        manager_attr: str = None
    ) -> 'AgentTeam':
        """
        Create a team from tuples with optional coordination pattern.

        Each tuple can be:
        - (AgentClass, "DisplayName")
        - (AgentClass, "DisplayName", "_custom_attr")
        - (AgentClass, "DisplayName", "_attr", "role")
        - (AgentClass, "DisplayName", "_attr", "role", priority)

        Example:
            # Simple team (swarm coordinates)
            team = AgentTeam.define(
                (ArchitectAgent, "Architect"),
                (DeveloperAgent, "Developer"),
            )

            # Pipeline team
            team = AgentTeam.define(
                (ArchitectAgent, "Architect", None, None, 3),  # First
                (DeveloperAgent, "Developer", None, None, 2),
                (TesterAgent, "Tester", None, None, 1),  # Last
                pattern=CoordinationPattern.PIPELINE,
            )

            # Parallel team with merge
            team = AgentTeam.define(
                (Reviewer1, "Security"),
                (Reviewer2, "Performance"),
                (Reviewer3, "Style"),
                pattern=CoordinationPattern.PARALLEL,
                merge_strategy=MergeStrategy.CONCAT,
            )
        """
        team = cls(
            pattern=pattern,
            merge_strategy=merge_strategy,
            timeout=timeout,
            manager_attr=manager_attr
        )

        for spec in specs:
            if len(spec) == 2:
                team.add(spec[0], spec[1])
            elif len(spec) == 3:
                team.add(spec[0], spec[1], spec[2])
            elif len(spec) == 4:
                team.add(spec[0], spec[1], spec[2], spec[3])
            elif len(spec) == 5:
                team.add(spec[0], spec[1], spec[2], spec[3], spec[4])
            else:
                raise ValueError(f"Invalid spec tuple: {spec}")

        return team


__all__ = [
    'AgentTeam',
    'AgentSpec',
    'TeamResult',
    'CoordinationPattern',
    'MergeStrategy',
]
