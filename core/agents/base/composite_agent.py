"""
CompositeAgent - Unified Agent/Swarm Bridge
=============================================

An agent that orchestrates sub-agents with optional DSPy signature,
bridging the Agent and Swarm hierarchies via the Bridge Pattern.

CompositeAgent extends BaseAgent only (NOT BaseSwarm). It wraps swarms
via delegation, preserving all learning hooks and agent lifecycle.

Three modes of use:
    1. Wrap a DomainSwarm:     CodingSwarm().to_composite()
    2. Compose sub-agents:     CompositeAgent.compose("Name", a=agent1, b=agent2)
    3. Nest composites:        CompositeAgent.compose("Outer", inner=composite1, ...)

Author: A-Team
Date: February 2026
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type

from .base_agent import BaseAgent, AgentRuntimeConfig, AgentResult

# Canonical definitions in foundation — breaks agents → swarms circular dependency
from Jotty.core.foundation.types.execution_types import CoordinationPattern, MergeStrategy

if TYPE_CHECKING:
    from Jotty.core.swarms.base.domain_swarm import DomainSwarm
    from Jotty.core.swarms.swarm_types import SwarmResult

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class CompositeAgentConfig(AgentRuntimeConfig):
    """Configuration for CompositeAgent.

    Extends AgentRuntimeConfig with coordination_pattern and merge_strategy.
    """
    coordination_pattern: CoordinationPattern = CoordinationPattern.PIPELINE
    merge_strategy: MergeStrategy = MergeStrategy.COMBINE


# =============================================================================
# UNIFIED RESULT BRIDGE
# =============================================================================

@dataclass
class UnifiedResult:
    """Bidirectional bridge between AgentResult and SwarmResult.

    Converts freely between the two result types so CompositeAgent
    can wrap swarms (SwarmResult) while exposing the agent interface
    (AgentResult).
    """
    success: bool
    output: Any
    name: str
    execution_time: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    agent_traces: List = field(default_factory=list)
    evaluation: Any = None
    improvements: List = field(default_factory=list)

    def to_agent_result(self) -> AgentResult:
        """Convert to AgentResult."""
        return AgentResult(
            success=self.success,
            output=self.output,
            agent_name=self.name,
            execution_time=self.execution_time,
            error=self.error,
            metadata=self.metadata,
        )

    def to_swarm_result(self) -> SwarmResult:
        """Convert to SwarmResult (lazy import avoids circular dep)."""
        from Jotty.core.swarms.swarm_types import SwarmResult
        return SwarmResult(
            success=self.success,
            swarm_name=self.name,
            domain=self.metadata.get('domain', 'general'),
            output=self.output if isinstance(self.output, dict) else {'result': self.output},
            execution_time=self.execution_time,
            agent_traces=self.agent_traces,
            evaluation=self.evaluation,
            improvements=self.improvements,
            error=self.error,
            metadata=self.metadata,
        )

    @classmethod
    def from_swarm_result(cls, result: SwarmResult) -> UnifiedResult:
        """Create from a SwarmResult."""
        return cls(
            success=result.success,
            output=result.output,
            name=result.swarm_name,
            execution_time=result.execution_time,
            error=result.error,
            metadata=getattr(result, 'metadata', {}),
            agent_traces=getattr(result, 'agent_traces', []),
            evaluation=getattr(result, 'evaluation', None),
            improvements=getattr(result, 'improvements', []),
        )

    @classmethod
    def from_agent_result(cls, result: AgentResult) -> UnifiedResult:
        """Create from an AgentResult."""
        return cls(
            success=result.success,
            output=result.output,
            name=result.agent_name,
            execution_time=result.execution_time,
            error=result.error,
            metadata=result.metadata,
        )


# =============================================================================
# COMPOSITE AGENT
# =============================================================================

# Supported coordination patterns → method names
_COORDINATION_DISPATCH: Dict[CoordinationPattern, str] = {
    CoordinationPattern.PIPELINE: '_execute_pipeline',
    CoordinationPattern.PARALLEL: '_execute_parallel',
    CoordinationPattern.CONSENSUS: '_execute_consensus',
}


class CompositeAgent(BaseAgent):
    """Agent that orchestrates sub-agents with optional DSPy signature.

    Extends BaseAgent only (Bridge Pattern). Wraps DomainSwarm via
    delegation — no multiple inheritance, no diamond problems.
    """

    def __init__(
        self,
        config: CompositeAgentConfig = None,
        signature: Optional[Type] = None,
        sub_agents: Optional[Dict[str, BaseAgent]] = None,
    ):
        config = config or CompositeAgentConfig(name="CompositeAgent")
        super().__init__(config)
        self.signature = signature
        self._sub_agents: Dict[str, BaseAgent] = sub_agents or {}
        self._wrapped_swarm: Optional[DomainSwarm] = None

    # =========================================================================
    # FACTORY METHODS
    # =========================================================================

    @classmethod
    def from_swarm(cls, swarm: DomainSwarm, signature: Optional[Type] = None) -> CompositeAgent:
        """Wrap a DomainSwarm as a CompositeAgent.

        Delegates execute() to the swarm, preserving all learning hooks.
        Uses the swarm's timeout_seconds (default 300s) instead of the
        agent-level LLM timeout (120s), since swarms run multiple phases.
        """
        config = CompositeAgentConfig(name=getattr(swarm.config, 'name', swarm.__class__.__name__))
        # Swarm execution spans multiple LLM calls — use swarm timeout
        swarm_timeout = getattr(swarm.config, 'timeout_seconds', 300)
        config.timeout = float(swarm_timeout)
        agent = cls(config=config, signature=signature)
        agent._wrapped_swarm = swarm
        return agent

    @classmethod
    def compose(
        cls,
        name: str,
        coordination: CoordinationPattern = CoordinationPattern.PIPELINE,
        merge_strategy: MergeStrategy = MergeStrategy.COMBINE,
        signature: Optional[Type] = None,
        **agents: BaseAgent,
    ) -> CompositeAgent:
        """Build a CompositeAgent from named sub-agents.

        Example::

            pipeline = CompositeAgent.compose(
                "DevCycle",
                code=CodingSwarm().to_composite(),
                test=TestingSwarm().to_composite(),
                coordination=CoordinationPattern.PIPELINE,
            )
        """
        config = CompositeAgentConfig(
            name=name,
            coordination_pattern=coordination,
            merge_strategy=merge_strategy,
        )
        # Compute timeout from sub-agents: sum for pipeline, max for parallel/consensus
        agent_timeouts = [a.config.timeout for a in agents.values()]
        if agent_timeouts:
            if coordination == CoordinationPattern.PIPELINE:
                config.timeout = sum(agent_timeouts)
            else:
                config.timeout = max(agent_timeouts)
        # Don't retry entire pipelines — sub-agents handle their own retries
        config.max_retries = 1
        return cls(config=config, signature=signature, sub_agents=agents)

    # =========================================================================
    # SUB-AGENT MANAGEMENT
    # =========================================================================

    def add_agent(self, name: str, agent: BaseAgent) -> CompositeAgent:
        """Add a named sub-agent. Returns self for chaining."""
        self._sub_agents[name] = agent
        return self

    def remove_agent(self, name: str) -> CompositeAgent:
        """Remove a named sub-agent. Returns self for chaining."""
        self._sub_agents.pop(name, None)
        return self

    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Get a sub-agent by name."""
        return self._sub_agents.get(name)

    @property
    def sub_agents(self) -> Dict[str, BaseAgent]:
        """Read-only snapshot of sub-agents."""
        return dict(self._sub_agents)

    # =========================================================================
    # I/O SCHEMA
    # =========================================================================

    def get_io_schema(self):
        """Get I/O schema from signature or wrapped swarm.

        Priority: explicit signature > wrapped swarm's schema > None.
        Result is cached on first call.
        """
        if hasattr(self, '_io_schema') and self._io_schema is not None:
            return self._io_schema
        from Jotty.core.agents._execution_types import AgentIOSchema
        if self.signature is not None:
            self._io_schema = AgentIOSchema.from_dspy_signature(
                self.config.name, self.signature
            )
        elif self._wrapped_swarm and hasattr(self._wrapped_swarm, 'get_io_schema'):
            self._io_schema = self._wrapped_swarm.get_io_schema()
        else:
            self._io_schema = None
        return self._io_schema

    # =========================================================================
    # EXECUTION
    # =========================================================================

    async def _execute_impl(self, **kwargs) -> Any:
        """Delegate to wrapped swarm or orchestrate sub-agents."""
        if self._wrapped_swarm:
            logger.info("Delegating to wrapped swarm: %s", self._wrapped_swarm.__class__.__name__)
            result = await self._wrapped_swarm.execute(**kwargs)
            return UnifiedResult.from_swarm_result(result)
        return await self._orchestrate(**kwargs)

    async def _orchestrate(self, **kwargs) -> UnifiedResult:
        """Route to coordination-specific execution method."""
        if not self._sub_agents:
            return UnifiedResult(
                success=False, output=None, name=self.config.name,
                execution_time=0.0, error="No sub-agents configured",
            )

        pattern = self.config.coordination_pattern
        method_name = _COORDINATION_DISPATCH.get(pattern)
        if method_name is None:
            logger.warning("Unsupported coordination pattern %s, falling back to pipeline", pattern.value)
            method_name = '_execute_pipeline'

        logger.info(
            "Orchestrating %d agents (%s): %s",
            len(self._sub_agents), pattern.value, list(self._sub_agents.keys()),
        )
        return await getattr(self, method_name)(**kwargs)

    # =========================================================================
    # COORDINATION PATTERNS
    # =========================================================================

    async def _execute_pipeline(self, **kwargs) -> UnifiedResult:
        """Execute sub-agents sequentially, chaining output forward.

        Dict outputs merge as keyword args. Non-dict outputs replace
        the ``task`` key so downstream agents see the chained result.
        """
        start_time = time.time()
        current_output = None
        all_metadata: Dict[str, Any] = {}

        # Log wiring map between pipeline stages for debuggability
        agents_list = list(self._sub_agents.items())
        for i in range(len(agents_list) - 1):
            src_name, src_agent = agents_list[i]
            tgt_name, tgt_agent = agents_list[i + 1]
            src_schema = src_agent.get_io_schema() if hasattr(src_agent, 'get_io_schema') else None
            tgt_schema = tgt_agent.get_io_schema() if hasattr(tgt_agent, 'get_io_schema') else None
            if src_schema and tgt_schema:
                wiring = src_schema.wire_to(tgt_schema)
                logger.info("Pipeline wiring %s -> %s: %s", src_name, tgt_name, wiring)

        for name, agent in self._sub_agents.items():
            agent_kwargs = dict(kwargs)
            if current_output is not None:
                if isinstance(current_output, dict):
                    agent_kwargs.update(current_output)
                else:
                    agent_kwargs['task'] = current_output
                    agent_kwargs['previous_output'] = current_output

            logger.debug("Pipeline stage '%s' starting", name)
            result = await agent.execute(**agent_kwargs)

            if not result.success:
                logger.warning("Pipeline failed at '%s': %s", name, result.error)
                return UnifiedResult(
                    success=False, output=result.output, name=self.config.name,
                    execution_time=time.time() - start_time,
                    error=f"Pipeline failed at '{name}': {result.error}",
                    metadata={'failed_agent': name},
                )

            current_output, all_metadata[name] = self._extract_output(result)

        return UnifiedResult(
            success=True, output=current_output, name=self.config.name,
            execution_time=time.time() - start_time,
            metadata={'pipeline_stages': list(self._sub_agents.keys()), **all_metadata},
        )

    async def _execute_parallel(self, **kwargs) -> UnifiedResult:
        """Execute all sub-agents concurrently, merge results."""
        start_time = time.time()
        results, errors = await self._gather_results(**kwargs)

        outputs = {
            name: self._extract_output(r)[0]
            for name, r in results.items()
        }
        merged = self._merge_outputs(outputs)

        if errors:
            logger.warning("Parallel errors: %s", errors)

        return UnifiedResult(
            success=bool(outputs), output=merged, name=self.config.name,
            execution_time=time.time() - start_time,
            error="; ".join(f"{k}: {v}" for k, v in errors.items()) if errors else None,
            metadata={'parallel_agents': list(self._sub_agents.keys()), 'errors': errors},
        )

    async def _execute_consensus(self, **kwargs) -> UnifiedResult:
        """Execute all sub-agents, majority vote on success."""
        start_time = time.time()
        results, errors = await self._gather_results(**kwargs)

        successes = len(results)
        total = successes + len(errors)
        majority_success = successes > total / 2

        successful_outputs = [self._extract_output(r)[0] for r in results.values()]
        consensus_output = successful_outputs[0] if successful_outputs else None

        logger.info(
            "Consensus: %d/%d succeeded (majority=%s)",
            successes, total, "yes" if majority_success else "no",
        )

        return UnifiedResult(
            success=majority_success, output=consensus_output, name=self.config.name,
            execution_time=time.time() - start_time,
            metadata={
                'consensus_agents': list(results.keys()),
                'votes_success': successes, 'votes_total': total,
            },
        )

    # =========================================================================
    # SHARED HELPERS
    # =========================================================================

    async def _gather_results(
        self, **kwargs
    ) -> Tuple[Dict[str, AgentResult], Dict[str, str]]:
        """Run all sub-agents concurrently.

        Returns:
            (successful_results, errors) — successful_results maps agent
            name to AgentResult; errors maps agent name to error string.
        """
        agent_names = list(self._sub_agents.keys())

        async def _run(name: str, agent: BaseAgent) -> Tuple[str, AgentResult]:
            return name, await agent.execute(**kwargs)

        coros = [_run(n, a) for n, a in self._sub_agents.items()]
        completed = await asyncio.gather(*coros, return_exceptions=True)

        results: Dict[str, AgentResult] = {}
        errors: Dict[str, str] = {}

        for i, item in enumerate(completed):
            name = agent_names[i]
            if isinstance(item, Exception):
                logger.warning("Agent '%s' raised: %s", name, item)
                errors[name] = str(item)
            else:
                _, result = item
                if result.success:
                    results[name] = result
                else:
                    errors[name] = result.error or "Unknown error"

        return results, errors

    @staticmethod
    def _extract_output(result: AgentResult) -> Tuple[Any, Dict[str, Any]]:
        """Extract raw output and metadata from AgentResult.

        Unwraps nested UnifiedResult (from inner CompositeAgents).
        """
        if isinstance(result.output, UnifiedResult):
            return result.output.output, result.output.metadata
        return result.output, result.metadata

    def _merge_outputs(self, outputs: Dict[str, Any]) -> Any:
        """Merge parallel outputs per configured merge strategy."""
        if not outputs:
            return None

        strategy = self.config.merge_strategy

        if strategy == MergeStrategy.COMBINE:
            return outputs
        elif strategy == MergeStrategy.FIRST:
            return next(iter(outputs.values()))
        elif strategy == MergeStrategy.CONCAT:
            return "\n".join(str(v) for v in outputs.values())
        elif strategy == MergeStrategy.BEST:
            best = max(outputs, key=lambda k: len(str(outputs[k])))
            return outputs[best]

        return outputs

    # =========================================================================
    # EXECUTE OVERRIDE — unwrap UnifiedResult into AgentResult
    # =========================================================================

    async def execute(self, **kwargs) -> AgentResult:
        """Execute and unwrap UnifiedResult if present.

        BaseAgent.execute() wraps _execute_impl output in AgentResult.
        When _execute_impl returns UnifiedResult (swarm wrap or orchestration),
        this override converts it to a proper AgentResult preserving metadata.
        """
        result = await super().execute(**kwargs)

        if isinstance(result.output, UnifiedResult):
            unified = result.output
            return AgentResult(
                success=unified.success,
                output=unified.output,
                agent_name=self.config.name,
                execution_time=result.execution_time,
                retries=result.retries,
                error=unified.error,
                metadata={
                    **unified.metadata,
                    'agent_traces': unified.agent_traces,
                    'evaluation': unified.evaluation,
                    'improvements': unified.improvements,
                },
            )

        return result

    # =========================================================================
    # UTILITY
    # =========================================================================

    def __repr__(self) -> str:
        if self._wrapped_swarm:
            return f"CompositeAgent(wraps={self._wrapped_swarm.__class__.__name__}, name='{self.config.name}')"
        return (
            f"CompositeAgent(name='{self.config.name}', "
            f"agents={list(self._sub_agents.keys())}, "
            f"coordination={self.config.coordination_pattern.value})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize composite agent state."""
        base = super().to_dict()
        base.update({
            'type': 'composite',
            'wrapped_swarm': self._wrapped_swarm.__class__.__name__ if self._wrapped_swarm else None,
            'sub_agents': {
                name: agent.to_dict() if hasattr(agent, 'to_dict') else str(agent)
                for name, agent in self._sub_agents.items()
            },
            'coordination': self.config.coordination_pattern.value,
            'merge_strategy': self.config.merge_strategy.value,
            'has_signature': self.signature is not None,
        })
        return base


__all__ = [
    'CompositeAgent',
    'CompositeAgentConfig',
    'UnifiedResult',
    'CoordinationPattern',
    'MergeStrategy',
]
