"""
Swarm Intelligence Module
==========================

Multi-agent coordination with learning and specialization.

Capabilities:
1. Emergent specialization: agents specialize based on performance history
2. Consensus voting: trust-weighted decisions across agents
3. Online adaptation: learn during execution, not just after
4. Collective memory: shared experience buffer across agents
5. Dynamic routing: route tasks to best-fit agents via stigmergy + trust
6. Session isolation: per-context isolated agent sessions
7. Self-curriculum: generate training tasks targeting agent weaknesses
8. MorphAgent scoring: RCS/RDS/TRAS alignment metrics

Architecture: Sub-modules are extracted for maintainability.
All classes are re-exported here for backward compatibility.
"""

import asyncio
import threading
import time
import logging
import hashlib
import math
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False

logger = logging.getLogger(__name__)

# =============================================================================
# RE-EXPORTS FROM EXTRACTED MODULES
# =============================================================================

from .swarm_data_structures import (
    AgentSpecialization,
    AgentProfile,
    ConsensusVote,
    SwarmDecision,
    AgentSession,
    # arXiv swarm enhancements
    HandoffContext,
    Coalition,
    AuctionBid,
    GossipMessage,
    SupervisorNode,
)

from .morph_scoring import MorphScores, MorphScorer
try:
    from .morph_scoring import TaskAgentAlignmentSignature
except ImportError:
    pass  # DSPy not available

from .stigmergy import StigmergySignal, StigmergyLayer
from .benchmarking import SwarmMetrics, SwarmBenchmarks
from .byzantine_verification import ByzantineVerifier, ConsistencyChecker
from .curriculum_generator import SyntheticTask, CurriculumGenerator
from .tool_management import ToolManager
from .metrics_collector import MetricsCollector

# Protocol modules (now composed, not inherited)
from .protocols import CoordinationMixin, RoutingMixin, ResilienceMixin, LifecycleMixin
from ._consensus_mixin import ConsensusMixin
from ._session_mixin import SessionMixin
from ._morph_mixin import MorphMixin


# =============================================================================
# SWARM INTELLIGENCE ENGINE
# =============================================================================

class SwarmIntelligence:
    """
    Swarm intelligence coordinator using composition.

    Each concern is a delegate object whose methods are accessible
    via __getattr__ forwarding for backward compatibility.

    Delegates:
    - _coordination: handoff, auction, coalition, gossip, supervisor hierarchy
    - _routing: task routing, circuit breakers
    - _resilience: failure recovery, backpressure
    - _lifecycle: agent lifecycle management
    - _consensus: swarm consensus voting
    - _session: session management
    - _morph: MorphAgent scoring

    Features:
    - Emergent specialization
    - Swarm consensus
    - Online adaptation
    - Dynamic task routing
    - Session isolation
    - Agent-to-agent messaging
    """

    DEFAULT_COLLECTIVE_MEMORY_LIMIT = 200

    def __init__(self, config=None, collective_memory_limit: int = None):
        self.config = config
        self.collective_memory_limit = (
            collective_memory_limit
            if collective_memory_limit is not None
            else self.DEFAULT_COLLECTIVE_MEMORY_LIMIT
        )

        # Thread-safe lock for shared mutable state (agent_profiles,
        # collective_memory, adaptation_buffer). Prevents races when
        # multiple agents record results concurrently in multi-agent mode.
        self._state_lock = threading.Lock()

        # Agent profiles (emergent specialization)
        self.agent_profiles: Dict[str, AgentProfile] = {}

        # Session management (moltbot pattern)
        self.sessions: Dict[str, AgentSession] = {}

        # Collective memory (shared across swarm, bounded to prevent leak)
        # Uses collective_memory_limit as maxlen so runtime and persistence
        # are consistent — no silent data loss on save/load cycles.
        self.collective_memory: deque = deque(maxlen=self.collective_memory_limit)
        self.memory_embeddings: Dict[str, Any] = {}

        # Online adaptation buffer
        self.adaptation_buffer: List[Dict] = []
        self.adaptation_interval = getattr(config, 'adaptation_interval', 5)

        # Consensus history (bounded)
        self.consensus_history: deque = deque(maxlen=200)

        # Stigmergy layer (indirect coordination via shared artifacts)
        self.stigmergy = StigmergyLayer()

        # Swarm benchmarks (performance tracking)
        self.benchmarks = SwarmBenchmarks()

        # Byzantine fault tolerance (verify agent claims)
        self.byzantine = ByzantineVerifier(self)

        # Multi-agent consistency checker (repurposed Byzantine for parallel teams)
        self.consistency_checker = ConsistencyChecker(self.byzantine)

        # Observability: process-wide metrics collector
        self.metrics = MetricsCollector.get_global()

        # DrZero-inspired curriculum generator (self-generated training tasks)
        self.curriculum_generator = CurriculumGenerator(config)

        # MorphAgent-inspired scorer (RCS/RDS/TRAS)
        self.morph_scorer = MorphScorer(config)

        # Agent0: Dynamic tool management
        self.tool_manager = ToolManager()

        # Track swarm-level MorphAgent scores over time (bounded)
        self.morph_score_history: deque = deque(maxlen=100)

        # Training mode configuration (Agent0 inspired)
        self._training_mode = False
        self._memory_system = None

        # RL loop: reference to TD-Lambda learner for RL-informed routing
        # Set via connect_td_learner() from the swarm's learning mixin
        self._td_learner = None

        # =================================================================
        # COORDINATION PROTOCOLS
        # =================================================================
        # Wired:   handoff (relay paradigm), coalition (fanout paradigm),
        #          supervisor_tree (build_supervisor_tree called externally)
        # =================================================================

        # Handoff management (SwarmAgentic pattern)
        # WIRED: Used by _paradigm_relay in Orchestrator._execute_multi_agent
        self.pending_handoffs: Dict[str, HandoffContext] = {}
        self.handoff_history: deque = deque(maxlen=200)

        # Coalition management (SwarmAgentic pattern)
        # WIRED: Used by fanout paradigm in Orchestrator._execute_multi_agent
        self.coalitions: Dict[str, Coalition] = {}
        self.agent_coalitions: Dict[str, str] = {}  # agent -> coalition_id

        # Hierarchical supervisor tree (SwarmSys O(log n) coordination)
        self.supervisor_tree: Dict[str, SupervisorNode] = {}
        self._tree_built = False

        # ================================================================
        # PROTOCOL METHODS (via composition)
        #
        # Methods from these mixin classes are accessible on SwarmIntelligence
        # via __getattr__ forwarding. All methods receive `self` (the SI
        # instance) as their first argument, so they access SI's shared state
        # directly. This is effectively multiple inheritance with explicit
        # dispatch — chosen to keep the class declaration flat.
        #
        # Wired into execution:
        #   CoordinationMixin: initiate_handoff, form_coalition
        #   RoutingMixin:      smart_route, get_best_agent_for_task
        #   ConsensusMixin:    swarm_consensus
        #   SessionMixin:      get_or_create_session
        #   MorphMixin:        compute_morph_scores, get_morph_report
        #
        # Infrastructure only (not called from Orchestrator.run):
        #   ResilienceMixin:   handle_agent_failure, apply_backpressure
        #   LifecycleMixin:    agent_lifecycle management
        # ================================================================
        self._delegate_classes = (
            CoordinationMixin, RoutingMixin, ResilienceMixin,
            LifecycleMixin, ConsensusMixin, SessionMixin, MorphMixin,
        )
        # Build a flat method lookup table (name → unbound method)
        # for O(1) dispatch instead of O(n) delegate iteration.
        self._method_table: Dict[str, Any] = {}
        for MixinClass in self._delegate_classes:
            for attr_name in dir(MixinClass):
                if attr_name.startswith('_'):
                    continue
                method = getattr(MixinClass, attr_name, None)
                if callable(method) and attr_name not in self._method_table:
                    self._method_table[attr_name] = method

        logger.info("SwarmIntelligence initialized (DrZero + MorphAgent + arXiv Swarm patterns)")

    def __getattr__(self, name: str):
        """Forward attribute lookups to protocol mixin methods.

        Methods are bound to self (SwarmIntelligence instance) so they
        access shared state directly.
        """
        # Avoid infinite recursion
        if name in ('_method_table', '_delegate_classes'):
            raise AttributeError(name)

        table = self.__dict__.get('_method_table')
        if table and name in table:
            import types
            return types.MethodType(table[name], self)

        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def connect_td_learner(self, td_learner) -> None:
        """
        Connect a TD-Lambda learner for RL-informed routing.

        This closes the RL loop: learned values from TD-Lambda now influence
        agent selection in smart_route() and get_best_agent_for_task().

        Args:
            td_learner: TDLambdaLearner instance with grouped_baseline
        """
        self._td_learner = td_learner
        logger.debug("TD-Lambda learner connected for RL-informed routing")

    def enable_training_mode(self, enabled: bool = True, memory_system=None) -> None:
        """
        Enable/disable curriculum-based training mode.

        Agent0 insight: Training mode generates tasks that target agent weaknesses.

        Args:
            enabled: Whether training mode is active
            memory_system: Optional SwarmMemory for context-aware tasks
        """
        self._training_mode = enabled

        if memory_system:
            self._memory_system = memory_system
            self.curriculum_generator.connect_memory(memory_system)

        logger.info(f"Training mode {'enabled' if enabled else 'disabled'}")

    def get_training_task(self, target_agent: str = None, tool_aware: bool = True) -> Optional[SyntheticTask]:
        """
        Get a curriculum-generated training task.

        Agent0: Uses tool-aware generation when tool_aware=True.

        Args:
            target_agent: Optionally target specific agent's weaknesses
            tool_aware: Use tool-aware task generation (Agent0 style)

        Returns:
            SyntheticTask or None if training mode disabled
        """
        if not self._training_mode:
            return None

        if tool_aware:
            return self.curriculum_generator.generate_tool_aware_task(
                profiles=self.agent_profiles,
                target_agent=target_agent,
                prefer_weak_tools=True
            )
        else:
            return self.curriculum_generator.generate_training_task(
                profiles=self.agent_profiles,
                target_agent=target_agent
            )

    def receive_executor_feedback(
        self,
        task_id: str,
        success: bool,
        tools_used: List[str],
        execution_time: float = 0.0,
        error_type: str = None,
        task_type: str = None
    ):
        """
        Receive feedback from executor after task completion.

        Agent0 closed-loop: Executor feedback → Curriculum adaptation.

        Args:
            task_id: Task identifier
            success: Whether task succeeded
            tools_used: List of tools used during execution
            execution_time: Time taken to execute
            error_type: Type of error if failed
            task_type: Type of task (for curriculum update)
        """
        # Forward to curriculum generator
        self.curriculum_generator.receive_executor_feedback(
            task_id=task_id,
            success=success,
            tools_used=tools_used,
            execution_time=execution_time,
            error_type=error_type
        )

        # Update curriculum difficulty if this was a synthetic task
        if task_type:
            task = SyntheticTask(
                task_id=task_id,
                task_type=task_type,
                description="",
                difficulty=0.5,
                target_agent=None
            )
            self.curriculum_generator.update_from_result(task, success, execution_time)

    # =========================================================================
    # EMERGENT SPECIALIZATION
    # =========================================================================

    def register_agent(self, agent_name: str) -> None:
        """Register an agent for tracking. Thread-safe (lock acquired by callers or here)."""
        # Avoid nested lock acquisition: check-then-set is safe because
        # _state_lock is already held by callers like record_task_result.
        # For standalone calls, the worst case is a benign double-init.
        if agent_name not in self.agent_profiles:
            self.agent_profiles[agent_name] = AgentProfile(agent_name=agent_name)

    def record_task_result(
        self,
        agent_name: str,
        task_type: str,
        success: bool,
        execution_time: float,
        context: Dict = None,
        is_multi_agent: bool = False,
        agents_count: int = 1
    ):
        """Record task result for specialization learning. Thread-safe."""
        with self._state_lock:
            self.register_agent(agent_name)
            self.agent_profiles[agent_name].update_task_result(task_type, success, execution_time)

            # Add to collective memory
            self.collective_memory.append({
                'agent': agent_name,
                'task_type': task_type,
                'success': success,
                'execution_time': execution_time,
                'context': context or {},
                'timestamp': time.time()
            })

            # Online adaptation
            self.adaptation_buffer.append({
                'agent': agent_name,
                'task_type': task_type,
                'success': success
            })
            if len(self.adaptation_buffer) >= self.adaptation_interval:
                self._perform_online_adaptation()

        # Stigmergy: deposit success/warning signals
        if success:
            self.deposit_success_signal(agent_name, task_type, execution_time)
        else:
            self.deposit_warning_signal(agent_name, task_type, "Task failed")

        # Benchmarks: record run
        if is_multi_agent:
            self.benchmarks.record_multi_agent_run(task_type, execution_time, agents_count, success)
        else:
            self.benchmarks.record_single_agent_run(task_type, execution_time, success)

        # Observability: record to metrics collector
        swarm_name = context.get('swarm', 'unknown') if context else 'unknown'
        self.metrics.record_task(
            swarm=swarm_name,
            agent=agent_name,
            task_type=task_type,
            success=success,
            duration=execution_time,
        )

    def get_agent_specialization(self, agent_name: str) -> AgentSpecialization:
        """Get current specialization of an agent."""
        if agent_name in self.agent_profiles:
            return self.agent_profiles[agent_name].specialization
        return AgentSpecialization.GENERALIST

    def get_specialization_summary(self) -> Dict[str, str]:
        """Get summary of all agent specializations."""
        return {
            name: profile.specialization.value
            for name, profile in self.agent_profiles.items()
        }

    # =========================================================================
    # DYNAMIC TASK ROUTING
    # =========================================================================

    def get_best_agent_for_task(
        self,
        task_type: str,
        available_agents: List[str],
        task_description: str = None,
        use_morph_scoring: bool = True
    ) -> Optional[str]:
        """
        Route task to best-fit agent based on learned performance.

        Enhanced with MorphAgent TRAS scoring for better task-agent alignment.

        Uses:
        - MorphAgent TRAS (Task-Role Alignment Score) - NEW
        - MorphAgent RCS (Role Clarity Score) as filter - NEW
        - Historical success rate
        - Specialization match
        - Trust score
        - Stigmergy routing signals
        """
        if not available_agents:
            return None

        # Ensure all agents are registered
        for agent_name in available_agents:
            self.register_agent(agent_name)

        # Build profile dict for available agents
        profiles = {name: self.agent_profiles[name] for name in available_agents}

        # Strategy 1: Use MorphAgent TRAS scoring if enabled and task description available
        min_rcs = getattr(self.config, 'morph_min_rcs', 0.3)
        if use_morph_scoring and task_description and self.morph_scorer:
            best = self.morph_scorer.get_best_agent_by_tras(
                profiles=profiles,
                task=task_description,
                task_type=task_type,
                min_rcs=min_rcs
            )
            if best:
                logger.debug(f"MorphAgent TRAS routing: {task_type} -> {best}")
                return best

        # Strategy 2: Check stigmergy routing signals
        route_signals = self.stigmergy.get_route_signals(task_type)
        if route_signals:
            # Filter to available agents
            available_signals = {a: s for a, s in route_signals.items() if a in available_agents}
            if available_signals:
                best_from_stigmergy = max(available_signals.keys(), key=lambda a: available_signals[a])
                stigmergy_threshold = getattr(self.config, 'stigmergy_routing_threshold', 0.5)
                if available_signals[best_from_stigmergy] > stigmergy_threshold:
                    logger.debug(f"Stigmergy routing: {task_type} -> {best_from_stigmergy}")
                    return best_from_stigmergy

        # Strategy 3: Fallback to traditional scoring + RL advantage
        best_agent = None
        best_score = -1.0

        # Get RL baseline for this task type (closes the RL loop)
        rl_baseline = 0.5
        if self._td_learner:
            grouped = getattr(self._td_learner, 'grouped_baseline', None)
            if grouped and grouped.group_counts.get(task_type, 0) >= 2:
                rl_baseline = grouped.get_baseline(task_type)

        for agent_name in available_agents:
            profile = self.agent_profiles[agent_name]

            # Base: success rate for this task type
            success_rate = profile.get_success_rate(task_type)

            # RL advantage: how this agent compares to the learned baseline
            # Positive = agent outperforms expectations for this task type
            rl_advantage = (success_rate - rl_baseline) if self._td_learner else 0.0

            # Bonus for specialization match
            spec_bonus = 0.0
            expected_spec = self._task_type_to_specialization(task_type)
            if profile.specialization == expected_spec:
                spec_bonus = 0.2

            # Trust score weight
            trust_weight = profile.trust_score

            # MorphAgent RCS bonus (clear roles get preference)
            rcs_bonus = 0.0
            if self.morph_scorer:
                rcs, _ = self.morph_scorer.compute_rcs(profile)
                rcs_bonus = rcs * 0.1  # Up to 0.1 bonus for clear roles

            # Combined score with RL advantage (15% weight)
            score = (
                success_rate * 0.30 +
                trust_weight * 0.20 +
                spec_bonus * 0.15 +
                rcs_bonus * 0.15 +
                (0.5 + rl_advantage) * 0.20  # Center RL advantage around 0.5
            )

            if score > best_score:
                best_score = score
                best_agent = agent_name

        return best_agent

    def _task_type_to_specialization(self, task_type: str) -> AgentSpecialization:
        """Map task type to expected specialization."""
        mapping = {
            'aggregation': AgentSpecialization.AGGREGATOR,
            'analysis': AgentSpecialization.ANALYZER,
            'transformation': AgentSpecialization.TRANSFORMER,
            'validation': AgentSpecialization.VALIDATOR,
            'planning': AgentSpecialization.PLANNER,
            'filtering': AgentSpecialization.EXECUTOR,
            'generation': AgentSpecialization.EXECUTOR,
        }
        return mapping.get(task_type, AgentSpecialization.GENERALIST)

    # =========================================================================
    # SWARM CONSENSUS
    # =========================================================================

    # gather_consensus — see _consensus_mixin.py

    # =========================================================================
    # ONLINE ADAPTATION
    # =========================================================================

    def _perform_online_adaptation(self) -> None:
        """
        Adapt routing and specialization based on recent performance.

        Called periodically during execution, not just at end.
        """
        if not self.adaptation_buffer:
            return

        # Analyze recent performance
        recent_by_agent = defaultdict(list)
        for item in self.adaptation_buffer:
            recent_by_agent[item['agent']].append(item['success'])

        # Check for struggling agents (thresholds from config)
        struggle_threshold = getattr(self.config, 'adaptation_struggle_threshold', 0.3)
        excel_threshold = getattr(self.config, 'adaptation_excel_threshold', 0.8)
        trust_decrease = getattr(self.config, 'trust_decrease_on_struggle', 0.1)
        trust_increase = getattr(self.config, 'trust_increase_on_excel', 0.05)
        trust_min = getattr(self.config, 'trust_min', 0.1)

        for agent_name, results in recent_by_agent.items():
            recent_rate = sum(results) / len(results)
            profile = self.agent_profiles.get(agent_name)

            if profile and recent_rate < struggle_threshold and len(results) >= 3:
                # Agent is struggling - trigger adaptation
                logger.info(f"Online adaptation: {agent_name} struggling ({recent_rate:.0%}), may need different task types")
                profile.trust_score = max(trust_min, profile.trust_score - trust_decrease)
            elif profile and recent_rate > excel_threshold and len(results) >= 3:
                # Agent is excelling - boost trust
                profile.trust_score = min(1.0, profile.trust_score + trust_increase)

        # Clear buffer
        self.adaptation_buffer = []

    # =========================================================================
    # SESSION MANAGEMENT (moltbot pattern)
    # =========================================================================

    # create_session, get_session, session_send, session_history, sessions_list — see _session_mixin.py

    # =========================================================================
    # STIGMERGY INTEGRATION
    # =========================================================================

    def deposit_success_signal(self, agent: str, task_type: str, execution_time: float = 0.0) -> None:
        """
        Deposit success signal so other agents can learn from this success.

        Creates two signals:
        1. A 'success' signal for general awareness
        2. A 'route' signal for task routing recommendations
        """
        # Success signal
        self.stigmergy.deposit(
            signal_type='success',
            content={'agent': agent, 'task_type': task_type},
            agent=agent,
            strength=1.0,
            metadata={'execution_time': execution_time}
        )

        # Route signal (for task routing)
        self.stigmergy.deposit(
            signal_type='route',
            content={'agent': agent, 'task_type': task_type},
            agent=agent,
            strength=1.0
        )

        logger.debug(f"Stigmergy: Deposited success signal for {agent} on {task_type}")

    def deposit_warning_signal(self, agent: str, task_type: str, warning: str) -> None:
        """Deposit warning signal so other agents can avoid mistakes."""
        self.stigmergy.deposit(
            signal_type='warning',
            content={'agent': agent, 'task_type': task_type, 'warning': warning},
            agent=agent,
            strength=0.8
        )

    def get_stigmergy_recommendation(self, task_type: str) -> Optional[str]:
        """
        Get agent recommendation from pheromone signals.

        Returns the agent with the strongest route signal for this task type.
        """
        route_signals = self.stigmergy.get_route_signals(task_type)

        if not route_signals:
            return None

        # Return agent with highest accumulated strength
        best_agent = max(route_signals.keys(), key=lambda a: route_signals[a])
        return best_agent

    def get_warnings_for_task(self, task_type: str) -> List[str]:
        """Get warnings from stigmergy for a task type."""
        warnings = []
        for signal in self.stigmergy.sense(signal_type='warning', min_strength=0.3):
            content = signal.content
            if isinstance(content, dict) and content.get('task_type') == task_type:
                warnings.append(content.get('warning', ''))
        return [w for w in warnings if w]

    # =========================================================================
    # COLLECTIVE INTELLIGENCE
    # =========================================================================

    def get_swarm_wisdom(self, query: str, task_type: str = None) -> Dict[str, Any]:
        """
        Get collective wisdom from the swarm for a task.

        Returns:
        - Best agent recommendation
        - Similar past experiences
        - Success patterns
        - Warnings from failures
        """
        wisdom = {
            'recommended_agent': None,
            'similar_experiences': [],
            'success_patterns': [],
            'warnings': [],
            'confidence': 0.0
        }

        # Get best agent
        available = list(self.agent_profiles.keys())
        if task_type and available:
            wisdom['recommended_agent'] = self.get_best_agent_for_task(task_type, available)

        # Find similar past experiences
        if self.collective_memory:
            recent = list(self.collective_memory)[-50:]  # deque doesn't support slicing
            for mem in recent:
                if task_type and mem.get('task_type') == task_type:
                    wisdom['similar_experiences'].append({
                        'agent': mem['agent'],
                        'success': mem['success'],
                        'execution_time': mem['execution_time']
                    })

        # Extract patterns
        successes = [m for m in wisdom['similar_experiences'] if m['success']]
        failures = [m for m in wisdom['similar_experiences'] if not m['success']]

        if successes:
            wisdom['success_patterns'].append(
                f"{len(successes)} successful executions for {task_type} tasks"
            )

        if failures:
            wisdom['warnings'].append(
                f"{len(failures)} failures recorded - consider validation"
            )

        # Confidence based on data
        total = len(wisdom['similar_experiences'])
        if total > 0:
            wisdom['confidence'] = min(1.0, total / 10)  # Max confidence at 10+ examples

        return wisdom

    def format_swarm_context(self, query: str, task_type: str = None) -> str:
        """Format swarm wisdom as context for agents."""
        wisdom = self.get_swarm_wisdom(query, task_type)

        lines = ["# Swarm Intelligence Context:\n"]

        if wisdom['recommended_agent']:
            lines.append(f"## Recommended Agent: {wisdom['recommended_agent']}")

        if wisdom['success_patterns']:
            lines.append("\n## Success Patterns:")
            for pattern in wisdom['success_patterns']:
                lines.append(f"  - {pattern}")

        if wisdom['warnings']:
            lines.append("\n## Warnings:")
            for warning in wisdom['warnings']:
                lines.append(f" - {warning}")

        # Add specialization info
        specs = self.get_specialization_summary()
        if specs:
            lines.append("\n## Agent Specializations:")
            for agent, spec in specs.items():
                lines.append(f"  - {agent}: {spec}")

        return "\n".join(lines)

    # =========================================================================
    # MORPHAGENT SCORING INTEGRATION
    # =========================================================================

    # compute_morph_scores, get_swarm_health, optimize_profiles_morph, format_morph_report — see _morph_mixin.py

    def get_swarm_status(self) -> Dict[str, Any]:
        """
        Get comprehensive swarm status.

        Includes: load distribution, pending work, coalitions, failures.
        """
        agents = list(self.agent_profiles.keys())

        # Load distribution
        loads = {a: self.get_agent_load(a) for a in agents}
        avg_load = sum(loads.values()) / len(loads) if loads else 0

        # Failure rates
        failure_rates = {a: self.get_failure_rate(a) for a in agents}
        avg_failure = sum(failure_rates.values()) / len(failure_rates) if failure_rates else 0

        status = {
            "agent_count": len(agents),
            "avg_load": avg_load,
            "overloaded_agents": self.find_overloaded_agents(),
            "idle_agents": self.find_idle_agents(),
            "pending_handoffs": len(self.pending_handoffs),
            "active_coalitions": len(self.coalitions),
            "active_auctions": len(self.active_auctions),
            "avg_failure_rate": avg_failure,
            "collective_memory_size": len(self.collective_memory),
            "tree_built": self._tree_built,
            "queue_size": len(getattr(self, 'priority_queue', [])),
        }

        # Health score (0-1)
        health = 1.0
        if avg_load > 0.8:
            health -= 0.2
        if avg_failure > 0.3:
            health -= 0.3
        if len(status["overloaded_agents"]) > len(agents) * 0.3:
            health -= 0.2

        status["health_score"] = max(0, health)

        return status

    # =========================================================================
    # CONTEXT CONDENSATION (inspired by Cline's condense tool)
    # =========================================================================

    def condense_collective_memory(self, keep_recent: int = 20) -> str:
        """
        Compress old collective memory entries into statistical summaries.

        Instead of feeding hundreds of raw episode records to agents,
        old entries are aggregated into per-task-type statistics.
        Recent entries are kept verbatim for recency value.

        Returns a condensed summary string suitable for agent context.
        No LLM call — pure aggregation (KISS).
        """
        with self._state_lock:
            entries = list(self.collective_memory)

        if len(entries) <= keep_recent:
            return ""  # Nothing to condense

        old_entries = entries[:-keep_recent]
        # Aggregate stats per task_type
        stats: Dict[str, Dict] = {}
        for entry in old_entries:
            tt = entry.get('task_type', 'unknown')
            if tt not in stats:
                stats[tt] = {
                    'successes': 0, 'failures': 0,
                    'total_time': 0.0, 'agents': set(),
                }
            s = stats[tt]
            if entry.get('success'):
                s['successes'] += 1
            else:
                s['failures'] += 1
            s['total_time'] += entry.get('execution_time', 0.0)
            s['agents'].add(entry.get('agent', '?'))

        # Build condensed summary
        lines = [f"[Condensed history: {len(old_entries)} episodes]"]
        for tt, s in sorted(stats.items()):
            total = s['successes'] + s['failures']
            rate = s['successes'] / total if total else 0
            avg_t = s['total_time'] / total if total else 0
            agents_str = ', '.join(sorted(s['agents']))
            lines.append(
                f"  {tt}: {rate:.0%} success ({total} runs, "
                f"avg {avg_t:.1f}s, agents: {agents_str})"
            )
        return '\n'.join(lines)

    # =========================================================================
    # PERSISTENCE
    # =========================================================================

    def save(self, path: str) -> None:
        """Save swarm intelligence state."""
        import json

        data = {
            'agent_profiles': {
                name: {
                    'agent_name': p.agent_name,
                    'specialization': p.specialization.value,
                    'task_success': p.task_success,
                    'helped_others': p.helped_others,
                    'received_help': p.received_help,
                    'consensus_agreements': p.consensus_agreements,
                    'consensus_disagreements': p.consensus_disagreements,
                    'avg_execution_time': p.avg_execution_time,
                    'total_tasks': p.total_tasks,
                    'trust_score': p.trust_score,
                }
                for name, p in self.agent_profiles.items()
            },
            # deque is already bounded by collective_memory_limit — no extra truncation needed
            'collective_memory': list(self.collective_memory),
            'stigmergy': self.stigmergy.to_dict(),  # Persist stigmergy state
            'benchmarks': self.benchmarks.to_dict(),  # Persist benchmark data
            'curriculum': self.curriculum_generator.to_dict(),  # DrZero curriculum state
            'morph_score_history': list(self.morph_score_history)[-50:],  # MorphAgent score history
            'tool_manager': self.tool_manager.to_dict(),  # Agent0 tool management state
            # Byzantine verification state (claim history + stats)
            'byzantine': self.byzantine.to_dict(),
            # Circuit breaker state (agent -> {state, failures, last_failure})
            'circuit_breakers': getattr(self, 'circuit_breakers', {}),
            # Consensus history (bounded deque)
            'consensus_history': list(self.consensus_history)[-200:],
            # arXiv swarm enhancements
            'handoff_history': [
                {'task_id': h.task_id, 'from': h.from_agent, 'to': h.to_agent,
                 'task_type': h.task_type, 'progress': h.progress, 'chain': h.handoff_chain,
                 'status': getattr(h, 'status', 'completed'),
                 'timestamp': getattr(h, 'timestamp', None),
                 'metadata': getattr(h, 'metadata', {}),
                }
                for h in list(self.handoff_history)[-50:]
            ],
            'tree_built': self._tree_built,
            'priority_queue': getattr(self, 'priority_queue', [])[-100:],
        }

        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Saved swarm intelligence: {len(self.agent_profiles)} profiles, {len(self.stigmergy.signals)} stigmergy signals, curriculum tasks={self.curriculum_generator.total_generated}")

    def load(self, path: str) -> bool:
        """Load swarm intelligence state."""
        import json
        from pathlib import Path

        if not Path(path).exists():
            return False

        try:
            with open(path, 'r') as f:
                data = json.load(f)

            # Restore profiles (enum-safe: bad specialization values fallback to GENERALIST)
            for name, p_data in data.get('agent_profiles', {}).items():
                try:
                    spec = AgentSpecialization(p_data.get('specialization', 'generalist'))
                except (ValueError, KeyError):
                    logger.warning(f"Unknown specialization '{p_data.get('specialization')}' for '{name}', defaulting to GENERALIST")
                    spec = AgentSpecialization.GENERALIST
                try:
                    profile = AgentProfile(
                        agent_name=p_data.get('agent_name', name),
                        specialization=spec,
                        task_success={k: tuple(v) if isinstance(v, list) else v for k, v in p_data.get('task_success', {}).items()},
                        helped_others=p_data.get('helped_others', 0),
                        received_help=p_data.get('received_help', 0),
                        consensus_agreements=p_data.get('consensus_agreements', 0),
                        consensus_disagreements=p_data.get('consensus_disagreements', 0),
                        avg_execution_time=p_data.get('avg_execution_time', 0.0),
                        total_tasks=p_data.get('total_tasks', 0),
                        trust_score=p_data.get('trust_score', 0.5),
                    )
                    self.agent_profiles[name] = profile
                except Exception as prof_err:
                    logger.warning(f"Could not load profile '{name}': {prof_err}")

            self.collective_memory = deque(data.get('collective_memory', []), maxlen=self.collective_memory_limit)

            # Load stigmergy state into the EXISTING instance (don't replace
            # the object reference — SwarmLearningPipeline may share it).
            if 'stigmergy' in data:
                loaded_stig = StigmergyLayer.from_dict(data['stigmergy'])
                # Merge loaded signals into existing stigmergy (LP may have
                # already loaded its own copy; avoid clobbering the reference)
                for sig_id, sig in loaded_stig.signals.items():
                    if sig_id not in self.stigmergy.signals:
                        self.stigmergy.signals[sig_id] = sig

            # Load benchmarks
            if 'benchmarks' in data:
                self.benchmarks = SwarmBenchmarks.from_dict(data['benchmarks'])

            # Load DrZero curriculum state
            if 'curriculum' in data:
                self.curriculum_generator = CurriculumGenerator.from_dict(data['curriculum'], self.config)

            # Load MorphAgent score history
            if 'morph_score_history' in data:
                self.morph_score_history = deque(data['morph_score_history'], maxlen=100)

            # Load Agent0 tool manager state
            if 'tool_manager' in data:
                self.tool_manager = ToolManager.from_dict(data['tool_manager'])

            # Load Byzantine verification state
            if 'byzantine' in data:
                self.byzantine.restore_from_dict(data['byzantine'])

            # Load circuit breaker state
            if 'circuit_breakers' in data:
                self.circuit_breakers = data['circuit_breakers']

            # Load consensus history
            if 'consensus_history' in data:
                self.consensus_history = deque(data['consensus_history'], maxlen=200)

            # Load handoff history
            if 'handoff_history' in data:
                self.handoff_history = deque(maxlen=200)
                for h_data in data['handoff_history']:
                    try:
                        handoff = HandoffContext(
                            task_id=h_data.get('task_id', ''),
                            from_agent=h_data.get('from', ''),
                            to_agent=h_data.get('to', ''),
                            task_type=h_data.get('task_type', ''),
                            progress=h_data.get('progress', 0.0),
                            handoff_chain=h_data.get('chain', []),
                            timestamp=h_data.get('timestamp', 0.0) or 0.0,
                        )
                        self.handoff_history.append(handoff)
                    except Exception:
                        pass  # Skip malformed entries

            # Load arXiv swarm state
            self._tree_built = data.get('tree_built', False)
            if self._tree_built and self.agent_profiles:
                self.build_supervisor_tree()  # Rebuild from current profiles

            # Load priority queue
            self.priority_queue = data.get('priority_queue', [])

            logger.info(f"Loaded swarm intelligence: {len(self.agent_profiles)} profiles, {len(self.stigmergy.signals)} stigmergy signals, curriculum tasks={self.curriculum_generator.total_generated}")
            return True

        except Exception as e:
            logger.warning(f"Could not load swarm intelligence: {e}")
            return False


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'SwarmIntelligence',
    'AgentProfile',
    'AgentSpecialization',
    'ConsensusVote',
    'SwarmDecision',
    'AgentSession',
    'StigmergySignal',
    'StigmergyLayer',
    'SwarmMetrics',
    'SwarmBenchmarks',
    'ByzantineVerifier',
    # DrZero-inspired curriculum
    'CurriculumGenerator',
    'SyntheticTask',
    # MorphAgent-inspired scoring
    'MorphScorer',
    'MorphScores',
    # Agent0 tool management
    'ToolManager',
    # arXiv swarm enhancements (SwarmSys, SwarmAgentic)
    'HandoffContext',
    'Coalition',
    'AuctionBid',
    'GossipMessage',
    'SupervisorNode',
]
