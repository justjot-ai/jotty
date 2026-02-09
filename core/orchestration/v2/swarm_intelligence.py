"""
World-Class Swarm Intelligence Module
=====================================

Implements advanced swarm intelligence patterns:

1. EMERGENT SPECIALIZATION: Agents naturally specialize based on performance
2. SWARM CONSENSUS: Agents vote on decisions for better outcomes
3. ONLINE ADAPTATION: Learn during execution, not just after
4. COLLECTIVE MEMORY: Shared experiences across all agents
5. DYNAMIC ROUTING: Route tasks to best-fit agents automatically
6. SESSION ISOLATION: Per-context isolated agent sessions (moltbot pattern)
7. AGENT-TO-AGENT MESSAGING: Direct inter-agent communication tools
8. SELF-CURRICULUM: DrZero-inspired self-generated training tasks
9. MORPHAGENT SCORES: RCS/RDS/TRAS for profile optimization (NEW)

Inspired by: biological swarms, moltbot architecture, multi-agent RL, DrZero, MorphAgent

Architecture: Classes are extracted into sub-modules for maintainability.
All classes are re-exported here for backward compatibility.
"""

import asyncio
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
from .byzantine_verification import ByzantineVerifier
from .curriculum_generator import SyntheticTask, CurriculumGenerator
from .tool_management import ToolManager

# Protocol mixins (extracted from this file for modularity)
from .protocols import CoordinationMixin, RoutingMixin, ResilienceMixin, LifecycleMixin
# Feature mixins (extracted for maintainability)
from ._consensus_mixin import ConsensusMixin
from ._session_mixin import SessionMixin
from ._morph_mixin import MorphMixin


# =============================================================================
# SWARM INTELLIGENCE ENGINE
# =============================================================================

class SwarmIntelligence(
    CoordinationMixin, RoutingMixin, ResilienceMixin, LifecycleMixin,
    ConsensusMixin, SessionMixin, MorphMixin
):
    """
    World-class swarm intelligence coordinator.

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

        # Agent profiles (emergent specialization)
        self.agent_profiles: Dict[str, AgentProfile] = {}

        # Session management (moltbot pattern)
        self.sessions: Dict[str, AgentSession] = {}

        # Collective memory (shared across swarm, bounded to prevent leak)
        self.collective_memory: deque = deque(maxlen=1000)
        self.memory_embeddings: Dict[str, Any] = {}

        # Online adaptation buffer
        self.adaptation_buffer: List[Dict] = []
        self.adaptation_interval = 5  # Adapt every N experiences

        # Consensus history (bounded)
        self.consensus_history: deque = deque(maxlen=200)

        # Stigmergy layer (indirect coordination via shared artifacts)
        self.stigmergy = StigmergyLayer()

        # Swarm benchmarks (performance tracking)
        self.benchmarks = SwarmBenchmarks()

        # Byzantine fault tolerance (verify agent claims)
        self.byzantine = ByzantineVerifier(self)

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

        # =================================================================
        # ARXIV SWARM ENHANCEMENTS
        # =================================================================

        # Handoff management (SwarmAgentic pattern)
        self.pending_handoffs: Dict[str, HandoffContext] = {}
        self.handoff_history: deque = deque(maxlen=200)

        # Coalition management (SwarmAgentic pattern)
        self.coalitions: Dict[str, Coalition] = {}
        self.agent_coalitions: Dict[str, str] = {}  # agent -> coalition_id

        # Auction management (SwarmSys contract-net)
        self.active_auctions: Dict[str, List[AuctionBid]] = {}  # task_id -> bids

        # Gossip protocol (SwarmSys O(log n) dissemination)
        self.gossip_inbox: Dict[str, List[GossipMessage]] = {}  # agent -> messages
        self.gossip_seen: Dict[str, bool] = {}  # message_id -> seen globally

        # Hierarchical supervisor tree (SwarmSys O(log n) coordination)
        self.supervisor_tree: Dict[str, SupervisorNode] = {}
        self._tree_built = False

        logger.info("SwarmIntelligence initialized (DrZero + MorphAgent + arXiv Swarm patterns)")

    def enable_training_mode(self, enabled: bool = True, memory_system=None):
        """
        Enable/disable curriculum-based training mode.

        Agent0 insight: Training mode generates tasks that target agent weaknesses.

        Args:
            enabled: Whether training mode is active
            memory_system: Optional HierarchicalMemory for context-aware tasks
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

    def register_agent(self, agent_name: str):
        """Register an agent for tracking."""
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
        """Record task result for specialization learning."""
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

        # deque auto-bounds at maxlen=1000

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
        if use_morph_scoring and task_description and self.morph_scorer:
            best = self.morph_scorer.get_best_agent_by_tras(
                profiles=profiles,
                task=task_description,
                task_type=task_type,
                min_rcs=0.3  # Require minimum role clarity
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
                if available_signals[best_from_stigmergy] > 0.5:  # Strong signal
                    logger.debug(f"Stigmergy routing: {task_type} -> {best_from_stigmergy}")
                    return best_from_stigmergy

        # Strategy 3: Fallback to traditional scoring
        best_agent = None
        best_score = -1.0

        for agent_name in available_agents:
            profile = self.agent_profiles[agent_name]

            # Base: success rate for this task type
            success_rate = profile.get_success_rate(task_type)

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

            # Combined score
            score = (
                success_rate * 0.4 +
                trust_weight * 0.25 +
                spec_bonus * 0.15 +
                rcs_bonus * 0.2
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

    def _perform_online_adaptation(self):
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

        # Check for struggling agents
        for agent_name, results in recent_by_agent.items():
            recent_rate = sum(results) / len(results)
            profile = self.agent_profiles.get(agent_name)

            if profile and recent_rate < 0.3 and len(results) >= 3:
                # Agent is struggling - trigger adaptation
                logger.info(f"Online adaptation: {agent_name} struggling ({recent_rate:.0%}), may need different task types")
                profile.trust_score = max(0.1, profile.trust_score - 0.1)
            elif profile and recent_rate > 0.8 and len(results) >= 3:
                # Agent is excelling - boost trust
                profile.trust_score = min(1.0, profile.trust_score + 0.05)

        # Clear buffer
        self.adaptation_buffer = []

    # =========================================================================
    # SESSION MANAGEMENT (moltbot pattern)
    # =========================================================================

    # create_session, get_session, session_send, session_history, sessions_list — see _session_mixin.py

    # =========================================================================
    # STIGMERGY INTEGRATION
    # =========================================================================

    def deposit_success_signal(self, agent: str, task_type: str, execution_time: float = 0.0):
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

    def deposit_warning_signal(self, agent: str, task_type: str, warning: str):
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
            for mem in self.collective_memory[-50:]:  # Recent memories
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
                lines.append(f"  - ⚠️ {warning}")

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
    # PERSISTENCE
    # =========================================================================

    def save(self, path: str):
        """Save swarm intelligence state."""
        import json

        limit = self.collective_memory_limit
        if len(self.collective_memory) > limit:
            logger.info(f"Truncating collective_memory: {len(self.collective_memory)} → {limit} items")

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
            'collective_memory': list(self.collective_memory)[-limit:],  # Keep recent
            'stigmergy': self.stigmergy.to_dict(),  # Persist stigmergy state
            'benchmarks': self.benchmarks.to_dict(),  # Persist benchmark data
            'curriculum': self.curriculum_generator.to_dict(),  # DrZero curriculum state
            'morph_score_history': list(self.morph_score_history)[-50:],  # MorphAgent score history
            'tool_manager': self.tool_manager.to_dict(),  # Agent0 tool management state
            # arXiv swarm enhancements
            'handoff_history': [
                {'task_id': h.task_id, 'from': h.from_agent, 'to': h.to_agent,
                 'task_type': h.task_type, 'progress': h.progress, 'chain': h.handoff_chain}
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

            self.collective_memory = deque(data.get('collective_memory', []), maxlen=1000)

            # Load stigmergy state
            if 'stigmergy' in data:
                self.stigmergy = StigmergyLayer.from_dict(data['stigmergy'])

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
