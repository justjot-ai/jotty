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
from collections import defaultdict
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


# =============================================================================
# SWARM INTELLIGENCE ENGINE
# =============================================================================

class SwarmIntelligence:
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

        # Collective memory (shared across swarm)
        self.collective_memory: List[Dict] = []
        self.memory_embeddings: Dict[str, Any] = {}

        # Online adaptation buffer
        self.adaptation_buffer: List[Dict] = []
        self.adaptation_interval = 5  # Adapt every N experiences

        # Consensus history
        self.consensus_history: List[SwarmDecision] = []

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

        # Track swarm-level MorphAgent scores over time
        self.morph_score_history: List[Dict[str, Any]] = []

        # Training mode configuration (Agent0 inspired)
        self._training_mode = False
        self._memory_system = None

        # =================================================================
        # ARXIV SWARM ENHANCEMENTS
        # =================================================================

        # Handoff management (SwarmAgentic pattern)
        self.pending_handoffs: Dict[str, HandoffContext] = {}
        self.handoff_history: List[HandoffContext] = []

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

        # Bound collective memory
        if len(self.collective_memory) > 1000:
            self.collective_memory = self.collective_memory[-1000:]

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

    async def gather_consensus(
        self,
        question: str,
        options: List[str],
        agents: List[str],
        vote_func: Callable[[str, str, List[str]], Tuple[str, float, str]]
    ) -> SwarmDecision:
        """
        Gather consensus from multiple agents.

        Args:
            question: The question to decide
            options: Available options
            agents: Agents participating in consensus
            vote_func: Function(agent_name, question, options) -> (decision, confidence, reasoning)

        Returns:
            SwarmDecision with final consensus
        """
        votes = []

        # Gather votes (can be parallelized)
        for agent_name in agents:
            try:
                decision, confidence, reasoning = vote_func(agent_name, question, options)
                votes.append(ConsensusVote(
                    agent_name=agent_name,
                    decision=decision,
                    confidence=confidence,
                    reasoning=reasoning
                ))
            except Exception as e:
                logger.warning(f"Agent {agent_name} failed to vote: {e}")

        if not votes:
            return SwarmDecision(
                question=question,
                votes=[],
                final_decision=options[0] if options else "",
                consensus_strength=0.0,
                dissenting_views=[]
            )

        # Weighted voting based on confidence and trust
        vote_weights = defaultdict(float)
        for vote in votes:
            self.register_agent(vote.agent_name)
            trust = self.agent_profiles[vote.agent_name].trust_score
            weight = vote.confidence * trust
            vote_weights[vote.decision] += weight

        # Find winner
        final_decision = max(vote_weights.keys(), key=lambda k: vote_weights[k])
        total_weight = sum(vote_weights.values())
        consensus_strength = vote_weights[final_decision] / total_weight if total_weight > 0 else 0.0

        # Find dissenting views
        dissenting = [
            f"{v.agent_name}: {v.reasoning}"
            for v in votes
            if v.decision != final_decision
        ]

        decision = SwarmDecision(
            question=question,
            votes=votes,
            final_decision=final_decision,
            consensus_strength=consensus_strength,
            dissenting_views=dissenting
        )

        # Update consensus stats
        for vote in votes:
            profile = self.agent_profiles[vote.agent_name]
            if vote.decision == final_decision:
                profile.consensus_agreements += 1
            else:
                profile.consensus_disagreements += 1

        self.consensus_history.append(decision)

        return decision

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

    def create_session(self, agent_name: str, context: str = "main") -> str:
        """Create isolated session for an agent."""
        session_id = hashlib.md5(f"{agent_name}:{context}:{time.time()}".encode()).hexdigest()[:12]

        self.sessions[session_id] = AgentSession(
            session_id=session_id,
            agent_name=agent_name,
            context=context
        )

        return session_id

    def get_session(self, session_id: str) -> Optional[AgentSession]:
        """Get session by ID."""
        return self.sessions.get(session_id)

    def session_send(self, session_id: str, from_agent: str, content: str, metadata: Dict = None):
        """Send message to a session (moltbot sessions_send pattern)."""
        session = self.sessions.get(session_id)
        if session:
            session.add_message(from_agent, content, metadata)
            return True
        return False

    def session_history(self, session_id: str, limit: int = 20) -> List[Dict]:
        """Get session history (moltbot sessions_history pattern)."""
        session = self.sessions.get(session_id)
        if session:
            return session.messages[-limit:]
        return []

    def sessions_list(self, agent_name: str = None) -> List[Dict]:
        """List sessions (moltbot sessions_list pattern)."""
        sessions = []
        for sid, session in self.sessions.items():
            if agent_name is None or session.agent_name == agent_name:
                sessions.append({
                    'session_id': sid,
                    'agent': session.agent_name,
                    'context': session.context,
                    'message_count': len(session.messages),
                    'last_active': session.last_active
                })
        return sessions

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

    def compute_morph_scores(self, task: str = None, task_type: str = None) -> Dict[str, MorphScores]:
        """
        Compute MorphAgent scores (RCS/RDS/TRAS) for all agents.

        Args:
            task: Optional task description for TRAS computation
            task_type: Optional task type for TRAS computation

        Returns:
            Dict of agent_name -> MorphScores
        """
        if not self.morph_scorer:
            return {}

        scores = self.morph_scorer.compute_all_scores(
            profiles=self.agent_profiles,
            task=task,
            task_type=task_type
        )

        # Record in history
        self.morph_score_history.append({
            'timestamp': time.time(),
            'scores': {name: {'rcs': s.rcs, 'rds': s.rds, 'tras': s.tras} for name, s in scores.items()},
            'task_context': task[:50] if task else ''
        })

        # Keep bounded
        if len(self.morph_score_history) > 100:
            self.morph_score_history = self.morph_score_history[-100:]

        return scores

    def get_swarm_health(self) -> Dict[str, Any]:
        """
        Get overall swarm health using MorphAgent metrics.

        Returns comprehensive health assessment:
        - avg_rcs: Average Role Clarity (are roles well-defined?)
        - rds: Role Differentiation (is swarm diverse?)
        - avg_trust: Average trust score
        - specialization_coverage: How many specializations are covered
        - recommendations: Improvement suggestions
        """
        health = {
            'avg_rcs': 0.5,
            'rds': 0.5,
            'avg_trust': 0.5,
            'specialization_coverage': 0.0,
            'agent_count': len(self.agent_profiles),
            'total_tasks': sum(p.total_tasks for p in self.agent_profiles.values()),
            'recommendations': []
        }

        if not self.agent_profiles:
            health['recommendations'].append("No agents registered - add agents to swarm")
            return health

        # Compute MorphAgent scores
        if self.morph_scorer:
            # RCS for each agent
            rcs_scores = []
            for profile in self.agent_profiles.values():
                rcs, _ = self.morph_scorer.compute_rcs(profile)
                rcs_scores.append(rcs)
            health['avg_rcs'] = sum(rcs_scores) / len(rcs_scores) if rcs_scores else 0.5

            # RDS (swarm-level)
            health['rds'] = self.morph_scorer.compute_rds(self.agent_profiles)

        # Average trust
        trust_scores = [p.trust_score for p in self.agent_profiles.values()]
        health['avg_trust'] = sum(trust_scores) / len(trust_scores) if trust_scores else 0.5

        # Specialization coverage
        unique_specs = set(p.specialization for p in self.agent_profiles.values())
        health['specialization_coverage'] = len(unique_specs) / len(AgentSpecialization)

        # Generate recommendations
        if health['avg_rcs'] < 0.4:
            health['recommendations'].append(
                "Low role clarity - consider warmup training to specialize agents"
            )

        if health['rds'] < 0.4:
            health['recommendations'].append(
                "Low role differentiation - agents are too similar, consider diversifying"
            )

        if health['avg_trust'] < 0.5:
            health['recommendations'].append(
                "Low average trust - some agents have inconsistent performance"
            )

        if health['total_tasks'] < 10:
            health['recommendations'].append(
                "Limited task history - consider warmup() for self-training"
            )

        if not health['recommendations']:
            health['recommendations'].append("Swarm health is good - no issues detected")

        return health

    def optimize_profiles_morph(self, num_iterations: int = 5, threshold: float = 0.1) -> Dict[str, Any]:
        """
        MorphAgent-inspired profile optimization.

        Iteratively improves agent profiles by:
        1. Computing RCS/RDS scores
        2. Identifying low-scoring agents
        3. Generating curriculum tasks targeting weaknesses
        4. Simulating improvement through task type rebalancing

        This is used during warmup phase to optimize agent differentiation.

        Args:
            num_iterations: Max optimization iterations
            threshold: Convergence threshold for score improvement

        Returns:
            Optimization results with before/after scores
        """
        if not self.agent_profiles:
            return {'success': False, 'reason': 'No agents to optimize'}

        results = {
            'iterations': 0,
            'initial_rds': 0.0,
            'final_rds': 0.0,
            'initial_avg_rcs': 0.0,
            'final_avg_rcs': 0.0,
            'improvements': []
        }

        # Initial scores
        if self.morph_scorer:
            results['initial_rds'] = self.morph_scorer.compute_rds(self.agent_profiles)
            rcs_scores = [self.morph_scorer.compute_rcs(p)[0] for p in self.agent_profiles.values()]
            results['initial_avg_rcs'] = sum(rcs_scores) / len(rcs_scores) if rcs_scores else 0.5

        prev_score = results['initial_rds'] + results['initial_avg_rcs']

        for iteration in range(num_iterations):
            results['iterations'] = iteration + 1

            # Find agents with low RCS (unclear roles)
            low_rcs_agents = []
            for name, profile in self.agent_profiles.items():
                if self.morph_scorer:
                    rcs, components = self.morph_scorer.compute_rcs(profile)
                    if rcs < 0.5:
                        low_rcs_agents.append((name, rcs, components))

            # Generate curriculum tasks targeting low-RCS agents
            for agent_name, rcs, components in low_rcs_agents[:3]:  # Top 3 worst
                # Identify which component is lowest
                if components.get('focus', 1.0) < 0.5:
                    # Agent needs to focus - generate tasks in their best type
                    profile = self.agent_profiles[agent_name]
                    if profile.task_success:
                        best_type = max(
                            profile.task_success.keys(),
                            key=lambda t: profile.task_success[t][0] / max(1, profile.task_success[t][1])
                        )
                        results['improvements'].append(
                            f"{agent_name}: Focus training on {best_type} (RCS: {rcs:.2f})"
                        )

            # Compute new scores
            if self.morph_scorer:
                new_rds = self.morph_scorer.compute_rds(self.agent_profiles)
                new_rcs_scores = [self.morph_scorer.compute_rcs(p)[0] for p in self.agent_profiles.values()]
                new_avg_rcs = sum(new_rcs_scores) / len(new_rcs_scores) if new_rcs_scores else 0.5

                new_score = new_rds + new_avg_rcs

                # Check convergence
                if abs(new_score - prev_score) < threshold:
                    break

                prev_score = new_score
                results['final_rds'] = new_rds
                results['final_avg_rcs'] = new_avg_rcs

        return results

    def format_morph_report(self) -> str:
        """Generate human-readable MorphAgent scores report."""
        lines = [
            "# MorphAgent Scores Report",
            "=" * 50,
            ""
        ]

        if not self.agent_profiles:
            lines.append("No agents registered.")
            return "\n".join(lines)

        # Swarm-level RDS
        if self.morph_scorer:
            rds = self.morph_scorer.compute_rds(self.agent_profiles)
            lines.append(f"## Swarm Role Differentiation (RDS): {rds:.2f}")
            lines.append(f"   {'✓ Good diversity' if rds >= 0.5 else '⚠️ Agents too similar'}")
            lines.append("")

        # Per-agent RCS
        lines.append("## Per-Agent Role Clarity (RCS)")
        lines.append("-" * 40)

        for name, profile in self.agent_profiles.items():
            if self.morph_scorer:
                rcs, components = self.morph_scorer.compute_rcs(profile)
                status = "✓" if rcs >= 0.5 else "⚠️"
                lines.append(f"  {status} {name}: RCS={rcs:.2f}")
                lines.append(f"      Focus: {components.get('focus', 0):.2f}, "
                           f"Consistency: {components.get('consistency', 0):.2f}, "
                           f"Specialization: {components.get('specialization', 0):.2f}")

        # Health summary
        lines.append("")
        health = self.get_swarm_health()
        lines.append("## Health Summary")
        lines.append(f"  - Average RCS: {health['avg_rcs']:.2f}")
        lines.append(f"  - RDS: {health['rds']:.2f}")
        lines.append(f"  - Average Trust: {health['avg_trust']:.2f}")
        lines.append(f"  - Specialization Coverage: {health['specialization_coverage']:.1%}")

        if health['recommendations']:
            lines.append("")
            lines.append("## Recommendations")
            for rec in health['recommendations']:
                lines.append(f"  - {rec}")

        return "\n".join(lines)

    # =========================================================================
    # AGENT HANDOFF (SwarmAgentic Pattern)
    # =========================================================================

    def initiate_handoff(
        self,
        task_id: str,
        from_agent: str,
        to_agent: str,
        task_type: str,
        context: Dict = None,
        partial_result: Any = None,
        progress: float = 0.0,
        priority: int = 5
    ) -> HandoffContext:
        """
        Initiate task handoff between agents with context preservation.

        SwarmAgentic pattern: Seamless task transfer without losing state.

        Args:
            task_id: Unique task identifier
            from_agent: Agent initiating handoff
            to_agent: Agent receiving task
            task_type: Type of task being handed off
            context: Task context to preserve
            partial_result: Any partial work completed
            progress: Completion progress 0-1
            priority: Task priority 1-10

        Returns:
            HandoffContext for tracking
        """
        handoff = HandoffContext(
            task_id=task_id,
            from_agent=from_agent,
            to_agent=to_agent,
            task_type=task_type,
            context=context or {},
            partial_result=partial_result,
            progress=progress,
            priority=priority
        )
        handoff.add_to_chain(from_agent)

        self.pending_handoffs[task_id] = handoff

        # Notify via gossip
        self.gossip_broadcast(
            origin_agent=from_agent,
            message_type="handoff",
            content={"task_id": task_id, "to": to_agent, "type": task_type}
        )

        logger.info(f"Handoff initiated: {from_agent} → {to_agent} for task {task_id}")
        return handoff

    def accept_handoff(self, task_id: str, agent: str) -> Optional[HandoffContext]:
        """
        Accept a pending handoff.

        Returns the handoff context for the receiving agent to continue work.
        """
        handoff = self.pending_handoffs.pop(task_id, None)
        if handoff and handoff.to_agent == agent:
            handoff.add_to_chain(agent)
            self.handoff_history.append(handoff)
            logger.info(f"Handoff accepted: {agent} received task {task_id}")
            return handoff
        return None

    def reject_handoff(self, task_id: str, agent: str, reason: str = "") -> bool:
        """
        Reject a handoff and find alternative agent.

        Returns True if successfully rerouted, False if no alternative.
        """
        handoff = self.pending_handoffs.get(task_id)
        if not handoff or handoff.to_agent != agent:
            return False

        # Find alternative via auction
        available = [a for a in self.agent_profiles.keys()
                     if a != agent and a not in handoff.handoff_chain]

        if not available:
            logger.warning(f"Handoff rejected, no alternatives: {task_id}")
            return False

        # Quick auction for rerouting
        best = self.get_best_agent_for_task(handoff.task_type, available)
        if best:
            handoff.to_agent = best
            logger.info(f"Handoff rerouted: {task_id} → {best} (rejected by {agent}: {reason})")
            return True

        return False

    def get_pending_handoffs(self, agent: str) -> List[HandoffContext]:
        """Get all pending handoffs for an agent."""
        return [h for h in self.pending_handoffs.values() if h.to_agent == agent]

    # =========================================================================
    # HIERARCHICAL SUPERVISOR TREE (SwarmSys O(log n) Pattern)
    # =========================================================================

    def build_supervisor_tree(self, agents: List[str] = None, branching_factor: int = 3):
        """
        Build hierarchical supervisor tree for O(log n) coordination.

        SwarmSys pattern: Layered supervisors reduce communication complexity
        from O(n) to O(log n).

        Args:
            agents: List of agents (uses all registered if None)
            branching_factor: Children per supervisor (default 3)
        """
        agents = agents or list(self.agent_profiles.keys())
        if not agents:
            return

        self.supervisor_tree.clear()
        import math

        # Level 0: All agents as leaves
        level = 0
        current_level = []
        for i, agent in enumerate(agents):
            node_id = f"L{level}_{i}"
            node = SupervisorNode(
                node_id=node_id,
                agent_name=agent,
                level=level,
                supervised_agents=[agent]
            )
            self.supervisor_tree[node_id] = node
            current_level.append(node_id)

        # Build supervisor levels until we have a single root
        while len(current_level) > 1:
            level += 1
            next_level = []

            for i in range(0, len(current_level), branching_factor):
                children = current_level[i:i + branching_factor]
                if not children:
                    continue

                # Pick best agent as supervisor (highest trust)
                child_agents = [self.supervisor_tree[c].agent_name for c in children]
                supervisor_agent = max(
                    child_agents,
                    key=lambda a: self.agent_profiles.get(a, AgentProfile(a)).trust_score
                )

                node_id = f"L{level}_{len(next_level)}"
                supervised = []
                for c in children:
                    supervised.extend(self.supervisor_tree[c].supervised_agents)
                    self.supervisor_tree[c].parent = node_id

                node = SupervisorNode(
                    node_id=node_id,
                    agent_name=supervisor_agent,
                    level=level,
                    children=children,
                    supervised_agents=supervised
                )
                self.supervisor_tree[node_id] = node
                next_level.append(node_id)

            current_level = next_level

        self._tree_built = True
        logger.info(f"Supervisor tree built: {len(agents)} agents, {level + 1} levels, O(log {len(agents)}) = O({level + 1})")

    def get_supervisor(self, agent: str) -> Optional[str]:
        """Get the supervisor agent for a given agent."""
        for node in self.supervisor_tree.values():
            if node.agent_name == agent and node.parent:
                parent_node = self.supervisor_tree.get(node.parent)
                if parent_node:
                    return parent_node.agent_name
        return None

    def get_supervised_agents(self, supervisor: str) -> List[str]:
        """Get all agents supervised by a given supervisor."""
        for node in self.supervisor_tree.values():
            if node.agent_name == supervisor:
                return node.supervised_agents
        return []

    def route_via_hierarchy(self, task_type: str, from_agent: str = None) -> Optional[str]:
        """
        Route task through hierarchy for O(log n) routing.

        Instead of checking all agents, routes through supervisor tree.
        """
        if not self._tree_built:
            self.build_supervisor_tree()

        # Find root
        root = None
        for node in self.supervisor_tree.values():
            if node.parent is None and node.level > 0:
                root = node
                break

        if not root:
            # Fallback to flat routing
            return self.get_best_agent_for_task(task_type, list(self.agent_profiles.keys()))

        # Traverse down tree finding best path
        current = root
        while current.children:
            best_child = None
            best_score = -1

            for child_id in current.children:
                child = self.supervisor_tree.get(child_id)
                if not child:
                    continue

                # Score based on task success in subtree
                subtree_score = 0
                for agent in child.supervised_agents:
                    profile = self.agent_profiles.get(agent)
                    if profile:
                        subtree_score += profile.get_success_rate(task_type)

                if subtree_score > best_score:
                    best_score = subtree_score
                    best_child = child

            if best_child:
                current = best_child
            else:
                break

        return current.agent_name

    # =========================================================================
    # GOSSIP PROTOCOL (SwarmSys O(log n) Dissemination)
    # =========================================================================

    def gossip_broadcast(
        self,
        origin_agent: str,
        message_type: str,
        content: Dict[str, Any],
        ttl: int = 3
    ) -> str:
        """
        Broadcast message via gossip protocol.

        SwarmSys pattern: O(log n) information spread without central coordinator.

        Args:
            origin_agent: Agent originating the message
            message_type: Type of message (info, warning, route, capability)
            content: Message content
            ttl: Time-to-live in hops

        Returns:
            Message ID
        """
        msg_id = hashlib.md5(f"{origin_agent}:{message_type}:{time.time()}".encode()).hexdigest()[:12]

        message = GossipMessage(
            message_id=msg_id,
            content=content,
            origin_agent=origin_agent,
            message_type=message_type,
            ttl=ttl,
            seen_by=[origin_agent]
        )

        # Distribute to random subset of agents (gossip fanout)
        import random
        all_agents = [a for a in self.agent_profiles.keys() if a != origin_agent]
        fanout = min(3, len(all_agents))  # Gossip to 3 random agents
        targets = random.sample(all_agents, fanout) if all_agents else []

        for target in targets:
            if target not in self.gossip_inbox:
                self.gossip_inbox[target] = []
            self.gossip_inbox[target].append(message)

        self.gossip_seen[msg_id] = True
        logger.debug(f"Gossip broadcast: {message_type} from {origin_agent} to {len(targets)} agents")
        return msg_id

    def gossip_receive(self, agent: str) -> List[GossipMessage]:
        """
        Receive and process gossip messages for an agent.

        Agent processes messages and propagates if TTL > 0.
        """
        messages = self.gossip_inbox.pop(agent, [])
        to_propagate = []

        for msg in messages:
            if msg.mark_seen(agent):
                to_propagate.append(msg)

        # Propagate with reduced TTL
        for msg in to_propagate:
            import random
            other_agents = [a for a in self.agent_profiles.keys()
                          if a != agent and a not in msg.seen_by]
            if other_agents:
                target = random.choice(other_agents)
                if target not in self.gossip_inbox:
                    self.gossip_inbox[target] = []
                self.gossip_inbox[target].append(msg)

        return messages

    def gossip_query(self, query_type: str, agent: str = None) -> List[Dict]:
        """
        Query recent gossip messages by type.

        Args:
            query_type: Message type to filter
            agent: Optional agent to filter by origin
        """
        results = []
        for inbox in self.gossip_inbox.values():
            for msg in inbox:
                if msg.message_type == query_type:
                    if agent is None or msg.origin_agent == agent:
                        results.append({
                            "id": msg.message_id,
                            "content": msg.content,
                            "from": msg.origin_agent,
                            "age": time.time() - msg.created_at
                        })
        return results

    # =========================================================================
    # AUCTION-BASED TASK ALLOCATION (SwarmSys Contract-Net)
    # =========================================================================

    def start_auction(
        self,
        task_id: str,
        task_type: str,
        task_description: str = "",
        deadline_seconds: float = 5.0
    ) -> str:
        """
        Start auction for task allocation.

        SwarmSys contract-net pattern: Agents bid based on capability.

        Args:
            task_id: Unique task identifier
            task_type: Type of task
            task_description: Optional description
            deadline_seconds: Auction duration

        Returns:
            Auction task_id
        """
        self.active_auctions[task_id] = []

        # Broadcast auction announcement via gossip
        self.gossip_broadcast(
            origin_agent="auctioneer",
            message_type="auction",
            content={
                "task_id": task_id,
                "task_type": task_type,
                "description": task_description,
                "deadline": time.time() + deadline_seconds
            }
        )

        logger.info(f"Auction started: {task_id} ({task_type})")
        return task_id

    def submit_bid(
        self,
        task_id: str,
        agent_name: str,
        estimated_time: float = 10.0,
        confidence: float = 0.8,
        reasoning: str = ""
    ) -> Optional[AuctionBid]:
        """
        Submit bid for an auction.

        Args:
            task_id: Auction task_id
            agent_name: Bidding agent
            estimated_time: Estimated completion time
            confidence: Confidence level 0-1
            reasoning: Optional explanation

        Returns:
            AuctionBid if accepted
        """
        if task_id not in self.active_auctions:
            return None

        profile = self.agent_profiles.get(agent_name)
        if not profile:
            self.register_agent(agent_name)
            profile = self.agent_profiles[agent_name]

        # Calculate bid value from profile
        bid_value = profile.trust_score

        # Specialization match (check if agent specializes in this task type)
        expected_spec = self._task_type_to_specialization(task_id.split("_")[0] if "_" in task_id else "general")
        spec_match = 1.0 if profile.specialization == expected_spec else 0.5

        # Current load (based on pending handoffs)
        current_load = len([h for h in self.pending_handoffs.values() if h.to_agent == agent_name]) / 5.0
        current_load = min(1.0, current_load)

        bid = AuctionBid(
            agent_name=agent_name,
            task_id=task_id,
            bid_value=bid_value,
            estimated_time=estimated_time,
            confidence=confidence,
            specialization_match=spec_match,
            current_load=current_load,
            reasoning=reasoning
        )

        self.active_auctions[task_id].append(bid)
        logger.debug(f"Bid submitted: {agent_name} for {task_id} (score: {bid.score:.2f})")
        return bid

    def close_auction(self, task_id: str) -> Optional[str]:
        """
        Close auction and determine winner.

        Returns winning agent name or None.
        """
        bids = self.active_auctions.pop(task_id, [])
        if not bids:
            return None

        # Sort by combined score
        bids.sort(key=lambda b: b.score, reverse=True)
        winner = bids[0]

        logger.info(f"Auction closed: {task_id} → {winner.agent_name} (score: {winner.score:.2f})")
        return winner.agent_name

    def auto_auction(
        self,
        task_id: str,
        task_type: str,
        available_agents: List[str] = None
    ) -> Optional[str]:
        """
        Run instant auction (no delay) for immediate task allocation.

        Convenience method combining start, bids, and close.
        """
        agents = available_agents or list(self.agent_profiles.keys())
        if not agents:
            return None

        self.start_auction(task_id, task_type)

        for agent in agents:
            self.submit_bid(task_id, agent)

        return self.close_auction(task_id)

    # =========================================================================
    # COALITION FORMATION (SwarmAgentic Dynamic Teams)
    # =========================================================================

    def form_coalition(
        self,
        task_type: str,
        required_roles: List[str] = None,
        min_agents: int = 2,
        max_agents: int = 5
    ) -> Optional[Coalition]:
        """
        Form dynamic coalition for complex tasks.

        SwarmAgentic pattern: Assemble optimal team based on capabilities.

        Args:
            task_type: Type of task requiring coalition
            required_roles: Specific roles needed (e.g., ["analyzer", "validator"])
            min_agents: Minimum team size
            max_agents: Maximum team size

        Returns:
            Coalition if successfully formed
        """
        import random

        available = [a for a in self.agent_profiles.keys()
                    if a not in self.agent_coalitions]

        if len(available) < min_agents:
            logger.warning(f"Not enough agents for coalition: {len(available)} < {min_agents}")
            return None

        # Score agents for this task type
        scored = []
        for agent in available:
            profile = self.agent_profiles[agent]
            score = (
                profile.get_success_rate(task_type) * 0.4 +
                profile.trust_score * 0.3 +
                (1.0 if profile.specialization.value in (required_roles or []) else 0.5) * 0.3
            )
            scored.append((agent, score, profile.specialization.value))

        scored.sort(key=lambda x: x[1], reverse=True)

        # Select team
        selected = []
        roles_filled = {}

        # First, fill required roles
        for role in (required_roles or []):
            for agent, score, spec in scored:
                if agent not in selected and spec == role:
                    selected.append(agent)
                    roles_filled[agent] = role
                    break

        # Then add highest-scored until max
        for agent, score, spec in scored:
            if len(selected) >= max_agents:
                break
            if agent not in selected:
                selected.append(agent)
                roles_filled[agent] = spec

        if len(selected) < min_agents:
            return None

        # Pick leader (highest trust)
        leader = max(selected, key=lambda a: self.agent_profiles[a].trust_score)

        coalition_id = hashlib.md5(f"coalition:{task_type}:{time.time()}".encode()).hexdigest()[:12]

        coalition = Coalition(
            coalition_id=coalition_id,
            task_type=task_type,
            leader=leader,
            members=selected,
            roles=roles_filled
        )

        # Register coalition
        self.coalitions[coalition_id] = coalition
        for agent in selected:
            self.agent_coalitions[agent] = coalition_id

        # Announce via gossip
        self.gossip_broadcast(
            origin_agent=leader,
            message_type="coalition",
            content={
                "coalition_id": coalition_id,
                "task_type": task_type,
                "members": selected,
                "leader": leader
            }
        )

        logger.info(f"Coalition formed: {coalition_id} with {len(selected)} agents, leader: {leader}")
        return coalition

    def dissolve_coalition(self, coalition_id: str):
        """Dissolve a coalition after task completion."""
        coalition = self.coalitions.pop(coalition_id, None)
        if coalition:
            for agent in coalition.members:
                self.agent_coalitions.pop(agent, None)
            coalition.active = False
            logger.info(f"Coalition dissolved: {coalition_id}")

    def get_coalition(self, agent: str) -> Optional[Coalition]:
        """Get coalition an agent belongs to."""
        coalition_id = self.agent_coalitions.get(agent)
        return self.coalitions.get(coalition_id) if coalition_id else None

    def coalition_broadcast(self, coalition_id: str, message: Dict[str, Any]):
        """Broadcast message to all coalition members."""
        coalition = self.coalitions.get(coalition_id)
        if not coalition:
            return

        for agent in coalition.members:
            if agent not in self.gossip_inbox:
                self.gossip_inbox[agent] = []
            self.gossip_inbox[agent].append(GossipMessage(
                message_id=f"cb_{coalition_id}_{time.time()}",
                content=message,
                origin_agent=coalition.leader,
                message_type="coalition_msg",
                ttl=1  # Direct delivery only
            ))

    # =========================================================================
    # INTEGRATED ROUTING (Combines All Patterns)
    # =========================================================================

    def smart_route(
        self,
        task_id: str,
        task_type: str,
        task_description: str = "",
        prefer_coalition: bool = False,
        use_auction: bool = False,
        use_hierarchy: bool = True
    ) -> Dict[str, Any]:
        """
        Smart routing combining all arXiv swarm patterns.

        Integrates: handoff, hierarchy, auction, coalition, gossip.

        Args:
            task_id: Task identifier
            task_type: Type of task
            task_description: Optional description
            prefer_coalition: Form coalition for complex tasks
            use_auction: Use auction for allocation
            use_hierarchy: Use hierarchical routing

        Returns:
            Dict with routing decision and metadata
        """
        result = {
            "task_id": task_id,
            "assigned_agent": None,
            "coalition": None,
            "method": "direct",
            "confidence": 0.5
        }

        available = list(self.agent_profiles.keys())
        if not available:
            return result

        # Strategy 1: Coalition for complex tasks
        if prefer_coalition:
            coalition = self.form_coalition(task_type, min_agents=2, max_agents=4)
            if coalition:
                result["assigned_agent"] = coalition.leader
                result["coalition"] = coalition.coalition_id
                result["method"] = "coalition"
                result["confidence"] = 0.9
                return result

        # Strategy 2: Auction for competitive allocation
        if use_auction:
            winner = self.auto_auction(task_id, task_type, available)
            if winner:
                result["assigned_agent"] = winner
                result["method"] = "auction"
                result["confidence"] = 0.85
                return result

        # Strategy 3: Hierarchical routing
        if use_hierarchy and self._tree_built:
            agent = self.route_via_hierarchy(task_type)
            if agent:
                result["assigned_agent"] = agent
                result["method"] = "hierarchy"
                result["confidence"] = 0.8
                return result

        # Strategy 4: MorphAgent TRAS scoring
        if task_description:
            profiles = {a: self.agent_profiles[a] for a in available}
            best = self.morph_scorer.get_best_agent_by_tras(
                profiles=profiles,
                task=task_description,
                task_type=task_type
            )
            if best:
                result["assigned_agent"] = best
                result["method"] = "morph_tras"
                result["confidence"] = 0.75
                return result

        # Strategy 5: Fallback to simple routing
        best = self.get_best_agent_for_task(task_type, available, task_description)
        result["assigned_agent"] = best
        result["method"] = "simple"
        result["confidence"] = 0.6
        return result

    # =========================================================================
    # WORK-STEALING (Idle agents steal from busy ones)
    # =========================================================================

    def get_agent_load(self, agent: str) -> float:
        """
        Get current load of an agent (0-1).

        Based on: pending handoffs, coalition membership, recent task count.
        """
        load = 0.0

        # Pending handoffs to this agent
        pending = len([h for h in self.pending_handoffs.values() if h.to_agent == agent])
        load += min(0.4, pending * 0.1)

        # Coalition membership
        if agent in self.agent_coalitions:
            load += 0.2

        # Recent tasks (from collective memory)
        recent = [m for m in self.collective_memory[-20:]
                  if m.get('agent') == agent and time.time() - m.get('timestamp', 0) < 60]
        load += min(0.4, len(recent) * 0.1)

        return min(1.0, load)

    def find_overloaded_agents(self, threshold: float = 0.7) -> List[str]:
        """Find agents with load above threshold."""
        return [a for a in self.agent_profiles.keys()
                if self.get_agent_load(a) > threshold]

    def find_idle_agents(self, threshold: float = 0.3) -> List[str]:
        """Find agents with load below threshold."""
        return [a for a in self.agent_profiles.keys()
                if self.get_agent_load(a) < threshold]

    def work_steal(self, idle_agent: str) -> Optional[HandoffContext]:
        """
        Idle agent steals work from overloaded agent.

        Work-stealing pattern: Automatic load balancing.

        Returns:
            HandoffContext if work was stolen, None otherwise.
        """
        overloaded = self.find_overloaded_agents()
        if not overloaded:
            return None

        # Find task to steal (pending handoff to overloaded agent)
        for busy_agent in overloaded:
            for task_id, handoff in list(self.pending_handoffs.items()):
                if handoff.to_agent == busy_agent:
                    # Steal this task
                    handoff.to_agent = idle_agent
                    handoff.add_to_chain(busy_agent)  # Record original target

                    logger.info(f"Work stolen: {idle_agent} took {task_id} from {busy_agent}")

                    # Notify via gossip
                    self.gossip_broadcast(
                        origin_agent=idle_agent,
                        message_type="work_steal",
                        content={"task_id": task_id, "from": busy_agent, "to": idle_agent}
                    )

                    return handoff

        return None

    def balance_load(self) -> List[Dict]:
        """
        Rebalance work across the swarm.

        Returns list of rebalancing actions taken.
        """
        actions = []
        idle = self.find_idle_agents()
        overloaded = self.find_overloaded_agents()

        for idle_agent in idle:
            if not overloaded:
                break

            result = self.work_steal(idle_agent)
            if result:
                actions.append({
                    "action": "work_steal",
                    "from": result.handoff_chain[-1] if result.handoff_chain else "unknown",
                    "to": idle_agent,
                    "task_id": result.task_id
                })
                # Recalculate overloaded
                overloaded = self.find_overloaded_agents()

        if actions:
            logger.info(f"Load balanced: {len(actions)} tasks redistributed")

        return actions

    # =========================================================================
    # FAILURE RECOVERY (Auto-retry with different agent)
    # =========================================================================

    def record_failure(
        self,
        task_id: str,
        agent: str,
        task_type: str,
        error_type: str = "unknown",
        context: Dict = None
    ) -> Optional[str]:
        """
        Record task failure and auto-reassign to different agent.

        Failure recovery pattern: Automatic retry with alternative.

        Returns:
            New assigned agent or None if no alternatives.
        """
        # Update agent profile (reduce trust)
        if agent in self.agent_profiles:
            profile = self.agent_profiles[agent]
            profile.trust_score = max(0.1, profile.trust_score - 0.1)
            profile.update_task_result(task_type, False, 0.0)

        # Find alternative agent
        failed_agents = [agent]
        if task_id in self.pending_handoffs:
            failed_agents.extend(self.pending_handoffs[task_id].handoff_chain)

        available = [a for a in self.agent_profiles.keys() if a not in failed_agents]

        if not available:
            logger.warning(f"Task {task_id} failed, no alternatives available")
            return None

        # Use auction to find best alternative
        new_agent = self.auto_auction(f"{task_id}_retry", task_type, available)

        if new_agent:
            # Create handoff with failure context
            self.initiate_handoff(
                task_id=f"{task_id}_retry",
                from_agent=agent,
                to_agent=new_agent,
                task_type=task_type,
                context={**(context or {}), "retry_reason": error_type, "failed_agent": agent},
                progress=0.0
            )
            logger.info(f"Task {task_id} reassigned: {agent} (failed) → {new_agent}")

        return new_agent

    def get_failure_rate(self, agent: str, task_type: str = None) -> float:
        """Get failure rate for an agent (optionally for specific task type)."""
        profile = self.agent_profiles.get(agent)
        if not profile:
            return 0.5

        if task_type:
            return 1.0 - profile.get_success_rate(task_type)

        # Overall failure rate
        total_success = sum(s for s, t in profile.task_success.values())
        total_tasks = sum(t for s, t in profile.task_success.values())
        return 1.0 - (total_success / total_tasks) if total_tasks > 0 else 0.5

    # =========================================================================
    # PRIORITY QUEUE (Handle urgent tasks first)
    # =========================================================================

    def __init_priority_queue(self):
        """Initialize priority queue if not exists."""
        if not hasattr(self, 'priority_queue'):
            self.priority_queue: List[Dict] = []

    def enqueue_task(
        self,
        task_id: str,
        task_type: str,
        priority: int = 5,
        deadline: float = None,
        context: Dict = None
    ):
        """
        Add task to priority queue.

        Priority 1-10 (10 = most urgent).
        """
        self.__init_priority_queue()

        task = {
            "task_id": task_id,
            "task_type": task_type,
            "priority": priority,
            "deadline": deadline or (time.time() + 3600),
            "context": context or {},
            "enqueued_at": time.time()
        }

        # Insert in priority order (higher priority first)
        inserted = False
        for i, t in enumerate(self.priority_queue):
            if priority > t["priority"]:
                self.priority_queue.insert(i, task)
                inserted = True
                break
            elif priority == t["priority"]:
                # Same priority: earlier deadline first
                if task["deadline"] < t["deadline"]:
                    self.priority_queue.insert(i, task)
                    inserted = True
                    break

        if not inserted:
            self.priority_queue.append(task)

        logger.debug(f"Task enqueued: {task_id} (priority={priority})")

    def dequeue_task(self) -> Optional[Dict]:
        """Get highest priority task from queue."""
        self.__init_priority_queue()

        if not self.priority_queue:
            return None

        return self.priority_queue.pop(0)

    def peek_queue(self, n: int = 5) -> List[Dict]:
        """Peek at top N tasks in queue."""
        self.__init_priority_queue()
        return self.priority_queue[:n]

    def escalate_priority(self, task_id: str, new_priority: int):
        """Escalate task priority (reposition in queue)."""
        self.__init_priority_queue()

        # Find and remove task
        task = None
        for i, t in enumerate(self.priority_queue):
            if t["task_id"] == task_id:
                task = self.priority_queue.pop(i)
                break

        if task:
            # Re-enqueue with new priority (only pass valid params)
            self.enqueue_task(
                task_id=task["task_id"],
                task_type=task["task_type"],
                priority=new_priority,
                deadline=task.get("deadline"),
                context=task.get("context")
            )
            logger.info(f"Task escalated: {task_id} → priority {new_priority}")

    # =========================================================================
    # TASK DECOMPOSITION (Split complex tasks)
    # =========================================================================

    def decompose_task(
        self,
        task_id: str,
        task_type: str,
        subtasks: List[Dict],
        parallel: bool = True
    ) -> List[str]:
        """
        Decompose complex task into subtasks.

        Args:
            task_id: Parent task ID
            task_type: Type of parent task
            subtasks: List of {"type": str, "context": dict, "priority": int}
            parallel: Whether subtasks can run in parallel

        Returns:
            List of subtask IDs assigned to agents.
        """
        subtask_ids = []

        for i, sub in enumerate(subtasks):
            sub_id = f"{task_id}_sub_{i}"
            sub_type = sub.get("type", task_type)
            sub_priority = sub.get("priority", 5)
            sub_context = sub.get("context", {})
            sub_context["parent_task"] = task_id
            sub_context["subtask_index"] = i
            sub_context["parallel"] = parallel

            # Route subtask to best agent
            route = self.smart_route(
                task_id=sub_id,
                task_type=sub_type,
                task_description=sub_context.get("description", "")
            )

            if route["assigned_agent"]:
                # Create handoff for subtask
                self.initiate_handoff(
                    task_id=sub_id,
                    from_agent="decomposer",
                    to_agent=route["assigned_agent"],
                    task_type=sub_type,
                    context=sub_context,
                    priority=sub_priority
                )
                subtask_ids.append(sub_id)

        logger.info(f"Task decomposed: {task_id} → {len(subtask_ids)} subtasks")
        return subtask_ids

    def aggregate_subtask_results(
        self,
        parent_task_id: str,
        results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Aggregate results from completed subtasks.

        Args:
            parent_task_id: Parent task ID
            results: Dict of subtask_id -> result

        Returns:
            Aggregated result dict.
        """
        aggregated = {
            "parent_task": parent_task_id,
            "subtask_count": len(results),
            "successful": sum(1 for r in results.values() if r.get("success", False)),
            "results": results,
            "aggregated_at": time.time()
        }

        # Calculate overall success
        aggregated["overall_success"] = aggregated["successful"] == aggregated["subtask_count"]

        return aggregated

    # =========================================================================
    # BYZANTINE CONSENSUS (Fault-tolerant agreement)
    # =========================================================================

    def byzantine_vote(
        self,
        question: str,
        options: List[str],
        voters: List[str] = None,
        threshold: float = 0.67
    ) -> Dict[str, Any]:
        """
        Byzantine fault-tolerant voting.

        Requires 2/3 majority for consensus (can tolerate 1/3 faulty nodes).

        Args:
            question: Question to vote on
            options: Available options
            voters: Participating agents (all if None)
            threshold: Required majority (default 2/3)

        Returns:
            Dict with decision, consensus reached, vote distribution.
        """
        voters = voters or list(self.agent_profiles.keys())

        # Collect votes (weighted by trust)
        votes = defaultdict(float)
        vote_details = []

        for agent in voters:
            profile = self.agent_profiles.get(agent, AgentProfile(agent))

            # Agent votes for option based on specialization match
            # (In real implementation, this would call agent's vote method)
            best_option = options[0]
            best_score = 0

            for opt in options:
                # Score based on past success with similar tasks
                score = profile.get_success_rate(opt) + profile.trust_score * 0.5
                if score > best_score:
                    best_score = score
                    best_option = opt

            # Weight vote by trust
            weight = profile.trust_score
            votes[best_option] += weight
            vote_details.append({
                "agent": agent,
                "vote": best_option,
                "weight": weight
            })

        # Determine winner
        total_weight = sum(votes.values())
        if total_weight == 0:
            return {
                "decision": options[0],
                "consensus": False,
                "reason": "no_votes",
                "votes": vote_details
            }

        winner = max(votes.keys(), key=lambda k: votes[k])
        winner_share = votes[winner] / total_weight

        consensus = winner_share >= threshold

        result = {
            "decision": winner,
            "consensus": consensus,
            "share": winner_share,
            "threshold": threshold,
            "votes": vote_details,
            "distribution": dict(votes)
        }

        if consensus:
            logger.info(f"Byzantine consensus reached: {winner} ({winner_share:.0%})")
        else:
            logger.warning(f"Byzantine consensus FAILED: {winner} only {winner_share:.0%} < {threshold:.0%}")

        return result

    # =========================================================================
    # CIRCUIT BREAKER (Stop sending to failing agents)
    # =========================================================================

    def __init_circuit_breakers(self):
        """Initialize circuit breakers if not exists."""
        if not hasattr(self, 'circuit_breakers'):
            self.circuit_breakers: Dict[str, Dict] = {}  # agent -> {state, failures, last_failure}

    def get_circuit_state(self, agent: str) -> str:
        """Get circuit breaker state: 'closed' (ok), 'open' (blocked), 'half-open' (testing)."""
        self.__init_circuit_breakers()
        cb = self.circuit_breakers.get(agent, {})
        return cb.get('state', 'closed')

    def record_circuit_failure(self, agent: str, threshold: int = 3, cooldown: float = 60.0):
        """
        Record failure for circuit breaker.

        After `threshold` failures, circuit opens (blocks agent).
        After `cooldown` seconds, circuit becomes half-open (allows one test).
        """
        self.__init_circuit_breakers()

        if agent not in self.circuit_breakers:
            self.circuit_breakers[agent] = {'state': 'closed', 'failures': 0, 'last_failure': 0}

        cb = self.circuit_breakers[agent]
        cb['failures'] += 1
        cb['last_failure'] = time.time()

        if cb['failures'] >= threshold:
            cb['state'] = 'open'
            logger.warning(f"Circuit OPEN for {agent} after {cb['failures']} failures")

    def record_circuit_success(self, agent: str):
        """Record success - resets circuit breaker."""
        self.__init_circuit_breakers()

        if agent in self.circuit_breakers:
            self.circuit_breakers[agent] = {'state': 'closed', 'failures': 0, 'last_failure': 0}

    def check_circuit(self, agent: str, cooldown: float = 60.0) -> bool:
        """
        Check if agent is available (circuit not open).

        Returns True if agent can receive tasks.
        """
        self.__init_circuit_breakers()

        cb = self.circuit_breakers.get(agent)
        if not cb:
            return True

        if cb['state'] == 'closed':
            return True

        if cb['state'] == 'open':
            # Check if cooldown passed
            if time.time() - cb['last_failure'] > cooldown:
                cb['state'] = 'half-open'
                logger.info(f"Circuit HALF-OPEN for {agent} (testing)")
                return True
            return False

        # half-open - allow one test
        return True

    def get_available_agents(self, agents: List[str] = None) -> List[str]:
        """Get agents with closed or half-open circuits."""
        agents = agents or list(self.agent_profiles.keys())
        return [a for a in agents if self.check_circuit(a)]

    # =========================================================================
    # BACKPRESSURE (Slow down when overwhelmed)
    # =========================================================================

    def calculate_backpressure(self) -> float:
        """
        Calculate swarm backpressure (0-1).

        High backpressure means swarm is overwhelmed.
        """
        if not self.agent_profiles:
            return 0.0

        # Factors contributing to backpressure
        avg_load = sum(self.get_agent_load(a) for a in self.agent_profiles) / len(self.agent_profiles)
        pending_ratio = min(1.0, len(self.pending_handoffs) / max(1, len(self.agent_profiles) * 3))
        queue_pressure = min(1.0, len(getattr(self, 'priority_queue', [])) / 20)

        backpressure = (avg_load * 0.4 + pending_ratio * 0.4 + queue_pressure * 0.2)
        return min(1.0, backpressure)

    def should_accept_task(self, priority: int = 5) -> bool:
        """
        Check if swarm should accept new task based on backpressure.

        High priority tasks (>=8) always accepted.
        """
        if priority >= 8:
            return True

        backpressure = self.calculate_backpressure()

        # Accept based on priority vs backpressure
        # Priority 5 needs backpressure < 0.7
        # Priority 3 needs backpressure < 0.5
        threshold = 0.5 + (priority - 5) * 0.1
        return backpressure < threshold

    # =========================================================================
    # EMERGENT LEADERSHIP (Dynamic leader election)
    # =========================================================================

    def elect_leader(self, candidates: List[str] = None, task_type: str = None) -> Optional[str]:
        """
        Elect leader based on trust, success rate, and availability.

        Emergent leadership pattern: Best performer leads.
        """
        candidates = candidates or list(self.agent_profiles.keys())
        available = self.get_available_agents(candidates)

        if not available:
            return None

        def score_candidate(agent: str) -> float:
            profile = self.agent_profiles.get(agent, AgentProfile(agent))
            score = profile.trust_score * 0.4

            if task_type:
                score += profile.get_success_rate(task_type) * 0.3
            else:
                # Overall success
                total_s = sum(s for s, t in profile.task_success.values())
                total_t = sum(t for s, t in profile.task_success.values())
                score += (total_s / total_t if total_t > 0 else 0.5) * 0.3

            # Low load bonus
            load = self.get_agent_load(agent)
            score += (1 - load) * 0.2

            # Experience bonus
            score += min(0.1, profile.total_tasks / 100)

            return score

        leader = max(available, key=score_candidate)
        logger.info(f"Leader elected: {leader} (score: {score_candidate(leader):.2f})")
        return leader

    # =========================================================================
    # ADAPTIVE TIMEOUT (Adjust based on task/agent history)
    # =========================================================================

    def get_adaptive_timeout(self, agent: str, task_type: str, base_timeout: float = 30.0) -> float:
        """
        Calculate adaptive timeout based on agent's historical performance.

        Slow agents get more time, fast agents get less.
        """
        profile = self.agent_profiles.get(agent)
        if not profile or profile.total_tasks < 3:
            return base_timeout

        # Use agent's average execution time
        avg_time = profile.avg_execution_time

        # Add buffer based on variance (if we had it, use 1.5x for now)
        timeout = avg_time * 2.0

        # Clamp to reasonable bounds
        return max(base_timeout * 0.5, min(base_timeout * 3, timeout))

    # =========================================================================
    # AGENT LIFECYCLE (Spawn/retire dynamically)
    # =========================================================================

    def should_spawn_agent(self, task_type: str = None) -> bool:
        """
        Determine if new agent should be spawned.

        Based on: load, queue size, specialization gaps.
        """
        backpressure = self.calculate_backpressure()
        if backpressure < 0.7:
            return False

        # Check specialization gap
        if task_type:
            specialists = [a for a, p in self.agent_profiles.items()
                         if p.get_success_rate(task_type) > 0.8]
            if len(specialists) < 2:
                return True

        return backpressure > 0.85

    def should_retire_agent(self, agent: str) -> bool:
        """
        Determine if agent should be retired.

        Based on: low trust, high failure rate, idle time.
        """
        profile = self.agent_profiles.get(agent)
        if not profile:
            return False

        # Low trust
        if profile.trust_score < 0.2:
            return True

        # High failure rate with enough history
        if profile.total_tasks >= 10:
            total_s = sum(s for s, t in profile.task_success.values())
            total_t = sum(t for s, t in profile.task_success.values())
            if total_t > 0 and total_s / total_t < 0.3:
                return True

        # Circuit breaker open for too long
        cb = self.circuit_breakers.get(agent, {}) if hasattr(self, 'circuit_breakers') else {}
        if cb.get('state') == 'open' and time.time() - cb.get('last_failure', 0) > 300:
            return True

        return False

    def retire_agent(self, agent: str):
        """Remove agent from swarm."""
        if agent in self.agent_profiles:
            del self.agent_profiles[agent]

        # Clean up related state
        self.agent_coalitions.pop(agent, None)
        if hasattr(self, 'circuit_breakers'):
            self.circuit_breakers.pop(agent, None)

        # Reassign pending handoffs
        for task_id, handoff in list(self.pending_handoffs.items()):
            if handoff.to_agent == agent:
                available = [a for a in self.agent_profiles.keys() if a != agent]
                if available:
                    new_agent = self.get_best_agent_for_task(handoff.task_type, available)
                    if new_agent:
                        handoff.to_agent = new_agent

        logger.info(f"Agent retired: {agent}")

    # =========================================================================
    # PARALLEL EXECUTION (Speed up multi-agent tasks)
    # =========================================================================

    async def execute_parallel(
        self,
        tasks: List[Dict],
        timeout_per_task: float = 30.0
    ) -> List[Dict]:
        """
        Execute multiple tasks in parallel across agents.

        Args:
            tasks: List of {"task_id", "task_type", "func", "args", "kwargs"}
            timeout_per_task: Timeout per task in seconds

        Returns:
            List of results with success/failure status
        """
        import asyncio

        async def run_task(task: Dict) -> Dict:
            task_id = task.get("task_id", "unknown")
            func = task.get("func")
            args = task.get("args", [])
            kwargs = task.get("kwargs", {})

            start = time.time()
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=timeout_per_task
                    )
                else:
                    result = func(*args, **kwargs)

                return {
                    "task_id": task_id,
                    "success": True,
                    "result": result,
                    "execution_time": time.time() - start
                }
            except asyncio.TimeoutError:
                return {
                    "task_id": task_id,
                    "success": False,
                    "error": "timeout",
                    "execution_time": timeout_per_task
                }
            except Exception as e:
                return {
                    "task_id": task_id,
                    "success": False,
                    "error": str(e),
                    "execution_time": time.time() - start
                }

        # Run all tasks concurrently
        results = await asyncio.gather(*[run_task(t) for t in tasks], return_exceptions=True)

        # Process results
        processed = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                processed.append({
                    "task_id": tasks[i].get("task_id", f"task_{i}"),
                    "success": False,
                    "error": str(r)
                })
            else:
                processed.append(r)

        return processed

    async def parallel_map(
        self,
        items: List[Any],
        func,
        max_concurrent: int = 5
    ) -> List[Any]:
        """
        Apply function to items in parallel with concurrency limit.

        Useful for processing multiple concepts, papers, etc.

        Args:
            items: Items to process
            func: Async function to apply
            max_concurrent: Max concurrent executions

        Returns:
            List of results in order
        """
        import asyncio

        semaphore = asyncio.Semaphore(max_concurrent)

        async def limited_func(item, idx):
            async with semaphore:
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(item)
                    else:
                        return func(item)
                except Exception as e:
                    logger.warning(f"parallel_map item {idx} failed: {e}")
                    return None

        results = await asyncio.gather(*[
            limited_func(item, i) for i, item in enumerate(items)
        ])

        return list(results)

    # =========================================================================
    # SMART CACHING (Reduce redundant LLM calls)
    # =========================================================================

    def __init_cache(self):
        """Initialize result cache."""
        if not hasattr(self, '_result_cache'):
            self._result_cache: Dict[str, Dict] = {}
            self._cache_hits = 0
            self._cache_misses = 0

    def cache_result(self, key: str, result: Any, ttl: float = 3600.0):
        """
        Cache a result with TTL.

        Args:
            key: Cache key
            result: Result to cache
            ttl: Time-to-live in seconds (default 1 hour)
        """
        self.__init_cache()
        self._result_cache[key] = {
            "result": result,
            "cached_at": time.time(),
            "ttl": ttl
        }

    def get_cached(self, key: str) -> Optional[Any]:
        """
        Get cached result if not expired.

        Returns None if not cached or expired.
        """
        self.__init_cache()

        entry = self._result_cache.get(key)
        if not entry:
            self._cache_misses += 1
            return None

        # Check TTL
        age = time.time() - entry["cached_at"]
        if age > entry["ttl"]:
            del self._result_cache[key]
            self._cache_misses += 1
            return None

        self._cache_hits += 1
        return entry["result"]

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        self.__init_cache()
        total = self._cache_hits + self._cache_misses
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": self._cache_hits / total if total > 0 else 0,
            "size": len(self._result_cache)
        }

    def clear_cache(self, pattern: str = None):
        """Clear cache entries (optionally matching pattern)."""
        self.__init_cache()
        if pattern:
            import fnmatch
            keys_to_delete = [k for k in self._result_cache if fnmatch.fnmatch(k, pattern)]
            for k in keys_to_delete:
                del self._result_cache[k]
        else:
            self._result_cache.clear()

    # =========================================================================
    # INCREMENTAL PROCESSING (Stream results as they complete)
    # =========================================================================

    async def execute_incremental(
        self,
        tasks: List[Dict],
        on_complete=None,
        on_error=None
    ):
        """
        Execute tasks and yield results incrementally as they complete.

        Args:
            tasks: List of {"task_id", "func", "args", "kwargs"}
            on_complete: Callback(task_id, result) called on each completion
            on_error: Callback(task_id, error) called on each error

        Yields:
            Results as they complete (not in order)
        """
        import asyncio

        async def run_task(task: Dict):
            task_id = task.get("task_id", "unknown")
            func = task.get("func")
            args = task.get("args", [])
            kwargs = task.get("kwargs", {})

            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)

                if on_complete:
                    on_complete(task_id, result)
                return {"task_id": task_id, "success": True, "result": result}

            except Exception as e:
                if on_error:
                    on_error(task_id, e)
                return {"task_id": task_id, "success": False, "error": str(e)}

        # Use as_completed for incremental results
        pending = [asyncio.create_task(run_task(t)) for t in tasks]

        for coro in asyncio.as_completed(pending):
            result = await coro
            yield result

    # =========================================================================
    # CHUNKED PROCESSING (Process large batches efficiently)
    # =========================================================================

    async def process_in_chunks(
        self,
        items: List[Any],
        chunk_size: int,
        process_func,
        delay_between_chunks: float = 0.1
    ) -> List[Any]:
        """
        Process items in chunks to avoid overwhelming LLM.

        Useful for processing many concepts without timeouts.

        Args:
            items: All items to process
            chunk_size: Items per chunk
            process_func: Async function to process a chunk
            delay_between_chunks: Delay between chunks

        Returns:
            All results combined
        """
        import asyncio

        results = []
        for i in range(0, len(items), chunk_size):
            chunk = items[i:i + chunk_size]

            # Check backpressure
            if not self.should_accept_task(priority=5):
                logger.warning(f"Backpressure high, waiting before chunk {i // chunk_size}")
                await asyncio.sleep(1.0)

            chunk_results = await process_func(chunk)
            results.extend(chunk_results if isinstance(chunk_results, list) else [chunk_results])

            if delay_between_chunks > 0 and i + chunk_size < len(items):
                await asyncio.sleep(delay_between_chunks)

        return results

    # =========================================================================
    # SWARM HEALTH MONITORING
    # =========================================================================

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
            'collective_memory': self.collective_memory[-limit:],  # Keep recent
            'stigmergy': self.stigmergy.to_dict(),  # Persist stigmergy state
            'benchmarks': self.benchmarks.to_dict(),  # Persist benchmark data
            'curriculum': self.curriculum_generator.to_dict(),  # DrZero curriculum state
            'morph_score_history': self.morph_score_history[-50:],  # MorphAgent score history
            'tool_manager': self.tool_manager.to_dict(),  # Agent0 tool management state
            # arXiv swarm enhancements
            'handoff_history': [
                {'task_id': h.task_id, 'from': h.from_agent, 'to': h.to_agent,
                 'task_type': h.task_type, 'progress': h.progress, 'chain': h.handoff_chain}
                for h in self.handoff_history[-50:]
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

            self.collective_memory = data.get('collective_memory', [])

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
                self.morph_score_history = data['morph_score_history']

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
