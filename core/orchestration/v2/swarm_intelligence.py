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

        logger.info("SwarmIntelligence initialized (DrZero curriculum + MorphAgent scoring)")

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
]
