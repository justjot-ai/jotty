"""
SwarmLearningPipeline - Extracted from SwarmManager
====================================================

All learning-related initialization, persistence, and post-episode hooks.
SwarmManager delegates to this class for learning concerns.
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from Jotty.core.foundation.data_structures import JottyConfig, EpisodeResult
from Jotty.core.foundation.agent_config import AgentConfig
from Jotty.core.foundation.robust_parsing import AdaptiveWeightGroup

logger = logging.getLogger(__name__)


class SwarmLearningPipeline:
    """
    Manages all learning components: RL, credit assignment, memory consolidation,
    transferable learning, swarm intelligence, and MAS learning.

    Extracted from SwarmManager to reduce god-object coupling.
    """

    def __init__(self, config: JottyConfig):
        self.config = config
        self.episode_count = 0
        self._init_components()

    def _init_components(self):
        """Initialize all learning components."""
        from Jotty.core.learning.learning_coordinator import LearningCoordinator as LearningManager
        from Jotty.core.learning.predictive_marl import (
            LLMTrajectoryPredictor, DivergenceMemory,
            CooperativeCreditAssigner,
        )
        from Jotty.core.memory.consolidation_engine import (
            BrainStateMachine, BrainModeConfig, AgentAbstractor,
        )
        from Jotty.core.agents.axon import SmartAgentSlack
        from Jotty.core.agents.feedback_channel import FeedbackChannel
        from Jotty.core.orchestration.v2.swarm_learner import SwarmLearner
        from Jotty.core.learning.transfer_learning import TransferableLearningStore
        from .swarm_intelligence import SwarmIntelligence

        # Core learning manager (wraps Q-learner)
        self.learning_manager = LearningManager(self.config)

        # Trajectory prediction (MARL)
        self.trajectory_predictor = None
        try:
            self.trajectory_predictor = LLMTrajectoryPredictor(self.config, horizon=5)
        except Exception as e:
            logger.warning(f"Trajectory predictor unavailable: {e}")

        # Divergence memory for storing prediction errors
        self.divergence_memory = DivergenceMemory(self.config)

        # Cooperative credit assignment
        self.cooperative_credit = CooperativeCreditAssigner(self.config)

        # Brain state machine for consolidation
        brain_config = BrainModeConfig()
        self.brain_state = BrainStateMachine(brain_config)

        # Agent abstractor for scalable role tracking
        self.agent_abstractor = AgentAbstractor(brain_config)

        # Inter-agent communication
        self.agent_slack = SmartAgentSlack(enable_cooperation=True)
        self.feedback_channel = FeedbackChannel()

        # Swarm learner for prompt evolution
        self.swarm_learner = SwarmLearner(self.config)

        # Transferable learning (cross-swarm, cross-goal)
        self.transfer_learning = TransferableLearningStore(self.config)

        # Swarm intelligence (emergent specialization, consensus, routing)
        self.swarm_intelligence = SwarmIntelligence(self.config)

        # Adaptive credit assignment weights
        self.credit_weights = AdaptiveWeightGroup({
            'base_reward': 0.3,
            'cooperation_bonus': 0.4,
            'predictability_bonus': 0.3,
        })

        # -----------------------------------------------------------------
        # Previously dormant modules — now wired (DRY: init once, use many)
        # -----------------------------------------------------------------
        from .stigmergy import StigmergyLayer
        from .byzantine_verification import ByzantineVerifier
        from .credit_assignment import CreditAssignment
        from .adaptive_learning import AdaptiveLearning
        from .curriculum_generator import CurriculumGenerator

        # Stigmergy: ant-colony pheromone trails for agent routing
        self.stigmergy = StigmergyLayer(decay_rate=0.1, max_signals=500)

        # Byzantine: detect and penalize agents that lie about success
        self.byzantine_verifier = ByzantineVerifier(self.swarm_intelligence)

        # Credit assignment: which agent/improvement actually helped?
        self.credit_assigner = CreditAssignment()

        # Adaptive learning: dynamic learning rate + exploration balance
        self.adaptive_learning = AdaptiveLearning(base_learning_rate=1.0)

        # Curriculum: self-generated training tasks (DrZero-inspired)
        self.curriculum_generator = CurriculumGenerator(
            config=self.config,
            memory_system=None,  # Wired later if HierarchicalMemory available
        )

        # Training task queue (filled by post_episode when exploration needed)
        self._pending_training_tasks: list = []

        # Paradigm effectiveness tracker (auto paradigm selection)
        # KISS: Simple dict, no new classes. Persisted with stigmergy.
        self._paradigm_stats: Dict[str, Dict[str, int]] = {
            'fanout': {'runs': 0, 'successes': 0},
            'relay': {'runs': 0, 'successes': 0},
            'debate': {'runs': 0, 'successes': 0},
            'refinement': {'runs': 0, 'successes': 0},
        }

    # =========================================================================
    # Persistence paths
    # =========================================================================

    def _get_learning_path(self) -> Path:
        base = getattr(self.config, 'base_path', None)
        if base:
            return Path(base) / 'swarm_learnings.json'
        return Path.home() / '.jotty' / 'swarm_learnings.json'

    def _get_transfer_learning_path(self) -> Path:
        base = getattr(self.config, 'base_path', None)
        if base:
            return Path(base) / 'transfer_learnings.json'
        return Path.home() / '.jotty' / 'transfer_learnings.json'

    def _get_swarm_intelligence_path(self) -> Path:
        base = getattr(self.config, 'base_path', None)
        if base:
            return Path(base) / 'swarm_intelligence.json'
        return Path.home() / '.jotty' / 'swarm_intelligence.json'

    def _get_credit_weights_path(self) -> Path:
        base = getattr(self.config, 'base_path', None)
        if base:
            return Path(base) / 'credit_weights.json'
        return Path.home() / '.jotty' / 'credit_weights.json'

    def _get_stigmergy_path(self) -> Path:
        base = getattr(self.config, 'base_path', None)
        if base:
            return Path(base) / 'stigmergy.json'
        return Path.home() / '.jotty' / 'stigmergy.json'

    # =========================================================================
    # Auto-load / Auto-save
    # =========================================================================

    def auto_load(self):
        """Load previous learnings at startup."""
        # Q-learner state
        learning_path = self._get_learning_path()
        if learning_path.exists():
            try:
                self.learning_manager.q_learner.load_state(str(learning_path))
                q_summary = self.learning_manager.get_q_table_summary()
                logger.info(f"Auto-loaded {q_summary['size']} Q-entries from {learning_path}")
            except Exception as e:
                logger.debug(f"Could not auto-load Q-learnings: {e}")

        # Transferable learnings
        transfer_path = self._get_transfer_learning_path()
        if self.transfer_learning.load(str(transfer_path)):
            logger.info(f"Auto-loaded transferable learnings from {transfer_path}")

        # Swarm intelligence
        si_path = self._get_swarm_intelligence_path()
        if self.swarm_intelligence.load(str(si_path)):
            specs = self.swarm_intelligence.get_specialization_summary()
            logger.info(f"Auto-loaded swarm intelligence: {len(specs)} agent profiles")

        # Adaptive credit weights
        credit_path = self._get_credit_weights_path()
        if credit_path.exists():
            try:
                with open(credit_path, 'r') as f:
                    credit_data = json.load(f)
                self.credit_weights = AdaptiveWeightGroup.from_dict(credit_data)
                logger.info(f"Auto-loaded credit weights: {self.credit_weights}")
            except Exception as e:
                logger.debug(f"Could not auto-load credit weights: {e}")

        # Stigmergy pheromone trails + paradigm stats
        stig_path = self._get_stigmergy_path()
        if stig_path.exists():
            try:
                from .stigmergy import StigmergyLayer
                with open(stig_path, 'r') as f:
                    stig_data = json.load(f)
                self.stigmergy = StigmergyLayer.from_dict(stig_data)
                # Restore paradigm stats if present (saved alongside stigmergy)
                if 'paradigm_stats' in stig_data:
                    self._paradigm_stats.update(stig_data['paradigm_stats'])
                logger.info(
                    f"Auto-loaded stigmergy: {len(self.stigmergy.signals)} signals"
                )
            except Exception as e:
                logger.debug(f"Could not auto-load stigmergy: {e}")

    def auto_save(self, mas_learning=None, swarm_terminal=None, provider_registry=None):
        """Save learnings after execution."""
        # Q-learner state
        learning_path = self._get_learning_path()
        try:
            learning_path.parent.mkdir(parents=True, exist_ok=True)
            self.learning_manager.q_learner.save_state(str(learning_path))
        except Exception as e:
            logger.debug(f"Could not auto-save Q-learnings: {e}")

        # Transferable learnings
        transfer_path = self._get_transfer_learning_path()
        try:
            self.transfer_learning.save(str(transfer_path))
        except Exception as e:
            logger.debug(f"Could not auto-save transfer learnings: {e}")

        # Swarm intelligence
        si_path = self._get_swarm_intelligence_path()
        try:
            self.swarm_intelligence.save(str(si_path))
        except Exception as e:
            logger.debug(f"Could not auto-save swarm intelligence: {e}")

        # Adaptive credit weights
        credit_path = self._get_credit_weights_path()
        try:
            credit_path.parent.mkdir(parents=True, exist_ok=True)
            with open(credit_path, 'w') as f:
                json.dump(self.credit_weights.to_dict(), f, indent=2)
        except Exception as e:
            logger.debug(f"Could not auto-save credit weights: {e}")

        # Stigmergy pheromone trails + paradigm stats
        stig_path = self._get_stigmergy_path()
        try:
            stig_path.parent.mkdir(parents=True, exist_ok=True)
            stig_data = self.stigmergy.to_dict()
            # Include paradigm stats in same file (DRY: no new path)
            stig_data['paradigm_stats'] = self._paradigm_stats
            with open(stig_path, 'w') as f:
                json.dump(stig_data, f, indent=2)
        except Exception as e:
            logger.debug(f"Could not auto-save stigmergy: {e}")

        # Provider registry
        if provider_registry:
            try:
                base = getattr(self.config, 'base_path', None)
                if base:
                    provider_path = Path(base) / 'provider_learnings.json'
                else:
                    provider_path = Path.home() / '.jotty' / 'provider_learnings.json'
                provider_registry.save_state(str(provider_path))
            except Exception as e:
                logger.debug(f"Could not auto-save provider learnings: {e}")

        # MAS Learning
        if mas_learning:
            try:
                if swarm_terminal:
                    mas_learning.sync_from_terminal(swarm_terminal)
                mas_learning.save_all()
            except Exception as e:
                logger.debug(f"Could not auto-save MAS learnings: {e}")

    # =========================================================================
    # Post-episode learning hooks
    # =========================================================================

    def post_episode(
        self,
        result: EpisodeResult,
        goal: str,
        agents: list,
        architect_prompts: list,
        mas_learning=None,
        swarm_terminal=None,
    ):
        """
        Post-episode learning: swarm learner, brain consolidation, NeuroChunk tiering.
        Called at end of both single-agent and multi-agent execution.
        """
        self.episode_count += 1
        episode_reward = 1.0 if result.success else 0.0

        # 1. SwarmLearner: record episode, conditionally update prompts
        try:
            trajectory = result.trajectory or []
            insights = []
            if hasattr(result, 'tagged_outputs') and result.tagged_outputs:
                insights = [str(t) for t in result.tagged_outputs[:5]]
            self.swarm_learner.record_episode(trajectory, result.success, insights)

            if self.swarm_learner.should_update_prompts():
                for prompt_path in architect_prompts:
                    try:
                        with open(prompt_path, 'r') as f:
                            current = f.read()
                        updated, changes = self.swarm_learner.update_prompt(prompt_path, current)
                        if changes:
                            logger.info(f"Prompt '{prompt_path}' evolved with {len(changes)} changes")
                    except Exception as e:
                        logger.debug(f"Prompt update skipped for {prompt_path}: {e}")
        except Exception as e:
            logger.debug(f"SwarmLearner recording skipped: {e}")

        # 2. Brain consolidation
        try:
            experience = {
                'content': str(result.output)[:500] if result.output else '',
                'context': {'goal': goal, 'episode': self.episode_count},
                'reward': episode_reward,
                'agent': 'swarm',
            }
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(self.brain_state.process_experience(experience))
            else:
                loop.run_until_complete(self.brain_state.process_experience(experience))
        except Exception as e:
            logger.debug(f"Brain consolidation skipped: {e}")

        # 3. NeuroChunk tiering
        try:
            self.learning_manager.promote_demote_memories(episode_reward)
            self.learning_manager.prune_tier3()
        except Exception as e:
            logger.debug(f"NeuroChunk tiering skipped: {e}")

        # 4. Agent abstractor
        try:
            if hasattr(result, 'agent_contributions') and result.agent_contributions:
                for agent_name, contrib in result.agent_contributions.items():
                    success = getattr(contrib, 'decision_correct', result.success)
                    self.agent_abstractor.update_agent(agent_name, success)
            else:
                agent_name = getattr(result, 'agent_name', agents[0].name if agents else 'unknown')
                self.agent_abstractor.update_agent(agent_name, result.success)
        except Exception as e:
            logger.debug(f"Agent abstractor update skipped: {e}")

        # 5. Transferable learning store
        try:
            query = goal[:200] if goal else ''
            agent_name = getattr(result, 'agent_name', agents[0].name if agents else 'unknown')
            self.transfer_learning.record_experience(
                query=query,
                agent=agent_name,
                action=goal[:100],
                reward=episode_reward,
                success=result.success,
                error=str(getattr(result, 'error', None) or ''),
                context={'episode': self.episode_count},
            )
        except Exception as e:
            logger.debug(f"Transfer learning record skipped: {e}")

        # 6. Swarm intelligence (emergent specialization)
        try:
            task_type = self.transfer_learning.extractor.extract_task_type(goal)
            execution_time = getattr(result, 'execution_time', 0.0)
            agent_name = getattr(result, 'agent_name', agents[0].name if agents else 'unknown')
            self.swarm_intelligence.record_task_result(
                agent_name=agent_name,
                task_type=task_type,
                success=result.success,
                execution_time=execution_time,
                context={'goal': goal[:100], 'episode': self.episode_count},
            )
        except Exception as e:
            logger.debug(f"Swarm intelligence record skipped: {e}")

        # 7. MAS Learning
        try:
            if mas_learning:
                task_type = self.transfer_learning.extractor.extract_task_type(goal) if hasattr(self, 'transfer_learning') else 'general'
                execution_time = getattr(result, 'execution_time', 0.0)

                if hasattr(result, 'agent_contributions') and result.agent_contributions:
                    agent_performances = {}
                    for agent_name, contrib in result.agent_contributions.items():
                        success = getattr(contrib, 'decision_correct', result.success)
                        agent_time = getattr(contrib, 'execution_time', execution_time / len(result.agent_contributions))
                        mas_learning.record_agent_task(
                            agent_type=agent_name,
                            task_type=task_type,
                            success=success,
                            time_taken=agent_time,
                        )
                        agent_performances[agent_name] = {
                            'success': success,
                            'success_rate': 1.0 if success else 0.0,
                            'avg_time': agent_time,
                        }
                else:
                    agent_name = getattr(result, 'agent_name', agents[0].name if agents else 'unknown')
                    mas_learning.record_agent_task(
                        agent_type=agent_name,
                        task_type=task_type,
                        success=result.success,
                        time_taken=execution_time,
                    )
                    agent_performances = {
                        agent_name: {
                            'success': result.success,
                            'success_rate': 1.0 if result.success else 0.0,
                            'avg_time': execution_time,
                        }
                    }

                stigmergy_signals = len(self.swarm_intelligence.stigmergy.signals) if hasattr(self.swarm_intelligence, 'stigmergy') else 0
                mas_learning.record_session(
                    task_description=goal,
                    agent_performances=agent_performances,
                    fixes_applied=getattr(swarm_terminal, '_fix_history', []) if swarm_terminal else [],
                    stigmergy_signals=stigmergy_signals,
                    total_time=execution_time,
                    success=result.success,
                )
        except Exception as e:
            logger.debug(f"MAS Learning record skipped: {e}")

        # 8. Stigmergy: deposit success/failure pheromone signals
        try:
            task_type = self.transfer_learning.extractor.extract_task_type(goal)
            agent_name = getattr(result, 'agent_name', agents[0].name if agents else 'unknown')

            if result.success:
                # Reinforce successful agent-task paths
                self.stigmergy.deposit(
                    signal_type='success',
                    content={'task_type': task_type, 'goal': goal[:100]},
                    agent=agent_name,
                    strength=0.8 + (0.2 * episode_reward),
                )
                # Also deposit a route signal so future routing can use it
                self.stigmergy.deposit(
                    signal_type='route',
                    content={'task_type': task_type, 'agent': agent_name},
                    agent=agent_name,
                    strength=0.7,
                )
            else:
                # Deposit weak warning signal
                self.stigmergy.deposit(
                    signal_type='warning',
                    content={'task_type': task_type, 'goal': goal[:100]},
                    agent=agent_name,
                    strength=0.4,
                )
        except Exception as e:
            logger.debug(f"Stigmergy deposit skipped: {e}")

        # 9. Byzantine verification: check agent claims vs actual results
        try:
            if hasattr(result, 'agent_contributions') and result.agent_contributions:
                for agent_name, contrib in result.agent_contributions.items():
                    claimed = getattr(contrib, 'decision_correct', result.success)
                    self.byzantine_verifier.verify_claim(
                        agent=agent_name,
                        claimed_success=claimed,
                        actual_result=result,
                        task_type=task_type if 'task_type' in dir() else 'general',
                    )
            else:
                agent_name = getattr(result, 'agent_name', agents[0].name if agents else 'unknown')
                self.byzantine_verifier.verify_claim(
                    agent=agent_name,
                    claimed_success=result.success,
                    actual_result=result,
                )
        except Exception as e:
            logger.debug(f"Byzantine verification skipped: {e}")

        # 10. Credit assignment: record which agent/approach deserves credit
        try:
            self.credit_assigner.record_improvement_application(
                improvement={'learned_pattern': goal[:200], 'task': goal[:100]},
                student_score=0.0,
                teacher_score=0.0,
                final_score=episode_reward,
                context={'task': goal[:100], 'episode': self.episode_count},
            )
        except Exception as e:
            logger.debug(f"Credit assignment skipped: {e}")

        # 11. Adaptive learning: adjust learning rate based on score trajectory
        try:
            lr_state = self.adaptive_learning.update_score(episode_reward)
            if lr_state.get('is_plateau'):
                logger.info(
                    f"📈 Adaptive learning: plateau detected "
                    f"(lr={lr_state['learning_rate']:.2f}, "
                    f"explore={lr_state['exploration_rate']:.2f})"
                )
        except Exception as e:
            logger.debug(f"Adaptive learning skipped: {e}")

        # 12. Credit-driven pruning: every 10 episodes, prune low-value learnings
        try:
            if self.episode_count % 10 == 0 and self.episode_count > 0:
                before_count = len(self.transfer_learning.experiences)
                if before_count > 20:
                    # Build improvement-like records from experiences
                    improvements = [
                        {
                            'learned_pattern': exp.get('query', '')[:200],
                            'task': exp.get('action', ''),
                            'timestamp': exp.get('timestamp', ''),
                        }
                        for exp in self.transfer_learning.experiences
                    ]
                    pruned = self.credit_assigner.prune_low_impact_improvements(
                        improvements,
                        min_credit_threshold=0.1,
                        min_application_count=1,
                    )
                    # Map pruned back to experiences
                    pruned_patterns = {imp['learned_pattern'] for imp in pruned}
                    self.transfer_learning.experiences = [
                        exp for exp in self.transfer_learning.experiences
                        if exp.get('query', '')[:200] in pruned_patterns
                        or exp.get('query', '') == ''  # Keep entries without patterns
                    ]
                    after_count = len(self.transfer_learning.experiences)
                    if after_count < before_count:
                        logger.info(
                            f"🧹 Credit pruning: {before_count} → {after_count} "
                            f"experiences (removed {before_count - after_count} low-value)"
                        )
        except Exception as e:
            logger.debug(f"Credit-driven pruning skipped: {e}")

        # 13. Curriculum: queue training tasks when exploration is recommended
        try:
            recommendation = self.adaptive_learning._get_recommendation()
            if recommendation == 'increase_exploration':
                task = self.curriculum_generator.generate_training_task(profiles={})
                self._pending_training_tasks.append(task)
                logger.info(
                    f"🎓 Curriculum: queued training task "
                    f"(difficulty={task.difficulty:.2f}): {task.description[:60]}"
                )
                # Keep queue bounded
                if len(self._pending_training_tasks) > 10:
                    self._pending_training_tasks = self._pending_training_tasks[-10:]
        except Exception as e:
            logger.debug(f"Curriculum queuing skipped: {e}")

        logger.debug(f"Post-episode learning complete (episode #{self.episode_count})")

    def pop_training_task(self):
        """Pop the next queued training task (or None if queue empty)."""
        if self._pending_training_tasks:
            return self._pending_training_tasks.pop(0)
        return None

    def pending_training_count(self) -> int:
        """How many training tasks are waiting."""
        return len(self._pending_training_tasks)

    def learn_from_result(self, result: EpisodeResult, agent_config: AgentConfig, workflow_learner=None):
        """Learn from a successful execution result."""
        if not result.success:
            return
        if not workflow_learner:
            return

        metadata = getattr(agent_config, 'metadata', {}) or {}
        workflow_learner.learn_from_execution(
            task_type=metadata.get('task_type', 'unknown'),
            operations=metadata.get('operations', []),
            tools_used=metadata.get('integrations', []),
            success=True,
            execution_time=getattr(result, 'duration', 0.0),
            metadata={'agent': agent_config.name},
        )

    # =========================================================================
    # Query methods
    # =========================================================================

    def get_transferable_context(self, query: str, agent: str = None) -> str:
        """Get transferable learnings as context string."""
        return self.transfer_learning.get_relevant_context(query, agent=agent)

    def get_swarm_wisdom(self, query: str) -> str:
        """Get swarm intelligence wisdom for a query."""
        try:
            task_type = self.transfer_learning.extractor.extract_task_type(query)
            best = self.swarm_intelligence.get_best_agent_for_task(task_type)
            specs = self.swarm_intelligence.get_specialization_summary()
            parts = []
            if best:
                parts.append(f"Best agent for this task: {best}")
            if specs:
                parts.append(f"Agent specializations: {specs}")
            return "\n".join(parts)
        except Exception as e:
            logger.debug(f"Swarm wisdom unavailable: {e}")
            return ""

    def get_agent_specializations(self) -> Dict[str, str]:
        """Get agent specialization summary."""
        return self.swarm_intelligence.get_specialization_summary()

    def get_best_agent_for_task(self, query: str) -> Optional[str]:
        """Get the best agent for a given task."""
        try:
            task_type = self.transfer_learning.extractor.extract_task_type(query)
            return self.swarm_intelligence.get_best_agent_for_task(task_type)
        except Exception:
            return None

    # =========================================================================
    # Stigmergy queries
    # =========================================================================

    def get_stigmergy_route(self, task_type: str) -> Optional[str]:
        """Use stigmergy pheromone trails to suggest the best agent for a task type."""
        try:
            signals = self.stigmergy.sense(signal_type='route')
            # Filter for this task type and find strongest
            best_agent, best_strength = None, 0.0
            for sig in signals:
                content = sig.content if hasattr(sig, 'content') else {}
                if isinstance(content, dict) and content.get('task_type') == task_type:
                    if sig.strength > best_strength:
                        best_agent = content.get('agent')
                        best_strength = sig.strength
            return best_agent
        except Exception:
            return None

    def get_stigmergy_warnings(self, task_type: str = None) -> list:
        """Get stigmergy warning signals, optionally filtered by task type."""
        try:
            signals = self.stigmergy.sense(signal_type='warning')
            if task_type:
                return [
                    s for s in signals
                    if isinstance(getattr(s, 'content', {}), dict)
                    and s.content.get('task_type') == task_type
                ]
            return signals
        except Exception:
            return []

    # =========================================================================
    # Byzantine trust queries
    # =========================================================================

    def get_agent_trust(self, agent_name: str) -> float:
        """Get trust score for an agent (0.0-1.0). Low = unreliable."""
        try:
            profiles = self.byzantine_verifier.si.agent_profiles
            if agent_name in profiles:
                return profiles[agent_name].trust_score
            return 1.0  # Unknown agents get full trust
        except Exception:
            return 1.0  # Default: trust unknown agents

    def is_agent_trusted(self, agent_name: str, threshold: float = 0.3) -> bool:
        """Check if an agent is trusted above threshold."""
        return self.get_agent_trust(agent_name) >= threshold

    # =========================================================================
    # Credit assignment queries
    # =========================================================================

    def get_credit_stats(self) -> dict:
        """Get credit assignment statistics."""
        return self.credit_assigner.get_credit_statistics()

    # =========================================================================
    # Adaptive learning queries
    # =========================================================================

    def get_learning_state(self) -> dict:
        """Get adaptive learning state: rate, exploration, convergence info."""
        state = self.adaptive_learning.get_state()
        return {
            'learning_rate': state.learning_rate,
            'exploration_rate': state.exploration_rate,
            'is_plateau': state.is_plateau,
            'is_converging': state.is_converging,
            'improvement_velocity': state.improvement_velocity,
            'should_stop': self.adaptive_learning.should_stop_early(),
        }

    # =========================================================================
    # Paradigm effectiveness tracking (auto paradigm selection)
    # =========================================================================

    def record_paradigm_result(self, paradigm: str, success: bool):
        """Record the outcome of a discussion paradigm run."""
        if paradigm not in self._paradigm_stats:
            self._paradigm_stats[paradigm] = {'runs': 0, 'successes': 0}
        self._paradigm_stats[paradigm]['runs'] += 1
        if success:
            self._paradigm_stats[paradigm]['successes'] += 1

    def recommend_paradigm(self, task_type: str = None) -> str:
        """
        Recommend the best discussion paradigm based on historical success rates.

        Uses Thompson Sampling (beta distribution) for explore/exploit:
        - Paradigms with more successes are preferred.
        - Paradigms with few runs get explored.
        - Falls back to 'fanout' if no data yet.

        KISS: ~15 lines, no external deps. DRY: Uses existing _paradigm_stats.
        """
        import random

        best_paradigm = 'fanout'
        best_score = -1.0

        for paradigm, stats in self._paradigm_stats.items():
            runs = stats['runs']
            successes = stats['successes']
            if runs == 0:
                # Unexplored paradigm gets a random draw from uniform prior
                score = random.random()
            else:
                # Thompson sampling: draw from Beta(successes+1, failures+1)
                failures = runs - successes
                score = random.betavariate(successes + 1, failures + 1)

            if score > best_score:
                best_score = score
                best_paradigm = paradigm

        return best_paradigm

    def get_paradigm_stats(self) -> Dict[str, Any]:
        """Get paradigm effectiveness stats with success rates."""
        stats = {}
        for paradigm, data in self._paradigm_stats.items():
            runs = data['runs']
            successes = data['successes']
            stats[paradigm] = {
                **data,
                'success_rate': successes / runs if runs > 0 else None,
            }
        return stats

    # =========================================================================
    # Curriculum generation
    # =========================================================================

    def generate_training_tasks(self, agent_profiles: dict = None, count: int = 3) -> list:
        """Generate synthetic training tasks for agents (DrZero self-curriculum)."""
        try:
            tasks = []
            profiles = agent_profiles or {}
            for _ in range(count):
                task = self.curriculum_generator.generate_training_task(
                    profiles=profiles,
                )
                tasks.append(task)
            return tasks
        except Exception as e:
            logger.debug(f"Curriculum generation failed: {e}")
            return []
