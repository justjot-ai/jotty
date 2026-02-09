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

        logger.debug(f"Post-episode learning complete (episode #{self.episode_count})")

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
