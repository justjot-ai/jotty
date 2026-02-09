"""
SwarmLearningPipeline - Extracted from SwarmManager
====================================================

All learning-related initialization, persistence, and post-episode hooks.
SwarmManager delegates to this class for learning concerns.

Includes EffectivenessTracker: measures whether the system actually
improves over time. Without measurable improvement, "self-improving"
is just a label.
"""

import asyncio
import json
import logging
import time as _time
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from collections import defaultdict, deque

from Jotty.core.foundation.data_structures import JottyConfig, EpisodeResult
from Jotty.core.foundation.agent_config import AgentConfig
from Jotty.core.foundation.robust_parsing import AdaptiveWeightGroup

logger = logging.getLogger(__name__)


# =========================================================================
# EFFECTIVENESS TRACKER - Measures actual improvement over time
# =========================================================================

class EffectivenessTracker:
    """
    Tracks whether the system actually improves over time.

    Compares success rate in the recent window vs. the historical window.
    If recent > historical, the system is genuinely improving.

    KISS: Two deques (recent, historical), one method to record, one to query.
    No LLM calls, no fancy math. Just windowed success rates.

    Usage:
        tracker = EffectivenessTracker()
        tracker.record("analysis", success=True, quality=0.8)
        tracker.record("analysis", success=False, quality=0.2)
        stats = tracker.improvement_report()
        # → {'analysis': {'recent_rate': 0.75, 'historical_rate': 0.5, 'trend': +0.25}}
    """

    def __init__(self, recent_window: int = 20, historical_window: int = 100):
        """
        Args:
            recent_window: Number of recent episodes to consider "current"
            historical_window: Number of older episodes for baseline comparison
        """
        self.recent_window = recent_window
        self.historical_window = historical_window

        # Per task_type: deque of (timestamp, success: bool, quality: float)
        self._records: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=recent_window + historical_window)
        )

        # Global (all task types combined)
        self._global: deque = deque(maxlen=recent_window + historical_window)

    def record(self, task_type: str, success: bool, quality: float = 0.0,
               agent: str = ""):
        """Record a task outcome. Call after every execution."""
        entry = (_time.time(), success, max(0.0, min(1.0, quality)), agent)
        self._records[task_type].append(entry)
        self._global.append(entry)

    def _split_windows(self, records: deque) -> Tuple[List, List]:
        """Split records into recent and historical windows."""
        items = list(records)
        if len(items) <= self.recent_window:
            return items, []
        recent = items[-self.recent_window:]
        historical = items[:-self.recent_window]
        return recent, historical

    def _rate(self, records: List) -> Tuple[float, float]:
        """Compute (success_rate, avg_quality) from record list."""
        if not records:
            return 0.0, 0.0
        successes = sum(1 for _, s, _, _ in records if s)
        avg_quality = sum(q for _, _, q, _ in records) / len(records)
        return successes / len(records), avg_quality

    def improvement_report(self) -> Dict[str, Any]:
        """
        Get improvement report across all task types.

        Returns dict with per-task-type and global trends.
        A positive 'trend' means the system is improving.
        """
        report = {}

        for task_type, records in self._records.items():
            recent, historical = self._split_windows(records)
            recent_rate, recent_quality = self._rate(recent)
            hist_rate, hist_quality = self._rate(historical)

            report[task_type] = {
                'recent_success_rate': round(recent_rate, 3),
                'historical_success_rate': round(hist_rate, 3),
                'trend': round(recent_rate - hist_rate, 3),
                'recent_quality': round(recent_quality, 3),
                'historical_quality': round(hist_quality, 3),
                'quality_trend': round(recent_quality - hist_quality, 3),
                'total_episodes': len(records),
                'improving': recent_rate > hist_rate and len(historical) >= 5,
            }

        # Global stats
        recent, historical = self._split_windows(self._global)
        recent_rate, recent_quality = self._rate(recent)
        hist_rate, hist_quality = self._rate(historical)

        report['_global'] = {
            'recent_success_rate': round(recent_rate, 3),
            'historical_success_rate': round(hist_rate, 3),
            'trend': round(recent_rate - hist_rate, 3),
            'recent_quality': round(recent_quality, 3),
            'total_episodes': len(self._global),
            'improving': recent_rate > hist_rate and len(historical) >= 5,
        }

        return report

    def is_improving(self, task_type: str = None) -> bool:
        """Quick check: is the system improving for a given task type (or globally)?"""
        report = self.improvement_report()
        key = task_type or '_global'
        return report.get(key, {}).get('improving', False)

    def to_dict(self) -> Dict:
        """Serialize for persistence."""
        return {
            task_type: [
                {'t': t, 's': s, 'q': q, 'a': a}
                for t, s, q, a in records
            ]
            for task_type, records in self._records.items()
        }

    @classmethod
    def from_dict(cls, data: Dict, recent_window: int = 20,
                  historical_window: int = 100) -> 'EffectivenessTracker':
        """Deserialize from persistence."""
        tracker = cls(recent_window, historical_window)
        for task_type, entries in data.items():
            for e in entries:
                entry = (e.get('t', 0), e.get('s', False), e.get('q', 0.0), e.get('a', ''))
                tracker._records[task_type].append(entry)
                tracker._global.append(entry)
        return tracker


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
        # UNIFIED: LP owns the canonical StigmergyLayer. SI gets a reference
        # to the same instance so writes and reads go through one store.
        self.stigmergy = StigmergyLayer(decay_rate=0.1, max_signals=500)
        self.swarm_intelligence.stigmergy = self.stigmergy  # unify — no duplicate

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
        # Keyed by task_type → paradigm → {runs, successes}.
        self._paradigm_stats: Dict[str, Dict[str, Dict[str, int]]] = {}

        # Effectiveness tracker: measures whether system actually improves.
        # This is what makes "self-improving" a verifiable claim, not a label.
        self.effectiveness = EffectivenessTracker(
            recent_window=20, historical_window=100
        )

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

        # Stigmergy pheromone trails + paradigm stats + effectiveness data + episode count
        stig_path = self._get_stigmergy_path()
        if stig_path.exists():
            try:
                from .stigmergy import StigmergyLayer
                with open(stig_path, 'r') as f:
                    stig_data = json.load(f)
                self.stigmergy = StigmergyLayer.from_dict(stig_data)
                # Re-unify: SI must point to the same loaded instance
                self.swarm_intelligence.stigmergy = self.stigmergy
                # Restore paradigm stats if present (saved alongside stigmergy)
                if 'paradigm_stats' in stig_data:
                    loaded = stig_data['paradigm_stats']
                    # Backward compat: old format was {paradigm: {runs, successes}}
                    # New format is {task_type: {paradigm: {runs, successes}}}
                    if loaded and isinstance(next(iter(loaded.values()), None), dict):
                        first_val = next(iter(loaded.values()))
                        if 'runs' in first_val:
                            loaded = {'_global': loaded}
                    self._paradigm_stats.update(loaded)
                # Restore effectiveness tracker
                if 'effectiveness' in stig_data:
                    self.effectiveness = EffectivenessTracker.from_dict(
                        stig_data['effectiveness']
                    )
                # Restore episode count (cumulative across sessions)
                if 'episode_count' in stig_data:
                    self.episode_count = stig_data['episode_count']
                logger.info(
                    f"Auto-loaded stigmergy: {len(self.stigmergy.signals)} signals, "
                    f"effectiveness: {self.effectiveness._global.__len__()} records"
                )
            except Exception as e:
                logger.warning(f"Could not auto-load stigmergy: {e}")

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

        # Stigmergy pheromone trails + paradigm stats + effectiveness data + episode count
        stig_path = self._get_stigmergy_path()
        try:
            stig_path.parent.mkdir(parents=True, exist_ok=True)
            stig_data = self.stigmergy.to_dict()
            stig_data['paradigm_stats'] = self._paradigm_stats
            stig_data['effectiveness'] = self.effectiveness.to_dict()
            stig_data['episode_count'] = self.episode_count
            with open(stig_path, 'w') as f:
                json.dump(stig_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not auto-save stigmergy: {e}")

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
    # REWARD COMPUTATION — continuous signal instead of binary {0, 1}
    # =========================================================================

    @staticmethod
    def _compute_episode_reward(result, goal: str) -> float:
        """
        Compute a continuous reward in [0, 1] from the execution result.

        The old approach: 1.0 if result.success else 0.0
        This is binary — TD-Lambda has no gradient to follow.

        New approach: multi-dimensional heuristic decomposition.
        Each dimension is cheap to compute (no LLM call) and provides
        real gradient information for the learning algorithms.

        Dimensions (weighted average):
          substance  (0.30): Is the output actually substantive?
          efficiency (0.15): How fast relative to a 120s baseline?
          tool_use   (0.15): Did tool calls produce useful output?
          structure  (0.15): Does output have organized structure?
          no_errors  (0.25): Absence of error indicators in output

        The success flag is a floor: if the agent reports failure,
        the max reward is capped at 0.3 (allowing partial credit for
        useful partial output on failures).
        """
        # Extract output text
        output_text = ""
        if hasattr(result, 'output') and result.output:
            output_text = str(result.output)
        elif isinstance(result, dict):
            output_text = str(result.get('output', ''))
        output_text = output_text.strip()

        success = bool(getattr(result, 'success', False))

        # --- Dimension 1: Substance (0-1) ---
        # Longer, more substantive output scores higher, with diminishing returns.
        # Empty = 0, 100 chars = 0.3, 500 = 0.6, 1000+ = 0.85, 3000+ = 1.0
        char_count = len(output_text)
        if char_count == 0:
            substance = 0.0
        elif char_count < 50:
            substance = 0.1
        elif char_count < 200:
            substance = 0.3
        elif char_count < 500:
            substance = 0.5
        elif char_count < 1000:
            substance = 0.7
        elif char_count < 3000:
            substance = 0.85
        else:
            substance = 1.0

        # --- Dimension 2: Efficiency (0-1) ---
        # Faster execution relative to 120s baseline scores higher.
        exec_time = getattr(result, 'execution_time', 60.0)
        if exec_time <= 0:
            exec_time = 60.0  # Unknown defaults to neutral
        # Sigmoid-like: 5s → 0.95, 30s → 0.75, 60s → 0.5, 120s → 0.25, 300s → 0.05
        import math
        efficiency = 1.0 / (1.0 + math.exp((exec_time - 60) / 30))

        # --- Dimension 3: Tool usage effectiveness (0-1) ---
        # If the trajectory shows tool calls, check how many produced output.
        trajectory = getattr(result, 'trajectory', []) or []
        tool_calls = 0
        tool_successes = 0
        for step in trajectory:
            if isinstance(step, dict):
                action = step.get('action', '')
                if 'tool' in str(action).lower() or step.get('tool_name'):
                    tool_calls += 1
                    step_output = step.get('output', step.get('result', ''))
                    if step_output and len(str(step_output)) > 10:
                        tool_successes += 1
        if tool_calls > 0:
            tool_use = tool_successes / tool_calls
        else:
            tool_use = 0.5  # No tools used — neutral

        # --- Dimension 4: Structure (0-1) ---
        # Does the output show organized thinking? Check for structure markers.
        structure_signals = 0
        lower = output_text.lower()[:3000]
        if any(m in lower for m in ['\n#', '\n##', '\n###']):
            structure_signals += 1  # Headings
        if any(m in lower for m in ['\n- ', '\n* ', '\n1.', '\n2.']):
            structure_signals += 1  # Lists
        if '```' in output_text:
            structure_signals += 1  # Code blocks
        if any(m in lower for m in ['in conclusion', 'summary', 'therefore', 'key finding']):
            structure_signals += 1  # Conclusions
        structure = min(1.0, structure_signals / 3.0)

        # --- Dimension 5: Error absence (0-1) ---
        # Penalize outputs containing error indicators.
        error_indicators = [
            'error:', 'exception:', 'traceback', 'failed to',
            'could not', 'unable to', 'i cannot', "i can't",
            'not supported', 'invalid', 'timeout',
        ]
        error_count = sum(1 for ind in error_indicators if ind in lower)
        no_errors = max(0.0, 1.0 - error_count * 0.25)

        # --- Weighted combination ---
        reward = (
            0.30 * substance +
            0.15 * efficiency +
            0.15 * tool_use +
            0.15 * structure +
            0.25 * no_errors
        )

        # Success floor/ceiling: failure caps at 0.3, success has no cap
        if not success:
            reward = min(reward, 0.3)

        # Clamp to [0, 1]
        reward = max(0.0, min(1.0, reward))

        logger.debug(
            f"Episode reward: {reward:.3f} "
            f"(substance={substance:.2f}, efficiency={efficiency:.2f}, "
            f"tool_use={tool_use:.2f}, structure={structure:.2f}, "
            f"no_errors={no_errors:.2f}, success={success})"
        )

        return reward

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
        episode_reward = self._compute_episode_reward(result, goal)

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
            logger.warning(f"SwarmLearner recording failed: {e}")

        # 2. Brain consolidation (fire-and-forget in running loop)
        try:
            experience = {
                'content': str(result.output)[:500] if result.output else '',
                'context': {'goal': goal, 'episode': self.episode_count},
                'reward': episode_reward,
                'agent': 'swarm',
            }
            try:
                asyncio.get_running_loop()
                # We're inside a running event loop — schedule as background task
                asyncio.ensure_future(self.brain_state.process_experience(experience))
            except RuntimeError:
                pass  # No running loop — skip async consolidation
        except Exception as e:
            logger.warning(f"Brain consolidation failed: {e}")

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
            logger.warning(f"Swarm intelligence record failed: {e}")

        # 6b. Stigmergy: deposit outcome signal for agent routing
        try:
            task_type = self.transfer_learning.extractor.extract_task_type(goal)
            agent_name = getattr(result, 'agent_name', agents[0].name if agents else 'unknown')
            self.stigmergy.record_outcome(
                agent=agent_name,
                task_type=task_type,
                success=result.success,
                quality=episode_reward,
            )
        except Exception as e:
            logger.warning(f"Stigmergy outcome recording failed: {e}")

        # 6b-ii. Stigmergy APPROACH tracking (useful in single-agent mode).
        #        Extract which tools/skills were used from the trajectory,
        #        then record approach outcome so future executions of the same
        #        task type get actionable "use X, avoid Y" guidance.
        try:
            task_type = self.transfer_learning.extractor.extract_task_type(goal)
            agent_name = getattr(result, 'agent_name', agents[0].name if agents else 'unknown')
            trajectory = getattr(result, 'trajectory', []) or []

            # Extract tools/skills used from trajectory AND from output
            tools_used = []
            for step in trajectory:
                if isinstance(step, dict):
                    tool = step.get('skill') or step.get('tool_name') or step.get('tool', '')
                    if tool and tool not in tools_used:
                        tools_used.append(str(tool))
            # Also check output's skills_used (autonomous agent populates this)
            _out = getattr(result, 'output', None)
            if not tools_used:
                if hasattr(_out, 'skills_used') and _out.skills_used:
                    tools_used = list(_out.skills_used)
                elif isinstance(_out, dict):
                    tools_used = list(_out.get('skills_used', []))

            # Build approach summary from skill sequence (much more useful than 'execute')
            skill_steps = []
            for step in trajectory[:8]:
                if isinstance(step, dict):
                    skill = step.get('skill') or step.get('tool_name', '')
                    desc = step.get('description', step.get('action', ''))
                    if skill:
                        skill_steps.append(f"{skill}: {str(desc)[:40]}" if desc else skill)
            if skill_steps:
                approach_summary = ' → '.join(skill_steps)
            elif tools_used:
                approach_summary = f"Used: {', '.join(tools_used[:5])}"
            else:
                approach_summary = goal[:100]

            self.stigmergy.record_approach_outcome(
                task_type=task_type,
                approach_summary=approach_summary,
                tools_used=tools_used,
                success=result.success,
                quality=episode_reward,
                agent=agent_name,
            )
        except Exception as e:
            logger.warning(f"Stigmergy approach recording failed: {e}")

        # 6c. Effectiveness tracker: measure actual improvement over time
        try:
            task_type = self.transfer_learning.extractor.extract_task_type(goal)
            agent_name = getattr(result, 'agent_name', agents[0].name if agents else 'unknown')
            self.effectiveness.record(
                task_type=task_type,
                success=result.success,
                quality=episode_reward,
                agent=agent_name,
            )
        except Exception as e:
            logger.debug(f"Effectiveness tracking skipped: {e}")

        # 7. MAS Learning (session recording — MASLearning.record_session)
        try:
            if mas_learning:
                execution_time = getattr(result, 'execution_time', 0.0)

                # Collect agent names used
                agents_used = []
                if hasattr(result, 'agent_contributions') and result.agent_contributions:
                    agents_used = list(result.agent_contributions.keys())
                else:
                    agent_name = getattr(result, 'agent_name', agents[0].name if agents else 'unknown')
                    agents_used = [agent_name]

                stigmergy_signals = len(self.swarm_intelligence.stigmergy.signals) if hasattr(self.swarm_intelligence, 'stigmergy') else 0
                mas_learning.record_session(
                    task_description=goal,
                    agents_used=agents_used,
                    total_time=execution_time,
                    success=result.success,
                    stigmergy_signals=stigmergy_signals,
                    output_quality=episode_reward,
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
            logger.warning(f"Stigmergy deposit failed: {e}")

        # 9. Byzantine verification: quality check + claim verification
        #    Single-agent: verify_output_quality (heuristic quality check)
        #    Multi-agent: verify_claim (cross-agent consistency)
        try:
            task_type_for_byz = task_type if 'task_type' in dir() else 'general'

            if hasattr(result, 'agent_contributions') and result.agent_contributions:
                # Multi-agent: check each agent's claim
                for agent_name, contrib in result.agent_contributions.items():
                    claimed = getattr(contrib, 'decision_correct', result.success)
                    self.byzantine_verifier.verify_claim(
                        agent=agent_name,
                        claimed_success=claimed,
                        actual_result=result,
                        task_type=task_type_for_byz,
                    )
            else:
                # Single-agent: use quality verification (not meaningless self-check)
                agent_name = getattr(result, 'agent_name', agents[0].name if agents else 'unknown')
                quality = self.byzantine_verifier.verify_output_quality(
                    agent=agent_name,
                    claimed_success=result.success,
                    output=result,
                    goal=goal,
                    task_type=task_type_for_byz,
                )
                if not quality['quality_ok']:
                    logger.warning(
                        f"Byzantine quality issues for {agent_name}: "
                        f"{quality['issues']}"
                    )
        except Exception as e:
            logger.warning(f"Byzantine verification failed: {e}")

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

    def learn_from_result(self, result: EpisodeResult, agent_config: AgentConfig, workflow_learner=None, goal: str = ''):
        """Learn from a successful execution result."""
        if not result.success:
            return
        if not workflow_learner:
            return

        # Extract task_type: prefer goal text (most reliable), then result fields
        from Jotty.core.learning.transfer_learning import PatternExtractor
        task_type = 'unknown'
        if goal:
            try:
                task_type = self.transfer_learning.extractor.extract_task_type(goal)
            except Exception:
                pass
        if task_type == 'unknown' and hasattr(result, 'task_type') and result.task_type:
            raw = result.task_type.value if hasattr(result.task_type, 'value') else str(result.task_type)
            task_type = PatternExtractor.normalize_task_type(raw)
        if task_type == 'unknown' and hasattr(result, 'task') and result.task:
            try:
                task_type = self.transfer_learning.extractor.extract_task_type(str(result.task))
            except Exception:
                pass
        # Always normalize (creation→generation, research→analysis, etc.)
        task_type = PatternExtractor.normalize_task_type(task_type)

        # Extract skills/tools actually used from result or its output
        skills_used = []
        if hasattr(result, 'skills_used') and result.skills_used:
            skills_used = list(result.skills_used)
        if not skills_used and hasattr(result, 'output'):
            _out = result.output
            if isinstance(_out, dict):
                skills_used = list(_out.get('skills_used', []))
            elif hasattr(_out, 'skills_used') and _out.skills_used:
                skills_used = list(_out.skills_used)

        # Extract operations from trajectory steps
        operations = []
        trajectory = getattr(result, 'trajectory', []) or []
        for step in trajectory[:10]:
            if isinstance(step, dict):
                action = step.get('skill') or step.get('action') or step.get('tool_name') or step.get('step', '')
                if action:
                    operations.append(str(action))
            elif step:
                operations.append(str(step))
        # Also extract from output.steps if available (autonomous agent execution results)
        if not operations and hasattr(result, 'output'):
            _out = result.output
            _steps = getattr(_out, 'steps', None) or (isinstance(_out, dict) and _out.get('steps'))
            if _steps and isinstance(_steps, (list, tuple)):
                for s in _steps[:10]:
                    if isinstance(s, dict):
                        skill = s.get('skill', s.get('action', ''))
                        if skill:
                            operations.append(str(skill))

        logger.info(
            f"📝 Workflow learning: task_type={task_type}, "
            f"operations={operations[:5]}, tools={skills_used[:5]}"
        )
        workflow_learner.learn_from_execution(
            task_type=task_type,
            operations=operations,
            tools_used=skills_used,
            success=True,
            execution_time=getattr(result, 'execution_time', 0.0),
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

    def get_effectiveness_report(self) -> dict:
        """
        Get measurable improvement data.

        Returns per-task-type success rate trends (recent vs historical).
        A positive 'trend' value means the system is genuinely improving.
        """
        return self.effectiveness.improvement_report()

    # =========================================================================
    # Paradigm effectiveness tracking (auto paradigm selection)
    # =========================================================================

    _PARADIGMS = ('fanout', 'relay', 'debate', 'refinement')

    def _ensure_paradigm_bucket(self, task_type: str):
        """Lazy-create the stats bucket for a task_type."""
        if task_type not in self._paradigm_stats:
            self._paradigm_stats[task_type] = {
                p: {'runs': 0, 'successes': 0} for p in self._PARADIGMS
            }

    def record_paradigm_result(
        self, paradigm: str, success: bool, task_type: str = '_global',
    ):
        """Record the outcome of a discussion paradigm run for a task type."""
        self._ensure_paradigm_bucket(task_type)
        bucket = self._paradigm_stats[task_type]
        if paradigm not in bucket:
            bucket[paradigm] = {'runs': 0, 'successes': 0}
        bucket[paradigm]['runs'] += 1
        if success:
            bucket[paradigm]['successes'] += 1

        # Also update _global so there's always a fallback
        if task_type != '_global':
            self.record_paradigm_result(paradigm, success, '_global')

    def recommend_paradigm(self, task_type: str = None) -> str:
        """
        Recommend the best discussion paradigm based on historical success rates.

        Per-task-type: "debate is great for analysis, relay is great for writing."
        Falls back to _global stats when task_type has < 5 data points.

        Uses Thompson Sampling (Beta distribution) for explore/exploit:
        - Paradigms with more successes are preferred.
        - Paradigms with few runs get explored.
        - Falls back to 'fanout' if no data yet.

        KISS: ~25 lines, no external deps. DRY: Uses existing _paradigm_stats.
        """
        import random

        # Pick the best bucket: task-specific if enough data, else global
        bucket = None
        key = task_type or '_global'
        if key in self._paradigm_stats:
            total_runs = sum(
                s['runs'] for s in self._paradigm_stats[key].values()
            )
            if total_runs >= 5:
                bucket = self._paradigm_stats[key]

        if bucket is None and '_global' in self._paradigm_stats:
            bucket = self._paradigm_stats['_global']

        if not bucket:
            # No data at all — return random paradigm
            return random.choice(list(self._PARADIGMS))

        best_paradigm = 'fanout'
        best_score = -1.0

        for paradigm in self._PARADIGMS:
            stats = bucket.get(paradigm, {'runs': 0, 'successes': 0})
            runs = stats['runs']
            successes = stats['successes']
            if runs == 0:
                score = random.random()
            else:
                failures = runs - successes
                score = random.betavariate(successes + 1, failures + 1)

            if score > best_score:
                best_score = score
                best_paradigm = paradigm

        return best_paradigm

    def get_paradigm_stats(self, task_type: str = None) -> Dict[str, Any]:
        """
        Get paradigm effectiveness stats with success rates.

        Args:
            task_type: If provided, return stats for that task type.
                       If None, return _global stats with per-type breakdown.
        """
        def _format_bucket(bucket):
            out = {}
            for paradigm, data in bucket.items():
                runs = data['runs']
                successes = data['successes']
                out[paradigm] = {
                    **data,
                    'success_rate': successes / runs if runs > 0 else None,
                }
            return out

        if task_type:
            bucket = self._paradigm_stats.get(task_type, {})
            return _format_bucket(bucket)

        # Return global + per-type summary
        result = {}
        for tt, bucket in self._paradigm_stats.items():
            result[tt] = _format_bucket(bucket)
        return result

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
