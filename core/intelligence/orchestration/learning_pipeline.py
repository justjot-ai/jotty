"""
SwarmLearningPipeline - Extracted from Orchestrator
====================================================

All learning-related initialization, persistence, and post-episode hooks.
Orchestrator delegates to this class for learning concerns.

Includes EffectivenessTracker: measures whether the system actually
improves over time. Without measurable improvement, "self-improving"
is just a label.
"""

import asyncio
import json
import logging
import time as _time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from Jotty.core.infrastructure.foundation.agent_config import AgentConfig
from Jotty.core.infrastructure.foundation.data_structures import (
    EpisodeResult,
    SwarmConfig,
    SwarmLearningConfig,
)
from Jotty.core.infrastructure.foundation.robust_parsing import AdaptiveWeightGroup

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

    def __init__(self, recent_window: int = 20, historical_window: int = 100) -> None:
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

    def record(self, task_type: str, success: bool, quality: float = 0.0, agent: str = "") -> Any:
        """Record a task outcome. Call after every execution."""
        entry = (_time.time(), success, max(0.0, min(1.0, quality)), agent)
        self._records[task_type].append(entry)
        self._global.append(entry)

    def _split_windows(self, records: deque) -> Tuple[List, List]:
        """Split records into recent and historical windows."""
        items = list(records)
        if len(items) <= self.recent_window:
            return items, []
        recent = items[-self.recent_window :]
        historical = items[: -self.recent_window]
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
                "recent_success_rate": round(recent_rate, 3),
                "historical_success_rate": round(hist_rate, 3),
                "trend": round(recent_rate - hist_rate, 3),
                "recent_quality": round(recent_quality, 3),
                "historical_quality": round(hist_quality, 3),
                "quality_trend": round(recent_quality - hist_quality, 3),
                "total_episodes": len(records),
                "improving": recent_rate > hist_rate and len(historical) >= 5,
            }

        # Global stats
        recent, historical = self._split_windows(self._global)
        recent_rate, recent_quality = self._rate(recent)
        hist_rate, hist_quality = self._rate(historical)

        report["_global"] = {
            "recent_success_rate": round(recent_rate, 3),
            "historical_success_rate": round(hist_rate, 3),
            "trend": round(recent_rate - hist_rate, 3),
            "recent_quality": round(recent_quality, 3),
            "total_episodes": len(self._global),
            "improving": recent_rate > hist_rate and len(historical) >= 5,
        }

        return report

    def is_improving(self, task_type: str = None) -> bool:
        """Quick check: is the system improving for a given task type (or globally)?"""
        report = self.improvement_report()
        key = task_type or "_global"
        return report.get(key, {}).get("improving", False)

    def to_dict(self) -> Dict:
        """Serialize for persistence."""
        return {
            task_type: [{"t": t, "s": s, "q": q, "a": a} for t, s, q, a in records]
            for task_type, records in self._records.items()
        }

    @classmethod
    def from_dict(
        cls, data: Dict, recent_window: int = 20, historical_window: int = 100
    ) -> "EffectivenessTracker":
        """Deserialize from persistence."""
        tracker = cls(recent_window, historical_window)
        for task_type, entries in data.items():
            for e in entries:
                entry = (e.get("t", 0), e.get("s", False), e.get("q", 0.0), e.get("a", ""))
                tracker._records[task_type].append(entry)
                tracker._global.append(entry)
        return tracker


class SwarmLearningPipeline:
    """
    Manages all learning components: RL, credit assignment, memory consolidation,
    transferable learning, swarm intelligence, and MAS learning.

    Extracted from Orchestrator to reduce god-object coupling.
    """

    def __init__(self, config: SwarmConfig) -> None:
        self.config = config
        self.episode_count = 0
        self._init_components()

    def _init_components(self) -> Any:
        """Initialize all learning components."""
        from Jotty.core.intelligence.learning.learning_coordinator import (
            LearningManager as LearningManager,
        )
        from Jotty.core.intelligence.learning.predictive_marl import (
            DivergenceMemory,
            LLMTrajectoryPredictor,
        )
        from Jotty.core.intelligence.learning.transfer_learning import TransferableLearningStore
        from Jotty.core.intelligence.memory.consolidation_engine import (
            AgentAbstractor,
            BrainModeConfig,
            BrainStateMachine,
        )
        from Jotty.core.intelligence.orchestration.swarm_learner import SwarmLearner
        from Jotty.core.modes.agent.axon import SmartAgentSlack
        from Jotty.core.modes.agent.feedback_channel import FeedbackChannel

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
        self.credit_weights = AdaptiveWeightGroup(
            {
                "base_reward": 0.3,
                "cooperation_bonus": 0.4,
                "predictability_bonus": 0.3,
            }
        )

        # -----------------------------------------------------------------
        # Previously dormant modules — now wired (DRY: init once, use many)
        # -----------------------------------------------------------------
        from .adaptive_learning import AdaptiveLearning
        from .byzantine_verification import ByzantineVerifier
        from .credit_assignment import CreditAssignment
        from .curriculum_generator import CurriculumGenerator
        from .stigmergy import StigmergyLayer

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
            memory_system=None,  # Wired later if SwarmMemory available
        )

        # Training task queue (filled by post_episode when exploration needed)
        self._pending_training_tasks: list = []

        # Paradigm effectiveness tracker (auto paradigm selection)
        # Keyed by task_type → paradigm → {runs, successes}.
        self._paradigm_stats: Dict[str, Dict[str, Dict[str, int]]] = {}

        # Effectiveness tracker: measures whether system actually improves.
        # This is what makes "self-improving" a verifiable claim, not a label.
        self.effectiveness = EffectivenessTracker(recent_window=20, historical_window=100)

        # TD(λ) learner with HRPO grouped baselines (was implemented but never wired)
        from Jotty.core.intelligence.learning.adaptive_components import AdaptiveLearningRate
        from Jotty.core.intelligence.learning.td_lambda import TDLambdaLearner

        self._adaptive_lr = AdaptiveLearningRate(self.config)
        self.td_learner = TDLambdaLearner(self.config, adaptive_lr=self._adaptive_lr)

    # =========================================================================
    # Persistence paths
    # =========================================================================

    def _get_learning_path(self) -> Path:
        base = getattr(self.config, "base_path", None)
        if base:
            return Path(base) / "swarm_learnings.json"
        return Path.home() / ".jotty" / "swarm_learnings.json"

    def _get_transfer_learning_path(self) -> Path:
        base = getattr(self.config, "base_path", None)
        if base:
            return Path(base) / "transfer_learnings.json"
        return Path.home() / ".jotty" / "transfer_learnings.json"

    def _get_swarm_intelligence_path(self) -> Path:
        base = getattr(self.config, "base_path", None)
        if base:
            return Path(base) / "swarm_intelligence.json"
        return Path.home() / ".jotty" / "swarm_intelligence.json"

    def _get_credit_weights_path(self) -> Path:
        base = getattr(self.config, "base_path", None)
        if base:
            return Path(base) / "credit_weights.json"
        return Path.home() / ".jotty" / "credit_weights.json"

    def _get_stigmergy_path(self) -> Path:
        base = getattr(self.config, "base_path", None)
        if base:
            return Path(base) / "stigmergy.json"
        return Path.home() / ".jotty" / "stigmergy.json"

    def _get_td_lambda_path(self) -> Path:
        base = getattr(self.config, "base_path", None)
        if base:
            return Path(base) / "td_lambda.json"
        return Path.home() / ".jotty" / "td_lambda.json"

    # =========================================================================
    # CHECKPOINTS — snapshot/restore learning state (Cline-inspired)
    # =========================================================================

    def save_checkpoint(self, label: str = "") -> str:
        """
        Save a snapshot of all learning state files.

        Returns the checkpoint directory path. Use restore_checkpoint()
        with this path to roll back if training degrades performance.
        """
        import shutil
        import time as _t

        ts = int(_t.time())
        tag = f"_{label}" if label else ""
        base = Path(getattr(self.config, "base_path", None) or (Path.home() / ".jotty"))
        checkpoint_dir = base / "checkpoints" / f"cp_{ts}{tag}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        files_to_backup = [
            self._get_learning_path(),
            self._get_transfer_learning_path(),
            self._get_swarm_intelligence_path(),
            self._get_credit_weights_path(),
            self._get_stigmergy_path(),
            self._get_td_lambda_path(),
        ]

        copied = 0
        for src in files_to_backup:
            if src.exists():
                shutil.copy2(src, checkpoint_dir / src.name)
                copied += 1

        logger.info(f"Checkpoint saved: {checkpoint_dir} ({copied} files)")
        return str(checkpoint_dir)

    def restore_checkpoint(self, checkpoint_dir: str) -> int:
        """
        Restore learning state from a checkpoint.

        Args:
            checkpoint_dir: Path returned by save_checkpoint()

        Returns:
            Number of files restored
        """
        import shutil

        cp_path = Path(checkpoint_dir)
        if not cp_path.is_dir():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")

        targets = {
            "swarm_learnings.json": self._get_learning_path(),
            "transfer_learnings.json": self._get_transfer_learning_path(),
            "swarm_intelligence.json": self._get_swarm_intelligence_path(),
            "credit_weights.json": self._get_credit_weights_path(),
            "stigmergy.json": self._get_stigmergy_path(),
            "td_lambda.json": self._get_td_lambda_path(),
        }

        restored = 0
        for filename, dest in targets.items():
            src = cp_path / filename
            if src.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
                restored += 1

        # Reload the restored state into live objects
        self.auto_load()

        logger.info(f"Checkpoint restored: {checkpoint_dir} ({restored} files)")
        return restored

    def list_checkpoints(self) -> list:
        """List available checkpoints, newest first."""
        base = Path(getattr(self.config, "base_path", None) or (Path.home() / ".jotty"))
        cp_base = base / "checkpoints"
        if not cp_base.exists():
            return []
        dirs = sorted(cp_base.iterdir(), reverse=True)
        return [str(d) for d in dirs if d.is_dir()]

    # =========================================================================
    # Versioned JSON I/O
    # =========================================================================

    _SCHEMA_VERSION = "2.0"
    _MIGRATIONS: Dict[Tuple[str, str], Any] = {}  # {("old_major", "new_major"): migrate_fn}

    @staticmethod
    def _migrate(data: dict, from_major: str, to_major: str) -> Optional[dict]:
        """Attempt to migrate data between major schema versions."""
        fn = SwarmLearningPipeline._MIGRATIONS.get((from_major, to_major))
        if fn:
            return fn(data)
        return None

    def _save_versioned(self, path: Path, data: dict) -> None:
        """Save data as versioned JSON: {"schema_version": "2.0", "data": {...}}."""
        path.parent.mkdir(parents=True, exist_ok=True)
        envelope = {"schema_version": self._SCHEMA_VERSION, "data": data}
        with open(path, "w") as f:
            json.dump(envelope, f, indent=2)

    def _load_versioned(self, path: Path) -> dict:
        """Load versioned JSON; handles both enveloped and legacy bare-dict formats.

        Returns the data dict (unwrapped from envelope).
        Logs a warning if the major version differs.
        """
        with open(path, "r") as f:
            raw = json.load(f)

        # New envelope format
        if isinstance(raw, dict) and "schema_version" in raw and "data" in raw:
            file_ver = str(raw["schema_version"])
            file_major = file_ver.split(".")[0]
            expected_major = self._SCHEMA_VERSION.split(".")[0]
            if file_major != expected_major:
                logger.warning(
                    f"Schema version mismatch in {path.name}: "
                    f"file={file_ver}, expected={self._SCHEMA_VERSION}"
                )
                # Attempt migration
                migrated = self._migrate(raw["data"], file_major, expected_major)
                if migrated is not None:
                    logger.info(f"Migrated {path.name} from v{file_ver} to v{self._SCHEMA_VERSION}")
                    return migrated
                # No migration path — return empty to avoid loading incompatible data
                return {}
            return raw["data"]

        # Legacy bare-dict format (pre-versioning)
        return raw

    # =========================================================================
    # Auto-load / Auto-save
    # =========================================================================

    def auto_load(self) -> None:
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

        # Adaptive credit weights (versioned)
        credit_path = self._get_credit_weights_path()
        if credit_path.exists():
            try:
                credit_data = self._load_versioned(credit_path)
                self.credit_weights = AdaptiveWeightGroup.from_dict(credit_data)
                logger.info(f"Auto-loaded credit weights: {self.credit_weights}")
            except Exception as e:
                logger.debug(f"Could not auto-load credit weights: {e}")

        # TD-Lambda grouped baselines (versioned)
        td_path = self._get_td_lambda_path()
        if td_path.exists():
            try:
                from Jotty.core.intelligence.learning.td_lambda import GroupedValueBaseline

                td_data = self._load_versioned(td_path)
                self.td_learner.grouped_baseline = GroupedValueBaseline.from_dict(
                    td_data, config=self.config
                )
                logger.info(
                    f"Auto-loaded TD-Lambda: "
                    f"{td_data.get('group_counts', {}).__len__()} group baselines"
                )
            except Exception as e:
                logger.debug(f"Could not auto-load TD-Lambda state: {e}")

        # Stigmergy pheromone trails + paradigm stats + effectiveness data + episode count (versioned)
        stig_path = self._get_stigmergy_path()
        if stig_path.exists():
            try:
                from .stigmergy import StigmergyLayer

                stig_data = self._load_versioned(stig_path)
                self.stigmergy = StigmergyLayer.from_dict(stig_data)
                # Re-unify: SI must point to the same loaded instance
                self.swarm_intelligence.stigmergy = self.stigmergy
                # Restore paradigm stats if present (saved alongside stigmergy)
                if "paradigm_stats" in stig_data:
                    loaded = stig_data["paradigm_stats"]
                    # Backward compat: old format was {paradigm: {runs, successes}}
                    # New format is {task_type: {paradigm: {runs, successes}}}
                    if loaded and isinstance(next(iter(loaded.values()), None), dict):
                        first_val = next(iter(loaded.values()))
                        if "runs" in first_val:
                            loaded = {"_global": loaded}
                    self._paradigm_stats.update(loaded)
                # Restore effectiveness tracker
                if "effectiveness" in stig_data:
                    self.effectiveness = EffectivenessTracker.from_dict(stig_data["effectiveness"])
                # Restore episode count (cumulative across sessions)
                if "episode_count" in stig_data:
                    self.episode_count = stig_data["episode_count"]
                logger.info(
                    f"Auto-loaded stigmergy: {len(self.stigmergy.signals)} signals, "
                    f"effectiveness: {self.effectiveness._global.__len__()} records"
                )
            except Exception as e:
                logger.warning(f"Could not auto-load stigmergy: {e}")

    def auto_save(
        self, mas_learning: Any = None, swarm_terminal: Any = None, provider_registry: Any = None
    ) -> None:
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

        # Adaptive credit weights (versioned)
        credit_path = self._get_credit_weights_path()
        try:
            self._save_versioned(credit_path, self.credit_weights.to_dict())
        except Exception as e:
            logger.debug(f"Could not auto-save credit weights: {e}")

        # TD-Lambda grouped baselines (versioned)
        td_path = self._get_td_lambda_path()
        try:
            self._save_versioned(td_path, self.td_learner.grouped_baseline.to_dict())
        except Exception as e:
            logger.debug(f"Could not auto-save TD-Lambda state: {e}")

        # Stigmergy pheromone trails + paradigm stats + effectiveness data + episode count (versioned)
        stig_path = self._get_stigmergy_path()
        try:
            stig_data = self.stigmergy.to_dict()
            stig_data["paradigm_stats"] = self._paradigm_stats
            stig_data["effectiveness"] = self.effectiveness.to_dict()
            stig_data["episode_count"] = self.episode_count
            self._save_versioned(stig_path, stig_data)
        except Exception as e:
            logger.warning(f"Could not auto-save stigmergy: {e}")

        # Provider registry
        if provider_registry:
            try:
                base = getattr(self.config, "base_path", None)
                if base:
                    provider_path = Path(base) / "provider_learnings.json"
                else:
                    provider_path = Path.home() / ".jotty" / "provider_learnings.json"
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
    def _compute_episode_reward(result: Any, goal: str) -> float:
        """
        Compute a continuous reward in [0, 1] from the execution result.

        Multi-dimensional heuristic decomposition with information density
        and relevance scoring. Each dimension is cheap to compute (no LLM call)
        and provides real gradient information for the learning algorithms.

        Dimensions (weighted average):
          substance  (0.25): Information density (unique 4-grams ratio)
          efficiency (0.10): How fast relative to a 120s baseline?
          tool_use   (0.15): Did tool calls produce useful output?
          structure  (0.10): Does output have organized structure?
          no_errors  (0.20): Absence of error indicators in output
          relevance  (0.20): Does output address the goal?

        The success flag is a floor: if the agent reports failure,
        the max reward is capped at 0.3 (allowing partial credit for
        useful partial output on failures).
        """
        import math

        # Extract output text
        output_text = ""
        if hasattr(result, "output") and result.output:
            output_text = str(result.output)
        elif isinstance(result, dict):
            output_text = str(result.get("output", ""))
        output_text = output_text.strip()

        success = bool(getattr(result, "success", False))
        lower = output_text.lower()[:3000]

        # --- Dimension 1: Substance via information density (0-1) ---
        # Uses compression-ratio proxy: unique 4-grams / total 4-grams.
        # Padded/repetitive text gets penalized. Base length score caps at 500 chars.
        char_count = len(output_text)
        if char_count == 0:
            substance = 0.0
        else:
            # Base length score: caps at 500 chars
            length_score = min(1.0, char_count / 500.0)
            # Information density: unique 4-grams / total 4-grams
            words = lower.split()
            if len(words) >= 4:
                ngrams = [" ".join(words[i : i + 4]) for i in range(len(words) - 3)]
                density = len(set(ngrams)) / len(ngrams) if ngrams else 0.5
            else:
                density = 0.5  # Too short to measure
            # Combine: length matters but repetitive padding is penalized
            substance = length_score * (0.4 + 0.6 * density)

        # --- Dimension 2: Efficiency (0-1) ---
        # Faster execution relative to 120s baseline scores higher.
        exec_time = getattr(result, "execution_time", 60.0)
        if exec_time <= 0:
            exec_time = 60.0  # Unknown defaults to neutral
        # Sigmoid-like: 5s → 0.95, 30s → 0.75, 60s → 0.5, 120s → 0.25, 300s → 0.05
        efficiency = 1.0 / (1.0 + math.exp((exec_time - 60) / 30))

        # --- Dimension 3: Tool usage effectiveness (0-1) ---
        # If the trajectory shows tool calls, check how many produced output.
        trajectory = getattr(result, "trajectory", []) or []
        tool_calls = 0
        tool_successes = 0
        for step in trajectory:
            if isinstance(step, dict):
                action = step.get("action", "")
                if "tool" in str(action).lower() or step.get("tool_name"):
                    tool_calls += 1
                    step_output = step.get("output", step.get("result", ""))
                    if step_output and len(str(step_output)) > 10:
                        tool_successes += 1
        if tool_calls > 0:
            tool_use = tool_successes / tool_calls
        else:
            tool_use = 0.3  # Slight penalty for not using available tools

        # --- Dimension 4: Structure (0-1) ---
        # Does the output show organized thinking? Check for structure markers.
        structure_signals = 0
        if any(m in lower for m in ["\n#", "\n##", "\n###"]):
            structure_signals += 1  # Headings
        if any(m in lower for m in ["\n- ", "\n* ", "\n1.", "\n2."]):
            structure_signals += 1  # Lists
        if "```" in output_text:
            structure_signals += 1  # Code blocks
        if any(m in lower for m in ["in conclusion", "summary", "therefore", "key finding"]):
            structure_signals += 1  # Conclusions
        structure = min(1.0, structure_signals / 3.0)

        # --- Dimension 5: Error absence (0-1) ---
        # Penalize outputs containing error indicators.
        error_indicators = [
            "error:",
            "exception:",
            "traceback",
            "failed to",
            "could not",
            "unable to",
            "i cannot",
            "i can't",
            "not supported",
            "invalid",
            "timeout",
        ]
        error_count = sum(1 for ind in error_indicators if ind in lower)
        no_errors = max(0.0, 1.0 - error_count * 0.25)

        # --- Dimension 6: Relevance (0-1) ---
        # Check that output mentions key terms from the goal.
        goal_lower = goal.lower() if goal else ""
        goal_words = set(
            w
            for w in goal_lower.split()
            if len(w) > 3
            and w
            not in {
                "this",
                "that",
                "with",
                "from",
                "about",
                "what",
                "have",
                "been",
                "will",
                "would",
                "could",
                "should",
                "their",
                "there",
                "them",
                "then",
                "than",
                "your",
            }
        )
        if goal_words and lower:
            matches = sum(1 for w in goal_words if w in lower)
            relevance = min(1.0, matches / max(1, len(goal_words) * 0.5))
        else:
            relevance = 0.5  # No goal words to check — neutral

        # --- Weighted combination ---
        reward = (
            0.25 * substance
            + 0.10 * efficiency
            + 0.15 * tool_use
            + 0.10 * structure
            + 0.20 * no_errors
            + 0.20 * relevance
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
            f"no_errors={no_errors:.2f}, relevance={relevance:.2f}, "
            f"success={success})"
        )

        return reward

    # =========================================================================
    # Post-episode learning pipeline
    # =========================================================================

    _DEFAULT_LEARNING_STEPS = (
        "td_lambda",
        "swarm_learner",
        "brain_consolidation",
        "neurochunk_tiering",
        "agent_abstractor",
        "transfer_learning",
        "swarm_intelligence",
        "stigmergy",
        "effectiveness",
        "mas_learning",
        "byzantine",
        "credit_assignment",
        "auditor_fixes",
        "adaptive_learning",
        "effectiveness_intervention",
        "credit_pruning",
        "curriculum",
    )

    def _record_stigmergy(
        self, agent_name: Any, task_type: Any, result: Any, episode_reward: Any, goal: Any
    ) -> Any:
        """Consolidated stigmergy: outcome + approach in one call (was 3 blocks)."""
        # 1. Record outcome (already creates route/warning signals internally)
        self.stigmergy.record_outcome(
            agent=agent_name,
            task_type=task_type,
            success=result.success,
            quality=episode_reward,
        )
        # 2. Extract tools and record approach
        trajectory = getattr(result, "trajectory", []) or []
        tools_used = []
        for step in trajectory:
            if isinstance(step, dict):
                tool = step.get("skill") or step.get("tool_name") or step.get("tool", "")
                if tool and tool not in tools_used:
                    tools_used.append(str(tool))
        if not tools_used:
            _out = getattr(result, "output", None)
            if hasattr(_out, "skills_used") and _out.skills_used:
                tools_used = list(_out.skills_used)
            elif isinstance(_out, dict):
                tools_used = list(_out.get("skills_used", []))
        skill_steps = []
        for step in trajectory[:8]:
            if isinstance(step, dict):
                skill = step.get("skill") or step.get("tool_name", "")
                desc = step.get("description", step.get("action", ""))
                if skill:
                    skill_steps.append(f"{skill}: {str(desc)[:40]}" if desc else skill)
        approach = (
            " -> ".join(skill_steps)
            if skill_steps
            else (f"Used: {', '.join(tools_used[:5])}" if tools_used else goal[:100])
        )
        self.stigmergy.record_approach_outcome(
            task_type=task_type,
            approach_summary=approach,
            tools_used=tools_used,
            success=result.success,
            quality=episode_reward,
            agent=agent_name,
        )

    def _run_learning_steps(self, ctx: dict) -> Any:
        """Run all enabled learning steps with uniform error handling."""
        enabled = getattr(self.config, "learning_components", None)
        steps = enabled if enabled else self._DEFAULT_LEARNING_STEPS
        for step_name in steps:
            method = getattr(self, f"_step_{step_name}", None)
            if not method:
                continue
            try:
                method(ctx)
            except Exception as e:
                logger.debug(f"Learning step '{step_name}' failed: {e}")

    def post_episode(
        self,
        result: EpisodeResult,
        goal: str,
        agents: list,
        architect_prompts: list,
        mas_learning: Any = None,
        swarm_terminal: Any = None,
    ) -> Any:
        """Post-episode learning: run all enabled learning steps."""
        self.episode_count += 1
        episode_reward = self._compute_episode_reward(result, goal)
        try:
            task_type = self.transfer_learning.extractor.extract_task_type(goal)
        except Exception:
            task_type = "general"
        agent_name = getattr(result, "agent_name", agents[0].name if agents else "unknown")

        ctx = {
            "result": result,
            "goal": goal,
            "agents": agents,
            "architect_prompts": architect_prompts,
            "episode_reward": episode_reward,
            "task_type": task_type,
            "agent_name": agent_name,
            "mas_learning": mas_learning,
            "swarm_terminal": swarm_terminal,
        }
        self._run_learning_steps(ctx)
        logger.debug(f"Post-episode learning complete (episode #{self.episode_count})")

    # -- Individual learning steps (each receives ctx dict) --

    def _step_td_lambda(self, ctx: Any) -> Any:
        """TD-Lambda: update grouped value baselines via TD(0)."""
        self.td_learner.start_episode(ctx["goal"], task_type=ctx["task_type"])
        self.td_learner.update(
            state={"goal": ctx["goal"]},
            action={"type": ctx["task_type"], "agent": ctx["agent_name"]},
            reward=ctx["episode_reward"],
            next_state={"completed": True},
        )
        self._adaptive_lr.record_success(ctx["result"].success)

    def _step_swarm_learner(self, ctx: Any) -> Any:
        """SwarmLearner: record episode, conditionally update prompts."""
        result = ctx["result"]
        trajectory = result.trajectory or []
        insights = []
        if hasattr(result, "tagged_outputs") and result.tagged_outputs:
            insights = [str(t) for t in result.tagged_outputs[:5]]
        self.swarm_learner.record_episode(trajectory, result.success, insights)

        if self.swarm_learner.should_update_prompts():
            for prompt_path in ctx["architect_prompts"]:
                try:
                    with open(prompt_path, "r") as f:
                        current = f.read()
                    updated, changes = self.swarm_learner.update_prompt(prompt_path, current)
                    if changes:
                        logger.info(f"Prompt '{prompt_path}' evolved with {len(changes)} changes")
                except Exception as e:
                    logger.debug(f"Prompt update skipped for {prompt_path}: {e}")

    def _step_brain_consolidation(self, ctx: Any) -> Any:
        """Brain consolidation (fire-and-forget in running loop)."""
        experience = {
            "content": str(ctx["result"].output)[:500] if ctx["result"].output else "",
            "context": {"goal": ctx["goal"], "episode": self.episode_count},
            "reward": ctx["episode_reward"],
            "agent": "swarm",
        }
        try:
            asyncio.get_running_loop()
            asyncio.ensure_future(self.brain_state.process_experience(experience))
        except RuntimeError:
            pass  # No running loop — skip async consolidation

    def _step_neurochunk_tiering(self, ctx: Any) -> Any:
        """NeuroChunk tiering: promote/demote/prune memories."""
        self.learning_manager.promote_demote_memories(ctx["episode_reward"])
        self.learning_manager.prune_tier3()

    def _step_agent_abstractor(self, ctx: Any) -> Any:
        """Agent abstractor: update agent role profiles."""
        result = ctx["result"]
        if hasattr(result, "agent_contributions") and result.agent_contributions:
            for contrib_agent, contrib in result.agent_contributions.items():
                success = getattr(contrib, "decision_correct", result.success)
                self.agent_abstractor.update_agent(contrib_agent, success)
        else:
            self.agent_abstractor.update_agent(ctx["agent_name"], result.success)

    def _step_transfer_learning(self, ctx: Any) -> Any:
        """Transferable learning store: record experience."""
        query = ctx["goal"][:200] if ctx["goal"] else ""
        self.transfer_learning.record_experience(
            query=query,
            agent=ctx["agent_name"],
            action=ctx["goal"][:100],
            reward=ctx["episode_reward"],
            success=ctx["result"].success,
            error=str(getattr(ctx["result"], "error", None) or ""),
            context={"episode": self.episode_count},
        )

    def _step_swarm_intelligence(self, ctx: Any) -> Any:
        """Swarm intelligence: record task result for specialization."""
        execution_time = getattr(ctx["result"], "execution_time", 0.0)
        self.swarm_intelligence.record_task_result(
            agent_name=ctx["agent_name"],
            task_type=ctx["task_type"],
            success=ctx["result"].success,
            execution_time=execution_time,
            context={"goal": ctx["goal"][:100], "episode": self.episode_count},
        )

    def _step_stigmergy(self, ctx: Any) -> Any:
        """Stigmergy: consolidated outcome + approach recording."""
        self._record_stigmergy(
            ctx["agent_name"],
            ctx["task_type"],
            ctx["result"],
            ctx["episode_reward"],
            ctx["goal"],
        )

    def _step_effectiveness(self, ctx: Any) -> Any:
        """Effectiveness tracker: measure actual improvement over time."""
        self.effectiveness.record(
            task_type=ctx["task_type"],
            success=ctx["result"].success,
            quality=ctx["episode_reward"],
            agent=ctx["agent_name"],
        )

    def _step_mas_learning(self, ctx: Any) -> Any:
        """MAS Learning: session recording."""
        mas_learning = ctx["mas_learning"]
        if not mas_learning:
            return
        result = ctx["result"]
        execution_time = getattr(result, "execution_time", 0.0)
        if hasattr(result, "agent_contributions") and result.agent_contributions:
            agents_used = list(result.agent_contributions.keys())
        else:
            agents_used = [ctx["agent_name"]]
        stigmergy_signals = (
            len(self.swarm_intelligence.stigmergy.signals)
            if hasattr(self.swarm_intelligence, "stigmergy")
            else 0
        )
        mas_learning.record_session(
            task_description=ctx["goal"],
            agents_used=agents_used,
            total_time=execution_time,
            success=result.success,
            stigmergy_signals=stigmergy_signals,
            output_quality=ctx["episode_reward"],
        )

    def _step_byzantine(self, ctx: Any) -> Any:
        """Byzantine verification: quality check + claim verification."""
        result = ctx["result"]
        if hasattr(result, "agent_contributions") and result.agent_contributions:
            for contrib_agent, contrib in result.agent_contributions.items():
                claimed = getattr(contrib, "decision_correct", result.success)
                self.byzantine_verifier.verify_claim(
                    agent=contrib_agent,
                    claimed_success=claimed,
                    actual_result=result,
                    task_type=ctx["task_type"],
                )
        else:
            quality = self.byzantine_verifier.verify_output_quality(
                agent=ctx["agent_name"],
                claimed_success=result.success,
                output=result,
                goal=ctx["goal"],
                task_type=ctx["task_type"],
            )
            if not quality["quality_ok"]:
                logger.warning(
                    f"Byzantine quality issues for {ctx['agent_name']}: " f"{quality['issues']}"
                )

    def _step_credit_assignment(self, ctx: Any) -> Any:
        """Credit assignment: record which agent/approach deserves credit."""
        self.credit_assigner.record_improvement_application(
            improvement={"learned_pattern": ctx["goal"][:200], "task": ctx["goal"][:100]},
            student_score=0.0,
            teacher_score=0.0,
            final_score=ctx["episode_reward"],
            context={"task": ctx["goal"][:100], "episode": self.episode_count},
        )

    def _step_auditor_fixes(self, ctx: Any) -> Any:
        """Auditor fix_instructions -> negative TD signal + procedural memory."""
        result = ctx["result"]
        goal = ctx["goal"]
        task_type = ctx["task_type"]
        agent_name = ctx["agent_name"]

        fix_texts = []
        for attr in ("auditor_results", "validation_results"):
            for vr in getattr(result, attr, None) or []:
                fi = getattr(vr, "fix_instructions", None)
                if fi and isinstance(fi, str) and fi.strip():
                    fix_texts.append(fi.strip())

        if fix_texts and self.td_learner is not None:
            self.td_learner.update(
                state={"goal": goal},
                action={"type": task_type, "agent": agent_name, "source": "auditor_fix"},
                reward=-0.1,
                next_state={"fix_instructions": True},
            )
            logger.debug(f"TD(-0.1) signal from {len(fix_texts)} auditor fix_instructions")

        if fix_texts:
            combined = "; ".join(fix_texts[:3])
            self.transfer_learning.record_experience(
                query=f"fix:{goal[:100]}",
                agent=agent_name,
                action=combined[:500],
                reward=-0.1,
                success=False,
                error="auditor_fix_instructions",
                context={"task_type": task_type, "episode": self.episode_count},
            )
            logger.debug(f"Recorded {len(fix_texts)} fix_instructions as procedural memory")

    def _step_adaptive_learning(self, ctx: Any) -> Any:
        """Adaptive learning: adjust learning rate based on score trajectory."""
        lr_state = self.adaptive_learning.update_score(ctx["episode_reward"])
        if lr_state.get("is_plateau"):
            logger.info(
                f"Adaptive learning: plateau detected "
                f"(lr={lr_state['learning_rate']:.2f}, "
                f"explore={lr_state['exploration_rate']:.2f})"
            )

    def _step_effectiveness_intervention(self, ctx: Any) -> Any:
        """Effectiveness-driven intervention: boost exploration on stagnation."""
        task_type = ctx["task_type"]
        if not self.effectiveness.is_improving(task_type) and self.episode_count >= 10:
            al = self.adaptive_learning
            old_explore = getattr(al.state, "exploration_rate", 0.3)
            new_explore = min(0.8, old_explore + 0.2)
            al.state.exploration_rate = new_explore
            task = self.curriculum_generator.generate_training_task(
                profiles={},
                focus_task_type=task_type,
            )
            self._pending_training_tasks.append(task)
            if len(self._pending_training_tasks) > 10:
                self._pending_training_tasks = self._pending_training_tasks[-10:]
            logger.info(
                f"Effectiveness intervention for '{task_type}': "
                f"exploration {old_explore:.2f} -> {new_explore:.2f}, "
                f"queued curriculum task"
            )

    def _step_credit_pruning(self, ctx: Any) -> Any:
        """Credit-driven pruning: every 10 episodes, prune low-value learnings."""
        if self.episode_count % 10 != 0 or self.episode_count == 0:
            return
        before_count = len(self.transfer_learning.experiences)
        if before_count <= 20:
            return
        improvements = [
            {
                "learned_pattern": exp.get("query", "")[:200],
                "task": exp.get("action", ""),
                "timestamp": exp.get("timestamp", ""),
            }
            for exp in self.transfer_learning.experiences
        ]
        pruned = self.credit_assigner.prune_low_impact_improvements(
            improvements,
            min_credit_threshold=0.1,
            min_application_count=1,
        )
        pruned_patterns = {imp["learned_pattern"] for imp in pruned}
        self.transfer_learning.experiences = [
            exp
            for exp in self.transfer_learning.experiences
            if exp.get("query", "")[:200] in pruned_patterns or exp.get("query", "") == ""
        ]
        after_count = len(self.transfer_learning.experiences)
        if after_count < before_count:
            logger.info(
                f"Credit pruning: {before_count} -> {after_count} "
                f"experiences (removed {before_count - after_count} low-value)"
            )

    def _step_curriculum(self, ctx: Any) -> Any:
        """Curriculum: queue training tasks when exploration is recommended."""
        recommendation = self.adaptive_learning._get_recommendation()
        if recommendation == "increase_exploration":
            task = self.curriculum_generator.generate_training_task(profiles={})
            self._pending_training_tasks.append(task)
            logger.info(
                f"Curriculum: queued training task "
                f"(difficulty={task.difficulty:.2f}): {task.description[:60]}"
            )
            if len(self._pending_training_tasks) > 10:
                self._pending_training_tasks = self._pending_training_tasks[-10:]

    def pop_training_task(self) -> Any:
        """Pop the next queued training task (or None if queue empty)."""
        if self._pending_training_tasks:
            return self._pending_training_tasks.pop(0)
        return None

    def pending_training_count(self) -> int:
        """How many training tasks are waiting."""
        return len(self._pending_training_tasks)

    def learn_from_result(
        self,
        result: EpisodeResult,
        agent_config: AgentConfig,
        workflow_learner: Any = None,
        goal: str = "",
    ) -> None:
        """Learn from a successful execution result."""
        if not result.success:
            return
        if not workflow_learner:
            return

        # Extract task_type: prefer goal text (most reliable), then result fields
        from Jotty.core.intelligence.learning.transfer_learning import PatternExtractor

        task_type = "unknown"
        if goal:
            try:
                task_type = self.transfer_learning.extractor.extract_task_type(goal)
            except Exception as e:
                logger.debug(f"Task type extraction from goal failed: {e}")
        if task_type == "unknown" and hasattr(result, "task_type") and result.task_type:
            raw = (
                result.task_type.value
                if hasattr(result.task_type, "value")
                else str(result.task_type)
            )
            task_type = PatternExtractor.normalize_task_type(raw)
        if task_type == "unknown" and hasattr(result, "task") and result.task:
            try:
                task_type = self.transfer_learning.extractor.extract_task_type(str(result.task))
            except Exception as e:
                logger.debug(f"Task type extraction from result.task failed: {e}")
        # Always normalize (creation→generation, research→analysis, etc.)
        task_type = PatternExtractor.normalize_task_type(task_type)

        # Extract skills/tools actually used from result or its output
        skills_used = []
        if hasattr(result, "skills_used") and result.skills_used:
            skills_used = list(result.skills_used)
        if not skills_used and hasattr(result, "output"):
            _out = result.output
            if isinstance(_out, dict):
                skills_used = list(_out.get("skills_used", []))
            elif hasattr(_out, "skills_used") and _out.skills_used:
                skills_used = list(_out.skills_used)

        # Extract operations from trajectory steps
        operations = []
        trajectory = getattr(result, "trajectory", []) or []
        for step in trajectory[:10]:
            if isinstance(step, dict):
                action = (
                    step.get("skill")
                    or step.get("action")
                    or step.get("tool_name")
                    or step.get("step", "")
                )
                if action:
                    operations.append(str(action))
            elif step:
                operations.append(str(step))
        # Also extract from output.steps if available (autonomous agent execution results)
        if not operations and hasattr(result, "output"):
            _out = result.output
            _steps = getattr(_out, "steps", None) or (isinstance(_out, dict) and _out.get("steps"))
            if _steps and isinstance(_steps, (list, tuple)):
                for s in _steps[:10]:
                    if isinstance(s, dict):
                        skill = s.get("skill", s.get("action", ""))
                        if skill:
                            operations.append(str(skill))

        logger.info(
            f" Workflow learning: task_type={task_type}, "
            f"operations={operations[:5]}, tools={skills_used[:5]}"
        )
        workflow_learner.learn_from_execution(
            task_type=task_type,
            operations=operations,
            tools_used=skills_used,
            success=True,
            execution_time=getattr(result, "execution_time", 0.0),
            metadata={"agent": agent_config.name},
        )

    # =========================================================================
    # Query methods
    # =========================================================================

    def get_transferable_context(self, query: str, agent: str = None) -> str:
        """Get transferable learnings as context string."""
        return self.transfer_learning.get_relevant_context(query, agent=agent)

    def get_swarm_wisdom(self, query: str) -> str:
        """Get swarm intelligence wisdom for a query.

        Delegates to SwarmIntelligence.get_swarm_wisdom() (rich dict) and
        formats the result as a human-readable string for CLI/API callers.
        """
        try:
            task_type = self.transfer_learning.extractor.extract_task_type(query)
            wisdom = self.swarm_intelligence.get_swarm_wisdom(query, task_type=task_type)
            parts = []
            if wisdom.get("recommended_agent"):
                parts.append(f"Best agent for this task: {wisdom['recommended_agent']}")
            specs = self.swarm_intelligence.get_specialization_summary()
            if specs:
                parts.append(f"Agent specializations: {specs}")
            for w in wisdom.get("warnings", [])[:3]:
                parts.append(f"Warning: {w}")
            if wisdom.get("confidence", 0) > 0:
                parts.append(f"Confidence: {wisdom['confidence']:.0%}")
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
        except Exception as e:
            logger.debug(f"Best agent lookup failed: {e}")
            return None

    # =========================================================================
    # Stigmergy queries
    # =========================================================================

    def get_stigmergy_route(self, task_type: str) -> Optional[str]:
        """Use stigmergy pheromone trails to suggest the best agent for a task type."""
        try:
            signals = self.stigmergy.sense(signal_type="route")
            # Filter for this task type and find strongest
            best_agent, best_strength = None, 0.0
            for sig in signals:
                content = sig.content if hasattr(sig, "content") else {}
                if isinstance(content, dict) and content.get("task_type") == task_type:
                    if sig.strength > best_strength:
                        best_agent = content.get("agent")
                        best_strength = sig.strength
            return best_agent
        except Exception as e:
            logger.debug(f"Stigmergy route lookup failed: {e}")
            return None

    def get_stigmergy_warnings(self, task_type: str = None) -> list:
        """Get stigmergy warning signals, optionally filtered by task type."""
        try:
            signals = self.stigmergy.sense(signal_type="warning")
            if task_type:
                return [
                    s
                    for s in signals
                    if isinstance(getattr(s, "content", {}), dict)
                    and s.content.get("task_type") == task_type
                ]
            return signals
        except Exception as e:
            logger.debug(f"Stigmergy warnings lookup failed: {e}")
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
        except Exception as e:
            logger.debug(f"Agent trust lookup failed for '{agent_name}': {e}")
            return 1.0  # Default: trust unknown agents

    def is_agent_trusted(self, agent_name: str, threshold: float = 0.3) -> bool:
        """Check if an agent is trusted above threshold."""
        return self.get_agent_trust(agent_name) >= threshold

    # =========================================================================
    # Multi-agent ordering (single place for routing logic — used by SwarmRouter)
    # =========================================================================

    def order_agents_for_goal(self, goal: str, agents: List[Any]) -> List[Any]:
        """
        Order agents for a multi-agent run using learning: Byzantine trust,
        stigmergy route strength, and MorphAgent TRAS. Single place for this
        logic so Orchestrator and SwarmRouter stay DRY.

        Returns a new list (does not mutate input). Caller should assign back
        if they want to persist the order (e.g. self.agents = lp.order_agents_for_goal(...)).
        """
        if not agents:
            return list(agents)
        ordered = list(agents)
        try:
            task_type = self.transfer_learning.extractor.extract_task_type(goal)
        except Exception as e:
            logger.debug(f"Task type extraction failed for ordering: {e}")
            return ordered

        # 1. Filter by Byzantine trust (keep only trusted)
        trusted = [
            a for a in ordered if self.is_agent_trusted(getattr(a, "name", str(a)), threshold=0.2)
        ]
        if trusted:
            ordered = trusted

        # 2. Reorder by stigmergy pheromone strength (strongest first)
        routes = self.stigmergy.get_route_signals(task_type)
        if routes:
            ordered.sort(key=lambda a: routes.get(getattr(a, "name", str(a)), 0.0), reverse=True)

        # 3. MorphAgent TRAS + combined score with stigmergy
        si = self.swarm_intelligence
        if (
            si
            and len(ordered) >= 2
            and getattr(si, "morph_scorer", None)
            and getattr(si, "agent_profiles", None)
        ):
            profiles = si.agent_profiles
            tras_scores = {}
            for agent_cfg in ordered:
                name = getattr(agent_cfg, "name", str(agent_cfg))
                if name in profiles:
                    tras, _ = si.morph_scorer.compute_tras(
                        task=goal,
                        task_type=task_type,
                        profile=profiles[name],
                        use_llm=False,
                    )
                    tras_scores[name] = tras
            if tras_scores and max(tras_scores.values()) > min(tras_scores.values()):

                def _combined_score(a: Any) -> Any:
                    name = getattr(a, "name", str(a))
                    route_s = routes.get(name, 0.0) if routes else 0.0
                    tras_s = tras_scores.get(name, 0.5)
                    return 0.6 * route_s + 0.4 * tras_s

                ordered.sort(key=_combined_score, reverse=True)
        return ordered

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
            "learning_rate": state.learning_rate,
            "exploration_rate": state.exploration_rate,
            "is_plateau": state.is_plateau,
            "is_converging": state.is_converging,
            "improvement_velocity": state.improvement_velocity,
            "should_stop": self.adaptive_learning.should_stop_early(),
        }

    def get_effectiveness_report(self) -> dict:
        """
        Get measurable improvement data.

        Returns per-task-type success rate trends (recent vs historical).
        A positive 'trend' value means the system is genuinely improving.
        """
        return self.effectiveness.improvement_report()

    def get_td_lambda_stats(self) -> dict:
        """
        Get TD-Lambda grouped learning statistics.

        Returns group baselines, counts, and advantage info per task type.
        """
        stats = self.td_learner.get_grouped_learning_stats()
        stats["adaptive_lr"] = self._adaptive_lr.get_adapted_alpha()
        return stats

    # =========================================================================
    # Paradigm effectiveness tracking (auto paradigm selection)
    # =========================================================================

    _PARADIGMS = ("fanout", "relay", "debate", "refinement")

    def _ensure_paradigm_bucket(self, task_type: str) -> Any:
        """Lazy-create the stats bucket for a task_type."""
        if task_type not in self._paradigm_stats:
            self._paradigm_stats[task_type] = {
                p: {"runs": 0, "successes": 0} for p in self._PARADIGMS
            }

    def record_paradigm_result(
        self, paradigm: str, success: bool, task_type: str = "_global"
    ) -> Any:
        """Record the outcome of a discussion paradigm run for a task type."""
        self._ensure_paradigm_bucket(task_type)
        bucket = self._paradigm_stats[task_type]
        if paradigm not in bucket:
            bucket[paradigm] = {"runs": 0, "successes": 0}
        bucket[paradigm]["runs"] += 1
        if success:
            bucket[paradigm]["successes"] += 1

        # Also update _global so there's always a fallback
        if task_type != "_global":
            self.record_paradigm_result(paradigm, success, "_global")

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
        key = task_type or "_global"
        if key in self._paradigm_stats:
            total_runs = sum(s["runs"] for s in self._paradigm_stats[key].values())
            if total_runs >= 5:
                bucket = self._paradigm_stats[key]

        if bucket is None and "_global" in self._paradigm_stats:
            bucket = self._paradigm_stats["_global"]

        if not bucket:
            # No data at all — return random paradigm
            return random.choice(list(self._PARADIGMS))

        best_paradigm = "fanout"
        best_score = -1.0

        for paradigm in self._PARADIGMS:
            stats = bucket.get(paradigm, {"runs": 0, "successes": 0})
            runs = stats["runs"]
            successes = stats["successes"]
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

        def _format_bucket(bucket: Any) -> Any:
            out = {}
            for paradigm, data in bucket.items():
                runs = data["runs"]
                successes = data["successes"]
                out[paradigm] = {
                    **data,
                    "success_rate": successes / runs if runs > 0 else None,
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
