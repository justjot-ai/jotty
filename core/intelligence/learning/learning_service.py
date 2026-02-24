"""
LearningService - Unified Learning for All Execution Units
===========================================================

Single service shared by agents, swarms, pipelines, and orchestrator.
Replaces scattered learning mixins with a clean service interface.

Responsibilities:
    record()   — Record any execution outcome
    query()    — Get recommendations for a given context
    reflect()  — Mid-execution reflection and adjustment
    transfer() — Transfer learning patterns between domains
    start/end_episode() — Episode lifecycle for multi-step tracking

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                   LearningService (singleton)                │
    │  ┌───────────┐ ┌───────────┐ ┌──────────┐ ┌────────────┐  │
    │  │ TD-Lambda │ │ Patterns  │ │ Reflect  │ │  Transfer  │  │
    │  │ (values)  │ │ (extract) │ │ (adjust) │ │  (cross)   │  │
    │  └─────┬─────┘ └─────┬─────┘ └────┬─────┘ └──────┬─────┘  │
    │        └──────────────┴────────────┴──────────────┘        │
    │                        LearningStore                        │
    │                    (SQLite, persistent)                      │
    └─────────────────────────────────────────────────────────────┘

    Used by:  Agent.execute()  |  Swarm.execute()  |  Orchestrator.run()

Author: Jotty Team
Date: February 2026
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

# Extracted modules — keep public names importable from this module
from .domain_classifier import (
    _get_related_domains,
)
from .learning_store import (
    DistilledLesson,
    EpisodeRecord,
    LearningStore,
    PatternRecord,
    ValueEstimate,
)
from .pattern_extractor import PatternExtractor
from .reflection_engine import ReflectionEngine
from .response_analyzer import (
    _CITATION_RE,
    _CODE_BLOCK_RE,
    analyze_response,
)

logger = logging.getLogger(__name__)

# Lazy import — avoids loading sentence-transformers at module level
_embedding_service = None


def _get_embeddings():
    """Lazy-load the EmbeddingService singleton."""
    global _embedding_service
    if _embedding_service is None:
        from .embeddings import EmbeddingService

        _embedding_service = EmbeddingService.get_instance()
    return _embedding_service


# =============================================================================
# EPISODE TRACKER (for multi-step execution tracking)
# =============================================================================


class ActiveEpisode:
    """Tracks a running execution episode with its steps."""

    def __init__(
        self,
        episode_id: str,
        unit_type: str,
        unit_name: str,
        domain: str,
        task_type: str,
        context: Dict[str, Any],
    ) -> None:
        self.episode_id = episode_id
        self.unit_type = unit_type
        self.unit_name = unit_name
        self.domain = domain
        self.task_type = task_type
        self.context = context
        self.start_time = time.time()
        self.steps: List[Dict[str, Any]] = []
        self.reflections: List[Dict[str, Any]] = []
        self.parent_episode_id: Optional[str] = None

    def add_step(
        self, step_name: str, action: Dict[str, Any], outcome: Dict[str, Any], success: bool
    ) -> None:
        self.steps.append(
            {
                "step": len(self.steps) + 1,
                "name": step_name,
                "action": action,
                "outcome": outcome,
                "success": success,
                "timestamp": time.time(),
            }
        )


# =============================================================================
# LEARNING SERVICE
# =============================================================================


class LearningService:
    """
    Unified learning service for all Jotty execution units.

    Singleton per process. All agents, swarms, pipelines, and the orchestrator
    share this service. Learning data persists in SQLite via LearningStore.

    Key design principles:
    1. ANY execution unit can record outcomes and query for guidance
    2. Learning happens at multiple timescales (intra-step, post-episode, cross-episode)
    3. Patterns transfer across domains automatically
    4. Self-evolving: extracts patterns from accumulated data

    Usage:
        service = LearningService.get_instance()

        # Record an outcome
        service.record("CodingSwarm", "swarm", "coding", "code_generation",
                       context={"task": "REST API"}, action={"paradigm": "pipeline"},
                       outcome={"code": "...", "tests_pass": True},
                       success=True, quality=0.85)

        # Query for guidance
        guidance = service.query("coding", "code_generation",
                                 context={"task": "REST API"})

        # Mid-execution reflection
        adjustment = service.reflect(episode_id, step=3,
                                      observation="Test failures on edge cases",
                                      unit_name="TestWriter")

        # Transfer learning
        patterns = service.transfer("coding", "devops")
    """

    _instance: Optional["LearningService"] = None

    def __init__(self, store: Optional[LearningStore] = None) -> None:
        from Jotty.core.infrastructure.foundation.configs.learning import LearningConfig

        self._config = LearningConfig()
        self._store = store or LearningStore.get_instance()
        self._active_episodes: Dict[str, ActiveEpisode] = {}

        # In-memory caches for hot-path queries (avoid DB hits)
        self._success_rate_cache: Dict[str, Tuple[float, int, float]] = (
            {}
        )  # key -> (rate, total, ts)
        self._pattern_cache: Dict[str, Tuple[List[PatternRecord], float]] = {}
        self._cache_ttl = 60.0  # Cache TTL in seconds

        # Pattern extraction thresholds
        self._min_episodes_for_pattern = 2
        self._pattern_extraction_interval = 1  # Extract patterns on every record
        self._record_count = 0

        # Auto-transfer: track episodes per domain, trigger transfer every N episodes.
        # Seed counts from DB so transfer fires even after process restart.
        self._domain_episode_counts: Dict[str, int] = self._seed_domain_counts()
        self._auto_transfer_interval = 10  # transfer after every 10 episodes in a domain
        self._last_transfer_counts: Dict[str, int] = {}  # domain -> count at last transfer

        # Auto-optimize: DSPy re-optimization + crystallization triggers
        self._last_optimize_counts: Dict[str, int] = {}  # domain -> count at last optimize
        self._dspy_optimize_interval = 15  # re-optimize every N new episodes per domain

        # Auto-reflect: trigger post-execution reflection on significant outcomes
        self._auto_reflect_interval = 10  # reflect every N episodes
        self._auto_reflect_quality_threshold = self._config.auto_reflect_quality_threshold

        # Background LLM judge tasks (fire-and-forget)
        self._judge_tasks: set = set()
        self._judged_episodes: set = set()  # Dedup: episode IDs already judged or in-flight

        # Domain-aware score blending overrides
        # Code-producing domains: trust LLM more (word-count heuristic penalizes short code)
        # Prose domains (research, creation, finance): default 0.6/0.4 is correct
        _DEFAULT_BLEND_OVERRIDES = {
            "coding": (0.85, 0.15),
            "algorithms": (0.85, 0.15),
            "math": (0.80, 0.20),
            "devops": (0.85, 0.15),
            "testing": (0.85, 0.15),
            "data_analysis": (0.80, 0.20),
        }
        self._blend_overrides = self._config.judge_blend_overrides or _DEFAULT_BLEND_OVERRIDES

        # Extracted subsystems
        self._pattern_extractor = PatternExtractor(self._store, self._min_episodes_for_pattern)
        self._reflection_engine = ReflectionEngine(
            self._store, self._auto_reflect_quality_threshold
        )

        logger.info("LearningService initialized")

    @classmethod
    def get_instance(cls, store: Optional[LearningStore] = None) -> "LearningService":
        """Get or create the singleton LearningService."""
        if cls._instance is None:
            cls._instance = cls(store)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None

    def _seed_domain_counts(self) -> Dict[str, int]:
        """Seed domain episode counts from DB so auto-transfer works
        even after process restart (without re-recording all episodes)."""
        try:
            conn = self._store._get_conn()
            rows = conn.execute(
                "SELECT domain, COUNT(*) as cnt FROM episodes " "WHERE domain != '' GROUP BY domain"
            ).fetchall()
            return {r["domain"]: r["cnt"] for r in rows}
        except Exception:
            return {}

    # =========================================================================
    # RECORD — Record any execution outcome
    # =========================================================================

    def record(
        self,
        unit_name: str,
        unit_type: str,
        domain: str,
        task_type: str,
        context: Dict[str, Any],
        action: Dict[str, Any],
        outcome: Dict[str, Any],
        success: bool,
        quality: float = 0.0,
        execution_time: float = 0.0,
        cost: float = 0.0,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        parent_episode_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Record an execution outcome from any unit (agent, swarm, pipeline).

        Args:
            unit_name: Name of the execution unit (e.g., "CodingSwarm", "ResearchAgent")
            unit_type: Type: "agent", "swarm", "pipeline", "orchestrator"
            domain: Domain (e.g., "coding", "research", "devops")
            task_type: Specific task type (e.g., "code_generation", "stock_analysis")
            context: Execution context (task description, inputs)
            action: What was done (paradigm, tools used, model chosen)
            outcome: Results produced
            success: Whether the execution succeeded
            quality: Quality score 0.0-1.0
            execution_time: Seconds taken
            cost: USD cost of LLM calls
            error_type: Classified error category (if failed)
            error_message: Error details (if failed)
            parent_episode_id: Parent episode ID (for nested episodes)
            metadata: Additional metadata

        Returns:
            Episode ID
        """
        if not metadata or "original_episode_id" not in metadata:
            episode_id = f"ep_{unit_name}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        else:
            episode_id = metadata.pop("original_episode_id")

        # Auto-enrich outcome with structural analysis when text content is
        # available.  This enables all pattern extraction strategies (structural,
        # causal, quality-contrast, cross-domain transfer) to work regardless
        # of whether the caller ran analyze_response() themselves.
        if "quality_score" not in outcome:  # not already enriched
            content_text = ""
            for _key in ("content", "response", "response_excerpt", "result"):
                _val = outcome.get(_key, "")
                if isinstance(_val, str) and len(_val) > 100:
                    content_text = _val
                    break
            if content_text:
                goal_text = ""
                if isinstance(context, dict):
                    goal_text = str(context.get("goal", context.get("message", "")))
                try:
                    analysis = analyze_response(content_text, goal_text)
                    # Merge analysis into outcome — don't override caller's keys
                    for _ak, _av in analysis.items():
                        if _ak not in outcome:
                            outcome[_ak] = _av
                    # Use analysis quality if caller didn't provide one
                    if quality == 0.0 and analysis.get("quality_score", 0) > 0:
                        quality = analysis["quality_score"]
                except Exception:
                    pass  # analysis failure must never block recording

        episode = EpisodeRecord(
            episode_id=episode_id,
            unit_type=unit_type,
            unit_name=unit_name,
            domain=domain,
            task_type=task_type,
            context=self._truncate_dict(context),
            action=self._truncate_dict(action),
            outcome=self._truncate_dict(outcome),
            success=success,
            quality=quality,
            execution_time=execution_time,
            cost=cost,
            error_type=error_type,
            error_message=error_message[:500] if error_message else None,
            parent_episode_id=parent_episode_id,
            metadata=metadata or {},
        )

        try:
            self._store.save_episode(episode)
        except Exception as e:
            logger.warning(f"Failed to save episode: {e}")
            # Continue with TD updates and pattern extraction even if DB save failed —
            # in-memory learning still benefits from this data point.

        # Update value estimates via TD learning
        self._update_values(domain, task_type, action, success, quality)

        # Per-skill credit-weighted Q-table updates (from algorithmic_credit)
        # When a multi-agent episode completes, distribute credit proportionally
        # so skills that contributed more get higher Q-updates.
        self._update_skill_credits(domain, task_type, action, outcome, success, quality)

        # Invalidate caches
        self._invalidate_cache(domain, task_type)

        # Periodic pattern extraction — run for triggering domain AND
        # sweep all domains that have accumulated enough episodes
        self._record_count += 1
        if self._record_count % self._pattern_extraction_interval == 0:
            try:
                self._extract_patterns(domain)
            except Exception as e:
                logger.debug(f"Pattern extraction failed for {domain}: {e}")

            # Sweep: extract patterns for other domains that have enough data
            try:
                conn = self._store._get_conn()
                rows = conn.execute(
                    "SELECT DISTINCT domain FROM episodes WHERE domain != ? AND domain != ''",
                    (domain,),
                ).fetchall()
                for row in rows:
                    other_domain = row["domain"]
                    try:
                        self._extract_patterns(other_domain)
                    except Exception:
                        pass
            except Exception as e:
                logger.debug(f"Pattern sweep failed: {e}")

        # ── Auto-transfer: propagate patterns to related domains ──
        self._domain_episode_counts[domain] = self._domain_episode_counts.get(domain, 0) + 1
        domain_count = self._domain_episode_counts[domain]
        last_transfer = self._last_transfer_counts.get(domain, 0)
        if (
            domain
            and domain_count >= self._auto_transfer_interval
            and domain_count - last_transfer >= self._auto_transfer_interval
        ):
            self._last_transfer_counts[domain] = domain_count
            related = _get_related_domains(domain)
            for target in related[:3]:  # Top 3 related domains
                try:
                    transferred = self.transfer(domain, target)
                    if transferred:
                        logger.debug(
                            f"Auto-transferred {len(transferred)} patterns "
                            f"from {domain} → {target}"
                        )
                except Exception as e:
                    logger.debug(f"Auto-transfer {domain}→{target} failed: {e}")

        # ── Auto-reflect: learn from significant outcomes ──
        if domain and self._record_count % self._auto_reflect_interval == 0:
            try:
                self._auto_reflect(
                    episode_id, domain, task_type, context, outcome, success, quality, unit_name
                )
            except Exception as e:
                logger.debug(f"Auto-reflect failed: {e}")

        # Fire-and-forget LLM judge for successful episodes.
        # The judge also triggers judge-informed distillation when it completes.
        # Gate: success + any non-trivial content (no char-count floor — short
        # outputs like code/diagrams are perfectly valid).
        _content = (
            outcome.get("content", "")
            or outcome.get("response_excerpt", "")
            or outcome.get("output", "")
            or outcome.get("result", "")
        )
        _judge_will_distill = False
        if isinstance(_content, str) and len(_content) > 50 and success:
            _goal = str(context.get("goal", context.get("message", task_type)))
            self.schedule_background_judge(
                episode_id, _goal, _content, domain, quality, unit_name=unit_name
            )
            _judge_will_distill = True  # judge handles distillation with feedback

        # ── Compute & store embedding for the goal text ──────────────────
        _goal_text = ""
        if isinstance(context, dict):
            _goal_text = str(context.get("goal", context.get("message", "")))
        if _goal_text and len(_goal_text) > 20:
            try:
                emb = _get_embeddings()
                if emb.available:
                    from .embeddings import EmbeddingService

                    vec = emb.embed(_goal_text)
                    if vec is not None:
                        self._store.save_embedding(episode_id, EmbeddingService.serialize(vec))
            except Exception as exc:
                logger.debug("Embedding computation failed: %s", exc)

        # ── Schedule fact distillation for good episodes (mem0 pattern) ──
        # Skip if the judge is already handling distillation (with its feedback).
        # Only run standalone distillation for episodes the judge won't cover.
        _content_len = len(_content) if isinstance(_content, str) else 0
        if not _judge_will_distill and _content_len > 50 and success:
            self._schedule_fact_distillation(
                episode_id, _goal_text or task_type, _content, domain, unit_name
            )

        # ── Reflexion: generate reflection on failures (Shinn et al.) ──
        if not success or quality < self._config.reflexion_quality_threshold:
            try:
                from .advanced_learning import Reflexion

                reflexion = Reflexion.get_instance()
                _output = str(_content)[:1500] if isinstance(_content, str) else ""
                reflexion.reflect_on_failure(
                    episode_id=episode_id,
                    unit_name=unit_name,
                    goal=_goal_text or task_type,
                    output=_output,
                    error_type=error_type or "",
                    error_message=error_message or "",
                )
            except Exception as e:
                logger.debug(f"Reflexion generation failed: {e}")

        # ── VoyagerSkillLib: extract reusable patterns from high-quality successes ──
        if success and quality >= self._config.pattern_extract_quality_threshold:
            try:
                from .advanced_learning import VoyagerSkillLib

                skill_lib = VoyagerSkillLib.get_instance()
                approach = str(action.get("paradigm", action.get("model", "")))
                skill_lib.extract_skill_pattern(
                    episode_id=episode_id,
                    domain=domain,
                    task_type=task_type,
                    goal=_goal_text or task_type,
                    approach=approach,
                    quality=quality,
                )
            except Exception as e:
                logger.debug(f"VoyagerSkillLib extraction failed: {e}")

        # ── Auto-crystallize + auto-DSPy-optimize ──────────────────────
        # Periodic check: when enough gold episodes accumulate for a domain,
        # (1) re-optimize DSPy prompts (incremental, even pre-crystallization)
        # (2) attempt crystallization (graduation to hardened SOP)
        # This closes the loop: record() -> learn -> optimize -> crystallize
        # without requiring explicit run_probation() or manual calls.
        if domain and success and domain_count >= 8:
            self._maybe_auto_optimize(domain, task_type, domain_count)

        logger.debug(
            f"Recorded: {unit_name} ({unit_type}) domain={domain} "
            f"task={task_type} success={success} quality={quality:.2f}"
        )
        return episode_id

    def _auto_reflect(
        self,
        episode_id: str,
        domain: str,
        task_type: str,
        context: Dict[str, Any],
        outcome: Dict[str, Any],
        success: bool,
        quality: float,
        unit_name: str,
    ) -> None:
        """Delegate to ReflectionEngine.auto_reflect()."""
        self._reflection_engine.auto_reflect(
            episode_id=episode_id,
            domain=domain,
            task_type=task_type,
            context=context,
            outcome=outcome,
            success=success,
            quality=quality,
            unit_name=unit_name,
            get_value_fn=self._store.get_value,
            make_state_key_fn=self._make_state_key,
        )

    # =========================================================================
    # QUERY — Get recommendations for a given context
    # =========================================================================

    def query(
        self,
        domain: str,
        task_type: str = "",
        context: Optional[Dict[str, Any]] = None,
        unit_name: str = "",
    ) -> Dict[str, Any]:
        """
        Query for guidance based on domain and task context.

        Returns recommendations, success rates, known patterns, and
        failure analysis to help the calling unit make better decisions.

        Args:
            domain: Execution domain
            task_type: Specific task type
            context: Current task context
            unit_name: Name of the querying unit

        Returns:
            Dict with:
                - success_rate: Historical success rate
                - patterns: Applicable behavioral patterns
                - recommendations: Specific action recommendations
                - failure_analysis: Recent failure patterns to avoid
                - best_action: Recommended action based on value estimates
        """
        result: Dict[str, Any] = {
            "has_learning": False,
            "success_rate": 0.0,
            "total_episodes": 0,
            "patterns": [],
            "recommendations": [],
            "failure_analysis": [],
            "best_action": None,
            "improving": False,
        }

        try:
            # 1. Success rate
            rate, total = self._get_cached_success_rate(domain, task_type)
            result["success_rate"] = rate
            result["total_episodes"] = total
            result["has_learning"] = total > 0

            # 2. Applicable patterns
            patterns = self._get_cached_patterns(domain)
            result["patterns"] = [
                {
                    "description": p.description,
                    "recommendation": p.recommendation,
                    "confidence": p.confidence,
                    "type": p.pattern_type,
                }
                for p in patterns[:5]
            ]

            # 3. Build recommendations from patterns + value estimates
            recommendations = []
            for p in patterns[:3]:
                if p.confidence >= 0.6:
                    recommendations.append(p.recommendation)
            result["recommendations"] = recommendations

            # 4. Failure/low-quality analysis (even at high success rate,
            #    low-quality episodes teach what to avoid)
            failures = self._store.get_failure_analysis(domain, task_type, limit=5)
            result["failure_analysis"] = [
                {
                    "error_type": f.get("error_type", "unknown"),
                    "description": f.get("error_message", "")[:200],
                    "task_type": f.get("task_type", ""),
                }
                for f in failures
            ]

            # 5. Best action from value estimates
            best = self._get_best_action(domain, task_type)
            if best:
                result["best_action"] = best

            # 6. Improvement trend
            report = self._store.get_improvement_report(domain)
            result["improving"] = report.get("improving", False)

        except Exception as e:
            logger.debug(f"Query failed: {e}")

        return result

    # =========================================================================
    # REFLECT — Mid-execution reflection and adjustment
    # =========================================================================

    def reflect(
        self,
        episode_id: str,
        step: int,
        observation: str,
        unit_name: str,
        analysis: str = "",
    ) -> Dict[str, Any]:
        """Delegate to ReflectionEngine.reflect()."""
        return self._reflection_engine.reflect(
            episode_id=episode_id,
            step=step,
            observation=observation,
            unit_name=unit_name,
            analysis=analysis,
        )

    def post_execution_reflect(
        self,
        episode_id: str,
        goal: str,
        content: str,
        domain: str,
        quality_score: float,
        execution_time: float,
    ) -> None:
        """Delegate to ReflectionEngine.post_execution_reflect()."""
        self._reflection_engine.post_execution_reflect(
            episode_id=episode_id,
            goal=goal,
            content=content,
            domain=domain,
            quality_score=quality_score,
            execution_time=execution_time,
        )

    def mark_reflection_applied(
        self, episode_id: str, step: int, improvement: Optional[float] = None
    ) -> None:
        """Delegate to ReflectionEngine.mark_reflection_applied()."""
        self._reflection_engine.mark_reflection_applied(
            episode_id=episode_id,
            step=step,
            improvement=improvement,
        )

    # =========================================================================
    # TRANSFER — Cross-domain pattern transfer
    # =========================================================================

    def transfer(self, source_domain: str, target_domain: str) -> List[Dict[str, Any]]:
        """
        Find patterns from source_domain that may apply to target_domain.

        Args:
            source_domain: Domain to transfer FROM
            target_domain: Domain to transfer TO

        Returns:
            List of transferable patterns with recommendations
        """
        # Get patterns from source domain with moderate confidence threshold
        patterns = self._store.get_patterns(domain=source_domain, min_confidence=0.4)

        # Transfer any pattern with sufficient confidence — the strict type
        # filter was preventing all transfer (most patterns lack these types).
        _TRANSFERABLE_TYPES = {
            "success_strategy",
            "tool_preference",
            "failure_avoidance",
            "speed_optimization",
            "cross_domain_transfer",
        }

        transferable = []
        for p in patterns:
            is_typed = p.pattern_type in _TRANSFERABLE_TYPES
            is_applicable = target_domain in p.applicable_domains
            is_high_confidence = p.confidence >= 0.7 and p.description
            if is_applicable or is_typed or is_high_confidence:
                transferable.append(
                    {
                        "pattern": p.description,
                        "recommendation": p.recommendation,
                        "confidence": p.confidence,
                        "source": p.source_domain,
                        "type": p.pattern_type,
                    }
                )

                # Auto-register pattern for target domain if not already
                if target_domain not in p.applicable_domains:
                    p.applicable_domains.append(target_domain)
                    self._store.save_pattern(p)

        return transferable

    # =========================================================================
    # EPISODE LIFECYCLE — For multi-step execution tracking
    # =========================================================================

    def start_episode(
        self,
        unit_name: str,
        unit_type: str,
        domain: str,
        task_type: str,
        context: Dict[str, Any],
        parent_episode_id: Optional[str] = None,
    ) -> str:
        """
        Start tracking a multi-step execution episode.

        Returns episode_id for use in subsequent step tracking and reflection.
        """
        episode_id = f"ep_{unit_name}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        episode = ActiveEpisode(
            episode_id=episode_id,
            unit_type=unit_type,
            unit_name=unit_name,
            domain=domain,
            task_type=task_type,
            context=context,
        )
        episode.parent_episode_id = parent_episode_id
        self._active_episodes[episode_id] = episode
        return episode_id

    def record_step(
        self,
        episode_id: str,
        step_name: str,
        action: Dict[str, Any],
        outcome: Dict[str, Any],
        success: bool,
    ) -> None:
        """Record a step within an active episode."""
        episode = self._active_episodes.get(episode_id)
        if episode:
            episode.add_step(step_name, action, outcome, success)

    def end_episode(
        self,
        episode_id: str,
        success: bool,
        quality: float = 0.0,
        cost: float = 0.0,
        outcome: Optional[Dict[str, Any]] = None,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        action_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        End an active episode and persist it.

        Args:
            action_metadata: Rich metadata from the caller (provider, model, domain, etc.)
                             Merged with step-based action summary. This is the primary
                             source of action data — step counts are supplementary.

        Returns the episode_id for reference.
        """
        episode = self._active_episodes.pop(episode_id, None)
        if not episode:
            return self.record(
                unit_name="unknown",
                unit_type="unknown",
                domain="",
                task_type="",
                context={},
                action=action_metadata or {},
                outcome=outcome or {},
                success=success,
                quality=quality,
                cost=cost,
                metadata={"original_episode_id": episode_id},
            )

        execution_time = time.time() - episode.start_time

        # Build action: merge caller metadata (primary) with step summary (supplementary)
        action_summary: Dict[str, Any] = {}
        if action_metadata:
            action_summary.update(action_metadata)
        action_summary["steps"] = len(episode.steps)
        if episode.steps:
            action_summary["step_names"] = [s["name"] for s in episode.steps]
            action_summary["step_successes"] = [s["success"] for s in episode.steps]

        return self.record(
            unit_name=episode.unit_name,
            unit_type=episode.unit_type,
            domain=episode.domain,
            task_type=episode.task_type,
            context=episode.context,
            action=action_summary,
            outcome=outcome or {"steps": episode.steps},
            success=success,
            quality=quality,
            execution_time=execution_time,
            cost=cost,
            error_type=error_type,
            error_message=error_message,
            parent_episode_id=episode.parent_episode_id,
            metadata={"original_episode_id": episode_id},
        )

    # =========================================================================
    # BUILD CONTEXT — Generate learning context string for agent prompts
    # =========================================================================

    def build_context_string(
        self, domain: str, task_type: str = "", unit_name: str = "", goal: str = ""
    ) -> str:
        """
        Build learning context — ADAPTIVE: only inject when there's
        a real signal to give. Silence when the model is already succeeding.

        Injecting generic instructions into a capable model's prompt is
        actively harmful — it wastes context window and causes verbose,
        unfocused output. Only speak when we have:
        - Real failure patterns to correct (success_rate < 90%)
        - Concrete structural template from a highly-relevant prior response
        """
        guidance = self.query(domain, task_type, unit_name=unit_name)

        total = guidance.get("total_episodes", 0)
        rate = guidance.get("success_rate", 0.0)

        if not guidance.get("has_learning"):
            # No episodes for this exact domain — try cross-domain transfer
            for related in _get_related_domains(domain):
                related_guidance = self.query(related, "")
                if (
                    related_guidance.get("has_learning")
                    and related_guidance.get("total_episodes", 0) >= 2
                ):
                    # Borrow patterns from related domain, adapting language
                    related_patterns = related_guidance.get("patterns", [])
                    if related_patterns:
                        hints = []
                        for p in related_patterns[:3]:
                            rec = p.get("recommendation", "")
                            if not rec:
                                continue
                            # Adapt "For X tasks:" to target domain
                            if rec.startswith(f"For {related} tasks:"):
                                rec = f"For {domain} tasks:" + rec[len(f"For {related} tasks:") :]
                            hints.append(rec)
                        if hints:
                            return f"[Guidance adapted from {related} experience]\n" + "\n".join(
                                f"  - {h}" for h in hints
                            )
                    break

            # No cross-domain patterns either — provide task-aware bootstrap
            # guidance based on what the task_type/domain typically needs.
            return self._bootstrap_guidance(domain, task_type)

        # ADAPTIVE GATE: "First, do no harm."
        #
        # When the model succeeds, ANY context injection risks degradation.
        # Distilled lessons can over-constrain the model by nudging it toward
        # one specific approach. Only inject substantive context when there are
        # real failures to correct.
        #
        # Tiers:
        #   1. Succeeding (rate >= 90%): silence — let the model do its thing
        #   2. Cold start (0-2 episodes, no failures): lightweight bootstrap only
        #   3. Failures present: full corrective context (lessons + retrieval)
        has_failures = rate < 0.90 and total >= 3
        is_cold_start = total < 3
        has_early_failures = is_cold_start and total > 0 and rate < 0.90
        is_succeeding = rate >= 0.90 and total >= 1

        if not has_failures and not is_cold_start:
            # Stable success: distilled lessons + reflexion (failure
            # avoidance is valuable even when currently succeeding).
            parts = []
            lessons = self.retrieve_distilled_lessons(
                domain, goal=goal, agent_name=unit_name, top_k=3
            )
            if lessons:
                lesson_lines = [f"- {l['lesson']}" for l in lessons[:3]]
                parts.append(f"[Learned patterns for {domain}]\n" + "\n".join(lesson_lines))

            # Include reflexion even during success — past failure insights
            # prevent regression into previously-fixed failure modes.
            try:
                from .advanced_learning import Reflexion

                reflexion = Reflexion.get_instance()
                past = reflexion.get_relevant_reflections(unit_name=unit_name or domain, limit=2)
                if past:
                    parts.append(
                        "[Avoid past mistakes]\n" + "\n".join(f"  - {r}" for r in past[:2])
                    )
            except Exception:
                pass

            if parts:
                return "\n\n".join(parts)
            return self._maintenance_guidance(domain, task_type)

        # Cold start succeeding: use distilled lessons if available,
        # otherwise fall back to bootstrap guidance.
        if is_cold_start and is_succeeding and not has_early_failures:
            cold_lessons = self.retrieve_distilled_lessons(
                domain, goal=goal, agent_name=unit_name, top_k=3
            )
            if cold_lessons:
                lesson_lines = [f"- {l['lesson']}" for l in cold_lessons[:3]]
                return f"[Learned patterns for {domain}]\n" + "\n".join(lesson_lines)
            return self._bootstrap_guidance(domain, task_type)

        # ── Active correction mode: failures present ─────────────
        # Full context: distilled lessons + retrieval + failure hints.
        # Budget: 2000 chars total.

        MAX_TOTAL = self._config.context_max_total
        MAX_RETRIEVAL = self._config.context_max_retrieval
        MAX_ABSTRACT = self._config.context_max_abstract

        # ── 1. Concrete few-shot retrieval (for failing domains) ────
        retrieval = ""
        try:
            retrieval = (
                self.build_retrieval_context(
                    domain,
                    task_type,
                    goal=goal,
                    agent_name=unit_name,
                )
                or ""
            )
            if retrieval and len(retrieval) < 100:
                retrieval = ""
        except Exception:
            pass

        # ── 2. Abstract guidance (compact) ────────────────────────────
        parts: List[str] = []
        best = self.get_best_approach_for_domain(domain, task_type)

        if best:
            feats = best.get("structural_features", [])
            if feats:
                parts.append(f"[Proven {domain} approach: {', '.join(feats[:5])}]")

        if has_failures or has_early_failures:
            failures = guidance.get("failure_analysis", [])
            if failures:
                seen_descs: set = set()
                unique_failures: list = []
                for f in failures:
                    desc = f.get("description", f.get("error_type", ""))
                    if desc and desc not in seen_descs:
                        seen_descs.add(desc)
                        unique_failures.append(desc[:100])
                if unique_failures:
                    parts.append(f"Gaps to address ({rate:.0%} success, {total} tasks):")
                    for desc in unique_failures[:2]:
                        parts.append(f"  - {desc}")

        # Per-(state,action) lessons
        try:
            stored_lessons = self._store.get_lessons(
                domain=domain,
                unit_name=unit_name if unit_name else None,
                limit=3,
            )
            if stored_lessons:
                for sl in stored_lessons[:2]:
                    parts.append(f"  - {sl}")
        except Exception:
            pass

        abstract = "\n".join(parts)
        if len(abstract) > MAX_ABSTRACT:
            abstract = abstract[:MAX_ABSTRACT]

        # ── 3. Distilled lessons (mem0 pattern) ──────────────────────────
        distilled = ""
        try:
            dist_lessons = self.retrieve_distilled_lessons(
                domain, goal=goal, agent_name=unit_name, top_k=3
            )
            if dist_lessons:
                d_parts = ["[Learned from prior tasks]"]
                for dl in dist_lessons[:3]:
                    if dl.get("lesson"):
                        d_parts.append(f"  - {dl['lesson']}")
                if len(d_parts) > 1:
                    distilled = "\n".join(d_parts)
        except Exception:
            pass

        # ── 3b. Reflexion retrieval (past failure insights) ─────────────
        reflections_str = ""
        if has_failures or has_early_failures:
            try:
                from .advanced_learning import Reflexion

                reflexion = Reflexion.get_instance()
                past = reflexion.get_relevant_reflections(unit_name=unit_name or "auto", limit=2)
                if past:
                    reflections_str = "[Past failure insights]\n" + "\n".join(
                        f"  - {r}" for r in past[:2]
                    )
            except Exception:
                pass

        # ── 4. Assemble: distilled → reflections → retrieval → abstract ───
        MAX_TOTAL_EXPANDED = 2000  # more budget now that we have richer signals
        sections = []
        if distilled:
            sections.append(distilled[:400])
        if reflections_str:
            sections.append(reflections_str[:400])
        if retrieval:
            sections.append(retrieval[:MAX_RETRIEVAL])
        if abstract:
            sections.append(abstract)

        if not sections:
            return ""

        result = "\n".join(sections)
        if len(result) > MAX_TOTAL_EXPANDED:
            result = result[:MAX_TOTAL_EXPANDED]
        return result

    @staticmethod
    def _bootstrap_guidance(domain: str, task_type: str = "") -> str:
        """
        Zero-episode bootstrap: provide structural guidance for domains
        that have never been seen before. This prevents cold-start silence
        where the model gets no learning context at all.

        Based on domain conventions — what successful responses in each
        domain typically contain. Fades out once real episodes exist.
        """
        _DOMAIN_BOOTSTRAP: Dict[str, str] = {
            "coding": (
                "[Quality guidance for coding tasks]\n"
                "  - Wrap ALL code in markdown fenced blocks (```python)\n"
                "  - Include complete, runnable code with class/function definitions\n"
                "  - Add test functions (def test_*) with 5+ assertions\n"
                "  - Use section headings to organize (## Implementation, ## Tests)"
            ),
            "algorithms": (
                "[Quality guidance for algorithm tasks]\n"
                "  - Include complete implementations in code blocks (```python)\n"
                "  - Add test functions (def test_*) with assertions verifying correctness\n"
                "  - Analyze time/space complexity with O() notation\n"
                "  - Use section headings (## Algorithm, ## Analysis, ## Tests)"
            ),
            "compiler_design": (
                "[Quality guidance for compiler/interpreter tasks]\n"
                "  - Include complete implementations: tokenizer, parser, evaluator\n"
                "  - Define proper classes (Token, AST nodes, Environment)\n"
                "  - Add test functions covering arithmetic, functions, closures\n"
                "  - Include 5+ assertions validating each language feature"
            ),
            "data_science": (
                "[Quality guidance for data science tasks]\n"
                "  - Include complete class implementations with fit/predict methods\n"
                "  - Add mathematical formulations (equations, proofs)\n"
                "  - Write test functions with assertions verifying correctness\n"
                "  - Use code blocks (```python) for all implementations"
            ),
            "system_design": (
                "[Quality guidance for system design tasks]\n"
                "  - Wrap code examples in markdown fenced blocks (```python)\n"
                "  - Include architecture diagrams or structured descriptions\n"
                "  - Analyze trade-offs (latency, throughput, consistency)\n"
                "  - Use tables for comparisons where appropriate"
            ),
            "math": (
                "[Quality guidance for math tasks]\n"
                "  - Include formal definitions and theorem statements\n"
                "  - Provide step-by-step proofs with justification\n"
                "  - Use mathematical notation and formulas\n"
                "  - Add concrete examples to illustrate abstract concepts"
            ),
            "research": (
                "[Quality guidance for research tasks]\n"
                "  - Structure with clear sections (Introduction, Analysis, Conclusion)\n"
                "  - Cite sources and provide evidence for claims\n"
                "  - Compare multiple perspectives or approaches\n"
                "  - Include a summary/conclusion section"
            ),
        }
        return _DOMAIN_BOOTSTRAP.get(domain, "")

    def _maintenance_guidance(self, domain: str, task_type: str) -> str:
        """Lightweight hint for high-performing domains.

        When the adaptive gate would have silenced all context (model already
        performing well), provide a short structural hint from the best
        historical episode. ~200-400 chars max.
        """
        best = self.get_best_approach_for_domain(domain, task_type)
        if not best:
            return ""
        feats = best.get("structural_features", [])
        if not feats:
            return ""
        return f"[Proven approach for {domain}: {', '.join(feats[:5])}]"[:400]

    # =========================================================================
    # IMPROVEMENT REPORT
    # =========================================================================

    def get_best_approach_for_domain(
        self, domain: str, task_type: str = ""
    ) -> Optional[Dict[str, Any]]:
        """
        Return the best historical approach for a domain: what action metadata,
        structural features, and quality score characterized the top episode.

        This gives the system a concrete template to follow, not just averages.
        """
        try:
            episodes = self._store.query_episodes(domain=domain, limit=50)
            successful = [e for e in episodes if e.success and e.quality > 0]
            if not successful:
                return None

            best = max(successful, key=lambda e: e.quality)
            outcome = best.outcome or {}

            approach: Dict[str, Any] = {
                "quality": best.quality,
                "execution_time": best.execution_time,
                "structural_features": [],
                "outline": None,
                "action": best.action,
            }

            # Extract outline from stored digest if available
            excerpt = outcome.get("response_excerpt", "")
            if "Outline:" in excerpt:
                for line in excerpt.split("\n"):
                    if line.startswith("Outline:"):
                        approach["outline"] = line[len("Outline:") :].strip()
                        break

            if outcome.get("has_code"):
                approach["structural_features"].append(
                    f"{outcome.get('code_block_count', 'multiple')} code blocks"
                )
            if outcome.get("has_headings"):
                approach["structural_features"].append("section headings")
            if outcome.get("has_citations"):
                approach["structural_features"].append("cited references")
            if outcome.get("has_math") and domain not in (
                "coding",
                "algorithms",
                "compiler_design",
            ):
                approach["structural_features"].append("math formulations")
            if outcome.get("has_table"):
                approach["structural_features"].append("comparison tables")
            wc = outcome.get("word_count", 0)
            if wc:
                approach["structural_features"].append(f"~{wc} words")

            return approach
        except Exception as e:
            logger.debug(f"get_best_approach_for_domain failed: {e}")
            return None

    async def llm_judge_quality(
        self,
        goal: str,
        content: str,
        domain: str,
        heuristic_score: float,
    ) -> float:
        """
        Use LLM judge to score response quality. Returns 0.0-1.0.
        Calls llm_judge_quality_with_feedback() and returns the score only.
        """
        result = await self.llm_judge_quality_with_feedback(goal, content, domain, heuristic_score)
        return result.get("score", heuristic_score)

    async def llm_judge_quality_with_feedback(
        self,
        goal: str,
        content: str,
        domain: str,
        heuristic_score: float,
    ) -> Dict[str, Any]:
        """
        Use a strong LLM (Sonnet) to evaluate response quality with a detailed
        rubric. Returns structured feedback that can inform lesson extraction.

        Returns:
            {
                "score": float (0.0-1.0),
                "accuracy": float,
                "completeness": float,
                "structure": float,
                "actionability": float,
                "depth": float,
                "strengths": [str, ...],
                "improvements": [str, ...],
                "verdict": str,
            }
        """
        fallback = {
            "score": heuristic_score,
            "verdict": "heuristic-only",
            "strengths": [],
            "improvements": [],
        }
        if len(content) < 20:
            return fallback

        try:
            import asyncio as _aio
            import json as _json

            digest = self._build_judge_digest(content, goal, domain)

            prompt = (
                f"You are an expert evaluator. Score this {domain} response on 5 dimensions "
                f"(each 0.0-1.0) and provide specific feedback.\n\n"
                f"TASK: {goal[:500]}\n\n"
                f"{digest}\n\n"
                f"RUBRIC:\n"
                f"  accuracy:      Are facts correct? Are claims realistic and verifiable?\n"
                f"  completeness:  Does it cover ALL aspects the task requested?\n"
                f"  structure:     Is it well-organized with clear sections and flow?\n"
                f"  actionability: Can someone directly USE this output for its intended purpose?\n"
                f"  depth:         Does it go beyond surface-level? Specific details, not generic?\n\n"
                f"Reply with ONLY a JSON object:\n"
                f'{{"accuracy": 0.X, "completeness": 0.X, "structure": 0.X, '
                f'"actionability": 0.X, "depth": 0.X, '
                f'"strengths": ["specific strength 1", "specific strength 2"], '
                f'"improvements": ["what would make this better 1", "what is missing"]}}'
            )

            def _sync_judge(p: str) -> str:
                import anthropic

                client = anthropic.Anthropic()
                resp = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=300,
                    messages=[{"role": "user", "content": p}],
                )
                return getattr(resp.content[0], "text", "").strip()

            loop = _aio.get_running_loop()
            text = await loop.run_in_executor(None, _sync_judge, prompt)

            if "```" in text:
                text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")

            parsed = _json.loads(text)

            dims = ["accuracy", "completeness", "structure", "actionability", "depth"]
            for d in dims:
                parsed[d] = max(0.0, min(1.0, float(parsed.get(d, 0.5))))

            parsed["score"] = round(sum(parsed[d] for d in dims) / len(dims), 3)
            parsed["verdict"] = "llm-judged"
            parsed.setdefault("strengths", [])
            parsed.setdefault("improvements", [])

            return parsed

        except Exception as e:
            logger.debug(f"LLM judge with feedback failed: {e}")
            return fallback

    # =========================================================================
    # THOMPSON SAMPLING — Principled Bayesian exploration
    # =========================================================================

    def _build_judge_digest(self, content: str, goal: str, domain: str) -> str:
        """
        Build a structural digest of a response for the LLM judge.

        Instead of an arbitrary character cutoff, extract:
        1. Table of contents (headings hierarchy)
        2. Structural stats (code blocks, tables, citations, math, word count)
        3. Representative samples from distinct sections
        4. Opening paragraph (framing quality)
        5. A code sample if present (correctness signal)
        """
        lines = content.split("\n")
        words = content.split()

        # 1. Extract headings as table of contents (skip code blocks)
        headings = []
        in_code = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```"):
                in_code = not in_code
                continue
            if in_code:
                continue
            if stripped.startswith("#"):
                level = len(stripped) - len(stripped.lstrip("#"))
                title = stripped.lstrip("# ").strip()
                if title:
                    headings.append(("  " * (level - 1)) + f"- {title}")

        # 2. Structural inventory
        code_blocks = _CODE_BLOCK_RE.findall(content)
        tables = re.findall(r"(\|.+\|(?:\n\|.+\|)+)", content)
        citations = _CITATION_RE.findall(content)
        math_exprs = re.findall(r"[=∑∫∂∇λαβγ]|\\frac|O\(n", content)
        numbered_lists = re.findall(r"(?:^|\n)\s*\d+\.\s+(.+)", content)

        stats_lines = [
            f"RESPONSE STRUCTURE ({len(words)} words, {len(content)} chars):",
            f"  Sections: {len(headings)}",
            f"  Code blocks: {len(code_blocks)}",
            f"  Tables: {len(tables)}",
            f"  Citations/references: {len(citations)}",
            f"  Math expressions: {len(math_exprs)}",
            f"  Numbered list items: {len(numbered_lists)}",
        ]

        # 3. Table of contents
        if headings:
            stats_lines.append("\nTABLE OF CONTENTS:")
            stats_lines.extend(headings[:20])

        # 4. Opening paragraph (first non-empty, non-heading block up to 300 chars)
        opening = ""
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                opening = stripped
                break
        if opening:
            stats_lines.append(f"\nOPENING:\n  {opening[:300]}")

        # 5. Representative samples — one paragraph from each of the first
        #    few sections (not just start/middle which may land mid-sentence)
        sections: List[Tuple[str, str]] = []
        current_heading = "(intro)"
        current_text: List[str] = []
        in_code_block = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```"):
                in_code_block = not in_code_block
            if not in_code_block and stripped.startswith("#") and stripped.lstrip("# ").strip():
                if current_text:
                    body = "\n".join(current_text).strip()
                    if len(body) > 50:
                        sections.append((current_heading, body))
                current_heading = stripped.lstrip("# ").strip()
                current_text = []
            else:
                current_text.append(line)
        if current_text:
            body = "\n".join(current_text).strip()
            if len(body) > 50:
                sections.append((current_heading, body))

        if sections:
            # Pick ~3 evenly spaced sections
            indices = [0]
            if len(sections) > 2:
                indices.append(len(sections) // 2)
            if len(sections) > 1:
                indices.append(len(sections) - 1)

            stats_lines.append("\nSAMPLE EXCERPTS (one from each section):")
            for idx in indices:
                heading, body = sections[idx]
                # Take first meaningful paragraph (skip blanks)
                paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
                sample = paragraphs[0] if paragraphs else body
                stats_lines.append(f"\n  [{heading}]")
                stats_lines.append(f"  {sample[:400]}")

        # 6. One code sample if present (first block, truncated)
        if code_blocks:
            block = code_blocks[0]
            block_lines = block.split("\n")
            if len(block_lines) > 15:
                sample_code = (
                    "\n".join(block_lines[:12]) + f"\n  ... ({len(block_lines)} lines total)"
                )
            else:
                sample_code = block
            stats_lines.append(f"\nCODE SAMPLE:\n{sample_code[:500]}")

        return "\n".join(stats_lines)

    # =========================================================================
    # THOMPSON SAMPLING — Principled Bayesian exploration
    # =========================================================================

    def _get_arm_stats(self, domain: str) -> Dict[str, Dict[str, float]]:
        """
        Build Beta distribution parameters for each arm from episode history.
        Arms: temperature values, model choices, paradigm choices.
        Returns {arm_key: {"alpha": float, "beta": float, "n": int}}
        """
        episodes = self._store.query_episodes(domain=domain, limit=100)
        arms: Dict[str, Dict[str, float]] = {}

        for ep in episodes:
            if not ep.success or ep.quality <= 0:
                continue
            action = ep.action or {}
            q = ep.quality

            # Temperature arm
            temp = action.get("temperature")
            if temp is not None:
                key = f"temp={temp}"
                if key not in arms:
                    arms[key] = {"alpha": 1.0, "beta": 1.0, "n": 0}
                arms[key]["alpha"] += q
                arms[key]["beta"] += 1.0 - q
                arms[key]["n"] += 1

            # Model arm
            model = action.get("model")
            if model and model != "default":
                key = f"model={model}"
                if key not in arms:
                    arms[key] = {"alpha": 1.0, "beta": 1.0, "n": 0}
                arms[key]["alpha"] += q
                arms[key]["beta"] += 1.0 - q
                arms[key]["n"] += 1

            # Paradigm arm
            paradigm = action.get("paradigm")
            if paradigm:
                key = f"paradigm={paradigm}"
                if key not in arms:
                    arms[key] = {"alpha": 1.0, "beta": 1.0, "n": 0}
                arms[key]["alpha"] += q
                arms[key]["beta"] += 1.0 - q
                arms[key]["n"] += 1

            # Tool set arm
            tools = action.get("tools_used")
            if tools and isinstance(tools, str):
                key = f"tools={tools}"
                if key not in arms:
                    arms[key] = {"alpha": 1.0, "beta": 1.0, "n": 0}
                arms[key]["alpha"] += q
                arms[key]["beta"] += 1.0 - q
                arms[key]["n"] += 1

        return arms

    def _thompson_sample(self, arms: Dict[str, Dict[str, float]], prefix: str) -> Optional[str]:
        """
        Thompson Sampling: sample from Beta(alpha, beta) for each arm
        matching prefix, return the arm with highest sample.
        """
        import random as _rng

        candidates = {k: v for k, v in arms.items() if k.startswith(prefix)}
        if not candidates:
            return None

        best_key = None
        best_sample = -1.0
        for key, stats in candidates.items():
            sample = _rng.betavariate(stats["alpha"], stats["beta"])
            if sample > best_sample:
                best_sample = sample
                best_key = key

        return best_key

    # =========================================================================
    # EXECUTION STEERING — Learning changes actual behavior
    # =========================================================================

    def get_optimal_execution_params(
        self, domain: str, task_type: str = "", goal: str = ""
    ) -> Dict[str, Any]:
        """
        Return optimal execution parameters derived from Thompson Sampling
        over learned Beta distributions.

        Steers: model, temperature, paradigm, tool selection.
        Principled exploration via posterior sampling — no epsilon needed.

        Returns dict with:
            model: str — recommended model (or None for default)
            temperature: float — recommended temperature
            paradigm: str — recommended paradigm (direct/relay/debate/refinement)
            tools_hint: list — recommended tools to enable
            strategy: str — how the decision was made
            exploration: bool — whether this is an under-explored arm
            exploration_reason: str — what's being explored
        """
        import random

        params: Dict[str, Any] = {
            "model": None,
            "temperature": None,
            "paradigm": None,
            "tools_hint": None,
            "strategy": "default",
            "exploration": False,
            "exploration_reason": "",
        }

        episodes = self._store.query_episodes(domain=domain, limit=50)
        successful = [e for e in episodes if e.success and e.quality > 0]

        if len(successful) < 2:
            paradigms = ["direct", "relay", "debate", "refinement"]
            temps = [0.3, 0.5, 0.7]
            if len(successful) == 0:
                params["temperature"] = random.choice(temps)
                params["paradigm"] = random.choice(paradigms)
                params["exploration"] = True
                params["exploration_reason"] = "cold_start: seeding first arms"
                params["strategy"] = "thompson_cold_start"
                return params
            params["exploration"] = True
            params["temperature"] = random.choice(temps)
            params["paradigm"] = random.choice(paradigms)
            params["exploration_reason"] = f"cold_start: only {len(successful)} episodes"
            params["strategy"] = "thompson_cold_start"
            params["tools_hint"] = self._recommend_tools(domain, goal)
            return params

        # --- Thompson Sampling over learned arms ---
        arms = self._get_arm_stats(domain)

        # Temperature selection via Thompson Sampling
        temp_arm = self._thompson_sample(arms, "temp=")
        if temp_arm:
            params["temperature"] = float(temp_arm.split("=")[1])
            stats = arms[temp_arm]
            if stats["n"] < 3:
                params["exploration"] = True
                params["exploration_reason"] = (
                    f"thompson: {temp_arm} under-explored (n={stats['n']})"
                )

        # Model selection via Thompson Sampling
        model_arm = self._thompson_sample(arms, "model=")
        if model_arm:
            params["model"] = model_arm.split("=", 1)[1]
            stats = arms[model_arm]
            if stats["n"] < 3:
                params["exploration"] = True
                params["exploration_reason"] = (
                    f"thompson: {model_arm} under-explored (n={stats['n']})"
                )

        # Paradigm selection via Thompson Sampling
        paradigm_arm = self._thompson_sample(arms, "paradigm=")
        if paradigm_arm:
            params["paradigm"] = paradigm_arm.split("=", 1)[1]
            stats = arms[paradigm_arm]
            if stats["n"] < 3:
                params["exploration"] = True
                params["exploration_reason"] = f"thompson: {paradigm_arm} under-explored"

        # Tool routing based on domain patterns
        params["tools_hint"] = self._recommend_tools(domain, goal)

        # --- Inject unseen arms for exploration ---
        # If we've never tried certain arms, inject them so Thompson can explore
        all_temps_tried = {k for k in arms if k.startswith("temp=")}
        all_paradigms_tried = {k for k in arms if k.startswith("paradigm=")}

        available_paradigms = {"direct", "relay", "debate", "refinement"}
        tried_paradigms = {k.split("=")[1] for k in all_paradigms_tried}
        untried = available_paradigms - tried_paradigms

        if untried and random.random() < 0.3:  # 30% chance to try new paradigm
            chosen = random.choice(list(untried))
            params["paradigm"] = chosen
            params["exploration"] = True
            params["exploration_reason"] = f"thompson: trying untried paradigm={chosen}"

        if not all_temps_tried and not params.get("temperature"):
            params["temperature"] = random.choice([0.3, 0.5, 0.7])
            params["exploration"] = True
            params["exploration_reason"] = "thompson: no temp history, seeding"

        params["strategy"] = "thompson_sampling" if len(successful) >= 2 else "default"
        return params

    def _recommend_tools(self, domain: str, goal: str = "") -> Optional[List[str]]:
        """
        Recommend which tools to enable based on domain and learned patterns.
        Returns tool list or None for default.
        """
        domain_tool_map = {
            "coding": ["code_interpreter", "file_write"],
            "research": ["web_search", "arxiv_search"],
            "economics": ["web_search", "calculator"],
            "data_science": ["code_interpreter", "calculator", "file_write"],
            "system_design": [],
        }

        base = domain_tool_map.get(domain)
        if base is None:
            return None

        # Check if web_search helped in past episodes
        episodes = self._store.query_episodes(domain=domain, limit=20)
        web_eps = [
            e
            for e in episodes
            if e.action.get("tools_used") and "web_search" in str(e.action.get("tools_used", ""))
        ]
        non_web_eps = [e for e in episodes if e.success and e.quality > 0 and e not in web_eps]

        if web_eps and non_web_eps:
            web_q = sum(e.quality for e in web_eps if e.quality > 0) / max(len(web_eps), 1)
            non_q = sum(e.quality for e in non_web_eps) / max(len(non_web_eps), 1)
            if web_q > non_q + 0.05 and "web_search" not in base:
                base = base + ["web_search"]

        return base if base else None

    # =========================================================================
    # RETRIEVAL-AUGMENTED LEARNING — Retrieve actual prior response excerpts
    # =========================================================================

    def retrieve_similar_responses(
        self,
        domain: str,
        task_type: str = "",
        goal: str = "",
        top_k: int = 2,
        agent_name: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Retrieve excerpts from the best prior responses using vector embeddings.

        Uses sentence-transformers (384-dim) for semantic similarity when available,
        falls back to TF-IDF. Also retrieves distilled lessons (mem0 pattern).
        Supports per-agent filtering.
        """

        emb = _get_embeddings()
        now = time.time()
        goal_vec = None

        # ── Try embedding-based retrieval first ──────────────────────────
        if goal and emb.available:
            goal_vec = emb.embed(goal)

        if goal_vec is not None:
            scored = self._embedding_retrieval(domain, goal_vec, agent_name, now)
        else:
            scored = self._tfidf_retrieval(domain, goal, agent_name, now)

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, ep in scored[:top_k]:
            outcome = ep.outcome or {}
            actual_content = ""
            full_content_len = 0
            for _ck in ("content", "response", "result"):
                _cv = outcome.get(_ck, "")
                if isinstance(_cv, str) and len(_cv) > 200:
                    full_content_len = len(_cv)
                    actual_content = _cv[:1200]
                    break
            # content_length from outcome metadata is the TRUE full length
            # (the content field itself may be truncated to 1500 chars)
            stored_len = outcome.get("content_length", 0)
            if stored_len and stored_len > full_content_len:
                full_content_len = stored_len

            excerpt = outcome.get("response_excerpt", "") or actual_content
            if isinstance(excerpt, str) and len(excerpt) > 1000:
                excerpt = excerpt[:1000] + "..."

            if not excerpt:
                goal_text = str(ep.context.get("goal", ep.context.get("message", "")))
                if not goal_text:
                    continue
                excerpt = f"[Task: {goal_text[:200]}] Quality={ep.quality:.2f}"

            results.append(
                {
                    "domain": ep.domain,
                    "task_type": ep.task_type,
                    "quality": ep.quality,
                    "relevance_score": round(score, 3),
                    "excerpt": excerpt,
                    "actual_content": actual_content[:1200] if actual_content else "",
                    "full_content_len": full_content_len,
                    "goal_preview": str(ep.context.get("goal", ep.context.get("message", "")))[
                        :100
                    ],
                    "agent_name": ep.unit_name,
                }
            )

        return results

    def _embedding_retrieval(
        self,
        domain: str,
        goal_vec: Any,
        agent_name: str,
        now: float,
    ) -> List[Tuple[float, EpisodeRecord]]:
        """Retrieve episodes using embedding cosine similarity."""
        import math as _math

        import numpy as np

        from .embeddings import EmbeddingService

        emb = _get_embeddings()
        scored: List[Tuple[float, EpisodeRecord]] = []

        # Get episodes with stored embeddings
        ep_embs = self._store.get_episodes_with_embeddings(domain=domain, limit=200)
        if not ep_embs:
            for alt_domain in _get_related_domains(domain):
                ep_embs = self._store.get_episodes_with_embeddings(domain=alt_domain, limit=100)
                if ep_embs:
                    break
        if not ep_embs:
            ep_embs = self._store.get_episodes_with_embeddings(domain=None, limit=100)

        for ep, emb_blob in ep_embs:
            if ep.quality <= 0:
                continue
            ep_vec = EmbeddingService.deserialize(emb_blob)
            similarity = float(np.dot(goal_vec, ep_vec))

            age_h = max((now - ep.timestamp) / 3600, 0.01)
            recency = 0.15 * _math.exp(-age_h / 6)

            has_content = any(
                isinstance(ep.outcome.get(k, ""), str) and len(ep.outcome.get(k, "")) > 200
                for k in ("content", "response_excerpt")
            )
            content_bonus = 0.1 if has_content else 0.0

            # Per-agent boost: same agent gets +0.1
            agent_bonus = 0.1 if agent_name and ep.unit_name == agent_name else 0.0

            score = similarity * 0.45 + ep.quality * 0.30 + recency + content_bonus + agent_bonus
            scored.append((score, ep))

        return scored

    def _tfidf_retrieval(
        self,
        domain: str,
        goal: str,
        agent_name: str,
        now: float,
    ) -> List[Tuple[float, EpisodeRecord]]:
        """Fallback TF-IDF retrieval when embeddings unavailable."""
        import math as _math
        from collections import Counter as _Counter

        episodes = self._store.query_episodes(domain=domain, success_only=True, limit=50)
        if not episodes:
            for alt_domain in _get_related_domains(domain):
                episodes = self._store.query_episodes(
                    domain=alt_domain, success_only=True, limit=20
                )
                if episodes:
                    break
            if not episodes:
                episodes = self._store.query_episodes(domain="general", success_only=True, limit=20)

        if not episodes:
            return []

        episodes = [
            ep
            for ep in episodes
            if ep.outcome
            and (
                any(
                    isinstance(ep.outcome.get(k, ""), str) and len(ep.outcome.get(k, "")) > 100
                    for k in ("content", "response", "response_excerpt")
                )
                or (ep.context.get("goal") or ep.context.get("message"))
            )
        ]
        if not episodes:
            return []

        scored: List[Tuple[float, EpisodeRecord]] = []

        _stop = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "and",
            "or",
            "not",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "it",
            "this",
            "that",
            "do",
            "does",
            "did",
            "have",
            "has",
            "had",
            "will",
            "would",
            "could",
            "should",
            "may",
            "can",
            "must",
            "shall",
            "from",
            "by",
            "as",
            "if",
            "but",
            "so",
            "all",
            "each",
            "every",
            "any",
            "no",
            "your",
            "my",
            "their",
            "our",
            "its",
            "use",
            "using",
        }

        def _tokenize(text: str) -> List[str]:
            return [w for w in re.findall(r"\b[a-z]{3,}\b", text.lower()) if w not in _stop]

        if goal:
            goal_tokens = _tokenize(goal)
            goal_tf = _Counter(goal_tokens)
            doc_tokens_list = []
            for ep in episodes:
                ctx_text = str(ep.context.get("goal", ep.context.get("message", "")))
                doc_tokens_list.append(_tokenize(ctx_text))

            n_docs = len(doc_tokens_list) + 1
            df: Dict[str, int] = defaultdict(int)
            for dtoks in doc_tokens_list:
                for w in set(dtoks):
                    df[w] += 1
            for w in set(goal_tokens):
                df[w] += 1

            def _tfidf_vec(tf: Dict[str, int]) -> Dict[str, float]:
                return {w: c * _math.log(n_docs / max(df.get(w, 1), 1)) for w, c in tf.items()}

            def _cosine(v1: Dict[str, float], v2: Dict[str, float]) -> float:
                shared = set(v1) & set(v2)
                if not shared:
                    return 0.0
                dot = sum(v1[w] * v2[w] for w in shared)
                mag1 = _math.sqrt(sum(x * x for x in v1.values()))
                mag2 = _math.sqrt(sum(x * x for x in v2.values()))
                return dot / max(mag1 * mag2, 1e-10)

            goal_vec = _tfidf_vec(goal_tf)

            for i, ep in enumerate(episodes):
                if ep.quality <= 0:
                    continue
                doc_tf = _Counter(doc_tokens_list[i])
                doc_vec = _tfidf_vec(doc_tf)
                relevance = _cosine(goal_vec, doc_vec)
                age_h = max((now - ep.timestamp) / 3600, 0.01)
                recency = 0.15 * _math.exp(-age_h / 6)
                has_content = any(
                    isinstance(ep.outcome.get(k, ""), str) and len(ep.outcome.get(k, "")) > 200
                    for k in ("content", "response_excerpt")
                )
                content_bonus = 0.1 if has_content else 0.0
                agent_bonus = 0.1 if agent_name and ep.unit_name == agent_name else 0.0
                score = ep.quality * 0.4 + relevance * 0.4 + recency + content_bonus + agent_bonus
                scored.append((score, ep))
        else:
            for ep in episodes:
                if ep.quality <= 0:
                    continue
                age_h = max((now - ep.timestamp) / 3600, 0.01)
                recency = 0.15 * _math.exp(-age_h / 6)
                has_content = any(
                    isinstance(ep.outcome.get(k, ""), str) and len(ep.outcome.get(k, "")) > 200
                    for k in ("content", "response_excerpt")
                )
                content_bonus = 0.1 if has_content else 0.0
                agent_bonus = 0.1 if agent_name and ep.unit_name == agent_name else 0.0
                scored.append((ep.quality * 0.75 + recency + content_bonus + agent_bonus, ep))

        return scored

    def build_retrieval_context(
        self,
        domain: str,
        task_type: str = "",
        goal: str = "",
        agent_name: str = "",
    ) -> str:
        """
        Build few-shot learning context from the best prior responses.

        Uses vector embeddings for semantic matching when available,
        falls back to TF-IDF. Supports per-agent scoping.
        """
        similar = self.retrieve_similar_responses(
            domain,
            task_type,
            goal,
            top_k=3,
            agent_name=agent_name,
        )
        if not similar:
            return ""

        # Only include high-quality examples (Q >= 0.5)
        good = [r for r in similar if r["quality"] >= 0.5]
        if not good:
            return ""

        # Pick the single best by combined quality + relevance.
        best_resp = max(good, key=lambda r: r.get("relevance_score", r["quality"]))
        excerpt = best_resp.get("excerpt", "")
        if not excerpt or len(excerpt) < 50:
            return ""

        relevance = best_resp.get("relevance_score", 0.0)
        best_quality = best_resp["quality"]

        # ── Same-task detection ──
        # If relevance > 0.92, the retrieved response is for a nearly identical
        # task.  Injecting the actual answer makes the LLM copy/abbreviate
        # instead of thinking fresh (anti-pattern: RAG-as-summarizer).
        # Instead, inject structural guidance so the LLM aims for equal or
        # better quality WITHOUT seeing the previous text.
        if relevance > 0.92:
            content_len = best_resp.get("full_content_len", 0)
            if not content_len:
                actual_content = best_resp.get("actual_content", "")
                content_len = len(actual_content) if actual_content else len(excerpt)
            parts: List[str] = [
                f"[QUALITY BASELINE — your previous response scored Q={best_quality:.2f} "
                f"and was {content_len} chars]",
                f"Your response MUST be at least {content_len} characters.",
                "Cover ALL aspects requested. Do NOT summarize or abbreviate.",
            ]
            return "\n".join(parts)

        # ── Different-task: inject as few-shot example (normal RAG) ──
        parts: List[str] = []
        parts.append(f"[REFERENCE FROM SIMILAR TASK (Q={best_quality:.2f})]")

        if best_resp.get("goal_preview"):
            parts.append(f"Task: {best_resp['goal_preview']}")

        # Prefer actual response content over metadata digest.
        actual_content = best_resp.get("actual_content", "")
        if actual_content and len(actual_content) > len(excerpt):
            parts.append(f"Response:\n{actual_content[:1000]}")
        else:
            parts.append(f"Excerpt:\n{excerpt[:1000]}")

        return "\n".join(parts)

    # =========================================================================
    # ADAPTIVE JUDGE SCHEDULING
    # =========================================================================

    def should_judge_inline(self, domain: str) -> bool:
        """
        Decide whether to judge inline (blocking) or background (async).

        Cold start (<10 LLM-judged episodes): inline — we need accurate signal.
        Stable (>=10 LLM-judged episodes): background — heuristic is calibrated.
        """
        conn = self._store._get_conn()
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM episodes WHERE domain = ? "
            "AND outcome LIKE '%llm_judged%'",
            (domain,),
        ).fetchone()
        judged_count = row["cnt"] if row else 0
        return judged_count < 10

    # =========================================================================
    # BACKGROUND LLM JUDGE — Fire-and-forget quality assessment
    # =========================================================================

    def schedule_background_judge(
        self,
        episode_id: str,
        goal: str,
        content: str,
        domain: str,
        heuristic_quality: float,
        unit_name: str = "",
    ) -> None:
        """
        Schedule an LLM quality judge as a background task.
        Does NOT block the response. Updates the episode record asynchronously.

        The judge feedback is also passed to fact distillation so lessons
        are grounded in expert evaluation (not just the raw output).
        """
        import asyncio

        async def _judge_and_update() -> None:
            try:
                feedback = await self.llm_judge_quality_with_feedback(
                    goal=goal,
                    content=content,
                    domain=domain,
                    heuristic_score=heuristic_quality,
                )
                llm_score = feedback.get("score", heuristic_quality)
                w_llm, w_heur = self._blend_overrides.get(
                    domain, (self._config.judge_llm_weight, self._config.judge_heuristic_weight)
                )
                blended = llm_score * w_llm + heuristic_quality * w_heur

                self._store.update_episode_quality(
                    episode_id=episode_id,
                    quality=blended,
                    outcome_patch={
                        "llm_judged": True,
                        "llm_score": round(llm_score, 3),
                        "heuristic_quality": round(heuristic_quality, 3),
                        "judge_feedback": {
                            k: feedback[k]
                            for k in (
                                "accuracy",
                                "completeness",
                                "structure",
                                "actionability",
                                "depth",
                                "strengths",
                                "improvements",
                            )
                            if k in feedback
                        },
                    },
                )

                episodes = self._store.query_episodes(domain=domain, limit=1)
                if episodes:
                    ep = episodes[0]
                    self._update_values(domain, ep.task_type, ep.action, ep.success, blended)

                logger.info(
                    "Judge: %s score %.2f→%.2f (acc=%.2f comp=%.2f struct=%.2f act=%.2f depth=%.2f)",
                    episode_id,
                    heuristic_quality,
                    blended,
                    feedback.get("accuracy", 0),
                    feedback.get("completeness", 0),
                    feedback.get("structure", 0),
                    feedback.get("actionability", 0),
                    feedback.get("depth", 0),
                )

                # Now run judge-informed distillation (if content is rich enough)
                if len(content) > 300 and blended >= 0.2:
                    self._schedule_fact_distillation(
                        episode_id,
                        goal,
                        content,
                        domain,
                        unit_name or "auto",
                        judge_feedback=feedback,
                    )

            except Exception as e:
                logger.debug(f"Background judge failed for {episode_id}: {e}")

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_judge_and_update())
        except RuntimeError:
            logger.debug("No event loop for background judge, skipping")

    # =========================================================================
    # FACT DISTILLATION — mem0 pattern: extract lessons from episodes
    # =========================================================================

    def _schedule_fact_distillation(
        self,
        episode_id: str,
        goal: str,
        content: str,
        domain: str,
        agent_name: str,
        judge_feedback: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Extract facts from an episode using LLM (mem0 pattern).

        Cold start (<5 distilled lessons in domain): runs synchronously
        via thread pool so lessons are available for the very next task.
        After warm-up: runs as async background task.

        If judge_feedback is provided, it is included in the distillation
        prompt so lessons are grounded in the expert evaluation.
        """
        existing_lessons = self._store.get_distilled_lessons(domain=domain, limit=5)
        is_cold = len(existing_lessons) < 5

        if is_cold:
            self._sync_distill(episode_id, goal, content, domain, agent_name, judge_feedback)
            return

        async def _distill() -> None:
            try:
                lessons = await self._extract_lessons(
                    goal, content, domain, agent_name, judge_feedback
                )
                self._store_distilled_lessons(lessons, episode_id, domain, agent_name)
                logger.debug("Distilled %d lessons from episode %s", len(lessons), episode_id)
            except Exception as exc:
                logger.debug("Fact distillation failed for %s: %s", episode_id, exc)

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_distill())
        except RuntimeError:
            self._sync_distill(episode_id, goal, content, domain, agent_name, judge_feedback)

    def _sync_distill(
        self,
        episode_id: str,
        goal: str,
        content: str,
        domain: str,
        agent_name: str,
        judge_feedback: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Synchronous fact distillation — blocks calling thread."""
        import json as _json

        import anthropic

        prompt = self._build_distillation_prompt(goal, content, domain, judge_feedback)
        client = anthropic.Anthropic()

        # Retry once on empty/invalid response
        for attempt in range(2):
            try:
                response = client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = response.content[0].text.strip()  # type: ignore[union-attr]
                if not text:
                    logger.debug("Distillation returned empty response (attempt %d)", attempt + 1)
                    continue
                if "```" in text:
                    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")
                lessons = _json.loads(text)
                if isinstance(lessons, list):
                    parsed = [
                        {
                            "lesson": str(l.get("lesson", ""))[:200],
                            "type": str(l.get("type", "pattern")),
                            "sub_domain": str(l.get("sub_domain", "")),
                            "applies_to": str(l.get("applies_to", ""))[:100],
                            "confidence": min(1.0, max(0.0, float(l.get("confidence", 0.6)))),
                        }
                        for l in lessons[:3]
                        if l.get("lesson")
                    ]
                    if parsed:
                        self._store_distilled_lessons(parsed, episode_id, domain, agent_name)
                        logger.debug(
                            "Sync-distilled %d lessons for %s (cold start)",
                            len(parsed),
                            episode_id,
                        )
                        return
                elif isinstance(lessons, dict) and lessons.get("lesson"):
                    parsed = [
                        {
                            "lesson": str(lessons["lesson"])[:200],
                            "type": str(lessons.get("type", "pattern")),
                            "sub_domain": str(lessons.get("sub_domain", "")),
                            "applies_to": str(lessons.get("applies_to", ""))[:100],
                            "confidence": min(1.0, max(0.0, float(lessons.get("confidence", 0.6)))),
                        }
                    ]
                    self._store_distilled_lessons(parsed, episode_id, domain, agent_name)
                    return
            except _json.JSONDecodeError:
                logger.debug("Distillation JSON parse failed (attempt %d)", attempt + 1)
                continue
            except Exception as exc:
                logger.warning("Sync distillation failed for %s: %s", episode_id, exc)
                return
        logger.debug("Distillation failed after retries for %s", episode_id)

    def _build_distillation_prompt(
        self,
        goal: str,
        content: str,
        domain: str,
        judge_feedback: Optional[Dict[str, Any]] = None,
    ) -> str:
        # Fetch existing lessons so the LLM only extracts NEW insights
        existing_block = ""
        try:
            existing = self._store.get_distilled_lessons(domain=domain, limit=15)
            if existing:
                seen = [l.lesson for l in existing[:15]]
                existing_block = (
                    "\n\nALREADY KNOWN LESSONS (do NOT repeat these — extract only NEW insights):\n"
                    + "\n".join(f"- {s}" for s in seen)
                    + "\n"
                )
        except Exception:
            pass

        # Include expert judge feedback when available
        judge_block = ""
        if judge_feedback and judge_feedback.get("verdict") == "llm-judged":
            parts = ["\n\nEXPERT JUDGE EVALUATION:"]
            for dim in ("accuracy", "completeness", "structure", "actionability", "depth"):
                if dim in judge_feedback:
                    parts.append(f"  {dim}: {judge_feedback[dim]:.2f}/1.0")
            strengths = judge_feedback.get("strengths", [])
            improvements = judge_feedback.get("improvements", [])
            if strengths:
                parts.append("  Strengths: " + "; ".join(str(s) for s in strengths[:3]))
            if improvements:
                parts.append(
                    "  Improvements needed: " + "; ".join(str(s) for s in improvements[:3])
                )
            judge_block = "\n".join(parts) + "\n"

        return (
            f"Analyze this {domain} task execution and extract 2-3 concise, actionable lessons.\n\n"
            f"TASK: {goal[:500]}\n\n"
            f"OUTPUT (first 1500 chars):\n{content[:1500]}\n"
            f"{judge_block}"
            f"{existing_block}\n"
            f"For each lesson, provide:\n"
            f"- lesson: One sentence, specific and actionable (not generic advice)\n"
            f"- type: 'strategy' | 'mistake' | 'pattern' | 'tool_usage'\n"
            f"- sub_domain: A short tag for the specific sub-type of this task "
            f"(e.g. for domain 'mermaid': 'er', 'sequence', 'flowchart', 'state', 'gantt'; "
            f"for domain 'travel': 'itinerary', 'booking', 'visa'; "
            f"for domain 'coding': 'python', 'rust', 'frontend'). "
            f"Use '' (empty) if the lesson is generic to the whole domain.\n"
            f"- applies_to: When this lesson applies\n"
            f"- confidence: 0.0-1.0 how reliable this lesson is\n\n"
            f"CRITICAL: Focus on what the EXPERT JUDGE highlighted — especially the "
            f"improvements needed (learn from gaps) and strengths (learn what worked). "
            f"Only extract NON-OBVIOUS lessons that are NOT already listed above. "
            f"Skip generic advice like 'write tests' or 'handle errors'. "
            f"If this task taught nothing new beyond the known lessons, return an empty array [].\n\n"
            f"Respond as JSON array. Example:\n"
            f'[{{"lesson": "Use ||--o{{ for one-to-many relationships in ER diagrams", '
            f'"type": "pattern", "sub_domain": "er", '
            f'"applies_to": "Mermaid ER diagram generation", "confidence": 0.8}}]'
        )

    def _store_distilled_lessons(
        self,
        lessons: List[Dict[str, Any]],
        episode_id: str,
        domain: str,
        agent_name: str,
    ) -> None:
        """Store extracted lessons with embeddings.

        If a lesson has a ``sub_domain`` field, the domain is stored
        hierarchically as ``domain:sub_domain`` (e.g. ``mermaid:er``).
        Lessons with empty sub_domain stay under the parent domain.
        """
        emb_svc = _get_embeddings()
        for lesson_data in lessons:
            sub = str(lesson_data.get("sub_domain", "")).strip().lower()
            effective_domain = f"{domain}:{sub}" if sub else domain
            lesson = DistilledLesson(
                lesson_id=f"dl_{episode_id}_{uuid.uuid4().hex[:4]}",
                episode_id=episode_id,
                domain=effective_domain,
                agent_name=agent_name,
                lesson=lesson_data["lesson"],
                context_type=lesson_data.get("type", "pattern"),
                applicability=lesson_data.get("applies_to", ""),
                confidence=lesson_data.get("confidence", 0.6),
            )
            embedding_bytes = None
            if emb_svc.available:
                from .embeddings import EmbeddingService

                vec = emb_svc.embed(lesson.lesson)
                if vec is not None:
                    embedding_bytes = EmbeddingService.serialize(vec)
            self._store.save_distilled_lesson(lesson, embedding_bytes)

    async def _extract_lessons(
        self,
        goal: str,
        content: str,
        domain: str,
        agent_name: str,
        judge_feedback: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Use a cheap LLM to extract 2-3 concise lessons from an episode.

        Returns list of {"lesson": str, "type": str, "applies_to": str, "confidence": float}
        """
        prompt = self._build_distillation_prompt(goal, content, domain, judge_feedback)

        try:
            import json as _json

            import anthropic

            client = anthropic.AsyncAnthropic()
            response = await client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()  # type: ignore[union-attr]

            if "```" in text:
                text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")

            lessons = _json.loads(text)
            if isinstance(lessons, list):
                return [
                    {
                        "lesson": str(l.get("lesson", ""))[:200],
                        "type": str(l.get("type", "pattern")),
                        "sub_domain": str(l.get("sub_domain", "")),
                        "applies_to": str(l.get("applies_to", ""))[:100],
                        "confidence": min(1.0, max(0.0, float(l.get("confidence", 0.6)))),
                    }
                    for l in lessons[:3]
                    if l.get("lesson")
                ]
        except Exception as exc:
            logger.debug("LLM lesson extraction failed: %s", exc)

        return []

    def retrieve_distilled_lessons(
        self,
        domain: str,
        goal: str = "",
        agent_name: str = "",
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant distilled lessons using vector similarity.

        Supports hierarchical domains: if domain is "mermaid", retrieves
        both "mermaid" (shared) and "mermaid:er" / "mermaid:sequence" etc.,
        ranking sub-domain matches higher when the goal text matches.
        """
        emb_svc = _get_embeddings()

        if goal and emb_svc.available:
            goal_vec = emb_svc.embed(goal)
            if goal_vec is not None:
                return self._embedding_lesson_retrieval(domain, goal_vec, agent_name, top_k)

        # Fallback: hierarchical retrieval by confidence
        lessons = self._store.get_distilled_lessons(
            domain=domain,
            agent_name=agent_name or None,
            limit=top_k * 3,
            hierarchical=True,
        )

        # Score: sub-domain match bonus when goal hints at a sub-type
        goal_lower = goal.lower() if goal else ""
        scored = []
        for l in lessons:
            score = l.confidence
            # Boost lessons whose sub-domain appears in the goal text
            if ":" in l.domain:
                sub = l.domain.split(":", 1)[1]
                if sub and sub in goal_lower:
                    score += 0.4  # strong sub-domain match
                elif sub and sub not in goal_lower:
                    score -= 0.3  # wrong sub-domain — suppress
            scored.append((score, l))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            {
                "lesson": l.lesson,
                "type": l.context_type,
                "applies_to": l.applicability,
                "confidence": l.confidence,
                "agent": l.agent_name,
                "domain": l.domain,
            }
            for _, l in scored[:top_k]
        ]

    def _embedding_lesson_retrieval(
        self,
        domain: str,
        goal_vec: Any,
        agent_name: str,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Retrieve distilled lessons using embedding cosine similarity.

        Uses hierarchical domain matching so sub-domain lessons are
        included but ranked by semantic relevance to the current goal.
        """
        import numpy as np

        from .embeddings import EmbeddingService

        lesson_embs = self._store.get_distilled_lessons_with_embeddings(
            domain=domain,
            limit=100,
            hierarchical=True,
        )

        if not lesson_embs:
            lesson_embs = self._store.get_distilled_lessons_with_embeddings(domain=None, limit=100)
        if not lesson_embs:
            return []

        scored = []
        for lesson, emb_blob in lesson_embs:
            lesson_vec = EmbeddingService.deserialize(emb_blob)
            similarity = float(np.dot(goal_vec, lesson_vec))
            agent_bonus = 0.1 if agent_name and lesson.agent_name == agent_name else 0.0
            score = similarity * 0.6 + lesson.confidence * 0.3 + agent_bonus
            scored.append((score, lesson))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            {
                "lesson": lesson.lesson,
                "type": lesson.context_type,
                "applies_to": lesson.applicability,
                "confidence": lesson.confidence,
                "agent": lesson.agent_name,
                "domain": lesson.domain,
                "relevance": round(score, 3),
            }
            for score, lesson in scored[:top_k]
        ]

    # =========================================================================
    # TAUTOLOGY FILTER — Delegated to PatternExtractor
    # =========================================================================

    def _is_tautological_pattern(self, domain: str, recommendation: str) -> bool:
        """Delegate to PatternExtractor.is_tautological_pattern()."""
        return self._pattern_extractor.is_tautological_pattern(domain, recommendation)

    def _batch_tautology_filter(self, candidates: List[Tuple[str, str, str]]) -> Dict[str, bool]:
        """Delegate to PatternExtractor.batch_tautology_filter()."""
        return self._pattern_extractor.batch_tautology_filter(candidates)

    def purge_stale_tautological_patterns(self) -> int:
        """Delegate to PatternExtractor.purge_stale_tautological_patterns()."""
        return self._pattern_extractor.purge_stale_tautological_patterns()

    @staticmethod
    def _heuristic_tautology_check(domain: str, recommendation: str) -> bool:
        """Delegate to PatternExtractor.heuristic_tautology_check()."""
        return PatternExtractor.heuristic_tautology_check(domain, recommendation)

    # =========================================================================
    # EXPLORATION ANALYSIS — Close the A/B loop
    # =========================================================================

    def analyze_exploration_results(self, domain: str) -> Optional[Dict[str, Any]]:
        """
        Compare exploration episodes vs baseline for a domain.
        Updates value estimates if exploration found a better approach.

        This closes the A/B experimentation loop: explore → measure → learn.
        """
        episodes = self._store.query_episodes(domain=domain, limit=50)
        if len(episodes) < 3:
            return None

        def _is_exploration(ep: EpisodeRecord) -> bool:
            return bool(ep.action.get("exploration") or ep.metadata.get("exploration"))

        explore_eps = [e for e in episodes if _is_exploration(e) and e.quality > 0]
        baseline_eps = [e for e in episodes if not _is_exploration(e) and e.quality > 0]

        if not explore_eps or not baseline_eps:
            return None

        explore_avg = sum(e.quality for e in explore_eps) / len(explore_eps)
        baseline_avg = sum(e.quality for e in baseline_eps) / len(baseline_eps)
        delta = explore_avg - baseline_avg

        result = {
            "domain": domain,
            "explore_count": len(explore_eps),
            "baseline_count": len(baseline_eps),
            "explore_avg_quality": round(explore_avg, 3),
            "baseline_avg_quality": round(baseline_avg, 3),
            "delta": round(delta, 3),
            "exploration_wins": delta > 0.05,
        }

        # If exploration found a better approach, boost its value estimate
        if delta > 0.05 and len(explore_eps) >= 2:
            for ep in explore_eps:
                reason = ep.action.get("exploration_reason", "")
                if "model=" in reason:
                    model = reason.split("model=")[-1].strip()
                    state_key = f"{domain}:explored_model"
                    action_key = f"model={model}"
                    existing = self._store.get_value(state_key, action_key, domain)
                    new_val = explore_avg
                    if existing:
                        alpha = 0.2
                        new_val = existing.value + alpha * (explore_avg - existing.value)
                    self._store.save_value(
                        ValueEstimate(
                            state_key=state_key,
                            action_key=action_key,
                            domain=domain,
                            value=new_val,
                            td_error=delta,
                            update_count=(existing.update_count + 1) if existing else 1,
                        )
                    )
                    # Save as a pattern too
                    pattern_id = hashlib.md5(f"explore_win_{domain}_{model}".encode()).hexdigest()[
                        :12
                    ]
                    self._store.save_pattern(
                        PatternRecord(
                            pattern_id=pattern_id,
                            source_domain=domain,
                            pattern_type="exploration_win",
                            description=(
                                f"Exploration in {domain}: {model} scored "
                                f"{explore_avg:.2f} vs baseline {baseline_avg:.2f} "
                                f"(+{delta:.2f})"
                            ),
                            conditions={"domain": domain},
                            recommendation=f"Consider using {model} for {domain} tasks",
                            confidence=min(0.9, 0.5 + delta),
                            evidence_count=len(explore_eps),
                            applicable_domains=[domain],
                        )
                    )

            logger.info(
                f"Exploration win in {domain}: "
                f"{explore_avg:.3f} vs {baseline_avg:.3f} (+{delta:.3f})"
            )

        elif delta < -0.05 and len(explore_eps) >= 2:
            # Exploration was worse — record as pattern to avoid
            pattern_id = hashlib.md5(f"explore_loss_{domain}".encode()).hexdigest()[:12]
            self._store.save_pattern(
                PatternRecord(
                    pattern_id=pattern_id,
                    source_domain=domain,
                    pattern_type="exploration_loss",
                    description=(
                        f"Exploration in {domain} performed worse: "
                        f"{explore_avg:.2f} vs baseline {baseline_avg:.2f} "
                        f"({delta:.2f})"
                    ),
                    conditions={"domain": domain},
                    recommendation=f"Stick with default approach for {domain} tasks",
                    confidence=min(0.9, 0.5 + abs(delta)),
                    evidence_count=len(explore_eps),
                    applicable_domains=[domain],
                )
            )

        return result

    # =========================================================================
    # CURRICULUM — Identify weak domains and suggest practice
    # =========================================================================

    def identify_weak_domains(self, min_episodes: int = 2) -> List[Dict[str, Any]]:
        """
        Identify domains that need improvement based on quality scores.
        Returns domains sorted by weakness (lowest quality first).
        """
        conn = self._store._get_conn()
        rows = conn.execute(
            """SELECT domain, COUNT(*) as cnt, AVG(quality) as avg_q,
                      MIN(quality) as min_q, MAX(quality) as max_q,
                      AVG(CASE WHEN success = 1 THEN 1.0 ELSE 0.0 END) as sr
               FROM episodes
               WHERE quality > 0 AND domain != 'system' AND domain != ''
               GROUP BY domain
               HAVING cnt >= ?
               ORDER BY avg_q ASC""",
            (min_episodes,),
        ).fetchall()

        weak = []
        for row in rows:
            d = dict(row)
            if d["avg_q"] < 0.8 or d["sr"] < 0.9:
                weak.append(
                    {
                        "domain": d["domain"],
                        "episodes": d["cnt"],
                        "avg_quality": round(d["avg_q"], 3),
                        "min_quality": round(d["min_q"], 3),
                        "max_quality": round(d["max_q"], 3),
                        "success_rate": round(d["sr"], 3),
                        "needs": self._diagnose_weakness(d),
                    }
                )

        return weak

    def _diagnose_weakness(self, domain_stats: Dict[str, Any]) -> str:
        """Diagnose why a domain is weak and suggest what to practice."""
        domain = domain_stats["domain"]
        avg_q = domain_stats["avg_q"]
        sr = domain_stats["sr"]

        episodes = self._store.query_episodes(domain=domain, limit=20)
        if not episodes:
            return "insufficient data"

        # Analyze what's missing in low-quality responses
        issues = []
        for ep in episodes:
            out = ep.outcome or {}
            if out.get("goal_coverage", 1.0) < 0.6:
                issues.append("low_coverage")
            if out.get("structure_score", 1.0) < 0.3:
                issues.append("low_structure")
            if out.get("word_count", 1000) < 500:
                issues.append("too_short")
            if not out.get("has_code") and domain in ("coding", "data_science"):
                issues.append("missing_code")
            if not out.get("has_citations") and domain in ("research", "economics"):
                issues.append("missing_citations")

        if not issues:
            return "quality plateau — try deeper prompts"

        from collections import Counter

        top_issues = Counter(issues).most_common(3)
        diagnoses = {
            "low_coverage": "responses don't address all parts of the task",
            "low_structure": "responses lack clear structure (headings, lists)",
            "too_short": "responses are too shallow — need more depth",
            "missing_code": "coding responses should include implementations",
            "missing_citations": "research responses should cite specific sources",
        }
        return "; ".join(diagnoses.get(issue, issue) for issue, _ in top_issues)

    def generate_curriculum(self, top_n: int = 3, min_episodes: int = 1) -> List[Dict[str, Any]]:
        """
        Generate a learning curriculum: practice tasks for the weakest domains.

        Returns a list of suggested practice tasks with rationale.
        """
        weak = self.identify_weak_domains(min_episodes=min_episodes)
        if not weak:
            return []

        curriculum = []
        practice_templates = {
            "coding": [
                "Implement a {structure} in Python with full error handling, type hints, and 3 unit tests.",
                "Design a production-grade {pattern} system. Include: architecture, code, failure modes, and monitoring.",
            ],
            "economics": [
                "Analyze the economic impact of {topic}. Include: quantitative data, policy implications, and timeline.",
                "Compare {approach_a} vs {approach_b} economic policies. Cite Acemoglu, Autor, or Piketty. Use data.",
            ],
            "research": [
                "Write a literature review on {topic}. Cite at least 5 specific papers with years. Include methodology comparison.",
                "Propose a novel research approach to {problem}. Include: hypothesis, methodology, expected results, and limitations.",
            ],
            "system_design": [
                "Design a {system} handling {scale}. Include: architecture diagram (text), capacity planning, failure modes, and monitoring.",
                "Compare {tech_a} vs {tech_b} for {use_case}. Include: benchmarks, trade-offs, and recommendation with reasoning.",
            ],
            "data_science": [
                "Design an ML pipeline for {task}. Include: feature engineering, model selection, evaluation metrics, and deployment.",
                "Propose a novel approach combining {method_a} and {method_b}. Include: mathematical formulation and complexity analysis.",
            ],
            "general": [
                "Provide a comprehensive analysis of {topic} covering at least 5 dimensions. Use structured formatting.",
            ],
        }

        fills = {
            "coding": {"structure": "thread-safe priority queue", "pattern": "circuit breaker"},
            "economics": {
                "topic": "central bank digital currencies",
                "approach_a": "MMT",
                "approach_b": "Austrian school",
            },
            "research": {
                "topic": "transformer architecture scaling laws",
                "problem": "catastrophic forgetting in LLMs",
            },
            "system_design": {
                "system": "real-time recommendation engine",
                "scale": "1M requests/sec",
                "tech_a": "Kafka",
                "tech_b": "Pulsar",
                "use_case": "event streaming",
            },
            "data_science": {
                "task": "fraud detection",
                "method_a": "GNNs",
                "method_b": "temporal transformers",
            },
            "general": {"topic": "the future of remote work"},
        }

        for domain_info in weak[:top_n]:
            domain = domain_info["domain"]
            templates = practice_templates.get(domain, practice_templates["general"])
            domain_fills = fills.get(domain, fills["general"])

            for template in templates[:1]:
                task = template.format(**domain_fills)
                current_q = domain_info["avg_quality"]
                target_q = min(0.90, current_q + 0.02)
                curriculum.append(
                    {
                        "domain": domain,
                        "task": task,
                        "rationale": f"Domain '{domain}' quality={current_q:.2f}. "
                        f"Issue: {domain_info['needs']}",
                        "target_quality": round(target_q, 3),
                        "current_quality": current_q,
                    }
                )

        return curriculum

    # =========================================================================
    # AUTO-CURRICULUM — Execute practice tasks for weak domains
    # =========================================================================

    async def run_curriculum(
        self, max_tasks: int = 3, provider: str = "anthropic", min_episodes: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Auto-execute practice tasks for the weakest domains.

        This is the active learning loop:
        1. Identify weak domains
        2. Generate targeted practice tasks
        3. Execute them through the Orchestrator
        4. Record results (the learning system learns from these too)

        Returns list of practice results.
        """
        curriculum = self.generate_curriculum(top_n=max_tasks, min_episodes=min_episodes)
        if not curriculum:
            logger.info("Auto-curriculum: no weak domains, skipping")
            return []

        results = []
        try:
            import asyncio

            from Jotty.core.intelligence.orchestration.core.swarm_manager import (
                Orchestrator,
            )

            orch = object.__new__(Orchestrator)
            orch.config = type(
                "C",
                (),
                {
                    "domain": "general",
                    "base_path": None,
                    "learning_wait_timeout_seconds": 0,
                },
            )()
            orch.agents = []
            orch.mode = "single"
            orch.runners = {}
            orch._runners_built = False
            orch._efficiency_stats = {}
            orch._intelligence_metrics = {}
            orch._engine = None
            orch._learning_ready = asyncio.Event()
            orch._learning_ready.set()

            for item in curriculum[:max_tasks]:
                domain = item["domain"]
                task = item["task"]
                target_q = item["target_quality"]

                logger.info(
                    f"Auto-curriculum: practicing {domain} "
                    f"(current={item['current_quality']:.2f}, target={target_q:.2f})"
                )

                try:
                    result = await orch.chat(
                        message=task + "\n\nDo NOT use any tools.",
                        provider=provider,
                        learn=True,
                    )
                    content = getattr(result, "content", "")
                    success = getattr(result, "success", False)

                    response_analysis = analyze_response(content, task) if content else {}
                    quality = response_analysis.get("quality_score", 0.0)

                    results.append(
                        {
                            "domain": domain,
                            "task": task[:100],
                            "success": success,
                            "quality": round(quality, 3),
                            "target_quality": target_q,
                            "improved": quality >= target_q,
                            "content_length": len(content),
                        }
                    )

                    logger.info(
                        f"Auto-curriculum result: {domain} "
                        f"quality={quality:.3f} (target={target_q:.2f}) "
                        f"{'IMPROVED' if quality >= target_q else 'needs more practice'}"
                    )

                except Exception as e:
                    results.append(
                        {
                            "domain": domain,
                            "task": task[:100],
                            "success": False,
                            "quality": 0.0,
                            "target_quality": target_q,
                            "improved": False,
                            "error": str(e)[:200],
                        }
                    )
                    logger.warning(f"Auto-curriculum task failed for {domain}: {e}")

        except Exception as e:
            logger.error(f"Auto-curriculum execution failed: {e}")

        return results

    def improvement_report(self, domain: str = "") -> Dict[str, Any]:
        """Get improvement report for monitoring."""
        return self._store.get_improvement_report(domain)

    # =========================================================================
    # INTERNAL: Value updates, caching, pattern extraction
    # =========================================================================

    def record_outcome(
        self,
        unit_name: str,
        state: str,
        action: str,
        reward: float,
        domain: str = "",
        task_type: str = "",
    ) -> None:
        """
        Convenience method for callers that just have state/action/reward.

        Replaces LearningManager.record_outcome(). Records to SQLite and
        updates value estimates with lesson extraction.
        """
        success = reward > 0.5
        quality = max(0.0, min(1.0, reward))
        action_dict = {"action": action}

        self.record(
            unit_name=unit_name,
            unit_type="agent",
            domain=domain or "general",
            task_type=task_type or "general",
            context={"state": state},
            action=action_dict,
            outcome={"reward": reward},
            success=success,
            quality=quality,
        )

    def get_learned_context(self, state: str, domain: str = "", unit_name: str = "") -> str:
        """
        Get learning context for a state — backward-compat wrapper around
        build_context_string() + lessons from value estimates.

        Replaces LearningManager.get_learned_context().
        """
        parts: List[str] = []

        # Get lessons from value estimates for this domain/unit
        lessons = self._store.get_lessons(
            domain=domain or "general",
            unit_name=unit_name if unit_name else None,
            limit=10,
        )
        if lessons:
            parts.append("Prior lessons:")
            for lesson in lessons[:5]:
                parts.append(f"  {lesson}")

        # Also include pattern-based context
        context_str = self.build_context_string(
            domain=domain or "general",
            task_type="",
            unit_name=unit_name,
        )
        if context_str:
            parts.append(context_str)

        return "\n".join(parts) if parts else ""

    def _update_values(
        self,
        domain: str,
        task_type: str,
        action: Dict[str, Any],
        success: bool,
        quality: float,
        unit_name: str = "",
    ) -> None:
        """Update TD value estimates based on outcome, with lesson extraction."""
        state_key = self._make_state_key(domain, task_type)
        action_key = self._make_action_key(action)
        reward = quality if success else -0.1

        existing = self._store.get_value(state_key, action_key, domain)

        if existing:
            # TD(0) update: V(s) = V(s) + alpha * (reward - V(s))
            alpha = max(0.01, 0.1 / (1 + existing.update_count * 0.01))
            new_value = existing.value + alpha * (reward - existing.value)
            td_error = reward - existing.value
            count = existing.update_count + 1
            # Running average reward
            avg_reward = existing.avg_reward + (reward - existing.avg_reward) / count

            # Extract lesson from this experience
            lessons = list(existing.lessons)
            lesson = self._extract_lesson(state_key, action_key, reward, td_error)
            if lesson and lesson not in lessons:
                lessons.append(lesson)
                # Keep only the most recent 10 lessons
                if len(lessons) > 10:
                    lessons = lessons[-10:]

            updated = ValueEstimate(
                state_key=state_key,
                action_key=action_key,
                domain=domain,
                value=new_value,
                td_error=td_error,
                update_count=count,
                unit_name=unit_name or existing.unit_name,
                lessons=lessons,
                avg_reward=avg_reward,
            )
        else:
            lesson = self._extract_lesson(state_key, action_key, reward, reward)
            lessons = [lesson] if lesson else []

            updated = ValueEstimate(
                state_key=state_key,
                action_key=action_key,
                domain=domain,
                value=reward,
                td_error=reward,
                update_count=1,
                unit_name=unit_name,
                lessons=lessons,
                avg_reward=reward,
            )

        self._store.save_value(updated)

    def _update_skill_credits(
        self,
        domain: str,
        task_type: str,
        action: Dict[str, Any],
        outcome: Dict[str, Any],
        success: bool,
        quality: float,
    ) -> None:
        """Distribute credit across skills using cheap heuristic.

        Instead of giving every skill the same global reward, weight by
        step-level contribution. Uses data already in action/outcome —
        no LLM calls. Falls back to uniform credit when step data is absent.

        Integrates the credit assignment concept from algorithmic_credit.py
        via a zero-cost heuristic (Shapley fallback: proportional by success).
        """
        skills = action.get("skills", [])
        if not skills or not success:
            return

        try:
            from .facade import get_td_lambda

            td = get_td_lambda()
        except Exception:
            return

        # Extract per-step outcomes if available
        step_results = outcome.get("per_step_results", outcome.get("step_results", []))

        if step_results and len(step_results) == len(skills):
            # Credit by step success: succeeded steps get more credit
            credits = []
            for sr in step_results:
                if isinstance(sr, dict):
                    step_ok = sr.get("success", sr.get("ok", True))
                    credits.append(1.0 if step_ok else 0.2)
                else:
                    credits.append(0.7)  # ambiguous → moderate credit
        else:
            # Uniform credit when step data is absent
            credits = [1.0] * len(skills)

        total = sum(credits) or 1.0
        for skill, credit in zip(skills, credits):
            weighted_reward = quality * (credit / total) * len(skills)
            weighted_reward = max(0.0, min(1.0, weighted_reward))
            td.skill_q.update(task_type, skill, weighted_reward, domain=domain)

    def _maybe_auto_optimize(self, domain: str, task_type: str, domain_count: int) -> None:
        """Auto-trigger DSPy re-optimization and crystallization.

        Called from record() when a successful episode lands in a domain
        with enough accumulated gold data. Two actions:

        1. DSPy re-optimize: every _dspy_optimize_interval episodes per
           domain, gather gold episodes and run MIPROv2/BootstrapRS.
           This makes prompts incrementally better *before* graduation.

        2. Crystallize: after DSPy optimization (or on its own schedule),
           check if Q-values have converged enough to harden into an SOP.

        Both are cheap checks — the expensive work (DSPy compile, LLM
        calls) only fires when thresholds are crossed.
        """
        last_opt = self._last_optimize_counts.get(domain, 0)
        if domain_count - last_opt < self._dspy_optimize_interval:
            return

        self._last_optimize_counts[domain] = domain_count

        # 1. DSPy re-optimization (incremental, pre-crystallization)
        try:
            from .advanced_learning import DomainDSPyOptimizer

            optimizer = DomainDSPyOptimizer.get_instance()
            gold_count = len(optimizer._gather_training_data(domain))
            if gold_count >= 5:
                optimized = optimizer.optimize(domain)
                if optimized:
                    logger.info(
                        f"Auto-DSPy-optimize: {domain} re-optimized "
                        f"({gold_count} gold episodes, trigger at count={domain_count})"
                    )
        except Exception as e:
            logger.debug(f"Auto-DSPy-optimize failed for {domain}: {e}")

        # 2. Validate before crystallizing — prove learning is genuinely better
        try:
            from .validation import LearningValidator

            validator = LearningValidator.get_instance()
            report = validator.validate_domain(domain, task_type)
            if not report.overall_passed:
                logger.info(
                    f"Validation gate blocked crystallization for {domain}:{task_type} "
                    f"— {report.recommendation}: "
                    + "; ".join(f"{c.check}={'✓' if c.passed else '✗'}" for c in report.checks)
                )
                return
        except Exception as e:
            logger.debug(f"Validation check failed for {domain}:{task_type}: {e}")

        # 3. Auto-crystallize (graduation check) — only reached if validation passed
        try:
            from .crystallization import maybe_crystallize

            config = maybe_crystallize(task_type, domain)
            if config:
                logger.info(
                    f"Auto-crystallized {domain}:{task_type} — "
                    f"SOP={' → '.join(config.sop_roles)}"
                )
        except Exception as e:
            logger.debug(f"Auto-crystallize failed for {domain}:{task_type}: {e}")

    @staticmethod
    def _extract_lesson(
        state_key: str, action_key: str, reward: float, td_error: float
    ) -> Optional[str]:
        """
        Extract actionable natural language lesson from experience.

        Only records lessons for significant outcomes (large TD error or
        extreme rewards). Ported from q_learning.py, simplified.
        """
        if abs(td_error) < 0.1:
            return None

        # Build concise context from keys
        context = f"{state_key} -> {action_key}"
        if len(context) > 120:
            context = context[:120] + "..."

        if reward > 0.7:
            return f"SUCCESS: {context} (reward={reward:.2f})"
        elif reward < 0.3:
            return f"AVOID: {context} (reward={reward:.2f})"
        elif td_error > 0.2:
            expected = reward - td_error
            return f"DISCOVERY: {context} better than expected (actual={reward:.2f}, expected={expected:.2f})"
        elif td_error < -0.2:
            expected = reward - td_error
            return f"CAUTION: {context} worse than expected (actual={reward:.2f}, expected={expected:.2f})"

        return None

    def _extract_patterns(self, domain: str) -> None:
        """Delegate to PatternExtractor.extract_patterns()."""
        self._pattern_extractor.extract_patterns(domain)

    def _extract_causal_patterns(self, domain: str, episodes: List[EpisodeRecord]) -> None:
        """Delegate to PatternExtractor.extract_causal_patterns()."""
        self._pattern_extractor.extract_causal_patterns(domain, episodes)

    def _extract_transfer_patterns(self, domain: str, domain_episodes: List[EpisodeRecord]) -> None:
        """Delegate to PatternExtractor.extract_transfer_patterns()."""
        self._pattern_extractor.extract_transfer_patterns(domain, domain_episodes)

    def _get_best_action(self, domain: str, task_type: str) -> Optional[Dict[str, Any]]:
        """Find the highest-value action for this state."""
        state_key = self._make_state_key(domain, task_type)
        conn = self._store._get_conn()
        row = conn.execute(
            """SELECT action_key, value, update_count FROM value_estimates
               WHERE state_key = ? AND domain = ?
               ORDER BY value DESC LIMIT 1""",
            (state_key, domain),
        ).fetchone()

        if row and row["update_count"] >= 3:
            return {
                "action": row["action_key"],
                "expected_value": round(row["value"], 3),
                "confidence": min(1.0, row["update_count"] / 10),
            }
        return None

    def _get_cached_success_rate(self, domain: str, task_type: str) -> Tuple[float, int]:
        """Get success rate with caching."""
        key = f"{domain}:{task_type}"
        cached: Any = self._success_rate_cache.get(key)
        if cached and (time.time() - cached[2]) < self._cache_ttl:
            return cached[0], cached[1]

        rate, total = self._store.get_success_rate(domain, task_type)
        self._success_rate_cache[key] = (rate, total, time.time())
        return rate, total

    def _get_cached_patterns(self, domain: str) -> List[PatternRecord]:
        """Get patterns with caching."""
        cached = self._pattern_cache.get(domain)
        if cached and (time.time() - cached[1]) < self._cache_ttl:
            return cached[0]

        patterns = self._store.get_patterns(domain=domain, min_confidence=0.3)
        self._pattern_cache[domain] = (patterns, time.time())
        return patterns

    def _invalidate_cache(self, domain: str, task_type: str) -> None:
        """Invalidate relevant caches including related domains."""
        key = f"{domain}:{task_type}"
        self._success_rate_cache.pop(key, None)
        self._success_rate_cache.pop(f"{domain}:", None)
        self._pattern_cache.pop(domain, None)
        # Also invalidate "general" cache since aggregate stats change
        self._success_rate_cache.pop("general:", None)
        self._success_rate_cache.pop("general:general", None)
        self._pattern_cache.pop("general", None)

    @staticmethod
    def _make_state_key(domain: str, task_type: str) -> str:
        return f"{domain}:{task_type}" if task_type else domain

    @staticmethod
    def _make_action_key(action: Dict[str, Any]) -> str:
        parts = sorted(
            f"{k}={v}" for k, v in action.items() if isinstance(v, (str, int, float, bool))
        )
        return "|".join(parts[:5]) or "default"

    _CONTENT_KEYS = frozenset({"content", "response_excerpt", "response", "result"})

    @staticmethod
    def _truncate_dict(d: Dict[str, Any], max_str_len: int = 500) -> Dict[str, Any]:
        """Truncate string values in a dict to avoid DB bloat.

        Content-bearing keys (content, response_excerpt) get a 3000-char
        limit so retrieval has meaningful few-shot examples to inject.
        """
        result: Dict[str, Any] = {}
        for k, v in list(d.items())[:40]:
            if isinstance(v, str):
                limit = 3000 if k in LearningService._CONTENT_KEYS else max_str_len
                if len(v) > limit:
                    result[k] = v[:limit] + "..."
                else:
                    result[k] = v
            elif isinstance(v, dict):
                result[k] = LearningService._truncate_dict(v, max_str_len)
            else:
                result[k] = v
        return result


# =============================================================================
# MODULE-LEVEL CONVENIENCE
# =============================================================================


def get_learning_service(store: Optional[LearningStore] = None) -> LearningService:
    """Get the singleton LearningService instance."""
    return LearningService.get_instance(store)
