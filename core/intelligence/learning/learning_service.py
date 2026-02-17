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

import hashlib
import logging
import time
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from .learning_store import (
    EpisodeRecord,
    LearningStore,
    PatternRecord,
    ReflectionRecord,
    ValueEstimate,
)

logger = logging.getLogger(__name__)


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
        self._store = store or LearningStore.get_instance()
        self._active_episodes: Dict[str, ActiveEpisode] = {}

        # In-memory caches for hot-path queries (avoid DB hits)
        self._success_rate_cache: Dict[str, Tuple[float, float]] = {}  # key -> (rate, timestamp)
        self._pattern_cache: Dict[str, Tuple[List[PatternRecord], float]] = {}
        self._cache_ttl = 60.0  # Cache TTL in seconds

        # Pattern extraction thresholds
        self._min_episodes_for_pattern = 5
        self._pattern_extraction_interval = 20  # Extract patterns every N records
        self._record_count = 0

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
        episode_id = f"ep_{unit_name}_{int(time.time())}_{uuid.uuid4().hex[:6]}"

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
            return episode_id

        # Update value estimates via TD learning
        self._update_values(domain, task_type, action, success, quality)

        # Invalidate caches
        self._invalidate_cache(domain, task_type)

        # Periodic pattern extraction
        self._record_count += 1
        if self._record_count % self._pattern_extraction_interval == 0:
            try:
                self._extract_patterns(domain)
            except Exception as e:
                logger.debug(f"Pattern extraction failed: {e}")

        logger.debug(
            f"Recorded: {unit_name} ({unit_type}) domain={domain} "
            f"task={task_type} success={success} quality={quality:.2f}"
        )
        return episode_id

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

            # 4. Recent failure analysis
            if rate < 0.8:
                failures = self._store.get_failure_analysis(domain, task_type, limit=5)
                result["failure_analysis"] = [
                    {
                        "error_type": f.get("error_type", "unknown"),
                        "error_message": f.get("error_message", "")[:200],
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
        """
        Mid-execution reflection. Called when a step fails or produces
        unexpected results, enabling the system to adjust its approach
        for subsequent steps.

        Args:
            episode_id: Current episode ID
            step: Step number that triggered reflection
            observation: What happened (e.g., "Test failures on edge cases")
            unit_name: Which unit is reflecting
            analysis: Optional analysis of why it happened

        Returns:
            Dict with:
                - adjustment: Recommended adjustment for next steps
                - similar_failures: Past failures with similar patterns
                - recovery_strategies: Strategies that worked for similar issues
        """
        result: Dict[str, Any] = {
            "adjustment": "",
            "similar_failures": [],
            "recovery_strategies": [],
        }

        try:
            # Find similar past failures
            failures = self._store.get_failure_analysis(limit=20)
            similar = []
            obs_lower = observation.lower()
            for f in failures:
                err = (f.get("error_message", "") or "").lower()
                if any(word in err for word in obs_lower.split() if len(word) > 3):
                    similar.append(f)

            result["similar_failures"] = similar[:3]

            # Find patterns that succeeded in similar contexts
            patterns = self._store.get_patterns(
                pattern_type="failure_avoidance", min_confidence=0.5
            )
            recovery = [
                p.recommendation
                for p in patterns
                if any(word in p.description.lower() for word in obs_lower.split() if len(word) > 3)
            ]
            result["recovery_strategies"] = recovery[:3]

            # Build adjustment recommendation
            if recovery:
                result["adjustment"] = f"Based on {len(recovery)} past recoveries: {recovery[0]}"
            elif similar:
                # Learn from past: what was different about successes after similar failures
                result["adjustment"] = (
                    f"Similar failure seen {len(similar)} times before. "
                    f"Common error: {similar[0].get('error_type', 'unknown')}. "
                    f"Consider adjusting approach."
                )
            else:
                result["adjustment"] = "No prior data. Recording for future learning."

            # Save reflection for future use
            reflection = ReflectionRecord(
                reflection_id=f"ref_{uuid.uuid4().hex[:8]}",
                episode_id=episode_id,
                step=step,
                unit_name=unit_name,
                observation=observation[:500],
                analysis=analysis[:500] or result["adjustment"],
                adjustment=result["adjustment"][:500],
                applied=False,
            )
            self._store.save_reflection(reflection)

        except Exception as e:
            logger.debug(f"Reflect failed: {e}")
            result["adjustment"] = "Reflection unavailable, proceeding with default approach."

        return result

    def mark_reflection_applied(
        self, episode_id: str, step: int, improvement: Optional[float] = None
    ) -> None:
        """Mark a reflection as applied and record improvement delta."""
        reflections = self._store.get_reflections(episode_id=episode_id)
        for r in reflections:
            if r.step == step:
                r.applied = True
                r.improvement = improvement
                self._store.save_reflection(r)
                break

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
        patterns = self._store.get_patterns(domain=source_domain, min_confidence=0.6)

        transferable = []
        for p in patterns:
            if target_domain in p.applicable_domains or p.pattern_type in (
                "success_strategy",
                "tool_preference",
            ):
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
    ) -> str:
        """
        End an active episode and persist it.

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
                action={},
                outcome=outcome or {},
                success=success,
                quality=quality,
                cost=cost,
            )

        execution_time = time.time() - episode.start_time
        action_summary = {
            "steps": len(episode.steps),
            "step_names": [s["name"] for s in episode.steps],
            "step_successes": [s["success"] for s in episode.steps],
        }

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
        )

    # =========================================================================
    # BUILD CONTEXT — Generate learning context string for agent prompts
    # =========================================================================

    def build_context_string(self, domain: str, task_type: str = "", unit_name: str = "") -> str:
        """
        Build a human-readable learning context string for injection into
        agent system prompts or swarm context.

        Args:
            domain: Execution domain
            task_type: Specific task type
            unit_name: Name of the requesting unit

        Returns:
            String with learning insights for prompt injection
        """
        guidance = self.query(domain, task_type, unit_name=unit_name)

        if not guidance.get("has_learning"):
            return ""

        parts = []
        parts.append(f"[LEARNING CONTEXT — {domain}]")

        rate = guidance["success_rate"]
        total = guidance["total_episodes"]
        parts.append(f"Success rate: {rate:.0%} ({total} episodes)")

        if guidance.get("improving"):
            parts.append("Trend: IMPROVING")
        elif total > 10:
            parts.append("Trend: needs improvement")

        for rec in guidance.get("recommendations", [])[:2]:
            parts.append(f"Tip: {rec}")

        failures = guidance.get("failure_analysis", [])
        if failures:
            error_types = set(f.get("error_type", "") for f in failures if f.get("error_type"))
            if error_types:
                parts.append(f"Watch for: {', '.join(error_types)}")

        return "\n".join(parts)

    # =========================================================================
    # IMPROVEMENT REPORT
    # =========================================================================

    def improvement_report(self, domain: str = "") -> Dict[str, Any]:
        """Get improvement report for monitoring."""
        return self._store.get_improvement_report(domain)

    # =========================================================================
    # INTERNAL: Value updates, caching, pattern extraction
    # =========================================================================

    def _update_values(
        self, domain: str, task_type: str, action: Dict[str, Any], success: bool, quality: float
    ) -> None:
        """Update TD value estimates based on outcome."""
        state_key = self._make_state_key(domain, task_type)
        action_key = self._make_action_key(action)
        reward = quality if success else -0.1

        existing = self._store.get_value(state_key, action_key, domain)

        if existing:
            # TD(0) update: V(s) = V(s) + alpha * (reward - V(s))
            alpha = max(0.01, 0.1 / (1 + existing.update_count * 0.01))
            new_value = existing.value + alpha * (reward - existing.value)
            td_error = reward - existing.value

            updated = ValueEstimate(
                state_key=state_key,
                action_key=action_key,
                domain=domain,
                value=new_value,
                td_error=td_error,
                update_count=existing.update_count + 1,
            )
        else:
            updated = ValueEstimate(
                state_key=state_key,
                action_key=action_key,
                domain=domain,
                value=reward,
                td_error=reward,
                update_count=1,
            )

        self._store.save_value(updated)

    def _extract_patterns(self, domain: str) -> None:
        """
        Auto-extract behavioral patterns from accumulated episode data.

        This is the self-evolving part: as data accumulates, patterns
        emerge and are stored for future guidance.
        """
        episodes = self._store.query_episodes(domain=domain, limit=100)
        if len(episodes) < self._min_episodes_for_pattern:
            return

        # 1. Extract success strategies: What actions lead to high quality?
        successes = [e for e in episodes if e.success and e.quality >= 0.7]
        failures = [e for e in episodes if not e.success]

        if len(successes) >= 3:
            # Find common action patterns in successes
            action_counts: Dict[str, int] = defaultdict(int)
            for e in successes:
                for key, val in e.action.items():
                    action_counts[f"{key}={val}"] += 1

            for action_str, count in action_counts.items():
                if count >= 3:
                    confidence = count / len(successes)
                    pattern_id = hashlib.md5(f"success_{domain}_{action_str}".encode()).hexdigest()[
                        :12
                    ]

                    self._store.save_pattern(
                        PatternRecord(
                            pattern_id=pattern_id,
                            source_domain=domain,
                            pattern_type="success_strategy",
                            description=f"In {domain}, {action_str} correlates with success ({count}/{len(successes)} episodes)",
                            conditions={"domain": domain},
                            recommendation=f"When working on {domain} tasks, prefer {action_str}",
                            confidence=confidence,
                            evidence_count=count,
                            applicable_domains=[domain],
                        )
                    )

        # 2. Extract failure avoidance patterns
        if len(failures) >= 3:
            error_counts: Dict[str, int] = defaultdict(int)
            for e in failures:
                if e.error_type:
                    error_counts[e.error_type] += 1

            for error_type, count in error_counts.items():
                if count >= 2:
                    pattern_id = hashlib.md5(f"failure_{domain}_{error_type}".encode()).hexdigest()[
                        :12
                    ]

                    self._store.save_pattern(
                        PatternRecord(
                            pattern_id=pattern_id,
                            source_domain=domain,
                            pattern_type="failure_avoidance",
                            description=f"In {domain}, {error_type} errors occur frequently ({count} times)",
                            conditions={"domain": domain, "error_type": error_type},
                            recommendation=f"Add error handling for {error_type} in {domain} tasks",
                            confidence=min(0.9, count / len(failures)),
                            evidence_count=count,
                            applicable_domains=[domain],
                        )
                    )

        logger.debug(f"Pattern extraction complete for domain={domain}")

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
        cached = self._success_rate_cache.get(key)
        if cached and (time.time() - cached[1]) < self._cache_ttl:
            return cached[0], int(cached[1])  # Using cached timestamp slot for total

        rate, total = self._store.get_success_rate(domain, task_type)
        self._success_rate_cache[key] = (rate, time.time())
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
        """Invalidate relevant caches."""
        key = f"{domain}:{task_type}"
        self._success_rate_cache.pop(key, None)
        self._pattern_cache.pop(domain, None)

    @staticmethod
    def _make_state_key(domain: str, task_type: str) -> str:
        return f"{domain}:{task_type}" if task_type else domain

    @staticmethod
    def _make_action_key(action: Dict[str, Any]) -> str:
        parts = sorted(
            f"{k}={v}" for k, v in action.items() if isinstance(v, (str, int, float, bool))
        )
        return "|".join(parts[:5]) or "default"

    @staticmethod
    def _truncate_dict(d: Dict[str, Any], max_str_len: int = 500) -> Dict[str, Any]:
        """Truncate string values in a dict to avoid DB bloat."""
        result = {}
        for k, v in list(d.items())[:20]:
            if isinstance(v, str) and len(v) > max_str_len:
                result[k] = v[:max_str_len] + "..."
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
