"""
Reflection Engine - Mid-execution and post-execution reflection.

Handles reflection during and after execution, recording insights
for future learning.

Extracted from learning_service.py for modularity.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Optional

from .learning_store import LearningStore, ReflectionRecord
from .response_analyzer import analyze_response

logger = logging.getLogger(__name__)


class ReflectionEngine:
    """
    Handles reflection during and after execution episodes.

    Takes a LearningStore instance for persisting reflections and
    querying past failures and patterns.
    """

    def __init__(self, store: LearningStore, quality_threshold: float = 0.75) -> None:
        self._store = store
        self._quality_threshold = quality_threshold

    def auto_reflect(
        self,
        episode_id: str,
        domain: str,
        task_type: str,
        context: Dict[str, Any],
        outcome: Dict[str, Any],
        success: bool,
        quality: float,
        unit_name: str,
        get_value_fn: Any = None,
        make_state_key_fn: Any = None,
    ) -> None:
        """
        Automatic post-execution reflection for significant outcomes.

        Triggers on:
        - High-quality successes (quality >= threshold) to learn *what worked*
        - Failures to learn *what went wrong*
        - Large quality variance from domain average to learn *what changed*

        Args:
            get_value_fn: Callable(state_key, action_key, domain) -> ValueEstimate
            make_state_key_fn: Callable(domain, task_type) -> str
        """
        # Get domain baseline for comparison
        baseline_value = 0.5
        if get_value_fn and make_state_key_fn:
            state_key = make_state_key_fn(domain, task_type)
            baseline = get_value_fn(state_key, "", domain)
            baseline_value = baseline.value if baseline else 0.5

        # Determine if this outcome is significant enough to reflect on
        is_high_quality = success and quality >= self._quality_threshold
        is_failure = not success
        is_surprise = abs(quality - baseline_value) > 0.3  # Large deviation

        if not (is_high_quality or is_failure or is_surprise):
            return

        # Build observation
        goal = ""
        if isinstance(context, dict):
            goal = str(context.get("goal", context.get("message", "")))[:200]

        if is_high_quality:
            observation = (
                f"HIGH QUALITY outcome in {domain}/{task_type}: "
                f"quality={quality:.2f}, goal='{goal}'. "
                f"Structural features: "
                f"has_code={outcome.get('has_code', 'N/A')}, "
                f"has_headings={outcome.get('has_headings', 'N/A')}, "
                f"word_count={outcome.get('word_count', 'N/A')}, "
                f"goal_coverage={outcome.get('goal_coverage', 'N/A')}"
            )
            analysis = (
                f"Success pattern: quality {quality:.2f} exceeds threshold "
                f"{self._quality_threshold}. "
                f"Baseline for domain is {baseline_value:.2f}."
            )
        elif is_failure:
            observation = (
                f"FAILURE in {domain}/{task_type}: " f"quality={quality:.2f}, goal='{goal}'"
            )
            analysis = (
                f"Failure analysis: success=False, quality={quality:.2f}. "
                f"Baseline for domain is {baseline_value:.2f}."
            )
        else:  # is_surprise
            direction = "above" if quality > baseline_value else "below"
            observation = (
                f"SURPRISE outcome in {domain}/{task_type}: "
                f"quality={quality:.2f} is {direction} baseline {baseline_value:.2f}, "
                f"goal='{goal}'"
            )
            analysis = (
                f"Deviation: quality {quality:.2f} vs baseline {baseline_value:.2f} "
                f"(delta={quality - baseline_value:+.2f})."
            )

        self.reflect(
            episode_id=episode_id,
            step=0,
            observation=observation,
            unit_name=unit_name or "auto_reflect",
            analysis=analysis,
        )
        logger.debug(
            f"Auto-reflected on {episode_id}: "
            f"high_q={is_high_quality}, fail={is_failure}, surprise={is_surprise}"
        )

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

    def post_execution_reflect(
        self,
        episode_id: str,
        goal: str,
        content: str,
        domain: str,
        quality_score: float,
        execution_time: float,
    ) -> None:
        """
        Post-execution reflection: analyze what worked and record insights.

        Called after every successful execution to build strategic knowledge
        about what response patterns lead to quality outcomes.
        """
        analysis = analyze_response(content, goal)

        # Build observation from analysis
        obs_parts = [f"domain={domain}, quality={quality_score:.2f}, time={execution_time:.1f}s"]
        if analysis.get("has_code"):
            obs_parts.append(f"included {analysis.get('code_block_count', 0)} code blocks")
        if analysis.get("has_headings"):
            obs_parts.append("used section headings")
        if analysis.get("has_citations"):
            obs_parts.append("cited sources")
        if analysis.get("has_math"):
            obs_parts.append("included math formulations")
        obs_parts.append(f"word_count={analysis.get('word_count', 0)}")
        obs_parts.append(f"goal_coverage={analysis.get('goal_coverage', 0):.0%}")
        observation = "; ".join(obs_parts)

        # Build analysis string
        if quality_score >= 0.85:
            analysis_str = (
                f"HIGH QUALITY response in {domain}. Key factors: "
                f"structure={analysis.get('structure_score', 0):.2f}, "
                f"coverage={analysis.get('goal_coverage', 0):.2f}. "
                f"This approach should be reinforced."
            )
        elif quality_score >= 0.5:
            analysis_str = f"ADEQUATE response in {domain}. " f"Improvement areas: "
            if analysis.get("goal_coverage", 0) < 0.7:
                analysis_str += "increase goal coverage; "
            if analysis.get("structure_score", 0) < 0.3:
                analysis_str += "add more structure (headings, lists); "
            if analysis.get("word_count", 0) < 500:
                analysis_str += "increase depth; "
        else:
            analysis_str = (
                f"LOW QUALITY response in {domain}. "
                f"Significant improvement needed in coverage and depth."
            )

        try:
            self.reflect(
                episode_id=episode_id,
                step=0,
                observation=observation,
                unit_name="Orchestrator",
                analysis=analysis_str,
            )
        except Exception as e:
            logger.debug(f"Post-execution reflection failed: {e}")

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
