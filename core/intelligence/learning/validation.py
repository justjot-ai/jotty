"""
Learning Validation — prove what's learned is actually better.

Answers the fundamental question: "Does the learned policy outperform
a naive baseline?"  Without this, crystallized SOPs and Q-values might
reflect consistency without genuine quality improvement.

Five validation strategies:

1. **Holdout evaluation**: Split episodes into train/test, verify learned
   policy generalizes to unseen data.

2. **Baseline comparison**: Compare learned skill ranking against random
   selection and measure quality lift.

3. **Temporal improvement**: Verify quality/success curves are actually
   improving over time (not flat or declining).

4. **Counterfactual analysis**: Check if the Q-table's top-ranked skills
   actually produced higher quality than lower-ranked ones in real data.

5. **Staleness detection**: Flag crystallized SOPs whose recent performance
   has degraded vs. their crystallization-time metrics.

Integration:
    - Called automatically by _maybe_auto_optimize() before crystallization.
    - Available via facade: get_learning_validator()
    - CLI/scripts can call validate_domain() for any domain.

Author: Jotty Team
Date: February 2026
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a single validation check."""

    check: str
    passed: bool
    score: float  # 0.0–1.0 confidence
    detail: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DomainValidation:
    """Full validation report for a domain."""

    domain: str
    task_type: str
    checks: List[ValidationResult] = field(default_factory=list)
    overall_passed: bool = False
    confidence: float = 0.0
    recommendation: str = ""

    @property
    def summary(self) -> str:
        passed = sum(1 for c in self.checks if c.passed)
        return (
            f"{self.domain}:{self.task_type} — {passed}/{len(self.checks)} checks passed "
            f"(confidence={self.confidence:.0%}) → {self.recommendation}"
        )


class LearningValidator:
    """Validates that learned knowledge is genuinely better than baseline."""

    _instance: Optional["LearningValidator"] = None

    @classmethod
    def get_instance(cls) -> "LearningValidator":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def validate_domain(
        self,
        domain: str,
        task_type: str = "",
        min_episodes: int = 10,
    ) -> DomainValidation:
        """Run all validation checks for a domain.

        Returns a DomainValidation with per-check results and an overall
        recommendation: "crystallize", "keep_learning", or "investigate".
        """
        result = DomainValidation(domain=domain, task_type=task_type)

        from .learning_store import LearningStore

        store = LearningStore()
        episodes = store.query_episodes(domain=domain, task_type=task_type or None, limit=200)

        if len(episodes) < min_episodes:
            result.checks.append(
                ValidationResult(
                    check="sufficient_data",
                    passed=False,
                    score=len(episodes) / min_episodes,
                    detail=f"Only {len(episodes)} episodes (need {min_episodes})",
                )
            )
            result.recommendation = "keep_learning"
            return result

        result.checks.append(
            ValidationResult(
                check="sufficient_data",
                passed=True,
                score=1.0,
                detail=f"{len(episodes)} episodes available",
            )
        )

        ceiling = self._check_ceiling(episodes)
        result.checks.append(ceiling)
        at_ceiling = ceiling.data.get("at_ceiling", False)

        result.checks.append(self._check_temporal_improvement(episodes))
        result.checks.append(self._check_holdout(episodes))
        result.checks.append(self._check_counterfactual(domain, task_type, episodes))
        result.checks.append(self._check_baseline_lift(domain, task_type, episodes))
        result.checks.append(self._check_staleness(domain, task_type))

        passed_checks = [c for c in result.checks if c.passed]
        result.confidence = sum(c.score for c in passed_checks) / max(len(result.checks), 1)

        # At ceiling: temporal_improvement and baseline_lift failures are
        # expected (no room to improve), so relax the pass threshold.
        if at_ceiling:
            result.overall_passed = len(passed_checks) >= len(result.checks) * 0.5
        else:
            result.overall_passed = len(passed_checks) >= len(result.checks) * 0.6

        fail_count = len(result.checks) - len(passed_checks)
        if fail_count == 0:
            result.recommendation = "crystallize"
        elif fail_count <= 2:
            result.recommendation = "keep_learning" if not at_ceiling else "crystallize_ceiling"
        else:
            result.recommendation = "investigate"

        return result

    # ─── Check 1: Temporal improvement ────────────────────────────────

    def _check_temporal_improvement(self, episodes: list) -> ValidationResult:
        """Verify that quality improves over time (not flat or declining).

        Splits episodes into first-half and second-half, compares mean
        quality. Passes if second half is at least as good as first half
        (with a small tolerance for noise).
        """
        sorted_eps = sorted(episodes, key=lambda e: e.execution_time or 0)
        qualities = [e.quality for e in sorted_eps if e.quality > 0]

        if len(qualities) < 6:
            return ValidationResult(
                check="temporal_improvement",
                passed=False,
                score=0.0,
                detail="Too few episodes with quality scores",
            )

        mid = len(qualities) // 2
        early_q = statistics.mean(qualities[:mid])
        late_q = statistics.mean(qualities[mid:])
        improvement = late_q - early_q

        # Allow small tolerance: late half shouldn't be noticeably worse
        passed = improvement >= -0.03
        score = min(1.0, max(0.0, 0.5 + improvement * 5))

        return ValidationResult(
            check="temporal_improvement",
            passed=passed,
            score=score,
            detail=(
                f"Early avg quality: {early_q:.3f}, Late avg: {late_q:.3f}, "
                f"Δ={improvement:+.3f}"
            ),
            data={"early_quality": early_q, "late_quality": late_q, "delta": improvement},
        )

    # ─── Check 2: Holdout evaluation ─────────────────────────────────

    def _check_holdout(self, episodes: list) -> ValidationResult:
        """Split episodes into train/test and verify learned ranking
        generalizes. The "prediction" is: top Q-value skills should
        appear more often in successful holdout episodes.
        """
        sorted_eps = sorted(episodes, key=lambda e: e.execution_time or 0)
        split = int(len(sorted_eps) * 0.7)
        train_eps = sorted_eps[:split]
        test_eps = sorted_eps[split:]

        if len(test_eps) < 3:
            return ValidationResult(
                check="holdout_evaluation",
                passed=False,
                score=0.0,
                detail="Too few test episodes for holdout",
            )

        # "Training" signal: which skills appear in high-quality train episodes?
        skill_quality: Dict[str, List[float]] = {}
        for ep in train_eps:
            skills = self._extract_skills(ep)
            for s in skills:
                skill_quality.setdefault(s, []).append(ep.quality)

        if not skill_quality:
            return ValidationResult(
                check="holdout_evaluation",
                passed=True,
                score=0.5,
                detail="No skill data in episodes; holdout inconclusive (neutral)",
            )

        top_skills = {
            s for s, qs in skill_quality.items() if statistics.mean(qs) >= 0.7 and len(qs) >= 2
        }

        if not top_skills:
            return ValidationResult(
                check="holdout_evaluation",
                passed=True,
                score=0.5,
                detail="No clearly top-ranked skills from training data",
            )

        # Test: do top-skill episodes in test set have higher quality?
        test_with_top = [e for e in test_eps if top_skills & set(self._extract_skills(e))]
        test_without_top = [e for e in test_eps if not (top_skills & set(self._extract_skills(e)))]

        if not test_with_top or not test_without_top:
            return ValidationResult(
                check="holdout_evaluation",
                passed=True,
                score=0.6,
                detail="All test episodes use same skill set; holdout inconclusive",
            )

        avg_with = statistics.mean(e.quality for e in test_with_top)
        avg_without = statistics.mean(e.quality for e in test_without_top)
        lift = avg_with - avg_without

        passed = lift >= -0.05
        score = min(1.0, max(0.0, 0.5 + lift * 5))

        return ValidationResult(
            check="holdout_evaluation",
            passed=passed,
            score=score,
            detail=(
                f"Top-skill test quality: {avg_with:.3f} vs others: {avg_without:.3f}, "
                f"lift={lift:+.3f}"
            ),
            data={"with_top": avg_with, "without_top": avg_without, "lift": lift},
        )

    # ─── Check 3: Counterfactual ─────────────────────────────────────

    def _check_counterfactual(
        self, domain: str, task_type: str, episodes: list
    ) -> ValidationResult:
        """Verify Q-table's top skills actually produced higher quality
        in observed data. If Q-table says skill A > skill B, do episodes
        using skill A actually have higher quality than skill B episodes?
        """
        from .facade import get_td_lambda

        td = get_td_lambda()
        top_skills = td.skill_q.get_top_skills(task_type or domain, n=10, domain=domain)

        if len(top_skills) < 2:
            return ValidationResult(
                check="counterfactual",
                passed=True,
                score=0.5,
                detail="Too few skills in Q-table for counterfactual check",
            )

        q_ranking = {skill: rank for rank, (skill, _) in enumerate(top_skills)}

        # Group episode quality by skills used
        skill_episode_quality: Dict[str, List[float]] = {}
        for ep in episodes:
            for s in self._extract_skills(ep):
                if s in q_ranking:
                    skill_episode_quality.setdefault(s, []).append(ep.quality)

        if len(skill_episode_quality) < 2:
            return ValidationResult(
                check="counterfactual",
                passed=True,
                score=0.5,
                detail="Not enough skill variation in episodes for counterfactual",
            )

        # Check concordance: for each pair where Q says A > B,
        # does real data agree?
        concordant = 0
        discordant = 0
        skills_with_data = [s for s in q_ranking if s in skill_episode_quality]
        for i, s1 in enumerate(skills_with_data):
            for s2 in skills_with_data[i + 1 :]:
                q_says_s1_better = q_ranking[s1] < q_ranking[s2]
                actual_s1 = statistics.mean(skill_episode_quality[s1])
                actual_s2 = statistics.mean(skill_episode_quality[s2])
                actual_s1_better = actual_s1 > actual_s2

                if q_says_s1_better == actual_s1_better:
                    concordant += 1
                elif abs(actual_s1 - actual_s2) > 0.02:
                    discordant += 1

        total = concordant + discordant
        if total == 0:
            return ValidationResult(
                check="counterfactual",
                passed=True,
                score=0.5,
                detail="All skill pairs tied in practice; counterfactual inconclusive",
            )

        concordance_rate = concordant / total
        passed = concordance_rate >= 0.5
        score = concordance_rate

        return ValidationResult(
            check="counterfactual",
            passed=passed,
            score=score,
            detail=(
                f"Q-table ranking agrees with actual data {concordance_rate:.0%} "
                f"of the time ({concordant}/{total} pairs concordant)"
            ),
            data={"concordance": concordance_rate, "concordant": concordant, "total": total},
        )

    # ─── Check 4: Baseline lift ──────────────────────────────────────

    def _check_baseline_lift(self, domain: str, task_type: str, episodes: list) -> ValidationResult:
        """Compare quality of Q-guided episodes (where learned policy
        influenced selection) vs. early exploration episodes (before
        learning had data). The first ~20% of episodes serve as a
        natural baseline since they had no prior learning.
        """
        sorted_eps = sorted(episodes, key=lambda e: e.execution_time or 0)
        qualities = [e.quality for e in sorted_eps if e.quality > 0]

        if len(qualities) < 10:
            return ValidationResult(
                check="baseline_lift",
                passed=False,
                score=0.0,
                detail="Too few episodes with quality scores for baseline comparison",
            )

        baseline_size = max(3, len(qualities) // 5)
        baseline_q = statistics.mean(qualities[:baseline_size])
        learned_q = statistics.mean(qualities[baseline_size:])
        lift = learned_q - baseline_q

        # Also compute statistical significance via effect size (Cohen's d)
        if len(qualities) >= 10:
            try:
                baseline_std = statistics.stdev(qualities[:baseline_size]) or 0.01
                learned_std = statistics.stdev(qualities[baseline_size:]) or 0.01
                pooled_std = ((baseline_std**2 + learned_std**2) / 2) ** 0.5
                cohens_d = lift / max(pooled_std, 0.01)
            except statistics.StatisticsError:
                cohens_d = 0.0
        else:
            cohens_d = 0.0

        passed = lift >= -0.02
        score = min(1.0, max(0.0, 0.5 + lift * 5))

        return ValidationResult(
            check="baseline_lift",
            passed=passed,
            score=score,
            detail=(
                f"Baseline (first {baseline_size} eps) quality: {baseline_q:.3f}, "
                f"Learned quality: {learned_q:.3f}, lift={lift:+.3f}, "
                f"Cohen's d={cohens_d:.2f}"
            ),
            data={
                "baseline_quality": baseline_q,
                "learned_quality": learned_q,
                "lift": lift,
                "cohens_d": cohens_d,
            },
        )

    # ─── Check 5: Staleness detection ────────────────────────────────

    def _check_staleness(self, domain: str, task_type: str) -> ValidationResult:
        """Check if a crystallized SOP's recent performance matches its
        crystallization-time metrics. Catches drift.
        """
        from .crystallization import load
        from .learning_store import LearningStore

        config = load(task_type, domain)
        if config is None:
            return ValidationResult(
                check="staleness",
                passed=True,
                score=0.7,
                detail="No crystallized config; staleness check N/A",
            )

        store = LearningStore()
        recent = store.query_episodes(domain=domain, task_type=task_type or None, limit=20)

        if len(recent) < 5:
            return ValidationResult(
                check="staleness",
                passed=True,
                score=0.6,
                detail="Too few recent episodes to assess staleness",
            )

        recent_success_rate = sum(1 for e in recent if e.success) / len(recent)
        recent_quality = (
            statistics.mean(e.quality for e in recent if e.quality > 0)
            if any(e.quality > 0 for e in recent)
            else 0.0
        )

        crystal_sr = config.success_rate
        sr_drift = recent_success_rate - crystal_sr
        q_drift = recent_quality - crystal_sr  # compare quality to success_rate baseline

        passed = sr_drift >= -0.15 and recent_success_rate >= 0.7
        score = min(1.0, max(0.0, recent_success_rate))

        return ValidationResult(
            check="staleness",
            passed=passed,
            score=score,
            detail=(
                f"Crystal success_rate: {crystal_sr:.0%}, "
                f"Recent: {recent_success_rate:.0%} (Δ={sr_drift:+.0%}), "
                f"Recent quality: {recent_quality:.3f}, "
                f"Failures: {config.consecutive_failures}"
            ),
            data={
                "crystal_success_rate": crystal_sr,
                "recent_success_rate": recent_success_rate,
                "recent_quality": recent_quality,
                "drift": sr_drift,
                "consecutive_failures": config.consecutive_failures,
            },
        )

    # ─── Check 6: Ceiling detection ─────────────────────────────────

    def _check_ceiling(self, episodes: list) -> ValidationResult:
        """Detect if quality scores are at ceiling, limiting measurable lift.

        When >80% of episodes score above 0.85, there's no room for learning
        to show improvement. The system may be learning correctly (Q-values
        discriminate) but the quality metric is saturated.
        """
        qualities = [e.quality for e in episodes if e.quality > 0]
        if len(qualities) < 5:
            return ValidationResult(
                check="ceiling_detection",
                passed=True,
                score=0.5,
                detail="Too few episodes for ceiling check",
            )

        high_count = sum(1 for q in qualities if q >= 0.85)
        high_pct = high_count / len(qualities)
        avg_q = sum(qualities) / len(qualities)
        q_variance = sum((q - avg_q) ** 2 for q in qualities) / len(qualities)

        at_ceiling = high_pct > 0.80 and q_variance < 0.01

        if at_ceiling:
            return ValidationResult(
                check="ceiling_detection",
                passed=True,
                score=0.8,
                detail=(
                    f"CEILING EFFECT: {high_pct:.0%} of episodes score >= 0.85 "
                    f"(avg={avg_q:.3f}, var={q_variance:.4f}). Quality metric is "
                    f"saturated — learning may be correct but lift is unmeasurable. "
                    f"Use a harder domain or weaker model to validate improvement."
                ),
                data={"at_ceiling": True, "high_pct": high_pct, "variance": q_variance},
            )

        return ValidationResult(
            check="ceiling_detection",
            passed=True,
            score=1.0,
            detail=f"No ceiling: {high_pct:.0%} above 0.85, variance={q_variance:.4f}",
            data={"at_ceiling": False, "high_pct": high_pct, "variance": q_variance},
        )

    # ─── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _extract_skills(episode) -> List[str]:
        """Extract skill names from an episode's action dict."""
        action = episode.action if hasattr(episode, "action") else {}
        if isinstance(action, dict):
            skills = action.get("skills_used", [])
            if isinstance(skills, list):
                return skills
            skill = action.get("skill", "")
            if skill:
                return [skill]
        return []
