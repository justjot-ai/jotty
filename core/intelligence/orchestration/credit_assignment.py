"""
Credit Assignment and Counterfactual Credit Assignment for Improvements

Based on: "Counterfactual Credit Assignment in Multi-Agent Reinforcement Learning"
https://arxiv.org/abs/2011.09464

Implements:
1. Credit Assignment: Which improvements contributed to success?
2. Counterfactual Credit Assignment: What if improvement wasn't applied?
3. Improvement Prioritization: Rank improvements by credit scores
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ImprovementCredit:
    """Credit score for an improvement."""

    improvement_id: str
    improvement: Dict[str, Any]

    # Credit scores
    direct_credit: float = 0.0  # Direct impact (score delta)
    counterfactual_credit: float = 0.0  # Counterfactual impact
    combined_credit: float = 0.0  # Combined credit score

    # Metadata
    application_count: int = 0  # How many times applied
    success_count: int = 0  # How many times led to success
    avg_score_delta: float = 0.0  # Average score improvement
    last_applied: Optional[datetime] = None
    contexts: List[str] = field(default_factory=list)  # Contexts where applied

    # Quality metrics
    is_duplicate: bool = False
    similarity_score: float = 0.0  # Similarity to other improvements
    confidence: float = 0.5  # Confidence in improvement quality


class CreditAssignment:
    """
    Credit assignment system for improvements.

    Tracks which improvements contribute to success and assigns credit scores.
    """

    def __init__(self) -> None:
        self.improvement_credits: Dict[str, ImprovementCredit] = {}
        self.improvement_history: List[Dict[str, Any]] = []  # Track application history

    def record_improvement_application(
        self,
        improvement: Dict[str, Any],
        student_score: float,
        teacher_score: float,
        final_score: float,
        context: Dict[str, Any],
    ) -> ImprovementCredit:
        """
        Record when an improvement is applied and calculate credit.

        Args:
            improvement: Improvement dictionary
            student_score: Score before improvement
            teacher_score: Score from teacher (if available)
            final_score: Final score after improvement
            context: Context where improvement was applied

        Returns:
            ImprovementCredit with calculated credit scores
        """
        improvement_id = self._get_improvement_id(improvement)

        # Get or create credit record
        if improvement_id not in self.improvement_credits:
            self.improvement_credits[improvement_id] = ImprovementCredit(
                improvement_id=improvement_id, improvement=improvement
            )

        credit = self.improvement_credits[improvement_id]

        # Calculate direct credit (score delta)
        score_delta = final_score - student_score
        direct_credit = max(0.0, score_delta)  # Only positive credit

        # Update credit record
        credit.application_count += 1
        credit.last_applied = datetime.now()
        credit.contexts.append(str(context.get("task", "unknown"))[:100])

        # Update average score delta
        total_delta = credit.avg_score_delta * (credit.application_count - 1) + score_delta
        credit.avg_score_delta = total_delta / credit.application_count

        # Update direct credit (weighted average)
        if credit.application_count == 1:
            credit.direct_credit = direct_credit
        else:
            # Exponential moving average (recent applications weighted more)
            alpha = 0.3  # Learning rate
            credit.direct_credit = alpha * direct_credit + (1 - alpha) * credit.direct_credit

        # Track success
        if final_score >= 0.8:  # Threshold for success
            credit.success_count += 1

        # Calculate success rate
        success_rate = (
            credit.success_count / credit.application_count if credit.application_count > 0 else 0.0
        )

        # Update confidence based on success rate and consistency
        credit.confidence = (success_rate * 0.6) + (min(1.0, credit.avg_score_delta) * 0.4)

        # Record in history
        self.improvement_history.append(
            {
                "improvement_id": improvement_id,
                "student_score": student_score,
                "teacher_score": teacher_score,
                "final_score": final_score,
                "score_delta": score_delta,
                "timestamp": datetime.now().isoformat(),
                "context": context.get("task", "unknown"),
            }
        )

        logger.debug(
            f"Recorded improvement application: {improvement_id}, delta: {score_delta:.3f}, credit: {credit.direct_credit:.3f}"
        )

        return credit

    def calculate_counterfactual_credit(
        self,
        improvement_id: str,
        baseline_scores: List[float],
        with_improvement_scores: List[float],
    ) -> float:
        """
        Calculate counterfactual credit: What if improvement wasn't applied?

        Based on: Counterfactual = E[score | with improvement] - E[score | without improvement]

        Args:
            improvement_id: ID of improvement
            baseline_scores: Scores without improvement
            with_improvement_scores: Scores with improvement

        Returns:
            Counterfactual credit score
        """
        if not baseline_scores or not with_improvement_scores:
            return 0.0

        # Calculate expected values
        baseline_mean = sum(baseline_scores) / len(baseline_scores)
        with_improvement_mean = sum(with_improvement_scores) / len(with_improvement_scores)

        # Counterfactual credit = difference in expected values
        counterfactual_credit = with_improvement_mean - baseline_mean

        # Update credit record
        if improvement_id in self.improvement_credits:
            credit = self.improvement_credits[improvement_id]
            credit.counterfactual_credit = counterfactual_credit

            # Combined credit = weighted combination
            # Direct credit (60%) + Counterfactual credit (40%)
            credit.combined_credit = (credit.direct_credit * 0.6) + (counterfactual_credit * 0.4)

        logger.debug(f"Counterfactual credit for {improvement_id}: {counterfactual_credit:.3f}")

        return counterfactual_credit

    def detect_duplicates(
        self, improvements: List[Dict[str, Any]], similarity_threshold: float = 0.8
    ) -> Dict[str, List[str]]:
        """
        Detect duplicate or very similar improvements.

        Args:
            improvements: List of improvements to check
            similarity_threshold: Threshold for considering improvements similar

        Returns:
            Dictionary mapping improvement_id to list of similar improvement_ids
        """
        duplicates = {}

        for i, imp1 in enumerate(improvements):
            imp1_id = self._get_improvement_id(imp1)
            pattern1 = imp1.get("learned_pattern", "").lower()

            if imp1_id not in duplicates:
                duplicates[imp1_id] = []

            for j, imp2 in enumerate(improvements[i + 1 :], i + 1):
                imp2_id = self._get_improvement_id(imp2)
                pattern2 = imp2.get("learned_pattern", "").lower()

                # Calculate similarity (simple Jaccard similarity on words)
                similarity = self._calculate_similarity(pattern1, pattern2)

                if similarity >= similarity_threshold:
                    duplicates[imp1_id].append(imp2_id)
                    if imp2_id not in duplicates:
                        duplicates[imp2_id] = []
                    duplicates[imp2_id].append(imp1_id)

                    # Mark as duplicates in credit records
                    if imp1_id in self.improvement_credits:
                        self.improvement_credits[imp1_id].is_duplicate = True
                        self.improvement_credits[imp1_id].similarity_score = similarity
                    if imp2_id in self.improvement_credits:
                        self.improvement_credits[imp2_id].is_duplicate = True
                        self.improvement_credits[imp2_id].similarity_score = similarity

        return duplicates

    def prioritize_improvements(
        self,
        improvements: List[Dict[str, Any]],
        max_improvements: Optional[int] = None,
        min_credit_threshold: float = 0.1,
    ) -> List[Dict[str, Any]]:
        """
        Prioritize improvements based on credit scores.

        Args:
            improvements: List of improvements to prioritize
            max_improvements: Maximum number of improvements to return
            min_credit_threshold: Minimum credit score to include

        Returns:
            Prioritized list of improvements
        """
        # Calculate credits for all improvements
        prioritized = []

        for imp in improvements:
            imp_id = self._get_improvement_id(imp)

            # Get credit record
            if imp_id in self.improvement_credits:
                credit = self.improvement_credits[imp_id]

                # Use combined credit if available, otherwise direct credit
                credit_score = (
                    credit.combined_credit if credit.combined_credit > 0 else credit.direct_credit
                )

                # Penalize duplicates
                if credit.is_duplicate:
                    credit_score *= 0.5  # Reduce credit for duplicates

                # Filter by threshold
                if credit_score >= min_credit_threshold:
                    prioritized.append(
                        {
                            "improvement": imp,
                            "credit_score": credit_score,
                            "direct_credit": credit.direct_credit,
                            "counterfactual_credit": credit.counterfactual_credit,
                            "confidence": credit.confidence,
                            "application_count": credit.application_count,
                            "success_rate": (
                                credit.success_count / credit.application_count
                                if credit.application_count > 0
                                else 0.0
                            ),
                            "is_duplicate": credit.is_duplicate,
                        }
                    )
            else:
                # New improvement - give default credit
                prioritized.append(
                    {
                        "improvement": imp,
                        "credit_score": 0.5,  # Default credit for new improvements
                        "direct_credit": 0.0,
                        "counterfactual_credit": 0.0,
                        "confidence": 0.5,
                        "application_count": 0,
                        "success_rate": 0.0,
                        "is_duplicate": False,
                    }
                )

        # Sort by credit score (descending)
        prioritized.sort(key=lambda x: x["credit_score"], reverse=True)

        # Remove duplicates (keep highest credit)
        seen_patterns = set()
        deduplicated = []
        for item in prioritized:
            pattern = item["improvement"].get("learned_pattern", "").lower()
            pattern_hash = hash(pattern[:100])  # Use first 100 chars as hash

            if pattern_hash not in seen_patterns:
                seen_patterns.add(pattern_hash)
                deduplicated.append(item)
            elif item["credit_score"] > 0.5:  # Keep if high credit
                # Replace existing with higher credit
                for i, existing in enumerate(deduplicated):
                    if (
                        hash(existing["improvement"].get("learned_pattern", "").lower()[:100])
                        == pattern_hash
                    ):
                        if item["credit_score"] > existing["credit_score"]:
                            deduplicated[i] = item
                        break

        # Limit to max_improvements
        if max_improvements:
            deduplicated = deduplicated[:max_improvements]

        # Return just the improvements (sorted by priority)
        return [item["improvement"] for item in deduplicated]

    def prune_low_impact_improvements(
        self,
        improvements: List[Dict[str, Any]],
        min_credit_threshold: float = 0.2,
        min_application_count: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Prune low-impact improvements.

        Args:
            improvements: List of improvements
            min_credit_threshold: Minimum credit score to keep
            min_application_count: Minimum times improvement must be applied

        Returns:
            Filtered list of improvements
        """
        pruned = []

        for imp in improvements:
            imp_id = self._get_improvement_id(imp)

            if imp_id in self.improvement_credits:
                credit = self.improvement_credits[imp_id]
                credit_score = (
                    credit.combined_credit if credit.combined_credit > 0 else credit.direct_credit
                )

                # Keep if meets thresholds
                if (
                    credit_score >= min_credit_threshold
                    and credit.application_count >= min_application_count
                ):
                    pruned.append(imp)
            else:
                # Keep new improvements (haven't been evaluated yet)
                pruned.append(imp)

        return pruned

    def _get_improvement_id(self, improvement: Dict[str, Any]) -> str:
        """Generate unique ID for improvement."""
        pattern = improvement.get("learned_pattern", "")
        task = improvement.get("task", "")
        timestamp = improvement.get("timestamp", "")
        return f"{hash(pattern[:100])}_{hash(task[:50])}_{timestamp}"

    def _calculate_similarity(self, pattern1: str, pattern2: str) -> float:
        """Calculate similarity between two patterns (Jaccard similarity on words)."""
        words1 = set(pattern1.split())
        words2 = set(pattern2.split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def get_credit_statistics(self) -> Dict[str, Any]:
        """Get statistics about credit assignment."""
        if not self.improvement_credits:
            return {"total_improvements": 0}

        credits = list(self.improvement_credits.values())

        return {
            "total_improvements": len(credits),
            "avg_direct_credit": sum(c.direct_credit for c in credits) / len(credits),
            "avg_counterfactual_credit": sum(c.counterfactual_credit for c in credits)
            / len(credits),
            "avg_combined_credit": sum(c.combined_credit for c in credits) / len(credits),
            "total_applications": sum(c.application_count for c in credits),
            "total_successes": sum(c.success_count for c in credits),
            "duplicates": sum(1 for c in credits if c.is_duplicate),
        }
