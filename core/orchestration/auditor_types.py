"""
Auditor Types - Different Validation Strategies

Integrates OAgents verification strategies as different auditor types.
Based on OAgents list-wise and pair-wise verification approaches.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class AuditorType(Enum):
    """Types of auditors available."""
    SINGLE = "single"  # Single result validation (default, existing Auditor)
    LIST_WISE = "list_wise"  # Verify multiple results, merge best (OAgents approach)
    PAIR_WISE = "pair_wise"  # Pair-wise comparison (OAgents approach)
    CONFIDENCE_BASED = "confidence_based"  # Select based on confidence scores


@dataclass
class VerificationResult:
    """Result from verification of a single result."""
    result: Any
    score: float  # Verification score (0.0-1.0)
    confidence: float  # Confidence in result (0.0-1.0)
    reasoning: str  # Why this score
    passed: bool  # Whether it passed verification


@dataclass
class MergedResult:
    """Merged result from multiple verifications."""
    final_result: Any
    verification_score: float  # Best verification score
    confidence: float  # Confidence in merged result
    source_results: List[VerificationResult]  # All results that were verified
    merge_strategy: str  # Strategy used to merge
    reasoning: str  # Why this result was selected


class ListWiseAuditor:
    """
    List-wise verification auditor (OAgents approach).
    
    Verifies multiple results and merges the best one.
    This is the best-performing verification strategy in OAgents.
    
    Usage:
        auditor = ListWiseAuditor(verification_func=my_verify_func)
        merged = auditor.verify_and_merge(results)
    """
    
    def __init__(
        self,
        verification_func: Optional[callable] = None,
        merge_strategy: str = "best_score",  # "best_score", "consensus", "weighted"
        min_results: int = 2,
        max_results: int = 5
    ):
        """
        Initialize list-wise auditor.
        
        Args:
            verification_func: Function to verify a single result
                Signature: (result, context) -> VerificationResult
            merge_strategy: How to merge results ("best_score", "consensus", "weighted")
            min_results: Minimum number of results to verify
            max_results: Maximum number of results to verify
        """
        self.verification_func = verification_func
        self.merge_strategy = merge_strategy
        self.min_results = min_results
        self.max_results = max_results
    
    def verify_result(
        self,
        result: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> VerificationResult:
        """
        Verify a single result.
        
        Args:
            result: Result to verify
            context: Additional context for verification
            
        Returns:
            VerificationResult
        """
        if self.verification_func:
            return self.verification_func(result, context)
        
        # Default verification (simple pass/fail)
        # In practice, this should use LLM or custom logic
        return VerificationResult(
            result=result,
            score=0.8,  # Default score
            confidence=0.8,
            reasoning="Default verification (implement custom verification_func)",
            passed=True
        )
    
    def verify_and_merge(
        self,
        results: List[Any],
        context: Optional[Dict[str, Any]] = None
    ) -> MergedResult:
        """
        Verify multiple results and merge the best one.
        
        This is the list-wise verification strategy from OAgents.
        
        Args:
            results: List of results to verify
            context: Additional context for verification
            
        Returns:
            MergedResult with the best result
        """
        if len(results) < self.min_results:
            logger.warning(
                f"Only {len(results)} results provided, need at least {self.min_results}. "
                f"Using single result."
            )
            if results:
                verified = self.verify_result(results[0], context)
                return MergedResult(
                    final_result=results[0],
                    verification_score=verified.score,
                    confidence=verified.confidence,
                    source_results=[verified],
                    merge_strategy="single",
                    reasoning="Only one result available"
                )
            else:
                raise ValueError("No results to verify")
        
        # Limit to max_results
        results_to_verify = results[:self.max_results]
        
        # Verify all results
        verified_results: List[VerificationResult] = []
        for result in results_to_verify:
            verified = self.verify_result(result, context)
            verified_results.append(verified)
        
        # Merge based on strategy
        if self.merge_strategy == "best_score":
            return self._merge_best_score(verified_results)
        elif self.merge_strategy == "consensus":
            return self._merge_consensus(verified_results)
        elif self.merge_strategy == "weighted":
            return self._merge_weighted(verified_results)
        else:
            raise ValueError(f"Unknown merge strategy: {self.merge_strategy}")
    
    def _merge_best_score(
        self,
        verified_results: List[VerificationResult]
    ) -> MergedResult:
        """Merge by selecting result with best verification score."""
        best = max(verified_results, key=lambda r: r.score)
        
        return MergedResult(
            final_result=best.result,
            verification_score=best.score,
            confidence=best.confidence,
            source_results=verified_results,
            merge_strategy="best_score",
            reasoning=f"Selected result with highest verification score: {best.score:.2f}"
        )
    
    def _merge_consensus(
        self,
        verified_results: List[VerificationResult]
    ) -> MergedResult:
        """Merge by finding consensus among results."""
        # Find results that passed verification
        passed_results = [r for r in verified_results if r.passed]
        
        if not passed_results:
            # No consensus, use best score
            return self._merge_best_score(verified_results)
        
        # If all passed, use best score
        if len(passed_results) == len(verified_results):
            return self._merge_best_score(verified_results)
        
        # Use best of passed results
        best_passed = max(passed_results, key=lambda r: r.score)
        
        return MergedResult(
            final_result=best_passed.result,
            verification_score=best_passed.score,
            confidence=best_passed.confidence,
            source_results=verified_results,
            merge_strategy="consensus",
            reasoning=f"Consensus: {len(passed_results)}/{len(verified_results)} passed, selected best"
        )
    
    def _merge_weighted(
        self,
        verified_results: List[VerificationResult]
    ) -> MergedResult:
        """Merge by weighted average (if results are numeric)."""
        # For now, fall back to best score
        # Can be enhanced for numeric results
        return self._merge_best_score(verified_results)


class PairWiseAuditor:
    """
    Pair-wise verification auditor (OAgents approach).
    
    Compares results in pairs and selects the best.
    Less effective than list-wise but faster.
    
    Usage:
        auditor = PairWiseAuditor(comparison_func=my_compare_func)
        best = auditor.verify_and_select(results)
    """
    
    def __init__(
        self,
        comparison_func: Optional[callable] = None
    ):
        """
        Initialize pair-wise auditor.
        
        Args:
            comparison_func: Function to compare two results
                Signature: (result1, result2) -> (better_result, score_diff)
        """
        self.comparison_func = comparison_func
    
    def compare_results(
        self,
        result1: Any,
        result2: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, float]:
        """
        Compare two results and return the better one.
        
        Args:
            result1: First result
            result2: Second result
            context: Additional context
            
        Returns:
            (better_result, score_difference)
        """
        if self.comparison_func:
            return self.comparison_func(result1, result2, context)
        
        # Default: return first result (implement custom comparison)
        return result1, 0.0
    
    def verify_and_select(
        self,
        results: List[Any],
        context: Optional[Dict[str, Any]] = None
    ) -> MergedResult:
        """
        Verify results using pair-wise comparison.
        
        Args:
            results: List of results to compare
            context: Additional context
            
        Returns:
            MergedResult with the best result
        """
        if len(results) < 2:
            if results:
                return MergedResult(
                    final_result=results[0],
                    verification_score=0.8,
                    confidence=0.8,
                    source_results=[],
                    merge_strategy="pair_wise",
                    reasoning="Only one result available"
                )
            else:
                raise ValueError("No results to compare")
        
        # Pair-wise tournament: compare all pairs, select best
        best_result = results[0]
        best_score = 0.0
        
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                better, score_diff = self.compare_results(
                    results[i],
                    results[j],
                    context
                )
                
                if better == results[j] and score_diff > best_score:
                    best_result = results[j]
                    best_score = score_diff
        
        return MergedResult(
            final_result=best_result,
            verification_score=best_score + 0.5,  # Normalize
            confidence=0.8,
            source_results=[],
            merge_strategy="pair_wise",
            reasoning=f"Pair-wise comparison selected best result"
        )


class ConfidenceBasedAuditor:
    """
    Confidence-based auditor (select based on confidence scores).
    
    Simple strategy: select result with highest confidence.
    """
    
    def __init__(
        self,
        confidence_func: Optional[callable] = None
    ):
        """
        Initialize confidence-based auditor.
        
        Args:
            confidence_func: Function to get confidence score
                Signature: (result) -> float (0.0-1.0)
        """
        self.confidence_func = confidence_func
    
    def get_confidence(
        self,
        result: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Get confidence score for a result."""
        if self.confidence_func:
            return self.confidence_func(result, context)
        
        # Default confidence
        if hasattr(result, 'confidence'):
            return float(result.confidence)
        
        return 0.8  # Default
    
    def select_best(
        self,
        results: List[Any],
        context: Optional[Dict[str, Any]] = None
    ) -> MergedResult:
        """
        Select result with highest confidence.
        
        Args:
            results: List of results
            context: Additional context
            
        Returns:
            MergedResult with best result
        """
        if not results:
            raise ValueError("No results to select from")
        
        # Get confidence for each result
        scored_results = [
            (result, self.get_confidence(result, context))
            for result in results
        ]
        
        # Select best
        best_result, best_confidence = max(scored_results, key=lambda x: x[1])
        
        return MergedResult(
            final_result=best_result,
            verification_score=best_confidence,
            confidence=best_confidence,
            source_results=[],
            merge_strategy="confidence_based",
            reasoning=f"Selected result with highest confidence: {best_confidence:.2f}"
        )
