"""
Auditor Selector - Helper to Choose the Right Auditor Type

Helps end users choose the appropriate auditor type based on their needs.
"""
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class UseCase(Enum):
    """Common use cases for auditor selection."""
    CODE_GENERATION = "code_generation"
    DATA_EXTRACTION = "data_extraction"
    ANALYSIS = "analysis"
    CREATIVE_WRITING = "creative_writing"
    REASONING = "reasoning"
    GENERAL = "general"


class Priority(Enum):
    """Priority levels."""
    QUALITY = "quality"  # Quality is most important
    SPEED = "speed"  # Speed is most important
    COST = "cost"  # Cost is most important
    BALANCED = "balanced"  # Balance all factors


@dataclass
class AuditorRecommendation:
    """Recommendation for auditor type."""
    auditor_type: str
    reason: str
    trade_offs: Dict[str, Any]
    config: Dict[str, Any]


class AuditorSelector:
    """
    Helper class to select the right auditor type.
    
    Usage:
        selector = AuditorSelector()
        recommendation = selector.recommend(
            use_case=UseCase.CODE_GENERATION,
            priority=Priority.QUALITY,
            has_multiple_results=True
        )
        print(f"Use: {recommendation.auditor_type}")
        print(f"Reason: {recommendation.reason}")
    """
    
    def recommend(
        self,
        use_case: UseCase = UseCase.GENERAL,
        priority: Priority = Priority.BALANCED,
        has_multiple_results: bool = False,
        cost_sensitive: bool = False,
        latency_sensitive: bool = False,
        quality_critical: bool = False
    ) -> AuditorRecommendation:
        """
        Recommend auditor type based on requirements.
        
        Args:
            use_case: Type of task
            priority: What's most important (quality, speed, cost, balanced)
            has_multiple_results: Whether you have multiple results to verify
            cost_sensitive: Whether cost is a concern
            latency_sensitive: Whether latency is a concern
            quality_critical: Whether quality is critical
            
        Returns:
            AuditorRecommendation with auditor type and reasoning
        """
        # If no multiple results, use single
        if not has_multiple_results:
            return AuditorRecommendation(
                auditor_type="single",
                reason="Only single result available - use default single validation",
                trade_offs={
                    "cost": "Low",
                    "speed": "Fast",
                    "quality": "Baseline"
                },
                config={}
            )
        
        # Quality-critical use cases
        if quality_critical or priority == Priority.QUALITY:
            return self._recommend_for_quality(use_case)
        
        # Speed-critical use cases
        if latency_sensitive or priority == Priority.SPEED:
            return self._recommend_for_speed(use_case)
        
        # Cost-sensitive use cases
        if cost_sensitive or priority == Priority.COST:
            return self._recommend_for_cost(use_case)
        
        # Balanced (default)
        return self._recommend_balanced(use_case)
    
    def _recommend_for_quality(self, use_case: UseCase) -> AuditorRecommendation:
        """Recommend for quality-critical scenarios."""
        # List-wise is best for quality (OAgents research)
        return AuditorRecommendation(
            auditor_type="list_wise",
            reason="Quality is critical - list-wise verification provides best results (OAgents research)",
            trade_offs={
                "cost": "Higher (verifies multiple results)",
                "speed": "Slower (verifies all results)",
                "quality": "Best (selects best verified result)"
            },
            config={
                "enable_list_wise_verification": True,
                "list_wise_min_results": 3,
                "list_wise_max_results": 5,
                "list_wise_merge_strategy": "best_score"
            }
        )
    
    def _recommend_for_speed(self, use_case: UseCase) -> AuditorRecommendation:
        """Recommend for speed-critical scenarios."""
        # Confidence-based is fastest
        return AuditorRecommendation(
            auditor_type="confidence_based",
            reason="Speed is critical - confidence-based selection is fastest",
            trade_offs={
                "cost": "Low (no verification calls)",
                "speed": "Fastest (just selects highest confidence)",
                "quality": "Good (if confidence scores are accurate)"
            },
            config={
                "auditor_type": "confidence_based"
            }
        )
    
    def _recommend_for_cost(self, use_case: UseCase) -> AuditorRecommendation:
        """Recommend for cost-sensitive scenarios."""
        # Pair-wise is cheaper than list-wise
        return AuditorRecommendation(
            auditor_type="pair_wise",
            reason="Cost is a concern - pair-wise verification is cheaper than list-wise",
            trade_offs={
                "cost": "Medium (fewer verification calls than list-wise)",
                "speed": "Medium (faster than list-wise)",
                "quality": "Good (better than single, less than list-wise)"
            },
            config={
                "auditor_type": "pair_wise"
            }
        )
    
    def _recommend_balanced(self, use_case: UseCase) -> AuditorRecommendation:
        """Recommend balanced approach."""
        # Use case-specific recommendations
        if use_case == UseCase.CODE_GENERATION:
            # Code generation: quality matters, but speed too
            return AuditorRecommendation(
                auditor_type="list_wise",
                reason="Code generation: Quality matters, list-wise verification recommended",
                trade_offs={
                    "cost": "Higher",
                    "speed": "Slower",
                    "quality": "Best"
                },
                config={
                    "enable_list_wise_verification": True,
                    "list_wise_min_results": 2,
                    "list_wise_max_results": 4,
                    "list_wise_merge_strategy": "best_score"
                }
            )
        
        elif use_case == UseCase.DATA_EXTRACTION:
            # Data extraction: accuracy critical
            return AuditorRecommendation(
                auditor_type="list_wise",
                reason="Data extraction: Accuracy critical, use list-wise verification",
                trade_offs={
                    "cost": "Higher",
                    "speed": "Slower",
                    "quality": "Best"
                },
                config={
                    "enable_list_wise_verification": True,
                    "list_wise_min_results": 3,
                    "list_wise_max_results": 5,
                    "list_wise_merge_strategy": "consensus"
                }
            )
        
        elif use_case == UseCase.CREATIVE_WRITING:
            # Creative writing: speed matters more
            return AuditorRecommendation(
                auditor_type="confidence_based",
                reason="Creative writing: Speed matters, use confidence-based selection",
                trade_offs={
                    "cost": "Low",
                    "speed": "Fast",
                    "quality": "Good"
                },
                config={
                    "auditor_type": "confidence_based"
                }
            )
        
        else:
            # Default: list-wise (best quality)
            return AuditorRecommendation(
                auditor_type="list_wise",
                reason="Balanced approach: List-wise verification provides best quality",
                trade_offs={
                    "cost": "Higher",
                    "speed": "Slower",
                    "quality": "Best"
                },
                config={
                    "enable_list_wise_verification": True,
                    "list_wise_min_results": 2,
                    "list_wise_max_results": 5,
                    "list_wise_merge_strategy": "best_score"
                }
            )
    
    def get_comparison_table(self) -> str:
        """Get comparison table of auditor types."""
        return """
Auditor Types Comparison:

┌──────────────────┬──────────┬──────────┬──────────┬─────────────────────┐
│ Auditor Type     │ Quality  │ Speed    │ Cost     │ Best For            │
├──────────────────┼──────────┼──────────┼──────────┼─────────────────────┤
│ Single           │ Baseline │ Fastest  │ Lowest   │ Default, single     │
│                  │          │          │          │ result              │
├──────────────────┼──────────┼──────────┼──────────┼─────────────────────┤
│ List-wise        │ Best     │ Slower   │ Higher   │ Quality-critical    │
│                  │          │          │          │ tasks               │
├──────────────────┼──────────┼──────────┼──────────┼─────────────────────┤
│ Pair-wise        │ Good     │ Medium   │ Medium   │ Balanced approach  │
├──────────────────┼──────────┼──────────┼──────────┼─────────────────────┤
│ Confidence-based │ Good     │ Fast     │ Low      │ Speed-critical     │
│                  │          │          │          │ tasks               │
└──────────────────┴──────────┴──────────┴──────────┴─────────────────────┘

Research: List-wise verification performs best in OAgents benchmarks.
"""


def recommend_auditor(
    use_case: str = "general",
    priority: str = "balanced",
    has_multiple_results: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to recommend auditor type.
    
    Args:
        use_case: "code_generation", "data_extraction", "analysis", "creative_writing", "reasoning", "general"
        priority: "quality", "speed", "cost", "balanced"
        has_multiple_results: Whether you have multiple results
        **kwargs: Additional parameters
        
    Returns:
        Dict with recommendation
        
    Example:
        rec = recommend_auditor(
            use_case="code_generation",
            priority="quality",
            has_multiple_results=True
        )
        print(rec['auditor_type'])  # "list_wise"
    """
    selector = AuditorSelector()
    
    use_case_enum = UseCase(use_case)
    priority_enum = Priority(priority)
    
    recommendation = selector.recommend(
        use_case=use_case_enum,
        priority=priority_enum,
        has_multiple_results=has_multiple_results,
        **kwargs
    )
    
    return {
        "auditor_type": recommendation.auditor_type,
        "reason": recommendation.reason,
        "trade_offs": recommendation.trade_offs,
        "config": recommendation.config
    }
