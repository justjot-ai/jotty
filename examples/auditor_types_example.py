"""
Auditor Types Example

Demonstrates different auditor types:
- Single validation (default)
- List-wise verification (OAgents approach)
- Pair-wise verification
- Confidence-based selection
"""
import sys
from pathlib import Path

# Add Jotty to path
jotty_path = Path(__file__).parent.parent
sys.path.insert(0, str(jotty_path))

from core.orchestration.auditor_types import (
    AuditorType,
    ListWiseAuditor,
    PairWiseAuditor,
    ConfidenceBasedAuditor,
    VerificationResult
)
from core.foundation.data_structures import SwarmConfig


def example_list_wise_auditor():
    """Example: List-wise verification (best performing in OAgents)."""
    print("=" * 60)
    print("Example 1: List-Wise Auditor")
    print("=" * 60)
    
    # Create custom verification function
    def verify_result(result, context=None):
        """Custom verification logic."""
        # In practice, this would use LLM or domain-specific logic
        score = 0.8 if "correct" in str(result).lower() else 0.3
        return VerificationResult(
            result=result,
            score=score,
            confidence=0.8,
            reasoning=f"Verification score: {score:.2f}",
            passed=score > 0.5
        )
    
    # Create auditor
    auditor = ListWiseAuditor(
        verification_func=verify_result,
        merge_strategy="best_score",
        min_results=2,
        max_results=5
    )
    
    # Multiple results to verify
    results = [
        "Result 1: correct answer",
        "Result 2: wrong answer",
        "Result 3: correct answer with details",
    ]
    
    # Verify and merge
    merged = auditor.verify_and_merge(results)
    
    print(f"\n✅ Verified {len(results)} results")
    print(f"✅ Selected: {merged.final_result}")
    print(f"✅ Verification score: {merged.verification_score:.2f}")
    print(f"✅ Confidence: {merged.confidence:.2f}")
    print(f"✅ Strategy: {merged.merge_strategy}")
    print(f"✅ Reasoning: {merged.reasoning}")


def example_pair_wise_auditor():
    """Example: Pair-wise verification."""
    print("\n\n" + "=" * 60)
    print("Example 2: Pair-Wise Auditor")
    print("=" * 60)
    
    # Create custom comparison function
    def compare_results(result1, result2, context=None):
        """Compare two results."""
        # In practice, this would use LLM or domain-specific logic
        score1 = 0.8 if "correct" in str(result1).lower() else 0.3
        score2 = 0.8 if "correct" in str(result2).lower() else 0.3
        
        if score1 > score2:
            return result1, score1 - score2
        else:
            return result2, score2 - score1
    
    # Create auditor
    auditor = PairWiseAuditor(comparison_func=compare_results)
    
    # Multiple results to compare
    results = [
        "Result 1: correct answer",
        "Result 2: wrong answer",
        "Result 3: correct answer with details",
    ]
    
    # Verify and select
    merged = auditor.verify_and_select(results)
    
    print(f"\n✅ Compared {len(results)} results")
    print(f"✅ Selected: {merged.final_result}")
    print(f"✅ Verification score: {merged.verification_score:.2f}")
    print(f"✅ Strategy: {merged.merge_strategy}")


def example_confidence_based_auditor():
    """Example: Confidence-based selection."""
    print("\n\n" + "=" * 60)
    print("Example 3: Confidence-Based Auditor")
    print("=" * 60)
    
    # Create custom confidence function
    def get_confidence(result, context=None):
        """Get confidence score."""
        # In practice, this would extract from result or use LLM
        if hasattr(result, 'confidence'):
            return float(result.confidence)
        
        # Simple heuristic
        if "high" in str(result).lower():
            return 0.9
        elif "medium" in str(result).lower():
            return 0.6
        else:
            return 0.4
    
    # Create auditor
    auditor = ConfidenceBasedAuditor(confidence_func=get_confidence)
    
    # Multiple results with different confidence
    results = [
        "Result 1: high confidence answer",
        "Result 2: medium confidence answer",
        "Result 3: low confidence answer",
    ]
    
    # Select best
    merged = auditor.select_best(results)
    
    print(f"\n✅ Evaluated {len(results)} results")
    print(f"✅ Selected: {merged.final_result}")
    print(f"✅ Confidence: {merged.confidence:.2f}")
    print(f"✅ Strategy: {merged.merge_strategy}")


def example_config_integration():
    """Example: Using auditor types with SwarmConfig."""
    print("\n\n" + "=" * 60)
    print("Example 4: Config Integration")
    print("=" * 60)
    
    # Enable list-wise verification
    config = SwarmConfig(
        enable_list_wise_verification=True,
        list_wise_min_results=2,
        list_wise_max_results=5,
        list_wise_merge_strategy="best_score"
    )
    
    print(f"\n✅ Config auditor_type: {config.auditor_type}")
    print(f"✅ List-wise enabled: {config.enable_list_wise_verification}")
    print(f"✅ Min results: {config.list_wise_min_results}")
    print(f"✅ Max results: {config.list_wise_max_results}")
    print(f"✅ Merge strategy: {config.list_wise_merge_strategy}")
    
    # Use with ValidationManager
    from core.orchestration.managers.validation_manager import ValidationManager
    
    manager = ValidationManager(config)
    print(f"\n✅ ValidationManager auditor_type: {manager.auditor_type}")


if __name__ == "__main__":
    print("=" * 60)
    print("Auditor Types Examples")
    print("=" * 60)
    
    example_list_wise_auditor()
    example_pair_wise_auditor()
    example_confidence_based_auditor()
    example_config_integration()
    
    print("\n" + "=" * 60)
    print("Examples Complete")
    print("=" * 60)
    print("\n💡 Key Points:")
    print("  - List-wise verification is best performing (OAgents)")
    print("  - Pair-wise is faster but less effective")
    print("  - Confidence-based is simplest")
    print("  - All can be enabled via SwarmConfig (opt-in)")
