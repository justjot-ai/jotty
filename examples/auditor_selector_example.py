"""
Auditor Selector Example

Demonstrates how to choose the right auditor type for your use case.
"""
import sys
from pathlib import Path

# Add Jotty to path
jotty_path = Path(__file__).parent.parent
sys.path.insert(0, str(jotty_path))

from core.orchestration.auditor_selector import (
    AuditorSelector,
    UseCase,
    Priority,
    recommend_auditor
)
from core.foundation.data_structures import SwarmConfig


def example_code_generation():
    """Example: Code generation (quality critical)."""
    print("=" * 60)
    print("Example 1: Code Generation (Quality Critical)")
    print("=" * 60)
    
    selector = AuditorSelector()
    recommendation = selector.recommend(
        use_case=UseCase.CODE_GENERATION,
        priority=Priority.QUALITY,
        has_multiple_results=True,
        quality_critical=True
    )
    
    print(f"\n✅ Recommended Auditor: {recommendation.auditor_type}")
    print(f"✅ Reason: {recommendation.reason}")
    print(f"\nTrade-offs:")
    for key, value in recommendation.trade_offs.items():
        print(f"  - {key}: {value}")
    print(f"\nConfig:")
    for key, value in recommendation.config.items():
        print(f"  - {key}: {value}")


def example_creative_writing():
    """Example: Creative writing (speed critical)."""
    print("\n\n" + "=" * 60)
    print("Example 2: Creative Writing (Speed Critical)")
    print("=" * 60)
    
    selector = AuditorSelector()
    recommendation = selector.recommend(
        use_case=UseCase.CREATIVE_WRITING,
        priority=Priority.SPEED,
        has_multiple_results=True,
        latency_sensitive=True
    )
    
    print(f"\n✅ Recommended Auditor: {recommendation.auditor_type}")
    print(f"✅ Reason: {recommendation.reason}")
    print(f"\nTrade-offs:")
    for key, value in recommendation.trade_offs.items():
        print(f"  - {key}: {value}")


def example_data_extraction():
    """Example: Data extraction (accuracy critical)."""
    print("\n\n" + "=" * 60)
    print("Example 3: Data Extraction (Accuracy Critical)")
    print("=" * 60)
    
    selector = AuditorSelector()
    recommendation = selector.recommend(
        use_case=UseCase.DATA_EXTRACTION,
        priority=Priority.QUALITY,
        has_multiple_results=True
    )
    
    print(f"\n✅ Recommended Auditor: {recommendation.auditor_type}")
    print(f"✅ Reason: {recommendation.reason}")
    print(f"\nConfig:")
    for key, value in recommendation.config.items():
        print(f"  - {key}: {value}")


def example_convenience_function():
    """Example: Using convenience function."""
    print("\n\n" + "=" * 60)
    print("Example 4: Convenience Function")
    print("=" * 60)
    
    # Simple function call
    rec = recommend_auditor(
        use_case="code_generation",
        priority="quality",
        has_multiple_results=True
    )
    
    print(f"\n✅ Recommended: {rec['auditor_type']}")
    print(f"✅ Reason: {rec['reason']}")
    
    # Apply config
    config = SwarmConfig(**rec['config'])
    print(f"\n✅ Applied config:")
    print(f"   auditor_type: {config.auditor_type}")
    if hasattr(config, 'enable_list_wise_verification'):
        print(f"   enable_list_wise_verification: {config.enable_list_wise_verification}")


def example_comparison_table():
    """Example: Comparison table."""
    print("\n\n" + "=" * 60)
    print("Example 5: Comparison Table")
    print("=" * 60)
    
    selector = AuditorSelector()
    table = selector.get_comparison_table()
    print(table)


def example_single_result():
    """Example: Single result (no multiple results)."""
    print("\n\n" + "=" * 60)
    print("Example 6: Single Result (No Multiple Results)")
    print("=" * 60)
    
    selector = AuditorSelector()
    recommendation = selector.recommend(
        use_case=UseCase.GENERAL,
        priority=Priority.BALANCED,
        has_multiple_results=False  # Only one result
    )
    
    print(f"\n✅ Recommended Auditor: {recommendation.auditor_type}")
    print(f"✅ Reason: {recommendation.reason}")
    print(f"\nNote: When you only have one result, use default single validation.")


if __name__ == "__main__":
    print("=" * 60)
    print("Auditor Selector Examples")
    print("=" * 60)
    print("\nThis demonstrates how to choose the right auditor type.")
    
    example_code_generation()
    example_creative_writing()
    example_data_extraction()
    example_convenience_function()
    example_comparison_table()
    example_single_result()
    
    print("\n\n" + "=" * 60)
    print("Quick Reference")
    print("=" * 60)
    print("""
When to use each auditor type:

1. **Single** (default)
   - Only one result available
   - Fast, low cost
   - Baseline quality

2. **List-wise** (best quality)
   - Quality is critical
   - Code generation, data extraction
   - Multiple results available
   - Higher cost, slower

3. **Pair-wise** (balanced)
   - Balanced approach
   - Medium cost, medium speed
   - Good quality

4. **Confidence-based** (fastest)
   - Speed is critical
   - Creative writing, general tasks
   - Low cost, fast
   - Good quality (if confidence accurate)

Use AuditorSelector.recommend() to get recommendations!
""")
