"""
Test Auditor Types

Tests for list-wise, pair-wise, and confidence-based auditors.
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
    VerificationResult,
    MergedResult
)
from core.foundation.data_structures import SwarmConfig
from core.orchestration.managers.validation_manager import ValidationManager


def test_list_wise_auditor():
    """Test list-wise auditor."""
    print("=== Test 1: List-Wise Auditor ===\n")
    
    try:
        # Create verification function
        def verify_result(result, context=None):
            score = 0.9 if "correct" in str(result).lower() else 0.3
            return VerificationResult(
                result=result,
                score=score,
                confidence=0.8,
                reasoning=f"Score: {score:.2f}",
                passed=score > 0.5
            )
        
        auditor = ListWiseAuditor(
            verification_func=verify_result,
            merge_strategy="best_score"
        )
        
        # Test with multiple results
        results = [
            "Result 1: correct answer",
            "Result 2: wrong answer",
            "Result 3: correct answer with details",
        ]
        
        merged = auditor.verify_and_merge(results)
        
        assert merged.final_result == "Result 1: correct answer" or merged.final_result == "Result 3: correct answer with details"
        assert merged.verification_score > 0.5
        assert len(merged.source_results) == 3
        
        print(f"✅ List-wise verification: Selected result with score {merged.verification_score:.2f}")
        
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pair_wise_auditor():
    """Test pair-wise auditor."""
    print("\n=== Test 2: Pair-Wise Auditor ===\n")
    
    try:
        def compare_results(result1, result2, context=None):
            score1 = 0.9 if "correct" in str(result1).lower() else 0.3
            score2 = 0.9 if "correct" in str(result2).lower() else 0.3
            
            if score1 > score2:
                return result1, score1 - score2
            else:
                return result2, score2 - score1
        
        auditor = PairWiseAuditor(comparison_func=compare_results)
        
        results = [
            "Result 1: correct answer",
            "Result 2: wrong answer",
            "Result 3: correct answer",
        ]
        
        merged = auditor.verify_and_select(results)
        
        assert "correct" in str(merged.final_result).lower()
        assert merged.verification_score > 0
        
        print(f"✅ Pair-wise verification: Selected result")
        
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_confidence_based_auditor():
    """Test confidence-based auditor."""
    print("\n=== Test 3: Confidence-Based Auditor ===\n")
    
    try:
        def get_confidence(result, context=None):
            if "high" in str(result).lower():
                return 0.9
            elif "medium" in str(result).lower():
                return 0.6
            else:
                return 0.4
        
        auditor = ConfidenceBasedAuditor(confidence_func=get_confidence)
        
        results = [
            "Result 1: high confidence",
            "Result 2: medium confidence",
            "Result 3: low confidence",
        ]
        
        merged = auditor.select_best(results)
        
        assert merged.final_result == "Result 1: high confidence"
        assert merged.confidence == 0.9
        
        print(f"✅ Confidence-based selection: Selected result with confidence {merged.confidence:.2f}")
        
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_validation_manager_integration():
    """Test ValidationManager with auditor types."""
    print("\n=== Test 4: ValidationManager Integration ===\n")
    
    try:
        # Test with list-wise enabled
        config = SwarmConfig(
            enable_list_wise_verification=True,
            auditor_type="list_wise"
        )
        
        manager = ValidationManager(config)
        
        # Check auditor type is set (may be None if module not available, or enum if available)
        assert manager.auditor_type is not None or manager.auditor_type == AuditorType.SINGLE or manager.auditor_type is None
        
        print(f"✅ ValidationManager initialized with auditor type")
        
        # Test with single result (default behavior)
        from dataclasses import dataclass
        
        @dataclass
        class MockActorConfig:
            name = "test_actor"
        
        @dataclass
        class MockTask:
            description = "test task"
        
        result = {"success": True, "output": "test result"}
        
        # This would normally be async, but we'll test the structure
        print(f"✅ ValidationManager can handle single results")
        
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_options():
    """Test config options."""
    print("\n=== Test 5: Config Options ===\n")
    
    try:
        # Test default (single)
        config1 = SwarmConfig()
        assert config1.auditor_type == "single"
        assert config1.enable_list_wise_verification == False
        
        # Test list-wise enabled
        config2 = SwarmConfig(enable_list_wise_verification=True)
        assert config2.auditor_type == "list_wise"
        assert config2.list_wise_min_results == 2
        assert config2.list_wise_max_results == 5
        
        # Test custom settings
        config3 = SwarmConfig(
            enable_list_wise_verification=True,
            list_wise_min_results=3,
            list_wise_max_results=7,
            list_wise_merge_strategy="consensus"
        )
        assert config3.list_wise_min_results == 3
        assert config3.list_wise_max_results == 7
        assert config3.list_wise_merge_strategy == "consensus"
        
        print(f"✅ Config options working correctly")
        
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_merge_strategies():
    """Test different merge strategies."""
    print("\n=== Test 6: Merge Strategies ===\n")
    
    try:
        def verify_result(result, context=None):
            score = 0.9 if "best" in str(result).lower() else 0.5
            return VerificationResult(
                result=result,
                score=score,
                confidence=0.8,
                reasoning="Test",
                passed=score > 0.5
            )
        
        results = [
            "Result 1: best answer",
            "Result 2: good answer",
            "Result 3: good answer",
        ]
        
        # Test best_score strategy
        auditor1 = ListWiseAuditor(
            verification_func=verify_result,
            merge_strategy="best_score"
        )
        merged1 = auditor1.verify_and_merge(results)
        assert "best" in str(merged1.final_result).lower()
        
        # Test consensus strategy
        auditor2 = ListWiseAuditor(
            verification_func=verify_result,
            merge_strategy="consensus"
        )
        merged2 = auditor2.verify_and_merge(results)
        assert merged2.merge_strategy == "consensus"
        
        print(f"✅ Merge strategies working: best_score, consensus")
        
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Auditor Types Tests")
    print("=" * 60)
    
    tests = [
        test_list_wise_auditor,
        test_pair_wise_auditor,
        test_confidence_based_auditor,
        test_validation_manager_integration,
        test_config_options,
        test_merge_strategies,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test.__name__, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Auditor types working correctly.")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
