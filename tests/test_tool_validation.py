"""
Test Tool Validation Framework

Tests tool validation before registration.
"""
import sys
from pathlib import Path

# Add Jotty to path
jotty_path = Path(__file__).parent.parent
sys.path.insert(0, str(jotty_path))

from core.registry.tool_validation import ToolValidator, validate_tool_attributes, RegistryValidationResult


def test_valid_tool():
    """Test validation of valid tool."""
    print("=== Test 1: Valid Tool ===\n")
    
    try:
        def valid_tool(x: int) -> dict:
            """A valid tool."""
            return {"success": True, "result": x * 2}
        
        metadata = {
            "name": "valid_tool",
            "description": "A valid tool",
            "inputs": {
                "x": {"type": "integer", "description": "Input number"}
            },
            "output_type": "dict"
        }
        
        validator = ToolValidator(strict=False)
        result = validator.validate_tool(valid_tool, metadata)
        
        assert result.valid, f"Tool should be valid: {result.errors}"
        print(f"✅ Valid tool passed validation")
        
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_invalid_signature():
    """Test validation catches invalid signature."""
    print("\n=== Test 2: Invalid Signature ===\n")
    
    try:
        def tool_with_wrong_params(y: int) -> dict:  # Wrong parameter name
            return {"success": True}
        
        metadata = {
            "name": "tool",
            "description": "Tool",
            "inputs": {
                "x": {"type": "integer", "description": "Input"}
            },
            "output_type": "dict"
        }
        
        validator = ToolValidator(strict=False)
        result = validator.validate_tool(tool_with_wrong_params, metadata)
        
        # Should have errors (signature mismatch)
        assert not result.valid or len(result.errors) > 0 or len(result.warnings) > 0
        print(f"✅ Invalid signature detected: {result.errors or result.warnings}")
        
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_invalid_type():
    """Test validation catches invalid type."""
    print("\n=== Test 3: Invalid Type ===\n")
    
    try:
        def tool(params: dict) -> dict:
            return {"success": True}
        
        metadata = {
            "name": "tool",
            "description": "Tool",
            "inputs": {
                "x": {"type": "invalid_type", "description": "Input"}  # Invalid type
            },
            "output_type": "dict"
        }
        
        validator = ToolValidator(strict=False)
        result = validator.validate_tool(tool, metadata)
        
        assert not result.valid, "Should fail with invalid type"
        assert len(result.errors) > 0
        print(f"✅ Invalid type detected: {result.errors}")
        
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_missing_metadata():
    """Test validation catches missing metadata."""
    print("\n=== Test 4: Missing Metadata ===\n")
    
    try:
        def tool(params: dict) -> dict:
            return {"success": True}
        
        metadata = {
            "name": "tool",
            # Missing description, inputs, output_type
        }
        
        validator = ToolValidator(strict=False)
        result = validator.validate_tool(tool, metadata)
        
        assert not result.valid, "Should fail with missing metadata"
        assert len(result.errors) > 0
        print(f"✅ Missing metadata detected: {result.errors}")
        
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Tool Validation Tests")
    print("=" * 60)
    
    tests = [
        test_valid_tool,
        test_invalid_signature,
        test_invalid_type,
        test_missing_metadata,
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
        print("\n🎉 All tests passed! Tool validation working correctly.")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
