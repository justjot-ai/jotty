"""
Integration test for ParameterResolutionManager (Phase 2.4).

Tests parameter resolution from multiple sources:
- kwargs
- SharedContext
- IOManager
- Defaults
"""
import sys
from pathlib import Path

# Add Jotty to path
jotty_root = Path(__file__).parent
sys.path.insert(0, str(jotty_root))

from core.orchestration.managers.parameter_resolution_manager import ParameterResolutionManager, ResolutionResult
from core.foundation.data_structures import JottyConfig


class MockIOManager:
    """Mock IOManager for testing."""
    def __init__(self):
        self.outputs = {}

    def get_all_outputs(self):
        return self.outputs

    def register_output(self, actor_name, output):
        self.outputs[actor_name] = output


class MockOutput:
    """Mock actor output."""
    def __init__(self, fields):
        self.output_fields = fields


def test_parameter_resolution_from_kwargs():
    """Test parameter resolution from kwargs (highest priority)."""
    print("\n" + "="*70)
    print("TEST 1: Parameter Resolution from kwargs")
    print("="*70)

    config = JottyConfig()
    manager = ParameterResolutionManager(config)

    # Setup
    param_name = "data_source"
    param_info = {"annotation": "str", "required": True}
    kwargs = {"data_source": "api"}
    shared_context = {"data_source": "database"}  # Should NOT be used

    # Resolve
    result = manager.resolve_parameter(param_name, param_info, kwargs, shared_context)

    print(f"✅ Parameter '{param_name}' resolved from kwargs: {result}")
    assert result == "api", f"Expected 'api', got {result}"

    stats = manager.get_stats()
    print(f"📊 Resolution stats: {stats}")
    assert stats['total_resolutions'] == 1

    print("✅ TEST PASSED: kwargs has highest priority")


def test_parameter_resolution_from_shared_context():
    """Test parameter resolution from SharedContext (second priority)."""
    print("\n" + "="*70)
    print("TEST 2: Parameter Resolution from SharedContext")
    print("="*70)

    config = JottyConfig()
    manager = ParameterResolutionManager(config)

    # Setup
    param_name = "goal"
    param_info = {"annotation": "str", "required": True}
    kwargs = {}  # Empty
    shared_context = {"goal": "process data"}

    # Resolve
    result = manager.resolve_parameter(param_name, param_info, kwargs, shared_context)

    print(f"✅ Parameter '{param_name}' resolved from SharedContext: {result}")
    assert result == "process data", f"Expected 'process data', got {result}"

    print("✅ TEST PASSED: SharedContext used when kwargs empty")


def test_parameter_resolution_from_io_manager():
    """Test parameter resolution from IOManager (third priority)."""
    print("\n" + "="*70)
    print("TEST 3: Parameter Resolution from IOManager")
    print("="*70)

    config = JottyConfig()
    manager = ParameterResolutionManager(config)

    # Setup IOManager with actor output
    io_manager = MockIOManager()
    io_manager.register_output("Fetcher", MockOutput({"data": [1, 2, 3]}))

    param_name = "data"
    param_info = {"annotation": "list", "required": True}
    kwargs = {}
    shared_context = {}

    # Resolve
    result = manager.resolve_parameter(param_name, param_info, kwargs, shared_context, io_manager)

    print(f"✅ Parameter '{param_name}' resolved from IOManager: {result}")
    assert result == [1, 2, 3], f"Expected [1, 2, 3], got {result}"

    print("✅ TEST PASSED: IOManager used when kwargs and SharedContext empty")


def test_parameter_resolution_fallback_to_none():
    """Test parameter resolution falls back to None when not found."""
    print("\n" + "="*70)
    print("TEST 4: Parameter Resolution Fallback")
    print("="*70)

    config = JottyConfig()
    manager = ParameterResolutionManager(config)

    # Setup
    param_name = "unknown_param"
    param_info = {"annotation": "str", "required": False}
    kwargs = {}
    shared_context = {}

    # Resolve
    result = manager.resolve_parameter(param_name, param_info, kwargs, shared_context)

    print(f"✅ Parameter '{param_name}' not found, returned: {result}")
    assert result is None, f"Expected None, got {result}"

    print("✅ TEST PASSED: Returns None when parameter not found")


def test_stats_tracking():
    """Test statistics tracking."""
    print("\n" + "="*70)
    print("TEST 5: Statistics Tracking")
    print("="*70)

    config = JottyConfig()
    manager = ParameterResolutionManager(config)

    # Make several resolutions
    manager.resolve_parameter("p1", {}, {"p1": "v1"}, {})
    manager.resolve_parameter("p2", {}, {"p2": "v2"}, {})
    manager.resolve_parameter("p3", {}, {}, {"p3": "v3"})

    stats = manager.get_stats()
    print(f"📊 Stats after 3 resolutions: {stats}")

    assert stats['total_resolutions'] == 3

    print("✅ TEST PASSED: Statistics tracked correctly")


def run_all_tests():
    """Run all parameter resolution tests."""
    print("\n" + "🧪 "*35)
    print("PARAMETER RESOLUTION MANAGER INTEGRATION TESTS (Phase 2.4)")
    print("🧪 "*35)

    try:
        test_parameter_resolution_from_kwargs()
        test_parameter_resolution_from_shared_context()
        test_parameter_resolution_from_io_manager()
        test_parameter_resolution_fallback_to_none()
        test_stats_tracking()

        print("\n" + "✅ "*35)
        print("ALL PARAMETER RESOLUTION MANAGER TESTS PASSED!")
        print("✅ "*35)
        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
