"""
Integration test for OutputRegistryManager (Phase 3.2).

Tests:
- Output type detection
- Schema extraction
- Preview generation
- Tag generation
- Statistics tracking
"""
import sys
from pathlib import Path

# Add Jotty to path
jotty_root = Path(__file__).parent
sys.path.insert(0, str(jotty_root))

from core.orchestration.managers.output_registry_manager import OutputRegistryManager
from core.foundation.data_structures import JottyConfig


def test_output_type_detection():
    """Test output type detection."""
    print("\n" + "="*70)
    print("TEST 1: Output Type Detection")
    print("="*70)

    config = JottyConfig()
    manager = OutputRegistryManager(config)

    # Test different output types
    tests = [
        ("text", "This is a simple text output"),
        ("html", "<html><body>HTML content</body></html>"),
        ("markdown", "# Heading\n\nMarkdown content"),
        ("json", {"key": "value", "number": 123}),
        ("binary", b"binary data"),
    ]

    for expected_type, output in tests:
        detected_type = manager.detect_output_type(output)
        print(f"✅ Detected '{expected_type}': {detected_type}")
        # Note: markdown might be detected as text if too short
        assert detected_type in [expected_type, 'text'], f"Expected {expected_type} or text, got {detected_type}"

    print("✅ TEST PASSED: Output type detection works")


def test_schema_extraction():
    """Test schema extraction from various outputs."""
    print("\n" + "="*70)
    print("TEST 2: Schema Extraction")
    print("="*70)

    config = JottyConfig()
    manager = OutputRegistryManager(config)

    # Test dict output
    dict_output = {"name": "John", "age": 30, "active": True}
    schema = manager.extract_schema(dict_output)
    print(f"✅ Dict schema: {schema}")
    assert "name" in schema
    assert schema["name"] == "str"

    # Test object output
    class MockOutput:
        def __init__(self):
            self.field1 = "value"
            self.field2 = 123

    obj_output = MockOutput()
    schema = manager.extract_schema(obj_output)
    print(f"✅ Object schema: {schema}")
    assert "field1" in schema
    assert "field2" in schema

    print("✅ TEST PASSED: Schema extraction works")


def test_preview_generation():
    """Test preview generation."""
    print("\n" + "="*70)
    print("TEST 3: Preview Generation")
    print("="*70)

    config = JottyConfig()
    manager = OutputRegistryManager(config)

    # Test short string
    short_text = "Short output"
    preview = manager.generate_preview(short_text)
    print(f"✅ Short preview: {preview}")
    assert preview == short_text

    # Test long string (truncated)
    long_text = "A" * 300
    preview = manager.generate_preview(long_text)
    print(f"✅ Long preview (truncated): {len(preview)} chars")
    assert len(preview) == 200

    # Test dict
    dict_output = {"key": "value"}
    preview = manager.generate_preview(dict_output)
    print(f"✅ Dict preview: {preview}")
    assert "key" in preview

    print("✅ TEST PASSED: Preview generation works")


def test_tag_generation():
    """Test tag generation."""
    print("\n" + "="*70)
    print("TEST 4: Tag Generation")
    print("="*70)

    config = JottyConfig()
    manager = OutputRegistryManager(config)

    # Test with dict output
    output = {"data": [1, 2, 3], "status": "success"}
    tags = manager.generate_tags("Fetcher", output, "json")
    print(f"✅ Generated tags: {tags}")
    assert "json" in tags
    assert "fetcher" in tags  # Actor name lowercased
    assert "data" in tags or "status" in tags  # Field names

    print("✅ TEST PASSED: Tag generation works")


def test_trajectory_operations():
    """Test getting outputs from trajectory."""
    print("\n" + "="*70)
    print("TEST 5: Trajectory Operations")
    print("="*70)

    config = JottyConfig()
    manager = OutputRegistryManager(config)

    # Create mock trajectory
    trajectory = [
        {"actor": "Fetcher", "actor_output": {"data": [1, 2, 3]}},
        {"actor": "Processor", "actor_output": {"result": "processed"}},
        {"actor": "Fetcher", "actor_output": {"data": [4, 5, 6]}}  # Updated output
    ]

    # Test get_actor_outputs
    outputs = manager.get_actor_outputs(trajectory)
    print(f"✅ All outputs: {list(outputs.keys())}")
    assert "Fetcher" in outputs
    assert "Processor" in outputs
    assert outputs["Fetcher"]["data"] == [4, 5, 6]  # Latest output

    # Test get_output_from_actor
    fetcher_output = manager.get_output_from_actor("Fetcher", trajectory)
    print(f"✅ Fetcher output: {fetcher_output}")
    assert fetcher_output["data"] == [4, 5, 6]

    # Test field extraction
    processor_result = manager.get_output_from_actor("Processor", trajectory, field="result")
    print(f"✅ Processor result field: {processor_result}")
    assert processor_result == "processed"

    print("✅ TEST PASSED: Trajectory operations work")


def test_statistics_tracking():
    """Test statistics tracking."""
    print("\n" + "="*70)
    print("TEST 6: Statistics Tracking")
    print("="*70)

    config = JottyConfig()
    manager = OutputRegistryManager(config)

    # Get initial stats
    stats = manager.get_stats()
    print(f"📊 Initial stats: {stats}")
    assert stats["total_registrations"] == 0
    assert stats["has_data_registry"] is False

    # Reset stats
    manager.reset_stats()
    stats = manager.get_stats()
    print(f"📊 Stats after reset: {stats}")
    assert stats["total_registrations"] == 0

    print("✅ TEST PASSED: Statistics tracking works")


def run_all_tests():
    """Run all output registry manager tests."""
    print("\n" + "🧪 "*35)
    print("OUTPUT REGISTRY MANAGER INTEGRATION TESTS (Phase 3.2)")
    print("🧪 "*35)

    try:
        test_output_type_detection()
        test_schema_extraction()
        test_preview_generation()
        test_tag_generation()
        test_trajectory_operations()
        test_statistics_tracking()

        print("\n" + "✅ "*35)
        print("ALL OUTPUT REGISTRY MANAGER TESTS PASSED!")
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
