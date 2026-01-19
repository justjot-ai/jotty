#!/usr/bin/env python3
"""
Test A2UI Integration

Quick test to verify A2UI widgets work end-to-end.
"""

import sys
import json
from core.ui.a2ui import (
    format_task_list,
    format_card,
    is_a2ui_response,
    convert_to_a2ui_response
)

def test_task_list():
    """Test task list formatting."""
    print("🧪 Testing task list formatting...")

    tasks = [
        {
            "title": "Implement authentication",
            "subtitle": "Priority: High",
            "status": "in_progress",
            "icon": "circle"
        },
        {
            "title": "Write tests",
            "subtitle": "Priority: Medium",
            "status": "completed",
            "icon": "check_circle"
        }
    ]

    result = format_task_list(tasks, title="Sprint Tasks")

    assert result["role"] == "assistant", "Role should be assistant"
    assert isinstance(result["content"], list), "Content should be a list"
    assert len(result["content"]) > 0, "Content should not be empty"

    print("✅ Task list test passed!")
    print(json.dumps(result, indent=2))
    return result

def test_card():
    """Test card formatting."""
    print("\n🧪 Testing card formatting...")

    result = format_card(
        title="Build Status",
        subtitle="Last updated: now",
        body="All tests passing ✅"
    )

    assert is_a2ui_response(result), "Result should be A2UI format"

    print("✅ Card test passed!")
    print(json.dumps(result, indent=2))
    return result

def test_detection():
    """Test A2UI detection."""
    print("\n🧪 Testing A2UI detection...")

    # Test A2UI response
    a2ui_resp = format_card("Test", "Testing")
    assert is_a2ui_response(a2ui_resp), "Should detect A2UI response"

    # Test plain text
    plain_text = "Hello world"
    assert not is_a2ui_response(plain_text), "Should not detect plain text as A2UI"

    # Test conversion
    converted = convert_to_a2ui_response(plain_text)
    assert is_a2ui_response(converted), "Converted text should be A2UI"

    print("✅ Detection test passed!")

def main():
    """Run all tests."""
    print("=" * 60)
    print("A2UI Integration Test")
    print("=" * 60)

    try:
        test_task_list()
        test_card()
        test_detection()

        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
        return 0
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
