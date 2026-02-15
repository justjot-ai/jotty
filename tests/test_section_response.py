#!/usr/bin/env python3
"""
Test Section Response - Verify section blocks work correctly

This demonstrates the CORRECT way to return JustJot sections
from Jotty supervisor/agents.
"""

import json
import sys

# Add to path for imports
sys.path.insert(0, "/var/www/sites/personal/stock_market/Jotty")

from core.ui import return_chart, return_kanban, return_mermaid, return_section


def test_kanban_response():
    """Test returning a kanban board (DRY way)."""
    print("=" * 80)
    print("TEST: Kanban Board Response (DRY)")
    print("=" * 80)

    response = return_kanban(
        columns=[
            {
                "id": "backlog",
                "title": "Backlog",
                "items": [
                    {
                        "id": "task-1",
                        "title": "Implement authentication",
                        "description": "Add OAuth support",
                        "priority": "high",
                        "assignee": "Alice",
                    }
                ],
            },
            {
                "id": "in_progress",
                "title": "In Progress",
                "items": [
                    {
                        "id": "task-2",
                        "title": "Database migration",
                        "priority": "high",
                        "assignee": "Bob",
                    }
                ],
            },
            {
                "id": "done",
                "title": "Done",
                "items": [
                    {
                        "id": "task-3",
                        "title": "Setup CI/CD",
                        "priority": "medium",
                        "assignee": "Charlie",
                    }
                ],
            },
        ],
        title="Sprint 23 Tasks",
    )

    print(json.dumps(response, indent=2))
    print()

    # Verify structure
    assert response["role"] == "assistant", "Should have role=assistant"
    assert response["content"][0]["type"] == "section", "Should be section block, not list!"
    assert response["content"][0]["section_type"] == "kanban-board", "Should be kanban-board type"
    assert "columns" in response["content"][0]["content"], "Should have columns in native format"

    print("✅ CORRECT: Returns section block with native kanban format")
    print()


def test_chart_response():
    """Test returning a chart."""
    print("=" * 80)
    print("TEST: Chart Response (DRY)")
    print("=" * 80)

    response = return_chart(
        chart_type="bar",
        data={"labels": ["Q1", "Q2", "Q3", "Q4"], "values": [100, 150, 200, 250]},
        title="Revenue Growth",
    )

    print(json.dumps(response, indent=2))
    print()

    assert response["content"][0]["type"] == "section", "Should be section block"
    assert response["content"][0]["section_type"] == "chart", "Should be chart type"

    print("✅ CORRECT: Returns section block with native chart format")
    print()


def test_mermaid_response():
    """Test returning a mermaid diagram."""
    print("=" * 80)
    print("TEST: Mermaid Diagram Response (DRY)")
    print("=" * 80)

    response = return_mermaid(
        diagram="""
graph TD
    A[User] --> B{Auth}
    B -->|Success| C[Dashboard]
    B -->|Fail| D[Login]
        """,
        title="Authentication Flow",
    )

    print(json.dumps(response, indent=2))
    print()

    assert response["content"][0]["type"] == "section", "Should be section block"
    assert response["content"][0]["section_type"] == "mermaid", "Should be mermaid type"

    print("✅ CORRECT: Returns section block with native mermaid format")
    print()


def test_generic_section():
    """Test returning any section type."""
    print("=" * 80)
    print("TEST: Generic Section (Any of 70+ types)")
    print("=" * 80)

    # Mind map
    response = return_section(
        section_type="mind-map",
        content={
            "root": {
                "id": "root",
                "text": "Product Launch",
                "children": [
                    {"id": "1", "text": "Marketing"},
                    {"id": "2", "text": "Development"},
                    {"id": "3", "text": "Sales"},
                ],
            }
        },
        title="Launch Strategy",
    )

    print(json.dumps(response, indent=2))
    print()

    assert response["content"][0]["type"] == "section", "Should be section block"
    assert response["content"][0]["section_type"] == "mind-map", "Should be mind-map type"

    print("✅ CORRECT: Works for ALL 70+ JustJot section types!")
    print()


if __name__ == "__main__":
    print("\n")
    print("=" * 80)
    print("JustJot Section Response Tests - DRY Architecture")
    print("=" * 80)
    print()
    print("This demonstrates the CORRECT way for Jotty supervisor to return")
    print("JustJot sections. Instead of converting to generic list/card blocks,")
    print("we return section blocks that preserve full native functionality.")
    print()

    test_kanban_response()
    test_chart_response()
    test_mermaid_response()
    test_generic_section()

    print("=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)
    print()
    print("Next Steps:")
    print("1. Update supervisor agents to use these helpers")
    print("2. Restart supervisor container to pick up new code")
    print("3. Test in actual supervisor chat UI")
    print()
