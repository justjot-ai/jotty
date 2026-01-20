"""
JustJot Section Widgets Demo - DRY A2UI Integration

This demonstrates the DRY way to use JustJot's 70+ section renderers
in Jotty supervisor chat via A2UI section blocks.

KEY CONCEPT: Instead of converting sections to generic A2UI blocks (list, card),
we pass native content directly using format_section() which preserves full
functionality of each section type.
"""

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ui import format_section, A2UIBuilder


# ============================================================================
# Example 1: Kanban Board
# ============================================================================
def create_kanban_response():
    """Create kanban board with native column format (DRY way)."""
    return format_section(
        section_type="kanban-board",
        content={
            "columns": [
                {
                    "id": "backlog",
                    "title": "Backlog",
                    "items": [
                        {
                            "id": "task-1",
                            "title": "Implement user authentication",
                            "description": "Add OAuth support",
                            "priority": "high",
                            "assignee": "Alice",
                            "labels": ["security", "backend"]
                        },
                        {
                            "id": "task-2",
                            "title": "Design dashboard UI",
                            "description": "Create wireframes",
                            "priority": "medium",
                            "assignee": "Bob"
                        }
                    ]
                },
                {
                    "id": "in_progress",
                    "title": "In Progress",
                    "items": [
                        {
                            "id": "task-3",
                            "title": "Database migration",
                            "description": "Migrate to PostgreSQL",
                            "priority": "high",
                            "assignee": "Charlie"
                        }
                    ]
                },
                {
                    "id": "completed",
                    "title": "Completed",
                    "items": [
                        {
                            "id": "task-4",
                            "title": "Setup CI/CD pipeline",
                            "priority": "medium",
                            "assignee": "Alice"
                        }
                    ]
                }
            ]
        },
        title="Sprint 23 Tasks"
    )


# ============================================================================
# Example 2: Mermaid Diagram
# ============================================================================
def create_mermaid_response():
    """Create mermaid diagram."""
    return format_section(
        section_type="mermaid",
        content="""
graph TD
    A[User Request] --> B{Auth Check}
    B -->|Valid| C[Load Dashboard]
    B -->|Invalid| D[Show Login]
    C --> E[Render Charts]
    D --> F[OAuth Flow]
    F --> C
        """,
        title="System Architecture"
    )


# ============================================================================
# Example 3: Chart
# ============================================================================
def create_chart_response():
    """Create interactive chart."""
    return format_section(
        section_type="chart",
        content={
            "type": "bar",
            "title": "Q4 Revenue",
            "data": {
                "labels": ["October", "November", "December"],
                "datasets": [
                    {
                        "label": "Revenue",
                        "values": [45000, 52000, 61000],
                        "color": "#3B82F6"
                    },
                    {
                        "label": "Expenses",
                        "values": [32000, 35000, 38000],
                        "color": "#EF4444"
                    }
                ]
            }
        },
        title="Quarterly Performance"
    )


# ============================================================================
# Example 4: Data Table
# ============================================================================
def create_data_table_response():
    """Create data table with CSV content."""
    return format_section(
        section_type="data-table",
        content="""Name,Role,Department,Salary
Alice Johnson,Senior Developer,Engineering,120000
Bob Smith,Product Manager,Product,115000
Charlie Brown,Designer,Design,95000
Diana Prince,QA Engineer,Engineering,90000""",
        title="Team Directory"
    )


# ============================================================================
# Example 5: Mind Map
# ============================================================================
def create_mind_map_response():
    """Create mind map."""
    return format_section(
        section_type="mind-map",
        content={
            "root": {
                "id": "root",
                "text": "Product Launch Strategy",
                "children": [
                    {
                        "id": "marketing",
                        "text": "Marketing",
                        "children": [
                            {"id": "social", "text": "Social Media Campaign"},
                            {"id": "pr", "text": "Press Release"},
                            {"id": "influencer", "text": "Influencer Partnerships"}
                        ]
                    },
                    {
                        "id": "development",
                        "text": "Development",
                        "children": [
                            {"id": "features", "text": "Final Features"},
                            {"id": "testing", "text": "QA Testing"},
                            {"id": "deploy", "text": "Deployment"}
                        ]
                    },
                    {
                        "id": "sales",
                        "text": "Sales",
                        "children": [
                            {"id": "pricing", "text": "Pricing Strategy"},
                            {"id": "training", "text": "Sales Training"},
                            {"id": "outreach", "text": "Customer Outreach"}
                        ]
                    }
                ]
            }
        },
        title="Launch Plan"
    )


# ============================================================================
# Example 6: SWOT Analysis
# ============================================================================
def create_swot_response():
    """Create SWOT analysis."""
    return format_section(
        section_type="swot",
        content={
            "strengths": [
                "Strong engineering team",
                "Proven technology stack",
                "Low customer churn"
            ],
            "weaknesses": [
                "Limited marketing budget",
                "Small sales team",
                "No mobile app"
            ],
            "opportunities": [
                "Growing market demand",
                "Competitor acquisition",
                "International expansion"
            ],
            "threats": [
                "New competitors entering market",
                "Economic downturn",
                "Regulatory changes"
            ]
        },
        title="Q1 Strategic Analysis"
    )


# ============================================================================
# Example 7: Multiple Sections (Complex Response)
# ============================================================================
def create_complex_response():
    """Create response with multiple sections using builder."""
    builder = A2UIBuilder()

    # Add text intro
    builder.add_text("Here's your project status dashboard:", style="bold")

    # Add kanban board
    builder.add_section(
        section_type="kanban-board",
        content={
            "columns": [
                {"id": "todo", "title": "To Do", "items": [{"id": "1", "title": "Task 1"}]},
                {"id": "done", "title": "Done", "items": [{"id": "2", "title": "Task 2"}]}
            ]
        },
        title="Sprint Tasks"
    )

    # Add separator
    builder.add_separator()

    # Add chart
    builder.add_section(
        section_type="chart",
        content={
            "type": "line",
            "data": {
                "labels": ["Week 1", "Week 2", "Week 3"],
                "values": [10, 25, 40]
            }
        },
        title="Velocity Trend"
    )

    return builder.build()


# ============================================================================
# Example 8: All 70+ Section Types (Quick Reference)
# ============================================================================
def get_all_section_types_examples():
    """Quick reference for all JustJot section types."""
    return {
        # Content
        "text": format_section("text", "# Markdown content"),
        "code": format_section("code", '{"language": "python", "code": "print(\\"hello\\")"}'),
        "latex": format_section("latex", "E = mc^2"),
        "json": format_section("json", '{"key": "value"}'),
        "html": format_section("html", "<h1>Hello</h1>"),

        # Diagrams
        "mermaid": format_section("mermaid", "graph TD; A-->B;"),
        "excalidraw": format_section("excalidraw", "{}"),  # Excalidraw JSON
        "plantuml": format_section("plantuml", "@startuml\\nBob -> Alice\\n@enduml"),
        "gantt": format_section("gantt", '{"tasks": []}'),

        # Data & Charts
        "chart": format_section("chart", '{"type": "bar", "data": {}}'),
        "data-table": format_section("data-table", "Col1,Col2\\nVal1,Val2"),
        "csv": format_section("csv", "Name,Age\\nAlice,30"),

        # Project Management
        "kanban-board": format_section("kanban-board", '{"columns": []}'),
        "sprint-planning": format_section("sprint-planning", '{"sprints": []}'),
        "gantt": format_section("gantt", '{"tasks": []}'),
        "roadmap": format_section("roadmap", '{"phases": []}'),

        # Business
        "swot": format_section("swot", '{"strengths": [], "weaknesses": []}'),
        "business-model-canvas": format_section("business-model-canvas", '{}'),
        "okrs": format_section("okrs", '{"objectives": []}'),

        # Productivity
        "todos": format_section("todos", "- [ ] Task 1\\n- [x] Task 2"),
        "mind-map": format_section("mind-map", '{"root": {"text": "Root"}}'),
        "daily-journal": format_section("daily-journal", '{"date": "2024-01-01"}'),

        # ...and 50+ more!
    }


# ============================================================================
# Usage in Jotty Agent
# ============================================================================
def example_agent_handler(user_message: str) -> dict:
    """Example of how to use format_section in a Jotty agent."""
    if "kanban" in user_message.lower():
        return create_kanban_response()
    elif "chart" in user_message.lower():
        return create_chart_response()
    elif "diagram" in user_message.lower():
        return create_mermaid_response()
    else:
        return format_section(
            section_type="text",
            content=f"Echo: {user_message}"
        )


if __name__ == "__main__":
    import json

    print("=" * 80)
    print("JustJot Section Widgets Demo")
    print("=" * 80)

    # Test kanban
    print("\n1. Kanban Board:")
    print(json.dumps(create_kanban_response(), indent=2))

    # Test mermaid
    print("\n2. Mermaid Diagram:")
    print(json.dumps(create_mermaid_response(), indent=2))

    # Test chart
    print("\n3. Chart:")
    print(json.dumps(create_chart_response(), indent=2))

    # Test complex
    print("\n4. Complex Response (Multiple Sections):")
    print(json.dumps(create_complex_response(), indent=2))

    print("\n" + "=" * 80)
    print("✅ All examples generated successfully!")
    print("=" * 80)
