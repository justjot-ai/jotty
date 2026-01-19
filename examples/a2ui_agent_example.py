"""
Example: Agent that returns A2UI widgets

This example shows how agents can return structured A2UI widgets
instead of plain text responses.
"""

import asyncio
from jotty.core.ui.a2ui import format_task_list, format_card


# Example 1: Task list widget
def get_tasks_as_widget():
    """Return task list as A2UI widget."""
    tasks = [
        {
            "title": "Implement authentication",
            "subtitle": "Priority: High",
            "status": "in_progress",
            "icon": "circle",
            "metadata": [
                {"label": "Assignee", "value": "Alice"},
                {"label": "Due", "value": "2024-01-20"}
            ]
        },
        {
            "title": "Write documentation",
            "subtitle": "Priority: Medium",
            "status": "pending",
            "icon": "circle"
        },
        {
            "title": "Deploy to production",
            "subtitle": "Priority: High",
            "status": "completed",
            "icon": "check_circle"
        }
    ]

    return format_task_list(tasks, title="Current Sprint Tasks")


# Example 2: Status card widget
def get_status_as_widget():
    """Return status as A2UI card widget."""
    return format_card(
        title="Build Status",
        subtitle="Last updated: 2 minutes ago",
        body="All tests passing ✅\n\n5 components built successfully.\n2 warnings (non-blocking)."
    )


# Example 3: Custom agent that returns A2UI
class TaskAssistantAgent:
    """Agent that helps manage tasks using A2UI widgets."""

    async def run(self, goal: str, **kwargs):
        """
        Process goal and return A2UI widget response.

        Args:
            goal: User's request
            **kwargs: Additional context

        Returns:
            A2UI formatted response
        """
        goal_lower = goal.lower()

        if "task" in goal_lower or "todo" in goal_lower:
            return get_tasks_as_widget()
        elif "status" in goal_lower or "build" in goal_lower:
            return get_status_as_widget()
        else:
            # Fallback to text response (will be auto-converted)
            return "I can show you tasks or build status. Just ask!"


# Example 4: Integration with Jotty Conductor
async def main():
    """Example of using A2UI agent with Jotty."""
    from jotty.core.jotty import Jotty

    # Initialize Jotty
    jotty = Jotty()

    # Register agent
    agent = TaskAssistantAgent()
    jotty.conductor.register_actor("TaskAssistant", agent)

    # Run with A2UI response
    result = await jotty.conductor.run(
        goal="Show me current tasks",
        actor_name="TaskAssistant"
    )

    print("Result:", result)
    print("\nType:", type(result))

    # If result is dict with 'role' and 'content', it's A2UI format
    if isinstance(result, dict) and 'content' in result:
        print("\n✅ A2UI widget returned!")
        print("Content blocks:", len(result['content']))
    else:
        print("\n📝 Plain text returned")


if __name__ == "__main__":
    asyncio.run(main())
