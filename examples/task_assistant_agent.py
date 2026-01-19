#!/usr/bin/env python3
"""
Task Assistant Agent - Returns A2UI Widgets for Task Queries

This demonstrates how to create an agent that returns rich A2UI widgets
instead of plain text when responding to task-related queries.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict, Any, List
from core.ui.a2ui import format_task_list, format_card


class TaskAssistantAgent:
    """
    Agent that formats task responses as A2UI widgets.

    This should be registered with Jotty Conductor to handle task queries.
    """

    def __init__(self, state_manager=None):
        """
        Initialize agent.

        Args:
            state_manager: Optional state manager to query real tasks
        """
        self.state_manager = state_manager

    async def run(self, goal: str, **kwargs) -> Dict[str, Any]:
        """
        Process user query and return A2UI widget response.

        Args:
            goal: User's question/request
            **kwargs: Additional context

        Returns:
            A2UI formatted response (dict with role and content)
        """
        goal_lower = goal.lower()

        # Detect task-related queries
        if any(keyword in goal_lower for keyword in ['task', 'backlog', 'pending', 'completed']):
            return await self._handle_task_query(goal_lower)

        # Fallback to plain text
        return "I can help you with task queries. Try asking 'how many tasks in backlog?'"

    async def _handle_task_query(self, query: str) -> Dict[str, Any]:
        """Handle task-related queries with A2UI widgets."""

        # Mock data - replace with real state_manager queries
        if 'backlog' in query:
            tasks = await self._get_backlog_tasks()
            return format_task_list(
                tasks,
                title=f"Backlog Tasks ({len(tasks)})"
            )

        elif 'completed' in query or 'done' in query:
            tasks = await self._get_completed_tasks()
            return format_task_list(
                tasks,
                title=f"Completed Tasks ({len(tasks)})"
            )

        elif 'pending' in query or 'in progress' in query:
            tasks = await self._get_pending_tasks()
            return format_task_list(
                tasks,
                title=f"Active Tasks ({len(tasks)})"
            )

        else:
            # General task summary
            return format_card(
                title="Task Summary",
                subtitle="Current status",
                body="1 task in backlog\n0 tasks in progress\n5 tasks completed"
            )

    async def _get_backlog_tasks(self) -> List[Dict[str, Any]]:
        """Get backlog tasks - replace with real DB query."""
        # Mock data - in real implementation, query state_manager
        return [
            {
                "title": "Test minimal supervisor with real agent",
                "subtitle": "TASK-20260118-00001",
                "status": "backlog",
                "icon": "circle",
                "metadata": [
                    {"label": "Created", "value": "2026-01-18"},
                    {"label": "Priority", "value": "Medium"}
                ]
            }
        ]

    async def _get_completed_tasks(self) -> List[Dict[str, Any]]:
        """Get completed tasks - replace with real DB query."""
        return [
            {
                "title": "Enhancements in Supervisor Dashboard",
                "subtitle": "TASK-20260117-00001",
                "status": "completed",
                "icon": "check_circle",
            },
            {
                "title": "Enhancement of Browse Page",
                "subtitle": "TASK-20260116-00002",
                "status": "completed",
                "icon": "check_circle",
            }
        ]

    async def _get_pending_tasks(self) -> List[Dict[str, Any]]:
        """Get in-progress tasks - replace with real DB query."""
        return []


# Integration with Jotty Conductor
async def register_with_conductor(conductor):
    """
    Register TaskAssistantAgent with Jotty Conductor.

    Usage:
        from jotty.core.jotty import Jotty
        jotty = Jotty()
        await register_with_conductor(jotty.conductor)
    """
    agent = TaskAssistantAgent()
    conductor.register_actor("ChatAssistant", agent)
    print("✅ TaskAssistantAgent registered as ChatAssistant")


# Example: Testing the agent directly
async def main():
    """Test the agent with sample queries."""
    agent = TaskAssistantAgent()

    queries = [
        "How many tasks in backlog?",
        "Show me completed tasks",
        "What tasks are pending?"
    ]

    for query in queries:
        print(f"\n🔍 Query: {query}")
        result = await agent.run(query)

        import json
        print(json.dumps(result, indent=2))
        print("-" * 60)


if __name__ == "__main__":
    asyncio.run(main())
