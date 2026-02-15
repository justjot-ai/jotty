"""
ChatAssistant V2 - LLM-Driven Section Generation (DRY)

This version uses LLM tool calling to choose section types dynamically.
No hardcoded logic per section type - truly generic and scalable!

Key differences from V1:
- V1: if 'markdown' in query → _get_markdown_summary()  (70+ methods)
- V2: LLM chooses tool → return_text() or return_kanban()  (1 method)
"""

import json
import logging
from typing import Any, Dict, Optional

from anthropic import Anthropic

from Jotty.core.infrastructure.foundation.anthropic_client_kwargs import get_anthropic_client_kwargs
from Jotty.core.infrastructure.foundation.config_defaults import LLM_MAX_OUTPUT_TOKENS

from ..ui import return_chart, return_data_table, return_kanban, return_mermaid, return_section
from ..ui.status_taxonomy import status_mapper
from .section_tools import generate_section_tools, get_system_prompt

logger = logging.getLogger(__name__)


class ChatAssistantV2:
    """
    LLM-driven chat assistant with dynamic section generation.

    Instead of hardcoding logic for each section type, the LLM
    chooses the appropriate section format based on the query.

    Scales to 70+ section types with ZERO additional code!
    """

    def __init__(self, state_manager: Any = None, anthropic_api_key: Optional[str] = None) -> None:
        """
        Initialize ChatAssistant V2.

        Args:
            state_manager: State manager for task queries
            anthropic_api_key: Anthropic API key for Claude
        """
        self.state_manager = state_manager
        client_kwargs = get_anthropic_client_kwargs(api_key=anthropic_api_key)
        self.llm = Anthropic(**client_kwargs) if client_kwargs.get("api_key") else None

        # Auto-generate tool definitions from schemas
        self.tools = generate_section_tools()

        logger.info(f" ChatAssistant V2 initialized with {len(self.tools)} section tools")

    async def run(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Process user query using LLM tool calling.

        Args:
            **kwargs: May include 'goal', 'message', 'context'

        Returns:
            A2UI formatted response (LLM chose the section type!)
        """
        # Extract query
        goal = (
            kwargs.get("goal")
            or kwargs.get("message")
            or kwargs.get("context")
            or kwargs.get("task_description", "")
        )

        if not goal:
            return {"role": "assistant", "content": "No query provided"}

        logger.info(f"Processing query: {goal}")

        # Detect intent and fetch relevant data
        context = await self._fetch_context(goal)

        # LLM generates response with tool calling
        return await self._llm_generate(query=goal, context=context)

    async def _fetch_context(self, query: str) -> Dict[str, Any]:
        """
        Fetch relevant data based on query intent.

        Returns context dictionary that LLM can use.
        """
        context = {}

        # Fetch task data if query is about tasks
        if self._is_task_query(query) and self.state_manager:
            tasks = await self._fetch_tasks()
            context["tasks"] = tasks
            context["task_count"] = len(tasks)

            # Group by status (using generic taxonomy)
            by_status = {}
            for task in tasks:
                canonical_status = status_mapper.normalize(task.get("status", "backlog"))
                if canonical_status not in by_status:
                    by_status[canonical_status] = []
                by_status[canonical_status].append(task)

            context["tasks_by_status"] = by_status
            context["status_counts"] = {k: len(v) for k, v in by_status.items()}

        return context

    def _is_task_query(self, query: str) -> bool:
        """Detect if query is about tasks."""
        keywords = [
            "task",
            "backlog",
            "pending",
            "completed",
            "done",
            "todo",
            "in progress",
            "show",
            "list",
        ]
        return any(keyword in query.lower() for keyword in keywords)

    async def _fetch_tasks(self) -> list:
        """Fetch all tasks from state manager."""
        if not self.state_manager:
            return []

        try:
            return await self.state_manager.get_all_tasks()
        except Exception as e:
            logger.error(f"Failed to fetch tasks: {e}")
            return []

    async def _llm_generate(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use LLM with tool calling to generate response.

        LLM chooses which section type to return based on query.

        Args:
            query: User's question
            context: Relevant data (tasks, status, etc.)

        Returns:
            A2UI response (LLM chose the format!)
        """
        if not self.llm:
            # Fallback: return simple kanban if no LLM
            return self._fallback_response(context)

        # Build messages
        messages = [
            {
                "role": "user",
                "content": f"""
Query: {query}

Available data:
{json.dumps(context, indent=2)}

Choose the BEST section format for this query and generate appropriate content.
""",
            }
        ]

        try:
            # Claude API call with tool calling
            response = self.llm.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=LLM_MAX_OUTPUT_TOKENS,
                system=get_system_prompt(),  # Includes tool hints
                messages=messages,
                tools=self.tools,  # Auto-generated from schemas
                tool_choice={"type": "any"},  # LLM must use a tool
            )

            # Execute the tool LLM chose
            if response.stop_reason == "tool_use":
                tool_use = response.content[-1]  # Last content block is tool use
                return self._execute_tool(tool_use, context)

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")

        # Fallback on error
        return self._fallback_response(context)

    def _execute_tool(self, tool_use: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the tool LLM chose.

        Maps tool calls to section helper functions.

        Args:
            tool_use: Tool use block from Claude API
            context: Query context (for fallback data)

        Returns:
            A2UI formatted response
        """
        tool_name = tool_use.name
        tool_input = tool_use.input

        logger.info(f"LLM chose tool: {tool_name}")

        # Map tool name to helper function
        tool_map = {
            "return_kanban": lambda: return_kanban(
                columns=tool_input.get("content", {}).get("columns", []),
                title=tool_input.get("title"),
            ),
            "return_text": lambda: return_section(
                section_type="text",
                content=tool_input.get("content", ""),
                title=tool_input.get("title"),
            ),
            "return_chart": lambda: return_chart(
                chart_type=tool_input.get("content", {}).get("type", "bar"),
                data=tool_input.get("content", {}).get("data", {}),
                title=tool_input.get("title"),
            ),
            "return_mermaid": lambda: return_mermaid(
                diagram=tool_input.get("content", ""), title=tool_input.get("title")
            ),
            "return_data_table": lambda: return_data_table(
                csv_data=tool_input.get("content", ""), title=tool_input.get("title")
            ),
            # ... other tools auto-mapped
        }

        # Execute tool or use generic fallback
        if tool_name in tool_map:
            return tool_map[tool_name]()
        else:
            # Generic section helper (works for ANY type!)
            section_type = tool_name.replace("return_", "").replace("_", "-")
            return return_section(
                section_type=section_type,
                content=tool_input.get("content"),
                title=tool_input.get("title"),
            )

    def _fallback_response(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback response when LLM unavailable.

        Returns kanban board if tasks available.
        """
        tasks = context.get("tasks", [])
        if not tasks:
            return {"role": "assistant", "content": "No tasks found"}

        # Build kanban columns using status taxonomy
        columns = status_mapper.create_kanban_columns()

        for task in tasks:
            canonical_status = status_mapper.normalize(task.get("status", "backlog"))
            column_id = status_mapper.to_kanban_column(canonical_status)

            # Find column and add task
            for col in columns:
                if col["id"] == column_id:
                    col["items"].append(
                        {
                            "id": task.get("task_id", task.get("id")),
                            "title": task.get("title", "Untitled"),
                            "description": task.get("description"),
                            "priority": task.get("priority"),
                        }
                    )

        return return_kanban(columns=columns, title=f"Tasks ({len(tasks)} total)")


# Example usage
"""
# Initialize with LLM
import os
assistant = ChatAssistantV2(
    state_manager=state_manager,
    anthropic_api_key=os.getenv('ANTHROPIC_API_KEY')
)

# User query: "show tasks"
# LLM thinks: Visual format best → chooses return_kanban()
response = await assistant.run(goal="show tasks")
# Returns: kanban board

# User query: "summarize tasks in markdown"
# LLM thinks: Markdown requested → chooses return_text()
response = await assistant.run(goal="summarize tasks in markdown")
# Returns: markdown summary

# User query: "chart task completion over time"
# LLM thinks: Time-series data → chooses return_chart()
response = await assistant.run(goal="chart task completion over time")
# Returns: line chart

NO HARDCODED LOGIC - LLM CHOOSES EVERYTHING!
"""
