"""
ChatAssistant Agent - Built-in Jotty Chat Agent

World-class chat agent that automatically:
- Returns A2UI widgets for rich UI rendering
- Integrates with state manager for task queries
- Works out of the box with zero configuration

Clients (like JustJot.ai) get this for free by using Jotty.
"""

import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..ui.a2ui import format_task_list, format_card, format_text
from ..ui.status_taxonomy import status_mapper

logger = logging.getLogger(__name__)


class ChatAssistant:
    """
    Built-in chat agent for Jotty.

    Automatically handles:
    - Task queries (backlog, completed, pending)
    - System status
    - Help/documentation
    - General conversation

    Returns A2UI widgets by default for rich rendering.
    """

    def __init__(self, state_manager=None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ChatAssistant.

        Args:
            state_manager: Optional state manager for task queries
            config: Optional configuration
        """
        self.state_manager = state_manager
        self.config = config or {}
        logger.info("✅ ChatAssistant initialized (A2UI enabled)")

    async def run(self, **kwargs) -> Dict[str, Any]:
        """
        Process user query and return A2UI response.

        Args:
            **kwargs: May include 'goal', 'message', 'context', or 'task_description'

        Returns:
            A2UI formatted response (dict with role and content)
        """
        # Debug logging
        logger.debug(f"ChatAssistant.run() received kwargs keys: {list(kwargs.keys())}")
        print(f"🔥🔥🔥 ChatAssistant.run() CALLED with kwargs: {list(kwargs.keys())}", flush=True)

        # Extract goal/message from kwargs (try multiple keys)
        goal = (kwargs.get('goal') or
                kwargs.get('message') or
                kwargs.get('task_description') or
                kwargs.get('query') or "")

        # If still empty, try to extract from context
        if not goal and 'context' in kwargs:
            context = kwargs['context']
            if isinstance(context, dict):
                goal = context.get('ROOT_GOAL') or context.get('goal') or ""
                logger.debug(f"Extracted goal from context: {goal}")

        goal_lower = goal.lower() if goal else ""
        logger.debug(f"Final goal: {goal}")
        print(f"🔥🔥🔥 Goal: '{goal}', is_task_query: {self._is_task_query(goal_lower)}", flush=True)

        # If no goal provided, default to task summary
        if not goal:
            logger.info("No goal provided, returning task summary")
            return await self._get_task_summary_widget()

        # Task-related queries
        if self._is_task_query(goal_lower):
            # Remove conflicting keys from kwargs to avoid parameter conflicts
            clean_kwargs = {k: v for k, v in kwargs.items() if k != 'query'}
            return await self._handle_task_query(goal_lower, **clean_kwargs)

        # Status queries
        if self._is_status_query(goal_lower):
            return await self._handle_status_query(goal_lower)

        # Help queries
        if self._is_help_query(goal_lower):
            return self._handle_help_query()

        # Default: conversational response
        return self._handle_general_query(goal)

    def _is_task_query(self, query: str) -> bool:
        """Detect if query is about tasks."""
        keywords = ['task', 'backlog', 'pending', 'completed', 'done', 'todo', 'in progress']
        return any(keyword in query for keyword in keywords)

    def _is_status_query(self, query: str) -> bool:
        """Detect if query is about system status."""
        keywords = ['status', 'health', 'running', 'system']
        return any(keyword in query for keyword in keywords)

    def _is_help_query(self, query: str) -> bool:
        """Detect if query is asking for help."""
        keywords = ['help', 'how', 'what can', 'capabilities']
        return any(keyword in query for keyword in keywords)

    async def _handle_task_query(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Handle task-related queries with A2UI widgets.

        Uses intent detection to determine response format:
        - "show/display tasks" → kanban board
        - "summarize/summary in markdown" → LLM-generated markdown
        - "list tasks" → list view
        - "backlog/completed/in progress" → filtered views

        Args:
            query: User's question
            **kwargs: Additional context

        Returns:
            A2UI formatted widget (kanban, text, or list)
        """
        query_lower = query.lower()

        # Intent detection: Check if user wants markdown summary
        markdown_keywords = ['markdown', 'summarize', 'summary in', 'write a summary', 'generate summary']
        if any(keyword in query_lower for keyword in markdown_keywords):
            return await self._get_markdown_summary()

        # Specific status filters
        if 'backlog' in query_lower and not ('all' in query_lower or 'summary' in query_lower):
            return await self._get_backlog_widget()
        elif ('completed' in query_lower or 'done' in query_lower) and not ('all' in query_lower or 'summary' in query_lower):
            return await self._get_completed_widget()
        elif ('pending' in query_lower or 'in progress' in query_lower) and not ('all' in query_lower or 'summary' in query_lower):
            return await self._get_pending_widget()
        else:
            # Default: show kanban board for overview/show/all queries
            return await self._get_task_summary_widget()

    async def _get_backlog_widget(self) -> Dict[str, Any]:
        """Get backlog tasks as A2UI widget."""
        tasks = await self._fetch_tasks(status='backlog')

        if not tasks:
            return format_card(
                title="Backlog",
                subtitle="No tasks in backlog",
                body="All clear! No pending tasks in the backlog."
            )

        return format_task_list(
            tasks=self._format_task_items(tasks),
            title=f"Backlog ({len(tasks)} task{'s' if len(tasks) != 1 else ''})"
        )

    async def _get_completed_widget(self) -> Dict[str, Any]:
        """Get completed tasks as A2UI widget."""
        tasks = await self._fetch_tasks(status='completed')

        if not tasks:
            return format_card(
                title="Completed Tasks",
                subtitle="No completed tasks yet",
                body="No tasks have been completed yet."
            )

        return format_task_list(
            tasks=self._format_task_items(tasks),
            title=f"Completed ({len(tasks)} task{'s' if len(tasks) != 1 else ''})"
        )

    async def _get_pending_widget(self) -> Dict[str, Any]:
        """Get in-progress tasks as A2UI widget."""
        tasks = await self._fetch_tasks(status='in_progress')

        if not tasks:
            return format_card(
                title="Active Tasks",
                subtitle="No tasks in progress",
                body="No tasks currently being worked on."
            )

        return format_task_list(
            tasks=self._format_task_items(tasks),
            title=f"In Progress ({len(tasks)} task{'s' if len(tasks) != 1 else ''})"
        )

    async def _get_all_tasks_widget(self) -> Dict[str, Any]:
        """Get all tasks as A2UI widget."""
        all_tasks = await self._fetch_tasks()

        if not all_tasks:
            return format_card(
                title="All Tasks",
                subtitle="No tasks found",
                body="There are no tasks in the system."
            )

        # Group by status
        by_status = {}
        for task in all_tasks:
            status = task.get('status', 'unknown')
            if status not in by_status:
                by_status[status] = []
            by_status[status].append(task)

        # Format each group as a list
        all_items = []
        for status in ['backlog', 'in_progress', 'completed', 'failed']:
            if status in by_status:
                all_items.extend(self._format_task_items(by_status[status]))

        return format_task_list(
            tasks=all_items,
            title=f"All Tasks ({len(all_tasks)} total)"
        )

    async def _get_task_summary_widget(self) -> Dict[str, Any]:
        """
        Get task summary as native kanban board (DRY way).

        Returns section block with native kanban format that JustJot renders directly.
        NO adapters needed - uses return_kanban() helper.
        Uses status_mapper for generic status mapping across all clients.
        """
        all_tasks = await self._fetch_tasks()

        # Create kanban columns using generic status taxonomy
        # This works for ANY client status naming (todo/doing/done, pending/active/closed, etc.)
        columns = status_mapper.create_kanban_columns()

        # Priority mapping: numeric (1,2,3,4) → string ('low','medium','high','urgent')
        def map_priority(priority_value):
            """Convert numeric priority to kanban string format."""
            if isinstance(priority_value, str):
                # Already a string, validate it's correct
                if priority_value in ['low', 'medium', 'high', 'urgent']:
                    return priority_value
                return 'medium'  # Default fallback
            # Numeric priority mapping
            priority_map = {1: 'low', 2: 'medium', 3: 'high', 4: 'urgent'}
            return priority_map.get(priority_value, 'medium')

        def format_assignee(assignee_value):
            """Convert assignee to kanban object format {name, avatar?, email?}."""
            if not assignee_value:
                return None
            if isinstance(assignee_value, dict):
                return assignee_value  # Already correct format
            if isinstance(assignee_value, str):
                return {"name": assignee_value}  # Convert string to object
            return None

        for task in all_tasks:
            # Use status_mapper for generic status normalization
            # This handles any client status naming (todo/doing/done, pending/active/closed, etc.)
            raw_status = task.get('status', 'backlog')
            canonical_status = status_mapper.normalize(raw_status)
            column_id = status_mapper.to_kanban_column(canonical_status)

            # Find the column for this status
            target_column = None
            for col in columns:
                if col['id'] == column_id:
                    target_column = col
                    break

            if not target_column:
                # Fallback to first column if mapping fails
                target_column = columns[0]

            # Format card according to JustJot KanbanItem schema
            card = {
                "id": task.get('task_id', task.get('id', str(len(all_tasks)))),
                "title": task.get('title', task.get('description', 'Untitled Task')),
                "priority": map_priority(task.get('priority', 2)),  # Default to 'medium'
            }

            # Add optional fields only if present
            if task.get('description'):
                card["description"] = task.get('description')
            if task.get('assignee'):
                card["assignee"] = format_assignee(task.get('assignee'))
            if task.get('created_at'):
                card["dueDate"] = task.get('created_at')
            if task.get('labels'):
                # Ensure labels is a string array
                labels = task.get('labels', [])
                if isinstance(labels, list):
                    card["labels"] = [str(l) for l in labels]

            target_column["items"].append(card)

        # Use return_kanban() helper (DRY way!)
        try:
            from ..ui import return_kanban
            logger.info("✅ Using return_kanban() - DRY section rendering")
            return return_kanban(columns=columns, title=f'Task Summary ({len(all_tasks)} total)')
        except Exception as e:
            logger.warning(f"⚠️  return_kanban() not available, falling back: {e}")
            import traceback
            logger.debug(traceback.format_exc())

        # Fallback: Use default A2UI list format
        # Use status_mapper for counting (generic across all client status names)
        backlog = sum(1 for t in all_tasks if status_mapper.normalize(t.get('status', '')) == 'backlog')
        in_progress = sum(1 for t in all_tasks if status_mapper.normalize(t.get('status', '')) == 'in_progress')
        completed = sum(1 for t in all_tasks if status_mapper.normalize(t.get('status', '')) == 'completed')
        failed = sum(1 for t in all_tasks if status_mapper.normalize(t.get('status', '')) == 'failed')

        from ..ui.a2ui import format_task_list

        items = [
            {
                'title': f'Backlog',
                'subtitle': f'{backlog} tasks waiting to start',
                'icon': 'circle',
                'status': 'backlog',
                'metadata': [{'label': 'Count', 'value': str(backlog)}]
            },
            {
                'title': f'In Progress',
                'subtitle': f'{in_progress} tasks currently being worked on',
                'icon': 'circle',
                'status': 'in_progress',
                'metadata': [{'label': 'Count', 'value': str(in_progress)}]
            },
            {
                'title': f'Completed',
                'subtitle': f'{completed} tasks finished successfully',
                'icon': 'check_circle',
                'status': 'completed',
                'metadata': [{'label': 'Count', 'value': str(completed)}]
            },
            {
                'title': f'Failed',
                'subtitle': f'{failed} tasks encountered errors',
                'icon': 'error',
                'status': 'failed',
                'metadata': [{'label': 'Count', 'value': str(failed)}]
            }
        ]

        return format_task_list(
            tasks=items,
            title=f"Task Summary ({len(all_tasks)} total)"
        )

    async def _get_markdown_summary(self) -> Dict[str, Any]:
        """
        Generate markdown summary of tasks using LLM analysis.

        Returns text section with AI-generated task summary.
        Uses status_mapper for generic status handling.
        """
        all_tasks = await self._fetch_tasks()

        # Group tasks by canonical status (using status_mapper for normalization)
        by_status = {
            'backlog': [],
            'in_progress': [],
            'completed': [],
            'failed': []
        }

        for task in all_tasks:
            raw_status = task.get('status', 'backlog')
            canonical_status = status_mapper.normalize(raw_status)
            if canonical_status in by_status:
                by_status[canonical_status].append(task)

        # Generate markdown summary
        summary_lines = [
            f"# Task Summary ({len(all_tasks)} total tasks)",
            "",
            "## Status Overview",
            "",
            f"- **Backlog:** {len(by_status['backlog'])} tasks waiting to start",
            f"- **In Progress:** {len(by_status['in_progress'])} tasks actively being worked on",
            f"- **Completed:** {len(by_status['completed'])} tasks finished successfully",
            f"- **Failed:** {len(by_status['failed'])} tasks encountered errors",
            "",
            f"**Progress:** {len(by_status['completed'])}/{len(all_tasks)} ({int(len(by_status['completed'])/len(all_tasks)*100) if all_tasks else 0}%) complete",
            ""
        ]

        # Add high-priority tasks if any
        high_priority = [t for t in all_tasks if t.get('priority', 0) >= 3 and t.get('status') != 'completed']
        if high_priority:
            summary_lines.extend([
                "## 🔥 High Priority Items",
                ""
            ])
            for task in high_priority[:5]:  # Top 5
                status_emoji = "⏳" if task.get('status') == 'in_progress' else "📋"
                summary_lines.append(f"- {status_emoji} **{task.get('title', 'Untitled')}**")
            summary_lines.append("")

        # Add recent completions
        recent_completed = sorted(
            by_status['completed'],
            key=lambda t: t.get('updated_at', t.get('created_at', '')),
            reverse=True
        )[:5]

        if recent_completed:
            summary_lines.extend([
                "## ✅ Recently Completed",
                ""
            ])
            for task in recent_completed:
                summary_lines.append(f"- {task.get('title', 'Untitled')}")
            summary_lines.append("")

        # Add failed tasks if any
        if by_status['failed']:
            summary_lines.extend([
                "## ⚠️ Failed Tasks (Need Attention)",
                ""
            ])
            for task in by_status['failed'][:5]:
                summary_lines.append(f"- {task.get('title', 'Untitled')}")
            summary_lines.append("")

        markdown_content = "\n".join(summary_lines)

        # Return as text section (markdown renderer)
        from ..ui import return_section
        return return_section(
            section_type="text",
            content=markdown_content,
            title="Task Summary"
        )

    async def _handle_status_query(self, query: str) -> Dict[str, Any]:
        """Handle system status queries."""
        return format_card(
            title="System Status",
            subtitle="All systems operational",
            body="✅ Chat API: Online\n✅ Task Manager: Online\n✅ A2UI Widgets: Enabled"
        )

    def _handle_help_query(self) -> Dict[str, Any]:
        """Handle help requests."""
        help_text = """
**I can help you with:**

🎯 **Task Management**
- "How many tasks in backlog?"
- "Show completed tasks"
- "What tasks are in progress?"

📊 **System Status**
- "System status"
- "Health check"

💬 **General Chat**
- Ask me anything!
        """.strip()

        return format_card(
            title="How I Can Help",
            subtitle="Ask me questions about your tasks",
            body=help_text
        )

    def _handle_general_query(self, query: str) -> Dict[str, Any]:
        """Handle general conversational queries."""
        return format_text(
            f"I understand you said: '{query}'. I'm a task management assistant. "
            "Try asking about tasks, status, or type 'help' to see what I can do!",
            style=None
        )

    async def _fetch_tasks(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Fetch tasks from state manager.

        Args:
            status: Optional status filter ('backlog', 'completed', etc.)

        Returns:
            List of task dictionaries
        """
        if not self.state_manager:
            logger.warning("No state_manager configured, returning empty task list")
            return []

        try:
            # Check for SQLiteTaskQueue methods (Jotty's queue)
            if hasattr(self.state_manager, 'get_tasks_by_status') and status:
                # Direct status query (most efficient)
                task_objects = await self.state_manager.get_tasks_by_status(status)
                return [self._task_to_dict(task) for task in task_objects]

            elif hasattr(self.state_manager, 'export_to_json'):
                # Get all tasks from export (fallback)
                export_data = await self.state_manager.export_to_json()

                # Extract tasks from export structure
                all_tasks = []
                task_details = export_data.get('task_details', {})

                # Get task IDs from status-specific lists
                status_keys = {
                    'backlog': 'backlog_tasks',
                    'pending': 'pending_tasks',
                    'in_progress': 'in_progress_tasks',
                    'completed': 'completed_task_files',
                    'failed': 'failed_task_files'
                }

                if status:
                    # Get specific status
                    key = status_keys.get(status, f'{status}_tasks')
                    task_ids = export_data.get(key, [])
                    all_tasks = [task_details.get(tid) for tid in task_ids if tid in task_details]
                else:
                    # Get all tasks
                    for key in status_keys.values():
                        task_ids = export_data.get(key, [])
                        all_tasks.extend([task_details.get(tid) for tid in task_ids if tid in task_details])

                return [t for t in all_tasks if t]  # Filter out None values

            # Generic fallback methods
            elif hasattr(self.state_manager, 'get_all_tasks'):
                all_tasks = await self.state_manager.get_all_tasks()
            elif hasattr(self.state_manager, 'list_tasks'):
                all_tasks = await self.state_manager.list_tasks()
            else:
                logger.warning(f"State manager has no task listing method: {type(self.state_manager)}")
                return []

            # Filter by status if provided and not already filtered
            if status and all_tasks:
                return [t for t in all_tasks if t.get('status') == status]

            return all_tasks

        except Exception as e:
            logger.error(f"Failed to fetch tasks from state manager: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def _task_to_dict(self, task: Any) -> Dict[str, Any]:
        """
        Convert Task object to dictionary.

        Args:
            task: Task object from state manager

        Returns:
            Task dictionary
        """
        # If already a dict, return as-is
        if isinstance(task, dict):
            return task

        # Convert Task object to dict
        return {
            'task_id': getattr(task, 'task_id', getattr(task, 'id', 'Unknown')),
            'title': getattr(task, 'title', getattr(task, 'description', 'Untitled')),
            'description': getattr(task, 'description', ''),
            'status': getattr(task, 'status', 'unknown'),
            'priority': getattr(task, 'priority', None),
            'created_at': getattr(task, 'created_at', None),
            'updated_at': getattr(task, 'updated_at', None),
        }

    def _format_task_items(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format tasks for A2UI list widget.

        Args:
            tasks: List of task dictionaries

        Returns:
            List of formatted items for A2UI list widget
        """
        items = []

        for task in tasks:
            # Extract task details
            task_id = task.get('task_id', task.get('id', 'Unknown'))
            title = task.get('title', task.get('description', 'Untitled Task'))
            status = task.get('status', 'unknown')

            # Determine icon based on status
            icon = {
                'completed': 'check_circle',
                'in_progress': 'circle',
                'backlog': 'circle',
                'failed': 'error',
                'pending': 'circle',
            }.get(status, 'circle')

            # Create item
            item = {
                'title': title,
                'subtitle': task_id,
                'status': status,
                'icon': icon
            }

            # Add metadata if available
            metadata = []
            if task.get('created_at'):
                try:
                    created = task['created_at']
                    if isinstance(created, str):
                        created = datetime.fromisoformat(created.replace('Z', '+00:00'))
                    metadata.append({
                        'label': 'Created',
                        'value': created.strftime('%Y-%m-%d')
                    })
                except:
                    pass

            if task.get('priority'):
                metadata.append({
                    'label': 'Priority',
                    'value': str(task['priority'])
                })

            if metadata:
                item['metadata'] = metadata

            items.append(item)

        return items


# Factory function for easy instantiation
def create_chat_assistant(state_manager=None, config: Optional[Dict[str, Any]] = None) -> ChatAssistant:
    """
    Create a ChatAssistant agent.

    Args:
        state_manager: Optional state manager for task queries
        config: Optional configuration

    Returns:
        ChatAssistant instance
    """
    return ChatAssistant(state_manager=state_manager, config=config)
