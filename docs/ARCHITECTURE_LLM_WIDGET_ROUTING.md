# LLM-Based Widget Routing Architecture

## Current Problem (Keyword Matching Hack)

The ChatAssistant currently uses brittle keyword matching to route queries to widgets:

```python
if 'backlog' in query:
    return await self._get_backlog_widget()
elif 'completed' in query or 'done' in query:
    return await self._get_completed_widget()
# ... more keyword matching
```

**Problems:**
- Brittle: "show tasks summary" matches "show" → goes to wrong widget
- Not intelligent: Can't handle variations like "give me an overview" or "what's the status"
- Waste of LLM: We have an LLM that can understand intent, but we're using regex

## Proper Solution: LLM Tool Calling

The ChatAssistant should expose widgets as **tools** that the LLM can intelligently choose from.

### Architecture

```python
class ChatAssistant(BaseAgent):
    """
    Supervisor chat agent with LLM-driven widget selection.

    The LLM has access to widget tools and chooses the best visualization
    based on understanding the user's intent.
    """

    @jotty_method(
        name="show_kanban_board",
        description="Show all tasks in a visual kanban board organized by status (backlog, in progress, completed, failed). Best for task overview and visual project status."
    )
    async def show_kanban_board(self) -> Dict[str, Any]:
        """Visual kanban board with all tasks."""
        return await self._get_task_summary_widget()

    @jotty_method(
        name="show_task_list",
        description="Show all tasks in a simple list format. Best when user wants a quick text list or detailed task information."
    )
    async def show_task_list(self) -> Dict[str, Any]:
        """Simple list of all tasks."""
        return await self._get_all_tasks_widget()

    @jotty_method(
        name="show_backlog_tasks",
        description="Show only tasks in backlog status. Use when user specifically asks for backlog or pending tasks."
    )
    async def show_backlog_tasks(self) -> Dict[str, Any]:
        """Filtered view: backlog only."""
        return await self._get_backlog_widget()

    @jotty_method(
        name="show_completed_tasks",
        description="Show only completed tasks. Use when user asks for finished or done tasks."
    )
    async def show_completed_tasks(self) -> Dict[str, Any]:
        """Filtered view: completed only."""
        return await self._get_completed_widget()

    async def run(self, **kwargs) -> Dict[str, Any]:
        """
        Process user query using LLM to choose appropriate widget.

        The LLM will analyze the query and call the most appropriate
        widget tool (kanban board, list, filtered view, etc.)
        """
        goal = kwargs.get('goal') or kwargs.get('message') or ""

        # Let the conductor orchestrate with LLM tool calling
        # The LLM will see available widget tools and choose the best one
        return await self.conductor.run(
            goal=goal,
            context=kwargs.get('context', {}),
            available_tools=[
                'show_kanban_board',
                'show_task_list',
                'show_backlog_tasks',
                'show_completed_tasks',
                'show_in_progress_tasks',
                'get_task_by_id',
                'search_tasks'
            ]
        )
```

### How It Works

1. **User Query**: "show tasks summary"
2. **LLM Analysis**: Understands user wants overview of all tasks
3. **Tool Selection**: LLM calls `show_kanban_board()` tool (best for visual overview)
4. **Widget Return**: Kanban board section block returned
5. **Frontend**: Renders native JustJot.ai kanban board

### Example Queries & LLM Tool Selection

| User Query | LLM Chooses | Reasoning |
|------------|-------------|-----------|
| "show tasks summary" | `show_kanban_board()` | Visual overview requested |
| "what are the tasks" | `show_kanban_board()` | General overview → visual board |
| "list all tasks" | `show_task_list()` | "list" suggests text format |
| "show me backlog" | `show_backlog_tasks()` | Specific filter requested |
| "what's done" | `show_completed_tasks()` | Completed filter |
| "give me task TASK-20260120-00001" | `get_task_by_id()` | Specific task lookup |

### Benefits

✅ **Intelligent**: LLM understands variations and intent
✅ **Robust**: No brittle keyword matching
✅ **Extensible**: Add new widgets as tools, LLM automatically learns them
✅ **Self-documenting**: Tool descriptions teach LLM when to use each widget
✅ **Better UX**: Right widget for user's actual intent

### Migration Path

1. Add `@jotty_method` decorators to widget methods
2. Update `run()` to use conductor with tool calling
3. Remove all keyword matching logic
4. Add system prompt describing available widgets
5. Test with various query phrasings

### System Prompt

```
You are a task management assistant. You have access to several visualization tools:

- show_kanban_board: Visual board with columns for backlog/in progress/completed/failed
- show_task_list: Simple text list of all tasks
- show_backlog_tasks: Filter to show only backlog tasks
- show_completed_tasks: Filter to show only completed tasks
- show_in_progress_tasks: Filter to show only in-progress tasks

Choose the most appropriate visualization based on the user's request:
- For overview/summary/status: Use kanban board (visual)
- For "list" or detailed text: Use task list
- For specific status filters: Use filtered views

Always choose the tool that best matches user intent.
```

## Implementation Priority

**HIGH** - This is a core architectural improvement that makes the system more intelligent and maintainable.

Current status: Quick fix applied (keyword routing tweaked), but full LLM tool calling refactor still needed.
