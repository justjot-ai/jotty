# ChatAssistant - Built-in Jotty Chat Agent

**World-class SaaS SDK design: Zero configuration required!**

Jotty provides a built-in `ChatAssistant` agent that automatically:
- Returns A2UI widgets for rich UI rendering
- Integrates with your state manager for task queries
- Handles conversational queries
- Works out of the box with zero setup

## For Clients (Like JustJot.ai)

### Quick Start - 3 Lines of Code!

```python
from jotty.core.api import ChatAPI
from jotty.core.orchestration import Conductor

# Create conductor (your existing setup)
conductor = Conductor(...)

# Create Chat API with agent_id="ChatAssistant"
chat_api = ChatAPI(
    conductor=conductor,
    agent_id="ChatAssistant"  # 🎯 That's it! ChatAssistant auto-registers
)

# Use it immediately!
async for event in chat_api.stream(message="How many tasks in backlog?"):
    print(event)  # A2UI widgets automatically rendered!
```

**That's literally it!** No agent registration, no configuration, no setup.

### With State Manager (For Task Queries)

If you want the ChatAssistant to query your tasks:

```python
chat_api = ChatAPI(
    conductor=conductor,
    agent_id="ChatAssistant",
    state_manager=your_state_manager  # Optional: for task queries
)
```

The ChatAssistant will automatically:
- Query backlog, completed, and in-progress tasks
- Return them as beautiful A2UI widgets (cards, lists, status badges)
- Handle "how many tasks" type queries

### What You Get for Free

**Zero Code A2UI Widgets:**
```
User: "How many tasks in backlog?"

Response: [Rich Card Widget]
┌────────────────────────────────┐
│ 📋 Backlog (1 task)           │
├────────────────────────────────┤
│ ⭕ Test minimal supervisor    │
│    TASK-20260118-00001        │
│    Created: 2026-01-18         │
│    Priority: Medium            │
└────────────────────────────────┘
```

**Supported Queries:**
- "How many tasks in backlog?"
- "Show me completed tasks"
- "What's in progress?"
- "Show all tasks"
- "System status"
- "Help"

### HTTP Server Integration

If using JottyHTTPServer:

```python
from jotty.core.server import JottyHTTPServer

server = JottyHTTPServer(
    conductor=conductor,
    chat_agent_id="ChatAssistant",  # Auto-registers!
    state_manager=your_state_manager  # Optional
)

server.run(port=8080)
```

Done! Your `/api/chat/stream` endpoint now returns A2UI widgets.

## For Jotty SDK Developers

### Architecture

```
┌─────────────────────┐
│   JustJot.ai        │ Zero setup needed!
│   (Client)          │
└──────────┬──────────┘
           │ uses ChatAPI(agent_id="ChatAssistant")
           ↓
┌─────────────────────┐
│   Jotty SDK         │ Auto-registers ChatAssistant
│   (core/api)        │
└──────────┬──────────┘
           │ creates & wraps
           ↓
┌─────────────────────┐
│   ChatAssistant     │ Returns A2UI widgets
│   (core/agents)     │
└──────────┬──────────┘
           │ formats with
           ↓
┌─────────────────────┐
│   A2UI Formatter    │ format_task_list()
│   (core/ui)         │ format_card()
└─────────────────────┘
```

### How Auto-Registration Works

1. Client calls `ChatAPI(conductor, agent_id="ChatAssistant")`
2. ChatAPI checks if "ChatAssistant" exists in conductor.actors
3. If not found:
   - Creates `ChatAssistant` instance
   - Wraps in `AgentSpec` (Jotty's standard agent config)
   - Registers with conductor.actors dict
4. Everything works!

### Extending ChatAssistant

To add custom capabilities:

```python
from jotty.core.agents import ChatAssistant

class MyCustomAssistant(ChatAssistant):
    async def run(self, goal: str, **kwargs):
        # Add your custom logic
        if "custom" in goal.lower():
            return self._handle_custom_query(goal)

        # Fall back to parent for standard queries
        return await super().run(goal, **kwargs)
```

## DRY Principles in Action

**Before (Client code):**
```python
# Create agent
class MyChatAgent:
    def __init__(self, state_manager):
        self.state_manager = state_manager

    async def run(self, goal, **kwargs):
        # Query tasks
        tasks = await self.state_manager.get_tasks()
        # Format as A2UI
        return {"role": "assistant", "content": [...]}

# Create AgentSpec
agent_spec = AgentSpec(
    name="MyChatAgent",
    agent=MyChatAgent(state_manager),
    enable_architect=False,
    enable_auditor=False
)

# Register
conductor.actors["MyChatAgent"] = agent_spec

# Create ChatAPI
chat_api = ChatAPI(conductor, agent_id="MyChatAgent")
```

**After (Jotty provides):**
```python
chat_api = ChatAPI(conductor, agent_id="ChatAssistant", state_manager=state_manager)
```

**95% less code!** That's world-class SaaS SDK design.

## State Manager Interface

ChatAssistant expects state_manager to have:

```python
class StateManager:
    async def get_all_tasks(self) -> List[Dict[str, Any]]:
        """Return all tasks"""
        pass

    # OR

    async def list_tasks(self) -> List[Dict[str, Any]]:
        """Return all tasks"""
        pass
```

Task dictionary format:
```python
{
    "task_id": "TASK-20260118-00001",
    "title": "Task description",
    "status": "backlog",  # backlog, in_progress, completed, failed
    "created_at": "2026-01-18T10:00:00Z",  # Optional
    "priority": "High"  # Optional
}
```

## Testing

```bash
# Test ChatAssistant directly
python3 examples/task_assistant_agent.py

# Test integration
python3 test_a2ui_integration.py

# Test with real conductor
python3 -c "
from core.api import ChatAPI
from core.orchestration import Conductor

conductor = Conductor(...)
chat = ChatAPI(conductor, agent_id='ChatAssistant')
print('✅ ChatAssistant ready!')
"
```

## Summary

**For Clients:** Use `agent_id="ChatAssistant"` → Done!

**For Jotty:** Batteries included, auto-registration, zero config.

**Result:** World-class developer experience. 🚀
