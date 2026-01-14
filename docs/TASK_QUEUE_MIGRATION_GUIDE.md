# Task Queue Migration Guide

## Overview

Jotty now includes a generic, reusable task queue management system that preserves **all** supervisor functionality while enabling reuse across projects. Migration is **optional** and **painless** - supervisor can continue using its own StateManager OR switch to Jotty's TaskQueue seamlessly.

## Architecture

```
┌─────────────────────────────────────────┐
│     Jotty Task Queue System             │
│  (Generic, Reusable, Pluggable)         │
├─────────────────────────────────────────┤
│  • TaskQueue (Abstract Interface)      │
│  • SQLiteTaskQueue (Persistent)        │
│  • MemoryTaskQueue (Testing)           │
│  • SupervisorStateManagerAdapter       │
│    (Backward Compatibility)             │
└─────────────────────────────────────────┘
           │                    │
           ▼                    ▼
    ┌──────────┐        ┌──────────────┐
    │ SQLite   │        │ Supervisor   │
    │ Backend  │        │ StateManager │
    └──────────┘        └──────────────┘
```

## Key Features Preserved

✅ **All supervisor functionality preserved:**
- Task CRUD operations
- Priority-based scheduling (1-5)
- Status management (pending, in_progress, completed, failed, etc.)
- Agent type filtering (claude, cursor, opencode)
- PID tracking for running tasks
- Heartbeat mechanism
- Retry logic
- Subtasks support (via database schema)
- Task metadata (title, description, category, context_files, etc.)
- Task history/audit log
- Task templates
- Git worktree management
- Log file tracking
- Statistics and analytics
- Export to JSON
- PID validation/cleanup

## Migration Options

### Option 1: No Migration (Continue Using Supervisor StateManager)

**Status:** ✅ **Works as-is, no changes needed**

Supervisor continues using its own `StateManager` class. No migration required.

```python
# supervisor/server.py - No changes needed
from supervisor.state_manager import StateManager

state_manager = StateManager(db_path="/data/supervisor-state.db")
# ... existing code works as-is
```

### Option 2: Use Jotty's SQLiteTaskQueue (Drop-in Replacement)

**Status:** ✅ **Drop-in replacement, same database schema**

Supervisor can switch to Jotty's `SQLiteTaskQueue` which uses the same database schema.

```python
# supervisor/server.py - Minimal changes
from Jotty.core.queue import SQLiteTaskQueue

# Replace StateManager with SQLiteTaskQueue
task_queue = SQLiteTaskQueue(db_path="/data/supervisor-state.db")

# Use same API
task = await task_queue.dequeue(agent_type="claude")
await task_queue.update_status(task_id, "in_progress", pid=12345)
```

### Option 3: Use Supervisor Adapter (Backward Compatibility)

**Status:** ✅ **Zero code changes, wraps existing StateManager**

Supervisor can use `SupervisorStateManagerAdapter` to wrap its existing `StateManager`, making it compatible with Jotty's TaskQueue interface.

```python
# supervisor/server.py - Wrap existing StateManager
from supervisor.state_manager import StateManager
from Jotty.core.queue import SupervisorStateManagerAdapter

state_manager = StateManager(db_path="/data/supervisor-state.db")
task_queue = SupervisorStateManagerAdapter(state_manager)

# Now compatible with Jotty's TaskQueue interface
# Existing code continues to work
```

### Option 4: Use Jotty's TaskQueue in New Projects

**Status:** ✅ **Generic, reusable across projects**

New projects can use Jotty's TaskQueue directly:

```python
from Jotty.core.queue import SQLiteTaskQueue, Task
from Jotty.core.orchestration import Conductor
from Jotty.core.queue import TaskQueueManager

# Create queue
queue = SQLiteTaskQueue("/data/tasks.db")

# Create conductor
conductor = Conductor(actors=[...], task_queue=queue)

# Enqueue goals
task_id = await conductor.enqueue_goal(
    goal="Research and implement feature X",
    priority=7
)

# Process queue
manager = TaskQueueManager(conductor, queue, max_concurrent=3)
await manager.start()
```

## API Compatibility

### Supervisor StateManager → Jotty TaskQueue Mapping

| Supervisor Method | Jotty TaskQueue Method | Notes |
|-------------------|------------------------|-------|
| `get_next_pending_task(agent_type)` | `dequeue(filters={'agent_type': ...})` | Same functionality |
| `update_task_status(...)` | `update_status(...)` | Same signature |
| `get_running_count_by_agent(...)` | `get_running_count_by_agent(...)` | Same |
| `get_stats()` | `get_stats()` | Same format |
| `create_task(...)` | `create_task(...)` | Same signature |
| `get_task_by_task_id(...)` | `get_task(...)` | Same |
| `export_to_json()` | `export_to_json()` | Same format |
| `validate_pids()` | `validate_pids()` | Same |

## Database Schema Compatibility

✅ **Same SQLite schema** - Jotty's `SQLiteTaskQueue` uses the same database schema as supervisor, ensuring:
- No data migration needed
- Same table structure
- Same indexes
- Same constraints
- Backward compatible

## Step-by-Step Migration (Optional)

### Step 1: Install Jotty (if not already installed)

```bash
cd /path/to/Jotty
pip install -e .
```

### Step 2: Choose Migration Option

**Option A: Drop-in Replacement**
```python
# Replace StateManager import
from Jotty.core.queue import SQLiteTaskQueue as StateManager

# Rest of code unchanged
state_manager = StateManager(db_path="/data/supervisor-state.db")
```

**Option B: Use Adapter**
```python
# Wrap existing StateManager
from supervisor.state_manager import StateManager
from Jotty.core.queue import SupervisorStateManagerAdapter

state_manager = StateManager(db_path="/data/supervisor-state.db")
task_queue = SupervisorStateManagerAdapter(state_manager)

# Use task_queue instead of state_manager
```

### Step 3: Update Async Calls (if needed)

Supervisor's `StateManager` uses synchronous methods. Jotty's `TaskQueue` uses async methods. If using Option 2 or 4:

```python
# Before (synchronous)
task = state_manager.get_next_pending_task(agent_type="claude")
state_manager.update_task_status(task_id, "in_progress", pid=12345)

# After (asynchronous)
task = await task_queue.dequeue(filters={'agent_type': 'claude'})
await task_queue.update_status(task_id, "in_progress", pid=12345)
```

**Note:** `SupervisorStateManagerAdapter` handles sync→async conversion automatically.

### Step 4: Test

```bash
# Run supervisor tests
python -m pytest supervisor/tests/

# Verify database compatibility
python -c "
from Jotty.core.queue import SQLiteTaskQueue
queue = SQLiteTaskQueue('/data/supervisor-state.db')
import asyncio
stats = asyncio.run(queue.get_stats())
print(stats)
"
```

## Benefits of Migration

### ✅ Reusability
- Generic task queue usable across projects
- Not tied to supervisor's specific use case

### ✅ Integration
- Works seamlessly with Jotty's Conductor
- Can queue goals for agent execution
- Priority-based scheduling for agent tasks

### ✅ Scalability
- Redis backend support (future)
- Distributed task processing (future)

### ✅ Maintainability
- Single source of truth for task queue logic
- Consistent API across projects

## Backward Compatibility Guarantee

✅ **No Breaking Changes:**
- Supervisor can continue using `StateManager` as-is
- Same database schema
- Same API methods
- Same data format

✅ **Optional Migration:**
- Migration is optional, not required
- Can migrate gradually
- Can use adapter for zero-code migration

## Examples

### Example 1: Supervisor Using Jotty's Queue

```python
# supervisor/server.py
from Jotty.core.queue import SQLiteTaskQueue

# Initialize queue (same database)
task_queue = SQLiteTaskQueue("/data/supervisor-state.db")

# Use in orchestrator
async def orchestrate():
    while True:
        # Get next task
        task = await task_queue.dequeue(filters={'agent_type': 'claude'})
        if task:
            # Spawn task
            spawn_task(task.task_id)
        
        await asyncio.sleep(30)
```

### Example 2: Jotty Conductor with Task Queue

```python
from Jotty.core.queue import SQLiteTaskQueue, TaskQueueManager
from Jotty.core.orchestration import Conductor, AgentConfig

# Create queue
queue = SQLiteTaskQueue("/data/tasks.db")

# Create conductor with queue
conductor = Conductor(
    actors=[research_agent, coding_agent],
    task_queue=queue,  # Optional!
    use_langgraph=True
)

# Enqueue goals
task_id = await conductor.enqueue_goal(
    goal="Research and implement feature X",
    priority=7,
    agent_type="claude"
)

# Process queue automatically
manager = TaskQueueManager(conductor, queue, max_concurrent=3)
await manager.start()
```

### Example 3: Supervisor Adapter (Zero Code Changes)

```python
# supervisor/server.py - Minimal changes
from supervisor.state_manager import StateManager
from Jotty.core.queue import SupervisorStateManagerAdapter

# Wrap existing StateManager
state_manager = StateManager("/data/supervisor-state.db")
task_queue = SupervisorStateManagerAdapter(state_manager)

# Now compatible with Jotty's TaskQueue interface
# Existing code continues to work
# Can optionally use Jotty features
```

## FAQ

### Q: Do I need to migrate?
**A:** No! Migration is optional. Supervisor works as-is.

### Q: Will migration break existing functionality?
**A:** No! Same database schema, same API, backward compatible.

### Q: Can I use Jotty's queue in other projects?
**A:** Yes! It's generic and reusable.

### Q: What if I want to keep using supervisor's StateManager?
**A:** Use `SupervisorStateManagerAdapter` to wrap it, or don't migrate at all.

### Q: Does it support all supervisor features?
**A:** Yes! All features preserved.

## Conclusion

✅ **No migration pain** - Supervisor works as-is  
✅ **All functionality preserved** - Nothing lost  
✅ **Reusable across projects** - Generic design  
✅ **Optional migration** - Choose what works for you  

The task queue system is designed to be **optional**, **backward compatible**, and **reusable**. Supervisor can continue using its own StateManager, or optionally switch to Jotty's TaskQueue for better integration and reusability.
