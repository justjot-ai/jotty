# Unified Comprehensive Execution Mode

## Overview

Jotty's `ExecutionMode` is a **unified, comprehensive package** that provides:

- ✅ **Unified Interface**: Single API for chat and workflow
- ✅ **Learning**: Q-learning, TD(λ), predictive MARL (via Conductor)
- ✅ **Memory**: Hierarchical memory, consolidation (via Conductor)
- ✅ **Queue**: Optional but seamlessly integrated
- ✅ **All Capabilities**: Context management, data registry, etc.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│         Unified ExecutionMode                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Style: "chat" or "workflow"                           │
│  Execution: "sync" or "async"                          │
│                                                         │
│  ┌──────────────────────────────────────────────────┐ │
│  │         Core Capabilities (via Conductor)         │ │
│  │                                                   │ │
│  │  ✅ Learning: Q-learning, TD(λ), Predictive MARL │ │
│  │  ✅ Memory: Hierarchical, Consolidation          │ │
│  │  ✅ Context: SmartContextGuard, Compression      │ │
│  │  ✅ Data Registry: Agentic data discovery         │ │
│  │  ✅ Queue: Optional async task management         │ │
│  └──────────────────────────────────────────────────┘ │
│                                                         │
│  ┌──────────────────┐  ┌──────────────────────┐      │
│  │  Chat Style      │  │  Workflow Style      │      │
│  │  (sync only)     │  │  (sync or async)     │      │
│  │                  │  │                      │      │
│  │  Gets all       │  │  Gets all            │      │
│  │  capabilities   │  │  capabilities       │      │
│  └──────────────────┘  └──────────────────────┘      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Key Features

### 1. **Everyone Gets Learning**

All executions (chat, workflow, sync, async) benefit from:
- **Q-learning**: Value estimation for state-action pairs
- **TD(λ)**: Temporal difference learning
- **Predictive MARL**: Multi-agent reinforcement learning
- **Policy Exploration**: Alternative approaches when stuck

```python
# Chat gets learning
chat = ExecutionMode(conductor, style="chat")
# Every conversation learns from interactions

# Workflow gets learning
workflow = ExecutionMode(conductor, style="workflow")
# Every task execution improves future routing
```

### 2. **Everyone Gets Memory**

All executions benefit from:
- **Hierarchical Memory**: Context retention across sessions
- **Memory Consolidation**: Sharp wave ripple during "sleep"
- **Hippocampal Extraction**: What to remember (salience, novelty)
- **Shared Memory**: Cross-agent knowledge sharing

```python
# Memory is automatically available
result = await workflow.execute(goal="...")
# Context is retained for future executions
```

### 3. **Queue Integration**

Queue is optional but seamlessly integrated:

```python
# Without queue (sync)
workflow = ExecutionMode(conductor, style="workflow", execution="sync")
result = await workflow.execute(goal="...")

# With queue (async)
workflow = ExecutionMode(
    conductor, 
    style="workflow", 
    execution="async",
    queue=SQLiteTaskQueue(...)
)
task_id = await workflow.enqueue_task(goal="...")
await workflow.process_queue()
```

### 4. **Unified Nuances**

Chat and workflow handle their nuances automatically:

**Chat Style:**
- Formats conversation history
- Transforms events to chat-specific format
- Streaming text chunks
- Tool call visualization

**Workflow Style:**
- Direct goal execution
- Raw workflow events
- Task-oriented context
- Batch processing support

## Usage Examples

### Chat with Learning & Memory

```python
from Jotty.core.orchestration import ExecutionMode, Conductor, ChatMessage

conductor = Conductor(actors=[...])  # Gets learning & memory
chat = ExecutionMode(conductor, style="chat", execution="sync")

# Every conversation learns and remembers
async for event in chat.stream(
    goal="Explain transformers",
    history=[ChatMessage(role="user", content="What are transformers?")]
):
    if event["type"] == "text_chunk":
        print(event["content"], end="", flush=True)
```

### Workflow Sync with Learning & Memory

```python
workflow = ExecutionMode(conductor, style="workflow", execution="sync")

# Every execution learns and remembers
result = await workflow.execute(
    goal="Generate quarterly report",
    context={"quarter": "Q4"}
)
# Result includes learning & memory summaries
```

### Workflow Async with Learning & Memory + Queue

```python
from Jotty.core.queue import SQLiteTaskQueue

queue = SQLiteTaskQueue(db_path="tasks.db")
workflow = ExecutionMode(
    conductor=conductor,
    style="workflow",
    execution="async",
    queue=queue
)

# Enqueue tasks (will get learning & memory when executed)
task_id = await workflow.enqueue_task(
    goal="Generate report",
    priority=1
)

# Process queue (all tasks get learning & memory)
await workflow.process_queue()
```

## Supervisor Integration

Supervisor can now use ExecutionMode to get all capabilities:

```python
# supervisor/state_manager.py
from Jotty.core.orchestration import ExecutionMode, Conductor
from Jotty.core.queue import SQLiteTaskQueue

class StateManager:
    def __init__(self, db_path, conductor=None):
        self._queue = SQLiteTaskQueue(db_path=db_path)
        
        # Optional: Enable ExecutionMode for learning & memory
        if conductor:
            self.workflow = ExecutionMode(
                conductor=conductor,
                style="workflow",
                execution="async",
                queue=self._queue
            )
            # Now supervisor gets learning & memory!
```

## Benefits Summary

### ✅ Unified Interface
- Single API for all execution patterns
- Clear separation: style vs execution
- Consistent interface

### ✅ Everyone Gets Learning
- Chat learns from conversations
- Workflow learns from task executions
- Async tasks learn when processed
- All benefit from Q-learning, TD(λ), predictive MARL

### ✅ Everyone Gets Memory
- Context retention across sessions
- Memory consolidation
- Shared knowledge across agents
- Hippocampal extraction

### ✅ Queue Integration
- Optional but seamless
- Works with or without queue
- Async tasks get learning & memory too

### ✅ Backward Compatible
- Wrappers for old API
- Gradual migration path
- No breaking changes

## Migration Guide

### For Chat Applications

**Old:**
```python
from Jotty.core.orchestration import ChatMode
chat = ChatMode(conductor)
```

**New:**
```python
from Jotty.core.orchestration import ExecutionMode
chat = ExecutionMode(conductor, style="chat", execution="sync")
# Gets learning & memory automatically!
```

### For Workflow Applications

**Old:**
```python
from Jotty.core.orchestration import WorkflowMode
workflow = WorkflowMode(conductor)
```

**New:**
```python
from Jotty.core.orchestration import ExecutionMode
workflow = ExecutionMode(conductor, style="workflow", execution="sync")
# Gets learning & memory automatically!
```

### For Supervisor

**Current:**
```python
from Jotty.core.queue import SQLiteTaskQueue
queue = SQLiteTaskQueue(db_path)
```

**Enhanced:**
```python
from Jotty.core.orchestration import ExecutionMode, Conductor
from Jotty.core.queue import SQLiteTaskQueue

conductor = Conductor(actors=[...])
queue = SQLiteTaskQueue(db_path)
workflow = ExecutionMode(
    conductor=conductor,
    style="workflow",
    execution="async",
    queue=queue
)
# Supervisor gets learning & memory!
```

## Conclusion

**Unified ExecutionMode is comprehensive:**

- ✅ Single unified interface
- ✅ Everyone gets learning
- ✅ Everyone gets memory
- ✅ Queue optional but integrated
- ✅ All Jotty capabilities available
- ✅ Backward compatible
- ✅ Ready for production

**The key insight:** Chat and workflow are unified, and everyone benefits from Jotty's core capabilities automatically via Conductor.
