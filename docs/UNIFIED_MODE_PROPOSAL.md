# Unified Mode Architecture Proposal

## First Principles Analysis

### Current State
```
ChatMode      → Synchronous (conversational, streaming)
WorkflowMode  → Synchronous (task execution via Conductor)
Queue         → Async (separate module, used by supervisor)
```

### User's Insight
- **Workflow and Queue are conceptually related** - both handle task/workflow execution
- **Chat is fundamentally synchronous** - conversational, immediate responses
- **Workflow can be synchronous OR async** - depending on use case

### Proposed Unification

```
ChatMode      → Always Synchronous (conversational)
WorkflowMode  → Synchronous OR Async (with integrated queue management)
```

## Benefits of Unification

### 1. **Conceptual Clarity**
- **ChatMode**: "I want an immediate response" (synchronous)
- **WorkflowMode**: "I want a task executed" (sync or async)

### 2. **Better API Design**
```python
# Current (confusing - queue is separate)
from Jotty.core.queue import SQLiteTaskQueue
queue = SQLiteTaskQueue(...)
workflow = WorkflowMode(conductor, mode="dynamic")

# Proposed (clear - queue is part of workflow)
workflow_sync = WorkflowMode(conductor, execution="sync", mode="dynamic")
workflow_async = WorkflowMode(conductor, execution="async", queue=queue, mode="dynamic")
```

### 3. **Supervisor Integration**
```python
# Current: Supervisor uses queue directly
from Jotty.core.queue import SQLiteTaskQueue
queue = SQLiteTaskQueue(...)

# Proposed: Supervisor uses WorkflowMode(async)
from Jotty.core.orchestration import WorkflowMode
workflow = WorkflowMode(
    conductor=conductor,
    execution="async",
    queue=SQLiteTaskQueue(...),
    mode="dynamic"
)
await workflow.enqueue_task(...)
await workflow.process_queue()
```

### 4. **Unified Task Management**
- Queue operations become part of WorkflowMode
- Single API for both sync and async workflows
- Better separation: Chat (sync) vs Workflow (sync/async)

## Proposed Architecture

### ChatMode (Unchanged)
```python
class ChatMode:
    """
    Always synchronous conversational interface.
    
    Use for:
    - Interactive chat
    - Real-time conversations
    - Streaming responses
    """
    
    async def stream(
        self,
        message: str,
        history: Optional[List[ChatMessage]] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        # Synchronous execution (immediate response)
        ...
```

### WorkflowMode (Enhanced)
```python
class WorkflowMode:
    """
    Task execution mode - Synchronous or Async.
    
    Execution modes:
    - "sync": Immediate execution (current behavior)
    - "async": Queue-based execution (background processing)
    
    Use sync for:
    - Interactive workflows
    - Real-time task execution
    - When you need immediate results
    
    Use async for:
    - Background jobs
    - Long-running tasks
    - Supervisor-style orchestration
    - Batch processing
    """
    
    def __init__(
        self,
        conductor: Conductor,
        execution: str = "sync",  # "sync" or "async"
        queue: Optional[TaskQueue] = None,  # Required for async
        mode: str = "dynamic",  # "static" or "dynamic"
        agent_order: Optional[List[str]] = None,
        max_concurrent: int = 3,  # For async mode
        poll_interval: float = 1.0,  # For async mode
    ):
        self.conductor = conductor
        self.execution = execution
        self.mode = mode
        
        if execution == "async":
            if queue is None:
                raise ValueError("queue is required for async execution")
            self.queue = queue
            self.queue_manager = TaskQueueManager(
                conductor=conductor,
                task_queue=queue,
                max_concurrent=max_concurrent,
                poll_interval=poll_interval,
            )
        else:
            self.queue = None
            self.queue_manager = None
        
        # Setup orchestrator (same for both sync and async)
        graph_mode = GraphMode.STATIC if mode == "static" else GraphMode.DYNAMIC
        self.orchestrator = LangGraphOrchestrator(
            conductor=conductor,
            mode=graph_mode,
            agent_order=agent_order
        )
    
    # Synchronous execution (current behavior)
    async def execute(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        max_iterations: int = 100
    ) -> Dict[str, Any]:
        """Execute workflow synchronously (immediate execution)"""
        if self.execution != "sync":
            raise ValueError("execute() requires sync execution mode")
        return await self.orchestrator.run(
            goal=goal,
            context=context,
            max_iterations=max_iterations
        )
    
    # Async execution (queue-based)
    async def enqueue_task(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        priority: int = 3,
        **kwargs
    ) -> str:
        """Enqueue task for async execution"""
        if self.execution != "async":
            raise ValueError("enqueue_task() requires async execution mode")
        
        task = Task(
            title=goal,
            description=goal,
            priority=priority,
            status=TaskStatus.PENDING,
            metadata=context or {},
            **kwargs
        )
        return await self.queue.enqueue(task)
    
    async def process_queue(self):
        """Start processing queued tasks (async mode only)"""
        if self.execution != "async":
            raise ValueError("process_queue() requires async execution mode")
        await self.queue_manager.start()
    
    async def stop_processing(self):
        """Stop processing queued tasks"""
        if self.execution == "async" and self.queue_manager:
            await self.queue_manager.stop()
```

## Migration Impact

### Supervisor Migration

**Current:**
```python
# supervisor/state_manager.py
from Jotty.core.queue import SQLiteTaskQueue

class StateManager:
    def __init__(self, db_path):
        self._queue = SQLiteTaskQueue(db_path=db_path, init_schema=True)
```

**Proposed:**
```python
# Option 1: Keep direct queue access (backward compatible)
from Jotty.core.queue import SQLiteTaskQueue  # Still available

# Option 2: Use WorkflowMode(async) for future features
from Jotty.core.orchestration import WorkflowMode
workflow = WorkflowMode(
    conductor=conductor,
    execution="async",
    queue=SQLiteTaskQueue(...),
    mode="dynamic"
)
```

### Backward Compatibility

**Queue module remains available:**
- `Jotty.core.queue` still exists
- Supervisor can continue using direct queue access
- No breaking changes

**WorkflowMode becomes more powerful:**
- Sync mode: Current behavior (unchanged)
- Async mode: New capability (optional)

## Implementation Plan

### Phase 1: Enhance WorkflowMode (Non-Breaking)
1. Add `execution` parameter to `WorkflowMode.__init__`
2. Add `queue` parameter for async mode
3. Integrate `TaskQueueManager` for async execution
4. Add `enqueue_task()` and `process_queue()` methods
5. Keep sync mode as default (backward compatible)

### Phase 2: Update Documentation
1. Document sync vs async use cases
2. Update examples
3. Add migration guide for supervisor (optional)

### Phase 3: Optional Supervisor Migration
1. Supervisor can migrate to `WorkflowMode(async)` if desired
2. Or continue using direct queue access
3. Both approaches work

## Benefits Summary

### ✅ Conceptual Clarity
- Chat = Synchronous (always)
- Workflow = Synchronous OR Async (flexible)

### ✅ Better API
- Unified interface for workflow execution
- Queue becomes part of WorkflowMode, not separate
- Clear separation of concerns

### ✅ Supervisor Benefits
- Can use WorkflowMode(async) for advanced features
- Or continue with direct queue access
- No breaking changes

### ✅ Future-Proof
- Easy to add more execution modes
- Clear extension points
- Better architecture for scaling

## Recommendation

**Proceed with unification** - it's a natural evolution that:
1. ✅ Doesn't break existing code (backward compatible)
2. ✅ Improves conceptual clarity
3. ✅ Makes WorkflowMode more powerful
4. ✅ Keeps queue module available for direct access
5. ✅ Supervisor can migrate gradually (or not at all)

The key insight: **Workflow and Queue are related concepts** - unifying them makes the architecture cleaner and more intuitive.
