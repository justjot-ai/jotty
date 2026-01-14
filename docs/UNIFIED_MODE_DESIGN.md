# Unified Mode Design - Detailed Proposal

## Current State Analysis

### What Exists
1. **Conductor** already has queue support:
   - `enqueue_goal()` - Enqueue tasks
   - `process_queue()` - Process queued tasks
   - `task_queue` parameter in `__init__`

2. **WorkflowMode** currently:
   - Only supports synchronous execution
   - Uses Conductor's `run()` method directly

3. **Queue Module**:
   - Standalone task queue system
   - Used by supervisor directly

### The Gap
- WorkflowMode doesn't expose async/queue capabilities
- Queue is separate from WorkflowMode conceptually
- Supervisor uses queue directly (not through WorkflowMode)

## Proposed Design

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Jotty Modes                           │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │            ChatMode                              │  │
│  │  - Always Synchronous                            │  │
│  │  - Conversational interface                      │  │
│  │  - Streaming responses                           │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │         WorkflowMode                             │  │
│  │                                                   │  │
│  │  ┌──────────────────┐  ┌──────────────────────┐ │  │
│  │  │  Sync Execution   │  │  Async Execution     │ │  │
│  │  │                  │  │                      │ │  │
│  │  │  execute()       │  │  enqueue_task()      │ │  │
│  │  │  execute_stream()│  │  process_queue()     │ │  │
│  │  │                  │  │                      │ │  │
│  │  │  Uses:          │  │  Uses:               │ │  │
│  │  │  - Conductor    │  │  - Conductor         │ │  │
│  │  │  - Orchestrator │  │  - TaskQueue         │ │  │
│  │  │                 │  │  - QueueManager      │ │  │
│  │  └──────────────────┘  └──────────────────────┘ │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### WorkflowMode API

```python
class WorkflowMode:
    """
    Unified workflow execution mode.
    
    Supports both synchronous and asynchronous execution.
    """
    
    def __init__(
        self,
        conductor: Conductor,
        execution: str = "sync",  # "sync" or "async"
        queue: Optional[TaskQueue] = None,  # Required for async
        mode: str = "dynamic",  # "static" or "dynamic" (graph mode)
        agent_order: Optional[List[str]] = None,
        # Async-specific parameters
        max_concurrent: int = 3,
        poll_interval: float = 1.0,
    ):
        self.conductor = conductor
        self.execution = execution
        self.mode = mode
        
        # Setup orchestrator (same for both modes)
        graph_mode = GraphMode.STATIC if mode == "static" else GraphMode.DYNAMIC
        self.orchestrator = LangGraphOrchestrator(
            conductor=conductor,
            mode=graph_mode,
            agent_order=agent_order
        )
        
        # For async mode: configure Conductor with queue
        if execution == "async":
            if queue is None:
                raise ValueError("queue is required for async execution")
            # Conductor already supports task_queue parameter
            if not hasattr(conductor, 'task_queue') or conductor.task_queue is None:
                conductor.task_queue = queue
        
        self.queue = queue
    
    # ========== Synchronous Execution ==========
    
    async def execute(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        max_iterations: int = 100
    ) -> Dict[str, Any]:
        """
        Execute workflow synchronously (immediate execution).
        
        Use for:
        - Interactive workflows
        - Real-time task execution
        - When you need immediate results
        """
        if self.execution != "sync":
            raise ValueError("execute() requires sync execution mode. Use enqueue_task() for async.")
        
        return await self.orchestrator.run(
            goal=goal,
            context=context,
            max_iterations=max_iterations
        )
    
    async def execute_stream(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """Execute workflow synchronously with streaming events."""
        if self.execution != "sync":
            raise ValueError("execute_stream() requires sync execution mode")
        
        async for event in self.orchestrator.run_stream(goal=goal, context=context):
            yield event
    
    # ========== Asynchronous Execution ==========
    
    async def enqueue_task(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        priority: int = 3,
        **kwargs
    ) -> str:
        """
        Enqueue task for asynchronous execution.
        
        Use for:
        - Background jobs
        - Long-running tasks
        - Batch processing
        - Supervisor-style orchestration
        """
        if self.execution != "async":
            raise ValueError("enqueue_task() requires async execution mode. Use execute() for sync.")
        
        # Use Conductor's existing enqueue_goal method
        task_id = await self.conductor.enqueue_goal(
            goal=goal,
            priority=priority,
            **kwargs
        )
        
        if task_id is None:
            raise RuntimeError("Failed to enqueue task. Ensure Conductor has task_queue configured.")
        
        return task_id
    
    async def process_queue(
        self,
        max_tasks: Optional[int] = None,
        max_concurrent: int = 3
    ):
        """
        Start processing queued tasks (async mode only).
        
        This uses Conductor's process_queue() method which:
        - Polls queue for pending tasks
        - Executes tasks via Conductor.run()
        - Manages concurrency
        - Handles retries and errors
        """
        if self.execution != "async":
            raise ValueError("process_queue() requires async execution mode")
        
        await self.conductor.process_queue(
            max_tasks=max_tasks,
            max_concurrent=max_concurrent
        )
    
    async def stop_processing(self):
        """Stop processing queued tasks."""
        if self.execution == "async" and self.conductor.task_queue:
            # Conductor's process_queue runs until stopped
            # This would need to be implemented in Conductor
            pass
```

## Usage Examples

### Synchronous Workflow (Current Behavior)
```python
from Jotty.core.orchestration import WorkflowMode, Conductor

conductor = Conductor(actors=[...])
workflow = WorkflowMode(conductor, execution="sync", mode="dynamic")

# Immediate execution
result = await workflow.execute(
    goal="Generate quarterly report",
    context={"quarter": "Q4"}
)
```

### Asynchronous Workflow (New Capability)
```python
from Jotty.core.orchestration import WorkflowMode, Conductor
from Jotty.core.queue import SQLiteTaskQueue

# Setup
conductor = Conductor(actors=[...])
queue = SQLiteTaskQueue(db_path="tasks.db")

workflow = WorkflowMode(
    conductor=conductor,
    execution="async",
    queue=queue,
    mode="dynamic",
    max_concurrent=3
)

# Enqueue tasks
task_id1 = await workflow.enqueue_task(
    goal="Generate report Q1",
    priority=1
)
task_id2 = await workflow.enqueue_task(
    goal="Generate report Q2",
    priority=2
)

# Process queue (runs until stopped)
await workflow.process_queue()
```

### Supervisor Integration (Future)
```python
# supervisor/state_manager.py (optional migration)
from Jotty.core.orchestration import WorkflowMode, Conductor
from Jotty.core.queue import SQLiteTaskQueue

class StateManager:
    def __init__(self, db_path, conductor=None):
        self.queue = SQLiteTaskQueue(db_path=db_path)
        
        # Option: Use WorkflowMode for advanced features
        if conductor:
            self.workflow = WorkflowMode(
                conductor=conductor,
                execution="async",
                queue=self.queue,
                mode="dynamic"
            )
        else:
            self.workflow = None
    
    # Keep existing API (backward compatible)
    def create_task(self, ...):
        # Can use workflow.enqueue_task() or direct queue access
        ...
```

## Benefits

### 1. **Conceptual Clarity**
- **ChatMode**: Always synchronous (conversational)
- **WorkflowMode**: Synchronous OR asynchronous (flexible)

### 2. **Unified API**
- Single interface for workflow execution
- Clear separation: sync vs async
- Queue becomes part of WorkflowMode

### 3. **Backward Compatible**
- Sync mode is default (current behavior)
- Queue module still available for direct access
- Supervisor can migrate gradually

### 4. **Leverages Existing Code**
- Conductor already has `enqueue_goal()` and `process_queue()`
- WorkflowMode just exposes these capabilities
- Minimal new code needed

### 5. **Future-Proof**
- Easy to add more execution modes
- Clear extension points
- Better architecture for scaling

## Migration Path

### Phase 1: Enhance WorkflowMode (Non-Breaking)
1. Add `execution` parameter (default="sync")
2. Add `queue` parameter (optional, required for async)
3. Add `enqueue_task()` and `process_queue()` methods
4. Keep sync methods unchanged

### Phase 2: Documentation
1. Document sync vs async use cases
2. Update examples
3. Add migration guide

### Phase 3: Optional Supervisor Migration
1. Supervisor can use WorkflowMode(async) if desired
2. Or continue with direct queue access
3. Both approaches work

## Recommendation

**✅ Proceed with unification**

**Reasons:**
1. Natural evolution (Conductor already supports queue)
2. Better conceptual model (Workflow = sync OR async)
3. No breaking changes (backward compatible)
4. Supervisor can migrate gradually (or not at all)
5. Clear separation: Chat (sync) vs Workflow (sync/async)

**Key Insight:** Workflow and Queue are related concepts - unifying them makes the architecture cleaner and more intuitive.
