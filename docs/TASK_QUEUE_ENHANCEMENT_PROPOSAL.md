# Task Queue Management Enhancement Proposal for Jotty

## Executive Summary

**Proposal:** Enhance Jotty with a generic, reusable task queue management system that can be used across projects, complementing its multi-agent orchestration capabilities.

## Current State Analysis

### Jotty (Current)
- **Focus:** Multi-agent orchestration
- **Features:** LangGraph integration, dynamic/static modes, agent execution
- **Missing:** Task queue management, persistence, priority handling

### Supervisor (JustJot.ai)
- **Focus:** Task queue management for CLI tasks
- **Features:** 
  - SQLite-based task persistence
  - Priority-based scheduling
  - Task state management (pending, in_progress, completed, failed)
  - Concurrency control
  - Retry logic
  - Task metadata tracking

## Proposed Enhancement

### Architecture: Layered Task Queue System

```
┌─────────────────────────────────────────┐
│     Jotty Task Queue Manager            │
│  (Generic, Reusable, Pluggable)         │
├─────────────────────────────────────────┤
│  • Task Queue Interface                 │
│  • Priority Scheduling                  │
│  • Concurrency Control                  │
│  • Retry Logic                          │
│  • State Management                     │
│  • Persistence Backend (Pluggable)      │
└─────────────────────────────────────────┘
           │                    │
           ▼                    ▼
    ┌──────────┐        ┌──────────────┐
    │ SQLite   │        │ Redis/Memory │
    │ Backend  │        │ Backend      │
    └──────────┘        └──────────────┘
           │                    │
           ▼                    ▼
    ┌──────────────────────────────────┐
    │  Agent Orchestration (Existing)   │
    │  • LangGraph Integration         │
    │  • Dynamic/Static Modes           │
    └──────────────────────────────────┘
```

## Design Proposal

### 1. Core Components

#### `TaskQueue` (Abstract Base Class)
```python
class TaskQueue(ABC):
    """Generic task queue interface"""
    
    @abstractmethod
    async def enqueue(self, task: Task) -> str:
        """Add task to queue, returns task_id"""
        pass
    
    @abstractmethod
    async def dequeue(self, filters: Optional[Dict] = None) -> Optional[Task]:
        """Get next task from queue"""
        pass
    
    @abstractmethod
    async def update_status(self, task_id: str, status: TaskStatus, **kwargs):
        """Update task status"""
        pass
    
    @abstractmethod
    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID"""
        pass
```

#### `Task` (Data Model)
```python
@dataclass
class Task:
    task_id: str
    payload: Dict[str, Any]  # Task-specific data
    priority: int = 5  # 1-10, higher = more important
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)
    agent_type: Optional[str] = None  # For agent-specific queues
```

#### `TaskStatus` (Enum)
```python
class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"
```

### 2. Backend Implementations

#### `SQLiteTaskQueue` (Persistent)
```python
class SQLiteTaskQueue(TaskQueue):
    """SQLite-backed task queue for persistence"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()
    
    async def enqueue(self, task: Task) -> str:
        # Insert into SQLite with priority ordering
        pass
    
    async def dequeue(self, filters: Optional[Dict] = None) -> Optional[Task]:
        # SELECT with priority ordering, status filtering
        pass
```

#### `MemoryTaskQueue` (In-Memory)
```python
class MemoryTaskQueue(TaskQueue):
    """In-memory task queue for testing/development"""
    
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.queue: List[str] = []  # Priority-sorted task IDs
```

#### `RedisTaskQueue` (Distributed)
```python
class RedisTaskQueue(TaskQueue):
    """Redis-backed task queue for distributed systems"""
    
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)
```

### 3. Integration with Conductor

```python
class Conductor:
    def __init__(
        self,
        actors: List[ActorConfig],
        task_queue: Optional[TaskQueue] = None,  # NEW
        use_langgraph: bool = False,
        # ... existing params
    ):
        self.task_queue = task_queue or MemoryTaskQueue()
        # ... existing initialization
    
    async def enqueue_goal(self, goal: str, priority: int = 5, **kwargs) -> str:
        """Enqueue a goal as a task"""
        task = Task(
            task_id=uuid.uuid4().hex,
            payload={"goal": goal, **kwargs},
            priority=priority,
            metadata={"conductor_id": self.id}
        )
        return await self.task_queue.enqueue(task)
    
    async def process_queue(self, max_tasks: Optional[int] = None):
        """Process tasks from queue"""
        processed = 0
        while True:
            if max_tasks and processed >= max_tasks:
                break
            
            task = await self.task_queue.dequeue()
            if not task:
                await asyncio.sleep(1)  # Poll interval
                continue
            
            await self.task_queue.update_status(
                task.task_id, 
                TaskStatus.IN_PROGRESS,
                started_at=datetime.now()
            )
            
            try:
                # Execute via existing LangGraph orchestration
                result = await self.run(
                    goal=task.payload["goal"],
                    **{k: v for k, v in task.payload.items() if k != "goal"}
                )
                
                await self.task_queue.update_status(
                    task.task_id,
                    TaskStatus.COMPLETED,
                    completed_at=datetime.now(),
                    metadata={"result": result.to_dict()}
                )
                processed += 1
                
            except Exception as e:
                if task.retry_count < task.max_retries:
                    await self.task_queue.update_status(
                        task.task_id,
                        TaskStatus.RETRYING,
                        retry_count=task.retry_count + 1
                    )
                else:
                    await self.task_queue.update_status(
                        task.task_id,
                        TaskStatus.FAILED,
                        metadata={"error": str(e)}
                    )
```

### 4. Task Queue Manager (Standalone)

```python
class TaskQueueManager:
    """High-level task queue manager with agent orchestration"""
    
    def __init__(
        self,
        conductor: Conductor,
        task_queue: TaskQueue,
        max_concurrent: int = 3
    ):
        self.conductor = conductor
        self.task_queue = task_queue
        self.max_concurrent = max_concurrent
        self.running_tasks: Dict[str, asyncio.Task] = {}
    
    async def start(self):
        """Start processing tasks"""
        while True:
            # Check concurrency limit
            running_count = len([t for t in self.running_tasks.values() if not t.done()])
            
            if running_count < self.max_concurrent:
                task = await self.task_queue.dequeue()
                if task:
                    self.running_tasks[task.task_id] = asyncio.create_task(
                        self._process_task(task)
                    )
            
            # Clean up completed tasks
            self.running_tasks = {
                tid: t for tid, t in self.running_tasks.items() 
                if not t.done()
            }
            
            await asyncio.sleep(1)  # Poll interval
    
    async def _process_task(self, task: Task):
        """Process a single task"""
        # Use conductor to execute via LangGraph
        result = await self.conductor.run(task.payload["goal"])
        return result
```

## Benefits

### 1. **Reusability**
- Generic task queue can be used across projects
- Pluggable backends (SQLite, Redis, Memory)
- Not tied to specific use case

### 2. **Separation of Concerns**
- Task queue management separate from agent orchestration
- Can use queue without agents, or agents without queue
- Clean interfaces

### 3. **Scalability**
- Redis backend for distributed systems
- Concurrency control built-in
- Priority-based scheduling

### 4. **Integration**
- Works seamlessly with existing LangGraph orchestration
- Can queue goals for agent execution
- Maintains Jotty's agent-centric design

### 5. **Migration Path**
- Supervisor can use Jotty's TaskQueue
- JustJot.ai can leverage Jotty for task management
- Other projects can use task queue independently

## Use Cases

### 1. **JustJot.ai Supervisor Migration**
```python
# Supervisor uses Jotty's TaskQueue
from Jotty.core.queue import SQLiteTaskQueue
from Jotty.core.orchestration import Conductor

task_queue = SQLiteTaskQueue(db_path="/data/tasks.db")
conductor = Conductor(actors=[...], task_queue=task_queue)

# Enqueue CLI tasks
task_id = await conductor.enqueue_goal(
    goal="Implement feature X",
    priority=8,
    agent_type="claude"
)
```

### 2. **Background Agent Processing**
```python
# Process agent goals in background
queue_manager = TaskQueueManager(
    conductor=conductor,
    task_queue=task_queue,
    max_concurrent=5
)

# Start background processing
await queue_manager.start()
```

### 3. **Priority-Based Agent Execution**
```python
# High-priority task
await conductor.enqueue_goal("Critical bug fix", priority=10)

# Low-priority task
await conductor.enqueue_goal("Documentation update", priority=2)
```

## Implementation Plan

### Phase 1: Core Task Queue (Week 1)
- [ ] `Task` data model
- [ ] `TaskQueue` abstract interface
- [ ] `SQLiteTaskQueue` implementation
- [ ] `MemoryTaskQueue` implementation
- [ ] Basic tests

### Phase 2: Conductor Integration (Week 2)
- [ ] Add `task_queue` parameter to `Conductor`
- [ ] `enqueue_goal()` method
- [ ] `process_queue()` method
- [ ] Integration tests

### Phase 3: Task Queue Manager (Week 3)
- [ ] `TaskQueueManager` class
- [ ] Concurrency control
- [ ] Retry logic
- [ ] Monitoring/status endpoints

### Phase 4: Advanced Features (Week 4)
- [ ] `RedisTaskQueue` implementation
- [ ] Priority scheduling improvements
- [ ] Task filtering/search
- [ ] Metrics/observability

## API Design

### Simple Usage
```python
from Jotty.core.queue import SQLiteTaskQueue
from Jotty.core.orchestration import Conductor

# Create queue
queue = SQLiteTaskQueue("/data/tasks.db")

# Create conductor with queue
conductor = Conductor(
    actors=[research_agent, coding_agent],
    task_queue=queue,
    use_langgraph=True
)

# Enqueue goal
task_id = await conductor.enqueue_goal(
    goal="Research and implement feature X",
    priority=7
)

# Process queue (background)
import asyncio
asyncio.create_task(conductor.process_queue())

# Or use TaskQueueManager
from Jotty.core.queue import TaskQueueManager
manager = TaskQueueManager(conductor, queue, max_concurrent=3)
await manager.start()
```

### Advanced Usage
```python
# Custom task with metadata
task = Task(
    task_id="custom-123",
    payload={"goal": "Complex task", "params": {...}},
    priority=9,
    metadata={"project": "justjot", "user_id": "user-123"},
    agent_type="research"
)
task_id = await queue.enqueue(task)

# Filter tasks
task = await queue.dequeue(filters={
    "status": TaskStatus.PENDING,
    "agent_type": "research",
    "priority_min": 7
})

# Update status
await queue.update_status(
    task_id,
    TaskStatus.IN_PROGRESS,
    started_at=datetime.now(),
    metadata={"worker": "worker-1"}
)
```

## Considerations

### Pros ✅
- Generic, reusable across projects
- Complements agent orchestration
- Pluggable backends
- Clean separation of concerns
- Migration path for Supervisor

### Cons ⚠️
- Adds complexity to Jotty
- May be overkill for simple use cases
- Requires maintenance of queue backends
- Need to ensure it doesn't bloat Jotty

### Mitigation
- Make task queue **optional** (not required)
- Keep core Jotty focused on agent orchestration
- Task queue as separate module (`Jotty.core.queue`)
- Clear documentation on when to use

## Recommendation

**✅ YES - Enhance Jotty with Task Queue Management**

**Rationale:**
1. **Reusability:** Generic task queue benefits multiple projects
2. **Integration:** Natural fit with agent orchestration
3. **Migration Path:** Supervisor can use Jotty's queue
4. **Scalability:** Redis backend for distributed systems
5. **Optional:** Doesn't bloat core Jotty if made optional

**Implementation:**
- Keep it **optional** (Conductor works without queue)
- Separate module (`Jotty.core.queue`)
- Pluggable backends (SQLite, Redis, Memory)
- Clear API for simple and advanced use cases

---

**Status:** Proposal for Review  
**Next Steps:** Get approval, then implement Phase 1
