# Unified Execution Mode Proposal

## Current State Analysis

### ChatMode vs WorkflowMode - What's Actually Different?

**ChatMode:**
- Conversational interface
- Formats message history
- Transforms events to chat-specific format
- Streaming responses
- Uses Conductor + Orchestrator

**WorkflowMode:**
- Task-oriented interface
- Direct goal execution
- Workflow events
- Streaming support
- Uses Conductor + Orchestrator

### Key Insight

**They're 95% the same!**

Both:
- Use `Conductor` for agent orchestration
- Use `LangGraphOrchestrator` for execution
- Support streaming
- Execute agents
- Handle context
- Support sync/async

**Only differences:**
1. **Event transformation** - Chat transforms to chat events, Workflow uses raw events
2. **History handling** - Chat formats conversation history, Workflow uses context dict
3. **Interface style** - Chat is conversational, Workflow is task-oriented

## Proposed Unified Design

### Single ExecutionMode with Style Parameter

```python
class ExecutionMode:
    """
    Unified execution mode for all agent interactions.
    
    Supports:
    - Conversational style (chat-like)
    - Task-oriented style (workflow-like)
    - Synchronous execution
    - Asynchronous execution (with queue)
    """
    
    def __init__(
        self,
        conductor: Conductor,
        style: str = "workflow",  # "chat" or "workflow"
        execution: str = "sync",  # "sync" or "async"
        queue: Optional[TaskQueue] = None,  # Required for async
        mode: str = "dynamic",  # "static" or "dynamic" (graph mode)
        agent_order: Optional[List[str]] = None,
        # Async-specific
        max_concurrent: int = 3,
        poll_interval: float = 1.0,
    ):
        self.conductor = conductor
        self.style = style  # "chat" or "workflow"
        self.execution = execution  # "sync" or "async"
        self.mode = mode
        
        # Setup orchestrator (same for both styles)
        graph_mode = GraphMode.STATIC if mode == "static" else GraphMode.DYNAMIC
        self.orchestrator = LangGraphOrchestrator(
            conductor=conductor,
            mode=graph_mode,
            agent_order=agent_order
        )
        
        # For async mode
        if execution == "async":
            if queue is None:
                raise ValueError("queue is required for async execution")
            if not hasattr(conductor, 'task_queue') or conductor.task_queue is None:
                conductor.task_queue = queue
        self.queue = queue
    
    # ========== Synchronous Execution ==========
    
    async def execute(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        history: Optional[List[ChatMessage]] = None,  # For chat style
        max_iterations: int = 100
    ) -> Dict[str, Any]:
        """
        Execute synchronously.
        
        For workflow style: Returns workflow result
        For chat style: Returns chat response
        """
        if self.execution != "sync":
            raise ValueError("execute() requires sync execution mode")
        
        # Prepare context based on style
        if self.style == "chat":
            # Format conversation history
            conversation_context = self._format_history(history or [])
            context = {
                "messages": [msg.to_dict() for msg in (history or [])],
                "conversation_history": conversation_context,
                "user_message": goal,
                **(context or {})
            }
        else:
            # Workflow style: use context as-is
            context = context or {}
        
        # Execute via orchestrator
        result = await self.orchestrator.run(
            goal=goal,
            context=context,
            max_iterations=max_iterations
        )
        
        # Transform result based on style
        if self.style == "chat":
            return self._transform_to_chat_response(result)
        else:
            return result.to_dict() if hasattr(result, 'to_dict') else result
    
    async def stream(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        history: Optional[List[ChatMessage]] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream execution events.
        
        For workflow style: Raw workflow events
        For chat style: Chat-specific events (reasoning, tool calls, text chunks)
        """
        if self.execution != "sync":
            raise ValueError("stream() requires sync execution mode")
        
        # Prepare context based on style
        if self.style == "chat":
            conversation_context = self._format_history(history or [])
            context = {
                "messages": [msg.to_dict() for msg in (history or [])],
                "conversation_history": conversation_context,
                "user_message": goal,
                **(context or {})
            }
        else:
            context = context or {}
        
        # Stream via orchestrator
        async for event in self.orchestrator.run_stream(goal=goal, context=context):
            if self.style == "chat":
                # Transform to chat events
                for chat_event in self._transform_to_chat_events(event):
                    yield chat_event
            else:
                # Workflow events as-is
                yield event
    
    # ========== Asynchronous Execution ==========
    
    async def enqueue_task(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        priority: int = 3,
        **kwargs
    ) -> str:
        """Enqueue task for async execution (workflow style only)"""
        if self.execution != "async":
            raise ValueError("enqueue_task() requires async execution mode")
        
        if self.style == "chat":
            raise ValueError("Chat style doesn't support async execution")
        
        task_id = await self.conductor.enqueue_goal(
            goal=goal,
            priority=priority,
            **kwargs
        )
        
        if task_id is None:
            raise RuntimeError("Failed to enqueue task")
        
        return task_id
    
    async def process_queue(
        self,
        max_tasks: Optional[int] = None,
        max_concurrent: int = 3
    ):
        """Process queued tasks (async mode, workflow style only)"""
        if self.execution != "async":
            raise ValueError("process_queue() requires async execution mode")
        
        if self.style == "chat":
            raise ValueError("Chat style doesn't support async execution")
        
        await self.conductor.process_queue(
            max_tasks=max_tasks,
            max_concurrent=max_concurrent
        )
    
    # ========== Helper Methods ==========
    
    def _format_history(self, history: List[ChatMessage]) -> str:
        """Format message history for conversational context"""
        return "\n".join([
            f"{msg.role}: {msg.content}"
            for msg in history
        ])
    
    def _transform_to_chat_events(self, event: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Transform workflow events to chat-specific events"""
        # Same logic as current ChatMode._transform_to_chat_events
        chat_events = []
        
        if event.get("type") != "agent_complete":
            return chat_events
        
        result = event.get("result", {})
        agent_name = event.get("agent")
        
        # Agent selection
        chat_events.append({
            "type": "agent_selected",
            "agent": agent_name,
            "timestamp": time.time()
        })
        
        # Reasoning
        if result.get("reasoning"):
            chat_events.append({
                "type": "reasoning",
                "content": result["reasoning"],
                "timestamp": time.time()
            })
        
        # Tool calls and results
        for tool_call in result.get("tool_calls", []):
            chat_events.append({
                "type": "tool_call",
                "tool": tool_call.get("name"),
                "args": tool_call.get("arguments"),
                "timestamp": time.time()
            })
        
        for tool_result in result.get("tool_results", []):
            chat_events.append({
                "type": "tool_result",
                "result": tool_result.get("result"),
                "timestamp": time.time()
            })
        
        # Text chunks
        response_text = result.get("response", "")
        if response_text:
            sentences = response_text.split(". ")
            for sentence in sentences:
                if sentence.strip():
                    chat_events.append({
                        "type": "text_chunk",
                        "content": sentence.strip() + ". ",
                        "timestamp": time.time()
                    })
        
        # Done event
        chat_events.append({
            "type": "done",
            "final_message": ChatMessage(
                role="assistant",
                content=response_text,
                timestamp=time.time()
            ).to_dict(),
            "timestamp": time.time()
        })
        
        return chat_events
    
    def _transform_to_chat_response(self, result) -> Dict[str, Any]:
        """Transform workflow result to chat response"""
        if hasattr(result, 'final_output'):
            return {
                "message": result.final_output,
                "agent_outputs": result.actor_outputs if hasattr(result, 'actor_outputs') else {},
                "metadata": result.metadata if hasattr(result, 'metadata') else {}
            }
        return {"message": str(result)}
```

## Usage Examples

### Chat Style (Synchronous)
```python
from Jotty.core.orchestration import ExecutionMode, Conductor

conductor = Conductor(actors=[...])
chat = ExecutionMode(conductor, style="chat", execution="sync")

# Single message
response = await chat.execute(
    goal="Explain transformers",
    history=[ChatMessage(role="user", content="What are transformers?")]
)

# Streaming
async for event in chat.stream(
    goal="Create idea about RAG",
    history=[...]
):
    if event["type"] == "text_chunk":
        print(event["content"], end="", flush=True)
```

### Workflow Style (Synchronous)
```python
workflow = ExecutionMode(conductor, style="workflow", execution="sync")

# Immediate execution
result = await workflow.execute(
    goal="Generate quarterly report",
    context={"quarter": "Q4"}
)

# Streaming
async for event in workflow.stream(goal="...", context={...}):
    print(f"Agent {event['agent']} completed")
```

### Workflow Style (Asynchronous)
```python
from Jotty.core.queue import SQLiteTaskQueue

queue = SQLiteTaskQueue(db_path="tasks.db")
workflow = ExecutionMode(
    conductor=conductor,
    style="workflow",
    execution="async",
    queue=queue
)

# Enqueue tasks
task_id = await workflow.enqueue_task(
    goal="Generate report",
    priority=1
)

# Process queue
await workflow.process_queue()
```

## Benefits of Unification

### 1. **Single API**
- One mode for all execution patterns
- Clear parameters: `style` and `execution`
- Consistent interface

### 2. **Code Reuse**
- Shared orchestrator logic
- Shared Conductor integration
- Only style-specific transformation differs

### 3. **Flexibility**
- Easy to add new styles (e.g., "batch", "interactive")
- Easy to add new execution modes
- Clear extension points

### 4. **Simpler Mental Model**
- One mode, different styles
- Clear separation: style (chat/workflow) vs execution (sync/async)

## Migration Path

### Phase 1: Create Unified ExecutionMode
1. Create new `ExecutionMode` class
2. Support both `style="chat"` and `style="workflow"`
3. Support both `execution="sync"` and `execution="async"`

### Phase 2: Deprecate Old Modes (Optional)
1. Keep `ChatMode` and `WorkflowMode` as thin wrappers
2. They delegate to `ExecutionMode` internally
3. Mark as deprecated with migration guide

### Phase 3: Update Usage
1. Update JustJot.ai chat to use `ExecutionMode(style="chat")`
2. Update supervisor to use `ExecutionMode(style="workflow", execution="async")`
3. Update examples and docs

## Recommendation

**✅ Unify into single ExecutionMode**

**Reasons:**
1. They're 95% the same code
2. Only difference is event transformation and history handling
3. Single API is simpler and more maintainable
4. Easy to extend with new styles
5. Clear separation: style vs execution

**Alternative:** Keep separate but share base class
- Less breaking change
- But more code duplication
- Less clean architecture
