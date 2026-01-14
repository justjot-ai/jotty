# Jotty Execution Modes

Jotty provides two execution modes for different use cases:

1. **WorkflowMode** - Task-oriented multi-agent collaboration
2. **ChatMode** - Conversational interface with streaming responses

---

## Quick Start

```python
from Jotty import Conductor
from Jotty.core.orchestration import create_workflow, create_chat, ChatMessage

# Setup
conductor = Conductor(actors=[...])

# Workflow: Execute task
workflow = create_workflow(conductor, mode="dynamic")
result = await workflow.execute(goal="Generate report")

# Chat: Interactive conversation
chat = create_chat(conductor, agent_id="research-assistant")
async for event in chat.stream(message="What is AI?"):
    if event["type"] == "text_chunk":
        print(event["content"], end="")
```

---

## WorkflowMode

**Use for**: Background jobs, pipelines, multi-step tasks

### Features
- ✅ Multi-agent collaboration
- ✅ Static (predefined) or dynamic (adaptive) routing
- ✅ Batch execution with final result
- ✅ Optional streaming for progress updates

### Usage

#### Dynamic Workflow (Adaptive Routing)
```python
workflow = create_workflow(conductor, mode="dynamic")

result = await workflow.execute(
    goal="Analyze market trends and create report",
    context={"market": "tech", "year": 2026}
)

print(result["final_output"])  # Complete report
print(result["actor_outputs"])  # Individual agent outputs
```

**When to use**: Complex tasks where agent order isn't known upfront

#### Static Workflow (Predefined Order)
```python
workflow = create_workflow(
    conductor,
    mode="static",
    agent_order=["Research", "Writer", "Editor"]
)

result = await workflow.execute(
    goal="Create article about quantum computing"
)
```

**When to use**: ETL pipelines, report generation, known sequences

#### Streaming Workflow
```python
async for event in workflow.execute_stream(goal="Long task"):
    if event["type"] == "agent_start":
        print(f"Starting: {event['agent']}")
    elif event["type"] == "agent_complete":
        print(f"Done: {event['agent']}")
```

**When to use**: Long-running tasks with UI progress bars

### Return Value
```python
{
    "success": bool,
    "final_output": str,
    "actor_outputs": {
        "AgentName": {...}
    },
    "metadata": {...}
}
```

---

## ChatMode

**Use for**: Interactive conversations, user-facing chat

### Features
- ✅ Conversational interface with message history
- ✅ Streaming responses (text chunks)
- ✅ Tool execution visibility
- ✅ Single-agent or multi-agent chat
- ✅ Chain-of-thought reasoning display

### Usage

#### Single-Agent Chat
```python
chat = create_chat(conductor, agent_id="research-assistant")

history = [
    ChatMessage(role="user", content="Previous question"),
    ChatMessage(role="assistant", content="Previous answer")
]

async for event in chat.stream(message="New question", history=history):
    if event["type"] == "text_chunk":
        print(event["content"], end="", flush=True)
```

**When to use**: Specialized chatbot (research bot, coding assistant)

#### Multi-Agent Chat
```python
# Dynamic routing (agents collaborate adaptively)
chat = create_chat(conductor, mode="dynamic")

# Static routing (predefined agent order)
chat = create_chat(
    conductor,
    mode="static",
    agent_order=["Analyst", "Writer"]
)

async for event in chat.stream(message="Complex question"):
    # Handle events...
```

**When to use**: Complex questions requiring multiple experts

### Event Types

#### `agent_selected`
Agent chosen to handle request
```python
{
    "type": "agent_selected",
    "agent": "research-assistant",
    "timestamp": 1234567890.0
}
```

#### `reasoning`
Chain-of-thought reasoning process
```python
{
    "type": "reasoning",
    "content": "Let me search for information about...",
    "timestamp": 1234567890.0
}
```

#### `tool_call`
Agent calling a tool/function
```python
{
    "type": "tool_call",
    "tool": "search",
    "args": {"query": "AI trends"},
    "timestamp": 1234567890.0
}
```

#### `tool_result`
Result from tool execution
```python
{
    "type": "tool_result",
    "result": {"data": "..."},
    "timestamp": 1234567890.0
}
```

#### `text_chunk`
Progressive text response (for streaming UI)
```python
{
    "type": "text_chunk",
    "content": "Based on the results, ",
    "timestamp": 1234567890.0
}
```

#### `done`
Conversation turn complete
```python
{
    "type": "done",
    "final_message": {
        "role": "assistant",
        "content": "Full response text",
        "timestamp": 1234567890.0
    },
    "tool_calls": [...],
    "tool_results": [...],
    "timestamp": 1234567890.0
}
```

---

## Mode Comparison

| Feature | WorkflowMode | ChatMode |
|---------|--------------|----------|
| **Interface** | Task-oriented | Conversational |
| **Input** | Goal/Task | User message + history |
| **Output** | Complete result | Streaming events |
| **Use Case** | Background jobs | Interactive UI |
| **Message History** | ❌ | ✅ |
| **Progressive Display** | Optional | Default |
| **Tool Visibility** | Hidden | Visible |

### When to Use Each

**Use WorkflowMode when:**
- Running background tasks
- Building data pipelines
- Generating reports/documents
- Batch processing
- No user interaction needed

**Use ChatMode when:**
- User-facing chat interface
- Real-time interaction required
- Progressive response display
- Tool execution should be visible
- Message history matters

---

## Advanced Examples

### Workflow → Chat Hybrid

Use workflow internally, expose as chat externally:

```python
class SmartChat:
    def __init__(self, conductor):
        self.workflow = create_workflow(conductor, mode="dynamic")
        self.chat = create_chat(conductor, mode="dynamic")

    async def handle_message(self, message, history):
        # Simple questions: Direct chat
        if self.is_simple_question(message):
            async for event in self.chat.stream(message, history):
                yield event

        # Complex tasks: Use workflow, stream as chat events
        else:
            result = await self.workflow.execute(goal=message)
            yield {"type": "text_chunk", "content": result["final_output"]}
            yield {"type": "done", "final_message": {...}}
```

### Multi-Turn Chat with Tool Calling

```python
chat = create_chat(conductor, agent_id="assistant")

conversation = []

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    conversation.append(ChatMessage(role="user", content=user_input))

    print("Assistant: ", end="", flush=True)
    full_response = ""

    async for event in chat.stream(message=user_input, history=conversation):
        if event["type"] == "tool_call":
            print(f"\n[Calling {event['tool']}...]", end="")

        elif event["type"] == "text_chunk":
            print(event["content"], end="", flush=True)
            full_response += event["content"]

        elif event["type"] == "done":
            print()
            conversation.append(ChatMessage(
                role="assistant",
                content=full_response
            ))
```

---

## API Reference

### `create_workflow(conductor, mode="dynamic", agent_order=None)`
Create workflow mode instance.

**Parameters:**
- `conductor` (Conductor): Jotty conductor instance
- `mode` (str): "dynamic" or "static"
- `agent_order` (List[str], optional): Required for static mode

**Returns:** WorkflowMode instance

### `create_chat(conductor, agent_id=None, mode="dynamic")`
Create chat mode instance.

**Parameters:**
- `conductor` (Conductor): Jotty conductor instance
- `agent_id` (str, optional): Specific agent for single-agent chat
- `mode` (str): "dynamic" or "static" (for multi-agent)

**Returns:** ChatMode instance

### `ChatMessage(role, content, timestamp=None)`
Structured chat message.

**Fields:**
- `role` (str): "user", "assistant", "system", or "tool"
- `content` (str): Message content
- `timestamp` (float, optional): Unix timestamp (auto-generated if not provided)

---

## Integration with JustJot.ai

JustJot.ai uses ChatMode for interactive note-taking conversations:

```typescript
// TypeScript client (src/lib/ai/chat-orchestrator.ts)
export async function* streamChatWithJotty(
  message: string,
  conversationHistory: Message[]
) {
  const response = await fetch(`${SUPERVISOR_URL}/api/dspy/chat/stream`, {
    method: 'POST',
    body: JSON.stringify({
      message,
      history: conversationHistory,
      mode: 'dynamic'
    })
  });

  // Parse SSE stream and yield events
  for await (const event of parseSSE(response.body)) {
    yield event;
  }
}
```

See `/docs/architecture/JOTTY_CHAT_INTEGRATION_STRATEGY.md` for full integration details.

---

## FAQ

**Q: Can I switch modes at runtime?**
A: Yes, create new instances as needed. They're lightweight wrappers.

**Q: What's the performance difference?**
A: ChatMode adds ~10ms overhead for event transformation. Workflow is slightly faster for batch tasks.

**Q: Can ChatMode use static workflows?**
A: Yes! Set `mode="static"` and provide `agent_order`.

**Q: How does multi-agent chat work?**
A: Agents collaborate via workflow underneath, events are transformed to chat format.

**Q: Can I customize event transformation?**
A: Yes, subclass `ChatMode` and override `_transform_to_chat_events()`.

---

## See Also

- [Conductor Documentation](./CONDUCTOR.md)
- [LangGraph Integration](./LANGGRAPH_INTEGRATION.md)
- [DSPy MCP Agents](./DSPY_MCP_AGENTS.md)
- [Example Code](../examples/execution_modes_examples.py)
