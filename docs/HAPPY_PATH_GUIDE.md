# Jotty Happy Path Guide

**Goal:** Understand the complete flow from "user sends message via Telegram" to "response back"

**Date:** February 2025
**Audience:** New contributors and developers
**Prerequisite:** Read `JOTTY_ARCHITECTURE.md` first for context

---

## Quick Summary

```
User (Telegram) → Webhook → UnifiedGateway → ChannelRouter → ModeRouter
→ ChatExecutor → Skills → Response → Responders → Telegram → User
```

**Time to trace:** ~5 minutes
**Files touched:** 6 core files
**Layers crossed:** All 5 (Interface → Modes → Registry → Brain → Persistence)

---

## The Complete Flow (Step-by-Step)

### 📱 Step 1: User Sends Message

**What happens:**
- User opens Telegram and sends: `"Research AI trends"`
- Telegram servers POST to your webhook: `https://your-server.com/webhook/telegram`

**Telegram payload example:**
```json
{
  "message": {
    "message_id": 123,
    "from": {"id": 789, "first_name": "Alice"},
    "chat": {"id": 456},
    "text": "Research AI trends"
  }
}
```

---

### 🌐 Step 2: UnifiedGateway Receives Webhook

**File:** `Jotty/cli/gateway/server.py`
**Class:** `UnifiedGateway`
**Method:** `telegram_webhook()` (FastAPI endpoint)

**What it does:**
1. Receives POST at `/webhook/telegram`
2. Extracts message data from Telegram format
3. Creates a `MessageEvent` object
4. Hands off to `ChannelRouter`

**Code location:** Lines 404-441 in `server.py`

```python
# Create MessageEvent from Telegram data
event = MessageEvent(
    channel=ChannelType.TELEGRAM,
    channel_id=str(chat.get("id")),     # "456"
    user_id=str(user.get("id")),        # "789"
    user_name=user.get("first_name"),   # "Alice"
    content=text,                        # "Research AI trends"
    message_id=str(message.get("message_id")),
    raw_data=data
)

# Process async (non-blocking)
asyncio.create_task(self.router.handle_message(event))
return {"ok": True}  # Telegram expects immediate response
```

**Key classes used:**
- `MessageEvent` (dataclass from `channels.py`)
- `ChannelType.TELEGRAM` (enum from `sdk_types.py`)

---

### 🔀 Step 3: ChannelRouter Processes Message

**File:** `Jotty/cli/gateway/channels.py`
**Class:** `ChannelRouter`
**Method:** `handle_message(event: MessageEvent)`

**What it does:**
1. **Trust check** (if enabled): Verifies user is authorized
2. **Session management**: Retrieves or creates session with history
3. **Execution context**: Builds `ExecutionContext` with metadata
4. **Routing**: Calls `_process_with_jotty()` → hands to ModeRouter
5. **Response handling**: Sends response back via responders

**Code location:** Lines 129-254 in `channels.py`

**Key logic:**
```python
# Get or create persistent session
session_manager = self._get_session_manager()
sdk_session = await session_manager.get_or_create(
    user_id=event.user_id,
    channel=event.channel,
    channel_id=event.channel_id,
    user_name=event.user_name
)

# Add message to history
sdk_session.add_message("user", event.content, {...})

# Create execution context
exec_context = ExecutionContext(
    mode=ExecutionMode.CHAT,
    channel=event.channel,
    session_id=sdk_session.session_id,
    user_id=event.user_id,
    user_name=event.user_name,
)

# Process with ModeRouter (canonical path)
response_text = await self._process_with_jotty(event, session_data, exec_context)

# Send response back
await self._send_response(ResponseEvent(
    channel=event.channel,
    channel_id=event.channel_id,
    content=response_text,
    reply_to=event.message_id
))
```

**Key classes used:**
- `PersistentSessionManager` (from `sessions.py`)
- `ExecutionContext` (from `sdk_types.py`)
- `ModeRouter` (next step!)

---

### 🎯 Step 4: ModeRouter Routes to Execution Mode

**File:** `Jotty/core/api/mode_router.py`
**Class:** `ModeRouter`
**Method:** `chat(message: str, context: ExecutionContext)`

**What it does:**
1. Determines execution mode from context (CHAT in our case)
2. Routes to appropriate handler:
   - **CHAT** → `ChatExecutor` (TierExecutor with native LLM tool calling)
   - **WORKFLOW** → `AutoAgent` (multi-step agent)
   - **SKILL** → Direct skill execution
   - **AGENT** → Specific agent
3. Returns `RouteResult` with output

**Code location:** Lines 161-230 in `mode_router.py`

**Key method:**
```python
async def chat(
    self,
    message: str,
    context: ExecutionContext
) -> RouteResult:
    """Handle CHAT mode execution."""

    # Get ChatExecutor with callbacks
    executor = self._get_executor(context)

    # Add conversation history from context
    history = context.metadata.get("conversation_history", [])

    # Execute with native LLM tool calling
    result = await executor.chat(
        message,
        history=history,
        session_id=context.session_id
    )

    # Build RouteResult
    return RouteResult(
        success=True,
        content=result.response,
        mode=ExecutionMode.CHAT,
        skills_used=result.skills_used,
        execution_time=result.execution_time
    )
```

**Key classes used:**
- `ChatExecutor` (from `unified_executor.py`)
- `RouteResult` (dataclass, contains output)
- `ExecutionMode.CHAT` (enum)

---

### 🧠 Step 5: ChatExecutor Executes with Skills

**File:** `Jotty/core/orchestration/unified_executor.py`
**Class:** `ChatExecutor`
**Method:** `chat(message: str, history: List, session_id: str)`

**What it does:**
1. Configures LLM with tool calling (Claude/OpenAI native tools)
2. Loads available skills from `UnifiedRegistry`
3. Converts skills to Claude/OpenAI tool format
4. Calls LLM with tools
5. Executes tool calls (skills) if requested
6. Returns final response

**Simplified flow:**
```python
# Get skills from registry
from Jotty.core.registry import get_unified_registry
registry = get_unified_registry()
skills = registry.list_skills()  # 273 skills

# Convert to Claude tools format
tools = registry.get_claude_tools(skills)

# Call LLM with tools
from Jotty.core.foundation.unified_lm_provider import get_native_provider
provider = get_native_provider()  # Returns AnthropicProvider/OpenAIProvider

response = await provider.chat(
    messages=history + [{"role": "user", "content": message}],
    tools=tools,
    model="claude-3-5-sonnet-20241022"
)

# If LLM requests tool use (e.g., "web-search")
if response.tool_calls:
    for tool_call in response.tool_calls:
        skill = registry.get_skill(tool_call.name)
        result = await skill.execute(tool_call.params)
        # Feed result back to LLM

return ChatResult(response=final_text, skills_used=[...])
```

**Key classes used:**
- `UnifiedRegistry` (from `core/registry/unified_registry.py`)
- `AnthropicProvider` / `OpenAIProvider` (from `unified_lm_provider.py`)
- Individual skills (from `Jotty/skills/`)

**Example skill execution:**
If LLM calls `web-search` skill:
```python
# Registry executes skill
skill = registry.get_skill("web-search")
result = skill.tools[0].function({
    "query": "AI trends 2025",
    "max_results": 5
})
# Returns: {"results": [...], "summary": "..."}
```

---

### 💾 Step 6: Memory & Learning (Automatic)

**Happens in parallel** (if swarm is used):

**Memory:**
- `SwarmMemory` stores task outcome in 5-level memory
- Uses `BrainInspiredMemoryManager` for consolidation
- Persists to `~/jotty/intelligence/memory/`

**Learning:**
- `TDLambdaLearner` updates value estimates
- `ReasoningCreditAssigner` assigns credit to decisions
- Persists to `~/jotty/intelligence/learning/`

**Files:**
- `core/memory/cortex.py` - SwarmMemory
- `core/learning/td_lambda.py` - TDLambdaLearner

**Automatic hooks:**
- `BaseSwarm._pre_execute_learning()` - Load context
- `BaseSwarm._post_execute_learning()` - Store results

---

### 📤 Step 7: Response Flows Back

**Path:** `ChatExecutor` → `ModeRouter` → `ChannelRouter` → `ChannelResponderRegistry` → `telegram-sender` skill → Telegram API → User

**File:** `Jotty/cli/gateway/responders.py`
**Class:** `ChannelResponderRegistry`
**Method:** `send(response: ResponseEvent)`

**What it does:**
1. Gets responder for channel (Telegram)
2. Discovers `telegram-sender` skill from registry
3. Calls skill's `send_telegram_message_tool()`
4. Formats message for Telegram Markdown
5. POSTs to Telegram API

**Code:**
```python
# ChannelRouter sends response
await self._send_response(ResponseEvent(
    channel=ChannelType.TELEGRAM,
    channel_id="456",              # Chat ID
    content=response_text,         # "Here are the AI trends..."
    reply_to="123"                 # Original message ID
))

# ChannelResponderRegistry handles sending
registry = get_responder_registry()
await registry.send(response_event)

# Internally discovers telegram-sender skill
tool = registry._discover_skill("telegram-sender")
tool({
    "chat_id": "456",
    "message": "Here are the AI trends:\n\n1. ..."
})
```

**Telegram receives:**
```json
POST https://api.telegram.org/bot{TOKEN}/sendMessage
{
  "chat_id": "456",
  "text": "Here are the AI trends:\n\n1. ...",
  "reply_to_message_id": 123
}
```

**User sees response in Telegram!** ✅

---

## Files Touched (In Order)

| # | File | Class | Purpose |
|---|------|-------|---------|
| 1 | `cli/gateway/server.py` | `UnifiedGateway` | Receives webhook |
| 2 | `cli/gateway/channels.py` | `ChannelRouter` | Routes message + manages session |
| 3 | `core/api/mode_router.py` | `ModeRouter` | Routes to execution mode |
| 4 | `core/orchestration/unified_executor.py` | `ChatExecutor` | Executes with LLM + tools |
| 5 | `core/registry/unified_registry.py` | `UnifiedRegistry` | Provides skills (273) |
| 6 | `cli/gateway/responders.py` | `ChannelResponderRegistry` | Sends response back |

---

## Subsystem Facades Used

For each subsystem, here's which facade you'd use if building/debugging:

```python
# MEMORY - Session history and long-term memory
from Jotty.core.memory import get_memory_system, get_brain_manager
memory = get_memory_system()
brain = get_brain_manager()

# SKILLS - Access all 273 skills
from Jotty.core.skills import get_registry, list_skills
registry = get_registry()
skills = list_skills()

# CONTEXT - Token management
from Jotty.core.context import get_context_manager
ctx_mgr = get_context_manager()

# LEARNING - RL updates (for swarms)
from Jotty.core.learning import get_td_lambda
td = get_td_lambda()

# ORCHESTRATION - Swarm routing
from Jotty.core.orchestration import get_swarm_router
router = get_swarm_router()

# UTILITIES - Budget tracking
from Jotty.core.utils import get_budget_tracker
budget = get_budget_tracker()
```

---

## Configuration (SwarmLearningConfig vs SwarmConfig)

**IMPORTANT:** Since the config refactoring, there are now TWO configs:

### For Learning/RL/Orchestration (175 fields)
```python
from Jotty.core.foundation.data_structures import SwarmLearningConfig

config = SwarmLearningConfig(
    enable_rl=True,
    gamma=0.99,
    learning_rate=0.01,
    max_context_tokens=28000,
    # ... 170+ more fields for TD-Lambda, memory, budget, etc.
)
```

### For Domain Swarm Metadata (12 fields)
```python
from Jotty.core.swarms.swarm_types import SwarmConfig

config = SwarmConfig(
    name="CodingSwarm",
    domain="coding",
    version="1.0.0",
    # ... 9 more basic fields
)
```

**Rule of thumb:**
- Building a **swarm** with agents? → Use `SwarmConfig` (swarm_types)
- Configuring **learning/RL/orchestration**? → Use `SwarmLearningConfig` (data_structures)

---

## Common Variations

### Slack Instead of Telegram

**Only changes:**
- Webhook endpoint: `/webhook/slack` (line 444 in `server.py`)
- Responder: Slack responder (line 138 in `responders.py`)

**Everything else is identical!** Same flow through ModeRouter, ChatExecutor, etc.

### Workflow Mode Instead of Chat

**Changes at Step 4:**
```python
# ModeRouter routes to AutoAgent instead of ChatExecutor
context.mode = ExecutionMode.WORKFLOW

result = await router.workflow(
    goal="Research AI trends and create a report",
    context=context
)

# AutoAgent breaks into steps:
# 1. Research AI trends (uses web-search skill)
# 2. Analyze results
# 3. Generate report (uses document-generator skill)
# 4. Return final output
```

**File:** `core/agents/auto_agent.py`
**Execution:** Multi-step with planning

---

## Testing the Happy Path

### 1. Unit Test (Mock Everything)

```python
import pytest
from Jotty.cli.gateway.channels import ChannelRouter, MessageEvent, ChannelType

@pytest.mark.unit
@pytest.mark.asyncio
async def test_happy_path_telegram():
    """Test complete flow from Telegram message to response."""

    # Create router
    router = ChannelRouter()

    # Mock ModeRouter
    from unittest.mock import AsyncMock, patch
    with patch('Jotty.core.api.mode_router.get_mode_router') as mock_router:
        mock_router.return_value.chat = AsyncMock(
            return_value=RouteResult(
                success=True,
                content="Here are the AI trends...",
                mode=ExecutionMode.CHAT
            )
        )

        # Create message event
        event = MessageEvent(
            channel=ChannelType.TELEGRAM,
            channel_id="456",
            user_id="789",
            user_name="Alice",
            content="Research AI trends"
        )

        # Process
        response = await router.handle_message(event)

        # Verify
        assert "AI trends" in response
        mock_router.return_value.chat.assert_called_once()
```

### 2. Integration Test (Real LLM, Mock Telegram)

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_happy_path_real_llm():
    """Test with real LLM but mocked Telegram."""

    from Jotty.core.api.mode_router import get_mode_router
    from Jotty.core.foundation.types.sdk_types import ExecutionContext, ExecutionMode

    router = get_mode_router()
    context = ExecutionContext(
        mode=ExecutionMode.CHAT,
        channel=ChannelType.TELEGRAM,
        session_id="test"
    )

    result = await router.chat("What is 2+2?", context)

    assert result.success
    assert "4" in result.content.lower()
```

### 3. E2E Test (Manual)

```bash
# 1. Start gateway
python Jotty/cli/gateway/server.py

# 2. Send test message via curl (simulates Telegram)
curl -X POST http://localhost:8766/webhook/telegram \
  -H "Content-Type: application/json" \
  -d '{
    "message": {
      "message_id": 123,
      "from": {"id": 789, "first_name": "Test"},
      "chat": {"id": 456},
      "text": "Hello Jotty"
    }
  }'

# 3. Check logs for response
```

---

## Debugging Tips

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or specific loggers
logging.getLogger("Jotty.cli.gateway").setLevel(logging.DEBUG)
logging.getLogger("Jotty.core.api.mode_router").setLevel(logging.DEBUG)
```

### Trace Execution

Add prints at each step:

```python
# In ChannelRouter.handle_message()
print(f"[TRACE] Received: {event.content}")

# In ModeRouter.chat()
print(f"[TRACE] Routing to CHAT mode")

# In ChatExecutor.chat()
print(f"[TRACE] Executing with {len(tools)} tools")
```

### Check Session State

```python
from Jotty.cli.gateway.sessions import get_session_manager
manager = get_session_manager()

# List active sessions
sessions = manager.list_active()
print(f"Active sessions: {len(sessions)}")

# Get specific session
session = await manager.get_or_create("user_789", ...)
print(f"History: {session.get_history(10)}")
```

### Inspect Registry

```python
from Jotty.core.registry import get_unified_registry
registry = get_unified_registry()

# List all skills
skills = registry.list_skills()
print(f"Total skills: {len(skills)}")  # Should be 273

# Get specific skill
skill = registry.get_skill("web-search")
print(f"Skill: {skill.name}, Tools: {len(skill.tools)}")
```

---

## Performance Characteristics

| Stage | Typical Time | Notes |
|-------|--------------|-------|
| Webhook receive | < 10ms | FastAPI + async |
| Trust check | < 5ms | In-memory lookup |
| Session lookup | 10-50ms | Redis/MongoDB or memory |
| ModeRouter routing | < 5ms | Simple dispatch |
| ChatExecutor (LLM call) | 2-8s | Depends on model |
| Skill execution | 0.1-5s | Varies by skill |
| Response send | 100-500ms | Telegram API latency |
| **Total (typical)** | **3-15s** | End-to-end |

**Bottlenecks:**
- LLM API latency (70-90% of total time)
- External skill calls (web-search, API calls)

**Optimizations:**
- Streaming responses (reduce perceived latency)
- Skill result caching (LLMCallCache)
- Parallel skill execution (when independent)

---

## Next Steps

After understanding the happy path:

1. **Add a new skill** → See `CONTRIBUTING.md` "Adding New Components"
2. **Add a new channel** → Model after Telegram webhook in `server.py`
3. **Create a new swarm** → Use `DomainSwarm` pattern
4. **Customize learning** → Adjust `SwarmLearningConfig` parameters
5. **Build SDK client** → Use `/api/chat` endpoint

---

## Summary

**The Happy Path in One Sentence:**

> User sends message via Telegram → `UnifiedGateway` webhook → `ChannelRouter` with session → `ModeRouter` dispatch → `ChatExecutor` with LLM+tools → Skills execute → Response via `ChannelResponderRegistry` → Back to user.

**Key Takeaways:**
1. **UnifiedGateway** handles ALL external channels (Telegram, Slack, Discord, WhatsApp, WebSocket)
2. **ModeRouter** is the canonical execution path (not JottyCLI)
3. **ChatExecutor** uses native LLM tool calling (Claude/OpenAI tools format)
4. **UnifiedRegistry** provides all 273 skills
5. **ExecutionContext** carries metadata through the entire flow
6. **Sessions persist** across conversations (Redis/MongoDB backend)

**Most important files to understand:**
- `cli/gateway/server.py` - Entry point
- `cli/gateway/channels.py` - Message routing
- `core/api/mode_router.py` - Execution dispatch
- `core/orchestration/unified_executor.py` - LLM execution

---

**Questions?** Check `docs/JOTTY_ARCHITECTURE.md` or ask in GitHub Discussions.

**Want to contribute?** See `CONTRIBUTING.md` for workflow and test patterns.
