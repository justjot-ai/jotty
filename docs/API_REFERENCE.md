# Jotty API Reference

Complete API documentation for all subsystems.

---

## Table of Contents

1. [Memory System API](#memory-system-api)
2. [Learning System API](#learning-system-api)
3. [Context Management API](#context-management-api)
4. [Skills Registry API](#skills-registry-api)
5. [Orchestration API](#orchestration-api)
6. [Utilities API](#utilities-api)

---

## Memory System API

### `get_memory_system()`

Get the singleton memory system instance.

**Returns:** `MemorySystem`

**Example:**
```python
from Jotty.core.memory import get_memory_system

memory = get_memory_system()
```

---

### `MemorySystem.store()`

Store a memory at a specific level.

**Signature:**
```python
def store(
    content: str,
    level: str = "episodic",
    goal: str = "default",
    metadata: Dict[str, Any] = None
) -> str
```

**Parameters:**
- `content` (str): The memory content to store
- `level` (str): Memory level - one of: `"episodic"`, `"semantic"`, `"procedural"`, `"meta"`, `"causal"`
- `goal` (str): Associated goal or context
- `metadata` (Dict): Optional metadata (tags, timestamps, etc.)

**Returns:** `str` - Memory ID

**Example:**
```python
memory_id = memory.store(
    content="User prefers concise responses",
    level="meta",
    goal="communication",
    metadata={"source": "feedback", "confidence": 0.9}
)
```

**Memory Levels Explained:**
- **Episodic**: Specific events (fast decay, 3 days)
- **Semantic**: General knowledge (medium decay, 7 days)
- **Procedural**: How-to knowledge (medium decay, 7 days)
- **Meta**: Learning insights (no decay, permanent)
- **Causal**: Deep understanding (no decay, permanent)

---

### `MemorySystem.retrieve()`

Retrieve memories relevant to a query.

**Signature:**
```python
def retrieve(
    query: str,
    goal: str = "default",
    top_k: int = 5,
    level: str = None
) -> List[MemoryResult]
```

**Parameters:**
- `query` (str): Search query
- `goal` (str): Filter by goal/context
- `top_k` (int): Number of results to return
- `level` (str, optional): Filter by specific memory level

**Returns:** `List[MemoryResult]` - List of memory results

**MemoryResult Fields:**
- `content` (str): The memory content
- `level` (str): Memory level
- `relevance` (float): Relevance score (0-1)
- `memory_id` (str): Unique identifier
- `metadata` (Dict): Associated metadata

**Example:**
```python
results = memory.retrieve(
    query="How does the user like responses formatted?",
    goal="communication",
    top_k=3
)

for result in results:
    print(f"[{result.level}] {result.content}")
    print(f"Relevance: {result.relevance:.2f}")
```

---

### `MemorySystem.status()`

Get memory system status.

**Returns:** `Dict[str, Any]` with keys:
- `backend` (str): Backend type ("full", "simple", etc.)
- `total_memories` (int): Total memories stored
- `operations` (Dict): Operation counts

**Example:**
```python
status = memory.status()
print(f"Total memories: {status['total_memories']}")
```

---

## Learning System API

### `get_td_lambda()`

Get the singleton TD-Lambda learner instance.

**Returns:** `TDLambdaLearner`

**Example:**
```python
from Jotty.core.learning import get_td_lambda

td = get_td_lambda()  # gamma=0.99, lambda=0.95
```

---

### `TDLambdaLearner.update()`

Update value estimates using TD-Lambda algorithm.

**Signature:**
```python
def update(
    state: Dict[str, Any],
    action: Dict[str, Any],
    reward: float,
    next_state: Dict[str, Any],
    done: bool = False
) -> None
```

**Parameters:**
- `state` (Dict): Current state representation
- `action` (Dict): Action taken
- `reward` (float): Immediate reward received
- `next_state` (Dict): Resulting state
- `done` (bool): Whether episode terminated

**Example:**
```python
# Record a successful web search
td.update(
    state={"task": "research", "step": 1},
    action={"tool": "web-search", "query": "AI trends"},
    reward=0.8,  # Partial success
    next_state={"task": "research", "step": 2}
)

# Final success
td.update(
    state={"task": "research", "step": 3},
    action={"tool": "summarize"},
    reward=1.0,  # Full success
    next_state={"task": "research", "step": 4},
    done=True
)
```

**What Happens:**
- Eligibility traces updated (credit assignment)
- Value estimates adjusted for all states in trajectory
- Earlier actions get credit for final success (temporal credit assignment)

---

### `get_credit_assigner()`

Get the credit assignment system (Shapley values).

**Returns:** `ReasoningCreditAssigner`

**Example:**
```python
from Jotty.core.learning import get_credit_assigner

credit = get_credit_assigner()
```

---

## Context Management API

### `get_context_manager()`

Get a context manager instance.

**Signature:**
```python
def get_context_manager(
    max_tokens: int = 28000,
    safety_margin: float = 0.85
) -> SmartContextManager
```

**Parameters:**
- `max_tokens` (int): Maximum token limit
- `safety_margin` (float): Use this fraction of max (0.85 = 85%)

**Returns:** `SmartContextManager`

**Example:**
```python
from Jotty.core.context import get_context_manager

ctx = get_context_manager(max_tokens=16000, safety_margin=0.90)
```

---

### `SmartContextManager.register_todo()`

Register TODO list for preservation (never truncated).

**Signature:**
```python
def register_todo(self, todo_content: str) -> None
```

**Example:**
```python
ctx.register_todo("Complete Q1 report by March 1")
```

---

### `SmartContextManager.register_goal()`

Register current goal for preservation.

**Signature:**
```python
def register_goal(self, goal: str) -> None
```

**Example:**
```python
ctx.register_goal("Create comprehensive data analysis")
```

---

### `SmartContextManager.add_chunk()`

Add a context chunk with priority.

**Signature:**
```python
def add_chunk(
    self,
    content: str,
    category: str,
    priority: ContextPriority = None
) -> None
```

**Parameters:**
- `content` (str): Chunk content
- `category` (str): Chunk category ("task", "memory", "trajectory", etc.)
- `priority` (ContextPriority): Priority level (CRITICAL, HIGH, MEDIUM, LOW)

**Example:**
```python
from Jotty.core.context.context_manager import ContextPriority

ctx.add_chunk(
    content="Recent analysis shows 15% growth",
    category="analysis",
    priority=ContextPriority.HIGH
)
```

---

### `SmartContextManager.build_context()`

Build final context that fits within token limits.

**Signature:**
```python
def build_context(
    self,
    system_prompt: str,
    user_input: str,
    additional_context: Dict[str, str] = None
) -> Dict[str, Any]
```

**Returns:** `Dict` with keys:
- `system_prompt` (str): System prompt (unchanged)
- `user_input` (str): User input (unchanged)
- `context` (str): Built context string
- `truncated` (bool): Whether truncation occurred
- `preserved` (Dict): What was preserved
- `stats` (Dict): Token usage statistics

**Example:**
```python
result = ctx.build_context(
    system_prompt="You are a data analyst",
    user_input="Analyze sales trends"
)

print(f"Truncated: {result['truncated']}")
print(f"Total tokens: {result['stats']['total_tokens']}")
print(f"Budget remaining: {result['stats']['budget_remaining']}")
```

---

## Skills Registry API

### `get_registry()`

Get the unified skills registry.

**Returns:** `UnifiedRegistry`

**Example:**
```python
from Jotty.core.skills import get_registry

registry = get_registry()
```

---

### `UnifiedRegistry.list_skills()`

List all available skills.

**Returns:** `List[Dict[str, Any]]` - List of skill metadata

**Example:**
```python
skills = registry.list_skills()
print(f"Total skills: {len(skills)}")

for skill in skills[:5]:
    print(f"- {skill['name']}: {skill['description']}")
```

---

### `UnifiedRegistry.discover_for_task()`

Discover skills relevant to a task.

**Signature:**
```python
def discover_for_task(
    self,
    task: str,
    max_skills: int = 10
) -> Dict[str, List[Dict]]
```

**Parameters:**
- `task` (str): Task description
- `max_skills` (int): Maximum skills to return

**Returns:** `Dict` with keys:
- `skills` (List[Dict]): Relevant skills
- `ui` (List[Dict]): Relevant UI components

**Example:**
```python
discovery = registry.discover_for_task(
    task="analyze data and create visualizations",
    max_skills=5
)

print(f"Found {len(discovery['skills'])} skills:")
for skill in discovery['skills']:
    print(f"- {skill['name']}")
```

---

### `UnifiedRegistry.get_claude_tools()`

Convert skills to Claude tool format.

**Signature:**
```python
def get_claude_tools(
    self,
    skill_names: List[str]
) -> List[Dict[str, Any]]
```

**Parameters:**
- `skill_names` (List[str]): List of skill names to convert

**Returns:** `List[Dict]` - Claude-compatible tool definitions

**Example:**
```python
tools = registry.get_claude_tools(['web-search', 'calculator'])

# Use with Claude API
import anthropic
client = anthropic.Anthropic()
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": "Search for AI news"}],
    tools=tools
)
```

---

## Orchestration API

### `Orchestrator`

Create and manage agent swarms.

**Constructor:**
```python
def __init__(
    self,
    agents: str = None,
    config: SwarmConfig = None
)
```

**Parameters:**
- `agents` (str): Natural language description of agents (e.g., "Researcher + Writer")
- `config` (SwarmConfig, optional): Advanced configuration

**Example:**
```python
from Jotty.core.orchestration import Orchestrator

swarm = Orchestrator(agents="Researcher + Analyst + Writer")
```

---

### `Orchestrator.run()`

Execute a task with the swarm.

**Signature:**
```python
async def run(
    self,
    goal: str,
    config: ExecutionConfig = None
) -> ExecutionResult
```

**Parameters:**
- `goal` (str): Task description
- `config` (ExecutionConfig, optional): Execution configuration

**Returns:** `ExecutionResult` with fields:
- `success` (bool): Whether execution succeeded
- `output` (str): Task output
- `latency_ms` (float): Execution time in milliseconds
- `llm_calls` (int): Number of LLM calls made
- `cost_usd` (float): Estimated cost in USD
- `error` (str, optional): Error message if failed

**Example:**
```python
result = await swarm.run(
    goal="Research top 3 AI trends and create summary"
)

print(f"Success: {result.success}")
print(f"Duration: {result.latency_ms / 1000:.2f}s")
print(f"Cost: ${result.cost_usd:.4f}")
print(f"\nOutput:\n{result.output}")
```

---

## Utilities API

### `get_budget_tracker()`

Get the budget tracking system.

**Returns:** `BudgetTracker`

**Example:**
```python
from Jotty.core.utils import get_budget_tracker

tracker = get_budget_tracker()
```

---

### `BudgetTracker.record_call()`

Record an LLM API call for cost tracking.

**Signature:**
```python
def record_call(
    self,
    agent_name: str,
    tokens_input: int,
    tokens_output: int,
    model: str = "gpt-4"
) -> None
```

**Example:**
```python
tracker.record_call(
    agent_name="researcher",
    tokens_input=1000,
    tokens_output=500,
    model="gpt-4o"
)
```

---

### `BudgetTracker.get_usage()`

Get usage statistics.

**Returns:** `Dict[str, Any]` with keys:
- `calls` (int): Total API calls
- `tokens_input` (int): Total input tokens
- `tokens_output` (int): Total output tokens
- `estimated_cost_usd` (float): Estimated cost

**Example:**
```python
usage = tracker.get_usage()
print(f"Total calls: {usage['calls']}")
print(f"Total cost: ${usage['estimated_cost_usd']:.4f}")
```

---

### `get_llm_cache()`

Get the LLM response cache.

**Returns:** `LLMCallCache`

**Example:**
```python
from Jotty.core.utils import get_llm_cache

cache = get_llm_cache()
```

---

### `LLMCallCache.get()`

Get cached response.

**Signature:**
```python
def get(self, key: str) -> Optional[CachedResponse]
```

**Returns:** `CachedResponse` or `None`

**CachedResponse Fields:**
- `response` (Dict): The cached response
- `timestamp` (float): When cached
- `hits` (int): Cache hit count

**Example:**
```python
cached = cache.get("prompt-hash-123")
if cached:
    print(f"Cache hit! Response: {cached.response}")
else:
    # Make actual LLM call
    response = make_llm_call()
    cache.set("prompt-hash-123", response)
```

---

### `get_tokenizer()`

Get the smart tokenizer.

**Returns:** `SmartTokenizer`

**Example:**
```python
from Jotty.core.utils import get_tokenizer

tokenizer = get_tokenizer()
count = tokenizer.count_tokens("Hello, world!")
print(f"Tokens: {count}")
```

---

## Complete Example: Multi-Agent Research Task

Putting it all together:

```python
import asyncio
from Jotty.core.orchestration import Orchestrator
from Jotty.core.memory import get_memory_system
from Jotty.core.learning import get_td_lambda
from Jotty.core.utils import get_budget_tracker

async def main():
    # Initialize subsystems
    memory = get_memory_system()
    td_lambda = get_td_lambda()
    tracker = get_budget_tracker()

    # Store context
    memory.store(
        content="Focus on practical applications, not theory",
        level="meta",
        goal="research"
    )

    # Create swarm
    swarm = Orchestrator(agents="Researcher + Analyst + Writer")

    # Execute task
    result = await swarm.run(
        goal="Research top 5 AI trends in healthcare and create summary"
    )

    # Check results
    if result.success:
        print("✅ Success!")
        print(f"\nOutput:\n{result.output}")
        print(f"\nStats:")
        print(f"- Duration: {result.latency_ms / 1000:.2f}s")
        print(f"- LLM calls: {result.llm_calls}")
        print(f"- Cost: ${result.cost_usd:.4f}")

        # Learning automatically happens
        # Next similar task will be faster and smarter!
    else:
        print(f"❌ Failed: {result.error}")

    # Check budget
    usage = tracker.get_usage()
    print(f"\nTotal budget used: ${usage['estimated_cost_usd']:.4f}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Error Handling

All API calls may raise `JottyError` or its subclasses:

```python
from Jotty.core.foundation.exceptions import (
    JottyError,
    ExecutionError,
    LLMError,
    ConfigurationError
)

try:
    result = await swarm.run(goal="Complex task")
except LLMError as e:
    print(f"LLM API error: {e}")
except ExecutionError as e:
    print(f"Execution failed: {e}")
except JottyError as e:
    print(f"General error: {e}")
```

---

## Configuration

### SwarmConfig

Advanced swarm configuration:

```python
from Jotty.core.foundation.data_structures import SwarmConfig

config = SwarmConfig(
    alpha=0.1,              # Learning rate
    gamma=0.99,             # Discount factor
    max_tokens=16000,       # Context window
    enable_learning=True    # Enable RL learning
)

swarm = Orchestrator(agents="Team", config=config)
```

---

## Type Definitions

### MemoryResult

```python
@dataclass
class MemoryResult:
    content: str
    level: str
    relevance: float
    memory_id: str
    metadata: Dict[str, Any]
```

### ExecutionResult

```python
@dataclass
class ExecutionResult:
    success: bool
    output: str
    latency_ms: float
    llm_calls: int
    cost_usd: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

---

For more details, see:
- **Getting Started**: `docs/GETTING_STARTED.md`
- **Architecture**: `docs/JOTTY_ARCHITECTURE.md`
- **Examples**: `examples/` directory
