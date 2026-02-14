# Jotty - AI Agent Framework

## Quick Reference for Claude

**Main Architecture Doc:** `docs/JOTTY_ARCHITECTURE.md` - READ THIS FIRST

## Common Tasks → Swarms (START HERE!)

| Task | Swarm | Quick Example |
|------|-------|---------------|
| **Generate learning materials** (K-12, Olympiad) | `olympiad_learning_swarm` | `learn_topic("economics", "5th Grade Economics", "Student")` |
| **Research ArXiv papers** | `arxiv_learning_swarm` | Research academic papers and create summaries |
| **Write/review code** | `coding_swarm` | Generate, test, and review code |
| **Testing & QA** | `testing_swarm` | Create test suites and validate code |
| **Research topics** | `research_swarm` | Deep research with web search |
| **Data analysis** | `data_analysis_swarm` | Analyze datasets and create visualizations |
| **DevOps tasks** | `devops_swarm` | Deployment, monitoring, infrastructure |

### Educational Content Example
```python
from Jotty.core.swarms.olympiad_learning_swarm import learn_topic

# Generate comprehensive learning material with PDF + HTML
result = await learn_topic(
    subject="general",           # or "mathematics", "physics", etc.
    topic="Economics for 5th Grade",
    student_name="Student",
    depth="deep",               # quick/standard/deep/marathon
    target="foundation",        # foundation/intermediate/advanced/olympiad
    send_telegram=True          # Auto-send to Telegram
)
# ✅ Generates: PDF (A4, professional) + HTML (interactive)
# ✅ Includes: Concepts, patterns, problems, examples, real-life scenarios
```

## Start Here: Discovery API

**Don't know what Jotty can do? Start with `capabilities()`:**

```python
from Jotty import capabilities
caps = capabilities()
# Returns: execution_paths, subsystems, swarms, skills_count, providers, utilities

from Jotty.core.capabilities import explain
print(explain("memory"))    # human-readable description of any subsystem
print(explain("learning"))
```

## Three Execution Paths

```python
from Jotty import Jotty
j = Jotty()

# 1. CHAT — conversational AI
j.router                    # ModeRouter for chat/workflow routing

# 2. WORKFLOW — multi-step automation
j.chat_executor             # ChatExecutor (direct LLM tool-calling, no agents)

# 3. SWARM — multi-agent coordination
from Jotty.core.orchestration import Orchestrator
swarm = Orchestrator(agents="Research AI startups")
result = await swarm.run(goal="Research AI startups")
```

## Subsystem Facades (direct access to any subsystem)

```python
# MEMORY — 5-level brain-inspired memory
from Jotty.core.memory import get_memory_system, get_brain_manager, get_rag_retriever
memory = get_memory_system()          # zero-config entry point
brain = get_brain_manager()           # BrainInspiredMemoryManager
rag = get_rag_retriever()             # LLMRAGRetriever

# LEARNING — RL, credit assignment, cooperation
from Jotty.core.learning import get_td_lambda, get_credit_assigner, get_reward_manager
td = get_td_lambda()                  # TDLambdaLearner (gamma=0.99)
credit = get_credit_assigner()        # ReasoningCreditAssigner

# CONTEXT — token management, compression, overflow protection
from Jotty.core.context import get_context_manager, get_context_guard, get_content_gate

# SKILLS — 164 skills, 8 providers
from Jotty.core.skills import get_registry, list_providers, list_skills
registry = get_registry()             # UnifiedRegistry
providers = list_providers()          # [{name, description, installed}, ...]

# ORCHESTRATION — hidden components surfaced
from Jotty.core.orchestration import (
    get_swarm_intelligence,            # SwarmIntelligence
    get_paradigm_executor,             # relay/debate/refinement
    get_ensemble_manager,              # ensemble methods
    get_provider_manager,              # LLM provider rotation
    get_swarm_router,                  # task→swarm routing
)

# UTILITIES — cost tracking, fault tolerance, caching
from Jotty.core.utils import get_budget_tracker, get_circuit_breaker, get_llm_cache, get_tokenizer
budget = get_budget_tracker()          # track LLM costs
breaker = get_circuit_breaker("svc")   # fault tolerance
cache = get_llm_cache()                # response caching
tok = get_tokenizer()                  # accurate token counting
```

## Usage Examples (how to actually use returned objects)

### Memory — store, retrieve, check status
```python
from Jotty.core.memory import get_memory_system
mem = get_memory_system()

# Store (levels: episodic, semantic, procedural, meta, causal)
mem_id = mem.store("Task X succeeded with approach Y", level="episodic",
                   goal="research", metadata={"reward": 1.0})

# Retrieve (returns List[MemoryResult] with .content, .level, .relevance)
results = mem.retrieve("How to handle task X?", goal="research", top_k=5)
for r in results:
    print(f"[{r.level}] {r.content} (relevance={r.relevance})")

# Status
status = mem.status()  # {'backend': 'full', 'operations': {...}, 'total_memories': N}
```

### Learning — TD-Lambda updates
```python
from Jotty.core.learning import get_td_lambda
td = get_td_lambda()  # gamma=0.99, lambda_trace=0.95

td.update(
    state={"task": "research", "agent": "researcher"},
    action={"tool": "web-search"},
    reward=1.0,
    next_state={"task": "research", "agent": "researcher", "step": 2},
)
```

### Budget Tracking — record LLM costs
```python
from Jotty.core.utils import get_budget_tracker
bt = get_budget_tracker()

bt.record_call("researcher", tokens_input=1000, tokens_output=500, model="gpt-4o")
bt.record_call("coder", tokens_input=500, tokens_output=200, model="gpt-4o-mini")

usage = bt.get_usage()  # {'calls': 2, 'tokens_input': 1500, 'tokens_output': 700, ...}
```

### LLM Cache — cache and retrieve responses
```python
from Jotty.core.utils import get_llm_cache
cache = get_llm_cache()

cache.set("prompt-hash-123", {"answer": "cached response"})
hit = cache.get("prompt-hash-123")  # returns CachedResponse or None
if hit:
    print(hit.response["answer"])   # NOTE: .response attribute, not subscript

stats = cache.stats()  # CacheStats with .hits, .misses, .hit_rate
```

### Circuit Breaker — fault tolerance
```python
from Jotty.core.utils import get_circuit_breaker
cb = get_circuit_breaker("openai-api")

cb.record_success()           # keeps circuit CLOSED
cb.record_failure(Exception("timeout"))  # increments failure count
print(cb.state)               # CircuitState.CLOSED / OPEN / HALF_OPEN
```

### Tokenizer — count tokens
```python
from Jotty.core.utils import get_tokenizer
tok = get_tokenizer()
count = tok.count_tokens("Hello world, this is a test")  # int
```

### Context Management — build context within token limits
```python
from Jotty.core.context import get_context_manager
ctx = get_context_manager()  # max_tokens=28000

ctx.register_goal("Research AI startups")
ctx.register_critical_memory("Budget is $0.50 max")
ctx.add_chunk("Previous research findings...", category="research")

result = ctx.build_context(
    system_prompt="You are a research assistant",
    user_input="Find recent AI startup funding rounds",
)
```

### Swarm Intelligence — agent routing and tracking
```python
from Jotty.core.orchestration import get_swarm_intelligence
si = get_swarm_intelligence()

si.register_agent("researcher")
si.record_task_result("researcher", task_type="web_search", success=True,
                      execution_time=2.5)
best = si.get_best_agent_for_task("web_search", available_agents=["researcher", "coder"])
```

## Top-Level Imports (shortcuts)

```python
from Jotty import (
    capabilities,         # discovery API
    MemorySystem,         # memory
    BudgetTracker,        # cost tracking
    CircuitBreaker,       # fault tolerance
    LLMCallCache,         # caching
    SmartTokenizer,       # tokenization
    ChatExecutor,         # direct LLM tool-calling
    SwarmIntelligence,    # learning intelligence
    ParadigmExecutor,     # relay/debate/refinement
    EnsembleManager,      # ensemble methods
    ModelTierRouter,      # model routing
)
```

## Jotty Class Properties

```python
from Jotty import Jotty
j = Jotty()
j.capabilities()          # structured map of everything
j.router                   # ModeRouter (chat/workflow/agent)
j.chat_executor            # ChatExecutor (fast LLM tool-calling)
j.registry                 # UnifiedRegistry (164 skills, 16 UI components)
```

## Testing Requirements

**MANDATORY**: Every code change to Jotty MUST include corresponding tests.

### Rules
1. Every new method/class gets a unit test
2. Every bug fix gets a regression test proving the fix
3. Tests use mocks — NEVER call real LLM providers
4. Tests run fast (< 1s each) and offline
5. Run `pytest tests/test_v3_execution.py -v` before considering work done

### V3 Test Patterns
- Use the `v3_executor` fixture from conftest.py (pre-wired with all mocks)
- Use `v3_observability_helpers` fixture for `assert_metrics_recorded()`, `assert_trace_exists()`, `assert_cost_tracked()` helpers
- Class-based: `class TestMyFeature:` with `@pytest.mark.unit` + `@pytest.mark.asyncio`
- Mock provider returns: `{'content': '...', 'usage': {'input_tokens': N, 'output_tokens': N}}`

### Running Tests
```bash
pytest tests/test_v3_execution.py -v        # V3 tests
pytest tests/ -m unit                       # All unit tests
pytest tests/ -m "not requires_llm"         # All offline tests
```

## The Five Layers (Top to Bottom)

```
1. INTERFACE    Telegram | Slack | Discord | WhatsApp | Web | CLI | SDK
       ↓
2. MODES        Chat (ChatAssistant) | API (MCP) | Workflow (AutoAgent)
       ↓
3. REGISTRY     UnifiedRegistry → Skills (Hands) + UI (Eyes) + Memory
       ↓
4. BRAIN        Swarms → Agents → SwarmIntelligence → TD-Lambda
       ↓
5. PERSISTENCE  ~/jotty/intelligence/*.json | ~/jotty/skills/
```

## Key Entry Points

| Entry Point | Command | Purpose |
|-------------|---------|---------|
| **CLI Interactive** | `python -m Jotty.cli` | REPL with slash commands |
| **CLI Single** | `python -m Jotty.cli -c "task"` | One-off execution |
| **Web Gateway** | `python Jotty/web.py` | HTTP/WS server (port 8766) |
| **Gateway Only** | `python -m Jotty.cli.gateway` | Webhooks for Telegram/Slack/etc |

## Legacy Imports (still work, but prefer facades above)

```python
from Jotty.core.registry import get_unified_registry
from Jotty.core.agents import AutoAgent, ChatAssistant
from Jotty.core.swarms import BaseSwarm, CodingSwarm
from Jotty.core.learning import TDLambdaLearner
from Jotty.core.api import JottyAPI
from Jotty.core.foundation.exceptions import JottyError
from Jotty.core.swarms.swarm_types import SwarmBaseConfig  # NOT SwarmConfig
from Jotty.core.memory.cortex import SwarmMemory
```

## Directory Structure

```
Jotty/
├── cli/                    # CLI application
│   ├── app.py              # JottyCLI main class
│   ├── gateway/            # UnifiedGateway + ChannelRouter
│   │   ├── server.py       # FastAPI + WebSocket server
│   │   ├── channels.py     # ChannelRouter + ChannelType enum
│   │   └── trust.py        # TrustManager (auth)
│   ├── commands/           # Slash commands (/run, /swarm, etc.)
│   ├── config/             # CLI configuration
│   │   └── schema.py       # CLIConfig dataclass
│   └── repl/               # REPL engine
├── core/
│   ├── agents/             # Agent implementations
│   │   ├── base/           # BaseAgent, DomainAgent, MetaAgent, AutonomousAgent
│   │   ├── auto_agent.py   # AutoAgent (workflow mode)
│   │   └── chat_assistant.py
│   ├── swarms/             # Swarm implementations
│   │   ├── base_swarm.py   # BaseSwarm with learning hooks
│   │   ├── domain_swarm.py # DomainSwarm (AgentTeam)
│   │   └── specialized/    # CodingSwarm, TestingSwarm, etc.
│   ├── registry/           # The unified registry system
│   │   ├── unified_registry.py  # MAIN ENTRY: get_unified_registry()
│   │   ├── skills_registry.py   # Skills (Hands) - 126 skills
│   │   ├── ui_registry.py       # UI (Eyes) - 16 components
│   │   └── api.py               # Registry HTTP API
│   ├── api/                # Programmatic API layer
│   │   ├── unified.py      # JottyAPI (main entry)
│   │   ├── chat_api.py     # ChatAPI
│   │   ├── workflow_api.py # WorkflowAPI
│   │   └── openapi.py      # OpenAPI 3.0 spec generator
│   ├── foundation/         # Cross-cutting concerns
│   │   ├── exceptions.py   # 30+ exception types
│   │   ├── data_structures.py  # SwarmConfig, etc.
│   │   └── agent_config.py # AgentConfig
│   ├── learning/           # TD-Lambda, memory systems
│   ├── memory/             # SwarmMemory (5 levels)
│   ├── orchestration/      # Orchestrator, SwarmIntelligence
│   └── integration/        # MCP, Claude CLI LM
├── skills/                 # Skill definitions (loaded lazily)
├── sdk/                    # Generated client libraries
├── web.py                  # Web server entry point
└── docs/                   # Documentation
    └── JOTTY_ARCHITECTURE.md  # MAIN ARCHITECTURE DOC
```

## How Learning Works

1. **Pre-execution**: `BaseSwarm._pre_execute_learning()` loads context from memory
2. **Execution**: Swarm runs agents with skills
3. **Post-execution**: `BaseSwarm._post_execute_learning()` stores to memory, updates TD-Lambda
4. **Persistence**: `SwarmIntelligence.save()` writes to `~/jotty/intelligence/{swarm}_{domain}.json`
5. **Next run**: Learning auto-loads from disk

## Adding New Components

### New Skill
```python
# skills/my-skill/skill.yaml
name: my-skill
description: "What this skill does"
tools:
  - my_tool

# skills/my-skill/tools.py
def my_tool(params: dict) -> dict:
    return {"result": "..."}
```

### New Swarm
```python
from Jotty.core.swarms.swarm_types import SwarmBaseConfig

class MySwarm(DomainSwarm):
    def __init__(self):
        super().__init__(SwarmBaseConfig(name="MySwarm", domain="my-domain"))
        self._define_agents([...])
```

### New Agent
```python
class MyAgent(DomainAgent):
    def __init__(self):
        super().__init__(signature=MySignature, config=DomainAgentConfig(name="MyAgent"))
```

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `ANTHROPIC_API_KEY` | Claude API access |
| `OPENAI_API_KEY` | OpenAI access |
| `GROQ_API_KEY` | Groq access (free tier) |
| `TELEGRAM_TOKEN` | Telegram bot token |
| `SLACK_SIGNING_SECRET` | Slack webhook verification |
| `DISCORD_PUBLIC_KEY` | Discord verification |
| `WHATSAPP_VERIFY_TOKEN` | WhatsApp webhook (default: "jotty") |

## Testing

```bash
# Run architecture verification
python3 docs/scripts/verify_architecture.py

# Test specific swarm
python3 -c "
import asyncio
from Jotty.core.swarms import CodingSwarm
swarm = CodingSwarm()
result = asyncio.run(swarm.execute('Test task'))
print(result)
"
```

## Common Patterns

### Get all skills
```python
from Jotty.core.skills import get_registry
registry = get_registry()
skills = registry.list_skills()  # 164 skills
```

### Discover for task
```python
discovery = registry.discover_for_task("create a chart")
# Returns: {'skills': [...], 'ui': [...]}
```

### Convert to Claude tools
```python
claude_tools = registry.get_claude_tools(['web-search', 'calculator'])
```

### Run workflow
```python
agent = AutoAgent()
result = await agent.execute("Research X, create report, send via telegram")
```

## Important Files

| File | Purpose |
|------|---------|
| `core/capabilities.py` | Discovery API — `capabilities()` and `explain()` |
| `core/memory/facade.py` | Memory subsystem facade |
| `core/learning/facade.py` | Learning subsystem facade |
| `core/context/facade.py` | Context subsystem facade |
| `core/skills/facade.py` | Skills/providers subsystem facade |
| `core/orchestration/facade.py` | Orchestration subsystem facade |
| `core/utils/facade.py` | Utilities subsystem facade |
| `core/registry/unified_registry.py` | Single entry point for all capabilities |
| `core/swarms/base_swarm.py` | Learning hooks (_pre/_post_execute_learning) |
| `core/orchestration/swarm_intelligence.py` | Learning state management |
| `cli/gateway/server.py` | UnifiedGateway (all webhooks) |
| `cli/gateway/channels.py` | ChannelRouter (message routing) |
| `cli/app.py` | JottyCLI (main CLI application) |

## What's NOT in This Codebase

- Frontend UI code is in separate `Jotty/ui/` Next.js app
- Telegram bot standalone is `telegram_bot/` (but gateway handles webhooks)
- JustJot.ai integration is in `common/justjot/`
