# Jotty - AI Agent Framework

## Quick Reference for Claude

**Main Architecture Doc:** `docs/JOTTY_ARCHITECTURE.md` - READ THIS FIRST

## 🏗️ New 5-Layer Architecture

Jotty is now organized in a clean 5-layer hierarchy:

```
Layer 1: INTERFACE      → External entry points (API, UI, CLI)
Layer 2: MODES          → Execution modes (Agent, Workflow, Execution)
Layer 3: CAPABILITIES   → Skills, Registry, Tools (273 skills!)
Layer 4: INTELLIGENCE   → Learning, Memory, Swarms, Orchestration
Layer 5: INFRASTRUCTURE → Foundation, Utils, Context, Monitoring
```

**Import Pattern:** `from Jotty.core.{layer}.{module} import X`

Example: `from Jotty.core.intelligence.memory.facade import get_memory_system`

---

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
from Jotty.core.intelligence.swarms.olympiad_learning_swarm import learn_topic

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

---

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

---

## Three Execution Paths

```python
from Jotty import Jotty
j = Jotty()

# 1. CHAT — conversational AI
j.router                    # ModeRouter for chat/workflow routing

# 2. WORKFLOW — multi-step automation
j.chat_executor             # ChatExecutor (direct LLM tool-calling, no agents)

# 3. SWARM — multi-agent coordination
from Jotty.core.intelligence.orchestration import Orchestrator
swarm = Orchestrator(agents="Research AI startups")
result = await swarm.run(goal="Research AI startups")
```

---

## Layer-by-Layer Access

### Layer 5: INFRASTRUCTURE (Foundation)

```python
# FOUNDATION — Core data structures, configs, types
from Jotty.core.infrastructure.foundation.data_structures import SwarmLearningConfig
from Jotty.core.infrastructure.foundation.agent_config import AgentConfig
from Jotty.core.infrastructure.foundation.exceptions import JottyError

# UTILS — Budget tracking, caching, circuit breakers
from Jotty.core.infrastructure.utils.facade import (
    get_budget_tracker,      # Track LLM costs
    get_circuit_breaker,     # Fault tolerance
    get_llm_cache,           # Response caching
    get_tokenizer,           # Token counting
)

# CONTEXT — Token management, compression
from Jotty.core.infrastructure.context.facade import (
    get_context_manager,     # Build context within token limits
    get_context_guard,       # Overflow protection
    get_content_gate,        # Content filtering
)

# MONITORING — Performance, safety, observability
from Jotty.core.infrastructure.monitoring.safety import SafetyMonitor
from Jotty.core.infrastructure.monitoring.observability import DistributedTracing
from Jotty.core.infrastructure.monitoring.monitoring import PerformanceTracker
```

### Layer 4: INTELLIGENCE (Brain)

```python
# MEMORY — 5-level brain-inspired memory
from Jotty.core.intelligence.memory.facade import (
    get_memory_system,       # Zero-config entry point
    get_brain_manager,       # BrainInspiredMemoryManager
    get_rag_retriever,       # LLMRAGRetriever
)

# LEARNING — RL, TD-Lambda, Q-Learning
from Jotty.core.intelligence.learning.facade import (
    get_td_lambda,           # TDLambdaLearner (gamma=0.99)
    get_credit_assigner,     # ReasoningCreditAssigner
    get_reward_manager,      # Reward management
)
from Jotty.core.intelligence.learning.td_lambda import TDLambdaLearner
from Jotty.core.intelligence.learning.q_learning import QLearningManager

# ORCHESTRATION — Swarm coordination
from Jotty.core.intelligence.orchestration.facade import (
    get_swarm_intelligence,  # SwarmIntelligence
    get_paradigm_executor,   # Relay/debate/refinement
    get_ensemble_manager,    # Ensemble methods
    get_provider_manager,    # LLM provider rotation
    get_swarm_router,        # Task→swarm routing
)

# SWARMS — Multi-agent coordination
from Jotty.core.intelligence.swarms.base_swarm import BaseSwarm
from Jotty.core.intelligence.swarms.base.domain_swarm import DomainSwarm
from Jotty.core.intelligence.swarms.coding_swarm import CodingSwarm
from Jotty.core.intelligence.swarms.research_swarm import ResearchSwarm
```

### Layer 3: CAPABILITIES (Skills & Tools)

```python
# SKILLS — 273 skills, 8 providers
from Jotty.core.capabilities.skills.facade import (
    get_registry,            # Skill registry
    list_providers,          # Available providers
    list_skills,             # All skills
)

# REGISTRY — Unified capability registry
from Jotty.core.capabilities.registry.unified_registry import (
    get_unified_registry,    # Single entry point
)
from Jotty.core.capabilities.registry.skills_registry import SkillsRegistry

# SEMANTIC — Database schema understanding, NL→SQL, visualization
from Jotty.core.capabilities.semantic.facade import (
    get_semantic_layer,      # Main semantic layer interface
    get_query_engine,        # NL→SQL query generation
    get_visualization_layer, # LIDA-based data visualization
    list_components,         # List all semantic components
    get_supported_databases, # List supported database types
)

# TOOLS — Tool management
from Jotty.core.capabilities.tools.content_generation import ContentGenerator

# SDK — Skill development kit
from Jotty.core.capabilities.sdk.skill_sdk.tool_helpers import format_json
```

### Layer 2: MODES (Execution Modes)

```python
# AGENT — Agent-based execution
from Jotty.core.modes.agent.base import (
    BaseAgent,               # Base agent class
    AgentRuntimeConfig,      # Runtime configuration
    AgentResult,             # Agent results
)
from Jotty.core.modes.agent.base.auto_agent import AutoAgent
from Jotty.core.modes.agent.base.chat_assistant import ChatAssistant
from Jotty.core.modes.agent.base.domain_agent import DomainAgent

# AUTONOMOUS — Autonomous execution
from Jotty.core.modes.agent.autonomous.intent_parser import IntentParser

# WORKFLOW — Workflow execution
from Jotty.core.modes.workflow.auto_workflow import AutoWorkflow
from Jotty.core.modes.workflow.research_workflow import ResearchWorkflow

# EXECUTION — Executors
from Jotty.core.modes.execution.executor import Executor
```

### Layer 1: INTERFACE (Entry Points)

```python
# API — Programmatic API
from Jotty.core.interface.api.unified import JottyAPI
from Jotty.core.interface.api.chat_api import ChatAPI
from Jotty.core.interface.api.workflow_api import WorkflowAPI

# USE CASES — Common use case implementations
from Jotty.core.interface.use_cases.chat.chat_executor import ChatExecutor
```

---

## Usage Examples

### Memory — Store, Retrieve, Check Status
```python
from Jotty.core.intelligence.memory.facade import get_memory_system
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

### Learning — TD-Lambda Updates
```python
from Jotty.core.intelligence.learning.facade import get_td_lambda
td = get_td_lambda()  # gamma=0.99, lambda_trace=0.95

td.update(
    state={"task": "research", "agent": "researcher"},
    action={"tool": "web-search"},
    reward=1.0,
    next_state={"task": "research", "agent": "researcher", "step": 2},
)
```

### Budget Tracking — Record LLM Costs
```python
from Jotty.core.infrastructure.utils.facade import get_budget_tracker
bt = get_budget_tracker()

bt.record_call("researcher", tokens_input=1000, tokens_output=500, model="gpt-4o")
bt.record_call("coder", tokens_input=500, tokens_output=200, model="gpt-4o-mini")

usage = bt.get_usage()  # {'calls': 2, 'tokens_input': 1500, 'tokens_output': 700, ...}
```

### LLM Cache — Cache and Retrieve Responses
```python
from Jotty.core.infrastructure.utils.facade import get_llm_cache
cache = get_llm_cache()

cache.set("prompt-hash-123", {"answer": "cached response"})
hit = cache.get("prompt-hash-123")  # returns CachedResponse or None
if hit:
    print(hit.response["answer"])   # NOTE: .response attribute, not subscript

stats = cache.stats()  # CacheStats with .hits, .misses, .hit_rate
```

### Context Management — Build Context Within Token Limits
```python
from Jotty.core.infrastructure.context.facade import get_context_manager
ctx = get_context_manager()  # max_tokens=28000

ctx.register_goal("Research AI startups")
ctx.register_critical_memory("Budget is $0.50 max")
ctx.add_chunk("Previous research findings...", category="research")

result = ctx.build_context(
    system_prompt="You are a research assistant",
    user_input="Find recent AI startup funding rounds",
)
```

### Semantic Layer — Natural Language to SQL
```python
from Jotty.core.capabilities.semantic.facade import get_semantic_layer

# From live database
layer = get_semantic_layer(
    db_type="postgresql",
    host="localhost",
    database="sales",
    user="admin",
    password="secret"
)

# Query with natural language
result = layer.query("Show total revenue by region this year")
print(result['generated_sql'])  # SELECT region, SUM(revenue) ...
print(result['results'])        # Query results (if execute=True)

# From DDL string (no database connection needed)
layer = get_semantic_layer(
    ddl="CREATE TABLE products (id INT, name VARCHAR(100), price DECIMAL)",
    dialect="postgresql"
)

# Generate LookML semantic model
lookml = layer.to_lookml()

# Suggest queries based on schema
suggestions = layer.suggest_queries(num_suggestions=5)
```

### Data Visualization — Charts from Natural Language
```python
from Jotty.core.capabilities.semantic.facade import get_visualization_layer

# Create viz layer from database
viz = get_visualization_layer(
    db_type="postgresql",
    host="localhost",
    database="sales"
)

# Generate charts from natural language
charts = viz.visualize(
    question="Show sales trends over the last 6 months",
    library="matplotlib",  # or seaborn, plotly, altair
    n=3  # Generate 3 chart variations
)

# Create multi-chart dashboard
dashboard = viz.dashboard(
    questions=[
        "Total revenue by region",
        "Monthly sales trends",
        "Top 10 products by quantity"
    ],
    library="plotly"
)

# Get suggested visualizations based on data
suggestions = viz.goals(n=5)  # Returns 5 visualization suggestions
```

### Semantic Skills — Use via Skill Wrappers
```python
# Option 1: Use semantic-sql-query skill
from skills.semantic_sql_query import query_database_natural_language

result = query_database_natural_language({
    "question": "Show customers with orders over $1000",
    "db_type": "postgresql",
    "host": "localhost",
    "database": "sales",
    "user": "admin",
    "password": "secret"
})

# Option 2: Use schema-analyzer skill
from skills.schema_analyzer import analyze_ddl_schema, generate_lookml_from_ddl

schema = analyze_ddl_schema({
    "ddl": "CREATE TABLE users (id INT PRIMARY KEY, email VARCHAR(255))",
    "dialect": "postgresql"
})

# Option 3: Use data-visualizer skill
from skills.data_visualizer import visualize_data_from_query

viz = visualize_data_from_query({
    "question": "Visualize sales by month",
    "db_type": "postgresql",
    "database": "sales",
    "library": "plotly",
    "output_format": "html"
})
```

---

## Top-Level Imports (Shortcuts)

```python
from Jotty import (
    capabilities,         # Discovery API
    MemorySystem,         # Memory
    BudgetTracker,        # Cost tracking
    CircuitBreaker,       # Fault tolerance
    LLMCallCache,         # Caching
    SmartTokenizer,       # Tokenization
    ChatExecutor,         # Direct LLM tool-calling
    SwarmIntelligence,    # Learning intelligence
    ParadigmExecutor,     # Relay/debate/refinement
    EnsembleManager,      # Ensemble methods
    ModelTierRouter,      # Model routing
)
```

---

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

---

## Directory Structure (5-Layer Architecture)

```
Jotty/
├── core/
│   ├── interface/           # Layer 1: Entry Points
│   │   ├── api/             # JottyAPI, ChatAPI, WorkflowAPI
│   │   ├── ui/              # UI components, formatters
│   │   └── use_cases/       # Common use case implementations
│   │
│   ├── modes/               # Layer 2: Execution Modes
│   │   ├── agent/
│   │   │   ├── base/        # BaseAgent, AutoAgent, ChatAssistant
│   │   │   └── autonomous/  # Intent parser, enhanced executor
│   │   ├── workflow/        # Auto workflows, research, learning
│   │   └── execution/       # Executors, intent classifiers
│   │
│   ├── capabilities/        # Layer 3: Skills & Tools
│   │   ├── skills/          # 273 skills (web-search, calculator, etc.)
│   │   ├── registry/        # Unified registry, skill registry
│   │   ├── tools/           # Content generation tools
│   │   ├── sdk/             # Skill development kit
│   │   └── semantic/        # Query engine, visualization
│   │
│   ├── intelligence/        # Layer 4: Brain Layer
│   │   ├── learning/        # TD-Lambda, Q-learning, RL
│   │   ├── memory/          # 5-level memory system
│   │   ├── orchestration/   # SwarmIntelligence, paradigms
│   │   ├── swarms/          # BaseSwarm, domain swarms
│   │   ├── reasoning/
│   │   │   └── experts/     # Expert agents, templates
│   │   └── optimization/    # Policy explorer
│   │
│   └── infrastructure/      # Layer 5: Foundation
│       ├── foundation/      # Data structures, configs, types
│       ├── utils/           # Budget tracker, cache, circuit breaker
│       ├── context/         # Context manager, chunker, compressor
│       ├── persistence/     # Swarm persistence
│       ├── integration/     # LLM providers, Lotus
│       ├── monitoring/      # Performance, safety, observability
│       ├── data/            # Feedback router, discovery
│       ├── metadata/        # Widget params, MCP metadata
│       ├── services/        # Command service
│       └── job_queue/       # Queue manager
│
├── cli/                     # CLI application (outside core/)
│   ├── app.py               # JottyCLI main class
│   ├── gateway/             # UnifiedGateway + ChannelRouter
│   ├── commands/            # Slash commands (/run, /swarm, etc.)
│   └── repl/                # REPL engine
│
├── skills/                  # Skill definitions (loaded lazily)
├── sdk/                     # Generated client libraries
└── web.py                   # Web server entry point
```

---

## Key Entry Points

| Entry Point | Command | Purpose |
|-------------|---------|---------|
| **CLI Interactive** | `python -m Jotty.core.interface.cli` | REPL with slash commands |
| **CLI Single** | `python -m Jotty.core.interface.cli -c "task"` | One-off execution |
| **Web Gateway** | `python -m Jotty.core.interface.web` | HTTP/WS server (port 8766) |
| **Gateway Only** | `python -m Jotty.core.interface.cli.gateway` | Webhooks for Telegram/Slack/etc |

---

## Important Files

| File | Purpose |
|------|---------|
| `core/capabilities.py` | Discovery API — `capabilities()` and `explain()` |
| `core/intelligence/memory/facade.py` | Memory subsystem facade |
| `core/intelligence/learning/facade.py` | Learning subsystem facade |
| `core/infrastructure/context/facade.py` | Context subsystem facade |
| `core/capabilities/skills/facade.py` | Skills/providers subsystem facade |
| `core/intelligence/orchestration/facade.py` | Orchestration subsystem facade |
| `core/infrastructure/utils/facade.py` | Utilities subsystem facade |
| `core/capabilities/registry/unified_registry.py` | Single entry point for all capabilities |
| `core/intelligence/swarms/base_swarm.py` | Learning hooks (_pre/_post_execute_learning) |
| `core/intelligence/orchestration/swarm_manager.py` | SwarmIntelligence (learning state management) |
| `cli/gateway/server.py` | UnifiedGateway (all webhooks) |
| `cli/gateway/channels.py` | ChannelRouter (message routing) |
| `cli/app.py` | JottyCLI (main CLI application) |

---

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

---

## Migration Notes (February 2026)

Jotty was reorganized from a flat 68-directory structure into a clean 5-layer hierarchy:

**Before:** `from Jotty.core.learning import TDLambdaLearner`
**After:** `from Jotty.core.intelligence.learning.td_lambda import TDLambdaLearner`

All imports have been updated. If you see old import paths in documentation or examples, update them to the new 5-layer structure.

---

## Type Hints

The codebase targets 100% type hint coverage. Use `Any` when the concrete type is dynamic or would require heavy imports; prefer concrete types for public APIs. `core/py.typed` marks the package as typed (PEP 561).

---

## What's NOT in This Codebase

- Frontend UI code is in separate `Jotty/ui/` Next.js app
- Telegram bot standalone is `telegram_bot/` (but gateway handles webhooks)
- JustJot.ai integration is in `common/justjot/`
