# Jotty Architecture

Jotty is a self-improving AI agent framework built on DSPy. It coordinates multi-agent swarms with reinforcement learning, hierarchical memory, and a unified skill registry. This document covers the full system architecture.

## System Overview

```
                        EXTERNAL CLIENTS
                 ┌──────────┬──────────────┐
                 │ Telegram │ Slack/Discord │  Web/CLI/SDK
                 └────┬─────┴──────┬───────┘
                      │            │
              ┌───────▼────────────▼────────┐
   Layer 1    │       INTERFACE LAYER        │  web.py, cli/app.py, cli/gateway/
              │  UnifiedGateway + JottyCLI   │
              └─────────────┬───────────────┘
                            │
              ┌─────────────▼───────────────┐
   Layer 2    │        MODES LAYER           │  core/agents/, core/api/
              │  Chat | Workflow | API       │
              │  ChatAssistant | AutoAgent   │
              └─────────────┬───────────────┘
                            │
              ┌─────────────▼───────────────┐
   Layer 3    │       REGISTRY LAYER         │  core/registry/
              │     UnifiedRegistry          │
              │  Skills (Hands) + UI (Eyes)  │
              └─────────────┬───────────────┘
                            │
              ┌─────────────▼───────────────┐
   Layer 4    │        BRAIN LAYER           │  core/orchestration/, core/swarms/
              │  Orchestrator + Swarms       │
              │  SwarmIntelligence + Agents   │
              └─────────────┬───────────────┘
                            │
              ┌─────────────▼───────────────┐
   Layer 5    │     PERSISTENCE LAYER        │  core/learning/, core/memory/
              │  TD-Lambda + SwarmMemory     │
              │  ~/jotty/intelligence/*.json │
              └─────────────────────────────┘
```

## Layer 1: Interface

Entry points that accept external requests and route them into Jotty.

### UnifiedGateway (`cli/gateway/server.py`)

FastAPI + WebSocket server handling all channels:

- **HTTP Webhooks**: `/webhook/telegram`, `/webhook/slack`, `/webhook/discord`, `/webhook/whatsapp`
- **WebSocket**: `/ws` for real-time bidirectional chat
- **REST API**: `/message` for generic HTTP requests
- **Health**: `/health`, `/docs` (OpenAPI)

### JottyCLI (`cli/app.py`)

Interactive REPL with slash commands (`/run`, `/swarm`, `/learn`, etc.). Also supports single-command mode via `python -m Jotty.cli -c "task"`.

### ChannelRouter (`cli/gateway/channels.py`)

Routes incoming messages from different channels through `ChannelType` enum (TELEGRAM, SLACK, DISCORD, WHATSAPP, WEB, CLI) with trust-level authentication via `TrustManager`.

### Web Server (`web.py`)

Standalone entry point that initializes JottyCLI and starts UnifiedGateway on port 8766.

## Layer 2: Modes

Three execution modes that determine how tasks are processed.

### ChatAssistant (`core/agents/chat_assistant.py`)

Conversational mode. Maintains chat history, handles multi-turn dialogue, streams responses. Used for interactive Q&A.

### AutoAgent (`core/agents/auto_agent.py`)

Workflow mode. Takes a goal, discovers skills, creates an execution plan, and runs it autonomously. Handles multi-step tasks with replanning on failure.

Key flow:
```
Goal → TaskType Inference → Skill Discovery → Plan Creation → Step Execution → Result
                                                     ↑                    │
                                                     └── Replan on Fail ──┘
```

### JottyAPI (`core/api/unified.py`)

Programmatic API combining chat and workflow use cases:
```python
api = JottyAPI(agents=[...], config=SwarmConfig(...))
result = await api.chat(message="Hello")         # Chat mode
result = await api.workflow(goal="Research X")    # Workflow mode
async for event in api.chat_stream(message="Hi"): # Streaming
    print(event)
```

## Layer 3: Registry

The capability discovery system. All skills and UI components are registered here.

### UnifiedRegistry (`core/registry/unified_registry.py`)

Single entry point combining two sub-registries:

```python
registry = get_unified_registry()

# Backend (Skills = "Hands")
registry.list_skills()                    # 140+ skills
registry.discover("get stock quote")      # Semantic discovery
registry.get_claude_tools(['web-search']) # Convert to Claude tool format

# Frontend (UI = "Eyes")
registry.list_ui_components()             # 16 UI components
registry.ui.convert_to_a2ui('chart', data)
```

### SkillsRegistry (`core/registry/skills_registry.py`)

Manages 140+ skills loaded from `skills/` directories. Each skill has:
- `tools.py`: Tool function implementations
- `SKILL.md`: Metadata (name, description, capabilities, dependencies)
- Optional `skill.yaml`: Schema definitions

Key features:
- **Lazy loading**: Skills loaded on first access
- **Capability tags**: `finance`, `research`, `devops`, `communicate`, etc.
- **Executor types**: `api`, `gui`, `hybrid`, `general` (for hybrid action routing)
- **Semantic discovery**: Task description -> ranked skill matches
- **MCP integration**: Tools exposed as Model Context Protocol endpoints

### UIRegistry (`core/registry/ui_registry.py`)

16 UI components for rendering agent outputs (charts, tables, forms, markdown, etc.).

## Layer 4: Brain

The orchestration and agent execution layer.

### Orchestrator (`core/orchestration/swarm_manager.py`)

Central coordinator. Uses composition pattern with lazy-initialized components:

```
Orchestrator
├── AgentFactory          — Creates and manages agent runners
├── ExecutionEngine       — Runs single/multi-agent execution
├── ParadigmExecutor      — Relay/debate/refinement paradigms
├── TrainingDaemon        — Background training loop
├── ProviderManager       — LLM provider rotation
├── EnsembleManager       — Multi-model ensemble execution
├── LearningDelegate      — Learning pipeline integration
├── MASZeroController     — MAS-ZERO evolution
├── ModelTierRouter       — Routes tasks to appropriate model tiers
└── SwarmRouter           — Routes tasks to specialized swarms
```

**Execution paradigms**:
- **Single Agent**: One agent handles the full task
- **Relay**: Agents pass output sequentially (A -> B -> C)
- **Debate**: Multiple agents solve independently, then vote
- **Refinement**: Initial solution refined iteratively

### SwarmIntelligence (`core/orchestration/swarm_intelligence.py`)

Multi-agent coordination engine:
- **Specialization tracking**: Agents develop expertise based on performance history
- **Consensus voting**: Trust-weighted decisions across agents
- **Stigmergy**: Indirect coordination via shared signal layer
- **Dynamic routing**: Route tasks to best-fit agents
- **MorphAgent scoring**: RCS/RDS/TRAS alignment metrics
- **Byzantine verification**: Detect and isolate faulty agents
- **Curriculum generation**: Synthetic training tasks targeting weaknesses

### BaseSwarm (`core/swarms/base_swarm.py`)

Foundation for all swarm types. Implements the self-improving loop:

```
Expert Agent → Reviewer Agent → Planner Agent → Actor Agent
      │              │               │               │
      └──────────────┴───────────────┴───────────────┘
                    SHARED RESOURCES
         Memory | Context | Bus | TD-Lambda Learner
```

Each cycle:
1. `_pre_execute_learning()`: Load context from memory
2. Execute task through agent pipeline
3. `_post_execute_learning()`: Store results, update TD-Lambda
4. Gold Standard evaluation for quality tracking

### Agent Hierarchy

```
BaseAgent (ABC)
├── DomainAgent          — Single-task executor wrapping DSPy signatures
│   ├── ExpertAgent      — Generates expert solutions
│   ├── ReviewerAgent    — Reviews and critiques
│   ├── PlannerAgent     — Creates improvement plans
│   └── ActorAgent       — Executes planned actions
├── MetaAgent            — Coordinates other agents
├── AutonomousAgent      — Self-directed with skill access
│   └── AutoAgent        — Workflow-mode agent
└── ChatAssistant        — Conversational agent
```

### AgenticPlanner (`core/agents/agentic_planner.py`)

Fully LLM-based planning (no hardcoded logic). Uses DSPy signatures:

| Signature | Purpose | Module Type |
|-----------|---------|-------------|
| `TaskTypeInferenceSignature` | Classify task type | `Predict` |
| `CapabilityInferenceSignature` | Infer needed capabilities | `Predict` |
| `SkillSelectionSignature` | Select skills for task | `ChainOfThought` |
| `ExecutionPlanningSignature` | Create step-by-step plan | `ChainOfThought` |
| `ReflectivePlanningSignature` | Replan after failure | `ChainOfThought` |

### SkillPlanExecutor (`core/agents/base/skill_plan_executor.py`)

Reusable planning/execution service. Any agent can use it:
```python
executor = SkillPlanExecutor(skills_registry)
result = await executor.plan_and_execute(task, discovered_skills)
```

Components:
- `ToolCallCache`: TTL+LRU cache preventing redundant tool executions
- `ParameterResolver` (in `step_processors.py`): Template/param resolution with auto-wiring
- `ToolResultProcessor` (in `step_processors.py`): JSON-aware output sanitization/truncation

## Layer 5: Persistence

### TD-Lambda Learning (`core/learning/td_lambda.py`)

Reinforcement learning with temporal difference:

- **TDLambdaLearner**: Episode-based TD(lambda) with eligibility traces
- **GroupedValueBaseline**: HRPO-inspired grouped learning for variance reduction

Key features:
- Per-task-type baselines with exponential moving averages
- Action-type dimension (`task_type:action_type` composite keys)
- Cross-domain transfer learning
- Adaptive learning rates

### SwarmMemory (`core/memory/cortex.py`)

Five-level hierarchical memory:

| Level | Decay | Content |
|-------|-------|---------|
| EPISODIC | Fast | Raw experiences, full detail |
| SEMANTIC | Slow | Abstracted patterns (LLM-extracted) |
| PROCEDURAL | Medium | Action sequences, how-to knowledge |
| META | None | Learning wisdom, never decays |
| CAUSAL | None | Why things work, causal reasoning |

Key features:
- LLM-based retrieval (no embeddings required)
- Goal-conditioned values with transfer
- Causal knowledge extraction
- Automatic consolidation via DSPy signatures
- Deduplication and compression (Shannon-inspired)

### Persistence Files

```
~/jotty/intelligence/
├── {swarm}_{domain}.json    # SwarmIntelligence state
├── td_baselines.json        # TD-Lambda learned values
└── memory/
    ├── episodic.json
    ├── semantic.json
    ├── procedural.json
    ├── meta.json
    └── causal.json
```

## Data Flow: End-to-End Task Execution

```
1. User sends "Get my portfolio P&L" via Telegram

2. INTERFACE: UnifiedGateway receives webhook
   → ChannelRouter identifies TELEGRAM channel
   → TrustManager authenticates user

3. MODES: AutoAgent receives task
   → AgenticPlanner.infer_task_type() → "finance"
   → AgenticPlanner.infer_capabilities() → ["finance", "data-fetch"]

4. REGISTRY: SkillsRegistry.discover("get portfolio P&L")
   → Returns ranked skills: [pmi-portfolio (0.95), pmi-market-data (0.72), ...]

5. BRAIN: AgenticPlanner.plan_execution()
   → Step 1: pmi-portfolio.get_portfolio_tool
   → Step 2: pmi-portfolio.get_pnl_summary_tool
   → _enrich_io_contracts() fixes field-level refs

6. BRAIN: SkillPlanExecutor executes plan
   → Step 1: API call to PMI → holdings data
   → Step 2: API call to PMI → P&L summary
   → ParameterResolver wires step outputs to inputs

7. PERSISTENCE: Post-execution learning
   → SwarmMemory.store() episodic memory
   → TDLambdaLearner.update() value estimates
   → GroupedValueBaseline tracks "finance:api" performance

8. INTERFACE: Response sent back to Telegram
```

## Configuration

### SwarmConfig (`core/foundation/data_structures.py`)

Central configuration dataclass with 8 view subclasses for scoped access:

| View | Fields |
|------|--------|
| `PersistenceView` | `persist_memories`, `auto_save_interval`, ... |
| `ExecutionView` | `max_actor_iters`, `paradigm`, `agent_count`, ... |
| `MonitoringView` | `enable_metrics`, `enable_tracing`, ... |
| `LearningView` | `td_lambda`, `learning_rate`, `discount_factor`, ... |
| `BudgetView` | `max_cost_per_task`, `budget_limit`, ... |
| `EnsembleView` | `ensemble_strategy`, `ensemble_size`, ... |
| `CommunicationView` | `enable_bus`, `broadcast_results`, ... |
| `ProviderView` | `provider`, `model`, `temperature`, ... |

Access pattern:
```python
config = SwarmConfig(max_actor_iters=5)
view = ExecutionView(config)
view.max_actor_iters  # 5 (proxied to parent)
```

### AgentRuntimeConfig (`core/agents/base/base_agent.py`)

Per-agent configuration with smart defaults from `config_defaults.py`:
- `model`, `temperature`, `max_tokens`, `max_retries`
- `enable_memory`, `enable_context`, `enable_monitoring`, `enable_skills`

## Exception Hierarchy

All Jotty exceptions inherit from `JottyError`:

```
JottyError
├── ConfigurationError
│   ├── InvalidConfigError      (10 usages - provider validation)
│   └── MissingConfigError
├── ExecutionError
│   ├── AgentExecutionError     (8 usages - agent failures)
│   ├── ToolExecutionError
│   ├── TimeoutError
│   └── CircuitBreakerError
├── ContextError
│   ├── ContextOverflowError    (custom: detected_tokens, max_tokens)
│   ├── CompressionError
│   └── ChunkingError
├── MemoryError
│   ├── MemoryRetrievalError
│   ├── MemoryStorageError
│   └── ConsolidationError
├── LearningError
│   ├── RewardCalculationError
│   ├── CreditAssignmentError
│   └── PolicyUpdateError
├── CommunicationError
│   ├── MessageDeliveryError
│   └── FeedbackRoutingError
├── ValidationError             (15 usages - param validation)
│   ├── InputValidationError    (7 usages - LM input checks)
│   └── OutputValidationError
├── PersistenceError
│   ├── StorageError
│   └── RetrievalError
└── IntegrationError
    ├── LLMError                (18+ usages - API failures)
    ├── DSPyError
    └── ExternalToolError
```

Every `JottyError` accepts `message`, `context` (dict), and `original_error` (for wrapping).

## Key Design Patterns

### Lazy Initialization
Heavy imports (DSPy, ML libraries) are deferred until first use to keep startup under 2 seconds:
```python
_dspy_module = None
def _get_dspy():
    global _dspy_module
    if _dspy_module is None:
        import dspy as _dspy
        _dspy_module = _dspy
    return _dspy_module
```

### Singleton with Reset
12 singletons provide `reset_*()` functions for test isolation:
```python
_instance = None
def get_instance():
    global _instance
    if _instance is None:
        _instance = MyClass()
    return _instance

def reset_instance():
    global _instance
    _instance = None
```

### Composition over Inheritance
`Orchestrator` composes managers rather than inheriting:
```python
class Orchestrator:
    def __init__(self, ...):
        self._agent_factory = AgentFactory(self)
        self._execution_engine = ExecutionEngine(self)
        self._paradigm_executor = ParadigmExecutor(self)
```

### ConfigView Proxy
Scoped configuration access without breaking flat attributes:
```python
class _ConfigView:
    _FIELDS = ()
    def __getattr__(self, name):
        if name in self._FIELDS:
            return getattr(self._parent, name)
        raise AttributeError(...)
```

## Skills System

140+ skills organized in `skills/` directories:

| Category | Skills | Examples |
|----------|--------|---------|
| Finance (PMI) | 32 tools across 7 packs | Market data, portfolio, trading, alerts |
| Web | 12 tools | Search, scrape, browse, screenshot |
| DevOps | 8 tools | Docker, SSH, monitoring |
| Communication | 6 tools | Telegram, email, notifications |
| Analysis | 5 tools | Sentiment, comparison, earnings |
| Automation | 16 tools | Android device control |
| n8n Workflows | Dynamic | 15+ workflow templates |
| AI/ML | 8 tools | Claude CLI, code generation |
| Utility | 20+ tools | Calculator, calendar, file ops |

Each skill registers with capabilities and executor type for intelligent routing.

## Testing

```bash
pytest tests/test_v3_execution.py -v          # Core execution (451 tests)
pytest tests/test_pmi_skills.py -v            # PMI skills (75 tests)
pytest tests/test_skill_plan_executor.py -v   # Plan executor (125 tests)
pytest tests/test_agentic_planner.py -v       # Planner (42 tests)
pytest tests/test_mas_bench.py -v             # MAS-Bench (49 tests)
pytest tests/ -m unit                         # All unit tests
```

All tests use mocks (no real LLM calls), run fast (< 1s each), and work offline.
