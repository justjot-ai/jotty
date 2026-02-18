# Core Flow and File Relationships

This document describes **flow** (how control and data move through `core/`) and **file relationships** (who imports whom). Every file is either shown in the flow with its relationships or explicitly marked as **NOT INTEGRATED** (no other core file imports it).

**Generated from static analysis of `core/**/*.py`.** Total: ~504 files; ~357 have at least one incoming import from elsewhere in core.

---

## 1. High-level flow (layers)

```
  EXTERNAL (SDK / CLI / Gateway / Web)
           │
           ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  LAYER: INTERFACE (core/interface/)                                       │
  │  Entry: mode_router, chat_api, workflow_api, unified (JottyAPI)            │
  │  → modalities (text/voice), ui (a2ui, schema_validator), interfaces       │
  └─────────────────────────────────────────────────────────────────────────┘
           │
           ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  LAYER: ORCHESTRATION (core/intelligence/orchestration/)                  │
  │  Hub: swarm_manager (Orchestrator)                                        │
  │  → execution (agent_runner, tier_executor, unified_executor, job_queue)   │
  │  → coordination (paradigm_executor, ensemble_manager, mas_zero)           │
  │  → routing (swarm_router, provider_manager, model_tier_router)            │
  │  → learning (training_daemon, learning_delegate, swarm_learning_pipeline)  │
  │  → swarms (_base, base, templates, domain swarms)                         │
  │  → intelligence (swarm_intelligence, curriculum, protocols)               │
  └─────────────────────────────────────────────────────────────────────────┘
           │
           ├──────────────────────────────────────────────────────────────────
           ▼                                                                  ▼
  ┌─────────────────────────────┐                    ┌─────────────────────────────────────┐
  │  REASONING (agents/planners) │                    │  CAPABILITIES + INFRASTRUCTURE       │
  │  core/intelligence/reasoning/│                    │  core/capabilities/                 │
  │  → agents (auto_agent,       │                    │  → registry (unified, skills, ui)   │
  │    chat_assistant,           │                    │  → prompts (composer, rules)        │
  │    swarm_agent, etc.)        │                    │  core/infrastructure/               │
  │  → planners (agentic_planner)│                    │  → foundation (data_structures,    │
  │  → executors, mixins, types  │                    │    agent_config, exceptions,        │
  │  → tools (inspector, axon)   │                    │    configs, types)                  │
  └─────────────────────────────┘                    │  → context, utils, persistence,     │
           │                                          │    integration, monitoring          │
           │                                          └─────────────────────────────────────┘
           │
           ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  LEARNING + MEMORY (core/intelligence/learning/, memory/)                 │
  │  → td_lambda, learning_coordinator, facade, cortex, llm_rag, etc.        │
  └─────────────────────────────────────────────────────────────────────────┘
```

**Rule of thumb:** Interface talks to Orchestration; Orchestration uses Reasoning, Capabilities, and Infrastructure; Learning/Memory are used by Orchestration and Reasoning.

---

## 2. Entry points (where external callers touch core)

These are the **roots** of the flow (no other core file imports them as the primary entry; they are imported by SDK/apps or by `core/__init__.py` / `core/jotty.py`).

| File | Role |
|------|------|
| `core/__init__.py` | Lazy facade; exposes Orchestrator, Cortex, Axon, etc. via `jotty.py` |
| `core/jotty.py` | Concrete imports: context, persistence, learning, memory, orchestration, interface.api |
| `core/interface/api/__init__.py` | Exports ModeRouter, ChatAPI, WorkflowAPI, JottyAPI |
| `core/interface/api/mode_router.py` | **Primary** routing (ExecutionContext, chat/workflow/skill/agent) |
| `core/interface/api/chat_api.py` | Chat API → Orchestrator |
| `core/interface/api/workflow_api.py` | Workflow API → Orchestrator |
| `core/interface/api/unified.py` | JottyAPI → Orchestrator, SwarmConfig, AgentConfig |
| `core/interface/api/registry.py` | get_unified_registry (capabilities) |
| `core/interface/api/agents.py` | AutoAgent, ChatAssistant (reasoning agents) |

All other entry-like files (e.g. `core/infrastructure/utils/facade.py`, `core/intelligence/learning/__init__.py`) are either **facades** intended to be used by apps/sdk or **package roots** (`__init__.py`). They are listed in the “Not integrated (intentional)” section where applicable.

---

## 3. Subsystem relationship maps

For each major subsystem we list **key files**, **depends on** (imports from core), and **used by** (imported by which core files). Only core-internal edges are shown.

### 3.1 Interface (`core/interface/`)

| File | Depends on (core) | Used by (core) |
|------|-------------------|----------------|
| `api/mode_router.py` | foundation.types.sdk_types | api/__init__, jotty (via api) |
| `api/chat_api.py` | intelligence.orchestration | api/__init__ |
| `api/workflow_api.py` | intelligence.orchestration | api/__init__ |
| `api/unified.py` | foundation.agent_config, foundation.data_structures, intelligence.orchestration | api/__init__, jotty |
| `api/registry.py` | capabilities.registry.unified_registry | — (entry) |
| `api/agents.py` | intelligence.reasoning.agents (auto_agent, chat_assistant) | — (entry) |
| `api/__init__.py` | chat_api, mode_router, openapi, unified, workflow_api | jotty |
| `interfaces/message.py` | — | interfaces/__init__ |
| `interfaces/host_provider.py` | — | orchestration.execution.agent_runner |
| `ui/schema_validator.py` | — | orchestration.execution.unified_executor |
| `ui/a2ui.py`, `ui/justjot_helper.py` | — | ui/__init__ |

Modalities (text/voice) and most of `interfaces/` have no core imports; they are self-contained or use stdlib only. **Integrated** via `interface` package and any app that imports them.

### 3.2 Orchestration hub (`core/intelligence/orchestration/core/swarm_manager.py`)

**Depends on (core):**

- `infrastructure.foundation`: agent_config, data_structures, exceptions, types.sdk_types
- `infrastructure.utils`: async_utils, budget_tracker
- `infrastructure.context`: context_manager
- `infrastructure.data`: data_registry, io_manager
- `infrastructure.integration.lotus`: integration
- `infrastructure.monitoring.metrics`: profiler
- `infrastructure.persistence`: shared_context
- `intelligence.learning`: learning_service, predictive_marl
- `intelligence.memory`: cortex
- `orchestration.coordination`: ensemble_manager, mas_zero_controller, paradigm_executor
- `orchestration.execution`: agent_runner, execution_orchestrator, swarm_dag_executor, swarm_warmup, unified_executor, validation_gate
- `orchestration.learning`: learning_delegate, mas_learning, swarm_learning_pipeline, training_daemon
- `orchestration.routing`: model_tier_router, provider_manager, swarm_provider_gateway, swarm_router
- `orchestration.state`: swarm_roadmap, swarm_state_manager, swarm_terminal
- `orchestration.swarms._base`: swarm_learning
- `orchestration`: swarm_code_generator, swarm_installer, swarm_researcher, zero_config_factory
- `reasoning`: agents.auto_agent, autonomous.intent_parser, planners.agentic_planner, tools.feedback_channel
- `capabilities.registry`: agui_component_registry, tool_validation (and prompts/registry via lazy or dotted imports)

**Used by (core):**

- `jotty.py`, `interface/api` (unified, chat_api, workflow_api)
- `orchestration.execution`: execution_orchestrator, swarm_dag_executor, swarm_warmup
- `orchestration.routing.provider_manager`
- `orchestration.core.swarm` (and possibly swarm_adapter)

So **swarm_manager** is the central orchestration node: it integrates foundation, context, data, learning, memory, execution, coordination, routing, state, swarms, and reasoning.

### 3.3 Orchestration execution (`core/intelligence/orchestration/execution/`)

| File | Depends on (core) | Used by (core) |
|------|-------------------|----------------|
| `agent_runner.py` | foundation (data_structures, exceptions), utils (async_utils, prompt_selector), learning (shaped_rewards, td_lambda), memory.cortex, execution.validation_gate, orchestration.prompts, reasoning.tools.inspector, interface.interfaces.host_provider | swarm_manager, state.swarm_terminal |
| `tier_executor.py` | foundation (data_structures, exceptions), monitoring.observability.tracing, coordination.paradigm_executor, tier_detector, types | execution/__init__ |
| `unified_executor.py` | llm_providers, capabilities.registry.skills_registry, interface.ui.schema_validator | swarm_manager |
| `validation_gate.py` | foundation.unified_lm_provider | swarm_manager, agent_runner, routing.model_tier_router |
| `intent_classifier.py` | capabilities.registry.skills_registry, foundation.unified_lm_provider | tier_executor (or similar) |
| `swarm_dag_executor.py` | foundation.data_structures, core.swarm_manager, reasoning.agents.auto_agent, planners.dag_agents | swarm_manager |
| `direct_chat_executor.py` | foundation.unified_lm_provider | — |
| `fact_retrieval_executor.py` | capabilities.registry, foundation.unified_lm_provider | — |
| `job_queue/*` | foundation.config_defaults (task.py); task, task_queue (internally) | swarm_manager, job_queue/__init__, infrastructure.job_queue/__init__ |

### 3.4 Orchestration learning (`core/intelligence/orchestration/learning/`)

| File | Depends on (core) | Used by (core) |
|------|-------------------|----------------|
| `swarm_learning_pipeline.py` | foundation (agent_config, data_structures, robust_parsing), intelligence.learning (adaptive_components, learning_coordinator, learning_service, predictive_marl, td_lambda, transfer_learning), memory.consolidation_engine, orchestration.intelligence (curriculum_generator, swarm_intelligence), orchestration.learning (adaptive_learning, byzantine_verification, credit_assignment, mas_learning, metrics_collector, stigmergy, swarm_learner), reasoning.tools (axon, feedback_channel) | learning_delegate, learning_pipeline |
| `learning_delegate.py` | foundation.agent_config, learning_pipeline, mas_learning | swarm_manager |
| `training_daemon.py` | foundation.data_structures | swarm_manager, facade |
| `mas_learning.py` | (internal orchestration) | swarm_manager, learning_delegate, core.swarm |
| `learning_pipeline.py` | swarm_learning_pipeline | learning_delegate |

### 3.5 Orchestration swarms (`core/intelligence/orchestration/swarms/`)

- **`_base/`**: swarm_learning, swarm_types, registry, evaluation, improvement_agents, mixins (_learning_mixin, _knowledge_mixin, _coordination_mixin), stage_config, pattern_selector, swarm_signatures.
  Used by: `base/`, `templates/*`, domain swarms (research, coding, olympiad_learning, etc.).
- **`base/`**: swarm_template, team_coordinator.
  swarm_template → _base (swarm_learning, swarm_types), team_coordinator.
  Used by: templates, domain swarms.
- **Templates** (e.g. `learning_swarm.py`, `testing_swarm.py`, `idea_writer_swarm.py`): depend on `reasoning.agents.swarm_agent`, `_base` (swarm_learning, swarm_signatures), `base` (SwarmTemplate, TeamCoordinator).
  **Integrated** via orchestration and swarm registry.
- **Domain swarms** (e.g. `research_swarm/swarm.py`, `coding_swarm/swarm.py`, `olympiad_learning_swarm/swarm.py`): depend on `_base`, `base`, and local `agents`, `types`, `signatures`.
  **Integrated** via orchestration and `swarm_manager` → swarm_router / registry.

### 3.6 Reasoning (`core/intelligence/reasoning/`)

| File / area | Depends on (core) | Used by (core) |
|-------------|-------------------|----------------|
| `agents/auto_agent.py` | (foundation, orchestration, capabilities as needed) | swarm_manager, intent/enhanced_executor, execution.swarm_dag_executor, swarm_code_generator, swarm_researcher |
| `agents/chat_assistant.py` | (typically foundation, inference) | interface.api.agents |
| `agents/swarm_agent.py` | (base, executors) | swarms templates and domain swarms |
| `planners/agentic_planner.py` | foundation.exceptions, mixins (inference, plan_utils, skill_selection) | swarm_manager, intent (enhanced_executor, intent_parser), execution.tier_executor, swarm_dag_executor, swarm_code_generator, swarm_researcher |
| `planners/swarm_resources_stub.py` | orchestration.core.swarm_resources | (planners) |
| `tools/inspector.py` | foundation.data_structures, learning.health_budget, memory.cortex | agent_runner, tier_executor, jotty |
| `tools/axon.py` | (infrastructure as needed) | jotty, swarm_learning_pipeline |
| `executors/skill_plan_executor.py` | foundation.exceptions, utils.async_utils, step_processors, tool_call_cache | autonomous_agent, etc. |
| `types/execution_types.py` | — | intent, planners, executors, many agents |

### 3.7 Capabilities (`core/capabilities/`)

| File | Depends on (core) | Used by (core) |
|------|-------------------|----------------|
| `registry/unified_registry.py` | skills_registry, ui_registry | interface.api.registry, swarm_manager (lazy), many execution/coordination files |
| `registry/skills_registry.py` | tool_execution_guard (integration) | unified_registry, swarm_manager, agent_runner, ensemble_manager, intent_classifier, fact_retrieval, pipelines, state.swarm_terminal, package_installer, etc. |
| `registry/tool_validation.py` | — | swarm_manager, agent_runner |
| `registry/ui_registry.py` | builtin_widgets | unified_registry |
| `prompts/composer.py` | rules | prompts/__init__ |
| `prompts/rules.py` | — | composer, prompts/__init__ |

### 3.8 Infrastructure (`core/infrastructure/`)

- **foundation**: data_structures, agent_config, exceptions, configs, types, tokenizer, llm_output_parser, robust_parsing, etc.
  **Used by:** almost every subsystem (orchestration, reasoning, learning, memory, capabilities, context, utils).
- **context**: context_manager, content_gate, chunker, compressor, models, utils, error_handling.
  **Used by:** swarm_manager, agent_runner, learning (algorithmic_foundations, q_learning), orchestration.state, etc.
- **utils**: tokenizer, async_utils, budget_tracker, provider_health, timeouts, facade, prompt_selector, context_logger, algorithmic_foundations (→ intelligence.learning.algorithmic_foundations).
  **Used by:** orchestration, reasoning, learning, memory, capabilities.
- **persistence**: vault, session_manager, shared_context.
  **Used by:** jotty, swarm_manager (shared_context).
- **integration**: guarded_tool_executor, tool_interceptor, tool_policy, lotus.
  **Used by:** capabilities (skills_registry), learning (tool_learning), swarms (_learning_mixin).
- **monitoring**: metrics (profiler, cost_tracker, etc.), observability (tracing), evaluation (gaia_adapter, etc.).
  **Used by:** tier_executor, swarm_manager (profiler), coordination (multi_stage, multi_strategy), etc.

### 3.9 Learning (`core/intelligence/learning/`)

| File | Depends on (core) | Used by (core) |
|------|-------------------|----------------|
| `facade.py` | foundation.configs, foundation.data_structures, learning_coordinator, learning_service, offline_learning, predictive_cooperation, reasoning_credit, shaped_rewards, td_lambda, tool_learning | mcp_tool_executor, tool_learning |
| `td_lambda.py` | foundation.configs.learning, foundation.data_structures, adaptive_components, memory.cortex | agent_runner, offline_learning, swarm_learning_pipeline, learning_coordinator |
| `learning_coordinator.py` | q_learning, td_lambda, memory.fallback_memory | facade, swarm_learning_pipeline |
| `offline_learning.py` | foundation.configs.learning, foundation.data_structures, memory.cortex, td_lambda | facade |
| `shaped_rewards.py` | (foundation/context as needed) | agent_runner |
| `algorithmic_foundations.py` | context.content_gate, memory.information_storage, algorithmic_credit | infrastructure.utils.algorithmic_foundations, swarm_learning_pipeline (via learning_coordinator/transfer) |
| `health_budget.py` | foundation.data_structures | reasoning.tools.inspector |
| `cortex` (memory) | — | td_lambda, offline_learning, inspector, agent_runner, swarm_manager, etc. |

### 3.10 Memory (`core/intelligence/memory/`)

| File | Depends on (core) | Used by (core) |
|------|-------------------|----------------|
| `cortex.py` | foundation.configs.memory, foundation.data_structures, consolidation_engine, llm_rag | jotty, agent_runner, inspector, td_lambda, offline_learning, swarm_manager, swarm_resources, memory_system, etc. |
| `llm_rag.py` | foundation.configs.memory, foundation.data_structures | cortex, memory/__init__ |
| `memory_system.py` | foundation.configs.memory, foundation.data_structures, consolidation_engine, cortex, fallback_memory, llm_rag, memory_orchestrator | facade, memory/__init__ |
| `consolidation_engine.py` | foundation.data_structures, foundation.robust_parsing | cortex, memory_system |
| `information_storage.py` | (foundation, etc.) | algorithmic_foundations (learning), infrastructure.data.information_storage |

---

## 4. Not integrated (no other core file imports this)

Files in this section have **no incoming import from any other core file** (excluding self or same-package re-exports). They are either:

- **Intentional entry points / facades** (e.g. `__init__.py`, `facade.py`, API roots), or
- **Standalone or optional** (e.g. benchmarks, examples, MCP server, optional adapters), or
- **Legacy / not yet wired** into the main flow.

**Convention:** If a file is an intentional public entry (facade, `__init__`, or documented API), it is marked **(intentional)**. Otherwise it is **(not integrated)** so you can decide whether to wire it in or remove it.

### 4.1 Intentional (entry points / facades / package roots)

- `core/__init__.py`
- `core/capabilities/__init__.py`, `capabilities/prompts/__init__.py`, `capabilities/skills/__init__.py`
- `core/capabilities/registry/api.py` (optional API surface)
- `core/infrastructure/__init__.py`, `infrastructure/foundation/__init__.py`, `infrastructure/foundation/configs/__init__.py`, `infrastructure/foundation/types/__init__.py`
- `core/infrastructure/integration/__init__.py`, `infrastructure/integration/lotus/__init__.py`
- `core/infrastructure/job_queue/__init__.py`, `infrastructure/metadata/__init__.py`
- `core/infrastructure/monitoring/evaluation/__init__.py`, `infrastructure/monitoring/metrics/__init__.py`, `infrastructure/monitoring/monitoring/__init__.py`, `infrastructure/monitoring/safety_gates/__init__.py`
- `core/infrastructure/persistence/__init__.py`
- `core/infrastructure/utils/__init__.py`, `infrastructure/utils/facade.py`
- `core/intelligence/__init__.py`, `intelligence/learning/__init__.py`
- `core/intelligence/orchestration/__init__.py`, `orchestration/communication/__init__.py`, `orchestration/coordination/__init__.py`, `orchestration/core/__init__.py`
- `core/intelligence/orchestration/execution/__init__.py`, `orchestration/execution/job_queue/__init__.py`, `orchestration/execution/memory/__init__.py`
- `core/intelligence/orchestration/facade.py`
- `core/intelligence/orchestration/intelligence/__init__.py`, `orchestration/intelligence/protocols/__init__.py`
- `core/intelligence/memory/__init__.py`

### 4.2 Not integrated (optional / standalone / legacy)

- `core/capabilities/registry/client_registration_helpers.py`
- `core/capabilities/registry/composite_skill.py`, `pipeline_skill.py`, `load_from_json.py`, `skills_manifest.py`, `widget_registry.py`
- `core/infrastructure/data/information_storage.py` (thin wrapper; see intelligence.memory.information_storage)
- `core/infrastructure/data/parameter_resolver.py`
- `core/infrastructure/foundation/helpful_errors.py`, `foundation/token_counter.py`
- `core/infrastructure/integration/lotus/benchmark.py`, `lotus/examples.py`
- `core/infrastructure/integration/mcp/__init__.py`, `mcp/server/__init__.py`, `mcp/server/memory_server.py`, `mcp/server_http.py`, `mcp/test_http_client.py`, `mcp_client.py`
- `core/infrastructure/monitoring/evaluation/gaia_dspy_optimizer.py`
- `core/infrastructure/monitoring/monitoring/*` (re-exports from metrics; monitoring/ is a thin wrapper)
- `core/infrastructure/persistence/scratchpad_persistence.py`
- `core/infrastructure/utils/api_client.py`, `env_loader.py`, `file_logger.py`, `profiler.py`, `rate_limiter.py`, `skill_status.py`, `smart_fetcher.py`, `tool_helpers.py`, `trajectory_parser.py`
- `core/intelligence/_comprehensive_protocols.py`
- `core/intelligence/learning/base_learning_manager.py`, `learning/rl_components.py`, `learning/utils.py`
- `core/intelligence/memory/justjot_sync_adapter.py`
- `core/intelligence/orchestration/coordination/_mas_zero_mixin.py`, `coordination/archive/_pipeline_utils.py`
- `core/intelligence/orchestration/core/swarm.py`, `core/swarm_adapter.py`
- `core/intelligence/orchestration/execution/fact_retrieval_executor.py`, `execution/job_queue/queue_manager.py`
- `core/intelligence/orchestration/intelligence/_ensemble_mixin.py`

Use this list to either **integrate** a file into the flow (add an import from a core component that should use it) or **document** it as an optional/standalone module.

---

## 5. Quick reference: one-line relationship summary

- **core/jotty.py**
  Imports: context (chunker, compressor, context_gradient, context_manager), data (data_registry, io_manager), foundation (agent_config, data_structures), persistence (vault, session_manager, shared_context), utils (algorithmic_foundations), learning (algorithmic_credit, predictive_cooperation, q_learning, td_lambda), memory (cortex), orchestration (swarm_manager, optimization_pipeline, state.swarm_roadmap), reasoning (tools.axon, tools.inspector), interface.api.
  **Integrated:** yes (main facade used by core/__init__ and SDK).

- **core/interface/api/***
  mode_router → sdk_types; chat_api, workflow_api, unified → Orchestrator (+ foundation where needed); registry → get_unified_registry; agents → auto_agent, chat_assistant.
  **Integrated:** yes (entry layer).

- **core/intelligence/orchestration/core/swarm_manager.py**
  **Integrated:** yes (central hub; see §3.2).

- **core/intelligence/orchestration/execution/agent_runner.py**
  **Integrated:** yes; used by swarm_manager, swarm_terminal; depends on foundation, utils, learning, memory, execution.validation_gate, reasoning.tools.inspector, interface.host_provider.

- **core/intelligence/reasoning/planners/agentic_planner.py**
  **Integrated:** yes; used by swarm_manager, intent, tier_executor, swarm_dag_executor, swarm_code_generator, swarm_researcher.

- **core/capabilities/registry/skills_registry.py**
  **Integrated:** yes; used by unified_registry, swarm_manager, agent_runner, ensemble_manager, intent_classifier, state.swarm_terminal, pipelines, and others.

- **core/infrastructure/foundation/data_structures.py**
  **Integrated:** yes; used by almost every subsystem (orchestration, reasoning, learning, memory, capabilities, context).

- **core/intelligence/learning/td_lambda.py**, **memory/cortex.py**
  **Integrated:** yes; used by agent_runner, offline_learning, swarm_learning_pipeline, inspector, and many others.

For any file not listed in §3 or §5, check §4: if it appears in “Not integrated (intentional)” it is an entry/facade; if in “Not integrated (optional/standalone/legacy)” it currently has no caller inside core.

---

## 6. How to use this document

- **Trace flow:** Start from §2 (entry points) or from `core/jotty.py`, then follow §3 by subsystem.
- **Check integration:** If a file is not under §3 and not in §4, it should have at least one core importer; if it appears in §4 “Not integrated”, decide whether to integrate or keep as optional.
- **Add a new file:** Prefer placing it in the right layer (interface / orchestration / reasoning / capabilities / infrastructure / learning / memory) and add one or more imports from an existing integrated file so the new file appears in the “used by” side and stays part of the flow.
- **Refactor:** Use the dependency directions in §3 to avoid inverting layers (e.g. infrastructure should not import orchestration; interface should not import memory directly except via orchestration/API).

This gives a single place to see **flow** and **every file's relationship** to the rest of core, with explicit **not integrated** status where nothing in core currently imports a file.

---

## 7. Index: core/ directories at a glance

| Directory | Role | Integration |
|-----------|------|-------------|
| `core/` | Root; `__init__.py` (lazy), `jotty.py` (concrete facade) | Entry |
| `core/capabilities/` | Registry (skills, UI), prompts (composer, rules) | Integrated via orchestration, interface, execution |
| `core/infrastructure/` | Foundation, context, utils, persistence, integration, monitoring, data, job_queue, metadata | Used by all layers |
| `core/interface/` | API (mode_router, chat/workflow/unified), modalities, ui, interfaces | Entry; used by jotty |
| `core/intelligence/learning/` | TD-Lambda, learning_coordinator, facade, shaped_rewards, algorithmic_* | Used by orchestration, reasoning |
| `core/intelligence/memory/` | Cortex, llm_rag, memory_system, consolidation, information_storage | Used by orchestration, reasoning, learning |
| `core/intelligence/orchestration/` | swarm_manager, execution, coordination, routing, learning, swarms, intelligence, state, llm_providers, pipelines, intent | Central hub; used by interface |
| `core/intelligence/reasoning/` | Agents, planners, executors, mixins, types, tools | Used by orchestration, interface |

**Legend:** **Entry** = external callers or core root; **Integrated** = at least one other core file imports from here; **Not integrated** = see §4 (either intentional facade or standalone).
