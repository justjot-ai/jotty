# Orchestrator Execution Flow

Complete architecture for `Orchestrator.run()` and `Orchestrator.chat()` — from goal to learned outcome.

---

## Public API

```python
# Universal execution (agents, swarms, pipelines — auto-detected or explicit)
result = await orchestrator.run(goal, learn=True)

# Conversational mode (tool-calling, streaming, history)
result = await orchestrator.chat(message, learn=True)

# Skip learning (tests, benchmarks, latency-critical)
result = await orchestrator.run(goal, learn=False)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `goal` / `message` | `str` | required | Task description or user message |
| `stream` | `bool` | `False` | Return `AsyncIterator[StreamEvent]` |
| `stages` | `list` | `None` | Pipeline stage definitions |
| `swarm` | `SwarmTemplate` | `None` | Explicit swarm template |
| `agent` | `BaseAgent` | `None` | Explicit single agent |
| `learn` | `bool` | `True` | Record outcomes + run post-episode learning |
| `status_callback` | `Callable` | `None` | Progress callback `(stage, detail)` |

---

## Complete Execution Flow

```
Orchestrator.run(goal, learn=True)
│
├── 1. PRE-EXECUTION (learn=True only)
│   ├── LearningService.build_context_string()     → inject past guidance
│   ├── Stigmergy.recommend_approach()             → "use/avoid" hints
│   ├── LearningService.improvement_report()       → cross-session trends
│   ├── EffectivenessTracker.improvement_report()  → current-session trends
│   ├── TransferLearning.get_relevant_learnings()  → similar past experiences
│   ├── MASLearning.get_execution_strategy()       → agent team recommendations
│   └── SwarmRouter.select_agent()                 → RL-guided agent selection
│
├── 2. EXECUTION (route based on parameters)
│   ├── Auto-detect  → ExecutionEngine.run()       (default: complexity-based)
│   ├── agent=X      → agent.execute(goal)         (single agent)
│   ├── swarm=X      → swarm.execute(goal)         (swarm template)
│   ├── stages=[...] → _run_pipeline(stages)       (multi-stage pipeline)
│   └── stream=True  → _run_stream(goal)           (async generator)
│   │
│   └── DURING EXECUTION (transparent, every tool call)
│       ├── ToolExecutionGuard.wrap_skill_tools()   → wraps ALL 164+ skills
│       │   ├── ToolInterceptor records: tool_name, args, result, success, latency
│       │   └── ProviderHealthManager records: success/failure per provider
│       └── ToolCallRegistry aggregates across all agents/skills
│
├── 3. POST-EXECUTION (learn=True only, fire-and-forget, never blocks result)
│   │
│   ├── 3a. LearningService.record()               → SQLite (single source of truth)
│   │   Records: unit_name, domain, task_type, context, action, outcome,
│   │   success, quality, execution_time
│   │
│   └── 3b. SwarmLearningPipeline.post_episode()   → 17 learning steps:
│       │
│       ├── _record_to_learning_service             → SQLite (unified persistence)
│       │   Also snapshots effectiveness every 5 episodes
│       │
│       ├── td_lambda                               → TD(λ) value baseline updates
│       │   Updates grouped value estimates via TD(0)
│       │   AdaptiveLearningRate adjusts λ based on success
│       │
│       ├── swarm_learner                           → Online prompt evolution
│       │   Records trajectory + insights
│       │   Conditionally updates architect prompts
│       │
│       ├── brain_consolidation                     → Brain-inspired memory consolidation
│       │   Fires async experience processing
│       │
│       ├── neurochunk_tiering                      → Memory tier promotion/demotion
│       │   Promotes high-reward, demotes low-reward memories
│       │   Prunes tier-3 (cold storage)
│       │
│       ├── agent_abstractor                        → Agent role profile updates
│       │   Updates specialization based on per-agent success
│       │
│       ├── transfer_learning                       → Transferable experience recording
│       │   Stores query→agent→action→reward for cross-task transfer
│       │
│       ├── swarm_intelligence                      → Specialization tracking
│       │   Records agent×task_type success matrix
│       │
│       ├── stigmergy                               → Ant-colony pheromone trails
│       │   Records outcome signals + approach signals
│       │   Decays old signals (evaporation)
│       │
│       ├── effectiveness                           → Improvement measurement
│       │   Windowed success rate: recent vs. historical
│       │   Per task_type and global
│       │
│       ├── mas_learning                            → Multi-agent session recording
│       │   Records session: agents used, time, success
│       │   Builds fix database (error→solution mappings)
│       │
│       ├── byzantine                               → Trust verification
│       │   Compares claimed success vs. actual result
│       │   Penalizes agents that over-report success
│       │
│       ├── credit_assignment                       → Who deserves credit?
│       │   Counterfactual credit assignment
│       │   Records improvement applications
│       │
│       ├── auditor_fixes                           → Negative learning from auditor
│       │   TD(-0.1) signal from fix_instructions
│       │   Records fixes as procedural memory
│       │
│       ├── adaptive_learning                       → Dynamic learning rate
│       │   Adjusts exploration/exploitation balance
│       │   Detects plateaus and convergence
│       │
│       ├── effectiveness_intervention              → Stagnation response
│       │   If not improving: boost exploration rate
│       │   Generates curriculum task for weak areas
│       │
│       ├── credit_pruning                          → Low-value pruning (every 10 episodes)
│       │   Removes low-impact transfer learnings
│       │   Keeps knowledge base lean
│       │
│       └── curriculum                              → Self-generated training
│           When exploration recommended: queue training task
│           Difficulty calibrated to current performance
│
│   └── 3c. Tool Learning Feedback (in swarm _learning_mixin)
│       ├── ToolLearningFeedback.feed_from_interceptor()
│       │   Reads ToolCallRegistry → feeds tool success/failure to TD-Lambda
│       │   Updates tool_success_rates and tool_avg_latencies
│       └── update_registry_with_learning()
│           Feeds learned statistics back to SkillsRegistry
│
└── 4. RETURN result to caller
```

---

## Tool Execution Guard — Where It Fits

```
                    ┌─────────────────────────────────┐
                    │     SkillsRegistry (164+ skills) │
                    │  ┌─────────────────────────────┐ │
                    │  │  SkillDefinition.tools       │ │
                    │  │  ┌─────────────────────────┐ │ │
                    │  │  │  ToolExecutionGuard      │ │ │    ← INJECTION POINT
                    │  │  │  ├── ToolInterceptor     │ │ │    Wraps every tool at load time
                    │  │  │  └── ProviderHealthMgr   │ │ │    Records success/failure/latency
                    │  │  └─────────────────────────┘ │ │
                    │  └─────────────────────────────┘ │
                    └─────────────────────────────────┘
                                    │
                    ┌───────────────┴──────────────────┐
                    │                                   │
            Path A: SkillPlanExecutor          Path B: Agent (tool calling)
            (explicit skill execution)         (LLM decides which tool)
                    │                                   │
                    └───────────┬───────────────────────┘
                                │
                    Post-execution: ToolLearningFeedback
                    reads interceptor data → TD-Lambda → SkillsRegistry
```

**Key insight:** `ToolExecutionGuard` is already correctly integrated. It wraps tools transparently at `SkillDefinition.tools` load time (line 437 of `skills_registry.py`). This single injection point covers ALL tool calls regardless of execution path. No changes needed — it's infrastructure-layer instrumentation that feeds into the learning pipeline automatically.

**Components:**

| Component | Location | Role |
|-----------|----------|------|
| `ToolInterceptor` | `infrastructure/integration/tool_interceptor.py` | Records every tool call (name, args, result, success, latency) |
| `ToolCallRegistry` | `infrastructure/integration/tool_interceptor.py` | Aggregates interceptors across all agents/skills |
| `ToolExecutionGuard` | `infrastructure/integration/guarded_tool_executor.py` | Singleton wrapper: combines ToolInterceptor + ProviderHealthManager |
| `tool_execution_guard.py` | `infrastructure/integration/tool_execution_guard.py` | Re-export shim for backward compat |
| `ToolLearningFeedback` | `intelligence/learning/tool_learning.py` | Reads interceptor data → feeds to TD-Lambda → updates SkillsRegistry |

---

## Data Flow: Two Persistence Paths (Unified)

```
                          ┌──────────────────────┐
                          │   LearningService     │ ← Single source of truth
                          │   (SQLite database)   │
                          │                       │
                          │  ├── episodes          │ ← Every execution recorded
                          │  ├── patterns          │ ← Extracted from episodes
                          │  ├── reflections       │ ← Self-improvement notes
                          │  └── value_estimates   │ ← State-action values
                          └──────────┬────────────┘
                                     │
                    ┌────────────────┤
                    │                │
          ┌────────┴───────┐  ┌─────┴──────────────────┐
          │  Orchestrator   │  │ SwarmLearningPipeline   │
          │  .run()/.chat() │  │ .post_episode()         │
          │  records via     │  │ records via              │
          │  LearningService │  │ LearningService +        │
          └────────────────┘  │ component-specific JSON   │
                               │                           │
                               │  JSON files (per-component)│
                               │  ├── swarm_learnings.json  │
                               │  ├── stigmergy.json        │
                               │  ├── credit_weights.json   │
                               │  ├── transfer_learnings.json│
                               │  └── swarm_intelligence.json│
                               └───────────────────────────┘
```

---

## Orchestrator.chat() Flow

```
Orchestrator.chat(message, learn=True)
│
├── 1. PRE-EXECUTION (learn=True only)
│   ├── LearningService.start_episode()        → episode_id
│   └── LearningService.query()                → domain guidance
│
├── 2. EXECUTION
│   └── ChatExecutor.execute(message, history)
│       ├── LLM tool-calling loop (max_steps)
│       ├── Tools wrapped by ToolExecutionGuard (transparent)
│       └── stream=True → async generator with token-level events
│
└── 3. POST-EXECUTION (learn=True only)
    └── LearningService.end_episode()          → SQLite
        Records: success, quality, cost, content_length, error
```

---

## Configuration

| Config | Default | Description |
|--------|---------|-------------|
| `learn=True` | `True` | Enable/disable all learning |
| `learning_wait_timeout_seconds` | `5.0` | Max wait for background learning init |
| `learning_components` | All 17 | Override which learning steps run |
| `recent_window` | `20` | EffectivenessTracker recent window |
| `historical_window` | `100` | EffectivenessTracker baseline window |

---

## Key Files

| File | Purpose |
|------|---------|
| `orchestration/core/swarm_manager.py` | Orchestrator: `run()`, `chat()`, learning wiring |
| `orchestration/learning/swarm_learning_pipeline.py` | 17-step post-episode learning pipeline |
| `orchestration/learning/stigmergy.py` | Ant-colony pheromone trails for agent routing |
| `orchestration/learning/byzantine_verification.py` | Trust verification: claimed vs. actual success |
| `orchestration/learning/mas_learning.py` | Multi-agent session history + fix database |
| `orchestration/learning/adaptive_learning.py` | Dynamic learning rate + exploration balance |
| `orchestration/learning/credit_assignment.py` | Counterfactual credit assignment |
| `orchestration/learning/metrics_collector.py` | Observability for swarm performance |
| `orchestration/learning/swarm_learner.py` | Online prompt evolution |
| `orchestration/learning/policy_explorer.py` | LLM-based policy exploration |
| `orchestration/learning/training_daemon.py` | Background self-improvement loop |
| `orchestration/learning/benchmarking.py` | Swarm performance benchmarking |
| `orchestration/learning/optimization_pipeline.py` | Generic iterative optimization |
| `intelligence/curriculum_generator.py` | Self-generated training tasks |
| `learning/learning_service.py` | Unified learning singleton (SQLite) |
| `learning/learning_store.py` | SQLite persistence layer |
| `learning/td_lambda.py` | TD(λ) reinforcement learning |
| `learning/q_learning.py` | Q-learning components |
| `learning/algorithmic_credit.py` | Shapley values, difference rewards |
| `learning/adaptive_components.py` | Adaptive learning rate, exploration |
| `learning/tool_learning.py` | Tool execution → TD-Lambda feedback loop |
| `infrastructure/integration/guarded_tool_executor.py` | ToolExecutionGuard singleton |
| `infrastructure/integration/tool_interceptor.py` | ToolInterceptor + ToolCallRegistry |
| `infrastructure/utils/provider_health.py` | Provider circuit breaker / health tracking |
| `capabilities/registry/skills_registry.py` | SkillDefinition.tools injection point |

---

*Last updated: 2026-02-17*
