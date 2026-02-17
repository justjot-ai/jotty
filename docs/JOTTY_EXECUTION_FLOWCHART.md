# Jotty Execution Flowchart — Orchestration to End

Step-by-step flow from orchestration entry points through to finished execution.

---

## 1. Entry Points (Where Execution Starts)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ ENTRY POINTS                                                                │
├─────────────────────────────────────────────────────────────────────────────┤
│ • Jotty.run(goal)              → TierExecutor.execute()                     │
│ • JottyAPI.chat_execute(msg)   → ChatUseCase → Conductor.run()              │
│ • JottyAPI.workflow_execute()  → WorkflowUseCase → Conductor.run()          │
│ • ModeRouter.route()           → ChatExecutor (unified) or workflow/agent   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Path A — TierExecutor (Jotty V3 / jotty.run)

Used when calling `Jotty().run(goal)` or any path that uses `TierExecutor` directly.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ TierExecutor.execute(goal, config, status_callback, **kwargs)                │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. INTENT CLASSIFICATION                                                     │
│    classify_task_intent(goal, attachments)                                   │
│    → intent_analysis.intent (e.g. fact_retrieval)                            │
│    → If fact_retrieval: force config.tier = DIRECT, hint_skills              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. TIER RESOLUTION                                                           │
│    if config.tier is None → TierDetector.adetect(goal)                       │
│    Start tracer (new_trace), then branch by tier                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          ▼                           ▼                           ▼
   ┌──────────────┐           ┌──────────────┐             ┌──────────────┐
   │ TIER 1       │           │ TIER 2       │             │ TIER 3       │
   │ DIRECT       │           │ AGENTIC     │             │ LEARNING     │
   └──────┬───────┘           └──────┬──────┘             └──────┬───────┘
          │                          │                            │
          ▼                          ▼                            ▼
   _execute_tier1             _execute_tier2              _execute_tier3
   (see Tier 1 flow)          (see Tier 2 flow)           (enrich goal with
                                                           memory → _execute_tier2)
          │                          │                            │
          │                    ┌─────┴─────┐                      │
          │                    ▼           ▼                      │
          │             TIER 4 RESEARCH   TIER 5 AUTONOMOUS        │
          │             _execute_tier4     _execute_tier5          │
          │             (see Tier 4/5)    (see Tier 4/5)           │
          │                                                       │
          └───────────────────────────┬───────────────────────────┘
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. RESULT WRAP-UP                                                            │
│    result.latency_ms, result.completed_at, tracer.end_trace()                │
│    metrics.record_execution(), return ExecutionResult                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Tier 1 (DIRECT) — Single LLM + Tools

```
_execute_tier1(goal, config, status_callback, **kwargs)
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. Resolve tools from registry (hint_skills if provided)                      │
│ 2. Build messages (system + user + history)                                  │
│ 3. LLM call with tools (streaming supported)                                │
│ 4. Parse response: tool_calls vs final text                                  │
└─────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 5. TOOL LOOP (while model returns tool_calls)                                │
│    For each tool_block: _execute_tool(tool_name, tool_input) → tool_result   │
│    Append tool results to messages, call LLM again                          │
└─────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 6. Extract final text; build ExecutionResult(output=..., tier=DIRECT)        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Tier 2 (AGENTIC) — Plan → Steps → Execute → Synthesize

```
_execute_tier2(goal, config, status_callback, **kwargs)
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. PLANNING                                                                  │
│    TaskPlanner (or planner from config) generates plan from goal             │
│    _parse_plan(goal, plan_result) → steps[]                                  │
└─────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. EXECUTE STEPS                                                             │
│    For each step: _execute_step(step, config)                                │
│    → LLM + tools per step, step.result = output                              │
└─────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. SYNTHESIS                                                                 │
│    Aggregate step results (or validator synthesis) → final output           │
│    Return ExecutionResult(output=..., tier=AGENTIC)                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Tier 3 (LEARNING) — Memory + Validation

```
_execute_tier3(goal, config, status_callback, **kwargs)
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. ENRICH GOAL WITH MEMORY                                                   │
│    _enrich_with_memory(goal, memory_context) → enriched_goal                 │
└─────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. RUN TIER 2                                                                │
│    _execute_tier2(enriched_goal, tier2_config, ...)                        │
└─────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. OPTIONAL VALIDATION RETRY                                                 │
│    If validator says retry: _execute_tier2(feedback_goal, ...)               │
│    Return ExecutionResult(tier=LEARNING)                                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Tier 4 (RESEARCH) & Tier 5 (AUTONOMOUS) — Swarm / Orchestrator

```
Tier 4: _execute_tier4(goal, config, ...)
Tier 5: _execute_tier5(goal, config, ...)  [optional sandbox]
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. SWARM SELECTION                                                           │
│    _select_swarm(goal, config.swarm_name) → swarm or None                    │
└─────────────────────────────────────────────────────────────────────────────┘
        │
        ├── swarm is not None ────────────────────────────────────────────────┐
        │                                                                      ▼
        │   ┌─────────────────────────────────────────────────────────────────┐
        │   │ Tier 5 only: if enable_sandbox → SandboxManager.execute(swarm,  │
        │   │               goal) else swarm.execute(task=goal, **kwargs)       │
        │   │ Tier 4:       swarm.execute(task=goal, **kwargs)                  │
        │   └─────────────────────────────────────────────────────────────────┘
        │                                      │
        │                                      ▼
        │   ┌─────────────────────────────────────────────────────────────────┐
        │   │ SwarmTemplate.execute() (see Swarm execute flow below)          │
        │   └─────────────────────────────────────────────────────────────────┘
        │
        └── swarm is None ────────────────────────────────────────────────────┐
                                                                               ▼
            ┌─────────────────────────────────────────────────────────────────┐
            │ _execute_with_swarm_manager(goal, config, ...)                   │
            │ → Create Orchestrator(config), await swarm_manager.run(goal)    │
            │ → ExecutionEngine.run() (Path B below)                          │
            └─────────────────────────────────────────────────────────────────┘
```

---

## 7. Swarm Execution (SwarmTemplate.execute)

When Tier 4/5 or Orchestrator invokes a domain swarm (e.g. ResearchSwarm, CodingSwarm):

```
SwarmTemplate.execute(*args, **kwargs)
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. _init_agents()                                                            │
│ 2. _pre_execute_learning()  — load state, warmup, tool analysis             │
└─────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. _execute_domain(*args, **kwargs)  [subclass implements]                    │
│    e.g. ResearchSwarm: research phases; CodingSwarm: design → code → review   │
│    May call execute_team(task) for multi-agent coordination                  │
└─────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. _post_execute_learning(success, execution_time, tools_used, ...)           │
│    — feedback, credit assignment, memory store                              │
└─────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 5. _validate_output_fields(result); attach traces; return SwarmResult      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Path B — Conductor (Orchestrator) Run

Used when using JottyAPI chat/workflow or when TierExecutor delegates to `_execute_with_swarm_manager`.

```
conductor.run(goal, **kwargs)
  conductor = create_swarm_manager(agents, config) → Orchestrator
        │
        ▼
Orchestrator.run(goal) → _ensure_engine().run(goal) → ExecutionEngine.run(goal)
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. INIT & GATE                                                               │
│    _ensure_runners(); wait _learning_ready (or skip if learning_wait_* = 0)  │
│    ValidationGate.decide(goal) [cached by goal+agent, TTL 60s] → mode        │
└─────────────────────────────────────────────────────────────────────────────┘
        │
        ├── mode single + gate DIRECT ────────────────────────────────────────┐
        │                                                                      ▼
        │   ┌─────────────────────────────────────────────────────────────────┐
        │   │ FAST PATH: ModelTierRouter.get_lm_for_mode(DIRECT)               │
        │   │ → Single LM call (e.g. lm(messages=[{role, content}]))           │
        │   │ → EpisodeResult(output=response); _save_learnings(); return     │
        │   └─────────────────────────────────────────────────────────────────┘
        │
        └── else (full pipeline) ─────────────────────────────────────────────┐
                                                                               ▼
            ┌─────────────────────────────────────────────────────────────────┐
            │ 2. MODEL TIER ROUTING (if not DIRECT)                            │
            │    get_model_for_mode(gate.mode) → configure dspy LM              │
            └─────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
            ┌─────────────────────────────────────────────────────────────────┐
            │ 3. ZERO-CONFIG (optional)                                        │
            │    _create_zero_config_agents(goal) → may set sm.agents, sm.mode  │
            │    single → multi if multiple sub-goals detected                  │
            └─────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
            ┌─────────────────────────────────────────────────────────────────┐
            │ 4. AUTONOMOUS SETUP (optional, single mode)                      │
            │    autonomous_setup(goal) — research/install/configure            │
            └─────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
            ┌─────────────────────────────────────────────────────────────────┐
            │ 5. ENSEMBLE (optional, single mode)                              │
            │    _execute_ensemble(goal, strategy, ...) → ensemble_context     │
            └─────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
            ┌───────────────┐                   ┌───────────────┐
            │ mode single   │                   │ mode multi     │
            └───────┬───────┘                   └───────┬───────┘
                    ▼                                   ▼
            _execute_single_agent(goal, gate_decision=…)  _execute_multi_agent(…)
                    │                                   │
                    ▼                                   ▼
            (see AgentRunner flow; reuses gate_decision) ParadigmExecutor (relay/debate/
                                              refinement) with N agents
```

---

## 9. AgentRunner (Single-Agent Full Pipeline)

When ExecutionEngine runs in single-agent mode:

```
AgentRunner.run(goal, **kwargs)
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. _setup_context(goal)   — Use gate_decision from kwargs if present       │
│    (from ExecutionEngine); else ValidationGate.decide() (cached). TD-lambda  │
└─────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. _gather_context(ctx)  — RAG/memory retrieval if enabled                  │
└─────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. _validate_architect(ctx)  — Architect produces/validates plan (or skip)   │
└─────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. _execute_agent(ctx)   — Agent runs (skills/tools, possibly multi-step)  │
└─────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 5. _validate_auditor_with_retry(ctx)  — Auditor checks; optional judge retry│
└─────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 6. _record_and_build_result(ctx)  — EpisodeResult, learning post_episode     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 10. End-to-End Summary Diagram

```
  [App/SDK]
       │
       ├── Jotty.run(goal) ──────────────► TierExecutor.execute(goal)
       │                                        │
       │                                        ├── Intent → Tier → Tier1/2/3/4/5
       │                                        ├── Tier 1: LLM + tools loop → ExecutionResult
       │                                        ├── Tier 2: Plan → Steps → Synthesize → ExecutionResult
       │                                        ├── Tier 3: Memory enrich → Tier2 → ExecutionResult
       │                                        └── Tier 4/5: Swarm.execute() or Orchestrator.run()
       │                                                      │
       │                                                      ▼
       │                                             SwarmTemplate.execute()
       │                                             → _pre_learning → _execute_domain → _post_learning
       │                                                      │
       │                                                      ▼
       │                                             ExecutionResult / SwarmResult
       │
       └── JottyAPI.chat_execute(msg) ──► ChatUseCase.execute() → ChatSessionExecutor.execute()
                                                │
                                                ▼
                                          conductor.run(goal)  [Orchestrator]
                                                │
                                                ▼
                                          ExecutionEngine.run(goal)
                                                │
                                                ├── ValidationGate → Fast path (direct LM) → EpisodeResult
                                                └── Full: zero-config → single/multi
                                                      │
                                                      ├── single → AgentRunner.run()
                                                      │           → setup → gather → architect → execute → auditor → record
                                                      └── multi  → ParadigmExecutor (relay/debate/refinement)
                                                │
                                                ▼
                                          EpisodeResult
```

---

## Key Types

| Type | Where | Meaning |
|------|--------|--------|
| **ExecutionResult** | TierExecutor | output, tier, success, latency_ms, metadata |
| **EpisodeResult** | Orchestrator / AgentRunner | output, success, trajectory, execution_time |
| **SwarmResult** | SwarmTemplate | success, output (domain-specific), agent_traces |

---

## Key Files

| Flow | Files |
|------|--------|
| TierExecutor | `core/intelligence/orchestration/execution/tier_executor.py` |
| ExecutionEngine | `core/intelligence/orchestration/core/swarm_manager.py` (ExecutionEngine class) |
| Orchestrator | `core/intelligence/orchestration/core/swarm_manager.py` (Orchestrator class) |
| AgentRunner | `core/intelligence/orchestration/execution/agent_runner.py` |
| SwarmTemplate | `core/intelligence/orchestration/swarms/base/swarm_template.py` |
| Chat use case | `core/intelligence/orchestration/use_cases/chat/chat_use_case.py`, `chat_executor.py` |
| Jotty V3 entry | `jotty.py` (Jotty.run → TierExecutor) |
| ValidationGate | `core/intelligence/orchestration/execution/validation_gate.py` (decide + cache) |

---

## 11. Optimization (implemented and remaining)

**Implemented (single gate path + cache + learning wait):**

1. **Pass gate decision to AgentRunner** — ExecutionEngine calls `ValidationGate.decide(goal)` once and passes the result in `kwargs["gate_decision"]`. AgentRunner reuses it when present (explicit override > passed decision > fresh `decide()`). This removes the redundant second `decide()` per full-pipeline run.
2. **Cache ValidationGate.decide()** — Decisions are cached by `(goal.strip()[:300], agent_name)` with configurable TTL (default 60s). Cache is pruned at 500 entries. Use `cache_ttl_seconds=0` or `enable_cache=False` to disable.
3. **Optional / shorter learning wait** — `SwarmLearningConfig.learning_wait_timeout_seconds` (default 5.0). Set to `0` to skip waiting for `_learning_ready` (latency-sensitive deployments).

**Best part of ValidationGate in the runner:** The runner does not duplicate the gate logic; it reuses the **same** gate decision (LLM or heuristic) produced by ExecutionEngine. So the “best part” (single cheap LLM call, safety rails, drift/sampling) stays in ValidationGate; the runner only consumes the result.

**Remaining (future):**

- Unify intent/tier (TierExecutor) with ValidationGate so one classification can drive both paths.
- Use async LM on ExecutionEngine fast path when available.
- Parallelize independent tool calls in Tier 1 tool loop.
