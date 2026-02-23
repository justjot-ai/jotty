# Jotty Learning Flow: Agent → Swarm → Orchestrator

## The Big Picture

```
┌─────────────────────────────────────────────────────────────┐
│                    EXECUTION TIME                           │
│                                                             │
│  Agent Planning ← Q-tables + Crystallized SOPs + Lessons    │
│       ↓                                                     │
│  Agent Executes (with injected learning context)            │
│       ↓                                                     │
│  Swarm Post-Processing (hot/cold split)                     │
│       ↓                                                     │
│  Orchestrator State Management                              │
│                                                             │
│                    LEARNING TIME                             │
│                                                             │
│  Q-table update → Judge scoring → Lesson distillation       │
│       ↓                                                     │
│  Pattern extraction → Crystallization check                 │
│       ↓                                                     │
│  Context ready for NEXT execution                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. Agent Level (AutonomousAgent)

**Before execution** — learning informs planning:
- `_try_crystallized()` loads hardened SOPs if the domain graduated (e.g., `research:travel` → proven role sequence + skill bindings)
- `get_role_guidance()` from StepQTable injects which roles work best with which skills
- `get_best_plan()` injects proven plan templates into the planner
- `build_context_string()` injects distilled lessons + reflexion insights into the prompt

**After execution** — results feed back:
- **SkillQTable**: records `(task_type, skill) → reward` with recency-weighted alpha
- **StepQTable**: records `(task_type, step_idx, role) → reward` + full plan template
- **Staleness canary**: calls `record_crystallized_outcome()` — if 3 consecutive failures, auto-decrystallizes

---

## 2. Swarm Level (SwarmLearningMixin)

Wraps every swarm execution with `_pre_execute_learning()` / `_post_execute_learning()`.

**Pre-execution:**
- Computes MorphAgent scores per agent (role consistency, diversity, tool reliability)
- Analyzes tool success rates
- Generates recommendations (e.g., "web-search failing → try alternative")
- Runs coordination protocols (circuit breakers, gossip, coalitions)

**Post-execution (MAPLE paper — hot/cold split):**

| Path | Timing | What happens |
|------|--------|-------------|
| **Hot** | Synchronous (~100ms) | Executor feedback, recompute MorphAgent scores, tool re-analysis, coordination protocols |
| **Cold** | Async background | LLM judge (5 dimensions), lesson distillation, pattern extraction, state persistence |

This split ensures learning never delays the user response.

---

## 3. LearningService (Central Hub)

All learning flows through `LearningService` — the single orchestrator.

### `record()` — called after every episode:
1. Auto-enriches outcome (code detection, word count, structural analysis)
2. Saves `EpisodeRecord` to SQLite
3. Updates TD value estimates
4. Triggers pattern extraction every N records
5. Auto-transfers patterns to related domains
6. **Fire-and-forget LLM judge** → scores accuracy, completeness, structure, actionability, depth
7. **Schedules fact distillation** → extracts 2-3 NEW lessons (dedup-aware, includes judge feedback)
   - Cold-start (<5 lessons): runs **synchronously** so lessons are ready for the next task
   - Warm: runs **async** in background

### `build_context_string()` — called before every episode:
- **Tier 1** (succeeding 90%+): Silence or lightweight hint
- **Tier 2** (cold-start, 0-2 episodes): Bootstrap guidance
- **Tier 3** (failures present): Full corrective context — distilled lessons + reflexion insights + retrieval + abstract patterns
- Budget-capped at 2000 chars

---

## 4. Q-Tables (TD-Lambda)

Two complementary tables:

| Table | Tracks | Key |
|-------|--------|-----|
| **SkillQTable** | Which skills work for which tasks | `(task_type, skill) → Q-value` |
| **StepQTable** | Which roles work at which steps, and which plan templates win | `(task_type, step, role) → Q-value` + `plan_history` |

Both use **recency-weighted alpha**: stale entries update faster so degraded skills don't retain artificially high Q-values.

**GroupedValueBaseline** provides hierarchical baselines for TD errors: `task_type:action → task_type → domain → global`.

---

## 5. Crystallization (Graduation Pipeline)

When a domain accumulates enough experience:

```
Exploration → Probation → Graduation → Hardened SOP
```

### `should_crystallize()` checks 5 convergence metrics:
- 25+ episodes
- 85%+ success rate
- 60%+ plan consistency
- 0.65+ role Q-values
- 8+ plans recorded

### `crystallize()` extracts a `CrystallizedConfig`:
- Top skills by success rate
- Best plan template (role sequence)
- Role→skill bindings with confidence = `Q × min(1.0, visits/10)`
- Distilled domain lessons baked into the SOP

### Staleness canary:
If a crystallized config leads to 3 consecutive failures, it auto-decrystallizes — the agent reverts to exploration with existing Q-data.

---

## 6. Orchestrator Level (SwarmIntelligence)

Manages cross-swarm state:
- **MorphAgent scores**: per-agent proficiency (RCS, RDS, TRAS)
- **Byzantine verifier**: detects agents claiming success with garbage output
- **Circuit breaker**: temporarily blocks repeatedly-failing agents
- **Effectiveness tracker**: measures whether the system is actually improving (recent vs. historical success rates)
- **Curriculum generator**: auto-generates training tasks for probation

---

## The Complete Loop

```
   ┌──────────── NEXT RUN ◄──────────────────────────┐
   │                                                   │
   ▼                                                   │
Agent reads:                                           │
  • Crystallized SOP (if graduated)                    │
  • Q-table role guidance + plan templates             │
  • Distilled lessons + reflexion insights             │
   │                                                   │
   ▼                                                   │
Agent executes with learned context                    │
   │                                                   │
   ▼                                                   │
Swarm hot-path: scores, feedback, protocols (100ms)    │
   │                                                   │
   ▼                                                   │
LearningService.record():                             │
  • Save episode → Q-table update (10ms)               │
  • LLM Judge scores 5 dims (background, 10-30s)       │
  • Distill 2-3 lessons (cold=sync, warm=async)        │
  • Pattern extraction → crystallization check         │
   │                                                   │
   └───────────────────────────────────────────────────┘
```

Every mechanism is **persistent** (SQLite + JSON on disk), **non-blocking** (hot/cold split), and **bidirectional** (learning informs execution, execution feeds learning).

---

## Key Files

| Component | File | Key Methods |
|-----------|------|-------------|
| Agent learning | `reasoning/agents/autonomous_agent.py` | `_try_crystallized()`, Q-table recording (~L1087-1128) |
| Swarm hooks | `orchestration/swarms/_base/_learning_mixin.py` | `_pre_execute_learning()`, `_post_execute_learning()` |
| LearningService | `learning/learning_service.py` | `record()`, `build_context_string()`, `_schedule_fact_distillation()` |
| Q-Tables | `learning/td_lambda.py` | `SkillQTable`, `StepQTable`, `GroupedValueBaseline` |
| Crystallization | `learning/crystallization.py` | `should_crystallize()`, `crystallize()`, `run_probation()`, `record_crystallized_outcome()` |
| Orchestrator | `orchestration/core/swarm_manager.py` | `SwarmIntelligence` |
| Swarm pipeline | `orchestration/swarms/_base/swarm_learning_pipeline.py` | Hot/cold evaluation split |
| Advanced learning | `learning/advanced_learning.py` | `LLMJudge`, `Reflexion`, `MCTSPlanner`, `VoyagerSkillLib` |
| Config | `infrastructure/foundation/configs/learning.py` | `LearningConfig` (single source of truth for all thresholds) |
