# Jotty Learning System — Self-Improving Agents

> **Status:** Production-ready (February 2026)
> **Key files:** `core/intelligence/learning/`

## Overview

Jotty agents improve through experience via 5 integrated mechanisms that work **by default** on every execution — no configuration needed.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        EXECUTION HAPPENS                                 │
│         Orchestrator.run()      |      Orchestrator.chat()               │
│              │                                                           │
│   ┌──────────┼──────────────┐                                            │
│   │ auto     │ swarm=X      │                                            │
│   ▼          ▼              │                                            │
│  Engine   Swarm.execute()   │  (internal — called by Orchestrator,       │
│  .run()   ._execute_domain()│   not a separate user-facing entry point)  │
│   │          │              │                                            │
│   └──────────┴──────────────┘                                            │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │ outcome recorded
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                  LearningService.record()  (automatic)                   │
│                                                                          │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐  │
│  │  Q-Tables   │  │  LLM Judge   │  │  Distillation│  │  Reflexion  │  │
│  │  (TD-λ RL)  │  │  (Sonnet)    │  │  (Haiku)     │  │  (on fail)  │  │
│  └──────┬──────┘  └──────┬───────┘  └──────┬───────┘  └──────┬──────┘  │
│         │                │                  │                  │         │
│         ▼                ▼                  ▼                  ▼         │
│  skill_q, step_q   5-dim scores      incremental        self-analysis   │
│  plan templates     + feedback        lessons             stored for     │
│  role→skill maps   strengths/gaps    (dedup-aware)       future runs    │
│         │                │                  │                            │
│         └────────────────┴──────────────────┘                            │
│                          │                                               │
│                          ▼                                               │
│              ┌───────────────────────┐                                   │
│              │   CRYSTALLIZATION     │                                   │
│              │  Probation → Graduate │                                   │
│              │  Hardened SOP config  │                                   │
│              └───────────────────────┘                                   │
└──────────────────────────────────────────────────────────────────────────┘
```

## 1. Q-Tables (TD-Lambda Reinforcement Learning)

**File:** `core/intelligence/learning/td_lambda.py`

Two Q-tables track what works:

### SkillQTable
Tracks which skills succeed for which task type + domain.

```python
# After 10 travel research tasks, the table might look like:
skill_q["research:travel"] = {
    "web-search": 0.97,      # very effective
    "claude-cli-llm": 0.95,  # effective for synthesis
    "file-operations": 0.90, # reliable for saving
    "browser-automation": 0.4 # less useful here
}
```

### StepQTable
Tracks plan templates (role sequences) and role→skill bindings.

```python
# Plan history for "research:travel":
plan_history = [
    (("research", "synthesize", "save"), 1.0),  # 3-step worked, reward=1.0
    (("research", "synthesize", "save"), 1.0),  # same template again
    (("research", "research", "synthesize", "save"), 0.8),  # 4-step variant
]

# Role guidance derived from history:
role_guidance = [
    {"role": "research",    "best_skill": "web-search",     "best_q": 0.97, "visits": 28},
    {"role": "synthesize",  "best_skill": "claude-cli-llm", "best_q": 0.95, "visits": 14},
    {"role": "save",        "best_skill": "file-operations", "best_q": 0.90, "visits": 14},
]
```

### Plan Normalization
Consecutive duplicate roles are collapsed for consistency tracking:
`(research, research, synthesize, save)` → `(research, synthesize, save)`

This prevents the 3-step and 4-step variants from being treated as completely different templates.

### How Q-Tables Feed into Planning
When `AutonomousAgent` plans a new task, it checks the Q-tables:
```
Q-table guidance injected: 3 roles, 2 plan templates for task_type=research
```
The planner sees which skills and plan structures have historically worked.

## 2. LLM Judge (Claude Sonnet)

**File:** `core/intelligence/learning/learning_service.py` → `llm_judge_quality_with_feedback()`

Every successful episode with >500 chars of content is evaluated by Claude Sonnet on 5 dimensions:

| Dimension | What It Measures |
|-----------|-----------------|
| **accuracy** | Are facts correct? Claims realistic and verifiable? |
| **completeness** | Does it cover ALL aspects the task requested? |
| **structure** | Well-organized with clear sections and flow? |
| **actionability** | Can someone directly USE this output? |
| **depth** | Goes beyond surface-level? Specific details? |

### Judge Output (structured JSON)
```json
{
    "accuracy": 0.85,
    "completeness": 0.70,
    "structure": 0.90,
    "actionability": 0.75,
    "depth": 0.65,
    "strengths": ["Well-organized with clear sections", "Good cost estimates"],
    "improvements": ["Missing visa information", "No local transport details"],
    "verdict": "llm-judged"
}
```

### Score Blending
Final quality = **0.6 × LLM judge** + **0.4 × heuristic** (word count, heading structure, keyword overlap).

The heuristic provides a fast baseline; the LLM provides nuanced evaluation.

### Trigger Conditions
- Episode must be successful (`success=True`)
- Content must be >500 characters
- Runs as fire-and-forget background task (doesn't block the user)
- Cold start: first 10 episodes are synchronous (3-5s) to seed quality data faster

## 3. Fact Distillation (Judge-Informed Lessons)

**File:** `core/intelligence/learning/learning_service.py` → `_build_distillation_prompt()`

After the judge scores an episode, Claude Haiku extracts 2-3 **incremental** lessons.

### Key Design Decisions

1. **Judge feedback drives lesson extraction:** The distillation prompt includes the judge's scores, strengths, and improvements. The LLM is instructed: *"Focus on what the EXPERT JUDGE highlighted — especially the improvements needed (learn from gaps) and strengths (learn what worked)."*

2. **Dedup-aware:** Before extracting, the prompt includes up to 15 existing lessons for the domain. The LLM is told: *"Only extract NON-OBVIOUS lessons that are NOT already listed above. If this task taught nothing new, return an empty array."*

3. **Incremental by design:** Task 1 learns basic lessons. Task 2 sees Task 1's lessons and only extracts what's NEW. Task 5 sees lessons from Tasks 1-4 and has a very high bar.

### Example Distillation Prompt
```
Analyze this travel task execution and extract 2-3 concise, actionable lessons.

TASK: Research top 5 places in Tokyo...

OUTPUT (first 1500 chars):
# Tokyo Travel Guide...

EXPERT JUDGE EVALUATION:
  accuracy: 0.85/1.0
  completeness: 0.70/1.0
  structure: 0.90/1.0
  Strengths: Well-organized; Good cost estimates
  Improvements needed: Missing visa info; No local transport details

ALREADY KNOWN LESSONS (do NOT repeat these):
- Always include cost estimates in USD for travel guides
- Structure guides with clear sections: overview, attractions, food, budget
...

CRITICAL: Focus on what the EXPERT JUDGE highlighted...
```

### Lesson Format
```json
[
    {
        "lesson": "Always include local transportation options (subway/bus passes, taxi apps) in travel guides",
        "type": "strategy",
        "applies_to": "travel research tasks",
        "confidence": 0.85
    }
]
```

### Storage
Lessons are persisted in SQLite (`distilled_lessons` table) with:
- `lesson_id`, `episode_id`, `domain`, `agent_name`
- `lesson` text, `context_type` (strategy/mistake/pattern/tool_usage)
- `confidence` score, `embedding` (for semantic retrieval)

## 4. Crystallization (Probation → Graduation)

**File:** `core/intelligence/learning/crystallization.py`

When an agent runs enough tasks with consistent success, it "graduates" — its proven knowledge is hardened into a `CrystallizedConfig`.

### Graduation Thresholds (Configurable)

| Threshold | Default | What It Means |
|-----------|---------|---------------|
| `min_episodes` | 25 | Minimum skill observations (~5-6 real tasks) |
| `min_success_rate` | 0.85 | Average plan reward must exceed this |
| `min_plan_consistency` | 0.60 | Top template must account for ≥60% of plans |
| `min_role_q` | 0.65 | Best skill per role must have Q ≥ this |
| `min_plans` | 8 | Minimum plan history entries |

Thresholds are configurable per probation run:
```python
result = await run_probation(
    task_type="creation",
    domain="mermaid",
    goals=[...],
    thresholds={
        "min_episodes": 10,
        "min_success_rate": 1.0,    # 100% for mermaid
        "min_plan_consistency": 0.70,
        "min_role_q": 0.80,
        "min_plans": 5,
    }
)
```

### CrystallizedConfig

When graduated, the agent gets:
```python
CrystallizedConfig(
    domain_key="creation:mermaid",
    task_type="creation",
    domain="mermaid",
    skills=["claude-cli-llm", "file-operations"],
    sop_roles=("generate", "save", "verify"),
    role_skill_map={
        "generate": "claude-cli-llm",
        "save": "file-operations",
        "verify": "file-operations"
    },
    prompt_guidance="Always validate Mermaid syntax...",
    success_rate=1.0,
    total_episodes=15,
)
```

### How It's Used
`AutonomousAgent` checks for a crystallized config before planning:
```python
config = load(task_type, domain)
if config:
    # Skip exploration — use proven SOP
    plan = config.to_plan_hint()  # Injected directly into planner
```

### Probation Pipeline

```python
from Jotty.core.intelligence.learning.crystallization import run_probation

result = await run_probation(
    task_type="research",
    domain="travel",
    goals=["Research Tokyo...", "Research Bali...", ...],
    max_tasks=15,
)
# result = {graduated: True, config: CrystallizedConfig, tasks_run: 6, ...}
```

Each iteration:
1. Execute a curriculum task via `Orchestrator.run()`
2. Learning records automatically (Q-tables, episodes, judge, distillation)
3. Check `maybe_crystallize()` — stop early if graduated
4. Continue until graduated or max_tasks reached

### Persistence
Crystallized configs are stored as JSON in `~/jotty/learning/crystallized/<key>.json`.

## 5. Reflexion (Self-Analysis on Failures)

**File:** `core/intelligence/learning/learning_service.py` → `_auto_reflect()`

When episodes fail or quality < 0.4, the system generates a self-analysis:
- What went wrong?
- What could have been done differently?
- What environmental factors contributed?

Reflections are stored in the episode record and inform future attempts.

## Integration Architecture

### All Paths Use the Same Pipeline

| Execution Path | Learning Integration |
|---------------|---------------------|
| `Orchestrator.run(goal)` | `learning.record()` → judge → distillation → Q-tables |
| `Orchestrator.run(goal, swarm=X)` | delegates to `Swarm.execute()` internally, same pipeline |
| `Orchestrator.chat()` (non-streaming) | `end_episode()` → `record()` → same pipeline |
| `Orchestrator.chat()` (streaming) | `end_episode()` → `record()` → same pipeline |
| Agent-level | `record()` via `BaseAgent` / `AgentRunner` |

> **Note:** `Swarm.execute()` is the swarm's internal API, not a user-facing entry point.
> The Orchestrator calls it via `run(goal, swarm=X)`, wrapping it with domain classification,
> learning context injection, budget awareness, and post-execution reflection. Calling
> `swarm.execute()` directly bypasses all of that.

### No Duplicate Processing
The `record()` method is the single point of entry. It:
1. Records the episode to SQLite
2. Updates Q-tables (skill + step)
3. Fires background LLM judge (which triggers distillation when done)
4. Runs reflexion on failures
5. Extracts patterns periodically
6. Auto-transfers learning between related domains

Previous duplicate judge calls in `Orchestrator.run()` and `Orchestrator.chat()` were removed — `record()` handles everything.

## API Reference

### Facade (Recommended Entry Point)
```python
from Jotty.core.intelligence.learning.facade import (
    get_learning_service,    # Unified: record/query/judge/distill
    get_td_lambda,           # Q-tables (skill + plan tracking)
    get_crystallized,        # Load graduated agent config
    list_components,         # List all learning components
)
```

### Crystallization
```python
from Jotty.core.intelligence.learning.crystallization import (
    run_probation,       # Probation pipeline (async)
    should_crystallize,  # Check graduation thresholds
    crystallize,         # Extract config from Q-tables
    load,                # Load graduated config from disk
    decrystallize,       # Remove a graduated config
    list_crystallized,   # List all graduated configs
    CrystallizedConfig,  # The config dataclass
)
```

### Top-Level Imports
```python
from Jotty import (
    LearningService,
    CrystallizedConfig,
    run_probation,
    should_crystallize,
)
```

### Capabilities Discovery
```python
from Jotty.core.capabilities import explain
print(explain("learning"))  # Full description of all 5 mechanisms
```

## File Map

| File | Purpose |
|------|---------|
| `learning_service.py` | Central orchestrator: record, query, judge, distill, reflect |
| `td_lambda.py` | TDLambdaLearner: SkillQTable + StepQTable, plan normalization |
| `crystallization.py` | Probation → graduation pipeline, threshold checks |
| `learning_store.py` | SQLite persistence for episodes, lessons, patterns |
| `response_analyzer.py` | Heuristic quality scoring (word count, headings, etc.) |
| `domain_classifier.py` | Classify task → (task_type, domain) |
| `pattern_extractor.py` | Cross-episode statistical pattern discovery |
| `reflection_engine.py` | Self-analysis on failures (Shinn et al. Reflexion) |
| `facade.py` | Clean entry points for all learning components |
| `embeddings.py` | Embedding service for semantic lesson retrieval |

## Design Principles

1. **Zero-config:** Works automatically for all agent executions. No setup needed.
2. **Non-blocking:** Judge and distillation run as background tasks. Never delays the user.
3. **Incremental:** Each task builds on previous lessons. No redundant learning.
4. **Configurable:** Thresholds, judge model, and distillation behavior are tunable per domain.
5. **Observable:** Every step logs — Q-table updates, judge scores, distilled lessons.
6. **Persisted:** SQLite for episodes/lessons, JSON for crystallized configs, pickle for Q-tables.
