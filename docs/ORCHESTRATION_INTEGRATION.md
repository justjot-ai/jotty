# Orchestration: Deep Learning Integration

## Architecture Overview

The orchestration module (`core/intelligence/orchestration/`) is organized into 9 subdirectories:

```
orchestration/
├── core/           Orchestrator, swarm entry point, adapter
├── intelligence/   SwarmIntelligence, consensus, ensemble, protocols
├── learning/       LearningPipeline, Stigmergy, Byzantine, MAS, Metrics
├── execution/      AgentRunner, ChatExecutor, sandbox, DAG executor
├── routing/        SwarmRouter, ModelTierRouter, ProviderManager
├── coordination/   ParadigmExecutor, EnsembleManager, MAS-Zero
├── state/          SwarmStateManager, SwarmRoadmap, SwarmTerminal
├── llm_providers/  Anthropic, OpenAI, Google adapters
└── templates/      Domain-specific swarm templates
```

All external imports go through `__init__.py` lazy loading — zero breaking changes.

---

## 4 Learning Components (Previously Dormant, Now Active)

| Component | File | Purpose |
|-----------|------|---------|
| **StigmergyLayer** | `learning/stigmergy.py` | Ant-colony pheromone trails for agent routing |
| **ByzantineVerifier** | `learning/byzantine_verification.py` | Detects agents that lie about success |
| **MASLearning** | `learning/mas_learning.py` | Multi-agent session recording & strategy |
| **MetricsCollector** | `learning/metrics_collector.py` | Observability & health reporting |

### Initialization Chain

```
Orchestrator.__init__()
  └── learning = LazyComponent → SwarmLearningPipeline(config)
        ├── stigmergy = StigmergyLayer(decay_rate=0.1)
        ├── byzantine_verifier = ByzantineVerifier(swarm_intelligence)
        ├── mas_learning = MASLearning()
        └── metrics = MetricsCollector.get_global()

Orchestrator._ensure_runners()
  └── for each AgentRunner:
        runner.inject_learning({
            "stigmergy": lp.stigmergy,
            "byzantine": lp.byzantine_verifier,
            "mas_learning": lp.mas_learning,
            "metrics": lp.metrics,
        })
```

---

## 7 Integration Hooks

### Hook 1: Post-Execution Learning (agent_runner.py)

**When**: After every agent task completes (success or failure)
**What**: Records outcome in all 4 systems

```
Agent completes task
  ├── stigmergy.record_outcome(agent, task_type, success, quality)
  ├── byzantine.verify_output_quality(agent, claimed_success, output, goal)
  │     └── If verification fails → override success=False
  ├── mas_learning.record_session(task, agents, time, success)
  └── metrics.record_task(swarm, agent, task_type, success, duration)
```

### Hook 2: Pre-Execution Strategy (swarm_manager.py)

**When**: Before assigning a task to an agent
**What**: Uses learned intelligence to pick the best agent

```
Task arrives
  ├── stigmergy.recommend_agent(task_type, candidates)
  │     └── Reorders candidates by pheromone signal strength
  ├── stigmergy.recommend_approach(task_type)
  │     └── Suggests tools/methods based on past success
  └── mas_learning.get_execution_strategy(goal, agents)
        └── Returns recommended_order, skip_agents, confidence
```

### Hook 3: Error Recovery (agent_runner.py)

**When**: Agent encounters an exception, before retry
**What**: Checks MAS learning fix database

```
Agent fails with error
  └── mas_learning.find_fix(error_message)
        ├── If fix found (success_rate >= 50%):
        │     └── Inject fix description into agent context
        └── If no fix: fall through to SwarmTerminal auto-fix
```

### Hook 4: Byzantine Consensus (_consensus_mixin.py)

**When**: During multi-agent voting in `_tally_weighted()`
**What**: Penalizes unverified agent claims

```
Consensus voting
  for each vote:
    ├── byzantine.verify_claim(agent, claimed_success, result)
    ├── If NOT verified: confidence *= 0.3 (70% penalty)
    └── weighted_score += confidence × trust
```

### Hook 5: Consistency Checking (paradigm_executor.py)

**When**: After debate rounds with 2+ agents
**What**: Detects hallucination via cross-agent consistency

```
Debate completes (2+ drafts)
  └── byzantine.consistency_checker.check_consistency(outputs)
        └── For each outlier: mark result.success = False
```

### Hook 6: TD-Lambda Metrics (learning_pipeline.py)

**When**: After TD-Lambda value update
**What**: Records learning convergence metrics

```
TD-Lambda update
  └── metrics.record_task(
        swarm="learning_pipeline",
        task_type="td_update:{task_type}",
        success=result.success
      )
```

### Hook 7: Health Report at Shutdown (swarm_manager.py)

**When**: Orchestrator shutdown
**What**: Logs metrics summary and effectiveness report

```
Orchestrator.shutdown()
  ├── metrics.get_report()
  │     └── Log: "Swarm Health: success=X%, tasks=N, trend=improving/declining"
  └── effectiveness.improvement_report()
        └── Log: "Effectiveness: recent=X%, ..."
```

---

## SwarmIntelligence Activated Features

### Result Caching (agent_runner.py)

```
Before execution:
  cache_key = md5(agent_name + goal)
  cached = swarm_intelligence.get_cached(cache_key)
  if cached: skip execution, return cached result

After execution:
  swarm_intelligence.cache_result(cache_key, output, ttl=1800)
```

### Trust-Weighted Consensus (paradigm_executor.py)

```
aggregate_results() with 2+ agents:
  for each agent result:
    trust = swarm_intelligence.agent_profiles[agent].trust_score
    quality = 0.8 if success else 0.2
    score = trust × quality
  → Select agent with highest score
```

### Task Decomposition & Priority Queue

Available via SwarmIntelligence → LifecycleMixin forwarding:
- `si.decompose_task(complex_task)` → subtasks
- `si.enqueue_task(task, priority)` / `si.dequeue_task()`
- `si.execute_parallel(subtasks)` / `si.parallel_map(fn, items)`

---

## Data Flow Diagram

```
  User Task
      │
      ▼
  ┌──────────────┐     ┌─────────────────┐
  │  Orchestrator │────▶│ Pre-Execution   │
  │  (Hook 2)    │     │ Strategy Hook   │
  └──────┬───────┘     │ • Stigmergy rec │
         │             │ • MAS strategy  │
         ▼             └─────────────────┘
  ┌──────────────┐
  │  AgentRunner │──── Cache check (Hook 4c)
  │  (Hooks 1,3) │
  └──────┬───────┘
         │
    ┌────┴────┐
    ▼         ▼
 Success    Failure
    │         │
    │    ┌────▼────────┐
    │    │ Error Recovery│ (Hook 3)
    │    │ MAS find_fix │
    │    └─────────────┘
    │
    ▼
  ┌──────────────────┐
  │ Post-Execution   │ (Hook 1)
  │ • Stigmergy      │ → pheromone trail
  │ • Byzantine      │ → trust verification
  │ • MAS Learning   │ → session record
  │ • Metrics        │ → observability
  └──────┬───────────┘
         │
         ▼
  ┌──────────────────┐
  │ Consensus/Debate │ (Hooks 4, 5)
  │ • Byzantine vote │ → penalize unverified
  │ • Consistency    │ → detect outliers
  └──────────────────┘
         │
         ▼
  ┌──────────────────┐
  │ Shutdown         │ (Hook 7)
  │ • Health report  │ → log metrics
  │ • Save learnings │ → persist stigmergy
  └──────────────────┘
```

---

## Dependency Injection Pattern

```python
# In SwarmLearningPipeline:
@property
def learning_components(self) -> Dict[str, Any]:
    return {
        "stigmergy": self.stigmergy,
        "byzantine": self.byzantine_verifier,
        "mas_learning": self.mas_learning,
        "metrics": self.metrics,
    }

# In Orchestrator._ensure_runners():
components = self.learning.learning_components
for runner in self.runners.values():
    runner.inject_learning(components)

# In AgentRunner.inject_learning():
def inject_learning(self, components: Dict[str, Any]) -> None:
    self._stigmergy = components.get("stigmergy")
    self._byzantine_verifier = components.get("byzantine")
    self._mas_learning = components.get("mas_learning")
    self._metrics = components.get("metrics")
```

---

## Archived Files

Moved to `archive/` subdirectories (available for review):

| File | From | Reason |
|------|------|--------|
| `hybrid_team_template.py` | templates/ | Superseded by execution/swarms/templates/ |
| `sequential_team_template.py` | templates/ | Superseded by execution/swarms/templates/ |
| `auto_provider_discovery.py` | routing/ | In lazy map but never consumed |
| `_pipeline_utils.py` | coordination/ | Utility functions never imported |

---

## Test Coverage

| Test File | Tests | Type | What It Covers |
|-----------|-------|------|----------------|
| `test_learning_integration_hooks.py` | 22 | Unit (mocked) | All 7 hooks, injection, caching |
| `test_real_llm_hooks.py` | 17 | Integration (real LLM) | Real API calls, component init, Orchestrator pipeline |

Run:
```bash
pytest tests/integration/test_learning_integration_hooks.py -v  # Fast, mocked
pytest tests/integration/test_real_llm_hooks.py -v              # Needs ANTHROPIC_API_KEY
```

---

## Pre-Existing Issues Found & Fixed

| Issue | File | Fix |
|-------|------|-----|
| `AutoAgent` wrong import path | `core/swarm_manager.py` | `agents.auto_agent` → correct |
| `agentic_planner` wrong relative import | `agents/auto_agent.py` | `.agentic_planner` → `..planners.agentic_planner` |
| `planner_signatures` wrong relative import | `planners/agentic_planner.py` | `.planner_signatures` → `..types.planner_signatures` |
| `TemplateAgentConfig` never defined | `templates/base.py` | Renamed to `AgentConfig` |
| 4 mixin imports pointing to wrong dir | `templates/swarm_ml_comprehensive.py` | `._mixin` → `skills.automl.reporting._mixin` |
| `SmartTokenizer` wrong relative import | `tools/axon.py` | `..utils.tokenizer` → absolute import |
| `axon` wrong import path | `learning/learning_pipeline.py` | `agent.axon` → `agent.tools.axon` |
| `feedback_channel` wrong import path | `learning/learning_pipeline.py` | `agent.feedback_channel` → `agent.tools.feedback_channel` |
| `ToolGuard` missing class | `execution/agent_runner.py` | Created stub in `capabilities/registry/tool_validation.py` |
| Top-level Jotty `__init__.py` stale paths | `Jotty/__init__.py` | Updated 5 orchestration lazy paths |
