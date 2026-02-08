# Modular Jotty Architecture: From Simple to Advanced

## Executive Summary

**Current State**: Jotty has 143,780 lines across 205 files - all features mandatory
**Target State**: Modular system from 1K lines (basic) to 50K lines (advanced)
**Inspiration**: AIME (2.3K lines, Hydra configs), MegaAgent (1K lines, prompts)

**Strategy**: Make Jotty configurable like a high-end car:
- **Basic** (1-5K lines): Multi-agent coordination only (MegaAgent equivalent)
- **Standard** (10-15K lines): + Memory + Task management
- **Premium** (25-35K lines): + Reinforcement Learning + Tool discovery
- **Advanced** (50-70K lines): All features enabled

---

## Complexity Breakdown by Module

### Current Jotty (143,780 lines)

| Module | Lines | % of Total | Category | Essential? |
|--------|-------|------------|----------|------------|
| **orchestration** | 20,071 | 14.0% | Core | ✅ Yes (but bloated) |
| **experts** | 6,326 | 4.4% | Domain-specific | ❌ Optional |
| **learning** | 6,198 | 4.3% | Advanced | ❌ Optional |
| **memory** | 5,208 | 3.6% | Advanced | ⚠️ Partial |
| **foundation** | 4,373 | 3.0% | Core | ✅ Yes |
| **data** | 3,425 | 2.4% | Core | ✅ Yes |
| **metadata** | 3,298 | 2.3% | Advanced | ❌ Optional |
| **agents** | 3,277 | 2.3% | Core | ✅ Yes |
| **server** | 2,899 | 2.0% | Infrastructure | ⚠️ Optional |
| **context** | 2,821 | 2.0% | Advanced | ❌ Optional |
| **integration** | 2,392 | 1.7% | Connectors | ❌ Optional |
| **utils** | 2,170 | 1.5% | Core | ✅ Yes |
| **use_cases** | 1,843 | 1.3% | Examples | ❌ Optional |
| **presets** | 1,677 | 1.2% | Templates | ❌ Optional |
| **queue** | 1,557 | 1.1% | Advanced | ❌ Optional |
| **persistence** | 1,355 | 0.9% | Advanced | ❌ Optional |
| **tools** | 952 | 0.7% | Core | ⚠️ Partial |
| **Other** | ~3,000 | 2.0% | Misc | ⚠️ Mixed |

**Analysis**:
- **Essential Core**: ~15K lines (foundation, data, agents, utils, basic tools)
- **Orchestration Bloat**: 20K lines (should be ~3K)
- **Advanced Features**: ~70K lines (learning, memory, metadata, context, queue, persistence)
- **Optional/Examples**: ~40K lines (experts, use_cases, presets, integration, server)

---

## Feature-Based Complexity Quantification

### Tier 0: Minimal (MegaAgent Equivalent)
**Target**: 1,000-2,000 lines
**Capabilities**:
- ✅ Multi-agent coordination
- ✅ Message passing
- ✅ Dynamic spawning (basic)
- ✅ Simple memory (in-memory dict)

**Required Modules**:
```
foundation/
  ├── agent_config.py         (87 lines)
  └── minimal_types.py        (100 lines - NEW)
agents/
  └── simple_agent.py         (200 lines - NEW)
orchestration/
  ├── simple_orchestrator.py  (500 lines - NEW)
  └── dynamic_spawner.py      (300 lines - EXISTS)
utils/
  └── minimal_utils.py        (200 lines - NEW)
```

**Total**: ~1,500 lines
**Config**: `mode: minimal`

### Tier 1: Basic (Entry Level)
**Target**: 5,000-8,000 lines
**Capabilities**:
- ✅ Everything from Tier 0
- ✅ Basic memory (ChromaDB)
- ✅ Task queue (simple)
- ✅ Tool registration (manual)
- ✅ State persistence (JSON)

**Additional Modules**:
```
memory/
  └── simple_memory.py        (500 lines - NEW)
queue/
  └── simple_queue.py         (300 lines - NEW)
tools/
  └── tool_registry.py        (200 lines - NEW)
persistence/
  └── json_persistence.py     (200 lines - NEW)
```

**Added**: ~1,200 lines
**Total**: ~2,700 lines (from Tier 0)
**Config**: `mode: basic`

### Tier 2: Standard (Most Users)
**Target**: 10,000-15,000 lines
**Capabilities**:
- ✅ Everything from Tier 1
- ✅ Hierarchical memory (3 levels)
- ✅ Markovian TODO with dependencies
- ✅ Parameter resolution
- ✅ Complexity assessment

**Additional Modules**:
```
memory/
  ├── hierarchical_memory.py  (1,500 lines - REFACTORED from cortex.py)
  └── consolidation.py        (500 lines)
orchestration/
  ├── markovian_todo.py       (800 lines - FROM roadmap.py)
  ├── parameter_resolver.py   (500 lines - SIMPLIFIED)
  └── complexity_assessor.py  (400 lines - EXISTS)
context/
  └── context_manager.py      (600 lines - SIMPLIFIED)
```

**Added**: ~4,300 lines
**Total**: ~7,000 lines (cumulative)
**Config**: `mode: standard`

### Tier 3: Premium (Power Users)
**Target**: 25,000-35,000 lines
**Capabilities**:
- ✅ Everything from Tier 2
- ✅ Reinforcement Learning (Q-learning, TD(λ))
- ✅ Tool auto-discovery
- ✅ Advanced state management
- ✅ Brain-inspired memory (5 levels)
- ✅ Credit assignment

**Additional Modules**:
```
learning/
  ├── q_learning.py           (1,200 lines)
  ├── td_lambda.py            (1,000 lines)
  ├── credit_assignment.py    (800 lines)
  └── shaped_rewards.py       (600 lines)
memory/
  ├── sharp_wave_ripple.py    (500 lines)
  └── hippocampal.py          (400 lines)
metadata/
  ├── tool_discovery.py       (800 lines)
  └── metadata_registry.py    (600 lines)
orchestration/
  ├── state_manager.py        (1,000 lines - REFACTORED)
  └── learning_manager.py     (1,200 lines - EXTRACTED from conductor)
```

**Added**: ~8,100 lines
**Total**: ~15,000 lines (cumulative)
**Config**: `mode: premium`

### Tier 4: Advanced (Researchers)
**Target**: 50,000-70,000 lines
**Capabilities**:
- ✅ Everything from Tier 3
- ✅ Multi-agent RL (MARL)
- ✅ Policy exploration
- ✅ Causal reasoning
- ✅ Expert integrations
- ✅ Full orchestration features

**Additional Modules**:
```
learning/
  ├── marl.py                 (1,500 lines)
  ├── policy_explorer.py      (800 lines)
  └── causal_learning.py      (1,000 lines)
experts/
  └── *.py                    (6,326 lines - domain-specific)
integration/
  └── *.py                    (2,392 lines - connectors)
server/
  └── *.py                    (2,899 lines - API server)
use_cases/
  └── *.py                    (1,843 lines - examples)
```

**Added**: ~15,000 lines
**Total**: ~30,000 lines (cumulative)
**Config**: `mode: advanced`

---

## Lessons from AIME (argmax-ai)

### What AIME Does Well

1. **Hydra-Based Configuration** 🏆
   ```yaml
   # aime/configs/aime.yaml
   defaults:
     - _self_
     - env: ???
     - world_model: rssmo

   freeze_model: True
   use_fp16: false
   batch_size: 50
   ```

   **Lesson**: Use Hydra for hierarchical, composable configs

2. **Minimal Core (2,313 lines)** 🏆
   - actor.py (173 lines)
   - env.py (241 lines)
   - data.py (273 lines)
   - models/ssm.py (913 lines)
   - dist.py (139 lines)

   **Lesson**: Focus on essential abstractions only

3. **Separate Scripts for Different Use Cases** 🏆
   - train_aime.py
   - train_bc.py
   - train_dreamer.py
   - Each ~150-300 lines

   **Lesson**: Don't put everything in one orchestrator

4. **Clear Module Boundaries** 🏆
   - Models (policy, ssm, base)
   - Environment wrapper
   - Data handling
   - Distribution/logging

   **Lesson**: Single responsibility principle

5. **Environment Abstraction** 🏆
   ```python
   # Supports multiple environments via config
   env=walker, cheetah, hopper, etc.
   environment_setup=mdp, lpomdp, visual
   ```

   **Lesson**: Abstract domains, don't hardcode

### What We Can Adopt

1. **Hydra Configs** instead of Python config classes
2. **Separate scripts** for different workflows (not one mega-orchestrator)
3. **Environment abstraction** for different domains
4. **Minimal core** with optional extensions
5. **Clear module boundaries** with single responsibility

---

## Proposed Modular Architecture

### Directory Structure

```
Jotty/
├── configs/                    # Hydra configs (YAML)
│   ├── mode/
│   │   ├── minimal.yaml        # Tier 0
│   │   ├── basic.yaml          # Tier 1
│   │   ├── standard.yaml       # Tier 2
│   │   ├── premium.yaml        # Tier 3
│   │   └── advanced.yaml       # Tier 4
│   ├── memory/
│   │   ├── none.yaml
│   │   ├── simple.yaml
│   │   ├── hierarchical.yaml
│   │   └── brain_inspired.yaml
│   ├── learning/
│   │   ├── none.yaml
│   │   ├── q_learning.yaml
│   │   ├── td_lambda.yaml
│   │   └── marl.yaml
│   └── orchestration/
│       ├── simple.yaml
│       ├── markovian.yaml
│       └── full.yaml
│
├── core/
│   ├── minimal/                # Tier 0 (1-2K lines)
│   │   ├── agent.py
│   │   ├── orchestrator.py
│   │   ├── spawner.py
│   │   └── memory.py
│   │
│   ├── basic/                  # Tier 1 (added ~3K lines)
│   │   ├── queue.py
│   │   ├── tools.py
│   │   └── persistence.py
│   │
│   ├── standard/               # Tier 2 (added ~5K lines)
│   │   ├── hierarchical_memory.py
│   │   ├── markovian_todo.py
│   │   ├── parameter_resolver.py
│   │   └── complexity_assessor.py
│   │
│   ├── premium/                # Tier 3 (added ~10K lines)
│   │   ├── learning/
│   │   │   ├── q_learning.py
│   │   │   ├── td_lambda.py
│   │   │   └── credit_assignment.py
│   │   ├── memory/
│   │   │   └── brain_inspired.py
│   │   └── metadata/
│   │       └── tool_discovery.py
│   │
│   └── advanced/               # Tier 4 (added ~15K lines)
│       ├── learning/
│       │   ├── marl.py
│       │   └── policy_explorer.py
│       ├── experts/
│       └── integration/
│
├── scripts/                    # Use-case specific (like AIME)
│   ├── run_simple.py          # Tier 0 example
│   ├── run_guide_generator.py # Tier 2 example
│   ├── run_rl_swarm.py        # Tier 3 example
│   └── run_full_system.py     # Tier 4 example
│
└── tests/                      # Tests for each tier
    ├── test_minimal/
    ├── test_basic/
    ├── test_standard/
    ├── test_premium/
    └── test_advanced/
```

### Configuration System

**Example: configs/mode/minimal.yaml**
```yaml
defaults:
  - _self_
  - memory: none
  - learning: none
  - orchestration: simple

mode: minimal
max_agents: 10
enable_spawning: true
enable_learning: false
enable_persistence: false
```

**Example: configs/mode/standard.yaml**
```yaml
defaults:
  - _self_
  - memory: hierarchical
  - learning: none
  - orchestration: markovian

mode: standard
max_agents: 50
enable_spawning: true
enable_learning: false
enable_persistence: true
enable_complexity_assessment: true
```

**Example: configs/mode/premium.yaml**
```yaml
defaults:
  - _self_
  - memory: brain_inspired
  - learning: td_lambda
  - orchestration: full

mode: premium
max_agents: 100
enable_spawning: true
enable_learning: true
enable_persistence: true
enable_complexity_assessment: true
enable_tool_discovery: true
```

### Usage Examples

**Minimal Mode** (MegaAgent equivalent):
```python
from jotty import Orchestrator
from hydra import compose, initialize

with initialize(config_path="configs"):
    cfg = compose(config_name="mode/minimal")

# Only ~1.5K lines loaded!
orchestrator = Orchestrator(cfg)
result = orchestrator.run(
    goal="Write hello world",
    agents=[SimpleAgent("writer")]
)
```

**Standard Mode** (Most users):
```python
with initialize(config_path="configs"):
    cfg = compose(config_name="mode/standard")

# ~7K lines loaded (includes memory, TODO, assessment)
orchestrator = Orchestrator(cfg)
result = orchestrator.run(
    goal="Generate 10-section guide",
    agents=[PlannerAgent(), ResearcherAgent()]
)
```

**Premium Mode** (RL enabled):
```python
with initialize(config_path="configs"):
    cfg = compose(config_name="mode/premium")

# ~15K lines loaded (includes RL, brain memory, tool discovery)
orchestrator = Orchestrator(cfg)

# Run multiple episodes to learn
for episode in range(100):
    result = orchestrator.run(
        goal="Complex task",
        agents=[...],
        enable_learning=True
    )
    # Q-values improve over time
```

---

## Testing Strategy

### Tier-Based Testing

**Tier 0 (Minimal) - 100% coverage required**
```python
# tests/test_minimal/test_orchestrator.py
def test_simple_coordination():
    """Ensure basic multi-agent works"""
    orchestrator = MinimalOrchestrator()
    result = orchestrator.run(goal="Hello", agents=[...])
    assert result.success

def test_spawning():
    """Ensure dynamic spawning works"""
    spawner = DynamicSpawner()
    agent = spawner.spawn(name="worker", ...)
    assert agent is not None
```

**Tier 1 (Basic) - 90% coverage required**
```python
# tests/test_basic/test_queue.py
def test_task_queue():
    """Ensure task queue works"""
    queue = SimpleQueue()
    queue.add_task(Task(...))
    task = queue.get_next()
    assert task is not None
```

**Tier 2 (Standard) - 80% coverage required**
```python
# tests/test_standard/test_markovian_todo.py
def test_dependency_tracking():
    """Ensure task dependencies work"""
    todo = MarkovianTODO()
    todo.add_task("task1", depends_on=[])
    todo.add_task("task2", depends_on=["task1"])
    next_task = todo.get_next()
    assert next_task.id == "task1"
```

**Tier 3+ - 70% coverage**
- RL tests can be stochastic
- Focus on integration tests
- Performance benchmarks

### Coverage Targets

| Tier | Coverage | Test Files | Estimated Tests |
|------|----------|------------|-----------------|
| Minimal | 100% | 5-10 | 50-100 |
| Basic | 90% | 10-15 | 100-150 |
| Standard | 80% | 15-25 | 150-250 |
| Premium | 70% | 25-40 | 250-400 |
| Advanced | 60% | 40-60 | 400-600 |

---

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
**Goal**: Create Hydra config system + Tier 0 (Minimal)

**Tasks**:
1. Install Hydra (`pip install hydra-core`)
2. Create config structure (configs/ directory)
3. Extract minimal orchestrator from conductor.py
4. Create simple agent wrapper
5. Test minimal mode (100% coverage)

**Deliverable**: Working minimal Jotty (1.5K lines) with tests

### Phase 2: Tiered Modules (Weeks 3-4)
**Goal**: Split existing code into tiers

**Tasks**:
1. Create Tier 1 modules (queue, tools, persistence)
2. Extract Tier 2 modules (hierarchical memory, markovian TODO)
3. Organize Tier 3 modules (learning, metadata)
4. Separate Tier 4 modules (experts, integration)
5. Update imports to be conditional

**Deliverable**: 4 working tiers with separate configs

### Phase 3: Refactor Conductor (Weeks 5-6)
**Goal**: Break 20K-line conductor into composable managers

**Tasks**:
1. Extract LearningManager (loaded only in Tier 3+)
2. Extract ValidationManager (loaded in Tier 2+)
3. Extract ExecutionManager (loaded always)
4. Create tier-aware Orchestrator factory
5. Remove circular dependencies

**Deliverable**: Modular conductor <3K lines core

### Phase 4: Testing Infrastructure (Weeks 7-8)
**Goal**: Comprehensive tests for all tiers

**Tasks**:
1. Write Tier 0 tests (100% coverage)
2. Write Tier 1 tests (90% coverage)
3. Write Tier 2 tests (80% coverage)
4. Write integration tests
5. Performance benchmarks per tier

**Deliverable**: 400+ tests across all tiers

### Phase 5: Documentation (Week 9)
**Goal**: Clear docs for each tier

**Tasks**:
1. Quick start guides per tier
2. Configuration reference
3. Migration guide (old → new)
4. Performance comparison
5. Example notebooks

**Deliverable**: Complete documentation

### Phase 6: Deprecation & Cleanup (Week 10)
**Goal**: Remove old monolithic code

**Tasks**:
1. Mark old imports as deprecated
2. Update existing code to use new tiers
3. Remove duplicates
4. Clean up 40K lines of optional code
5. Final benchmarks

**Deliverable**: Clean, modular Jotty

---

## Performance Targets

| Tier | Import Time | Memory | Use Case | Test Suite Time |
|------|-------------|--------|----------|-----------------|
| Minimal | <0.5s | <50MB | Quick scripts | <10s |
| Basic | <1s | <100MB | Simple apps | <30s |
| Standard | <2s | <200MB | Production | <60s |
| Premium | <5s | <500MB | Research | <2min |
| Advanced | <10s | <1GB | Full features | <5min |

---

## Success Metrics

### Code Metrics
- ✅ Minimal tier: 1-2K lines (vs current 143K)
- ✅ Standard tier: ~7K lines (vs current 143K)
- ✅ Clear module boundaries (vs circular dependencies)
- ✅ Each tier fully functional (vs all-or-nothing)

### User Metrics
- ✅ Beginner can use Minimal tier in <1 hour
- ✅ Most users happy with Standard tier
- ✅ Advanced users can enable features incrementally
- ✅ 90% of users use <50% of features

### Testing Metrics
- ✅ 100% coverage on Minimal tier
- ✅ 80%+ coverage on Standard tier
- ✅ All tiers have passing tests
- ✅ Performance benchmarks green

---

## Migration Path

**For existing Jotty users**:

1. **Immediate** (no code changes):
   ```python
   # Old way still works (backward compatible)
   from core.orchestration.conductor import MultiAgentsOrchestrator
   orchestrator = MultiAgentsOrchestrator(...)  # Loads everything (143K lines)
   ```

2. **Transition** (opt-in to configs):
   ```python
   # New way (choose your tier)
   from jotty import Orchestrator
   orchestrator = Orchestrator.from_config("standard")  # Loads ~7K lines
   ```

3. **Future** (deprecated warnings):
   ```python
   # Old imports show deprecation warning
   from core.orchestration.conductor import MultiAgentsOrchestrator
   # DeprecationWarning: Use Orchestrator.from_config() instead
   ```

4. **End state** (old code removed):
   ```python
   # Only new API available
   from jotty import Orchestrator
   ```

---

## Comparison: Before vs After

| Metric | Before | After (Standard) | After (Minimal) |
|--------|--------|------------------|-----------------|
| **Lines loaded** | 143,780 | 7,000 | 1,500 |
| **Import time** | 10s | 2s | 0.5s |
| **Memory** | 1GB+ | 200MB | 50MB |
| **Test time** | 10min+ | 60s | 10s |
| **Learning curve** | 2-4 weeks | 2-3 days | 2-3 hours |
| **Code clarity** | Poor | Good | Excellent |
| **Modularity** | None | High | Perfect |
| **Flexibility** | All-or-nothing | Configurable | Minimal |

---

## Next Steps

**Immediate**:
1. Get user approval for this plan
2. Create feature branch: `feature/modular-architecture`
3. Install Hydra
4. Start Phase 1 (Minimal tier extraction)

**This Week**:
- Extract minimal orchestrator
- Create Tier 0 configs
- Write Tier 0 tests
- Prove 1.5K lines works

**This Month**:
- Complete all 4 tiers
- Refactor conductor
- 400+ tests
- Documentation

**This Quarter**:
- Deprecate old code
- Remove 100K+ unnecessary lines
- Release Jotty 2.0 (modular)

---

## Open Questions for User

1. **Which tier should be default?**
   - Minimal (simplest, like MegaAgent)?
   - Standard (most features, recommended)?
   - Let user choose on first import?

2. **Backward compatibility duration?**
   - Keep old API for 6 months?
   - 1 year?
   - Forever with deprecation warnings?

3. **Testing priority?**
   - Start with Minimal (100% coverage)?
   - Or cover all tiers at 60% first?

4. **Feature flags vs separate imports?**
   - `if cfg.enable_learning:` in code?
   - Or separate import paths per tier?

5. **Should we rename project?**
   - Jotty 2.0?
   - Jotty Modular?
   - Keep "Jotty" but with tiers?

---

## Summary

**Current Problem**: Jotty is 143K lines, all mandatory, hard to learn
**Root Cause**: No modularity, features are entangled
**Solution**: Tiered architecture with Hydra configs
**Inspiration**: AIME (2.3K modular), MegaAgent (1K simple)

**Deliverable**:
- Tier 0: 1.5K lines (MegaAgent equivalent)
- Tier 2: 7K lines (recommended for most users)
- Tier 4: 30K lines (full featured, down from 143K)

**Benefit**: Users choose complexity level, code is cleaner, tests are faster, learning curve is gentler.

**Risk**: Large refactoring effort (~10 weeks)
**Mitigation**: Backward compatible, incremental rollout, thorough testing

**Ready to proceed?**
