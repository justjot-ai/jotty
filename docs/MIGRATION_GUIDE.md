# Jotty Migration Guide

This document lists all deprecated APIs and their replacements. All deprecated
APIs still work via backward-compatibility shims but emit warnings. The shims
will be removed in **v3.0**.

---

## 1. Use Cases (Layer 3 → Layer 2)

All execution use cases moved from `core/interface/use_cases/` to `core/modes/use_cases/`.

```python
# OLD (deprecated — emits DeprecationWarning)
from Jotty.core.interface.use_cases import ChatUseCase, WorkflowUseCase

# NEW
from Jotty.core.modes.use_cases import ChatUseCase, WorkflowUseCase
```

**Shim location:** `core/interface/use_cases/__init__.py` (`__getattr__` redirect)

---

## 2. Orchestrator → Jotty SDK

The `Orchestrator` class is deprecated in favor of the SDK client.

```python
# OLD (deprecated — emits DeprecationWarning on __init__)
from Jotty.core.intelligence.orchestration.swarm_manager import Orchestrator
sm = Orchestrator()
result = await sm.run("Research AI trends")

# NEW
from Jotty import Jotty
jotty = Jotty()
result = await jotty.run("Research AI trends")       # auto-detects tier
result = await jotty.swarm("task", swarm_name="coding")  # specific swarm
result = await jotty.autonomous("task")               # full features
```

**Shim location:** `core/intelligence/orchestration/swarm_manager.py`

---

## 3. Learning Manager → Learning Coordinator

```python
# OLD (deprecated — emits warning on import)
from Jotty.core.intelligence.learning.learning_manager import LearningManager

# NEW
from Jotty.core.intelligence.learning.learning_coordinator import LearningManager
# Or use the facade (recommended):
from Jotty.core.intelligence.learning.facade import get_td_lambda
```

**Shim location:** `core/intelligence/learning/learning_manager.py` (re-exports from coordinator)

---

## 4. SwarmBaseConfig → SwarmConfig

```python
# OLD (deprecated — emits DeprecationWarning via __getattr__)
from Jotty.core.intelligence.swarms.swarm_types import SwarmBaseConfig

# NEW
from Jotty.core.intelligence.swarms.swarm_types import SwarmConfig
```

**Shim location:** `core/intelligence/swarms/swarm_types.py`

---

## 5. ExpertAgent → SingleAgentOrchestrator

```python
# OLD (deprecated — emits DeprecationWarning on __init__)
from Jotty.core.intelligence.reasoning.experts.expert_agent import ExpertAgent
agent = ExpertAgent(domain="math")

# NEW
from Jotty.core.intelligence.orchestration import SingleAgentOrchestrator
agent = SingleAgentOrchestrator(enable_gold_standard_learning=True)
```

**Shim location:** `core/intelligence/reasoning/experts/expert_agent.py`
**Migration details:** `docs/PHASE_8_EXPERT_INTEGRATION_PROPOSAL.md`

---

## 6. Registry Functions

```python
# OLD (deprecated — emits logger.warning)
from Jotty.core.capabilities.registry.unified_registry import get_tools_registry
from Jotty.core.capabilities.registry.unified_registry import get_widget_registry

# NEW
from Jotty.core.capabilities.registry import get_unified_registry
registry = get_unified_registry()
skills = registry.skills       # replaces get_tools_registry()
ui = registry.ui               # replaces get_widget_registry()
```

**Shim location:** `core/capabilities/registry/unified_registry.py`

---

## 7. Context Guard → Context Manager

```python
# OLD (deprecated — docstring note)
from Jotty.core.infrastructure.context.facade import get_context_guard

# NEW
from Jotty.core.infrastructure.context.facade import get_context_manager
```

Both return the same `SmartContextManager` singleton.

**Shim location:** `core/infrastructure/context/facade.py`

---

## 8. IntentParser auto_agent Parameter

```python
# OLD (deprecated — parameter ignored)
from Jotty.core.modes.agent.autonomous.intent_parser import IntentParser
parser = IntentParser(auto_agent=some_agent)

# NEW
parser = IntentParser()  # uses TaskPlanner internally
```

**Shim location:** `core/modes/agent/autonomous/intent_parser.py`

---

## 9. Executor skip_swarm_selection Parameter

```python
# OLD (deprecated — parameter ignored with warning)
result = await executor.execute(goal, skip_swarm_selection=True)

# NEW — intent classification handles routing automatically
result = await executor.execute(goal)
```

**Shim location:** `core/modes/execution/executor.py`

---

## Summary

| Deprecated API | Replacement | Shim Type |
|----------------|-------------|-----------|
| `core.interface.use_cases.*` | `core.modes.use_cases.*` | `__getattr__` redirect |
| `Orchestrator` | `Jotty()` SDK client | Warning in `__init__` |
| `LearningManager` (old module) | `learning_coordinator` / facade | Module re-export |
| `SwarmBaseConfig` | `SwarmConfig` | `__getattr__` redirect |
| `ExpertAgent` | `SingleAgentOrchestrator` | Warning in `__init__` |
| `get_tools_registry()` | `get_unified_registry().skills` | Logger warning |
| `get_widget_registry()` | `get_unified_registry().ui` | Logger warning |
| `get_context_guard()` | `get_context_manager()` | Alias function |
| `IntentParser(auto_agent=)` | `IntentParser()` | Parameter ignored |
| `skip_swarm_selection=` | Automatic intent routing | Parameter ignored |

## Removal Timeline

- **Current:** All shims active, warnings emitted
- **v3.0:** All shims removed, old paths will raise `ImportError`
