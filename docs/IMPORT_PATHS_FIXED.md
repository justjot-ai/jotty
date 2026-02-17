# Import Path Fixes - dag_agents Module

**Date:** 2026-02-16
**Issue:** Incorrect import paths for dag_agents module
**Resolution:** Fixed all import paths to use correct location

---

## Problem

Multiple files were importing from incorrect path:
```python
# ❌ WRONG
from Jotty.core.intelligence.reasoning.dag_agents import SwarmResources
from ..agent.dag_agents import SwarmResources
```

**Actual location:**
```
core/intelligence/reasoning/planners/dag_agents.py
```

---

## Files Fixed (8 files)

### Absolute Imports (6 files)
1. `core/intelligence/swarms/research_swarm/swarm.py:138`
2. `core/intelligence/swarms/templates/research.py:154`
3. `core/intelligence/orchestration/swarm_dag_executor.py:52`
4. `core/intelligence/orchestration/swarm_dag_executor.py:244`
5. `core/execution/swarms/research_swarm/swarm.py:139`
6. `core/execution/swarms/templates/research.py:154`

### Relative Imports (2 files)
7. `core/intelligence/swarms/_base/swarm_learning.py:185`
8. `core/execution/swarms/_base/swarm_learning.py:185`

---

## Solution

**Correct import:**
```python
# ✅ CORRECT
from Jotty.core.intelligence.reasoning.planners.dag_agents import SwarmResources
from Jotty.core.intelligence.reasoning.planners.dag_agents import TaskBreakdownAgent, TodoCreatorAgent
```

---

## Changes Made

```bash
# Fixed absolute imports (6 files)
sed -i 's|from Jotty.core.intelligence.reasoning.dag_agents import|from Jotty.core.intelligence.reasoning.planners.dag_agents import|g' \
  core/intelligence/swarms/research_swarm/swarm.py \
  core/intelligence/swarms/templates/research.py \
  core/intelligence/orchestration/swarm_dag_executor.py \
  core/execution/swarms/research_swarm/swarm.py \
  core/execution/swarms/templates/research.py

# Fixed relative imports (2 files)
sed -i 's|from ..agent.dag_agents import|from Jotty.core.intelligence.reasoning.planners.dag_agents import|g' \
  core/intelligence/swarms/_base/swarm_learning.py \
  core/execution/swarms/_base/swarm_learning.py
```

---

## Verification

```bash
$ grep -rn "from.*dag_agents import" core/ --include="*.py" | grep -v "planners.dag_agents"
# No results = all fixed ✅
```

---

## Why This Matters

1. **No symlink hacks** - Proper imports instead of workarounds
2. **Correct module resolution** - Python can find the actual file
3. **Maintainable** - Clear, explicit import paths
4. **IDE-friendly** - IDEs can navigate to definitions
5. **Testing-ready** - Tests can import without errors

---

## Root Cause

The `dag_agents.py` file was moved from:
- `core/intelligence/reasoning/dag_agents.py` (old location)

To:
- `core/intelligence/reasoning/planners/dag_agents.py` (new location)

But imports were not updated throughout the codebase.

---

## Impact

**Before:**
- ❌ `ModuleNotFoundError: No module named 'Jotty.core.intelligence.reasoning.dag_agents'`
- ❌ Tests fail
- ❌ Swarms fail to initialize

**After:**
- ✅ Imports resolve correctly
- ✅ Tests can run
- ✅ Swarms can initialize (if other dependencies met)

---

## Follow-up Actions

Still need to address:
1. ✅ **Import paths** - FIXED
2. ⚠️ **API keys** - Need ANTHROPIC_API_KEY or OPENAI_API_KEY for LLM
3. ⚠️ **Method signatures** - Need to use correct swarm APIs
4. ⚠️ **Dependencies** - SwarmConfig import needed in some files

**Next:** Create proper test that uses correct swarm APIs with available API keys.
