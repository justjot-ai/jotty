# Swarm Issues and Fixes

**Date:** 2026-02-16
**Status:** In Progress

## Critical Issues Fixed

### 1. ResearchSwarm - Missing Agent Initialization ✅ FIXED

**Problem:**
```python
AttributeError: 'ResearchSwarm' object has no attribute '_data_fetcher'
```

**Root Cause:**
- Swarm defined AGENT_TEAM but never called `_init_agents()`
- All 10 agents (_data_fetcher, _web_searcher, etc.) were undefined

**Fix Applied:**
```python
# File: core/execution/swarms/research_swarm/swarm.py
# Line: 189 (in research method)

async def research(...):
    # Initialize agents if not already done
    self._init_agents()  # <-- ADDED THIS

    # Parse inputs...
    ticker = ticker or self._extract_ticker(query)
```

**Status:** ✅ Fixed

---

### 2. Import Path Errors (8 files) ✅ FIXED

**Problem:**
```python
ModuleNotFoundError: No module named 'Jotty.core.intelligence.reasoning.dag_agents'
```

**Root Cause:**
- Module moved from `core/intelligence/reasoning/dag_agents.py` to `core/intelligence/reasoning/planners/dag_agents.py`
- 8 swarm files still had old import path

**Files Fixed:**
1. core/execution/swarms/research_swarm/swarm.py
2. core/execution/swarms/templates/research.py
3. core/intelligence/swarms/research_swarm/swarm.py
4. core/intelligence/swarms/templates/research.py
5. core/intelligence/orchestration/swarm_dag_executor.py (2 locations)
6. core/intelligence/swarms/_base/swarm_learning.py
7. core/execution/swarms/_base/swarm_learning.py

**Fix:**
```python
# Before
from Jotty.core.intelligence.reasoning.dag_agents import get_swarm_resources

# After
from Jotty.core.intelligence.reasoning.planners.dag_agents import get_swarm_resources
```

**Status:** ✅ Fixed

---

### 3. Missing AgentRole Import ✅ FIXED

**Problem:**
```python
NameError: name 'AgentRole' is not defined
```

**File:** `core/execution/swarms/research_swarm/swarm.py`

**Fix:**
```python
# Added import at line 18
from .._base.swarm_types import AgentRole
```

**Status:** ✅ Fixed

---

### 4. Syntax Errors - Unmatched Parentheses (5 locations) ✅ FIXED

**Problem:**
```python
SyntaxError: unmatched ')' (swarm_learning.py, line 824)
```

**Root Cause:**
- Return statements commented out but parameters weren't
- Pattern: `# return func(` with `params)` on next lines

**File:** `core/intelligence/swarms/_base/swarm_learning.py`

**Lines Fixed:** 817, 958, 966, 984, 1120

**Fix:**
```python
# Before
# return self._swarm_intelligence.smart_route(
    task_id=task_id,
    task_type=task_type,
)

# After
return self._swarm_intelligence.smart_route(
    task_id=task_id,
    task_type=task_type,
)
```

**Status:** ✅ Fixed

---

### 5. SwarmResources Module Missing ✅ WORKAROUND

**Problem:**
```
SwarmResources not available: No module named 'Jotty.core.intelligence.swarms.agent'
```

**Solution:**
Created minimal stub at `core/intelligence/reasoning/planners/swarm_resources_stub.py`

**Impact:**
- Warning only - swarms work without SwarmResources
- All swarms check `if self._memory:` before using
- Graceful degradation by design

**Status:** ✅ Workaround in place (swarms functional)

---

## Performance Issues

### 6. Slow Import Time - 2+ Seconds ⚠️ DOCUMENTED

**Problem:**
```
OlympiadLearningSwarm import: 2.05s
ArxivLearningSwarm import: ~2s (estimated)
```

**Root Cause Analysis:**

1. **Signatures Module Bottleneck** (2.22s)
   - File: `olympiad_learning_swarm/signatures.py`
   - 792 lines, 12 DSPy Signature classes
   - Each signature class takes ~180ms to create
   - Imported eagerly by `__init__.py`

2. **Agents Module** (2.19s)
   - File: `olympiad_learning_swarm/agents.py`
   - 1,066 lines, 11 agent classes
   - Imports signatures module (cascade delay)

3. **Import Chain:**
   ```
   swarm.__init__.py (0s)
   ├─> agents.py (2.19s)
   │   └─> signatures.py (2.22s)  ← BOTTLENECK
   └─> swarm.py (fast)
   ```

**Impact:**
- Every swarm import adds 2+ seconds startup time
- CLI commands feel slow
- Test scripts take longer to start

**Potential Fixes:**

1. **Lazy Loading** (Recommended)
   ```python
   # In __init__.py - don't import agents/signatures eagerly
   __all__ = ["OlympiadLearningSwarm", "learn_topic"]

   def __getattr__(name):
       if name == "OlympiadLearningSwarm":
           from .swarm import OlympiadLearningSwarm
           return OlympiadLearningSwarm
       # ... lazy load agents on demand
   ```

2. **Move Signatures to Separate Module**
   - Create `signatures/` subdirectory
   - Only import needed signatures per agent
   - Reduce cascade imports

3. **DSPy Signature Optimization**
   - Investigate why DSPy signature creation is slow
   - May be Pydantic field validation overhead
   - Consider caching signature classes

**Status:** ⚠️ Documented, fix deferred

---

## Testing Status

### Swarms Tested with Real LLM

| Swarm | Status | Time | Notes |
|-------|--------|------|-------|
| OlympiadLearningSwarm | ✅ Working | 5+ min | Full multi-phase execution |
| ResearchSwarm | 🔧 Fixed | Pending | Agent init added |
| ArxivLearningSwarm | 📝 Ready | Pending | Correct API identified |
| CodingSwarm | ⏭️ Skipped | - | Complex API, deferred |

### Remaining Swarms (Not Tested)

- TestingSwarm
- ReviewSwarm
- DeploymentSwarm
- DataAnalysisSwarm
- DevOpsSwarm
- DebugSwarm
- MarketingSwarm
- PilotSwarm
- PerspectiveLearningSwarm

---

## Summary

**Bugs Fixed:** 4 critical + 1 workaround = 5 total
**Performance Issues:** 1 documented (slow imports)
**Swarms Validated:** 1 confirmed working (Olympiad)
**Swarms Fixed:** 2 ready for testing (Research, Arxiv)

**Next Steps:**
1. ✅ Complete systematic testing of all 13 swarms
2. 📝 Document slow import fix (lazy loading)
3. 🔧 Fix any additional issues found during testing
4. 📊 Create comprehensive test report

---

**Last Updated:** 2026-02-16
**In Progress:** Systematic testing of all swarms running...
