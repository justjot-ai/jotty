# RL Natural Dependencies - Debug Findings

**Date**: 2026-01-17
**Test**: `test_rl_natural_dependencies.py` + `test_rl_natural_deps_debug.py`

---

## ✅ What We Fixed

### 1. Phase 7 Refactoring Bug (`single_agent_orchestrator.py`)

**Problem**: Method `_run_actor()` used `self.actor` but attribute was renamed to `self.agent` in Phase 7.

**Location**: `core/orchestration/single_agent_orchestrator.py:1468-1586`

**Fix**: Changed all references from `self.actor` to `self.agent`:
- Line 1472: `self.actor.__class__.__name__` → `self.agent.__class__.__name__`
- Line 1473: `asyncio.iscoroutinefunction(self.actor)` → `asyncio.iscoroutinefunction(self.agent)`
- Line 1506: `hasattr(self.actor, 'forward')` → `hasattr(self.agent, 'forward')`
- Line 1555: `await self.actor(**agent_kwargs)` → `await self.agent(**agent_kwargs)`
- Line 1559: `lambda: self.actor(**agent_kwargs)` → `lambda: self.agent(**agent_kwargs)`

**Result**: Agents NOW execute successfully! ✅

---

## ✅ What We Proved

### 1. Agents ARE Executing

**Evidence**:
```
INFO [__main__] 🔍 VISUALIZER AGENT CALLED
ERROR [__main__] ❌ VISUALIZER FAILING: No summary available!
INFO [__main__] ❌ VISUALIZER returning: chart='', success=False

INFO [__main__] 🔍 PROCESSOR AGENT CALLED
ERROR [__main__] ❌ PROCESSOR FAILING: No sales_data available!
INFO [__main__] ❌ PROCESSOR returning: summary='', success=False

INFO [__main__] 🔍 FETCHER AGENT CALLED
INFO [__main__] ✅ FETCHER returning: sales_data=..., success=True
```

### 2. Natural Dependencies ARE Working

Agents correctly detect missing data:
- **Visualizer**: Checks for `summary`, fails if missing ✅
- **Processor**: Checks for `sales_data`, fails if missing ✅
- **Fetcher**: No dependencies, always succeeds ✅

**Agent Code Works Correctly**:
```python
class ProcessorAgentDebug(dspy.Module):
    def forward(self, **kwargs):
        sales_data = kwargs.get('sales_data', '')

        if not sales_data or sales_data == '':
            return dspy.Prediction(
                summary='',
                success=False,
                _reasoning="ERROR: Cannot process - no sales_data available!"
            )

        summary = f"Sales Summary: $1M in Q1 for US region"
        return dspy.Prediction(summary=summary, success=True)
```

---

## ❌ Remaining Issues

### Issue 1: Agents Receive Empty kwargs

**Problem**: All agents receive empty kwargs, no data flow between agents.

**Evidence**:
```
INFO [__main__] Received kwargs keys: []
INFO [__main__] 📊 sales_data value: '' (type: <class 'str'>)
INFO [__main__] 📊 summary value: '' (type: <class 'str'>)
```

**Why This Happens**:
1. Fetcher produces `sales_data` in its Prediction
2. IOManager registers output: `📦 Registered output from 'Fetcher': 0 fields`
3. But data is NOT passed to next agent's kwargs
4. Processor receives empty kwargs → fails

**Root Cause**: Data flow between agents via SharedContext/IOManager not working.

**Expected Behavior**:
1. Fetcher returns `Prediction(sales_data="...", success=True)`
2. IOManager extracts `sales_data` field
3. SharedContext stores `sales_data`
4. Processor kwargs should include `sales_data="..."`
5. Processor succeeds
6. Visualizer receives `summary` from Processor
7. Visualizer succeeds

**Current Behavior**:
1. Fetcher returns `Prediction(sales_data="...", success=True)` ✅
2. IOManager extracts: `📦 Registered output from 'Fetcher': 0 fields` ❌
3. Processor receives: `kwargs keys: []` ❌
4. Processor fails
5. All agents fail except Fetcher

---

### Issue 2: Tasks Marked as COMPLETED Despite Agent Failures

**Problem**: Agents return `success=False` but tasks are marked as `COMPLETED`.

**Evidence**:
```
# Agent fails:
ERROR [__main__] ❌ PROCESSOR FAILING: No sales_data available!
INFO [__main__] ❌ PROCESSOR returning: summary='', success=False

# But task marked as completed:
Execution order:
  ✅ Processor: COMPLETED

Overall success: True
```

**Why This Happens**:
1. Agent returns `Prediction(success=False)`
2. `SingleAgentOrchestrator` wraps it in `EpisodeResult(success=False)`
3. But task status in MarkovianTODO is set to `COMPLETED` regardless
4. Overall episode success is `True`

**Root Cause**: Task completion logic doesn't check agent `success` field.

**Expected Behavior**:
- If agent returns `Prediction(success=False)` → task status = `FAILED`
- Episode success = `False`
- Q-learning receives negative reward

**Current Behavior**:
- Agent returns `Prediction(success=False)` ✅
- Task status = `COMPLETED` ❌
- Episode success = `True` ❌
- Q-learning receives positive reward ❌
- No differentiation between agents
- Q-values stay identical
- Ordering never improves

---

### Issue 3: RL Not Learning from Failures

**Problem**: Ordering doesn't change over 15 episodes (Visualizer first 100% of time).

**Evidence**:
```
🎯 First Agent Selected Per Episode:
   Episode  1: Visualizer
   Episode  2: Visualizer
   Episode  3: Visualizer
   ...
   Episode 15: Visualizer

📊 Agent Selection Frequency:
   Visualizer: 15/15 episodes (100%)
   Processor: 0/15 episodes (0%)
   Fetcher: 0/15 episodes (0%)
```

**Why This Happens**: Chain reaction from Issues 1 & 2:
1. All agents receive empty kwargs
2. All agents fail (Visualizer, Processor) or succeed (Fetcher)
3. But all tasks marked as COMPLETED
4. All episodes marked as success=True
5. All agents get similar rewards
6. Q-values don't diverge
7. Selection remains random/sticky

**Expected Learning Progression** (with 50+ episodes):
```
Episodes 1-10:  Q-values diverging
  Visualizer: 0.50 → 0.42 (often fails when run first)
  Fetcher: 0.50 → 0.65 (succeeds, provides data)
  Processor: 0.50 → 0.58 (succeeds after Fetcher)

Episodes 11-30: Preference emerging
  Visualizer: 0.42 → 0.35 (decreasing)
  Fetcher: 0.65 → 0.78 (increasing - selected first more often)
  Processor: 0.58 → 0.70 (increasing after Fetcher)

Episodes 31+: Learned optimal order
  Fetcher selected first 90%+ of time
  Processor selected second
  Visualizer selected last
```

---

## 🔧 Fixes Needed

### Fix 1: Data Flow Between Agents

**Location**: Conductor or IOManager data passing

**Current**:
```python
# Fetcher produces output
result = await actor.arun(**kwargs)
# IOManager registers output
self.io_manager.register_output(actor_name, result.output)
# But data NOT passed to next agent
```

**Needed**:
```python
# Extract fields from Prediction
if isinstance(result.output, dspy.Prediction):
    for key, value in result.output._store.items():
        if not key.startswith('_'):  # Skip internal fields
            shared_context[key] = value

# Pass to next agent
next_kwargs = {
    **shared_context,  # Include previous agents' outputs
    **original_kwargs
}
result = await next_actor.arun(**next_kwargs)
```

### Fix 2: Task Status Based on Agent Success

**Location**: Task status update after agent execution

**Current**:
```python
# Task always marked as completed
task.status = TaskStatus.COMPLETED
```

**Needed**:
```python
# Check agent success field
if result.success:
    task.status = TaskStatus.COMPLETED
else:
    task.status = TaskStatus.FAILED
```

### Fix 3: Q-Learning Rewards Based on Success

**Current**:
```python
# All episodes get positive reward
reward = 1.0
```

**Needed**:
```python
# Reward based on actual success
if result.success:
    reward = 1.0
else:
    reward = -0.5  # Negative reward for failures
```

---

## 🎯 Summary

### What Works ✅:
1. Agents execute (after Phase 7 fix)
2. Natural dependencies correctly implemented
3. Agents detect missing data and fail
4. Q-value selection infrastructure runs
5. RL independence fix (tasks don't have sequential dependencies)

### What Doesn't Work ❌:
1. Data flow between agents (all receive empty kwargs)
2. Task status (marked COMPLETED even when agent fails)
3. Q-learning rewards (don't reflect actual success/failure)
4. Ordering improvement (Visualizer first 100% of time)

### Next Steps:
1. Fix data flow (highest priority - enables natural learning)
2. Fix task status based on agent success
3. Verify Q-learning receives proper rewards
4. Run 50+ episode test to see ordering improve

---

**Generated**: 2026-01-17
**Status**: Agents executing ✅, Natural dependencies implemented ✅, Data flow needs fix ❌
