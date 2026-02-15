# Technical Debt Analysis - Jotty

## 🔴 CRITICAL (Must Fix - Runtime Bugs)

### 1. Undefined Name Errors (2 instances)
**File:** `core/modes/execution/executor.py`
**Lines:** 474, 1432
**Error:** `F821 undefined name 'SwarmConfig'`
**Impact:** Runtime crash when code path is hit
**Effort:** 5 minutes
**Fix:**
```python
# Add import at top of file
from Jotty.core.infrastructure.foundation.data_structures import SwarmConfig
```

---

## 🟠 HIGH (Should Fix Soon - Quality Issues)

### 2. Deprecated SwarmConfig Inheritance (5 files)
**Files:**
- `core/intelligence/swarms/perspective_learning_swarm/types.py:69`
- `core/intelligence/swarms/coding_swarm/types.py:52`
- `core/intelligence/swarms/arxiv_learning_swarm/types.py:68`
- `core/intelligence/swarms/olympiad_learning_swarm/types.py:72`
- `core/intelligence/swarms/pilot_swarm/types.py:55`

**Impact:** Using deprecated API, will break in future
**Effort:** 15 minutes (5 files × 3 min each)
**Fix:**
```python
# Change from:
class MyConfig(SwarmConfig):
    ...

# To:
class MyConfig(SwarmBaseConfig):
    ...
```

### 3. Bare Except Clauses (2 instances)
**File:** `apps/telegram/bot.py`
**Lines:** 94, 141
**Error:** `E722 do not use bare 'except'`
**Impact:** Can hide bugs, catches SystemExit/KeyboardInterrupt
**Effort:** 2 minutes
**Fix:**
```python
# Change from:
except:
    ...

# To:
except Exception as e:
    logger.error(f"Error: {e}")
    ...
```

### 4. Import Shadowing (3 instances)
**Files:**
- `core/infrastructure/monitoring/evaluation/gaia_adapter.py:489` (ExecutionConfig)
- `core/modes/agent/planners/agentic_planner.py:331,454` (field)

**Impact:** Can cause confusion, potential bugs
**Effort:** 5 minutes
**Fix:**
```python
# Change loop variable name to avoid shadowing import
for execution_config in configs:  # was: for ExecutionConfig in configs
```

---

## 🟡 MEDIUM (Nice to Have - Code Cleanliness)

### 5. Unused Imports (18 instances)
**Impact:** Cluttered code, slower imports
**Effort:** 10 minutes (automated with tools)

**Files:**
- `apps/cli/app.py`: Path, prompt_toolkit, HistoryManager
- `apps/telegram/bot.py`: (none, just ordering issues)
- `core/capabilities/registry/skill_generator.py`: json (imported 3x), dspy
- `core/infrastructure/monitoring/evaluation/gaia_adapter.py`: re
- `core/intelligence/orchestration/direct_chat_executor.py`: Dict
- `core/intelligence/orchestration/validation_gate.py`: field, List
- `core/interface/api/mode_router.py`: ResponseFormat
- `core/modes/agent/planners/agentic_planner.py`: ExecutionTrajectory
- `core/modes/execution/executor.py`: re, SwarmLearningConfig

**Fix:**
```bash
# Automated cleanup
autoflake --remove-all-unused-imports --in-place <files>
```

### 6. Import Ordering (E402) (8 instances)
**Impact:** PEP 8 violation, minor readability issue
**Effort:** 5 minutes (automated)

**Files:**
- `apps/cli/app.py`: Lines 21-29, 33
- `apps/telegram/bot.py`: Lines 20, 21, 28, 32
- `core/interface/api/mode_router.py`: Line 32
- `core/modes/agent/planners/agentic_planner.py`: Lines 109-111, 203, 205

**Fix:**
```bash
# Automated cleanup
isort <files>
```

### 7. F-strings Missing Placeholders (11 instances)
**Impact:** Misleading syntax, should be regular strings
**Effort:** 5 minutes

**Files:**
- `core/infrastructure/monitoring/evaluation/gaia_adapter.py`: Lines 646, 654, 776
- `core/modes/agent/planners/agentic_planner.py`: Lines 214, 223, 534, 556, 623, 684, 695
- `core/modes/execution/executor.py`: Lines 996, 1386

**Fix:**
```python
# Change from:
logger.info(f"Starting process")  # No {} placeholders!

# To:
logger.info("Starting process")
```

### 8. Unused Variables (2 instances)
**Files:**
- `core/modes/execution/executor.py:1509`: variable 'si'
- `core/infrastructure/monitoring/evaluation/gaia_signatures.py:19`: redefinition

**Impact:** Dead code
**Effort:** 2 minutes

---

## 🟢 LOW (Optional - Style/Convention)

### 9. Wildcard Import (1 instance)
**File:** `core/infrastructure/foundation/types/__init__.py:18`
**Impact:** Makes dependencies unclear
**Effort:** 10 minutes (need to check what's actually used)

---

## 🧪 Test Failures (4 tests)

From MEMORY.md, pre-existing failures:
- `test_tier4_swarm_delegation`
- `test_swarm_returns_failure_result`
- `test_phase7_terminology.py`
- `test_phase8_expert_integration.py`

**Impact:** Unknown (tests might be outdated or reveal real bugs)
**Effort:** 1-2 hours (investigate each)

---

## Summary by Priority

| Priority | Count | Effort | Description |
|----------|-------|--------|-------------|
| 🔴 CRITICAL | 2 | 5 min | Undefined names (runtime crashes) |
| 🟠 HIGH | 10 | 25 min | Deprecated APIs, bare excepts, shadowing |
| 🟡 MEDIUM | 37 | 20 min | Unused imports, ordering, f-strings |
| 🟢 LOW | 1 | 10 min | Wildcard import |
| 🧪 TESTS | 4 | 2 hrs | Failing tests |

**Total Effort:** ~3 hours to clean all technical debt

---

## Recommended Cleanup Order

### Phase 1: Critical Bugs (5 minutes)
```bash
# Fix undefined SwarmConfig in executor.py
```

### Phase 2: High Priority (25 minutes)
```bash
# Fix deprecated SwarmConfig inheritance (5 files)
# Fix bare except clauses (2 instances)
# Fix import shadowing (3 instances)
```

### Phase 3: Automated Cleanup (15 minutes)
```bash
# Remove unused imports
autoflake --remove-all-unused-imports --remove-unused-variables --in-place core/**/*.py apps/**/*.py

# Fix import ordering
isort core/ apps/

# Fix f-strings (manual or with tool)
```

### Phase 4: Test Investigation (2 hours)
```bash
# Investigate 4 failing tests
pytest tests/test_tier4_swarm_delegation.py -v
# ... etc
```

