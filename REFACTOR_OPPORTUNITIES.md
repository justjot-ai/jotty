# Additional Refactor Opportunities

## Date: 2026-02-15

## Overview
After completing Layer 5 and Layer 3→2 cleanup, additional refactoring opportunities exist in:
- `core/interface/` (17 files)
- `core/modes/` (63 files, 2.8MB)
- `sdk/` (9 files)

---

## 🔍 Findings

### 1. **QUICK WIN: Delete Empty `core/modes/chat/` Directory**

**Status:** EMPTY (0 files)
```bash
$ ls -la core/modes/chat/
total 0
drwxr-xr-x. 2 opc opc 6 Feb 15 14:35 .
```

**Action:**
```bash
rmdir core/modes/chat/
```

**Savings:** Cleanup empty directory
**Risk:** Zero (it's empty)

---

### 2. **core/modes/agent/base/** - TOO MANY FILES (20+ files, 1.8MB)

**Current Structure:**
```
core/modes/agent/base/
├── __init__.py
├── section_tools.py           #
├── model_chat_agent.py         #
├── dag_agents.py               #
├── planner_signatures.py       #
├── _skill_selection_mixin.py   #
├── _plan_utils_mixin.py        # 1482 lines!
├── dspy_mcp_agent.py           #
├── auto_agent.py               #
├── _execution_types.py         # 1435 lines!
├── agent_factory.py            # 734 lines
├── agentic_planner.py          # 1011 lines
├── axon.py                     # 830 lines
├── chat_assistant.py           # 695 lines
├── chat_assistant_v2.py        #
├── dag_types.py                #
├── feedback_channel.py         #
├── inspector.py                # 1623 lines
├── skill_based_agent.py        #
├── task_breakdown_agent.py     #
├── base_agent.py               # 880 lines
├── autonomous_agent.py         # 905 lines
├── step_processors.py          # 934 lines
└── skill_plan_executor.py      # 1655 lines
```

**Issues:**
- 20+ files in a single directory (hard to navigate)
- Some very large files (1435-1655 lines)
- Unclear organization

**Proposed Reorganization:**
```
core/modes/agent/
├── base/
│   ├── __init__.py
│   ├── base_agent.py           # Core base agent
│   └── agent_factory.py        # Factory pattern
├── types/                       # ← NEW: Types & signatures
│   ├── execution_types.py      # (from _execution_types.py)
│   ├── dag_types.py
│   └── planner_signatures.py
├── mixins/                      # ← NEW: Mixin classes
│   ├── skill_selection.py      # (from _skill_selection_mixin.py)
│   └── plan_utils.py           # (from _plan_utils_mixin.py)
├── implementations/             # ← NEW: Concrete agents
│   ├── auto_agent.py
│   ├── autonomous_agent.py
│   ├── chat_assistant.py
│   ├── chat_assistant_v2.py
│   ├── dspy_mcp_agent.py
│   ├── model_chat_agent.py
│   ├── skill_based_agent.py
│   └── task_breakdown_agent.py
├── executors/                   # ← NEW: Execution engines
│   ├── skill_plan_executor.py
│   └── step_processors.py
├── planning/                    # ← NEW: Planning logic
│   ├── agentic_planner.py
│   └── dag_agents.py
└── tools/                       # ← NEW: Agent tools
    ├── section_tools.py
    ├── inspector.py
    ├── feedback_channel.py
    └── axon.py
```

**Benefits:**
- Clear separation by responsibility
- Easier to navigate (6 subdirs vs 20 files)
- Better discoverability
- Follows Single Responsibility Principle

**Risk:** Medium (many imports to update)
**Effort:** 2-3 hours
**Impact:** High (better developer experience)

---

### 3. **core/interface/** - Can Be Slimmed Further

**Current Structure (17 files):**
```
core/interface/
├── api/              # 152K - JottyAPI, ChatAPI, WorkflowAPI
│   ├── unified.py           # 248 lines
│   ├── chat_api.py          # 154 lines
│   ├── workflow_api.py      # 105 lines
│   ├── mode_router.py       # 555 lines ← Large!
│   ├── openapi.py           # 430 lines
│   └── openapi_generator.py # 326 lines
├── interfaces/       # 68K - Base interfaces
│   ├── message.py           # 424 lines
│   └── host_provider.py     # 176 lines
├── ui/               # 120K - A2UI formatting
│   ├── a2ui.py              # 506 lines
│   ├── justjot_helper.py    # 382 lines
│   ├── schema_validator.py  # 256 lines
│   └── status_taxonomy.py   # 178 lines
└── use_cases/        # 12K - Shim
```

**Observations:**
1. **mode_router.py (555 lines)** - Could be split:
   - `ModeRouter` class
   - Route handlers
   - Context builders

2. **openapi.py + openapi_generator.py** - Related files, could merge or move to sdk/

3. **ui/** is large (120K) - Could be moved:
   - Option A: Move to `sdk/ui/` (if it's part of SDK response formatting)
   - Option B: Keep in `core/interface/ui/` (if it's internal)

**Questions:**
- Is `ui/` part of the public SDK or internal?
- Should OpenAPI generation be in `sdk/` instead?

---

### 4. **sdk/** - Well Organized (9 files)

**Current Structure:**
```
sdk/
├── client.py                    # 40K - Main SDK client
├── __init__.py                  # Public exports
├── openapi_generator.py         # OpenAPI generation
├── generate_sdks.py             # Multi-language SDK generation
├── openapi.json                 # OpenAPI spec
├── test_*.py                    # Tests
└── generated/                   # Auto-generated SDKs
    ├── python/
    ├── typescript/
    └── ...
```

**Issues Found:**
SDK imports directly from `core.agents`, `core.api`, `core.registry`:
```python
from ..core.agents.chat_assistant import ChatAssistant  # ❌ Bypasses interface!
from ..core.agents import AutoAgent                      # ❌ Bypasses interface!
from ..core.api.mode_router import get_mode_router       # ✅ OK (interface layer)
from ..core.registry import get_unified_registry         # ❌ Should use interface!
```

**Problem:** SDK should ONLY import from `core/interface/api/`, NOT directly from core internals.

**Proposed Fix:**
1. **Create facade in `core/interface/api/registry.py`:**
   ```python
   # core/interface/api/registry.py
   from Jotty.core.registry import get_unified_registry as _get_registry

   def get_registry():
       """Get skill registry via interface layer."""
       return _get_registry()
   ```

2. **Create facade for agents in `core/interface/api/agents.py`:**
   ```python
   # core/interface/api/agents.py
   from Jotty.core.modes.agent.base import ChatAssistant, AutoAgent

   __all__ = ['ChatAssistant', 'AutoAgent']
   ```

3. **Update SDK imports:**
   ```python
   # sdk/client.py
   from ..core.interface.api.mode_router import get_mode_router  # ✅
   from ..core.interface.api.registry import get_registry         # ✅
   from ..core.interface.api.agents import ChatAssistant, AutoAgent  # ✅
   ```

**Benefits:**
- ✅ SDK respects layer boundaries
- ✅ Core can change without breaking SDK
- ✅ Proper separation of concerns

**Risk:** Low
**Effort:** 1 hour
**Impact:** High (architectural correctness)

---

### 5. **core/modes/use_cases/** vs **core/modes/execution/**

**Potential Overlap:**
```
core/modes/use_cases/
└── chat/chat_executor.py        # ChatExecutor (356 lines)

core/modes/execution/
└── executor.py                  # Executor (1836 lines)
```

**Questions:**
- Are these different executors or overlapping?
- Should they be unified?

**Need to investigate:**
```bash
# Check if they're related
grep -n "class.*Executor" core/modes/use_cases/chat/chat_executor.py
grep -n "class.*Executor" core/modes/execution/executor.py
```

---

## 📊 Summary

| Opportunity | Type | Files | Lines | Risk | Effort | Impact |
|-------------|------|-------|-------|------|--------|--------|
| 1. Delete empty chat/ | Quick Win | 0 | 0 | Zero | 1 min | Low |
| 2. Reorganize agent/base/ | Refactor | 20+ | 15K+ | Medium | 2-3 hrs | High |
| 3. Slim interface/ | Analysis | 17 | 4K | Low | 2 hrs | Medium |
| 4. Fix SDK imports | Architecture | 1 | 100 | Low | 1 hr | High |
| 5. Unify executors | Investigation | 2 | 2K | Medium | TBD | TBD |

---

## 🎯 Recommended Priority

### Phase 1: Quick Wins (10 minutes)
1. ✅ Delete `core/modes/chat/` (empty)
2. ✅ Fix SDK import violations (add facades)

### Phase 2: Architecture Improvements (3-4 hours)
3. Reorganize `core/modes/agent/base/` into subdirectories
4. Split `mode_router.py` if too large

### Phase 3: Deep Analysis (TBD)
5. Investigate executor overlap
6. Decide on UI location (interface vs sdk)
7. Decide on OpenAPI location (interface vs sdk)

---

## 🚀 Next Steps

**Ask User:**
1. Should we proceed with Phase 1 quick wins?
2. Should we reorganize `core/modes/agent/base/`?
3. Where should `ui/` live - interface or sdk?
4. Where should OpenAPI files live - interface or sdk?

**Then:**
- Execute approved refactors
- Update documentation
- Run tests
- Commit changes
