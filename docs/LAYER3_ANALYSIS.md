# Layer 3→2 Analysis: use_cases vs modes

## Current State

### use_cases/ (196K, 12 files)
```
core/interface/use_cases/
├── base.py                      # BaseUseCase, UseCaseConfig, UseCaseResult
├── chat/
│   ├── chat_context.py          # ChatContext, ChatMessage
│   ├── chat_executor.py         # ChatExecutor (356 lines)
│   ├── chat_orchestrator.py     # ChatOrchestrator
│   └── chat_use_case.py         # ChatUseCase (wraps executor)
└── workflow/
    ├── workflow_context.py      # WorkflowContext
    ├── workflow_executor.py     # WorkflowExecutor
    ├── workflow_orchestrator.py # WorkflowOrchestrator
    └── workflow_use_case.py     # WorkflowUseCase (wraps executor)
```

### orchestration/ (overlapping implementations)
```
core/intelligence/orchestration/
├── unified_executor.py          # ChatExecutor (1043 lines) ← NEWER, MORE FEATURED
└── ... (other files)
```

### modes/ (51 files)
```
core/modes/
├── agent/
│   └── base/
│       ├── chat_assistant.py    # ChatAssistant
│       └── ...
└── workflow/
    ├── auto_workflow.py         # AutoWorkflow
    ├── research_workflow.py     # ResearchWorkflow
    └── ...
```

---

## Usage Analysis

### Who imports use_cases?

1. **core/interface/api/unified.py** - JottyAPI
   ```python
   from Jotty.core.interface.use_cases import ChatUseCase, WorkflowUseCase
   ```

2. **core/interface/api/chat_api.py** - ChatAPI
   ```python
   from Jotty.core.interface.use_cases.chat import ChatUseCase, ChatMessage
   ```

3. **core/interface/api/workflow_api.py** - WorkflowAPI
   ```python
   from Jotty.core.interface.use_cases.workflow import WorkflowUseCase
   ```

4. **sdk/client.py** - Jotty SDK
   - Uses JottyAPI, ChatAPI, WorkflowAPI (indirect usage)

### Who imports from modes?

- **core/interface/api/chat_api.py**
  ```python
  from Jotty.core.modes.agent.base.chat_assistant import create_chat_assistant
  ```

- Various internal core modules

---

## Key Findings

### 1. TWO ChatExecutor Implementations

| File | Lines | Purpose |
|------|-------|---------|
| `use_cases/chat/chat_executor.py` | 356 | Simple wrapper around conductor |
| `orchestration/unified_executor.py` | 1043 | Full-featured native LLM tool-calling |

**Question:** Which one is actually used?
- **ChatUseCase** uses use_cases/ChatExecutor
- **ModeRouter** uses orchestration/ChatExecutor
- Both are active!

### 2. use_cases are WRAPPERS

Looking at the code:
- **ChatUseCase** wraps **ChatExecutor** + **ChatOrchestrator**
- **WorkflowUseCase** wraps **WorkflowExecutor** + **WorkflowOrchestrator**
- They provide:
  - Standardized error handling (BaseUseCase)
  - Common interface (execute, stream, validate)
  - Metadata extraction
  - Logging

### 3. modes/ are IMPLEMENTATIONS

- **ChatAssistant** - Actual chat agent implementation
- **AutoWorkflow** - Actual workflow implementation
- **ResearchWorkflow** - Research-specific workflow

---

## Feature Preservation Strategy

### Option A: Keep Both (Safe but messy)
- ✅ Zero risk of feature loss
- ❌ Maintains duplication
- ❌ Confusing architecture

### Option B: Consolidate (Clean but risky)
- ✅ Clean architecture
- ✅ Single source of truth
- ❌ Risk of breaking SDK/API

### Option C: Move use_cases to modes/use_cases/ (Recommended)
```
core/modes/
├── use_cases/           # ← MOVE from core/interface/
│   ├── base.py
│   ├── chat/
│   └── workflow/
├── agent/
├── workflow/
└── execution/
```

**Benefits:**
- ✅ Keeps all execution logic in one place (modes/)
- ✅ Preserves all features
- ✅ Simple import updates (from core.interface.use_cases → core.modes.use_cases)
- ✅ Backward compatibility via shim in core/interface/

---

## Import Impact

### Files to update (22 total):

**SDK Layer:**
- sdk/client.py (indirect via API)

**API Layer:**
- core/interface/api/unified.py
- core/interface/api/chat_api.py
- core/interface/api/workflow_api.py

**Core Layer:**
- core/__init__.py (re-exports)
- ... (others)

---

## Recommendation

**MOVE use_cases/ to modes/use_cases/** with backward compat shim:

1. Move `core/interface/use_cases/` → `core/modes/use_cases/`
2. Create shim in `core/interface/use_cases/__init__.py`:
   ```python
   # Deprecated: Moved to core/modes/use_cases/
   from Jotty.core.modes.use_cases import *
   ```
3. Update imports in api/ layer to use new path
4. Test thoroughly

This approach:
- ✅ Preserves ALL features
- ✅ Clean architecture (all execution in modes/)
- ✅ Backward compatible (shim prevents breakage)
- ✅ Can be done incrementally

---

## Next Steps

1. ✅ Complete this analysis
2. Get user approval
3. Execute move with shim
4. Update imports
5. Run tests
6. Remove shim after confirming no breakage
