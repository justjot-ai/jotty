# Jotty Refactoring Migration Guide

**Version:** 6.0  
**Date:** January 2026  
**Status:** Phase 1-5 Complete

## Overview

This guide helps developers migrate code after the Jotty v6.0 refactoring (Phases 1-5). All changes maintain **100% backward compatibility** through deprecation aliases, so existing code continues to work.

---

## Quick Reference: Name Changes

### Phase 1.3: Naming Convention Standardization

| Old Name | New Name | Type | Module |
|----------|----------|------|--------|
| `Conductor` | `MultiAgentsOrchestrator` | Class | `orchestration.conductor` |
| `ArchitectSignature` | `PlannerSignature` | DSPy Signature | `agents.inspector` |
| `AuditorSignature` | `ReviewerSignature` | DSPy Signature | `agents.inspector` |
| `SmartContextGuard` | `LLMContextManager` | Class | `context.context_guard` |
| `AgenticChunker` | `LLMChunkManager` | Class | `context.chunker` |

---

## Detailed Migration Instructions

### 1. Orchestrator Renaming

**Old Code:**
```python
from Jotty.core.orchestration.conductor import Conductor

conductor = Conductor(
    actors=actors,
    metadata_provider=provider,
    config=config
)
result = await conductor.run(goal="Extract data")
```

**New Code (Recommended):**
```python
from Jotty.core.orchestration.conductor import MultiAgentsOrchestrator

orchestrator = MultiAgentsOrchestrator(
    actors=actors,
    metadata_provider=provider,
    config=config
)
result = await orchestrator.run(goal="Extract data")
```

**Backward Compatible (Still Works):**
```python
# Old import still works with deprecation warning
from Jotty.core.orchestration.conductor import Conductor
conductor = Conductor(...)  # Shows DeprecationWarning
```

---

### 2. Validation Signatures

**Old Code:**
```python
from Jotty.core.agents.inspector import ArchitectSignature, AuditorSignature

architect = dspy.ChainOfThought(ArchitectSignature)
auditor = dspy.ChainOfThought(AuditorSignature)
```

**New Code (Recommended):**
```python
from Jotty.core.agents.inspector import PlannerSignature, ReviewerSignature

planner = dspy.ChainOfThought(PlannerSignature)
reviewer = dspy.ChainOfThought(ReviewerSignature)
```

**Backward Compatible (Still Works):**
```python
# Old names are aliases - no warning (simple assignment)
from Jotty.core.agents.inspector import ArchitectSignature, AuditorSignature
architect = dspy.ChainOfThought(ArchitectSignature)  # Works fine
```

---

### 3. Context Management

**Old Code:**
```python
from Jotty.core.context.context_guard import SmartContextGuard

guard = SmartContextGuard(max_tokens=28000)
context, metadata = await guard.build_context()
```

**New Code (Recommended):**
```python
from Jotty.core.context.context_guard import LLMContextManager

context_mgr = LLMContextManager(max_tokens=28000)
context, metadata = await context_mgr.build_context()
```

**Backward Compatible (Still Works):**
```python
# Old name is alias
from Jotty.core.context.context_guard import SmartContextGuard
guard = SmartContextGuard(max_tokens=28000)  # Works fine
```

---

### 4. Semantic Chunking

**Old Code:**
```python
from Jotty.core.context.chunker import AgenticChunker

chunker = AgenticChunker()
chunks = await chunker.chunk_and_process(content, task_context)
```

**New Code (Recommended):**
```python
from Jotty.core.context.chunker import LLMChunkManager

chunk_mgr = LLMChunkManager()
chunks = await chunk_mgr.chunk_and_process(content, task_context)
```

**Backward Compatible (Still Works):**
```python
# Old name is alias
from Jotty.core.context.chunker import AgenticChunker
chunker = AgenticChunker()  # Works fine
```

---

### 5. Data Structures (Phase 1.1)

**Old Code:**
```python
from Jotty.core.foundation.data_structures import (
    MemoryLevel, OutputTag, TaskStatus, MemoryEntry
)
```

**New Code (Recommended):**
```python
# Import from organized types package
from Jotty.core.foundation.types import (
    MemoryLevel, OutputTag, TaskStatus, MemoryEntry
)
```

**Backward Compatible (Still Works):**
```python
# Old import still works (re-exported from data_structures.py)
from Jotty.core.foundation.data_structures import (
    MemoryLevel, OutputTag, TaskStatus, MemoryEntry
)
```

---

### 6. TaskStatus Consolidation (Phase 1.2)

**Old Code:**
```python
# Multiple definitions in different modules
from Jotty.core.orchestration.roadmap import TaskStatus
from Jotty.core.queue.task import TaskStatus
from Jotty.core.use_cases.workflow.workflow_context import TaskStatus
```

**New Code (Recommended):**
```python
# Single canonical import
from Jotty.core.foundation.types import TaskStatus

# Or from any module (all import from canonical)
from Jotty.core.orchestration.roadmap import TaskStatus  # Same enum
```

**All TaskStatus imports now reference the same enum (10 statuses):**
- SUGGESTED, BACKLOG, PENDING, IN_PROGRESS
- COMPLETED, FAILED, BLOCKED, CANCELLED
- RETRYING, SKIPPED

---

### 7. Learning System Interface (Phase 5)

**New Feature (Optional):**

Phase 5 introduced abstract base classes for learning systems. These are **optional** and don't require migration of existing code.

```python
# New base classes available for future learners
from Jotty.core.learning import (
    BaseLearningManager,           # Base for all learners
    ValueBasedLearningManager,     # For TD(λ), Q-learning
    RewardShapingManager,          # For shaped rewards
    MultiAgentLearningManager,     # For MARL
)

# Existing learners still work without inheriting from base
from Jotty.core.learning import (
    TDLambdaLearner,      # Works as before
    LLMQPredictor,        # Works as before
    ShapedRewardManager,  # Works as before
)
```

**Use Cases:**
- Creating new custom learners (inherit from base classes)
- Dependency injection and testing (use base class types)
- Polymorphic learner swapping

---

## Naming Convention Rationale

### The *Manager Pattern

All subsystem components now use the `*Manager` suffix for consistency:

- `MemoryManager` - Memory management
- `StateManager` - State tracking
- `ToolManager` - Tool lifecycle
- `LLMContextManager` - Context budgeting
- `LLMChunkManager` - Semantic chunking
- `ParameterResolutionManager` - Parameter binding

**Exception:** `MultiAgentsOrchestrator` - Top-level orchestrator (not a manager)

### The LLM* Prefix

Components powered by LLMs use the `LLM*` prefix:

- `LLMContextManager` - LLM-powered context budgeting
- `LLMChunkManager` - LLM-powered chunking
- `LLMQPredictor` - LLM-based Q-value prediction
- `LLMRAGRetriever` - LLM-powered RAG retrieval

### Clear Role Names

Validation components use clear role names:

- `Planner` (was `Architect`) - Plans execution
- `Reviewer` (was `Auditor`) - Reviews outputs

**Why?** "Architect" and "Auditor" imply rigidity. "Planner" and "Reviewer" better reflect their advisory roles.

---

## Testing Your Migration

### 1. Import Tests

```python
# Test old imports still work
from Jotty.core.orchestration.conductor import Conductor
from Jotty.core.agents.inspector import ArchitectSignature
from Jotty.core.context.context_guard import SmartContextGuard
print("✓ Old imports work")

# Test new imports
from Jotty.core.orchestration.conductor import MultiAgentsOrchestrator
from Jotty.core.agents.inspector import PlannerSignature
from Jotty.core.context.context_guard import LLMContextManager
print("✓ New imports work")
```

### 2. Backward Compatibility Test

```python
# Verify aliases point to new classes
from Jotty.core.orchestration.conductor import Conductor, MultiAgentsOrchestrator
assert issubclass(Conductor, MultiAgentsOrchestrator)
print("✓ Conductor is MultiAgentsOrchestrator")

from Jotty.core.agents.inspector import ArchitectSignature, PlannerSignature
assert ArchitectSignature is PlannerSignature
print("✓ ArchitectSignature is PlannerSignature")
```

### 3. Run Existing Tests

```bash
# Your existing tests should pass without changes
pytest tests/
```

---

## Migration Timeline

### Immediate (No Action Required)
- All code continues to work with old names
- Deprecation warnings shown (for class-based aliases only)

### Recommended (Next Sprint)
- Update imports to use new names in new code
- Suppress deprecation warnings if needed

### Optional (Future)
- Gradually migrate existing code to new names
- Leverage learning system interfaces for new learners

---

## Breaking Changes

**None!** All changes maintain 100% backward compatibility.

Deprecation aliases will be removed in a **future major version** (v7.0+), with advance notice.

---

## Support

### Report Issues
- GitHub: [anthropics/jotty](https://github.com/anthropics/jotty)  
- Issues: Tag with `refactoring` label

### Questions
- Check: `docs/ARCHITECTURE.md` for system design
- Check: `docs/REFACTORING_SUMMARY.md` for changes overview

---

## Checklist for New Code

When writing new Jotty code:

- [ ] Use `MultiAgentsOrchestrator` instead of `Conductor`
- [ ] Use `PlannerSignature`/`ReviewerSignature` instead of `Architect`/`Auditor`
- [ ] Use `LLMContextManager` instead of `SmartContextGuard`
- [ ] Use `LLMChunkManager` instead of `AgenticChunker`
- [ ] Import types from `core.foundation.types` instead of `data_structures`
- [ ] Import `TaskStatus` from canonical location (`types` package)
- [ ] Consider using learning base classes for new learners

---

## Summary

**Phase 1-5 Complete:**
- ✅ Data structures organized into types package
- ✅ Duplicate classes removed
- ✅ Unified naming conventions (*Manager pattern)
- ✅ TaskStatus consolidated (single source of truth)
- ✅ Learning system interface created
- ✅ 100% backward compatibility maintained

**Next:** Enjoy cleaner, more maintainable code! 🎉
