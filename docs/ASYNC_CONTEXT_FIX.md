# Async Context Fix

## Issue
When `infer_task_type()` is called from async code (like `SwarmManager.run()`), DSPy throws:
```
dspy.configure(...) can only be called from the same async task that called it first. 
Please use `dspy.context(...)` in other async tasks instead.
```

## Root Cause
DSPy was configured in one async task context, but accessed from another async task context. DSPy requires using `dspy.context()` when accessing settings from different async tasks.

## Fix
Updated `infer_task_type()` to:
1. Detect if running in async context (`asyncio.get_running_loop()`)
2. Use `dspy.context(lm=lm)` for async contexts
3. Use `dspy.settings` directly for sync contexts

## Code Changes
- `core/agents/agentic_planner.py`: Added async context detection and `dspy.context()` usage

## Status
✅ Fixed - Async context issue resolved
