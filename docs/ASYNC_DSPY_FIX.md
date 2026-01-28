# DSPy Async Context Fix

## Issue
```
RuntimeError: dspy.configure(...) can only be called from the same async task that called it first. 
Please use `dspy.context(...)` in other async tasks instead.
```

## Root Cause
The code was trying to set `dspy.settings._current_signature` using `setattr()`, which internally calls `dspy.configure()`. This fails in async contexts because DSPy tracks which async task first called `configure()` and doesn't allow configuration changes from other async tasks.

## Solution
**Removed all `setattr(dspy.settings, '_current_signature', ...)` calls** because:
1. The signature is already baked into the DSPy modules when they're created:
   - `self.execution_planner = dspy.ChainOfThought(ExecutionPlanningSignature)`
   - `self.task_type_inferrer = dspy.ChainOfThought(TaskTypeInferenceSignature)`
   - `self.skill_selector = dspy.ChainOfThought(SkillSelectionSignature)`

2. DSPy modules already have access to their signatures - no need to set them globally.

## Changes Made

### 1. `plan_execution()` method (line ~555)
- **Before**: Set `_current_signature` before calling `execution_planner`
- **After**: Call `execution_planner` directly (signature already in module)

### 2. `infer_task_type()` method (line ~235)
- **Before**: Set `_current_signature` before calling `task_type_inferrer`
- **After**: Use `dspy.context(lm=lm)` for async contexts, call module directly

### 3. `select_skills()` method (line ~440)
- **Before**: Set `_current_signature` before calling `skill_selector`
- **After**: Call `skill_selector` directly (signature already in module)

## Testing
- ✓ Syntax check passed
- ✓ All `setattr(dspy.settings, '_current_signature', ...)` calls removed
- ✓ Async context handling preserved with `dspy.context(lm=lm)` where needed

## Notes
- The timeout restoration code was kept (it doesn't use `configure()`)
- `dspy.context()` is still used for async contexts to properly set the LM
- No functionality lost - signatures are still used, just accessed through the modules themselves
