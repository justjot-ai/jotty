# Auditor Task Success Validation Fix

## Problem
Auditor was marking failed tasks as VALID because it was validating "extraction quality" (whether the failure was well-documented) rather than "task success" (whether the task actually completed).

**Example**:
- Task failed: `ExecutionResult(success=False, steps_executed=0, errors=["No valid tools found"])`
- Auditor marked as VALID with reasoning: "accurately documents a critical system limitation"
- **Issue**: Task failed, so should be INVALID

## Root Cause
Auditor prompt said:
- `is_valid: "True if extraction is valid, False if issues found"`

This was ambiguous - it could mean:
1. ✅ "The extraction accurately represents what happened" (what LLM interpreted)
2. ❌ "The task succeeded" (what we actually want)

## Solution

### 1. Enhanced Prompt ✅
Updated `ReviewerSignature` to emphasize task success:

**Before**:
```
is_valid: "True if extraction is valid, False if issues found"
```

**After**:
```
is_valid: "True if task SUCCEEDED and extraction is valid, False if task failed or issues found. 
CRITICAL: Check if task actually completed successfully (success=True, outputs produced). 
A failed task (success=False, no outputs) should be marked INVALID even if the failure is well-documented."
```

**Location**: `Jotty/core/agents/inspector.py` (line ~92)

### 2. Failure Detection Override ✅
Added automatic detection of ExecutionResult failures:

**Checks for**:
- `success=False`
- `steps_executed=0`
- `No valid steps`
- `cannot proceed`
- `errors=`

**Action**: If failure detected but LLM marked as VALID, override to INVALID

**Location**: `Jotty/core/agents/inspector.py` (lines ~983-1005)

## Expected Behavior

**Before**:
```
Task: Generate code...
Result: ExecutionResult(success=False, steps_executed=0)
Auditor: ✅ VALID (confidence: 0.90)
Reasoning: "accurately documents system limitation"
```

**After**:
```
Task: Generate code...
Result: ExecutionResult(success=False, steps_executed=0)
Auditor: ⚠️  [AUDITOR OVERRIDE] Output indicates task failure but was marked VALID. Overriding to INVALID.
Auditor: ❌ INVALID (confidence: 0.90)
Tag: fail
```

## Benefits

1. ✅ **Correct validation** - Failed tasks marked as INVALID
2. ✅ **Better learning** - System learns from failures correctly
3. ✅ **Accurate metrics** - Success rates reflect actual task completion
4. ✅ **Override safety** - Catches LLM mistakes automatically

## Status
✅ Enhanced prompt with task success emphasis
✅ Added failure detection override
✅ Syntax check passed
✅ No linter errors

The auditor will now correctly mark failed tasks as INVALID, even if the failure is well-documented.
