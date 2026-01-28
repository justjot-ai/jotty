# Claude CLI Context Isolation Fix

**Date**: January 27, 2026  
**Status**: ✅ **FIXED**

---

## Problem

Claude CLI was picking up git/codebase context instead of responding to the actual prompt.

**Example**:
- **Prompt**: "Explain your reasoning"
- **Response**: "How can I help you today with your codebase?" ❌

---

## Root Cause

Claude CLI (`@anthropic-ai/claude-code`) is **workspace-aware by design**:
- Reads git status, file changes, workspace structure
- Includes this context automatically
- Designed for code assistance

Even with `-p` (print mode), it still includes workspace context.

---

## Solution Found ✅

### Option 1: Use `--system-prompt` ✅ **IMPLEMENTED**

**How it works**:
- `--system-prompt` tells Claude to ignore workspace context
- Claude follows the system prompt instruction
- Isolates from git/codebase

**Test Results**:
```
✅ TEST 4: Claude CLI with System Prompt
   Mentions codebase/git: False  ← WORKS!
```

**Implementation**:
```python
system_prompt = (
    "You are a helpful AI assistant. Only respond to the user's prompt. "
    "Ignore any workspace, git, codebase, or file system context. "
    "Focus solely on answering the user's question directly."
)
cmd = ["claude", "--model", model, "-p", "--system-prompt", system_prompt]
```

---

### Option 2: Run from Temp Directory ✅ **WORKS**

**How it works**:
- Run Claude CLI from a temp directory (no git repo)
- No workspace context to pick up
- Isolated environment

**Test Results**:
```
✅ TEST 3: Claude CLI from Temp Directory
   Mentions codebase/git: False  ← WORKS!
```

**Implementation**:
```python
import tempfile

with tempfile.TemporaryDirectory() as tmpdir:
    result = subprocess.run(
        cmd,
        input=prompt,
        cwd=tmpdir,  # Run from temp dir
        ...
    )
```

---

### Option 3: Both Combined ✅ **BEST**

**How it works**:
- Use `--system-prompt` + temp directory
- Double isolation
- Most reliable

---

## What Didn't Work ❌

### `--tools ""` Alone

**Test Results**:
```
❌ TEST 2: Claude CLI with --tools ""
   Mentions codebase/git: True  ← STILL PICKS UP CONTEXT
```

**Why**: Disabling tools doesn't disable workspace awareness.

---

## Implementation

### File: `core/llm/providers.py`

**Change**:
```python
# Before
cmd = ["claude", "--model", model, "-p"]

# After
system_prompt = (
    "You are a helpful AI assistant. Only respond to the user's prompt. "
    "Ignore any workspace, git, codebase, or file system context. "
    "Focus solely on answering the user's question directly."
)
cmd = ["claude", "--model", model, "-p", "--system-prompt", system_prompt]
```

---

## Test Results

### Before Fix

```
Prompt: "Explain your reasoning about roses and flowers"
Response: "I notice you've asked me to explain your reasoning about roses and flowers, 
          but this appears to be a test or example question that doesn't relate to 
          any actual software engineering task in your Jotty codebase..."
❌ Mentions codebase/git: True
```

### After Fix

```
Prompt: "Explain your reasoning about roses and flowers"
Response: "I'd be happy to explain reasoning about roses and flowers!

## Logical Relationship

**Roses are a subset of flowers.** This means:
- All roses are flowers (true)
- Not all flowers are roses (true)..."
✅ Mentions codebase/git: False
```

---

## Impact

### Before Fix

- ❌ Wrong responses (responds to workspace, not prompt)
- ❌ Unreliable tests (results vary by workspace state)
- ❌ Context contamination

### After Fix

- ✅ Correct responses (responds to actual prompt)
- ✅ Reliable tests (consistent results)
- ✅ Isolated context

---

## Summary

### Solution ✅

**Use `--system-prompt`** to tell Claude CLI to ignore workspace context.

**Why it works**:
- Claude follows system prompt instructions
- System prompt explicitly tells it to ignore workspace
- Isolates from git/codebase

### Alternative ✅

**Run from temp directory** (no git repo):
- No workspace context to pick up
- Isolated environment
- Also works

### Best Practice ✅

**Use both**:
- `--system-prompt` + temp directory
- Double isolation
- Most reliable

---

**Last Updated**: January 27, 2026  
**Status**: ✅ **FIXED** - Using `--system-prompt` to Isolate Context
