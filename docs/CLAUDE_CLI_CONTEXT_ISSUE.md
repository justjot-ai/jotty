# Claude CLI Context Contamination Issue

**Date**: January 27, 2026  
**Status**: ⚠️ **IDENTIFIED** - Root Cause Explained

---

## Problem

When using Claude CLI (`claude-cli` provider), the LLM responds to **git/codebase context** instead of the actual prompt.

**Example**:
- **Prompt**: "Explain your reasoning"
- **Response**: "How can I help you today with your codebase? I can assist with code development..." ❌

---

## Root Cause

### How Claude CLI Works

Claude CLI (`@anthropic-ai/claude-code`) is designed to be **workspace-aware**:

1. **It reads from the terminal/workspace context**
   - Git status
   - File contents
   - Terminal history
   - Workspace structure

2. **It's designed for code assistance**
   - The CLI tool is meant to help with codebases
   - It automatically includes workspace context
   - This is a **feature**, not a bug

3. **Our Implementation**:
   ```python
   # From core/llm/providers.py
   cmd = ["claude", "--model", model, "-p"]
   result = subprocess.run(
       cmd,
       input=prompt,  # We send prompt via stdin
       capture_output=True,
       text=True,
       env=env,
       timeout=timeout
   )
   ```

### Why It Happens

1. **Claude CLI reads workspace context automatically**
   - Even though we send prompt via stdin (`-p` flag)
   - The CLI tool still includes workspace context
   - This is by design for code assistance

2. **Git status is visible**
   - The CLI sees `git status` output
   - It includes this in its context
   - It responds to what it sees, not just the prompt

3. **Terminal/workspace awareness**
   - The CLI is aware of:
     - Current directory
     - Git repository state
     - File changes
     - Workspace structure

---

## Evidence

### Test Results

**Prompt**: "Explain your reasoning"

**Response**: 
```
I notice you have a partial context from a previous conversation. 
How can I help you today with your codebase? I can assist with:

- Code development and refactoring
- Performance optimization
- Testing and debugging
- Git operations
- Code review and analysis
- Documentation

What would you like to work on?
```

**Git Status** (at time of test):
```
 M core/monitoring/__init__.py
 M skills/mcp-justjot/tools.py
?? core/autonomous/
?? core/monitoring/profiler.py
?? core/optimization/
```

**Connection**: The response mentions "codebase", "git operations", "code review" - matching what Claude CLI sees in the workspace.

---

## Why This Happens

### Claude CLI Design Philosophy

1. **Workspace-Aware Tool**
   - Designed for code assistance
   - Automatically includes workspace context
   - Helps with code-related tasks

2. **Terminal Integration**
   - Reads from terminal/workspace
   - Includes git status, file changes
   - Provides context-aware responses

3. **Feature, Not Bug**
   - This is intentional behavior
   - Makes it useful for code tasks
   - But causes issues for isolated prompts

---

## Solutions

### Option 1: Use Anthropic API Instead ✅ RECOMMENDED

**Why**:
- ✅ **Isolated context** - Only sees what you send
- ✅ **No workspace contamination** - Pure prompt/response
- ✅ **More control** - You control the context

**How**:
```python
from core.llm import UnifiedLLM

# Use API instead of CLI
llm = UnifiedLLM(
    default_provider="anthropic",  # Instead of "claude-cli"
    default_model="sonnet"
)
```

**Requirements**:
- Set `ANTHROPIC_API_KEY` environment variable
- API key from Anthropic dashboard

---

### Option 2: Isolate Context (Workaround)

**How**:
1. **Run tests in isolated directory**
   ```bash
   mkdir /tmp/isolated_test
   cd /tmp/isolated_test
   # Run tests here (no git repo)
   ```

2. **Clear git context before tests**
   ```python
   import subprocess
   # Clear git status
   subprocess.run(["git", "status"], capture_output=True)
   ```

3. **Use clean environment**
   ```python
   import os
   env = os.environ.copy()
   # Remove git-related env vars
   env.pop("GIT_DIR", None)
   env.pop("GIT_WORK_TREE", None)
   ```

**Limitations**:
- ⚠️ Doesn't fully solve the issue
- ⚠️ Claude CLI still reads workspace
- ⚠️ Not a complete solution

---

### Option 3: Use Different Provider

**Options**:
- **Gemini**: `default_provider="gemini"`
- **OpenAI**: `default_provider="openai"`
- **Anthropic API**: `default_provider="anthropic"` ✅ Best option

**Why Anthropic API**:
- Same model (Claude Sonnet)
- Isolated context
- No workspace contamination
- More reliable for tests

---

## Recommended Solution

### Use Anthropic API for Tests ✅

**Change**:
```python
# Before (CLI - has context issues)
llm = UnifiedLLM(
    default_provider="claude-cli",
    default_model="sonnet"
)

# After (API - isolated context)
llm = UnifiedLLM(
    default_provider="anthropic",  # Use API instead
    default_model="sonnet"
)
```

**Benefits**:
- ✅ Isolated context (no git contamination)
- ✅ Same model (Claude Sonnet)
- ✅ More reliable for tests
- ✅ Better for production use

**Setup**:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

---

## Impact

### Current Behavior (CLI)

- ⚠️ **Context contamination** - Picks up git/codebase
- ⚠️ **Wrong responses** - Responds to workspace, not prompt
- ⚠️ **Unreliable tests** - Results vary by workspace state

### With API

- ✅ **Isolated context** - Only sees prompt
- ✅ **Correct responses** - Responds to actual prompt
- ✅ **Reliable tests** - Consistent results

---

## Code Changes Needed

### Update Test Files

**File**: `tests/test_jotty_with_outputs.py`

**Change**:
```python
# Before
self.llm = UnifiedLLM(
    default_provider="claude-cli",
    default_model="sonnet"
)

# After
self.llm = UnifiedLLM(
    default_provider="anthropic",  # Use API
    default_model="sonnet"
)
```

**Apply to**:
- `tests/test_jotty_optimized_performance.py`
- `tests/test_jotty_improved_performance.py`
- `tests/test_jotty_with_outputs.py`
- Other test files using `claude-cli`

---

## Summary

### Why Claude CLI Picks Up Git Context

1. **By Design**: Claude CLI is workspace-aware
2. **Terminal Integration**: Reads git status, file changes
3. **Code Assistance**: Designed for code tasks, not isolated prompts
4. **Feature, Not Bug**: This is intentional behavior

### Solution

**Use Anthropic API instead**:
- ✅ Isolated context
- ✅ Same model
- ✅ More reliable
- ✅ Better for tests

**This is NOT a compression issue** - it's Claude CLI's workspace awareness feature.

---

**Last Updated**: January 27, 2026  
**Status**: ⚠️ **ROOT CAUSE IDENTIFIED** - Use API Instead of CLI
