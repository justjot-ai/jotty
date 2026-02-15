# LLM Provider Consolidation Plan

**Date:** 2026-02-15
**Issue:** LLMs are being called in different ways across the codebase instead of using the unified provider.

---

## Problem

Different parts of the codebase are initializing and calling LLMs differently:

| Component | Current LLM Usage | Issue |
|-----------|------------------|-------|
| **ValidationGate** | `self._lm` (own instance) | Hitting rate limits, not using unified provider |
| **DirectChatExecutor** | `JottyClaudeProvider()` | Direct provider instance |
| **unified_executor** | DSPy with own config | Own configuration |
| **mode_router** | Calls executors | Inconsistent providers |

**Result:**
- ❌ Rate limit errors (each component hits limits separately)
- ❌ Inconsistent configuration (model, timeouts, etc.)
- ❌ No centralized caching/monitoring
- ❌ Duplicate API calls
- ❌ Higher costs

---

## Solution: Use UnifiedLMProvider Everywhere

**Single source of truth:**
```python
from Jotty.core.infrastructure.foundation.unified_lm_provider import configure_dspy_lm

# Configure once at app startup
lm = configure_dspy_lm()  # Auto-detects best provider

# All components use this same instance
```

**Benefits:**
- ✅ Centralized rate limiting
- ✅ Shared caching
- ✅ Consistent configuration
- ✅ Single point for monitoring/logging
- ✅ Lower costs (no duplicate calls)

---

## Current LLM Initialization Points

### 1. ValidationGate (`validation_gate.py`)

**Current:**
```python
def _init_lm(self) -> bool:
    try:
        import dspy
        self._lm = dspy.LM(model="claude-haiku-3-5-20241022", max_tokens=1)
        return True
    except Exception:
        return False
```

**Issues:**
- Creates own DSPy instance
- No connection to unified provider
- Hits rate limits independently
- Line 364: `response = self._lm(prompt=prompt)`

**Should be:**
```python
def _init_lm(self) -> bool:
    try:
        from Jotty.core.infrastructure.foundation.unified_lm_provider import configure_dspy_lm
        self._lm = configure_dspy_lm(provider="anthropic", model="claude-haiku-3-5-20241022")
        return True
    except Exception:
        return False
```

---

### 2. DirectChatExecutor (`direct_chat_executor.py`)

**Current:**
```python
def _get_provider(self) -> Any:
    if self._provider is None:
        from Jotty.core.infrastructure.foundation.jotty_claude_provider import JottyClaudeProvider
        self._provider = JottyClaudeProvider()
    return self._provider
```

**Issues:**
- Creates own provider instance
- Not using unified provider
- Different from what ValidationGate uses

**Should be:**
```python
def _get_provider(self) -> Any:
    if self._provider is None:
        from Jotty.core.infrastructure.foundation.unified_lm_provider import configure_dspy_lm
        self._provider = configure_dspy_lm(provider="anthropic", model=self.model)
    return self._provider
```

---

### 3. UnifiedExecutor (`unified_executor.py`)

**Check:** Does it use unified provider or own DSPy config?

**Should verify:**
```bash
grep -n "dspy\|LM\|provider" unified_executor.py
```

---

### 4. Mode Router (`mode_router.py`)

**Current:** Delegates to executors (indirect LLM usage)

**Action:** Ensure it passes unified provider to executors

---

## Consolidation Steps

### Step 1: Create Unified Provider Singleton

**File:** `core/infrastructure/foundation/llm_singleton.py` (NEW)

```python
"""
LLM Provider Singleton
======================

Single global LLM instance shared across entire application.
Prevents duplicate API calls and rate limit issues.
"""

from typing import Optional
from dspy.clients.base_lm import BaseLM

_global_lm: Optional[BaseLM] = None
_lm_lock = None


def get_global_lm(provider: Optional[str] = None, model: Optional[str] = None, **kwargs) -> BaseLM:
    """
    Get or create global LLM instance.

    All components should use this instead of creating their own.

    Args:
        provider: Provider name (anthropic, openai, etc.) - only used on first call
        model: Model name - only used on first call
        **kwargs: Additional config - only used on first call

    Returns:
        Shared LLM instance
    """
    global _global_lm, _lm_lock

    if _lm_lock is None:
        import threading
        _lm_lock = threading.Lock()

    with _lm_lock:
        if _global_lm is None:
            from .unified_lm_provider import configure_dspy_lm
            _global_lm = configure_dspy_lm(provider=provider, model=model, **kwargs)
        return _global_lm


def reset_global_lm():
    """Reset global LM (for testing)."""
    global _global_lm
    _global_lm = None
```

---

### Step 2: Update ValidationGate

**File:** `core/intelligence/orchestration/validation_gate.py`

**Change line ~348:**
```python
# OLD
def _init_lm(self) -> bool:
    try:
        import dspy
        self._lm = dspy.LM(model="claude-haiku-3-5-20241022", max_tokens=1)
        return True
    except Exception:
        return False

# NEW
def _init_lm(self) -> bool:
    try:
        from Jotty.core.infrastructure.foundation.llm_singleton import get_global_lm
        self._lm = get_global_lm(provider="anthropic", model="claude-haiku-3-5-20241022")
        return True
    except Exception as e:
        logger.warning(f"ValidationGate: LLM init failed: {e}")
        return False
```

---

### Step 3: Update DirectChatExecutor

**File:** `core/intelligence/orchestration/direct_chat_executor.py`

**Change _get_provider method:**
```python
# OLD
def _get_provider(self) -> Any:
    if self._provider is None:
        from Jotty.core.infrastructure.foundation.jotty_claude_provider import JottyClaudeProvider
        self._provider = JottyClaudeProvider()
    return self._provider

# NEW
def _get_provider(self) -> Any:
    if self._provider is None:
        from Jotty.core.infrastructure.foundation.llm_singleton import get_global_lm
        self._provider = get_global_lm(provider="anthropic", model=self.model)
    return self._provider
```

**Update execute method** to use DSPy call format:
```python
# OLD (if using direct provider)
response = await provider.chat_completion(...)

# NEW (using DSPy LM)
response = provider(prompt=f"System: {system_prompt}\n\nUser: {message}")
```

---

### Step 4: Verify UnifiedExecutor Uses Singleton

**File:** `core/intelligence/orchestration/unified_executor.py`

**Check:** Does it create its own DSPy instance?

**Should use:**
```python
from Jotty.core.infrastructure.foundation.llm_singleton import get_global_lm
lm = get_global_lm()
```

---

### Step 5: Initialize Global LM at App Startup

**File:** `apps/cli/app.py` (and others)

**Add in `__init__`:**
```python
def __init__(self, ...):
    # ... existing code ...

    # Initialize global LLM provider ONCE
    from Jotty.core.infrastructure.foundation.llm_singleton import get_global_lm
    self.lm = get_global_lm(provider="anthropic")
    logger.info(f"Global LLM provider initialized: {self.lm.model}")
```

---

## Rate Limit Handling

### Add Rate Limit Retry in Singleton

**Update `llm_singleton.py`:**
```python
class RateLimitedLM(BaseLM):
    """Wrapper that handles rate limits with exponential backoff."""

    def __init__(self, wrapped_lm: BaseLM):
        super().__init__(model=wrapped_lm.model)
        self._wrapped = wrapped_lm
        self._retry_after = 0

    def __call__(self, *args, **kwargs):
        import time
        from litellm import RateLimitError

        # Check if we're in cooldown
        if self._retry_after > 0:
            wait_time = self._retry_after - time.time()
            if wait_time > 0:
                logger.warning(f"Rate limit cooldown: waiting {wait_time:.1f}s")
                time.sleep(wait_time)
            self._retry_after = 0

        try:
            return self._wrapped(*args, **kwargs)
        except RateLimitError as e:
            # Parse retry-after from error message
            if "Try again in" in str(e):
                try:
                    seconds = int(str(e).split("Try again in ")[1].split(" ")[0])
                    self._retry_after = time.time() + seconds
                    logger.error(f"Rate limit hit, will retry after {seconds}s")
                except:
                    pass
            raise
```

---

## Expected Benefits

### Before Consolidation

**Scenario:** User types "hi"

1. ValidationGate creates DSPy instance → Makes Haiku call → **Rate limit**
2. Falls back to FULL mode
3. UnifiedExecutor creates own DSPy instance → Makes Sonnet call → **Rate limit**
4. Retries with exponential backoff
5. User waits 60+ seconds

**Cost:** 2+ rate-limited calls, 60s latency

---

### After Consolidation

**Scenario:** User types "hi"

1. ValidationGate uses global LM → Makes Haiku call (cached/shared rate limit)
2. Returns DIRECT mode
3. DirectChatExecutor uses same global LM → Makes Haiku call
4. Response in <1s

**Cost:** 1-2 calls (shared rate limit pool), <1s latency

---

## Migration Checklist

- [x] Create `llm_singleton.py` with global provider ✅
- [x] Update `ValidationGate._init_lm()` to use singleton ✅
- [x] Update `DirectChatExecutor._get_provider()` to use singleton ✅
- [x] Update `executor.py` auditor LM injection to use singleton ✅
- [x] Update `agentic_planner.py` fast LM to use singleton ✅
- [x] Update `skill_generator.py` to use singleton ✅
- [x] Update `gaia_adapter.py` to use singleton ✅
- [x] Update `gaia_signatures.py` to use singleton ✅
- [x] Check `unified_executor.py` LLM usage (no direct imports) ✅
- [x] Initialize global LM at app startup (app.py) ✅
- [ ] Test "hi" query (should be fast)
- [ ] Monitor rate limit errors (should decrease)
- [ ] Add rate limit handling wrapper (optional, future)
- [ ] Add caching layer (optional, future)

---

## Rollback Plan

If issues occur:

1. **Keep old code in git history:**
   ```bash
   git log validation_gate.py
   git checkout <commit> -- validation_gate.py
   ```

2. **Disable global LM:**
   ```python
   # In llm_singleton.py
   def get_global_lm(...):
       # return None  # Forces components to use own instances
   ```

---

## Next Steps

1. **Implement llm_singleton.py** (5 min)
2. **Update ValidationGate** (5 min)
3. **Update DirectChatExecutor** (5 min)
4. **Test with "hi"** (1 min)
5. **Verify no rate limits** (monitor)

**Total time:** ~20 minutes

---

**Ready to consolidate?** Let's start with creating the singleton and updating ValidationGate.
