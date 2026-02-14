# Error Handling Guide - Best Practices

## ✅ Good Error Handling Patterns

### 1. Use Specific Exceptions

**❌ Bad:**
```python
try:
    result = agent.execute(task)
except Exception as e:
    logger.error(f"Failed: {e}")
    return None
```

**✅ Good:**
```python
from core.foundation.exceptions import AgentExecutionError, TimeoutError

try:
    result = agent.execute(task)
except TimeoutError as e:
    logger.warning(f"Agent timed out: {e}. Retrying with longer timeout...")
    # Retry with adjusted timeout
except AgentExecutionError as e:
    logger.error(f"Agent execution failed: {e}")
    # Handle gracefully
except Exception as e:
    # Catch-all for unexpected errors - MUST log
    logger.exception(f"Unexpected error in agent execution: {e}")
    raise  # Re-raise unexpected errors
```

### 2. Add Context to Errors

**❌ Bad:**
```python
if not api_key:
    raise ValueError("API key missing")
```

**✅ Good:**
```python
from core.foundation.helpful_errors import MissingEnvVarError

if not api_key:
    raise MissingEnvVarError(
        "OPENAI_API_KEY",
        purpose="LLM calls to OpenAI GPT models"
    )
```

### 3. Provide Actionable Suggestions

**❌ Bad:**
```python
logger.error("JSON parsing failed")
```

**✅ Good:**
```python
from core.foundation.helpful_errors import JSONParseError

try:
    data = json.loads(response)
except json.JSONDecodeError as e:
    raise JSONParseError(
        source="LLM response",
        preview=response[:200],
        original_error=e
    )
# User sees:
# "Failed to parse JSON from LLM response
#  Response preview: {text...}
#  💡 Suggestion: Ensure your prompt clearly requests JSON format"
```

### 4. Log at Appropriate Levels

```python
import logging
logger = logging.getLogger(__name__)

# DEBUG: Detailed diagnostic info
logger.debug(f"Agent received input: {input_data}")

# INFO: Normal operation milestones
logger.info(f"Agent {name} started execution")

# WARNING: Recoverable issues
logger.warning(f"JSON parsing failed, using fallback")

# ERROR: Operation failed but application continues
logger.error(f"Agent execution failed: {e}")

# EXCEPTION: Same as ERROR but includes stack trace
logger.exception(f"Unexpected error: {e}")

# CRITICAL: Application cannot continue
logger.critical(f"Database connection lost")
```

## 📋 Exception Hierarchy

Use the right exception for the job:

```python
from core.foundation.exceptions import (
    # Configuration errors
    InvalidConfigError,      # Config values are wrong
    MissingConfigError,      # Required config missing

    # Execution errors
    AgentExecutionError,     # Agent failed to execute
    ToolExecutionError,      # Tool/function call failed
    TimeoutError,            # Operation timed out
    CircuitBreakerError,     # Circuit breaker is open

    # Context errors
    ContextOverflowError,    # Too much context for token limit
    CompressionError,        # Failed to compress content

    # Memory errors
    MemoryRetrievalError,    # Failed to retrieve memory
    MemoryStorageError,      # Failed to store memory

    # Learning errors
    RewardCalculationError,  # Reward calculation failed
    CreditAssignmentError,   # Credit assignment failed

    # Integration errors
    LLMError,                # LLM call failed
    DSPyError,               # DSPy-specific error
)
```

## 🔧 Common Patterns

### Pattern 1: Retry with Backoff

```python
import time
from core.foundation.exceptions import LLMError

max_retries = 3
for attempt in range(max_retries):
    try:
        result = llm.call(prompt)
        break  # Success
    except LLMError as e:
        if attempt == max_retries - 1:
            raise  # Final attempt failed

        wait_time = 2 ** attempt  # Exponential backoff
        logger.warning(f"LLM call failed (attempt {attempt+1}/{max_retries}): {e}. "
                      f"Retrying in {wait_time}s...")
        time.sleep(wait_time)
```

### Pattern 2: Fallback to Default

```python
from core.foundation.helpful_errors import JSONParseError

try:
    data = json.loads(response)
except json.JSONDecodeError as e:
    logger.warning(f"JSON parsing failed, using empty list. Error: {e}")
    logger.debug(f"Response that failed to parse: {response[:200]}...")
    data = []  # Safe fallback
```

### Pattern 3: Context Manager for Cleanup

```python
from contextlib import contextmanager

@contextmanager
def temp_model_context(model_name: str):
    """Switch LLM model temporarily."""
    old_model = current_model
    try:
        set_model(model_name)
        yield
    except Exception as e:
        logger.error(f"Error with model {model_name}: {e}")
        raise
    finally:
        set_model(old_model)  # Always restore

# Usage:
with temp_model_context("haiku"):
    result = fast_operation()
```

### Pattern 4: Validation with Clear Errors

```python
def validate_config(config: SwarmBaseConfig) -> None:
    """Validate configuration with helpful error messages."""

    if config.max_retries < 0:
        raise InvalidConfigError(
            message=f"max_retries must be >= 0, got {config.max_retries}",
            context={"field": "max_retries", "value": config.max_retries},
            suggestion="Set max_retries to 3 (recommended) or 0 to disable retries"
        )

    if config.timeout_seconds <= 0:
        raise InvalidConfigError(
            message=f"timeout_seconds must be > 0, got {config.timeout_seconds}",
            context={"field": "timeout_seconds", "value": config.timeout_seconds},
            suggestion="Set timeout_seconds to 300 (5 minutes) or adjust based on task complexity"
        )
```

## 🚫 Anti-Patterns (Don't Do This)

### 1. Silent Failures
```python
# ❌ DON'T
try:
    process()
except:
    pass  # Silent failure - bugs hide here!
```

### 2. Catching Too Broadly Without Logging
```python
# ❌ DON'T
try:
    result = complex_operation()
except Exception:
    return None  # What went wrong? We'll never know!
```

### 3. Bare Except
```python
# ❌ DON'T
try:
    do_something()
except:  # Catches EVERYTHING including KeyboardInterrupt!
    handle_error()
```

### 4. Swallowing Important Information
```python
# ❌ DON'T
try:
    result = parse_json(data)
except json.JSONDecodeError:
    logger.error("JSON error")  # Where? What data? What was expected?
    return {}
```

## ✅ Checklist for Error Handling

When adding error handling, ensure:

- [ ] Use specific exception types (not bare `except:` or `except Exception:`)
- [ ] Log at appropriate level (DEBUG/INFO/WARNING/ERROR)
- [ ] Include context in error messages (what failed, why, how to fix)
- [ ] Provide actionable suggestions when possible
- [ ] Re-raise unexpected errors after logging
- [ ] Clean up resources in `finally` blocks
- [ ] Test both success and failure paths
- [ ] Document expected exceptions in docstrings

## 📚 Related Documentation

- `core/foundation/exceptions.py` - Complete exception hierarchy
- `core/foundation/helpful_errors.py` - Error messages with suggestions
- `docs/JOTTY_ARCHITECTURE.md` - Overall architecture
- `Jotty/CLAUDE.md` - Quick reference

---

**Remember:** Good error messages save hours of debugging. Invest time making them helpful!
