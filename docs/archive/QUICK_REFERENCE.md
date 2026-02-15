# Interface Refactoring - Quick Reference

**For Developers:** What changed and how to use the new DRY patterns.

---

## 1. Creating New Use Cases

### ❌ OLD WAY (Don't do this)
```python
class MyUseCase(BaseUseCase):
    async def execute(self, goal, context=None, **kwargs):
        start_time = time.time()
        try:
            result = await self.executor.execute(...)
            execution_time = time.time() - start_time
            return self._create_result(...)
        except Exception as e:
            # 20 more lines of error handling...
```

### ✅ NEW WAY (Use the DRY wrapper)
```python
class MyUseCase(BaseUseCase):
    async def execute(self, goal, context=None, **kwargs):
        # Just call the wrapper!
        return await self._execute_with_error_handling(
            self.executor.execute,
            goal=goal,
            context=context
        )

    # Override these to customize behavior
    def _extract_output(self, result):
        return result.get("my_output_key")

    def _extract_metadata(self, result, execution_time):
        return {"custom": result.get("custom_field")}
```

**Benefit:** No more duplicate try/except/timing code. Just 3 lines!

---

## 2. Converting External Messages

### ❌ OLD WAY (Still works, but verbose)
```python
# Telegram
msg = JottyMessage.from_telegram(telegram_update, session_id="sess1")

# Web
msg = JottyMessage.from_web(request_data, user_id="user1")

# CLI
msg = JottyMessage.from_cli("hello", session_id="sess1")
```

### ✅ NEW WAY (Unified adapter - cleaner!)
```python
# Single entry point for all interfaces
msg = MessageAdapter.from_source(
    InterfaceType.TELEGRAM,  # or WEB, CLI
    data,                    # telegram_update, request_data, or text string
    session_id="sess1"
)
```

**Benefit:** Strategy pattern - one method handles all conversions.

**Note:** Old methods still work (they delegate to MessageAdapter).

---

## 3. Serializing Dataclasses

### ❌ OLD WAY (Manual dict construction)
```python
@dataclass
class MyData:
    name: str
    value: int

    def to_dict(self):
        return {
            "name": self.name,
            "value": self.value,
            # ... repeat for all fields
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            name=data.get("name", ""),
            value=data.get("value", 0),
            # ... repeat for all fields
        )
```

### ✅ NEW WAY (Use dataclass utilities)
```python
from dataclasses import asdict, fields as dataclass_fields

@dataclass
class MyData:
    name: str
    value: int

    def to_dict(self):
        # DRY: Let dataclasses do the work
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        # DRY: Auto-filter valid fields
        valid_fields = {f.name for f in dataclass_fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**kwargs)
```

**Benefit:** Automatic, safe, no manual field listing.

---

## 4. Creating Use Cases in APIs

### ❌ OLD WAY (Repeated creation logic)
```python
if agent_id:
    chat = ChatUseCase(
        conductor=self.conductor,
        agent_id=agent_id,
        config=UseCaseConfig(use_case_type=UseCaseType.CHAT)
    )
else:
    chat = self.chat
```

### ✅ NEW WAY (Use factory)
```python
# If you need custom config, use factory
chat = self._create_use_case(
    ChatUseCase,
    UseCaseType.CHAT,
    agent_id=agent_id  # kwargs passed through
) if agent_id else self.chat
```

**Benefit:** Eliminates 4x duplication across execute/stream methods.

---

## 5. Host Implementations

### ❌ OLD WAY (Magic values)
```python
class MyHost(Host):
    def notify(self, message, level="info"):
        # Inline dict construction
        logger.log({'info': 20, 'warning': 30}[level], message)
```

### ✅ NEW WAY (Named constants)
```python
class MyHost(Host):
    # DRY: Constants at class level
    _LOG_LEVELS = {'info': logging.INFO, 'warning': logging.WARNING}
    _ICONS = {'info': 'ℹ', 'warning': '⚠️'}

    def notify(self, message, level="info"):
        log_level = self._LOG_LEVELS.get(level, logging.INFO)
        icon = self._ICONS.get(level, '')
        logger.log(log_level, f"{icon} {message}")
```

**Benefit:** Clear, maintainable, easy to extend.

---

## Pattern Cheat Sheet

| Pattern | When to Use | Example |
|---------|-------------|---------|
| **Error Wrapper** | Any use case with execute() | `await self._execute_with_error_handling(...)` |
| **Factory Method** | Creating use cases with variations | `self._create_use_case(ChatUseCase, ...)` |
| **Message Adapter** | Converting external messages | `MessageAdapter.from_source(type, data)` |
| **Dataclass Utils** | Serializing dataclasses | `asdict(self)`, `dataclass_fields(cls)` |
| **Named Constants** | Repeated literals/configs | `_LOG_LEVELS = {...}` at class level |

---

## Migration Guide

**No migration needed!** All changes are backwards compatible.

**Existing code continues to work:**
- ✅ `JottyMessage.from_telegram()` still works
- ✅ Manual `to_dict()` implementations still work
- ✅ Existing use cases continue functioning

**To adopt new patterns:**
1. New use cases: Use `_execute_with_error_handling()`
2. New converters: Use `MessageAdapter.from_source()`
3. New dataclasses: Use `asdict()` and `dataclass_fields()`
4. New hosts: Extract constants to class level

---

## Questions?

**Q: Do I need to update existing code?**
A: No! Everything is backwards compatible.

**Q: When should I use the new patterns?**
A: For all new code. Migrate existing code opportunistically.

**Q: What if I need custom behavior?**
A: Override the hook methods (`_extract_output`, `_extract_metadata`, etc.)

**Q: Can I still use old methods?**
A: Yes! They delegate to new implementations internally.

---

## Examples in Codebase

**Error handling wrapper:**
- `use_cases/chat/chat_use_case.py:62`
- `use_cases/workflow/workflow_use_case.py:66`

**Factory method:**
- `api/unified.py:90-120`

**Message adapter:**
- `interfaces/message.py:367-480`

**Dataclass utilities:**
- `interfaces/message.py:38-57` (Attachment)
- `interfaces/message.py:304-328` (InternalEvent)

**Named constants:**
- `interfaces/host_provider.py:60-82` (NullHost)
- `interfaces/host_provider.py:84-126` (CLIHost)

---

**Updated:** 2026-02-15
**Status:** Production-ready ✅
