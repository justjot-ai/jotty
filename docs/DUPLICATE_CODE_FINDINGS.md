# Duplicate Code Findings in Jotty

Findings from scanning the codebase for repeated patterns. This doc is a living list; items marked ✅ have been partially or fully addressed.

---

## 1. Tool error responses (`{"success": False, "error": "..."}`)

**Location:** Many skills return error dicts manually instead of using the shared helper.

**Existing helper:** `Jotty.core.infrastructure.utils.tool_helpers`:
- `tool_error(error: str, ...)` → `{"success": False, "error": error, ...}`
- `require_params(params, required)` → returns `tool_error(...)` when a param is missing

**Duplication:** 100+ manual `return {"success": False, "error": "X is required"}` (or similar) across:
- `skills/document_tools/tools.py` ✅ (switched to `tool_error`)
- `skills/messaging_tools/tools.py` ✅ (switched to `tool_error`)
- `skills/claude-api-llm/tools.py`, `skills/terminal-session/tools.py`, `skills/browser-automation/tools.py`, `skills/justjot-converters/tools.py`, and many others

**Recommendation:** Where skills already import from `tool_helpers`, use `tool_error()` and `require_params()` for validation. No need to refactor every skill in one go; do it when touching a file.

---

## 2. Lazy manager / singleton (`_get_manager()`)

**Pattern:** Global `_manager = None`, then `def _get_manager(): global _manager; if _manager is None: ...; return _manager`.

**Duplication:**
- `skills/document_tools/tools.py` – `_get_manager()` → `OutputFormatManager()`
- `skills/messaging_tools/tools.py` – `_get_manager()` → `OutputChannelManager()`

**Status:** Same pattern in two places; logic is only 5–6 lines each. A shared helper (e.g. `lazy_singleton(factory)` in `tool_helpers` or `skills/_infrastructure`) would remove duplication but add indirection. **Left as-is** unless more skills adopt the same pattern.

---

## 3. Result object → tool dict (`_result_to_dict`)

**Pattern:** Convert a result-like object (with `.success`, `.error`, `.metadata`, plus a few extra fields) to a dict for tool response.

**Duplication:**
- `skills/document_tools/tools.py` – `_result_to_dict(result)` with `format`, `file_path`
- `skills/messaging_tools/tools.py` – `_result_to_dict(result)` with `channel`, `message_id`

**Fix:** ✅ Added `result_to_tool_dict(result, include=("format", "file_path"))` (and similar) in `core/infrastructure/utils/tool_helpers.py`. Both skills now use it; each keeps a thin `_result_to_dict()` that calls `result_to_tool_dict(..., include=(...))`.

---

## 4. Module-level singleton (`_instance = None`, `get_*()`)

**Pattern:** `_instance = None` and a `get_*()` that creates on first use.

**Duplication:** 20+ files, e.g.:
- `core/intelligence/reasoning/planners/swarm_resources_stub.py`
- `core/intelligence/orchestration/pipelines/automl_pipeline.py`
- `apps/cli/repl/session.py` (multiple)
- `skills/voice/tools.py`, `skills/browser-automation/tools.py`, `skills/openai-image-gen/tools.py`, etc.

**Recommendation:** Leave as-is. A generic `get_singleton(factory)` would require careful handling of constructor args and test resets. Prefer local pattern unless a module has several singletons.

---

## 5. Registry init pattern (`get_skills_registry(); reg.init(); reg.get_skill(...)`)

**Pattern:** Get registry, call `init()`, then `get_skill(name)`.

**Duplication:** Tests and some manager code (e.g. `document_tools/manager.py`, `messaging_tools/manager.py`).

**Recommendation:** No change. Short and clear; a helper would save one line at most.

---

## 6. Excluded-dirs / skip logic in registry

**Location:** `core/capabilities/registry/skills_registry.py` – `excluded_dirs` is a fixed set used in `_scan_skills_metadata()` and `_scan_plugin_skills()`.

**Status:** Single definition; no duplication.

---

## Summary

| Finding                         | Scope        | Action taken / recommendation                    |
|---------------------------------|-------------|--------------------------------------------------|
| Tool error dicts                | 100+ spots  | Use `tool_error` / `require_params`; done in document_tools, messaging_tools |
| Lazy _get_manager               | 2 files     | Leave as-is unless pattern spreads               |
| Result → tool dict              | 2 files     | ✅ Centralized as `result_to_tool_dict()` in tool_helpers |
| Module singleton                | 20+ files   | Leave as-is                                      |
| Registry get + init + get_skill | Tests/managers | No helper needed                              |
