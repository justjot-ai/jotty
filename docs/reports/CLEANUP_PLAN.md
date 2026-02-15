# Code Cleanup Plan - Redundant Files

**Date:** 2026-02-15
**Status:** Audit Complete

---

## Summary

After migrating to shared components, we have redundant old implementations that should be cleaned up.

---

## Apps Directory - Duplicates Found

### 1. CLI (apps/cli/)

| File | Size | Status | Action |
|------|------|--------|--------|
| **app.py** | 8KB | ✅ NEW - Uses shared components | **KEEP** |
| **app.py** | 48KB | ❌ OLD - Pre-migration | **ARCHIVE or DELETE** |
| **__main__.py** | 12KB | ⚠️ References old app.py | **UPDATE to use app_migrated** |
| **main.py** | 1KB | ⚠️ Entry point | **CHECK usage** |

**Issue:** `__main__.py` line 256 still imports from old `app.py`:
```python
from .app import JottyCLI  # OLD
```

**Should be:**
```python
from .app_migrated import JottyCLI  # NEW
```

---

### 2. Telegram (apps/telegram/)

| File | Size | Status | Action |
|------|------|--------|--------|
| **bot.py** | 19KB | ✅ NEW - Full implementation | **KEEP** |
| **bot_migrated.py** | 8KB | ⚠️ INTERMEDIATE - Partial migration | **ARCHIVE** |
| **bot.py** | 47KB | ❌ OLD - Pre-migration | **ARCHIVE or DELETE** |
| **renderer.py** | 8KB | ❌ OLD - Replaced by shared/renderers/telegram_renderer.py | **DELETE** |
| **__main__.py** | 2KB | ⚠️ Check which bot it uses | **UPDATE if needed** |

**Current:** Using `bot.py` via `start_telegram_bot_full.sh`

**Redundant:**
- `bot.py` - Old implementation (47KB)
- `bot_migrated.py` - Intermediate version (8KB)
- `renderer.py` - Replaced by `apps/shared/renderers/telegram_renderer.py`

---

### 3. Web (apps/web/)

| File | Status | Action |
|------|--------|--------|
| **backend/server.py** | ✅ NEW - Uses shared components | **KEEP** |
| **frontend/src/App.tsx** | ✅ NEW - Simplified standalone | **KEEP** |
| **frontend/src/App.css** | ✅ NEW | **KEEP** |
| **frontend/src/index.tsx** | ✅ NEW | **KEEP** |

**Status:** ✅ Clean - No duplicates found

---

### 4. Shared Components (apps/shared/)

| Directory/File | Status | Action |
|----------------|--------|--------|
| **models.py** | ✅ Core models | **KEEP** |
| **state.py** | ✅ State machine | **KEEP** |
| **interface.py** | ✅ Abstract base classes | **KEEP** |
| **events.py** | ✅ Event processor | **KEEP** |
| **renderers/terminal.py** | ✅ TUI renderer | **KEEP** |
| **renderers/telegram_renderer.py** | ✅ Telegram renderer | **KEEP** |
| **renderers/web.tsx** | ✅ Web renderer | **KEEP** |

**Status:** ✅ Clean - All files actively used

---

## Core Directory - Check for Redundancy

### Executors

| File | Status | Action |
|------|--------|--------|
| **core/intelligence/orchestration/direct_chat_executor.py** | ✅ NEW - Simple queries | **KEEP** |
| **core/intelligence/orchestration/unified_executor.py** | ✅ Complex queries | **KEEP** |
| **core/interface/api/mode_router.py** | ✅ Updated with ValidationGate | **KEEP** |

**Status:** ✅ Both executors needed (simple vs complex routing)

---

## Recommended Cleanup Actions

### Priority 1: Fix Active References

1. **Update CLI __main__.py** to use new app:
   ```bash
   # File: apps/cli/__main__.py
   # Line 256: Change
   from .app import JottyCLI  # OLD
   # To:
   from .app_migrated import JottyCLI  # NEW
   ```

2. **Update Telegram __main__.py** if needed:
   ```bash
   # File: apps/telegram/__main__.py
   # Check which bot it references and update to bot_migrated_full
   ```

3. **Update start scripts** to use new versions:
   ```bash
   # Already done: start_telegram_bot_full.sh uses bot.py ✅
   ```

---

### Priority 2: Archive Old Implementations

Create `apps/_archived/` directory and move old files:

```bash
mkdir -p apps/_archived/cli
mkdir -p apps/_archived/telegram

# Move old CLI
mv apps/cli/app.py apps/_archived/cli/app_old.py

# Move old Telegram
mv apps/telegram/bot.py apps/_archived/telegram/bot_old.py
mv apps/telegram/bot_migrated.py apps/_archived/telegram/bot_migrated_intermediate.py
mv apps/telegram/renderer.py apps/_archived/telegram/renderer_old.py
```

**Benefits:**
- ✅ Keeps old code for reference
- ✅ Cleans up working directory
- ✅ Easy to compare old vs new if needed
- ✅ Can delete `_archived/` later once confirmed stable

---

### Priority 3: Delete Truly Redundant Files

**AFTER** verifying everything works with new implementations:

```bash
# Delete archived files (optional, after 30 days of stable operation)
rm -rf apps/_archived/
```

**Only delete if:**
- ✅ New implementation tested thoroughly
- ✅ No references to old files found
- ✅ Git history preserved (old code accessible via git)

---

## Files to Keep (Active)

### Apps
- ✅ `apps/cli/app.py`
- ✅ `apps/telegram/bot.py`
- ✅ `apps/web/backend/server.py`
- ✅ `apps/web/frontend/src/*` (App.tsx, index.tsx, App.css)
- ✅ All `apps/shared/*` files

### Core
- ✅ `core/intelligence/orchestration/direct_chat_executor.py` (NEW)
- ✅ `core/intelligence/orchestration/unified_executor.py`
- ✅ `core/intelligence/orchestration/validation_gate.py`
- ✅ `core/intelligence/orchestration/model_tier_router.py`
- ✅ `core/interface/api/mode_router.py`

---

## Files to Archive/Delete

### Archive (move to apps/_archived/)
- ❌ `apps/cli/app.py` (48KB old implementation)
- ❌ `apps/telegram/bot.py` (47KB old implementation)
- ❌ `apps/telegram/bot_migrated.py` (8KB intermediate)
- ❌ `apps/telegram/renderer.py` (replaced by shared renderer)

### Check & Update
- ⚠️ `apps/cli/__main__.py` - Update to use app_migrated
- ⚠️ `apps/cli/main.py` - Check if still needed
- ⚠️ `apps/telegram/__main__.py` - Update if needed

---

## Verification Steps

Before archiving/deleting:

1. **Test TUI with new app:**
   ```bash
   python -m apps.cli.app_migrated
   # Should work without errors
   ```

2. **Test Telegram with new bot:**
   ```bash
   ./start_telegram_bot_full.sh
   # Should start bot.py
   ```

3. **Test Web:**
   ```bash
   # Backend
   python apps/web/backend/server.py

   # Frontend
   cd apps/web/frontend && npm start
   ```

4. **Grep for old file references:**
   ```bash
   grep -r "from.*\.app import" apps/
   grep -r "import.*bot\.py" apps/
   grep -r "renderer\.py" apps/
   ```

5. **Check startup scripts:**
   ```bash
   grep -r "app\.py" *.sh
   grep -r "bot\.py" *.sh
   ```

---

## Benefits of Cleanup

### Code Quality
- ✅ Clearer codebase structure
- ✅ No confusion about which file to use
- ✅ Easier onboarding for new developers
- ✅ Reduced maintenance burden

### Performance
- ✅ Faster IDE indexing (less code to index)
- ✅ Clearer import paths
- ✅ No accidental imports of old code

### Disk Space
- Old files: 48KB (app.py) + 47KB (bot.py) + 8KB (renderer.py) = ~103KB
- Not huge, but cleaner is better

---

## Rollback Plan

If new implementations have issues:

1. **Keep git history:**
   ```bash
   git log apps/cli/app.py
   git checkout <commit> -- apps/cli/app.py
   ```

2. **Restore from _archived:**
   ```bash
   cp apps/_archived/cli/app_old.py apps/cli/app.py
   ```

3. **Update __main__.py back to old import**

---

## Next Steps

1. ✅ **Update __main__.py files** to use new implementations
2. ✅ **Test all three platforms** (TUI, Telegram, Web)
3. ✅ **Create _archived directory** and move old files
4. ✅ **Update documentation** to reference new files only
5. ⏳ **Wait 30 days** to ensure stability
6. ⏳ **Delete _archived** once confirmed stable

---

## Summary

**Total Redundant Files:** 4 files (~103KB)
- `apps/cli/app.py` (48KB)
- `apps/telegram/bot.py` (47KB)
- `apps/telegram/bot_migrated.py` (8KB)
- `apps/telegram/renderer.py` (8KB)

**Action:** Archive (don't delete yet) and update references

**Timeline:**
- **Now:** Update __main__.py references
- **Day 0-7:** Test thoroughly
- **Day 30:** Delete archived files if stable

---

**Ready to execute cleanup?** Start with Priority 1 (fixing references).
