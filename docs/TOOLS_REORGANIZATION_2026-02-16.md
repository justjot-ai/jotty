# Tool Infrastructure Reorganization - 2026-02-16

## Summary

Moved tool-related infrastructure from `core/intelligence/orchestration/` to `skills/_tools/` to follow the established skills infrastructure pattern.

## Changes

### Files Moved

```bash
core/intelligence/orchestration/tool_generator.py  → skills/_tools/tool_generator.py
core/intelligence/orchestration/tool_management.py → skills/_tools/tool_management.py
```

### New Structure

```
skills/
├── _infrastructure/          # Skill infrastructure (base classes, registry)
│   ├── base.py               # SkillProvider base, SkillCategory
│   └── provider_registry.py  # ProviderRegistry, ProviderSelector
│
├── _providers/               # External skill providers
│   ├── n8n_provider.py       # n8n integration
│   ├── streamlit_provider.py # Streamlit apps
│   └── morph_provider.py     # Morph apps
│
├── _tools/                   # ✅ NEW - Tool management infrastructure
│   ├── __init__.py           # Public exports
│   ├── tool_generator.py     # UnifiedToolGenerator (1,139 lines)
│   └── tool_management.py    # ToolManager (339 lines)
│
├── web-search/               # Individual skills
└── [273 other skills]
```

## Why This Makes Sense

### Pattern Consistency

The `skills/` directory already has infrastructure subdirectories prefixed with `_`:
- `_infrastructure/` - Base classes for skill providers
- `_providers/` - External skill integrations
- `_tools/` - Tool generation and management (NEW)

This follows the Python convention where `_` prefix indicates "internal/infrastructure" code.

### Semantic Correctness

**tool_generator.py:**
- Converts skills from SkillsRegistry → Claude API tool format
- Generates tool schemas for input/output/visualization/skills
- Directly depends on SkillsRegistry
- **Role**: Skills infrastructure (formatting)

**tool_management.py:**
- Tracks tool/skill performance (success rates)
- Suggests replacements for failing tools
- Manages per-swarm tool selection
- **Role**: Skills infrastructure (performance tracking)

Both are **about skills**, not orchestration logic.

### What Orchestration Should Contain

`core/intelligence/orchestration/` is for:
- ✅ Swarm coordination (SwarmIntelligence)
- ✅ Multi-agent patterns (ParadigmExecutor)
- ✅ Agent execution (AgentRunner)
- ✅ Learning pipelines (SwarmLearner)

**NOT:**
- ❌ Tool schema generation (that's capabilities/skills)
- ❌ Tool performance tracking (that's learning about skills)

## Files Updated

### Import Changes

**core/intelligence/orchestration/unified_executor.py:**
```python
# Before
from .tool_generator import UnifiedToolGenerator

# After
from Jotty.skills._tools import UnifiedToolGenerator
```

**core/intelligence/orchestration/swarm_intelligence.py:**
```python
# Before
from .tool_management import ToolManager

# After
from Jotty.skills._tools import ToolManager
```

**core/intelligence/orchestration/__init__.py:**
```python
# Updated lazy imports
"UnifiedToolGenerator": ("Jotty.skills._tools", "UnifiedToolGenerator"),
"ToolDefinition": ("Jotty.skills._tools.tool_generator", "ToolDefinition"),
```

**skills/_tools/tool_management.py:**
```python
# Fixed import after tool_shed moved
# Before
from Jotty.core.infrastructure.metadata.tool_shed import ToolShedSchema

# After
from Jotty.core.capabilities.registry.tool_shed import ToolShedSchema
```

## Usage (No Breaking Changes)

### Public API Unchanged

Code using the public API continues to work:

```python
# Still works (via lazy imports)
from Jotty.core.intelligence.orchestration import UnifiedToolGenerator

# Also works (new direct import)
from Jotty.skills._tools import UnifiedToolGenerator, ToolManager
```

### Internal Usage

```python
# unified_executor.py uses UnifiedToolGenerator
from Jotty.skills._tools import UnifiedToolGenerator

executor = UnifiedExecutor()
executor.tool_generator = UnifiedToolGenerator()
tools = executor.tool_generator.generate_all_tools()
```

```python
# swarm_intelligence.py uses ToolManager
from Jotty.skills._tools import ToolManager

swarm = SwarmIntelligence()
swarm.tool_manager = ToolManager()
analysis = swarm.tool_manager.analyze_tools(success_rates, "research_swarm")
```

## Benefits

1. **✅ Clearer architecture** - Tool infrastructure lives with skills
2. **✅ Follows established pattern** - Consistent with `_infrastructure/`, `_providers/`
3. **✅ Better discoverability** - All skill-related infrastructure in one place
4. **✅ Semantic correctness** - Files are where they logically belong
5. **✅ No breaking changes** - Public API preserved via lazy imports

## Related Work

This continues the reorganization started with:
- Tool learning integration (2026-02-16)
- Registry cleanup (2026-02-16)
- Skill_sdk deletion (2026-02-15)

All part of establishing clean, discoverable architecture.

## File Descriptions

### tool_generator.py (1,139 lines)

**UnifiedToolGenerator** - Auto-generates Claude API tool definitions from:
1. **Input tools**: web_search, file_read, fetch_url
2. **Output tools**: save_docx, save_pdf, save_slides, send_telegram
3. **Visualization tools**: 70+ section types from schema registry
4. **Skills tools**: Dynamic from SkillsRegistry (273 skills)

**Purpose:** Single LLM call where Claude decides which tools to use, eliminating DSPy signatures.

**Key Methods:**
- `generate_all_tools()` - Generate all available tools
- `get_executor(tool_name)` - Get execution function for a tool
- `get_tools_by_category(category)` - Filter by input/output/visualization/skill

### tool_management.py (339 lines)

**ToolManager** - Dynamic tool management based on Agent0 learned performance:
- Tracks per-tool success rates (successes/total uses)
- Suggests replacements for failing tools (<60% success)
- Manages per-swarm tool additions/removals
- Auto-registers tools from usage tracking

**Purpose:** Learn which tools work well, adapt tool selection over time.

**Key Methods:**
- `analyze_tools(success_rates, swarm_name)` - Classify weak/strong tools
- `find_replacements(failing_tool)` - Suggest alternatives
- `get_active_tools(swarm_name, defaults)` - Merged tool list
- `auto_register_from_rates(tool_rates)` - Auto-populate registry

## Testing

No test changes required - both files are used in production:
- ✅ `unified_executor.py` uses `UnifiedToolGenerator` (3 call sites)
- ✅ `swarm_intelligence.py` uses `ToolManager` (4 call sites)

Integration verified - files are actually used, not just sitting unused.

## Git Commit

```bash
git mv core/intelligence/orchestration/tool_generator.py skills/_tools/tool_generator.py
git mv core/intelligence/orchestration/tool_management.py skills/_tools/tool_management.py

# Created: skills/_tools/__init__.py
# Updated: unified_executor.py, swarm_intelligence.py, orchestration/__init__.py
```

---

**Date:** 2026-02-16
**Status:** ✅ Complete
**Breaking Changes:** None (public API preserved)
**Pattern:** Follows `skills/_infrastructure/`, `skills/_providers/` convention
