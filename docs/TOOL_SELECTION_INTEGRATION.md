# Tool Selection Integration (2026-02-16)

## Overview

Integrated advanced tool selection capabilities into Jotty's skills registry:

1. **ToolShed** - LLM-based tool selection (smarter than keyword matching)
2. **ToolInterceptor** - Tool call monitoring for learning systems

---

## Architecture

### File Locations

| Component | Location | Purpose |
|-----------|----------|---------|
| **SkillsRegistry** | `core/capabilities/registry/skills_registry.py` | Main registry (keyword-based discovery) |
| **ToolShed** | `core/capabilities/registry/tool_shed.py` | LLM-based tool selection |
| **ToolInterceptor** | `core/infrastructure/integration/tool_interceptor.py` | Tool execution monitoring |

### Integration Points

```
SkillsRegistry
├── discover()              # Fast: Keyword-based (<100ms)
├── discover_agentic()      # Smart: LLM-based (~2000ms) [NEW]
├── find_tool_chain()       # Multi-step workflow planning [NEW]
└── get_tool_statistics()   # Usage stats for learning [NEW]

ToolInterceptor (in integration layer)
├── wrap_tools()           # Wrap tools for monitoring
├── get_all_calls()        # Get execution history
├── summary()              # Aggregate statistics
└── to_tagged_attempts()   # For ReVal/learning systems
```

---

## Usage Guide

### 1. Fast Keyword-Based Discovery (Current)

```python
from Jotty.core.capabilities.registry import get_skills_registry

registry = get_skills_registry()

# Keyword matching: name, description, capabilities, triggers
skills = registry.discover("Search for AI research papers")
# Returns: ['web-search', 'arxiv-learning', 'research-workflow', ...]
# Speed: ~100ms
```

**How it works:**
- Name match: +3 points/word
- Description match: +1 point/word
- Capability match: +2 points/capability
- Trigger match: +4 points (exact phrase)

---

### 2. Smart LLM-Based Discovery (NEW)

```python
from Jotty.core.capabilities.registry import get_skills_registry

registry = get_skills_registry()

# LLM reasoning instead of keywords
skills = registry.discover_agentic(
    task="Create a bar chart from SQL database",
    required_output="chart"
)
# LLM reasons: "Need SQL execution → data visualization"
# Returns: ['sql-executor', 'data-visualizer'] (in correct order)
# Speed: ~2000ms (requires LLM call)
```

**Advantages:**
- No hardcoded keywords
- Understands intent and context
- Automatically plans multi-step workflows
- Learning from usage statistics

**Fallback:**
- If DSPy not available → automatically falls back to keyword discovery
- Set `fallback_to_keyword=False` to disable fallback

---

### 3. Tool Chaining (NEW)

```python
from Jotty.core.capabilities.registry import get_skills_registry

registry = get_skills_registry()

# Find workflow: PDF → Excel conversion
chain = registry.find_tool_chain(
    start_capability="pdf",
    end_capability="excel",
    max_depth=5
)
# Returns: ['pdf_reader', 'table_extractor', 'excel_writer']
```

**Use cases:**
- Complex multi-step workflows
- Automatic capability bridging
- Document conversion pipelines

---

### 4. Tool Statistics for Learning (NEW)

```python
from Jotty.core.capabilities.registry import get_skills_registry

registry = get_skills_registry()

# Get usage statistics for a specific tool
stats = registry.get_tool_statistics("web_search_tool")
print(stats)
# {
#     'tool_name': 'web_search_tool',
#     'call_count': 42,
#     'success_rate': 0.95,
#     'avg_latency': 1.2
# }
```

**Integration with learning:**
- Feed statistics to TD-Lambda learner
- Q-learning for tool selection
- Improve discovery scoring over time

---

## Tool Interceptor Integration

### Basic Usage

```python
from Jotty.core.infrastructure.integration import ToolInterceptor

# Create interceptor for an executor
interceptor = ToolInterceptor("mcp_executor")

# Wrap tools before passing to LLM
original_tools = {
    "search": search_function,
    "calculate": calc_function
}
wrapped_tools = interceptor.wrap_tools(original_tools)

# Use wrapped tools in execution...
# (all calls are automatically tracked)

# After execution, get statistics
stats = interceptor.summary()
print(stats)
# {
#     'actor': 'mcp_executor',
#     'total_calls': 5,
#     'successful': 4,
#     'failed': 1,
#     'by_tool': {
#         'search': {'total': 3, 'successful': 3, 'failed': 0},
#         'calculate': {'total': 2, 'successful': 1, 'failed': 1}
#     }
# }
```

### Learning Integration

```python
from Jotty.core.infrastructure.integration import ToolInterceptor
from Jotty.core.intelligence.learning.facade import get_td_lambda

# Setup
interceptor = ToolInterceptor("executor")
wrapped_tools = interceptor.wrap_tools(tools)

# ... execute tools ...

# Feed to learning system
td = get_td_lambda()
for call in interceptor.get_all_calls():
    reward = 1.0 if call.success else -0.5

    td.update(
        state={"tool": call.tool_name, "args_count": len(call.args)},
        action={"execute": True},
        reward=reward,
        next_state={"tool": call.tool_name, "completed": True}
    )
```

### Multi-Actor Registry

```python
from Jotty.core.infrastructure.integration import ToolCallRegistry

# Global registry tracks all actors
registry = ToolCallRegistry()

# Get or create interceptor for each actor
interceptor1 = registry.get_or_create_interceptor("researcher")
interceptor2 = registry.get_or_create_interceptor("coder")

# ... use interceptors ...

# Aggregate statistics across all actors
global_stats = registry.summary()
print(global_stats)
# {
#     'total_calls': 10,
#     'successful': 8,
#     'failed': 2,
#     'by_actor': {
#         'researcher': {...},
#         'coder': {...}
#     }
# }
```

---

## When to Use Each Method

| Scenario | Method | Reason |
|----------|--------|--------|
| **Fast skill lookup** | `discover()` | Keyword matching is instant |
| **Complex intent understanding** | `discover_agentic()` | LLM understands nuance |
| **Multi-step workflows** | `find_tool_chain()` | Automatic workflow planning |
| **Learning from usage** | `get_tool_statistics()` | Improve selection over time |
| **Tool execution tracking** | `ToolInterceptor` | Feed learning system |

---

## Performance Comparison

```python
import time

# Keyword-based (fast)
start = time.time()
skills = registry.discover("Search for AI research")
print(f"Keyword: {time.time() - start:.3f}s")  # ~0.001s

# LLM-based (smart)
start = time.time()
skills = registry.discover_agentic(
    task="Search for AI research",
    required_output="papers"
)
print(f"LLM-based: {time.time() - start:.3f}s")  # ~2.0s
```

**Recommendation:**
- Use keyword discovery for most cases (fast, good enough)
- Use agentic discovery for complex/ambiguous tasks
- Use tool chaining for multi-step workflows
- Use statistics for continuous improvement

---

## Examples

### Example 1: Hybrid Discovery

```python
# Try fast first, fall back to smart if needed
def smart_discover(task, min_confidence=0.7):
    # Fast keyword-based attempt
    skills = registry.discover(task, max_results=10)

    # If top skill has low relevance, use LLM
    if skills and skills[0]['relevance_score'] < 5:
        print("Low confidence, using LLM discovery...")
        skills = registry.discover_agentic(task)

    return skills
```

### Example 2: Learning-Enhanced Discovery

```python
# Use statistics to boost successful tools
def learning_enhanced_discover(task):
    skills = registry.discover(task)

    # Boost skills based on success rate
    for skill in skills:
        for tool_name in skill['tools']:
            stats = registry.get_tool_statistics(tool_name)
            if stats and stats['success_rate'] > 0.8:
                skill['relevance_score'] += 2  # Boost proven tools

    # Re-sort by adjusted score
    skills.sort(key=lambda s: -s['relevance_score'])
    return skills
```

### Example 3: Complete Workflow

```python
# PDF → Analysis → Report workflow
task = "Analyze data from PDF and create report"

# Find complete chain
chain = registry.find_tool_chain("pdf", "report")
# Returns: ['pdf_reader', 'data_analyzer', 'report_generator']

# Get tools for each step
for tool_name in chain:
    # Find which skill provides this tool
    for skill in registry.loaded_skills.values():
        if tool_name in skill.tools:
            print(f"{tool_name} → {skill.name}")
            break
```

---

## Migration Notes

### No Breaking Changes

All existing code continues to work:
- `discover()` is unchanged
- Keyword-based discovery is still the default
- LLM-based features are opt-in

### Optional Dependencies

ToolShed requires DSPy:
```bash
pip install dspy-ai
```

If DSPy is not installed:
- `discover_agentic()` falls back to `discover()`
- `find_tool_chain()` returns empty list
- No errors, graceful degradation

---

## Future Enhancements

1. **Automatic fallback learning**: Track when keyword discovery fails and LLM succeeds
2. **Hybrid scoring**: Combine keyword + LLM confidence scores
3. **Tool execution prediction**: Predict which tools will succeed before calling
4. **Workflow templates**: Cache successful tool chains for reuse

---

## References

- Original files: `core/infrastructure/metadata/tool_shed.py`, `tool_interceptor.py`
- Integration PR: Tool Selection Integration (2026-02-16)
- Related: ProviderSelector (Q-learning for provider selection in `skills/_infrastructure/`)
