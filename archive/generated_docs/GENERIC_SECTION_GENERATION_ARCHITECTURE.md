# Generic Section Generation - True DRY Architecture

## Problem: Current Approach Doesn't Scale

**Anti-pattern (not DRY):**
```python
if 'markdown' in query:
    return _get_markdown_summary()
elif 'chart' in query:
    return _get_chart()
elif 'gantt' in query:
    return _get_gantt()
# ... 70+ section types = 70+ methods ❌
```

## Solution: LLM Chooses Section Type

**LLM has tools/functions for all section types:**

```python
# LLM system prompt
SECTION_TOOLS = """
You can return data in various formats using these helpers:

Project Management:
- return_kanban(columns=[{id, title, items}]) - Visual task board
- return_gantt(tasks=[{name, start, end}]) - Timeline chart
- return_roadmap(phases=[...]) - Product roadmap

Data Visualization:
- return_chart(type='bar|line|pie', data={...}) - Charts
- return_data_table(csv_data) - Tables

Documentation:
- return_text(markdown) - Markdown text
- return_mermaid(diagram) - Flowcharts

... (all 70+ types, 1 line each)

Choose the BEST format based on user query.
"""
```

**Generic handler:**
```python
class ChatAssistant:
    async def run(self, goal: str) -> Dict[str, Any]:
        # Detect intent
        if self._is_task_query(goal):
            tasks = await self._fetch_tasks()

            # LLM chooses section type and generates content
            return await self._llm_generate_response(
                query=goal,
                data={'tasks': tasks},
                available_tools=SECTION_HELPERS
            )

        # ... other intents

    async def _llm_generate_response(self, query, data, available_tools):
        """
        Let LLM choose section type and generate content.

        LLM has function calling access to all section helpers.
        """
        messages = [
            {"role": "system", "content": SECTION_TOOLS},
            {"role": "user", "content": f"{query}\n\nData: {data}"}
        ]

        # LLM calls appropriate helper
        response = await llm.chat(
            messages=messages,
            tools=available_tools,  # return_kanban, return_chart, etc.
            tool_choice="required"   # Must use a tool
        )

        # LLM chooses: return_kanban(...) or return_text(...) or return_chart(...)
        return response  # Already formatted correctly!
```

## Implementation Strategy

### Phase 1: Intent Detection (Generic)

```python
async def run(self, goal: str):
    """Single entry point - no hardcoded section logic."""

    # Generic intent detection
    if self._is_task_query(goal):
        context = {'tasks': await self._fetch_tasks()}
    elif self._is_status_query(goal):
        context = {'status': await self._get_status()}
    else:
        context = {}

    # LLM decides everything
    return await self._llm_generate(query=goal, context=context)
```

### Phase 2: LLM Tool Calling

```python
# Register all section helpers as tools
SECTION_TOOLS = {
    'return_kanban': {
        'description': 'Return kanban board for task tracking',
        'parameters': {
            'columns': {'type': 'array', 'description': 'Board columns'}
        }
    },
    'return_text': {
        'description': 'Return markdown text',
        'parameters': {
            'content': {'type': 'string', 'description': 'Markdown content'}
        }
    },
    # ... auto-generate from section schemas!
}

async def _llm_generate(self, query, context):
    response = await llm.chat(
        messages=[...],
        tools=SECTION_TOOLS,
        tool_choice='required'
    )

    # LLM chose: {"name": "return_text", "arguments": {"content": "# Summary\n..."}}
    tool_call = response.tool_calls[0]

    # Execute the tool
    if tool_call.name == 'return_kanban':
        return return_kanban(**tool_call.arguments)
    elif tool_call.name == 'return_text':
        return return_text(**tool_call.arguments)
    # ... dispatch to appropriate helper
```

### Phase 3: Auto-Generate Tool Definitions from Schemas

```python
# Extract from section-schemas.generated.json
def generate_tool_definitions():
    tools = []
    for section_type, schema in SECTION_SCHEMAS.items():
        tools.append({
            'name': f'return_{section_type.replace("-", "_")}',
            'description': schema['description'],
            'parameters': schema['schema'],  # JSON schema
            'hint': schema['llmHint']
        })
    return tools

# Auto-generated tool definitions for ALL 70+ types
SECTION_TOOLS = generate_tool_definitions()
```

## Benefits

✅ **True DRY:** No per-section methods, LLM chooses
✅ **Scalable:** Add section type = auto-available to LLM
✅ **Flexible:** LLM can combine multiple sections
✅ **Smart:** LLM interprets user intent naturally
✅ **Auto-validated:** Schema validator ensures correctness

## Example Usage

**User:** "show tasks"
**LLM thinks:** Query mentions tasks, visual format best → use `return_kanban`
**LLM calls:** `return_kanban(columns=[...])`
**Result:** Kanban board

**User:** "summarize tasks in markdown"
**LLM thinks:** User wants markdown → use `return_text`
**LLM calls:** `return_text(content="# Summary\n...")`
**Result:** Markdown summary

**User:** "chart task completion over time"
**LLM thinks:** Time-series data → use `return_chart`
**LLM calls:** `return_chart(type='line', data={...})`
**Result:** Line chart

## Migration Path

### Step 1: Add LLM to ChatAssistant
```python
from anthropic import Anthropic

class ChatAssistant:
    def __init__(self, state_manager, llm_client):
        self.state_manager = state_manager
        self.llm = llm_client  # Anthropic client
```

### Step 2: Convert Helpers to Tools
```python
TOOLS = [
    {
        "name": "return_kanban",
        "description": "Return kanban board for visual task tracking",
        "input_schema": {
            "type": "object",
            "properties": {
                "columns": {
                    "type": "array",
                    "description": "Board columns with tasks"
                }
            }
        }
    },
    # ... auto-generate rest
]
```

### Step 3: Replace Hardcoded Logic
```python
# Before (not DRY)
if 'markdown' in query:
    return self._get_markdown_summary()

# After (DRY)
response = await self.llm.chat(
    messages=[...],
    tools=TOOLS,
    tool_choice='required'
)
return self._execute_tool(response.tool_calls[0])
```

## Integration with Schema Architecture

The schema registry we built enables this:

```python
# Auto-generate LLM tool definitions from schemas
from core.ui.schema_validator import schema_registry

def generate_llm_tools():
    tools = []
    for section_type in schema_registry.list_sections():
        schema = schema_registry.get_schema(section_type)
        tools.append({
            'name': f'return_{section_type.replace("-", "_")}',
            'description': schema['description'],
            'input_schema': schema['schema'],  # JSON schema as tool input
        })
    return tools

# LLM now has tools for ALL sections automatically!
LLM_TOOLS = generate_llm_tools()
```

## Summary

**Instead of:**
- 70+ hardcoded methods (`_get_kanban`, `_get_chart`, ...)
- Manual intent detection (`if 'markdown' in query`)

**We have:**
- 1 generic LLM-driven generator
- LLM chooses section type based on query
- Auto-validated via schema registry
- Scales to 1000+ section types automatically

This is **truly DRY** - the LLM does the work, we just provide tools!
