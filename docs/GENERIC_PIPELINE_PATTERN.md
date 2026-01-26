# Generic Pipeline Pattern: Source → Processor → Sink

## Overview

The generic pipeline framework allows you to define composite skills declaratively as arrays of source/processor/sink steps. This follows the recommended architecture pattern.

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│   Source    │ --> │    Processor     │ --> │    Sink     │
│ (Data Gen)  │     │ (Transform)      │     │ (Output)    │
└─────────────┘     └──────────────────┘     └─────────────┘
```

## Implementation Status

✅ **Implemented**: Generic `PipelineSkill` framework
✅ **Implemented**: Template variable support (`{{variable}}`, `{{step.field}}`)
✅ **Implemented**: Type-safe step definitions (source/processor/sink)
✅ **Implemented**: Example: `search-summarize-pdf-telegram-v2`

## Usage

### Basic Pipeline Definition

```python
from core.registry.pipeline_skill import create_pipeline_skill, StepType

pipeline = [
    {
        "type": StepType.SOURCE.value,  # or "source"
        "skill": "web-search",
        "tool": "search_web_tool",
        "params": {
            "query": "{{topic}}",
            "max_results": "{{max_results}}"
        },
        "output_key": "source"
    },
    {
        "type": StepType.PROCESSOR.value,  # or "processor"
        "skill": "claude-cli-llm",
        "tool": "summarize_text_tool",
        "params": {
            "content": "{{source.results}}",
            "prompt": "Summarize these results"
        },
        "output_key": "processor"
    },
    {
        "type": StepType.SINK.value,  # or "sink"
        "skill": "telegram-sender",
        "tool": "send_telegram_file_tool",
        "params": {
            "file_path": "{{processor.pdf_path}}"
        },
        "output_key": "sink"
    }
]

skill = create_pipeline_skill(
    name='my-pipeline',
    description='My custom pipeline',
    pipeline=pipeline
)
```

### Template Variables

**From initial params:**
- `{{topic}}` → `params['topic']`
- `{{max_results}}` → `params['max_results']`

**From previous step results:**
- `{{source.results}}` → `results['source']['results']`
- `{{processor.pdf_path}}` → `results['processor']['pdf_path']`
- `{{processor.field.nested}}` → `results['processor']['field']['nested']`

**Function-based params (advanced):**
```python
"params": lambda p, r: {
    "content": format_results(r['source']['results']),
    "prompt": f"Summarize: {p['topic']}"
}
```

### Step Types

1. **SOURCE**: Data retrieval/generation
   - Examples: `web-search`, `last30days-claude-cli`, `youtube-downloader`
   - Output: Raw data

2. **PROCESSOR**: Data transformation
   - Examples: `claude-cli-llm`, `document-converter`, `text-utils`
   - Input: Previous step output
   - Output: Transformed data

3. **SINK**: Data output/delivery
   - Examples: `telegram-sender`, `remarkable-sender`, `file-operations`
   - Input: Final processed data
   - Output: Delivery confirmation

## Example: Search → Summarize → PDF → Telegram

See `skills/search-summarize-pdf-telegram-v2/` for a complete example.

## Benefits

1. **Declarative**: Define workflows as data structures
2. **Reusable**: Mix and match sources/processors/sinks
3. **Type-safe**: Clear separation of concerns
4. **DRY**: No code duplication
5. **Extensible**: Easy to add new steps

## Comparison: Specific vs Generic

### Option A: Specific Composite Skill (Original)
```python
async def search_summarize_pdf_telegram_tool(params):
    # Hardcoded workflow
    search_result = await search_web_tool(...)
    summary = await summarize_text_tool(...)
    pdf = await convert_to_pdf_tool(...)
    telegram = await send_telegram_file_tool(...)
```

**Pros**: Simple, fast to implement
**Cons**: Not reusable, hardcoded workflow

### Option B: Generic Pipeline (Recommended)
```python
pipeline = [
    {"type": "source", "skill": "web-search", ...},
    {"type": "processor", "skill": "claude-cli-llm", ...},
    {"type": "processor", "skill": "document-converter", ...},
    {"type": "sink", "skill": "telegram-sender", ...}
]
```

**Pros**: Declarative, reusable, extensible
**Cons**: Slightly more complex

## Migration Path

Both approaches work! You can:
1. Use specific composite skills for simple workflows
2. Use generic pipeline for complex/reusable workflows
3. Migrate specific skills to generic pipelines over time

## Next Steps

1. ✅ Generic pipeline framework implemented
2. ✅ Template variable support
3. ✅ Example pipeline skill created
4. 🔄 Create more example pipelines
5. 🔄 Add pipeline validation and error handling
6. 🔄 Add pipeline visualization/debugging tools
