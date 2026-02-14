# Search → Summarize → PDF → Telegram (Generic Pipeline)

Demonstrates generic Source → Processor → Sink pipeline pattern using declarative configuration.

## Description

This skill uses the generic pipeline framework (`core/registry/pipeline_skill.py`) to define workflows declaratively as arrays of source/processor/sink steps.


## Type
composite

## Base Skills
- web-search
- claude-cli-llm
- document-converter
- telegram-sender

## Execution
sequential


## Capabilities
- research
- document
- communicate

## Composite
Combines: web-search, claude-cli-llm, document-converter, telegram-sender
Use when: User wants to research a topic, create a PDF summary, and send it via Telegram

## Architecture

Uses `PipelineSkill` class which supports:
- **Declarative configuration**: Pipeline defined as array of steps
- **Template variables**: `{{variable}}` and `{{step.field}}` syntax
- **Type safety**: Explicit source/processor/sink types
- **Data flow**: Automatic result passing between steps

## Pipeline Definition

```python
pipeline = [
    {
        "type": "source",
        "skill": "web-search",
        "tool": "search_web_tool",
        "params": {"query": "{{topic}}"}
    },
    {
        "type": "processor",
        "skill": "claude-cli-llm",
        "tool": "summarize_text_tool",
        "params": {"content": "{{source.results}}"}
    },
    {
        "type": "processor",
        "skill": "document-converter",
        "tool": "convert_to_pdf_tool",
        "params": {"input_file": "{{processor.summary_path}}"}
    },
    {
        "type": "sink",
        "skill": "telegram-sender",
        "tool": "send_telegram_file_tool",
        "params": {"file_path": "{{processor.pdf_path}}"}
    }
]
```

## Benefits

1. **Declarative**: Define workflows as data structures
2. **Reusable**: Easy to create new pipelines by mixing steps
3. **Type-safe**: Clear source/processor/sink separation
4. **Template variables**: Automatic data flow between steps

## Usage

Same as regular composite skill:

```python
from skills.search_summarize_pdf_telegram_v2.tools import search_summarize_pdf_telegram_v2_tool

result = await search_summarize_pdf_telegram_v2_tool({
    'topic': 'multi agent systems',
    'max_results': 5,
    'send_telegram': True
})
```

## Triggers
- "search summarize pdf telegram v2"
- "create pdf"
- "generate pdf"
- "convert to pdf"
- "pdf"
- "send to telegram"
- "telegram message"
- "notify via telegram"

## Category
document-creation
