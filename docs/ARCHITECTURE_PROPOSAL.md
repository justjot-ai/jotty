# Architecture Proposal: Source → Processor → Sink Pattern

## Current State Analysis

### Existing Skills:
1. **Sources** (data generators):
   - `web-search`: Search web via DuckDuckGo
   - `last30days-claude-cli`: Research topics from last 30 days
   - `v2v-trending-search`: Search V2V.ai trending topics
   - `youtube-downloader`: Download YouTube transcripts
   - `arxiv-downloader`: Download arXiv papers

2. **Processors** (data transformers):
   - `document-converter`: Convert between formats (PDF, EPUB, DOCX, HTML, MD)
   - `content-repurposer`: Repurpose content for platforms (simple text-based)
   - `text-chunker`: Chunk text into smaller pieces
   - `text-utils`: Text manipulation utilities

3. **Sinks** (data outputs):
   - `telegram-sender`: Send messages/files to Telegram
   - `remarkable-sender`: Upload PDFs to reMarkable tablet
   - `file-operations`: Save files locally

### Missing Component:
- **LLM Processor**: No skill exists for Claude CLI LLM summarization/generation
- Need to create: `claude-cli-llm` skill for text summarization/generation

## Proposed Architecture

### Pattern: Source → Processor → Sink

```
┌─────────┐     ┌─────────────┐     ┌─────────┐
│ Source  │ --> │  Processor  │ --> │  Sink   │
└─────────┘     └─────────────┘     └─────────┘
```

### Generic Composite Skill Structure

```python
{
    "name": "search-summarize-pdf-telegram",
    "description": "Search → Summarize → PDF → Telegram",
    "pipeline": [
        {
            "type": "source",
            "skill": "web-search",
            "tool": "search_web_tool",
            "params": {"query": "{{topic}}", "max_results": 10}
        },
        {
            "type": "processor",
            "skill": "claude-cli-llm",
            "tool": "summarize_text_tool",
            "params": {
                "content": "{{source.results}}",
                "prompt": "Summarize these search results"
            }
        },
        {
            "type": "processor",
            "skill": "document-converter",
            "tool": "convert_to_pdf_tool",
            "params": {
                "input_file": "{{processor.summary_path}}",
                "output_file": "{{output_dir}}/summary.pdf"
            }
        },
        {
            "type": "sink",
            "skill": "telegram-sender",
            "tool": "send_telegram_file_tool",
            "params": {
                "file_path": "{{processor.pdf_path}}",
                "chat_id": "{{telegram_chat_id}}"
            }
        }
    ]
}
```

### Benefits:
1. **DRY**: Reuse existing skills, no duplication
2. **Composable**: Mix and match sources/processors/sinks
3. **Declarative**: Define workflows as data structures
4. **Type-safe**: Clear separation of concerns
5. **Extensible**: Easy to add new sources/processors/sinks

## Implementation Plan

### Phase 1: Create LLM Processor Skill
- Create `claude-cli-llm` skill
- Tool: `summarize_text_tool` (uses Claude CLI API)
- Tool: `generate_text_tool` (general text generation)

### Phase 2: Enhance Composite Skill Framework
- Add pipeline-based execution mode
- Support template variables ({{source.field}})
- Auto-detect source/processor/sink types
- Better error handling and data flow

### Phase 3: Create Generic Pipeline Builder
- Function: `create_pipeline_skill(pipeline_config)`
- Auto-generate composite skill from pipeline config
- Support multiple processors (chain)
- Support multiple sinks (fan-out)

## Example Use Cases

### 1. Search → Summarize → PDF → Telegram
```python
pipeline = [
    {"type": "source", "skill": "web-search", "tool": "search_web_tool"},
    {"type": "processor", "skill": "claude-cli-llm", "tool": "summarize_text_tool"},
    {"type": "processor", "skill": "document-converter", "tool": "convert_to_pdf_tool"},
    {"type": "sink", "skill": "telegram-sender", "tool": "send_telegram_file_tool"}
]
```

### 2. YouTube → Summarize → PDF → reMarkable
```python
pipeline = [
    {"type": "source", "skill": "youtube-downloader", "tool": "download_transcript_tool"},
    {"type": "processor", "skill": "claude-cli-llm", "tool": "summarize_text_tool"},
    {"type": "processor", "skill": "document-converter", "tool": "convert_to_pdf_tool"},
    {"type": "sink", "skill": "remarkable-sender", "tool": "send_to_remarkable_tool"}
]
```

### 3. Multiple Sources → Combine → Summarize → PDF
```python
pipeline = [
    {"type": "source", "skill": "web-search", "tool": "search_web_tool", "output_key": "web"},
    {"type": "source", "skill": "v2v-trending-search", "tool": "search_v2v_trending_tool", "output_key": "v2v"},
    {"type": "processor", "skill": "text-utils", "tool": "combine_texts_tool", "depends_on": ["web", "v2v"]},
    {"type": "processor", "skill": "claude-cli-llm", "tool": "summarize_text_tool"},
    {"type": "processor", "skill": "document-converter", "tool": "convert_to_pdf_tool"}
]
```

## Questions to Resolve

1. **Claude CLI API**: How to call Claude CLI? Is there an existing integration?
   - Check: `anthropic` Python package
   - Check: Environment variables for API keys
   - Check: CLI command `claude` if installed

2. **Template Variables**: How to handle `{{source.field}}` references?
   - Option A: String interpolation with result dict
   - Option B: Function-based params (current approach)
   - Option C: JSONPath/XPath-like queries

3. **Data Flow**: How to pass data between steps?
   - Current: Merge results into params
   - Proposed: Explicit data flow with named outputs
   - Need: Better type checking and validation

4. **Error Handling**: What happens if a step fails?
   - Current: Stop on required step failure
   - Proposed: Retry logic, fallback steps, partial success

## Recommendation

**Start Simple**: 
1. Create `claude-cli-llm` skill first
2. Create specific composite skill: `search-summarize-pdf-telegram`
3. Then generalize to pipeline pattern if needed

**Why**: 
- Validate LLM integration works
- Test the workflow end-to-end
- Then abstract to generic pattern

This follows YAGNI (You Aren't Gonna Need It) - build what's needed now, generalize later.
