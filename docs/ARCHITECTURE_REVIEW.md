# Architecture Review: Source → Processor → Sink Pattern

## Findings

### ✅ Existing Infrastructure:
1. **Claude CLI Integration**: `core/foundation/claude_cli_lm.py`
   - Uses `claude` CLI command
   - Supports JSON schema for structured output
   - Can be used for summarization/generation

2. **Composite Skill Framework**: `core/registry/composite_skill.py`
   - Supports sequential, parallel, mixed execution
   - Uses function-based params: `lambda p, r: {...}`
   - Already handles data flow between steps

3. **Existing Skills**:
   - **Sources**: `web-search`, `last30days-claude-cli`, `v2v-trending-search`
   - **Processors**: `document-converter`, `text-utils`
   - **Sinks**: `telegram-sender`, `remarkable-sender`

### ❌ Missing:
- **LLM Processor Skill**: No skill wrapper for Claude CLI LLM
- **Generic Pipeline Builder**: Current composite framework works but could be more declarative

## Proposed Implementation

### Option A: Simple Approach (Recommended)
Create specific composite skill first, then generalize if needed.

**Steps:**
1. Create `claude-cli-llm` skill with `summarize_text_tool`
2. Create `search-summarize-pdf-telegram` composite skill
3. Use existing composite framework (no changes needed)

**Pros:**
- Fast to implement
- Validates workflow end-to-end
- Follows YAGNI principle
- Can generalize later if needed

**Cons:**
- Less generic initially
- May need refactoring later

### Option B: Generic Pipeline Pattern
Enhance composite framework to support declarative pipeline configs.

**Steps:**
1. Create `claude-cli-llm` skill
2. Enhance `composite_skill.py` with pipeline mode
3. Add template variable support (`{{source.field}}`)
4. Create generic pipeline builder

**Pros:**
- More flexible and reusable
- Declarative configuration
- Better for complex workflows

**Cons:**
- More complex to implement
- May be overkill for current needs
- Requires more testing

## Recommendation: Option A

**Rationale:**
1. Current composite framework already supports the pattern via function params
2. Function params (`lambda p, r: {...}`) are flexible enough
3. Can always refactor to Option B later if needed
4. Faster to deliver value

## Implementation Plan

### Phase 1: Create LLM Skill
```python
# skills/claude-cli-llm/tools.py
def summarize_text_tool(params):
    """Summarize text using Claude CLI"""
    content = params.get('content')
    prompt = params.get('prompt', 'Summarize the following:')
    
    # Use ClaudeCLILM or direct subprocess call
    # Return: {'success': True, 'summary': '...'}
```

### Phase 2: Create Composite Skill
```python
# skills/search-summarize-pdf-telegram/tools.py
async def search_summarize_pdf_telegram_tool(params):
    # Step 1: Source - Search
    search_result = await search_web_tool({'query': params['topic']})
    
    # Step 2: Processor - Summarize
    content = format_search_results(search_result['results'])
    summary_result = await summarize_text_tool({
        'content': content,
        'prompt': f"Summarize these search results about {params['topic']}"
    })
    
    # Step 3: Processor - PDF
    pdf_result = await convert_to_pdf_tool({
        'input_file': create_temp_md(summary_result['summary']),
        'output_file': f"{output_dir}/summary.pdf"
    })
    
    # Step 4: Sink - Telegram
    telegram_result = await send_telegram_file_tool({
        'file_path': pdf_result['output_path']
    })
    
    return {'success': True, ...}
```

### Phase 3: Generalize (Future)
If we need more workflows, create:
- `create_pipeline_skill(pipeline_config)` helper
- Template variable system
- Better error handling

## Code Structure

```
skills/
├── claude-cli-llm/
│   ├── SKILL.md
│   └── tools.py  # summarize_text_tool, generate_text_tool
└── search-summarize-pdf-telegram/
    ├── SKILL.md
    └── tools.py  # Uses existing composite pattern
```

## Questions Resolved

1. **Claude CLI**: ✅ Found in `core/foundation/claude_cli_lm.py`
2. **Data Flow**: ✅ Current function params work (`lambda p, r: {...}`)
3. **Error Handling**: ✅ Existing framework handles failures
4. **Template Variables**: ✅ Function params provide flexibility

## Next Steps

1. ✅ Review architecture (this document)
2. Create `claude-cli-llm` skill
3. Create `search-summarize-pdf-telegram` composite skill
4. Test end-to-end
5. Document usage
