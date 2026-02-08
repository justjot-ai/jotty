# Notion Knowledge Pipeline Composite Skill

Complete knowledge management workflow: capture → research → implement.

## Description

This composite skill combines:
1. **Knowledge Capture** (Source): notion-knowledge-capture
2. **Research Documentation** (Processor): notion-research-documentation
3. **Implementation Planning** (Sink): notion-spec-to-implementation


## Type
composite

## Base Skills
- web-search
- claude-cli-llm
- notion

## Execution
sequential

## Usage

```python
from skills.notion_knowledge_pipeline.tools import notion_knowledge_pipeline_tool

# Full workflow
result = await notion_knowledge_pipeline_tool({
    'workflow_type': 'full',
    'knowledge_content': 'Key insight from meeting',
    'research_topic': 'AI agents',
    'spec_page_id': 'notion-page-id'
})

# Just capture
result = await notion_knowledge_pipeline_tool({
    'workflow_type': 'capture',
    'knowledge_content': 'Important concept',
    'knowledge_title': 'New Concept'
})
```

## Parameters

- `workflow_type` (str, required): 'capture', 'research', 'implement', or 'full'
- `knowledge_content` (str, optional): Content to capture
- `knowledge_title` (str, optional): Title for knowledge entry
- `content_type` (str, optional): 'concept', 'meeting', 'idea', 'note' (default: 'concept')
- `research_topic` (str, optional): Topic for research
- `research_output_format` (str, optional): 'brief', 'detailed', 'comprehensive' (default: 'detailed')
- `spec_page_id` (str, optional): Notion page ID for spec
- `plan_type` (str, optional): 'quick', 'detailed', 'comprehensive' (default: 'quick')
- `capture_knowledge` (bool, optional): Capture knowledge
- `research_and_document` (bool, optional): Research and document
- `create_implementation_plan` (bool, optional): Create plan

## Architecture

Source → Processor → Sink pattern:
- **Source**: Knowledge capture
- **Processor**: Research documentation
- **Sink**: Implementation planning

No code duplication - reuses existing skills.
