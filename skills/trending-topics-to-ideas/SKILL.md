---
name: discovering-trending-topics-to-ideas
description: "This composite skill: 1. **Source**: Get trending topics (from web search, V2V, or Reddit) 2. **For each topic** (parallel processing): - Gather more details via HTTP/web search - Synthesize information via Claude CLI LLM - Create JustJot idea with synthesized content as sections. Use when the user wants to create."
---

# Trending Topics → Ideas Composite Skill

Process trending topics: gather details → synthesize → create ideas with sections.

## Description

This composite skill:
1. **Source**: Get trending topics (from web search, V2V, or Reddit)
2. **For each topic** (parallel processing):
   - Gather more details via HTTP/web search
   - Synthesize information via Claude CLI LLM
   - Create JustJot idea with synthesized content as sections


## Type
composite

## Base Skills
- web-search
- mcp-justjot

## Execution
sequential


## Capabilities
- research

## Usage

```python
from skills.trending_topics_to_ideas.tools import trending_topics_to_ideas_tool

result = await trending_topics_to_ideas_tool({
    'source': 'reddit',  # 'reddit', 'v2v', or 'web'
    'query': 'AI trends',
    'max_topics': 5,
    'details_per_topic': 5,
    'create_ideas': True
})
```

## Parameters

- `source` (str, optional): Source for trending topics - 'reddit', 'v2v', 'web' (default: 'reddit')
- `query` (str, optional): Search query for trending topics
- `max_topics` (int, optional): Max topics to process (default: 5)
- `details_per_topic` (int, optional): Web search results per topic (default: 5)
- `create_ideas` (bool, optional): Create JustJot ideas (default: True)
- `synthesize_prompt` (str, optional): Custom synthesis prompt
- `use_mcp_client` (bool, optional): Use MCP client instead of HTTP API (default: True)

## Architecture

Uses parallel processing pattern:
- Source: Get trending topics (list)
- For each topic (parallel):
  - Processor: Gather details (web-search)
  - Processor: Synthesize (claude-cli-llm)
  - Sink: Create idea (mcp-justjot)

DRY: Reuses existing skills, no duplication.

## Workflow

```
Task Progress:
- [ ] Step 1: Get trending topics
- [ ] Step 2: Research each topic
- [ ] Step 3: Synthesize content
- [ ] Step 4: Create JustJot ideas
```

**Step 1: Get trending topics**
Fetch trending topics from Reddit, V2V, or web search.

**Step 2: Research each topic**
Gather additional details for each topic via web search.

**Step 3: Synthesize content**
Use Claude LLM to create structured summaries for each topic.

**Step 4: Create JustJot ideas**
Save each synthesized topic as a JustJot idea with sections.

## Triggers
- "trending topics to ideas"
- "create"

## Category
workflow-automation
