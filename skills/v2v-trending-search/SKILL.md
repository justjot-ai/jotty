---
name: searching-v2v-trending
description: "This skill searches V2V.ai for trending topics and generates research reports that can be converted to PDFs and sent to Telegram/reMarkable. Use when the user wants to search for, look up, find information."
---

# V2V Trending Search Skill

Search for trending topics on V2V.ai and generate reports.

## Description

This skill searches V2V.ai for trending topics and generates research reports that can be converted to PDFs and sent to Telegram/reMarkable.


## Type
composite

## Base Skills
- voice
- web-search

## Execution
sequential


## Capabilities
- media
- research

## Features

- Search V2V.ai for trending topics
- Generate markdown reports
- Integrates with PDF generation
- Can send to Telegram and reMarkable

## Usage

```python
from skills.v2v_trending_search.tools import search_v2v_trending_tool

result = await search_v2v_trending_tool({
    'query': 'multi agent systems',
    'format': 'markdown'
})
```

## Parameters

- `query` (str, optional): Search query (default: searches for trending topics)
- `format` (str, optional): Output format - 'markdown', 'json' (default: 'markdown')
- `max_results` (int, optional): Maximum results (default: 10)

## Workflow

```
Task Progress:
- [ ] Step 1: Search V2V.ai
- [ ] Step 2: Generate report
```

**Step 1: Search V2V.ai**
Query V2V.ai for trending topics matching the search query.

**Step 2: Generate report**
Format the search results into a structured markdown report.

## Triggers
- "v2v trending search"
- "search for"
- "look up"
- "find information"
- "generate"

## Category
data-retrieval
