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
