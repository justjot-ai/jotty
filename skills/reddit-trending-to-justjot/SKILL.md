# Reddit Trending → Markdown → JustJot Idea Pipeline

Search Reddit for trending topics, format as markdown, and create a JustJot idea.

## Description

This composite skill combines:
1. **web-search**: Search Reddit for trending topics
2. **text-utils**: Format results as markdown
3. **mcp-justjot**: Create JustJot idea with the content

## Usage

```python
from skills.reddit_trending_to_justjot.tools import reddit_trending_to_justjot_tool

result = await reddit_trending_to_justjot_tool({
    'topic': 'multi agent systems',
    'title': 'Multi-Agent Systems: Reddit Trends',
    'max_results': 10
})
```

## Parameters

- `topic` (str, required): Topic to search on Reddit
- `title` (str, optional): Idea title (default: auto-generated)
- `max_results` (int, optional): Max Reddit results (default: 10)
- `description` (str, optional): Idea description
- `tags` (list, optional): Tags for the idea

## Architecture

Uses generic pipeline pattern:
- Source: web-search (Reddit filter)
- Processor: Format as markdown
- Sink: mcp-justjot (create_idea_tool)

DRY: Reuses existing skills, no duplication.
