# Search to JustJot Idea Workflow

## Overview

Multi-agent skill that automatically researches any topic and creates a structured idea on JustJot.ai.

## Workflow

```
┌─────────────┐     Search      ┌──────────────┐     Summarize    ┌─────────────┐     Create      ┌──────────────┐
│   Topic     │ ──────────────> │  Web Search  │ ──────────────> │ Claude CLI  │ ─────────────> │ JustJot Idea │
│  (Input)    │                 │   (Source)   │                 │ (Processor) │                │    (Sink)    │
└─────────────┘                 └──────────────┘                 └─────────────┘                └──────────────┘
```

## Steps

1. **Source**: Web search for topic information
2. **Processor**: Summarize and structure content using Claude CLI LLM
3. **Sink**: Create idea on JustJot.ai via MCP client (direct MongoDB)

## Usage

```python
from core.registry.skills_registry import get_skills_registry

registry = get_skills_registry()
registry.init()

skill = registry.get_skill('search-to-justjot-idea')
tool = skill.tools.get('search_and_create_idea_tool')

result = await tool({
    'topic': 'multi-agent systems',
    'title': 'Multi-Agent Systems: Current Trends',
    'tags': ['ai', 'multi-agent', 'research'],
    'max_results': 10,
    'summary_length': 'comprehensive',
    'use_mcp_client': True
})
```

## Parameters

- `topic` (str, **required**): Search topic/query
- `title` (str, optional): Idea title (auto-generated if not provided)
- `description` (str, optional): Idea description
- `tags` (list, optional): Tags for the idea
- `max_results` (int, optional): Maximum search results (default: 10)
- `summary_length` (str, optional): 'brief', 'comprehensive', 'detailed' (default: 'comprehensive')
- `use_mcp_client` (bool, optional): Use MCP client instead of HTTP API (default: True)

## Features

- ✅ **Multi-agent workflow**: Combines search, summarization, and idea creation
- ✅ **MCP client**: Uses direct MongoDB access (faster than HTTP API)
- ✅ **Automatic fallback**: Falls back to HTTP API if MCP fails
- ✅ **Structured sections**: Creates idea with multiple sections from summary
- ✅ **Sources included**: Adds sources section with search result URLs
- ✅ **Async/sync handling**: Works with both async and sync tools

## Example Output

```python
{
    'success': True,
    'idea_id': '...',
    'title': 'Multi-Agent Systems: Current Trends and Applications',
    'sections': 6,
    'method': 'mcp-client',
    'message': 'Idea created successfully with 6 sections'
}
```

## Test Results

**Test**: Multi-agent systems research

**Result**:
- ✅ Successfully searched web (5 results)
- ✅ Generated comprehensive summary using Claude CLI
- ✅ Created idea on JustJot.ai via MCP client
- ✅ Created 6 sections:
  1. Overview/Introduction
  2. Key Concepts and Principles
  3. Current Applications and Use Cases
  4. Challenges and Limitations
  5. Future Directions
  6. Sources (with search result URLs)

**Method**: MCP client (direct MongoDB access)

## Dependencies

- `web-search` skill - Web search functionality
- `claude-cli-llm` skill - LLM summarization
- `mcp-justjot-mcp-client` skill - MCP client for JustJot.ai
- `mcp-justjot` skill - HTTP API fallback

## Configuration

**MongoDB URI** (for MCP client):
```bash
export MONGODB_URI="mongodb://planmyinvesting:aRpOVx2HYl6jS9LO@planmyinvesting.com:27017/planmyinvesting"
```

**MCP Server**: Already running on cmd.dev (2 instances)

## Benefits

1. **Automated Research**: No manual information gathering needed
2. **Intelligent Summarization**: Claude CLI provides structured summaries
3. **Direct Database Access**: MCP client is faster than HTTP API
4. **Structured Ideas**: Creates well-organized ideas with sections
5. **Source Attribution**: Includes sources for verification

## Next Steps

- Add support for multiple topics (parallel processing)
- Add PDF generation option
- Add Telegram/reMarkable delivery options
- Add custom section templates
