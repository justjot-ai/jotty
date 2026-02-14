# Search to JustJot Idea Skill

## Description

Multi-agent skill that searches for information on any topic, summarizes it using Claude CLI LLM, and creates a structured idea on JustJot.ai using MCP client (direct MongoDB access).


## Type
composite

## Base Skills
- web-search
- mcp-justjot

## Execution
sequential


## Capabilities
- research

## Workflow

1. **Source**: Search web for topic information
2. **Processor**: Summarize and structure content using Claude CLI LLM
3. **Sink**: Create idea on JustJot.ai via MCP client

## Usage

```python
from skills.search_to_justjot_idea.tools import search_and_create_idea_tool

result = await search_and_create_idea_tool({
    'topic': 'multi-agent systems',
    'title': 'Multi-Agent Systems: Current Trends and Applications',
    'tags': ['ai', 'multi-agent', 'research'],
    'description': 'Research on multi-agent systems architecture and applications'
})
```

## Parameters

- `topic` (str, required): Search topic/query
- `title` (str, optional): Idea title (auto-generated if not provided)
- `description` (str, optional): Idea description
- `tags` (list, optional): Tags for the idea
- `max_results` (int, optional): Maximum search results (default: 10)
- `summary_length` (str, optional): Summary length - 'brief', 'comprehensive', 'detailed' (default: 'comprehensive')

## Features

- ✅ Uses MCP client for direct MongoDB access (faster)
- ✅ Web search for information gathering
- ✅ Claude CLI LLM for intelligent summarization
- ✅ Structured idea creation with sections
- ✅ Automatic fallback to HTTP API if MCP fails

## Dependencies

- `web-search` skill
- `claude-cli-llm` skill
- `mcp-justjot` skill

## Example

```python
result = await search_and_create_idea_tool({
    'topic': 'multi-agent systems',
    'title': 'Multi-Agent Systems Research',
    'tags': ['ai', 'research'],
    'max_results': 15,
    'summary_length': 'comprehensive'
})

# Returns:
# {
#     'success': True,
#     'idea_id': '...',
#     'title': 'Multi-Agent Systems Research',
#     'sections': 3,
#     'message': 'Idea created successfully'
# }
```

## Triggers
- "search to justjot idea"
- "search for"
- "look up"
- "find information"
- "create"

## Category
data-retrieval
