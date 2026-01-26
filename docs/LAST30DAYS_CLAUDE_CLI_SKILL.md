# last30days-claude-cli Skill

A Jotty-native skill that researches topics from the last 30 days using Claude CLI LM (no API keys required).

## Overview

This skill replaces the original `last30days` skill's API key requirements with Claude CLI LM's built-in WebSearch tool. It searches Reddit, X/Twitter, and the web without needing OpenAI or xAI API keys.

## Installation

The skill is automatically available in Jotty's skills registry:

```python
from core.registry.skills_registry import get_skills_registry

registry = get_skills_registry()
registry.init()

skill = registry.get_skill('last30days-claude-cli')
```

## Usage

```python
# Get the tool
tool = skill.tools['last30days_claude_cli_tool']

# Execute research
result = await tool({
    'topic': 'AI trends 2026',
    'quick': True,  # Faster, fewer sources
    'emit': 'compact'  # or 'json' or 'md'
})
```

## Parameters

- `topic` (required): Topic to research
- `tool` (optional): Target tool (e.g., "ChatGPT", "Midjourney")
- `quick` (optional): Faster research, fewer sources (default: False)
- `deep` (optional): Comprehensive research, more sources (default: False)
- `emit` (optional): Output format - "compact", "json", or "md" (default: "compact")

## How It Works

1. **Uses Claude CLI LM**: Calls `claude --tools WebSearch` to search the web
2. **Searches Multiple Sources**: Looks for Reddit, X/Twitter, and web content
3. **Structured Output**: Returns results in your chosen format
4. **No API Keys**: Uses Claude CLI's built-in capabilities

## Example

```python
import asyncio
from core.registry.skills_registry import get_skills_registry

async def research_topic():
    registry = get_skills_registry()
    registry.init()
    
    skill = registry.get_skill('last30days-claude-cli')
    tool = skill.tools['last30days_claude_cli_tool']
    
    result = await tool({
        'topic': 'Claude Code skills',
        'deep': True,
        'emit': 'json'
    })
    
    if result['success']:
        print(result['output'])
    else:
        print(f"Error: {result['error']}")

asyncio.run(research_topic())
```

## Output Formats

### Compact (default)
```
Research Results: "AI trends"
Date Range: 2025-12-27 to 2026-01-26
Mode: Claude CLI WebSearch

Sources found: 5 Reddit, 3 X, 8 Web

Key Patterns:
  • Pattern 1
  • Pattern 2

Top Recommendations:
  • Recommendation 1
  • Recommendation 2
```

### JSON
```json
{
  "topic": "AI trends",
  "date_range": "2025-12-27 to 2026-01-26",
  "reddit": [...],
  "x": [...],
  "web": [...],
  "patterns": [...],
  "recommendations": [...]
}
```

### Markdown
Full markdown report with sections for Reddit, X, Web, patterns, and recommendations.

## Requirements

- Claude CLI installed (`npm install -g @anthropic-ai/claude-code`)
- Claude CLI configured with credentials
- WebSearch tool enabled in Claude CLI

## Differences from Original last30days

| Feature | Original | Claude CLI Version |
|---------|----------|-------------------|
| API Keys | Required (OpenAI/xAI) | Not required |
| Reddit Search | OpenAI Responses API | Claude CLI WebSearch |
| X/Twitter Search | xAI API | Claude CLI WebSearch |
| Web Search | Fallback | Primary method |
| Speed | Fast (API calls) | Slower (CLI calls) |
| Cost | API costs | Free (uses Claude CLI) |

## Troubleshooting

**Issue**: "I need permission to use the WebSearch tool"
- **Solution**: Ensure Claude CLI has WebSearch tool enabled in its configuration

**Issue**: Skill not found
- **Solution**: Make sure the skill is in `~/jotty/skills/last30days-claude-cli/` or restart the registry

**Issue**: Timeout errors
- **Solution**: Research can take 1-3 minutes. Use `quick: True` for faster results

## Integration with Agents

Agents can use this skill directly:

```python
from core.registry.skills_registry import get_skills_registry

class ResearchAgent:
    def __init__(self):
        self.registry = get_skills_registry()
        self.registry.init()
    
    async def research(self, topic: str):
        skill = self.registry.get_skill('last30days-claude-cli')
        tool = skill.tools['last30days_claude_cli_tool']
        return await tool({'topic': topic})
```

## Future Enhancements

- Better JSON parsing from Claude CLI responses
- Caching of research results
- Parallel search execution
- More sophisticated pattern extraction
