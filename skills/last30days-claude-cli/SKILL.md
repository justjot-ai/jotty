# last30days-claude-cli

Research any topic across Reddit + X + Web from the last 30 days using Jotty's web search tools (no API keys required).

## Description

This skill researches topics from the last 30 days using Jotty's built-in web-search skill (DuckDuckGo). Unlike the original last30days skill, this version doesn't require OpenAI or xAI API keys - it uses Jotty's web search tools directly.

## Features

- ✅ No API keys required - uses Claude CLI LM
- ✅ Searches Reddit, X/Twitter, and Web
- ✅ Returns structured research results
- ✅ Supports quick/deep research modes
- ✅ Generates copy-paste-ready prompts

## Usage

```python
from core.registry.skills_registry import get_skills_registry

registry = get_skills_registry()
registry.init()
skill = registry.get_skill('last30days-claude-cli')
tool = skill.tools['last30days_claude_cli']

result = await tool({
    'topic': 'AI trends 2026',
    'deep': True
})
```

## Parameters

- `topic` (required): Topic to research
- `tool` (optional): Target tool (e.g., "ChatGPT", "Midjourney")
- `quick` (optional): Faster research, fewer sources
- `deep` (optional): Comprehensive research
- `emit` (optional): Output format: "compact", "json", "md"

## Version

1.0.0

## Author

Jotty Framework

## Category

research
