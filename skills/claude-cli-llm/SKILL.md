# Claude CLI LLM Skill

Text generation and summarization using Claude CLI.

## Description

This skill provides LLM capabilities using Claude CLI for text summarization and generation.
Uses the `claude` command-line tool (no API keys required if CLI is authenticated).

## Usage

```python
from skills.claude_cli_llm.tools import summarize_text_tool

result = await summarize_text_tool({
    'content': 'Long text to summarize...',
    'prompt': 'Summarize the following in 3 bullet points:'
})
```

## Parameters

### summarize_text_tool
- `content` (str, required): Text content to summarize
- `prompt` (str, optional): Custom prompt (default: "Summarize the following:")
- `model` (str, optional): Claude model - 'sonnet', 'opus', 'haiku' (default: 'sonnet')
- `max_tokens` (int, optional): Maximum tokens in response

### generate_text_tool
- `prompt` (str, required): Text generation prompt
- `model` (str, optional): Claude model (default: 'sonnet')
- `max_tokens` (int, optional): Maximum tokens

## Requirements

- Claude CLI installed and authenticated
- Run `claude auth login` if not authenticated

## Architecture

Uses Claude CLI subprocess calls for simplicity.
Can be enhanced to use DSPy wrapper if needed.
