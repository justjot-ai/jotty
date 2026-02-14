---
name: search-summarize-pdf-telegram
description: "This composite skill combines: 1. **web-search**: Search the web for any topic 2. **claude-cli-llm**: Summarize search results using Claude 3. **document-converter**: Convert summary to PDF 4. **telegram-sender**: Send PDF to Telegram. Use when the user wants to create pdf, generate pdf, convert to pdf."
---

# Search → Summarize → PDF → Telegram Composite Skill

Search any topic, summarize with Claude CLI LLM, generate PDF, and send to Telegram.

## Description

This composite skill combines:
1. **web-search**: Search the web for any topic
2. **claude-cli-llm**: Summarize search results using Claude
3. **document-converter**: Convert summary to PDF
4. **telegram-sender**: Send PDF to Telegram


## Type
composite

## Base Skills
- web-search
- claude-cli-llm
- document-converter
- telegram-sender

## Execution
sequential


## Capabilities
- research
- document
- communicate

## Usage

```python
from skills.search_summarize_pdf_telegram.tools import search_summarize_pdf_telegram_tool

result = await search_summarize_pdf_telegram_tool({
    'topic': 'multi agent systems',
    'send_telegram': True,
    'telegram_chat_id': '810015653'
})
```

## Parameters

- `topic` (str, required): Search topic
- `max_results` (int, optional): Max search results (default: 10)
- `summarize_prompt` (str, optional): Custom summarization prompt
- `title` (str, optional): Report title
- `send_telegram` (bool, optional): Send to Telegram (default: True)
- `telegram_chat_id` (str, optional): Telegram chat ID
- `output_dir` (str, optional): Output directory
- `model` (str, optional): Claude model (default: 'sonnet')

## Architecture

Uses composite skill framework for DRY workflow composition.
Follows Source → Processor → Sink pattern:
- Source: web-search
- Processor: claude-cli-llm → document-converter
- Sink: telegram-sender

No code duplication - reuses existing skills.

## Triggers
- "search summarize pdf telegram"
- "create pdf"
- "generate pdf"
- "convert to pdf"
- "pdf"
- "send to telegram"
- "telegram message"
- "notify via telegram"

## Category
document-creation
