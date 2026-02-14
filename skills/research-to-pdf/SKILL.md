---
name: researchtopdf
description: "Research a topic, create comprehensive report, and optionally send to Telegram. Use when user wants to research and create PDF report."
---

# Research to PDF Skill

## Description
Composite skill that performs end-to-end research workflow: web search → LLM analysis → PDF generation → optional Telegram delivery. Consolidates 4 separate tools into one seamless operation.

## Type
composite

## Capabilities
- data-fetch
- analyze
- generate
- communicate

## Triggers
- "research and create PDF"
- "research to PDF"
- "create research report"
- "research and send"

## Category
research

## Base Skills
- web-search
- claude-cli-llm
- document-converter
- telegram-sender

## Execution Mode
sequential

## Tools

### research_to_pdf_tool
Research a topic, analyze results, create PDF report, and optionally send via Telegram (all-in-one).

**Parameters:**
- `topic` (str, required): Topic to research (e.g., "AI trends 2024")
- `depth` (str, optional): Research depth - "quick" (5 results), "standard" (10), "deep" (20). Defaults to "standard"
- `send_telegram` (bool, optional): Whether to send PDF to Telegram. Defaults to False
- `telegram_chat_id` (str, optional): Telegram chat ID if send_telegram is True

**Returns:**
- `success` (bool): Whether operation completed successfully
- `pdf_path` (str): Path to generated PDF report
- `topic` (str): Topic researched
- `sources_count` (int): Number of sources analyzed
- `summary_length` (int): Length of generated summary
- `telegram_sent` (bool): Whether PDF was sent to Telegram
- `error` (str, optional): Error message if failed

## Usage Examples
```python
# Example 1: Basic research to PDF
result = research_to_pdf_tool({
    'topic': 'Quantum Computing Advances 2024'
})
# Returns: {'success': True, 'pdf_path': '/path/to/report.pdf', ...}

# Example 2: Deep research with Telegram delivery
result = research_to_pdf_tool({
    'topic': 'Climate Tech Startups',
    'depth': 'deep',
    'send_telegram': True,
    'telegram_chat_id': '123456789'
})
```

## Requirements
- web-search skill
- claude-cli-llm skill
- document-converter skill
- telegram-sender skill (if send_telegram=True)

## Workflow
1. **Search**: Perform web search for topic
2. **Analyze**: Use LLM to analyze and synthesize findings
3. **Generate**: Create formatted PDF report
4. **Deliver**: Optionally send via Telegram

## Error Handling
Common errors and solutions:
- **Web search failed**: Check internet connection and search API availability
- **LLM analysis failed**: Verify LLM provider (Claude, OpenAI, Groq) is configured
- **PDF generation failed**: Check document-converter skill and filesystem permissions
- **Telegram send failed**: Verify `TELEGRAM_TOKEN` environment variable and chat_id
