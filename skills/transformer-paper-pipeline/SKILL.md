---
name: processing-transformer-papers
description: "Generates a research paper on Transformers using Claude CLI LLM, converts it to LaTeX PDF, and sends to Telegram. Use when the user wants to generate, send."
---

# Transformer Paper Pipeline

Generates a research paper on Transformers using Claude CLI LLM, converts it to LaTeX PDF, and sends to Telegram.


## Type
composite

## Base Skills
- arxiv-downloader
- document-converter

## Execution
sequential


## Capabilities
- research
- document

## Tools

### `generate_transformer_paper_tool`

Generates a comprehensive transformer paper and sends PDF to Telegram.

**Parameters:**
- `topic` (str, optional): Specific transformer topic (default: 'Transformers')
- `paper_length` (str, optional): 'short', 'medium', or 'long' (default: 'medium')
- `include_math` (bool, optional): Include mathematical equations (default: True)
- `output_dir` (str, optional): Output directory (default: './output/transformer_papers')
- `send_telegram` (bool, optional): Send PDF to Telegram (default: True)
- `telegram_chat_id` (str, optional): Telegram chat ID (uses env var if not provided)
- `compile_pdf` (bool, optional): Compile LaTeX to PDF (default: True)

**Returns:**
- `success` (bool): Whether generation succeeded
- `paper_content` (str): Generated paper content
- `tex_file` (str): Path to generated .tex file
- `pdf_file` (str, optional): Path to generated PDF if compiled
- `telegram_sent` (bool): Whether sent to Telegram
- `telegram_message_id` (int, optional): Telegram message ID if sent
- `error` (str, optional): Error message if failed

## Workflow

```
Task Progress:
- [ ] Step 1: Generate paper content
- [ ] Step 2: Compile LaTeX PDF
- [ ] Step 3: Send to Telegram
```

**Step 1: Generate paper content**
Use Claude LLM to write a comprehensive transformer research paper.

**Step 2: Compile LaTeX PDF**
Convert the paper content to LaTeX and compile to PDF.

**Step 3: Send to Telegram**
Deliver the compiled PDF to the specified Telegram chat.

## Triggers
- "transformer paper pipeline"
- "generate"
- "send"

## Category
content-creation
