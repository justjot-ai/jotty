---
name: sending-last30days-pdf-remarkable
description: "This composite skill combines: 1. **last30days-claude-cli**: Research topics from last 30 days 2. **document-converter**: Convert markdown to PDF 3. **remarkable-sender**: Upload PDF to reMarkable tablet. Use when the user wants to create pdf, generate pdf, convert to pdf."
---

# Last30Days → PDF → reMarkable Composite Skill

Research topics using last30days skill, generate PDF, and send to reMarkable tablet.

## Description

This composite skill combines:
1. **last30days-claude-cli**: Research topics from last 30 days
2. **document-converter**: Convert markdown to PDF
3. **remarkable-sender**: Upload PDF to reMarkable tablet


## Type
composite

## Base Skills
- last30days-claude-cli
- document-converter
- remarkable-sender

## Execution
sequential


## Capabilities
- research
- document
- communicate

## Usage

```python
from skills.last30days_to_pdf_remarkable.tools import last30days_to_pdf_remarkable_tool

result = await last30days_to_pdf_remarkable_tool({
    'topic': 'multi agent systems',
    'send_remarkable': True,
    'folder': '/Research'
})
```

## Parameters

- `topic` (str, required): Research topic
- `deep` (bool, optional): Deep research mode (default: False)
- `quick` (bool, optional): Quick research mode (default: False)
- `title` (str, optional): Report title
- `send_remarkable` (bool, optional): Send to reMarkable (default: True)
- `folder` (str, optional): reMarkable folder path (default: '/')
- `document_name` (str, optional): Document name on reMarkable
- `output_dir` (str, optional): Output directory

## Requirements

- `rmapi` must be installed and authenticated
- See: https://github.com/juruen/rmapi

## Architecture

Uses composite skill framework for DRY workflow composition.
No code duplication - reuses existing skills.

## Workflow

```
Task Progress:
- [ ] Step 1: Research recent topics
- [ ] Step 2: Generate PDF report
- [ ] Step 3: Send to reMarkable
```

**Step 1: Research recent topics**
Use last30days-claude-cli to research the topic from the past 30 days.

**Step 2: Generate PDF report**
Convert the research markdown into a formatted PDF document.

**Step 3: Send to reMarkable**
Upload the PDF to the reMarkable tablet for offline reading.

## Triggers
- "last30days to pdf remarkable"
- "create pdf"
- "generate pdf"
- "convert to pdf"
- "pdf"
- "send to remarkable"
- "remarkable tablet"

## Category
document-creation
