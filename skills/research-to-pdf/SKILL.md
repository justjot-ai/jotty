---
name: research-to-pdf
description: "Research any topic and generate a comprehensive PDF report with AI-synthesized analysis. Use when the user wants to research report, create report, pdf report."
---

# research-to-pdf

Research any topic and generate a comprehensive PDF report with AI-synthesized analysis.


## Type
composite

## Base Skills
- web-search
- claude-cli-llm
- document-converter

## Execution
sequential

**USE THIS SKILL when user asks to:**
- Research something AND generate/create PDF
- Compare topics AND send as PDF
- Analyze something AND create PDF report
- Any task with "pdf" + research/compare/analyze

**Key Features:**
- **AI Synthesis**: Converts raw search results into comprehensive 8-15 page analysis
- **Structured Reports**: Executive summary, detailed analysis, insights, recommendations
- **Deep Research**: Uses `last30days-claude-cli` for comprehensive web research
- **PDF Generation**: Professional PDF with proper formatting
- **No API keys required**: Uses Jotty's built-in web search and Claude CLI


## Capabilities
- research
- document


## Triggers
- "research report"
- "create report"
- "pdf report"
- "research to pdf"

## Category
document-creation

## Usage

```python
from core.registry.skills_registry import get_skills_registry

registry = get_skills_registry()
registry.init()

skill = registry.get_skill('research-to-pdf')
tool = skill.tools['research_to_pdf_tool']

result = await tool({
    'topic': 'multi agent systems',
    'deep': True,
    'title': 'Multi-Agent Systems Research',
    'author': 'Your Name'
})
```

## Parameters

- `topic` (required): Topic to research
- `output_dir` (optional): Output directory (default: ~/jotty/reports)
- `title` (optional): Report title (default: auto-generated from topic)
- `author` (optional): Report author (default: 'Jotty Framework')
- `page_size` (optional): PDF page size - 'a4', 'a5', 'a6', 'letter' (default: 'a4')
- `deep` (optional): Deep research mode (default: False)
- `quick` (optional): Quick research mode (default: False)

## Returns

- `success` (bool): Whether operation succeeded
- `pdf_path` (str): Path to generated PDF
- `md_path` (str): Path to markdown file
- `file_size` (int): PDF file size in bytes

## Version

1.0.0

## Author

Jotty Framework
