---
name: arxivtoreport
description: "Download ArXiv papers, analyze with AI, create comprehensive report. Use when user wants to research academic papers."
---

# ArXiv to Report Skill

## Description
Composite skill: ArXiv download → PDF extraction → AI analysis → Report generation

## Type
composite

## Capabilities
- data-fetch
- analyze
- generate

## Triggers
- "research arxiv paper"
- "analyze arxiv"
- "create report from paper"

## Category
research

## Base Skills
- arxiv-downloader
- document-converter
- claude-cli-llm

## Execution Mode
sequential

## Tools

### arxiv_to_report_tool
Download ArXiv paper, extract content, analyze, create report (all-in-one).

**Parameters:**
- `arxiv_id` (str, required): ArXiv ID (e.g., "2301.12345")
- `analysis_depth` (str, optional): "quick", "standard", "deep". Default: "standard"
- `output_format` (str, optional): "markdown", "pdf", "html". Default: "pdf"

**Returns:**
- `success` (bool): Whether operation succeeded
- `arxiv_id` (str): ArXiv paper ID
- `title` (str): Paper title
- `authors` (list): Paper authors
- `report_path` (str): Path to generated report
- `key_findings` (list): AI-extracted key findings

## Usage Examples
```python
result = arxiv_to_report_tool({
    'arxiv_id': '2301.12345',
    'analysis_depth': 'deep',
    'output_format': 'pdf'
})
```
