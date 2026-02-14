# NotebookLM PDF Generator

Generate PDFs using Google's NotebookLM by uploading markdown or text content.

## Description

This skill uploads markdown or text content to Google NotebookLM and generates a formatted PDF document. NotebookLM provides AI-powered document processing and can create well-formatted PDFs from uploaded content.


## Type
derived

## Base Skills
- pdf-tools


## Capabilities
- document

## Features

- Upload markdown or text files to NotebookLM
- Generate formatted PDFs with AI-enhanced content
- Support for various document types
- Automatic content formatting

## Usage

```python
from skills.notebooklm_pdf.tools import notebooklm_pdf_tool

result = await notebooklm_pdf_tool({
    'content': '# My Document\n\nContent here...',
    'content_file': '/path/to/file.md',  # Alternative to content
    'title': 'My Document Title',
    'output_file': '/path/to/output.pdf',
    'output_dir': '/path/to/outputs'  # Optional, defaults to stock_market/outputs
})
```

## Parameters

- `content` (str, optional): Markdown or text content to upload
- `content_file` (str, optional): Path to markdown/text file (alternative to content)
- `title` (str, optional): Document title
- `output_file` (str, optional): Output PDF path
- `output_dir` (str, optional): Output directory (defaults to stock_market/outputs)

## Requirements

- Google account with NotebookLM access
- Playwright or Selenium for browser automation (if API unavailable)
- Or NotebookLM API credentials (if available)

## Notes

NotebookLM may require authentication. The skill will attempt to use:
1. NotebookLM API (if available)
2. Browser automation with Playwright
3. Direct file upload methods

## Triggers
- "notebooklm pdf"
- "create pdf"
- "generate pdf"
- "convert to pdf"
- "pdf"
- "upload"

## Category
document-creation
