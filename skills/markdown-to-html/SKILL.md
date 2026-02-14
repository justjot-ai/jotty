---
name: converting-markdown
description: "Convert Markdown to styled HTML with syntax highlighting and TOC. Use when the user wants to convert markdown, markdown to html, render markdown."
---

# Markdown To Html Skill

Convert Markdown to styled HTML with syntax highlighting and TOC. Use when the user wants to convert markdown, markdown to html, render markdown.

## Type
base

## Capabilities
- document
- generate

## Reference
For detailed tool documentation, see [REFERENCE.md](REFERENCE.md).

## Workflow

```
Task Progress:
- [ ] Step 1: Parse input parameters
- [ ] Step 2: Execute operation
- [ ] Step 3: Return results
```

## Triggers
- "markdown"
- "convert markdown"
- "markdown to html"
- "render md"

## Category
document-creation

## Tools

### markdown_to_html_tool
Convert Markdown text to HTML.

**Parameters:**
- `markdown` (str, required): Markdown text
- `include_style` (bool, optional): Include CSS styling (default: true)
- `toc` (bool, optional): Generate table of contents (default: false)

**Returns:**
- `success` (bool)
- `html` (str): Rendered HTML

## Dependencies
None
