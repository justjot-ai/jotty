---
description: "Multi-format document generation (PDF, EPUB, HTML, DOCX, PPTX) from markdown. Use when the user wants to generate documents, convert markdown to PDF/EPUB/HTML, or create presentations."
---

# Document Tools

Orchestrates document-converter, epub-builder, and presenton skills for multi-format output.

## Type

composite

## Base Skills

- document-converter
- epub-builder
- presenton

## Capabilities

- document
- productivity
- conversion

## Triggers

- generate document
- create PDF
- convert to EPUB
- make presentation
- export to DOCX
- generate HTML

## Use When

- User wants to generate PDF, EPUB, HTML, DOCX, or presentation from markdown
- User wants multiple formats from one source file
- User wants rich EPUB with chapters or a slide deck

## Tools

### generate_pdf_tool

Generate PDF from a markdown file.

**Parameters:** markdown_path (required), title, author, page_size (default a4), output_path

### generate_epub_tool

Generate EPUB from a markdown file.

**Parameters:** markdown_path (required), title (required), author (required), output_path

### generate_epub_with_chapters_tool

Generate EPUB with chapters (epub-builder).

**Parameters:** chapters (list of {title, content}), title, author, description, language, output_path

### generate_html_tool

Generate HTML from markdown.

**Parameters:** markdown_path (required), title, standalone (default True), output_path

### generate_docx_tool

Generate DOCX from markdown.

**Parameters:** markdown_path (required), title, output_path

### generate_presentation_tool

Generate presentation (PPTX/PDF) from content.

**Parameters:** content (required), title (required), n_slides, export_as (pptx/pdf), tone

### generate_all_formats_tool

Generate multiple formats from one markdown file.

**Parameters:** markdown_path (required), formats (required, e.g. ["pdf","epub","html"]), title (required), author
