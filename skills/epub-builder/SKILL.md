---
name: building-epub-books
description: "Build EPUB e-books from text or markdown content using stdlib zipfile. Use when the user wants to create epub, build ebook, convert text to epub."
---

# Epub Builder Skill

Build EPUB e-books from text or markdown content using stdlib zipfile. Use when the user wants to create epub, build ebook, convert text to epub.

## Type
base

## Capabilities
- generate
- document

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
- "epub"
- "ebook"
- "e-book"
- "create epub"
- "build ebook"

## Category
document-creation

## Tools

### build_epub_tool
Build an EPUB e-book from chapters.

**Parameters:**
- `title` (str, required): Book title
- `author` (str, required): Author name
- `chapters` (list, required): List of {title, content} dicts
- `output_path` (str, optional): Output file path (default: title.epub)
- `language` (str, optional): Language code (default: en)
- `description` (str, optional): Book description

**Returns:**
- `success` (bool)
- `output_path` (str): Path to generated EPUB
- `chapter_count` (int): Number of chapters

## Dependencies
None
