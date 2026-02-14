---
name: generating-slugs
description: "Generate URL-friendly slugs from titles. Handle unicode, transliteration, custom separators."
---

# Slug Generator Skill

Generate URL-friendly slugs from titles. Handle unicode, transliteration, custom separators.

## Type
base

## Capabilities
- code

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
- "slug"
- "url slug"
- "slugify"
- "url-friendly"

## Category
development

## Tools

### slugify_tool
Generate a URL-friendly slug from text.

**Parameters:**
- `text` (str, required): Text to slugify
- `separator` (str): Separator character (default: -)
- `max_length` (int): Maximum slug length (default: 200)
- `lowercase` (bool): Force lowercase (default: true)

## Dependencies
None
