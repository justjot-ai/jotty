---
name: optimizing-seo-content
description: "Analyze text for SEO: keyword density, readability score, meta tag suggestions. Use when the user wants to check SEO, keyword density, readability, optimize content."
---

# Seo Content Optimizer Skill

Analyze text for SEO: keyword density, readability score, meta tag suggestions. Use when the user wants to check SEO, keyword density, readability, optimize content.

## Type
base

## Capabilities
- analyze

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
- "seo"
- "keyword density"
- "readability"
- "optimize content"
- "meta tags"

## Category
content-creation

## Tools

### analyze_seo_tool
Analyze text for SEO metrics.

**Parameters:**
- `text` (str, required): Text content to analyze
- `keywords` (list, optional): Target keywords to check density
- `title` (str, optional): Page title to analyze

**Returns:**
- `success` (bool)
- `word_count` (int): Total words
- `readability` (dict): Readability scores
- `keyword_density` (dict): Keyword frequencies
- `suggestions` (list): SEO improvement suggestions

## Dependencies
None
