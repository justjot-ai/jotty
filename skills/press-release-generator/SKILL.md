---
name: generating-press-releases
description: "Generate press release templates following AP style with proper structure. Use when the user wants to write press release, news release, PR template."
---

# Press Release Generator Skill

Generate press release templates following AP style with proper structure. Use when the user wants to write press release, news release, PR template.

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
- "press release"
- "news release"
- "PR template"
- "media release"
- "announcement"

## Category
content-creation

## Tools

### generate_press_release_tool
Generate a structured press release.

**Parameters:**
- `headline` (str, required): Press release headline
- `company` (str, required): Company name
- `body_points` (list, required): Key points to include
- `city` (str, optional): Dateline city (default: New York)
- `contact_name` (str, optional): Media contact name
- `contact_email` (str, optional): Media contact email
- `quote_attribution` (str, optional): Name for the quote
- `quote_title` (str, optional): Title of person quoted

**Returns:**
- `success` (bool)
- `press_release` (str): Formatted press release text
- `word_count` (int): Total word count

## Dependencies
None
