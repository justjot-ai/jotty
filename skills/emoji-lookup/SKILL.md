---
name: looking-up-emojis
description: "Search emojis by name/keyword, get emoji info, convert shortcodes to unicode. Built-in emoji database."
---

# Emoji Lookup Skill

Search emojis by name/keyword, get emoji info, convert shortcodes to unicode. Built-in emoji database.

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
- "emoji"
- "emoji search"
- "emoji lookup"
- "shortcode to emoji"

## Category
utilities

## Tools

### emoji_lookup_tool
Search and convert emojis.

**Parameters:**
- `action` (str, required): search, info, convert
- `query` (str): Search term (for search)
- `emoji` (str): Emoji char (for info)
- `shortcode` (str): Shortcode like :smile: (for convert)

## Dependencies
None
