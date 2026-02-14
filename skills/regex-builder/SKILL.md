---
name: building-regex
description: "Build regex patterns from descriptions. Common patterns: email, URL, phone, IP, date. Explain regex."
---

# Regex Builder Skill

Build regex patterns from descriptions. Common patterns: email, URL, phone, IP, date. Explain regex.

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
- "regex"
- "regular expression"
- "pattern"
- "regex builder"

## Category
development

## Tools

### regex_tool
Build, test, or explain regex patterns.

**Parameters:**
- `action` (str, required): preset, test, explain
- `name` (str): Preset name (for preset): email, url, phone, ip, date, uuid
- `pattern` (str): Regex pattern (for test/explain)
- `text` (str): Text to test against (for test)

## Dependencies
None
