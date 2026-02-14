---
name: comparing-json
description: "Compare two JSON objects and find additions, deletions, and modifications with JSON path."
---

# Json Diff Skill

Compare two JSON objects and find additions, deletions, and modifications with JSON path.

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
- "json diff"
- "compare json"
- "json difference"
- "diff objects"

## Category
development

## Tools

### json_diff_tool
Compare two JSON objects.

**Parameters:**
- `a` (dict/str, required): First JSON object or JSON string
- `b` (dict/str, required): Second JSON object or JSON string

**Returns:**
- `added` (list): Paths present in b but not a
- `removed` (list): Paths present in a but not b
- `modified` (list): Paths with different values

## Dependencies
None
