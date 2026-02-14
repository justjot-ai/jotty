---
name: generating-uuids
description: "Generate UUIDs (v1, v4, v5) and ULID identifiers. Use when the user wants to generate UUID, ULID, unique id."
---

# Uuid Generator Skill

Generate UUIDs (v1, v4, v5) and ULID identifiers. Use when the user wants to generate UUID, ULID, unique id.

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
- "uuid"
- "guid"
- "unique id"
- "ulid"
- "generate id"

## Category
development

## Tools

### generate_uuid_tool
Generate UUID identifiers.

**Parameters:**
- `version` (int, optional): UUID version 1 or 4 (default: 4)
- `count` (int, optional): Number of UUIDs (default: 1, max: 100)
- `uppercase` (bool, optional): Uppercase output (default: false)

**Returns:**
- `success` (bool)
- `uuids` (list): Generated UUIDs

## Dependencies
None
