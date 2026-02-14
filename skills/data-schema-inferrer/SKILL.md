---
name: inferring-data-schemas
description: "Infer JSON Schema from sample data. Pure Python. Use when the user wants to generate schema, infer types, JSON schema from data."
---

# Data Schema Inferrer Skill

Infer JSON Schema from sample data. Pure Python. Use when the user wants to generate schema, infer types, JSON schema from data.

## Type
base

## Capabilities
- analyze
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
- "schema"
- "json schema"
- "infer schema"
- "data types"
- "type inference"

## Category
data-analysis

## Tools

### infer_schema_tool
Infer JSON Schema from sample data.

**Parameters:**
- `data` (any, required): Sample JSON data (object, array, or primitive)
- `title` (str, optional): Schema title

**Returns:**
- `success` (bool)
- `schema` (dict): Inferred JSON Schema (draft-07)

## Dependencies
None
