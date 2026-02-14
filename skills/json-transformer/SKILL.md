---
name: transforming-json
description: "Transform, query, flatten, and merge JSON structures. Use when the user wants to transform JSON, flatten, merge, query, jq."
---

# Json Transformer Skill

Transform, query, flatten, and merge JSON structures. Use when the user wants to transform JSON, flatten, merge, query, jq.

## Type
base

## Capabilities
- code
- data-fetch

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
- "json"
- "transform json"
- "flatten json"
- "merge json"
- "jq"
- "jsonpath"

## Category
data-analysis

## Tools

### flatten_json_tool
Flatten nested JSON into dot-notation keys.

**Parameters:**
- `data` (dict, required): JSON object to flatten
- `separator` (str, optional): Key separator (default: ".")

**Returns:**
- `success` (bool)
- `result` (dict): Flattened key-value pairs

### merge_json_tool
Deep merge two or more JSON objects.

**Parameters:**
- `objects` (list, required): List of JSON objects to merge

**Returns:**
- `success` (bool)
- `result` (dict): Merged object

### query_json_tool
Query JSON with dot-notation path.

**Parameters:**
- `data` (dict, required): JSON to query
- `path` (str, required): Dot-notation path (e.g. "users.0.name")

**Returns:**
- `success` (bool)
- `result` (any): Value at path

## Dependencies
None
