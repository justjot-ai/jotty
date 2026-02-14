---
name: converting-number-bases
description: "Convert numbers between arbitrary bases (2-36). Support custom digit sets."
---

# Number Base Converter Skill

Convert numbers between arbitrary bases (2-36). Support custom digit sets.

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
- "base convert"
- "number base"
- "base 2"
- "base 16"
- "radix"

## Category
utilities

## Tools

### base_convert_tool
Convert numbers between arbitrary bases 2-36.

**Parameters:**
- `number` (str, required): The number to convert
- `from_base` (int, required): Source base (2-36)
- `to_base` (int, required): Target base (2-36)
- `custom_digits` (str): Custom digit set (optional)

## Dependencies
None
