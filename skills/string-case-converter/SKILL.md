---
name: converting-string-case
description: "Convert strings between camelCase, snake_case, kebab-case, PascalCase, UPPER_CASE. Use when the user wants to convert case, camelCase, snake_case."
---

# String Case Converter Skill

Convert strings between camelCase, snake_case, kebab-case, PascalCase, UPPER_CASE. Use when the user wants to convert case, camelCase, snake_case.

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
- "camelCase"
- "snake_case"
- "kebab-case"
- "PascalCase"
- "case convert"

## Category
development

## Tools

### convert_case_tool
Convert string between naming conventions.

**Parameters:**
- `text` (str, required): Text to convert
- `to_case` (str, required): Target: camelCase, snake_case, kebab-case, PascalCase, UPPER_CASE, Title Case

**Returns:**
- `success` (bool)
- `result` (str): Converted string

## Dependencies
None
