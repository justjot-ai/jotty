---
name: converting-roman-numerals
description: "Convert between Roman numerals and integers. Validate Roman numeral strings."
---

# Roman Numeral Converter Skill

Convert between Roman numerals and integers. Validate Roman numeral strings.

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
- "roman numeral"
- "roman to integer"
- "integer to roman"
- "roman convert"

## Category
utilities

## Tools

### roman_tool
Convert between Roman numerals and integers.

**Parameters:**
- `action` (str, required): to_roman, to_integer, validate
- `number` (int): Integer to convert (for to_roman)
- `roman` (str): Roman numeral string (for to_integer/validate)

## Dependencies
None
