---
name: converting-binary
description: "Convert between binary, decimal, hex, octal. Bitwise operations (AND, OR, XOR, NOT, shift)."
---

# Binary Converter Skill

Convert between binary, decimal, hex, octal. Bitwise operations (AND, OR, XOR, NOT, shift).

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
- "binary"
- "hex"
- "octal"
- "bitwise"
- "binary convert"

## Category
utilities

## Tools

### binary_convert_tool
Convert numbers and perform bitwise operations.

**Parameters:**
- `action` (str, required): convert or bitwise
- `value` (str): Number to convert (for convert)
- `from_base` (str): binary/decimal/hex/octal (for convert)
- `to_base` (str): binary/decimal/hex/octal (for convert)
- `op` (str): AND/OR/XOR/NOT/LSHIFT/RSHIFT (for bitwise)
- `a` (int): First operand (for bitwise)
- `b` (int): Second operand (for bitwise)

## Dependencies
None
