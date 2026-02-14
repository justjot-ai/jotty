---
name: converting-colors
description: "Convert colors between HEX, RGB, HSL, and named formats. Generate palettes and complementary colors. Use when the user wants to convert color, hex to rgb, color palette."
---

# Color Converter Skill

Convert colors between HEX, RGB, HSL, and named formats. Generate palettes and complementary colors. Use when the user wants to convert color, hex to rgb, color palette.

## Type
base

## Capabilities
- generate

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
- "color"
- "hex"
- "rgb"
- "hsl"
- "color palette"
- "convert color"

## Category
content-creation

## Tools

### convert_color_tool
Convert between color formats.

**Parameters:**
- `color` (str, required): Color value (hex, rgb, hsl, or named)
- `to_format` (str, optional): Target format: hex, rgb, hsl (default: all)

**Returns:**
- `success` (bool)
- `hex` (str): Hex value
- `rgb` (dict): RGB values
- `hsl` (dict): HSL values

## Dependencies
None
