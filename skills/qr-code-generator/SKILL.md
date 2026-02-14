---
name: generating-qr-codes
description: "Generate QR codes as SVG or ASCII art from text or URLs. Pure Python, no external deps. Use when the user wants to generate QR code, create QR."
---

# Qr Code Generator Skill

Generate QR codes as SVG or ASCII art from text or URLs. Pure Python, no external deps. Use when the user wants to generate QR code, create QR.

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
- "qr"
- "qr code"
- "generate qr"
- "barcode"

## Category
content-creation

## Tools

### generate_qr_tool
Generate a QR code from text or URL.

**Parameters:**
- `data` (str, required): Text or URL to encode
- `format` (str, optional): Output format: svg, ascii (default: svg)
- `size` (int, optional): Module size in pixels for SVG (default: 10)

**Returns:**
- `success` (bool)
- `qr_code` (str): QR code as SVG string or ASCII art
- `format` (str): Output format used

## Dependencies
None
