---
name: encoding-base64
description: "Encode and decode Base64, URL-safe Base64, and hex strings. Use when the user wants to encode, decode, base64, hex."
---

# Base64 Encoder Skill

Encode and decode Base64, URL-safe Base64, and hex strings. Use when the user wants to encode, decode, base64, hex.

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
- "base64"
- "encode"
- "decode"
- "hex encode"
- "url encode"

## Category
development

## Tools

### base64_encode_tool
Encode text or file to Base64.

**Parameters:**
- `text` (str, optional): Text to encode
- `encoding` (str, optional): base64, base64url, hex (default: base64)

**Returns:**
- `success` (bool)
- `encoded` (str): Encoded string

## Dependencies
None
