---
name: translating-morse-code
description: "Convert text to/from Morse code. Support letters, numbers, common punctuation."
---

# Morse Code Translator Skill

Convert text to/from Morse code. Support letters, numbers, common punctuation.

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
- "morse code"
- "morse translate"
- "text to morse"
- "morse to text"

## Category
utilities

## Tools

### morse_tool
Convert text to/from Morse code.

**Parameters:**
- `action` (str, required): encode or decode
- `text` (str): Text to encode (for encode)
- `morse` (str): Morse code to decode (for decode), dots and dashes separated by spaces

## Dependencies
None
