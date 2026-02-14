---
name: spelling-with-nato
description: "Convert text to NATO phonetic alphabet spelling. ABC -> Alfa Bravo Charlie."
---

# Nato Phonetic Alphabet Skill

Convert text to NATO phonetic alphabet spelling. ABC -> Alfa Bravo Charlie.

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
- "nato phonetic"
- "phonetic alphabet"
- "nato spelling"
- "spell out"

## Category
utilities

## Tools

### nato_tool
Convert text to/from NATO phonetic alphabet.

**Parameters:**
- `action` (str, required): encode or decode
- `text` (str): Text to spell out (for encode)
- `words` (str): NATO words to decode (for decode)

## Dependencies
None
