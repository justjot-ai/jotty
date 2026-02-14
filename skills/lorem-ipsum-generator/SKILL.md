---
name: generating-lorem-ipsum
description: "Generate Lorem Ipsum placeholder text in paragraphs, sentences, or words. Use when the user wants to generate placeholder text, lorem ipsum, dummy text."
---

# Lorem Ipsum Generator Skill

Generate Lorem Ipsum placeholder text in paragraphs, sentences, or words. Use when the user wants to generate placeholder text, lorem ipsum, dummy text.

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
- "lorem ipsum"
- "placeholder text"
- "dummy text"
- "filler text"

## Category
content-creation

## Tools

### lorem_ipsum_tool
Generate Lorem Ipsum placeholder text.

**Parameters:**
- `paragraphs` (int, optional): Number of paragraphs (default: 1)
- `sentences` (int, optional): Number of sentences (overrides paragraphs)
- `words` (int, optional): Number of words (overrides both)

**Returns:**
- `success` (bool)
- `text` (str): Generated Lorem Ipsum text

## Dependencies
None
