---
name: comparing-text
description: "Compare two texts or files and show differences. Use when the user wants to diff, compare, text difference."
---

# Diff Tool Skill

Compare two texts or files and show differences. Use when the user wants to diff, compare, text difference.

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
- "diff"
- "compare"
- "difference"
- "text diff"
- "file diff"

## Category
development

## Tools

### diff_text_tool
Show differences between two texts.

**Parameters:**
- `text_a` (str, required): First text
- `text_b` (str, required): Second text
- `context_lines` (int, optional): Lines of context (default: 3)

**Returns:**
- `success` (bool)
- `diff` (str): Unified diff output
- `changes` (int): Number of changed lines

## Dependencies
None
