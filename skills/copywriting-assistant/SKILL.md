---
name: assisting-copywriting
description: "Generate marketing copy using proven frameworks: AIDA, PAS, BAB, 4Ps, FAB. Use when the user wants to write marketing copy, ad copy, sales copy, AIDA framework."
---

# Copywriting Assistant Skill

Generate marketing copy using proven frameworks: AIDA, PAS, BAB, 4Ps, FAB. Use when the user wants to write marketing copy, ad copy, sales copy, AIDA framework.

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
- "copywriting"
- "marketing copy"
- "AIDA"
- "PAS"
- "ad copy"
- "sales copy"
- "landing page copy"

## Category
content-creation

## Tools

### generate_copy_tool
Generate marketing copy using a copywriting framework.

**Parameters:**
- `product` (str, required): Product or service name
- `audience` (str, required): Target audience
- `framework` (str, optional): AIDA, PAS, BAB, 4Ps, FAB (default: AIDA)
- `key_benefit` (str, optional): Main benefit to highlight
- `tone` (str, optional): professional, casual, urgent, friendly (default: professional)

**Returns:**
- `success` (bool)
- `copy` (dict): Copy organized by framework sections
- `framework` (str): Framework used

## Dependencies
None
