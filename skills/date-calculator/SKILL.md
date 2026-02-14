---
name: calculating-dates
description: "Calculate date differences, add/subtract durations, format dates, find business days. Use when the user wants to calculate date, days between, add days."
---

# Date Calculator Skill

Calculate date differences, add/subtract durations, format dates, find business days. Use when the user wants to calculate date, days between, add days.

## Type
base

## Capabilities
- analyze

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
- "date"
- "days between"
- "add days"
- "date difference"
- "business days"
- "timestamp"

## Category
workflow-automation

## Tools

### date_diff_tool
Calculate difference between two dates.

**Parameters:**
- `date_a` (str, required): First date (YYYY-MM-DD or ISO format)
- `date_b` (str, required): Second date

**Returns:**
- `success` (bool)
- `days` (int): Total days difference
- `weeks` (float): Weeks
- `months` (float): Approximate months
- `years` (float): Approximate years

## Dependencies
None
