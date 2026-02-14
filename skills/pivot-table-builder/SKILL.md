---
name: building-pivot-tables
description: "Create pivot tables from data with row/column grouping and aggregation. Pure Python. Use when the user wants to pivot table, group by, aggregate data."
---

# Pivot Table Builder Skill

Create pivot tables from data with row/column grouping and aggregation. Pure Python. Use when the user wants to pivot table, group by, aggregate data.

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
- "pivot"
- "pivot table"
- "group by"
- "aggregate"
- "crosstab"

## Category
data-analysis

## Tools

### pivot_table_tool
Create a pivot table from data.

**Parameters:**
- `data` (list, required): List of objects
- `rows` (str, required): Field for row grouping
- `columns` (str, optional): Field for column grouping
- `values` (str, required): Field to aggregate
- `aggfunc` (str, optional): sum, mean, count, min, max (default: sum)

**Returns:**
- `success` (bool)
- `table` (dict): Pivot table data
- `formatted` (str): ASCII formatted table

## Dependencies
None
