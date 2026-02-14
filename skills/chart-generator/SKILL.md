---
name: generating-charts
description: "Generate ASCII bar and line charts from data. Pure Python. Use when the user wants to create chart, bar chart, line chart, visualize data."
---

# Chart Generator Skill

Generate ASCII bar and line charts from data. Pure Python. Use when the user wants to create chart, bar chart, line chart, visualize data.

## Type
base

## Capabilities
- generate
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
- "chart"
- "bar chart"
- "line chart"
- "ascii chart"
- "visualize"

## Category
data-analysis

## Tools

### bar_chart_tool
Generate ASCII horizontal bar chart.

**Parameters:**
- `data` (dict, required): Label-value pairs
- `width` (int, optional): Chart width in chars (default: 50)
- `sort` (bool, optional): Sort by value descending (default: true)

**Returns:**
- `success` (bool)
- `chart` (str): ASCII bar chart

## Dependencies
None
