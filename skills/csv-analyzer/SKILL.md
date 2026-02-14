---
name: analyzing-csv
description: "Load, filter, aggregate, and summarize CSV files. Use when the user wants to analyze CSV, filter rows, aggregate data, summarize columns."
---

# Csv Analyzer Skill

Load, filter, aggregate, and summarize CSV files. Use when the user wants to analyze CSV, filter rows, aggregate data, summarize columns.

## Type
base

## Capabilities
- analyze
- data-fetch

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
- "csv"
- "analyze csv"
- "filter csv"
- "csv summary"
- "tabular data"

## Category
data-analysis

## Tools

### csv_summary_tool
Get summary statistics for a CSV file.

**Parameters:**
- `file_path` (str, required): Path to CSV file
- `delimiter` (str, optional): Column delimiter (default: ",")

**Returns:**
- `success` (bool)
- `rows` (int): Number of rows
- `columns` (list): Column names
- `stats` (dict): Per-column statistics

## Dependencies
None
