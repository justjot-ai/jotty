---
name: analyzing-logs
description: "Parse and summarize log files, find errors, count patterns. Pure Python. Use when the user wants to analyze logs, find errors, parse log file."
---

# Log Analyzer Skill

Parse and summarize log files, find errors, count patterns. Pure Python. Use when the user wants to analyze logs, find errors, parse log file.

## Type
base

## Capabilities
- analyze
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
- "log"
- "analyze log"
- "log file"
- "error log"
- "parse logs"

## Category
development

## Tools

### analyze_log_tool
Parse and summarize log content.

**Parameters:**
- `content` (str, optional): Log text content
- `file_path` (str, optional): Path to log file
- `level_filter` (str, optional): Filter by level (ERROR, WARN, INFO, DEBUG)

**Returns:**
- `success` (bool)
- `summary` (dict): Level counts, error messages, time range
- `errors` (list): Error log lines

## Dependencies
None
