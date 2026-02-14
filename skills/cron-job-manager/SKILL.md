---
name: managing-cron-jobs
description: "Create, validate, and explain cron expressions. Use when the user wants to create cron, schedule, crontab, explain cron."
---

# Cron Job Manager Skill

Create, validate, and explain cron expressions. Use when the user wants to create cron, schedule, crontab, explain cron.

## Type
base

## Capabilities
- code
- devops

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
- "cron"
- "crontab"
- "schedule"
- "cron expression"
- "recurring job"

## Category
development

## Tools

### explain_cron_tool
Explain a cron expression in human-readable format.

**Parameters:**
- `expression` (str, required): Cron expression (5 or 6 fields)

**Returns:**
- `success` (bool)
- `explanation` (str): Human-readable explanation
- `next_runs` (list): Next 5 execution times

## Dependencies
None
