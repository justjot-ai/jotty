---
name: tracking-habits
description: "Track daily habits with streaks, completion rates, and statistics. Uses JSON storage. Use when the user wants to track habit, habit streak, daily tracker."
---

# Habit Tracker Skill

Track daily habits with streaks, completion rates, and statistics. Uses JSON storage. Use when the user wants to track habit, habit streak, daily tracker.

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
- "habit"
- "habit tracker"
- "streak"
- "daily habit"
- "routine"
- "track habit"

## Category
workflow-automation

## Tools

### log_habit_tool
Log a habit completion for today.

**Parameters:**
- `habit` (str, required): Habit name
- `completed` (bool, optional): Whether completed (default: true)
- `date` (str, optional): Date YYYY-MM-DD (default: today)
- `notes` (str, optional): Optional notes
- `storage_file` (str, optional): JSON storage path

**Returns:**
- `success` (bool)
- `habit` (str): Habit name
- `current_streak` (int): Current consecutive day streak

### habit_stats_tool
Get habit statistics and streaks.

**Parameters:**
- `habit` (str, optional): Specific habit (default: all)
- `storage_file` (str, optional): JSON storage path

**Returns:**
- `success` (bool)
- `habits` (dict): Per-habit stats with streaks and completion rates

## Dependencies
None
