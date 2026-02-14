---
name: tracking-pomodoro
description: "Track Pomodoro work sessions, breaks, and productivity stats. Use when the user wants to pomodoro timer, focus timer, work session, productivity tracker."
---

# Pomodoro Timer Skill

Track Pomodoro work sessions, breaks, and productivity stats. Use when the user wants to pomodoro timer, focus timer, work session, productivity tracker.

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
- "pomodoro"
- "focus timer"
- "work session"
- "break timer"
- "productivity"

## Category
workflow-automation

## Tools

### start_pomodoro_tool
Start a Pomodoro session.

**Parameters:**
- `task` (str, required): Task description
- `work_minutes` (int, optional): Work duration (default: 25)
- `break_minutes` (int, optional): Break duration (default: 5)
- `long_break_minutes` (int, optional): Long break after 4 sessions (default: 15)

**Returns:**
- `success` (bool)
- `session` (dict): Session details with start time and durations

### pomodoro_stats_tool
Get Pomodoro productivity statistics.

**Parameters:**
- `storage_file` (str, optional): JSON storage path

**Returns:**
- `success` (bool)
- `total_sessions` (int): Total completed sessions
- `total_focus_minutes` (int): Total focus time
- `by_task` (dict): Sessions per task

## Dependencies
None
