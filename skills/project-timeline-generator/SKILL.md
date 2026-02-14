---
name: generating-project-timelines
description: "Generate text-based Gantt charts and project timelines from task lists. Use when the user wants to project timeline, Gantt chart, schedule tasks, project plan."
---

# Project Timeline Generator Skill

Generate text-based Gantt charts and project timelines from task lists. Use when the user wants to project timeline, Gantt chart, schedule tasks, project plan.

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
- "timeline"
- "gantt chart"
- "project plan"
- "schedule"
- "project timeline"
- "milestones"

## Category
workflow-automation

## Tools

### generate_timeline_tool
Generate a text-based Gantt chart / project timeline.

**Parameters:**
- `tasks` (list, required): List of {name, duration, start_day, depends_on (optional)}
- `title` (str, optional): Project title
- `scale` (str, optional): day, week (default: day)

**Returns:**
- `success` (bool)
- `gantt_chart` (str): Text-based Gantt chart
- `total_duration` (int): Total project duration in days
- `critical_path` (list): Tasks on critical path

## Dependencies
None
