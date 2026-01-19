# A2UI Widget Support in Jotty

Jotty now supports A2UI v0.8 widgets for rich UI rendering. Agents can return structured widgets (cards, lists, etc.) instead of plain text.

## Overview

**A2UI (Agent-to-UI)** is a specification for structured agent responses that can be rendered as rich UI components.

### Architecture

```
┌─────────────┐     ┌──────────────┐     ┌────────────────┐
│   Agent     │────▶│  Jotty       │────▶│  JustJot.ai    │
│  (returns   │     │  (streams    │     │  (renders      │
│   A2UI)     │     │   widgets)   │     │   widgets)     │
└─────────────┘     └──────────────┘     └────────────────┘
```

1. **Agent** returns A2UI formatted response
2. **Jotty** detects A2UI and streams as `a2ui_widget` events
3. **JustJot.ai** renders widgets using A2UIWidgetRenderer

## For Agent Developers

### Basic Usage

```python
from jotty.core.ui.a2ui import format_task_list, format_card, format_text

# Return task list
def my_agent_function(goal):
    tasks = [
        {
            "title": "Implement feature X",
            "status": "in_progress",
            "subtitle": "Priority: High",
            "icon": "circle"
        },
        {
            "title": "Write tests",
            "status": "completed",
            "icon": "check_circle"
        }
    ]
    return format_task_list(tasks, title="Current Tasks")

# Return status card
def get_build_status():
    return format_card(
        title="Build Status",
        subtitle="Last updated: 2 minutes ago",
        body="All tests passing ✅"
    )
```

### Advanced Usage (Builder API)

```python
from jotty.core.ui.a2ui import A2UIBuilder

builder = A2UIBuilder()
builder.add_card(title="Summary", body="5 tasks completed")
builder.add_list(items=[...])
builder.add_separator()
builder.add_text("Need help? Contact support.", style="italic")

return builder.build()
```

### Widget Types Supported

1. **Card** - Title, subtitle, body, footer
2. **List** - Items with title, subtitle, status, icons, metadata
3. **Text** - Plain text with optional styling (bold, italic)
4. **Image** - Image with caption
5. **Button** - Action buttons
6. **Separator** - Visual divider

## For Frontend Developers (JustJot.ai)

### Widget Registry

Widgets are defined in `SECTION_REGISTRY` and exposed via `/api/jotty/registry`.

### Rendering

The `SupervisorChatWidget` automatically detects and renders A2UI responses:

```tsx
// Automatically handled - no code needed!
// Widget detection and rendering is built-in
```

### Custom Widget Renderers

To add new widget types, update:
1. `SECTION_REGISTRY` in JustJot.ai
2. `A2UIWidgetRenderer.tsx` to handle new widget type

## Response Format

A2UI responses follow this structure:

```json
{
  "role": "assistant",
  "content": [
    {
      "type": "card",
      "title": "Task Status",
      "subtitle": "5 tasks total",
      "body": { "type": "text", "text": "3 completed, 2 pending" }
    },
    {
      "type": "list",
      "items": [
        {
          "title": "Task 1",
          "status": "completed",
          "icon": "check_circle"
        }
      ]
    }
  ]
}
```

## DRY Principles

✅ **Jotty** (SDK):
- Provides A2UI formatting utilities
- Handles widget streaming
- Generic and reusable

✅ **JustJot.ai** (Client):
- Consumes A2UI with minimal code
- Registers available widgets
- Renders using built-in components

## Examples

See `examples/a2ui_agent_example.py` for complete examples.

## Migration Guide

### Before (Plain Text)

```python
def my_agent(goal):
    return "Task 1: In Progress\nTask 2: Completed"
```

### After (A2UI Widgets)

```python
from jotty.core.ui.a2ui import format_task_list

def my_agent(goal):
    tasks = [
        {"title": "Task 1", "status": "in_progress"},
        {"title": "Task 2", "status": "completed"}
    ]
    return format_task_list(tasks)
```

**Benefits:**
- Rich UI rendering (cards, status badges, icons)
- Better user experience
- Structured data (easier to parse/filter)
- Backward compatible (plain text still works)

## Backward Compatibility

✅ **Plain text responses still work!** If an agent returns plain text, it's automatically rendered as markdown.

✅ **Gradual migration:** Update agents one at a time to use A2UI.
