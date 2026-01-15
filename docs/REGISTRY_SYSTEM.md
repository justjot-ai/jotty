# Jotty Tools and Widgets Registry System

## Overview

Jotty now has a **generic, reusable registry system** for managing AI tools and UI widgets. This system allows any project using Jotty to:

1. Register tools (MCP tools, AI capabilities)
2. Register widgets (UI components, section types)
3. Discover available tools/widgets via API
4. Validate tool/widget selections
5. Get default enabled sets

## Architecture

```
Jotty/core/registry/
├── __init__.py              # Main exports
├── widget_registry.py       # Widget (section type) registry
├── tools_registry.py        # Tools (MCP tools) registry
├── unified_registry.py      # Combined interface
├── justjot_loader.py        # Loader for JustJot.ai data
├── api.py                   # API handler (for web frameworks)
└── load_from_json.py        # CLI loader script
```

## Usage

### Python (Jotty)

```python
from Jotty.core.registry import UnifiedRegistry, get_unified_registry

# Get registry
registry = get_unified_registry()

# Register a tool
registry.tools.register(
    name='my_tool',
    description='Does something useful',
    category='general',
    mcp_enabled=True,
)

# Register a widget
registry.widgets.register(
    value='my_widget',
    label='My Widget',
    icon='📦',
    description='A useful widget',
    category='Content',
)

# Get all tools and widgets
all_data = registry.get_all()

# Validate selections
valid_tools = registry.validate_tools(['my_tool', 'other_tool'])
valid_widgets = registry.validate_widgets(['my_widget'])
```

### TypeScript/Next.js (JustJot.ai)

```typescript
// Fetch from Jotty registry API
const response = await fetch('/api/jotty/registry');
const data = await response.json();

if (data.success) {
  const { tools, widgets } = data.data;
  
  // Use tools
  const availableTools = tools.available;
  const toolCategories = tools.categories;
  
  // Use widgets
  const availableWidgets = widgets.available;
  const widgetCategories = widgets.categories;
}
```

## Integration with JustJot.ai

JustJot.ai automatically loads its tools and widgets into Jotty's registry via:

1. **API Endpoint**: `/api/jotty/registry` - Serves tools/widgets in Jotty format
2. **Auto-loading**: On API call, data is loaded into Jotty's Python registry
3. **Generic Components**: `SupervisorChatWidget` now uses Jotty registry

## Benefits

1. **Generic**: Works across any project using Jotty
2. **Extensible**: Easy to add new tools/widgets
3. **Discoverable**: API endpoints for discovery
4. **Validatable**: Built-in validation
5. **Reusable**: Same registry can be used by multiple projects

## Future Enhancements

- [ ] MCP tool auto-discovery
- [ ] Widget auto-discovery from section registry
- [ ] Tool/widget usage analytics
- [ ] Dynamic tool/widget loading
- [ ] Multi-project registry aggregation
