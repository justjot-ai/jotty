#!/usr/bin/env python3
"""
Widget Parameter Schema System
================================

Defines parameter schemas for widgets using JSON Schema format.
Enables:
- Auto-discovery of valid parameter values
- Runtime validation
- Auto-generated tool docstrings
- Type checking and enum constraints

**Architecture:**
- WidgetParamSchema: Defines parameters a widget accepts
- Validator: Validates params at runtime
- DocGenerator: Generates tool docstrings from schemas

**Example:**
```python
from Jotty.core.metadata import WidgetParamSchema, create_widget_with_params

# Define parameter schema
task_list_params = WidgetParamSchema(
    properties={
        "status": {
            "type": "string",
            "enum": ["backlog", "pending", "in_progress", "completed", "failed"],
            "description": "Task status filter",
            "examples": [
                {"value": "backlog", "description": "Tasks queued but not started (user's Task List list)"},
                {"value": "pending", "description": "Tasks ready to start"},
                {"value": "in_progress", "description": "Currently running tasks"},
                {"value": "completed", "description": "Finished tasks"},
                {"value": "failed", "description": "Failed tasks"}
            ]
        },
        "limit": {
            "type": "integer",
            "default": 100,
            "minimum": 1,
            "maximum": 1000,
            "description": "Maximum number of tasks to return"
        }
    },
    required=[]  # All params optional
)

# Create widget with param schema
widget = create_widget_with_params(
    widget_id="task_list",
    param_schema=task_list_params,
    ...
)

# Agent gets auto-generated docstring:
# \"\"\"
# Display task list widget.
#
# Parameters:
#   status (string, optional): Task status filter
#     Options:
#       - "backlog": Tasks queued but not started (user's Task List list)
#       - "pending": Tasks ready to start
#       - "in_progress": Currently running tasks
#       - "completed": Finished tasks
#       - "failed": Failed tasks
#   limit (integer, optional, default=100): Maximum number of tasks (1-1000)
# \"\"\"
```
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import json


@dataclass
class WidgetParamSchema:
    """
    JSON Schema for widget parameters.

    Defines what parameters a widget accepts, their types,
    valid values (enums), defaults, and documentation.
    """
    properties: Dict[str, Any] = field(default_factory=dict)
    required: List[str] = field(default_factory=list)

    def validate(self, params: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validate parameters against schema.

        Returns:
            (is_valid, error_message)
        """
        # Check required params
        for req_param in self.required:
            if req_param not in params:
                return False, f"Missing required parameter: {req_param}"

        # Validate each param
        for param_name, param_value in params.items():
            if param_name not in self.properties:
                # Allow unknown params (forward compatibility)
                continue

            param_schema = self.properties[param_name]

            # Type validation
            expected_type = param_schema.get("type")
            if expected_type:
                if not self._check_type(param_value, expected_type):
                    return False, f"Parameter '{param_name}' must be {expected_type}, got {type(param_value).__name__}"

            # Enum validation
            if "enum" in param_schema:
                if param_value not in param_schema["enum"]:
                    valid_values = ", ".join(f"'{v}'" for v in param_schema["enum"])
                    return False, f"Parameter '{param_name}' must be one of: {valid_values}. Got: '{param_value}'"

            # Range validation (for numbers)
            if expected_type in ["integer", "number"]:
                if "minimum" in param_schema and param_value < param_schema["minimum"]:
                    return False, f"Parameter '{param_name}' must be >= {param_schema['minimum']}"
                if "maximum" in param_schema and param_value > param_schema["maximum"]:
                    return False, f"Parameter '{param_name}' must be <= {param_schema['maximum']}"

        return True, None

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected JSON Schema type."""
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None)
        }

        expected_py_type = type_map.get(expected_type)
        if expected_py_type is None:
            return True  # Unknown type, allow it

        return isinstance(value, expected_py_type)

    def get_default_value(self, param_name: str) -> Any:
        """Get default value for parameter, if defined."""
        if param_name not in self.properties:
            return None
        return self.properties[param_name].get("default")

    def apply_defaults(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply default values to params dict."""
        result = params.copy()
        for param_name, param_schema in self.properties.items():
            if param_name not in result and "default" in param_schema:
                result[param_name] = param_schema["default"]
        return result


def generate_param_docstring(param_schema: WidgetParamSchema, widget_id: str) -> str:
    """
    Generate human-readable docstring from parameter schema.

    Creates clear documentation that agents can understand,
    including enum values with descriptions.
    """
    lines = [
        f"Display {widget_id.replace('_', ' ')} widget.",
        "",
        "Parameters:"
    ]

    if not param_schema.properties:
        lines.append("  None (no parameters required)")
        return "\n".join(lines)

    for param_name, param_def in param_schema.properties.items():
        # Build param signature
        param_type = param_def.get("type", "any")
        is_required = param_name in param_schema.required
        required_str = "required" if is_required else "optional"

        default_val = param_def.get("default")
        default_str = f", default={json.dumps(default_val)}" if default_val is not None else ""

        param_desc = param_def.get("description", "")
        lines.append(f"  {param_name} ({param_type}, {required_str}{default_str}): {param_desc}")

        # Add enum values with descriptions
        if "enum" in param_def:
            lines.append("    Options:")
            examples = param_def.get("examples", [])
            example_dict = {ex["value"]: ex.get("description", "") for ex in examples}

            for enum_val in param_def["enum"]:
                enum_desc = example_dict.get(enum_val, "")
                desc_str = f": {enum_desc}" if enum_desc else ""
                lines.append(f'      - "{enum_val}"{desc_str}')

        # Add range constraints
        if param_type in ["integer", "number"]:
            constraints = []
            if "minimum" in param_def:
                constraints.append(f"min={param_def['minimum']}")
            if "maximum" in param_def:
                constraints.append(f"max={param_def['maximum']}")
            if constraints:
                lines.append(f"    Range: {', '.join(constraints)}")

    return "\n".join(lines)


def generate_tool_examples(param_schema: WidgetParamSchema, widget_id: str) -> List[str]:
    """
    Generate example tool calls from parameter schema.

    Returns list of example strings like:
    - 'User says "show backlog tasks" → render_widget_tool(widget_id="task_list", params=\'{"status": "backlog"}\')'
    """
    examples = []

    # For each enum parameter, generate examples
    for param_name, param_def in param_schema.properties.items():
        if "enum" not in param_def:
            continue

        param_examples = param_def.get("examples", [])
        for example in param_examples:
            enum_val = example["value"]
            description = example.get("description", "")

            # Generate natural language trigger
            if "user_query" in example:
                trigger = example["user_query"]
            else:
                # Auto-generate from description
                trigger = f"show {description.lower()}" if description else f"{param_name}={enum_val}"

            # Generate tool call
            params_json = json.dumps({param_name: enum_val})
            tool_call = f'render_widget_tool(widget_id="{widget_id}", params=\'{params_json}\')'

            examples.append(f'User says "{trigger}" → {tool_call}')

    # Add default example (no params)
    examples.insert(0, f'User says "show {widget_id.replace("_", " ")}" → render_widget_tool(widget_id="{widget_id}")')

    return examples


# =============================================================================
# Standard Parameter Schemas (Reusable Templates)
# =============================================================================

# Status filter parameter (common for task/workflow widgets)
STATUS_PARAM_SCHEMA = {
    "status": {
        "type": "string",
        "enum": ["backlog", "pending", "in_progress", "completed", "failed"],
        "description": "Filter by task status",
        "examples": [
            {
                "value": "backlog",
                "description": "Tasks queued but not started (user's Task List list)",
                "user_query": "show backlog tasks"
            },
            {
                "value": "pending",
                "description": "Tasks ready to start (waiting for orchestrator)",
                "user_query": "show pending tasks"
            },
            {
                "value": "in_progress",
                "description": "Currently running tasks",
                "user_query": "show tasks in progress"
            },
            {
                "value": "completed",
                "description": "Successfully finished tasks",
                "user_query": "show completed tasks"
            },
            {
                "value": "failed",
                "description": "Tasks that encountered errors",
                "user_query": "show failed tasks"
            }
        ]
    }
}

# Limit parameter (common for lists)
LIMIT_PARAM_SCHEMA = {
    "limit": {
        "type": "integer",
        "default": 100,
        "minimum": 1,
        "maximum": 1000,
        "description": "Maximum number of items to return"
    }
}

# Time range parameter (common for analytics)
TIME_RANGE_PARAM_SCHEMA = {
    "time_range": {
        "type": "string",
        "enum": ["today", "week", "month", "year", "all"],
        "default": "week",
        "description": "Time range filter",
        "examples": [
            {"value": "today", "description": "Today's data only", "user_query": "show today's data"},
            {"value": "week", "description": "Last 7 days", "user_query": "show this week"},
            {"value": "month", "description": "Last 30 days", "user_query": "show this month"},
            {"value": "year", "description": "Last 365 days", "user_query": "show this year"},
            {"value": "all", "description": "All time data", "user_query": "show all data"}
        ]
    }
}

# Sort parameter (common for lists)
SORT_PARAM_SCHEMA = {
    "sort_by": {
        "type": "string",
        "enum": ["date", "priority", "status", "name"],
        "default": "date",
        "description": "Sort field"
    },
    "sort_order": {
        "type": "string",
        "enum": ["asc", "desc"],
        "default": "desc",
        "description": "Sort order (ascending or descending)"
    }
}
