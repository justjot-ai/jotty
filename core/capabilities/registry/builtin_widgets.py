"""
Builtin Widgets - Supervisor-Specific Widget Definitions
=========================================================

Contains widget definitions for supervisor/orchestrator UI components.
These extend Jotty's standard widget library with domain-specific widgets.

These widgets are used by JustJot.ai's supervisor chat and can be
imported by any client that needs orchestrator/task management widgets.

Usage:
    from Jotty.core.capabilities.registry.builtin_widgets import (
        get_supervisor_widgets,
        SUPERVISOR_WIDGET_DEFINITIONS,
    )
"""

from typing import Dict, Any, List, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class SupervisorWidgetFactory:
    """
    Factory for creating supervisor-specific A2UI widget definitions.

    Provides widget definitions that can be used with Jotty's A2UIWidgetProvider
    or any compatible widget rendering system.
    """

    @staticmethod
    def create_task_list_widget() -> Dict[str, Any]:
        """
        Create task list widget definition.

        Displays a list of supervisor tasks with status and progress.
        """
        return {
            'id': 'task_list',
            'name': 'Task List',
            'description': 'List of supervisor tasks with status and progress',
            'category': 'Supervisor',
            'component_tree': [
                {
                    'id': 'card-1',
                    'component_type': 'Card',
                    'props': {
                        'title': 'Supervisor Tasks',
                        'subtitle': '{{task_count}} tasks total'
                    },
                    'children': ['list-1']
                },
                {
                    'id': 'list-1',
                    'component_type': 'List',
                    'props': {
                        'items': '{{tasks}}',
                        'itemTemplate': 'task-item-1'
                    }
                },
                {
                    'id': 'task-item-1',
                    'component_type': 'Row',
                    'props': {
                        'alignment': 'center',
                        'distribution': 'spaceBetween'
                    },
                    'children': ['task-text-1', 'task-status-1']
                },
                {
                    'id': 'task-text-1',
                    'component_type': 'Text',
                    'props': {
                        'value': '{{item.title}}',
                        'style': 'body'
                    }
                },
                {
                    'id': 'task-status-1',
                    'component_type': 'Text',
                    'props': {
                        'value': '{{item.status}}',
                        'style': 'caption'
                    }
                }
            ],
            'data_schema': {
                'type': 'object',
                'properties': {
                    'task_count': {'type': 'integer'},
                    'tasks': {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'properties': {
                                'id': {'type': 'string'},
                                'title': {'type': 'string'},
                                'status': {'type': 'string'}
                            }
                        }
                    }
                }
            },
            'param_schema': {
                'properties': {
                    'status': {
                        'type': 'string',
                        'description': 'Filter by task status',
                        'enum': ['pending', 'in_progress', 'completed', 'failed']
                    },
                    'limit': {
                        'type': 'integer',
                        'description': 'Maximum number of tasks to return',
                        'default': 100
                    }
                },
                'required': []
            },
            'example_data': {
                'task_count': 3,
                'tasks': [
                    {'id': 'TASK-001', 'title': 'Fix authentication bug', 'status': 'in_progress'},
                    {'id': 'TASK-002', 'title': 'Add dark mode', 'status': 'pending'},
                    {'id': 'TASK-003', 'title': 'Update documentation', 'status': 'completed'}
                ]
            },
            'tags': ['tasks', 'supervisor', 'list']
        }

    @staticmethod
    def create_stats_card_widget() -> Dict[str, Any]:
        """
        Create stats card widget definition.

        Displays key metrics with icons and values.
        """
        return {
            'id': 'stats_card',
            'name': 'Stats Card',
            'description': 'Display key metrics with icons and values',
            'category': 'Supervisor',
            'component_tree': [
                {
                    'id': 'card-1',
                    'component_type': 'Card',
                    'props': {
                        'title': '{{title}}',
                        'subtitle': '{{subtitle}}'
                    },
                    'children': ['row-1']
                },
                {
                    'id': 'row-1',
                    'component_type': 'Row',
                    'props': {
                        'distribution': 'spaceAround'
                    },
                    'children': ['stat-1', 'stat-2', 'stat-3']
                },
                {
                    'id': 'stat-1',
                    'component_type': 'Column',
                    'props': {
                        'alignment': 'center'
                    },
                    'children': ['stat-1-icon', 'stat-1-value', 'stat-1-label']
                },
                {
                    'id': 'stat-1-icon',
                    'component_type': 'Icon',
                    'props': {
                        'name': '{{stats[0].icon}}',
                        'size': 'large'
                    }
                },
                {
                    'id': 'stat-1-value',
                    'component_type': 'Text',
                    'props': {
                        'value': '{{stats[0].value}}',
                        'style': 'h3'
                    }
                },
                {
                    'id': 'stat-1-label',
                    'component_type': 'Text',
                    'props': {
                        'value': '{{stats[0].label}}',
                        'style': 'caption'
                    }
                },
                {
                    'id': 'stat-2',
                    'component_type': 'Text',
                    'props': {'value': '{{stats[1].value}}'}
                },
                {
                    'id': 'stat-3',
                    'component_type': 'Text',
                    'props': {'value': '{{stats[2].value}}'}
                }
            ],
            'data_schema': {
                'type': 'object',
                'properties': {
                    'title': {'type': 'string'},
                    'subtitle': {'type': 'string'},
                    'stats': {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'properties': {
                                'icon': {'type': 'string'},
                                'value': {'type': 'string'},
                                'label': {'type': 'string'}
                            }
                        }
                    }
                }
            },
            'example_data': {
                'title': 'Supervisor Metrics',
                'subtitle': 'Last 24 hours',
                'stats': [
                    {'icon': 'check_circle', 'value': '12', 'label': 'Completed'},
                    {'icon': 'pending', 'value': '5', 'label': 'In Progress'},
                    {'icon': 'error', 'value': '2', 'label': 'Failed'}
                ]
            },
            'tags': ['stats', 'metrics', 'supervisor']
        }

    @staticmethod
    def create_orchestrator_status_widget() -> Dict[str, Any]:
        """
        Create orchestrator status widget definition.

        Displays real-time orchestrator status and metrics.
        """
        return {
            'id': 'orchestrator_status',
            'name': 'Orchestrator Status',
            'description': 'Real-time orchestrator status and metrics',
            'category': 'Supervisor',
            'component_tree': [
                {
                    'id': 'card-1',
                    'component_type': 'Card',
                    'props': {
                        'title': 'Orchestrator Status',
                        'subtitle': '{{update_time}}'
                    },
                    'children': ['col-1']
                },
                {
                    'id': 'col-1',
                    'component_type': 'Column',
                    'children': ['status-row', 'metrics-row']
                },
                {
                    'id': 'status-row',
                    'component_type': 'Row',
                    'props': {
                        'alignment': 'center',
                        'distribution': 'spaceBetween'
                    },
                    'children': ['status-label', 'status-value']
                },
                {
                    'id': 'status-label',
                    'component_type': 'Text',
                    'props': {
                        'value': 'Status:',
                        'style': 'body'
                    }
                },
                {
                    'id': 'status-value',
                    'component_type': 'Text',
                    'props': {
                        'value': '{{status}}',
                        'style': 'h4'
                    }
                },
                {
                    'id': 'metrics-row',
                    'component_type': 'Column',
                    'children': ['metric-1', 'metric-2', 'metric-3']
                },
                {
                    'id': 'metric-1',
                    'component_type': 'Text',
                    'props': {
                        'value': 'Active Tasks: {{active_tasks}}',
                        'style': 'body'
                    }
                },
                {
                    'id': 'metric-2',
                    'component_type': 'Text',
                    'props': {
                        'value': 'Total Completed: {{completed_tasks}}',
                        'style': 'body'
                    }
                },
                {
                    'id': 'metric-3',
                    'component_type': 'Text',
                    'props': {
                        'value': 'Success Rate: {{success_rate}}%',
                        'style': 'body'
                    }
                }
            ],
            'data_schema': {
                'type': 'object',
                'properties': {
                    'status': {'type': 'string'},
                    'update_time': {'type': 'string'},
                    'active_tasks': {'type': 'integer'},
                    'completed_tasks': {'type': 'integer'},
                    'success_rate': {'type': 'number'}
                }
            },
            'example_data': {
                'status': 'Running',
                'update_time': '2 minutes ago',
                'active_tasks': 2,
                'completed_tasks': 45,
                'success_rate': 89.5
            },
            'tags': ['orchestrator', 'status', 'realtime']
        }

    @staticmethod
    def create_alert_card_widget() -> Dict[str, Any]:
        """
        Create alert card widget definition.

        Displays alerts and notifications with customizable styling.
        """
        return {
            'id': 'alert_card',
            'name': 'Alert Card',
            'description': 'Display alerts and notifications',
            'category': 'Supervisor',
            'component_tree': [
                {
                    'id': 'card-1',
                    'component_type': 'Card',
                    'props': {
                        'title': '{{title}}',
                        'backgroundColor': '{{color}}'
                    },
                    'children': ['icon-1', 'text-1']
                },
                {
                    'id': 'icon-1',
                    'component_type': 'Icon',
                    'props': {
                        'name': '{{icon}}',
                        'size': 'medium'
                    }
                },
                {
                    'id': 'text-1',
                    'component_type': 'Text',
                    'props': {
                        'value': '{{message}}',
                        'style': 'body'
                    }
                }
            ],
            'data_schema': {
                'type': 'object',
                'properties': {
                    'title': {'type': 'string'},
                    'message': {'type': 'string'},
                    'icon': {'type': 'string'},
                    'color': {'type': 'string'}
                }
            },
            'example_data': {
                'title': 'Deployment Complete',
                'message': 'Successfully deployed to production at 3:45 PM',
                'icon': 'check_circle',
                'color': '#4CAF50'
            },
            'tags': ['alert', 'notification', 'info']
        }


# Pre-built widget definitions list
SUPERVISOR_WIDGET_DEFINITIONS: List[Dict[str, Any]] = [
    SupervisorWidgetFactory.create_task_list_widget(),
    SupervisorWidgetFactory.create_stats_card_widget(),
    SupervisorWidgetFactory.create_orchestrator_status_widget(),
    SupervisorWidgetFactory.create_alert_card_widget(),
]


def get_supervisor_widgets() -> List[Dict[str, Any]]:
    """
    Get list of supervisor widget definitions.

    Returns:
        List of widget definition dictionaries
    """
    return SUPERVISOR_WIDGET_DEFINITIONS.copy()


def get_supervisor_widget_by_id(widget_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a specific supervisor widget by ID.

    Args:
        widget_id: Widget identifier (e.g., 'task_list', 'stats_card')

    Returns:
        Widget definition dict or None if not found
    """
    for widget in SUPERVISOR_WIDGET_DEFINITIONS:
        if widget.get('id') == widget_id:
            return widget.copy()
    return None


def register_supervisor_widgets_to_catalog(
    catalog: Dict[str, Any],
    widget_definition_class: type,
    component_class: type
) -> Dict[str, Any]:
    """
    Register supervisor widgets into an existing widget catalog.

    This function converts the plain dict definitions into proper
    WidgetDefinition objects using the provided classes.

    Args:
        catalog: Existing widget catalog dictionary
        widget_definition_class: WidgetDefinition class to instantiate
        component_class: A2UIComponent class to instantiate

    Returns:
        Updated catalog with supervisor widgets added

    Example:
        from Jotty.core.infrastructure.metadata import WidgetDefinition, A2UIComponent
        from Jotty.core.capabilities.registry.builtin_widgets import register_supervisor_widgets_to_catalog

        catalog = get_standard_widget_catalog()
        catalog = register_supervisor_widgets_to_catalog(
            catalog, WidgetDefinition, A2UIComponent
        )
    """
    for widget_def in SUPERVISOR_WIDGET_DEFINITIONS:
        widget_id = widget_def['id']

        # Convert component_tree dicts to A2UIComponent objects
        component_tree = []
        for comp_dict in widget_def.get('component_tree', []):
            component = component_class(
                id=comp_dict.get('id', ''),
                component_type=comp_dict.get('component_type', 'Text'),
                props=comp_dict.get('props', {}),
                children=comp_dict.get('children', [])
            )
            component_tree.append(component)

        # Create WidgetDefinition
        widget = widget_definition_class(
            id=widget_id,
            name=widget_def.get('name', widget_id),
            description=widget_def.get('description', ''),
            category=widget_def.get('category', 'Supervisor'),
            component_tree=component_tree,
            data_schema=widget_def.get('data_schema', {}),
            example_data=widget_def.get('example_data', {}),
            tags=widget_def.get('tags', [])
        )

        catalog[widget_id] = widget
        logger.debug(f"Registered supervisor widget: {widget_id}")

    logger.info(f"Registered {len(SUPERVISOR_WIDGET_DEFINITIONS)} supervisor widgets")
    return catalog


# Alert type mappings for alert_card widget data provider
ALERT_TYPE_MAPPINGS: Dict[str, Dict[str, str]] = {
    'success': {'color': '#4CAF50', 'icon': 'check_circle', 'title': 'Success'},
    'error': {'color': '#F44336', 'icon': 'error', 'title': 'Error'},
    'warning': {'color': '#FF9800', 'icon': 'warning', 'title': 'Warning'},
    'info': {'color': '#2196F3', 'icon': 'info', 'title': 'Info'}
}


def get_alert_config(alert_type: str) -> Dict[str, str]:
    """
    Get alert configuration by type.

    Args:
        alert_type: Alert type ('success', 'error', 'warning', 'info')

    Returns:
        Dict with 'color', 'icon', and 'title' keys
    """
    return ALERT_TYPE_MAPPINGS.get(alert_type, ALERT_TYPE_MAPPINGS['info'])
