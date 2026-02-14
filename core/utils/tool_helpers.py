"""
Tool Helpers
============

Standardized helpers for skill tool functions.
Reduces boilerplate validation and error handling.

Usage:
    from Jotty.core.utils.tool_helpers import (
        tool_response, tool_error, validate_params, require_params
    )

    def my_tool(params: Dict[str, Any]) -> Dict[str, Any]:
        # Validate required params
        error = require_params(params, ['channel', 'message'])
        if error:
            return error

        # Do work...
        return tool_response(data={'id': '123'})
"""

import logging
import functools
from typing import Dict, Any, List, Optional, Callable

logger = logging.getLogger(__name__)

# =============================================================================
# PARAM ALIAS RESOLUTION (Single source of truth)
# =============================================================================
# Merges the centralized DEFAULT_PARAM_ALIASES with tool-specific aliases.
# Both tool_wrapper() and async_tool_wrapper() delegate here — no duplication.

try:
    from Jotty.core.foundation.data_structures import DEFAULT_PARAM_ALIASES
except ImportError:
    DEFAULT_PARAM_ALIASES = {}

# Tool-specific aliases that LLM planners commonly produce.
# Merged with DEFAULT_PARAM_ALIASES at module load.
_TOOL_PARAM_ALIASES: Dict[str, List[str]] = {
    'path': ['file_path', 'filepath', 'filename', 'file_name', 'file'],
    'content': ['text', 'data', 'body', 'file_content'],
    'command': ['cmd', 'shell_command', 'shell_cmd'],
    'script': ['script_content', 'script_code', 'code', 'python_code', 'script_path'],
    'query': ['search_query', 'q', 'search', 'question'],
    'url': ['link', 'href', 'website', 'page_url'],
    'message': ['msg', 'text_message'],
    'timeout': ['time_limit', 'max_time'],
}

# Merge: tool-specific aliases take priority, DEFAULT_PARAM_ALIASES fill gaps.
_PARAM_ALIASES: Dict[str, List[str]] = {**DEFAULT_PARAM_ALIASES, **_TOOL_PARAM_ALIASES}


def _normalize_param_aliases(params: Dict[str, Any], required_params: List[str]) -> None:
    """Resolve param aliases in-place for required params.

    Mutates *params*: if a required canonical key is missing but an alias
    is present, the alias is popped and assigned to the canonical key.
    """
    for canonical, aliases in _PARAM_ALIASES.items():
        if canonical in required_params and canonical not in params:
            for alias in aliases:
                if alias in params:
                    params[canonical] = params.pop(alias)
                    break


def tool_response(data: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Dict[str, Any]:
    """
    Create a successful tool response.

    Args:
        data: Optional dict of response data
        **kwargs: Additional fields to include

    Returns:
        Dict with success=True and provided data
    """
    result = {"success": True}
    if data:
        result.update(data)
    result.update(kwargs)
    return result


def tool_error(error: str, code: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
    """
    Create an error tool response.

    Args:
        error: Error message
        code: Optional error code
        **kwargs: Additional fields

    Returns:
        Dict with success=False and error details
    """
    result = {"success": False, "error": error}
    if code:
        result["code"] = code
    result.update(kwargs)
    return result


def require_params(
    params: Dict[str, Any],
    required: List[str]
) -> Optional[Dict[str, Any]]:
    """
    Validate required parameters.

    Args:
        params: Parameters dict
        required: List of required parameter names

    Returns:
        Error dict if validation fails, None if valid
    """
    for param in required:
        if not params.get(param):
            return tool_error(f"{param} parameter is required")
    return None


def validate_params(
    params: Dict[str, Any],
    schema: Dict[str, Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """
    Validate parameters against schema.

    Args:
        params: Parameters dict
        schema: Validation schema like:
            {
                'channel': {'required': True, 'type': str},
                'limit': {'required': False, 'type': int, 'max': 100}
            }

    Returns:
        Error dict if validation fails, None if valid
    """
    for name, rules in schema.items():
        value = params.get(name)

        # Required check
        if rules.get('required', False) and value is None:
            return tool_error(f"{name} parameter is required")

        if value is None:
            continue

        # Type check
        expected_type = rules.get('type')
        if expected_type and not isinstance(value, expected_type):
            return tool_error(f"{name} must be {expected_type.__name__}")

        # Min/max for numbers
        if isinstance(value, (int, float)):
            if 'min' in rules and value < rules['min']:
                return tool_error(f"{name} must be at least {rules['min']}")
            if 'max' in rules and value > rules['max']:
                return tool_error(f"{name} must be at most {rules['max']}")

        # Length for strings
        if isinstance(value, str):
            if 'max_length' in rules and len(value) > rules['max_length']:
                return tool_error(f"{name} cannot exceed {rules['max_length']} characters")

    return None


def tool_wrapper(
    required_params: Optional[List[str]] = None,
    log_errors: bool = True
) -> Callable:
    """
    Decorator for tool functions with automatic error handling.

    Args:
        required_params: List of required parameter names
        log_errors: Whether to log exceptions

    Usage:
        @tool_wrapper(required_params=['channel', 'message'])
        def send_message_tool(params: Dict[str, Any]) -> Dict[str, Any]:
            # No need for try/except or param validation
            channel = params['channel']
            message = params['message']
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(params: Dict[str, Any]) -> Dict[str, Any]:
            try:
                if required_params:
                    _normalize_param_aliases(params, required_params)

                # Validate required params
                if required_params:
                    error = require_params(params, required_params)
                    if error:
                        return error

                return func(params)

            except Exception as e:
                if log_errors:
                    logger.error(f"{func.__name__} error: {e}", exc_info=True)
                return tool_error(f"Failed to execute {func.__name__}: {str(e)}")

        # Stash required_params for ToolSchema introspection
        wrapper._required_params = required_params or []
        return wrapper
    return decorator


def async_tool_wrapper(
    required_params: Optional[List[str]] = None,
    log_errors: bool = True
) -> Callable:
    """
    Decorator for async tool functions with automatic error handling.

    Same as tool_wrapper but for async functions.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(params: Dict[str, Any]) -> Dict[str, Any]:
            try:
                if required_params:
                    _normalize_param_aliases(params, required_params)

                if required_params:
                    error = require_params(params, required_params)
                    if error:
                        return error

                return await func(params)

            except Exception as e:
                if log_errors:
                    logger.error(f"{func.__name__} error: {e}", exc_info=True)
                return tool_error(f"Failed to execute {func.__name__}: {str(e)}")

        # Stash required_params for ToolSchema introspection
        wrapper._required_params = required_params or []
        return wrapper
    return decorator
