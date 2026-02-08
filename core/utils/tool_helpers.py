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


def tool_response(
    data: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
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


def tool_error(
    error: str,
    code: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
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
                    error = require_params(params, required_params)
                    if error:
                        return error

                return await func(params)

            except Exception as e:
                if log_errors:
                    logger.error(f"{func.__name__} error: {e}", exc_info=True)
                return tool_error(f"Failed to execute {func.__name__}: {str(e)}")

        return wrapper
    return decorator
