"""
Parameter Validation Utilities
==============================

Provides reusable validation for skill tool parameters.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

logger = logging.getLogger(__name__)


from Jotty.core.infrastructure.foundation.exceptions import ValidationError  # noqa: F401


class ParamValidator:
    """
    Validates tool parameters with common validation rules.

    Usage:
        validator = ParamValidator(params)
        validator.require('arxiv_id', str)
        validator.optional('output_format', str, default='pdf', choices=['pdf', 'markdown', 'epub'])
        validator.validate_url('url')
        validator.validate_file_exists('file_path')

        # Get validated params
        validated = validator.validated

        # Or use context manager for auto error handling
        with ParamValidator(params) as v:
            v.require('arxiv_id')
    """

    # Common regex patterns
    PATTERNS = {
        "email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
        "url": r"^https?://[^\s]+$",
        "arxiv_id": r"^(\d{4}\.\d{4,5}(v\d+)?|[a-z-]+/\d{7})$",
        "youtube_url": r"^(https?://)?(www\.)?(youtube\.com|youtu\.be)/.+$",
        "kindle_email": r"^[a-zA-Z0-9._%+-]+@kindle\.com$",
    }

    def __init__(self, params: Dict[str, Any] = None) -> None:
        """
        Initialize validator with parameters.

        Args:
            params: Dictionary of parameters to validate
        """
        self.params = params or {}
        self.validated: Dict[str, Any] = {}
        self.errors: List[str] = []

    def __enter__(self) -> Any:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        # Don't suppress exceptions
        return False

    def require(self, name: str, param_type: Type = None, message: str = None) -> Any:
        """
        Require a parameter to be present and optionally validate its type.

        Args:
            name: Parameter name
            param_type: Expected type (str, int, float, bool, list, dict)
            message: Custom error message

        Returns:
            The validated value

        Raises:
            ValidationError: If parameter is missing or wrong type
        """
        value = self.params.get(name)

        if value is None:
            error_msg = message or f"'{name}' is required"
            self.errors.append(error_msg)
            raise ValidationError(error_msg, param=name)

        if param_type and not isinstance(value, param_type):
            error_msg = f"'{name}' must be {param_type.__name__}, got {type(value).__name__}"
            self.errors.append(error_msg)
            raise ValidationError(error_msg, param=name, value=value)

        self.validated[name] = value
        return value

    def optional(
        self, name: str, param_type: Type = None, default: Any = None, choices: List[Any] = None
    ) -> Any:
        """
        Get an optional parameter with type checking and default value.

        Args:
            name: Parameter name
            param_type: Expected type
            default: Default value if not provided
            choices: Valid choices for the value

        Returns:
            The value or default
        """
        value = self.params.get(name, default)

        if value is None:
            self.validated[name] = default
            return default

        if param_type and not isinstance(value, param_type):
            # Try to coerce if possible
            try:
                if param_type is bool:
                    value = str(value).lower() in ("true", "1", "yes")
                elif param_type is int:
                    value = int(value)
                elif param_type is float:
                    value = float(value)
                elif param_type is str:
                    value = str(value)
                else:
                    error_msg = f"'{name}' must be {param_type.__name__}"
                    self.errors.append(error_msg)
                    raise ValidationError(error_msg, param=name, value=value)
            except (ValueError, TypeError):
                error_msg = f"'{name}' must be {param_type.__name__}"
                self.errors.append(error_msg)
                raise ValidationError(error_msg, param=name, value=value)

        if choices and value not in choices:
            error_msg = f"'{name}' must be one of: {', '.join(str(c) for c in choices)}"
            self.errors.append(error_msg)
            raise ValidationError(error_msg, param=name, value=value)

        self.validated[name] = value
        return value

    def require_one_of(self, *names: str, message: str = None) -> str:
        """
        Require at least one of the specified parameters.

        Args:
            *names: Parameter names (at least one must be present)
            message: Custom error message

        Returns:
            Name of the first present parameter

        Raises:
            ValidationError: If none of the parameters are present
        """
        for name in names:
            if self.params.get(name) is not None:
                self.validated[name] = self.params[name]
                return name

        error_msg = message or f"At least one of {', '.join(names)} is required"
        self.errors.append(error_msg)
        raise ValidationError(error_msg)

    def validate_pattern(
        self, name: str, pattern: str, message: str = None, required: bool = True
    ) -> Optional[str]:
        """
        Validate a parameter against a regex pattern.

        Args:
            name: Parameter name
            pattern: Regex pattern (can be a key from PATTERNS or a custom pattern)
            message: Custom error message
            required: Whether the parameter is required

        Returns:
            The validated value or None
        """
        value = self.params.get(name)

        if value is None:
            if required:
                error_msg = f"'{name}' is required"
                self.errors.append(error_msg)
                raise ValidationError(error_msg, param=name)
            return None

        # Get pattern from PATTERNS if it's a key
        regex = self.PATTERNS.get(pattern, pattern)

        if not re.match(regex, str(value), re.IGNORECASE):
            error_msg = message or f"'{name}' has invalid format"
            self.errors.append(error_msg)
            raise ValidationError(error_msg, param=name, value=value)

        self.validated[name] = value
        return value

    def validate_url(self, name: str, required: bool = True) -> Optional[str]:
        """Validate a URL parameter."""
        return self.validate_pattern(
            name, "url", message=f"'{name}' must be a valid URL", required=required
        )

    def validate_email(self, name: str, required: bool = True) -> Optional[str]:
        """Validate an email parameter."""
        return self.validate_pattern(
            name, "email", message=f"'{name}' must be a valid email address", required=required
        )

    def validate_file_exists(self, name: str, required: bool = True) -> Optional[Path]:
        """
        Validate that a file path exists.

        Args:
            name: Parameter name
            required: Whether the parameter is required

        Returns:
            Path object or None
        """
        value = self.params.get(name)

        if value is None:
            if required:
                error_msg = f"'{name}' is required"
                self.errors.append(error_msg)
                raise ValidationError(error_msg, param=name)
            return None

        path = Path(value)
        if not path.exists():
            error_msg = f"File not found: {value}"
            self.errors.append(error_msg)
            raise ValidationError(error_msg, param=name, value=value)

        self.validated[name] = path
        return path

    def validate_dir_exists(
        self, name: str, create: bool = False, required: bool = False
    ) -> Optional[Path]:
        """
        Validate that a directory exists, optionally creating it.

        Args:
            name: Parameter name
            create: Create directory if it doesn't exist
            required: Whether the parameter is required

        Returns:
            Path object or None
        """
        value = self.params.get(name)

        if value is None:
            if required:
                error_msg = f"'{name}' is required"
                self.errors.append(error_msg)
                raise ValidationError(error_msg, param=name)
            return None

        path = Path(value)

        if not path.exists():
            if create:
                path.mkdir(parents=True, exist_ok=True)
            else:
                error_msg = f"Directory not found: {value}"
                self.errors.append(error_msg)
                raise ValidationError(error_msg, param=name, value=value)

        if not path.is_dir():
            error_msg = f"'{name}' is not a directory: {value}"
            self.errors.append(error_msg)
            raise ValidationError(error_msg, param=name, value=value)

        self.validated[name] = path
        return path

    def validate_range(
        self,
        name: str,
        min_val: Union[int, float] = None,
        max_val: Union[int, float] = None,
        required: bool = False,
        default: Union[int, float] = None,
    ) -> Optional[Union[int, float]]:
        """
        Validate a numeric parameter is within a range.

        Args:
            name: Parameter name
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            required: Whether the parameter is required
            default: Default value

        Returns:
            The validated number or default
        """
        value = self.params.get(name, default)

        if value is None:
            if required:
                error_msg = f"'{name}' is required"
                self.errors.append(error_msg)
                raise ValidationError(error_msg, param=name)
            return default

        try:
            num_value = float(value) if "." in str(value) else int(value)
        except (ValueError, TypeError):
            error_msg = f"'{name}' must be a number"
            self.errors.append(error_msg)
            raise ValidationError(error_msg, param=name, value=value)

        if min_val is not None and num_value < min_val:
            error_msg = f"'{name}' must be at least {min_val}"
            self.errors.append(error_msg)
            raise ValidationError(error_msg, param=name, value=value)

        if max_val is not None and num_value > max_val:
            error_msg = f"'{name}' must be at most {max_val}"
            self.errors.append(error_msg)
            raise ValidationError(error_msg, param=name, value=value)

        self.validated[name] = num_value
        return num_value

    def is_valid(self) -> bool:
        """Check if all validations passed."""
        return len(self.errors) == 0

    def get_errors(self) -> List[str]:
        """Get list of validation errors."""
        return self.errors


def validate_params(params: Dict[str, Any], schema: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate parameters against a schema.

    Args:
        params: Parameters to validate
        schema: Validation schema, e.g.:
            {
                'arxiv_id': {'required': True, 'type': str},
                'output_format': {'type': str, 'choices': ['pdf', 'markdown'], 'default': 'pdf'},
                'output_dir': {'type': str, 'dir_exists': True, 'create': True}
            }

    Returns:
        Validated parameters dict

    Raises:
        ValidationError: If validation fails
    """
    validator = ParamValidator(params)

    for name, rules in schema.items():
        required = rules.get("required", False)
        param_type = rules.get("type")
        default = rules.get("default")
        choices = rules.get("choices")

        if rules.get("file_exists"):
            validator.validate_file_exists(name, required=required)
        elif rules.get("dir_exists"):
            validator.validate_dir_exists(
                name, create=rules.get("create", False), required=required
            )
        elif rules.get("pattern"):
            validator.validate_pattern(name, rules["pattern"], required=required)
        elif rules.get("url"):
            validator.validate_url(name, required=required)
        elif rules.get("email"):
            validator.validate_email(name, required=required)
        elif rules.get("min") is not None or rules.get("max") is not None:
            validator.validate_range(
                name,
                min_val=rules.get("min"),
                max_val=rules.get("max"),
                required=required,
                default=default,
            )
        elif required:
            validator.require(name, param_type)
        else:
            validator.optional(name, param_type, default=default, choices=choices)

    return validator.validated
