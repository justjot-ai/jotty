"""
Helpful Error Messages - Make debugging delightful
===================================================

Enhanced exceptions with:
- Clear error messages
- Suggested fixes
- Context about what went wrong
- Links to documentation

Usage:
    from core.foundation.helpful_errors import raise_import_error, raise_config_error

    raise_import_error(
        "SwarmConfig",
        suggestion="Use 'SwarmBaseConfig' instead. See: Jotty/CLAUDE.md"
    )
"""

from typing import Any, Dict, List, Optional

from .exceptions import JottyError


class HelpfulError(JottyError):
    """Base for errors with helpful suggestions."""

    def __init__(
        self,
        message: str,
        suggestion: Optional[str] = None,
        doc_link: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        self.suggestion = suggestion
        self.doc_link = doc_link

        # Build helpful message
        parts = [message]

        if suggestion:
            parts.append(f"\n💡 Suggestion: {suggestion}")

        if doc_link:
            parts.append(f"\n📖 Docs: {doc_link}")

        full_message = "\n".join(parts)
        super().__init__(full_message, context=context, **kwargs)


# =============================================================================
# IMPORT ERRORS
# =============================================================================


class SwarmConfigImportError(HelpfulError):
    """Raised when trying to import deprecated SwarmConfig."""

    def __init__(self, attempted_import: str = "SwarmConfig") -> None:
        super().__init__(
            message=f"Cannot import '{attempted_import}' - this class has been renamed",
            suggestion="Use 'SwarmBaseConfig' instead:\n"
            "  from ..swarm_types import SwarmConfig\n"
            "  class MyConfig(SwarmConfig): ...",
            doc_link="Jotty/CLAUDE.md - Legacy Imports section",
        )


def raise_import_error(
    module: str, name: str, suggestion: Optional[str] = None, **kwargs: Any
) -> None:
    """Raise a helpful import error."""

    # Check for common mistakes
    if name == "SwarmConfig":
        raise SwarmConfigImportError(name)

    message = f"Cannot import '{name}' from '{module}'"

    if not suggestion:
        suggestion = (
            f"Check that:\n"
            f"  1. The module '{module}' exists\n"
            f"  2. '{name}' is defined in that module\n"
            f"  3. You're using the correct import path"
        )

    raise HelpfulError(message, suggestion=suggestion, **kwargs)


# =============================================================================
# CONFIGURATION ERRORS
# =============================================================================


class MissingEnvVarError(HelpfulError):
    """Raised when required environment variable is missing."""

    def __init__(self, var_name: str, purpose: Optional[str] = None) -> None:
        message = f"Required environment variable '{var_name}' is not set"

        suggestion = (
            f"Set {var_name} in your environment:\n" f"  export {var_name}='your-value-here'\n"
        )

        if purpose:
            suggestion += f"\nThis variable is used for: {purpose}"

        # Add common examples
        if "TELEGRAM" in var_name:
            suggestion += "\n\nGet a Telegram bot token from @BotFather"
        elif "OPENAI" in var_name or "API_KEY" in var_name:
            suggestion += "\n\nGet an API key from your provider's dashboard"

        super().__init__(message, suggestion=suggestion)


class InvalidConfigValueError(HelpfulError):
    """Raised when config value is invalid."""

    def __init__(
        self, field: str, value: Any, expected: str, valid_values: Optional[List[str]] = None
    ) -> None:
        message = f"Invalid value for '{field}': {value}"

        suggestion = f"Expected: {expected}"
        if valid_values:
            suggestion += f"\nValid values: {', '.join(map(str, valid_values))}"

        super().__init__(
            message,
            suggestion=suggestion,
            context={"field": field, "value": value, "expected": expected},
        )


def raise_config_error(
    field: str, issue: str, suggestion: Optional[str] = None, **kwargs: Any
) -> None:
    """Raise a helpful configuration error."""

    message = f"Configuration error in '{field}': {issue}"

    if not suggestion:
        suggestion = "Check your config and ensure all required fields are set correctly"

    raise HelpfulError(message, suggestion=suggestion, **kwargs)


# =============================================================================
# EXECUTION ERRORS
# =============================================================================


class TimeoutErrorWithSuggestion(HelpfulError):
    """Timeout with suggestions for fixing."""

    def __init__(self, operation: str, timeout: int, actual_time: Optional[float] = None) -> None:
        if actual_time:
            message = f"{operation} timed out after {actual_time:.1f}s (limit: {timeout}s)"
        else:
            message = f"{operation} exceeded timeout of {timeout}s"

        suggestion = (
            f"Options to fix:\n"
            f"  1. Increase timeout: config.timeout = {timeout * 2}\n"
            f"  2. Reduce input size if processing large content\n"
            f"  3. Use a faster model (e.g., 'haiku' instead of 'sonnet')"
        )

        super().__init__(
            message, suggestion=suggestion, context={"operation": operation, "timeout": timeout}
        )


class LLMError(HelpfulError):
    """LLM call failed with suggestions."""

    def __init__(self, provider: str, error: str, retry_count: int = 0) -> None:
        message = f"LLM call to {provider} failed: {error}"

        suggestion = "Common fixes:\n"

        if "API key" in error or "authentication" in error.lower():
            suggestion += (
                f"  1. Check {provider.upper()}_API_KEY is set correctly\n"
                f"  2. Verify API key is valid and not expired\n"
                f"  3. Check you have credits/quota remaining"
            )
        elif "rate limit" in error.lower():
            suggestion += (
                "  1. Wait a moment and retry\n"
                "  2. Reduce concurrent requests\n"
                "  3. Upgrade your API plan"
            )
        elif "timeout" in error.lower():
            suggestion += (
                "  1. Increase timeout setting\n"
                "  2. Reduce input size\n"
                "  3. Try again (may be temporary network issue)"
            )
        else:
            suggestion += (
                f"  1. Check your internet connection\n"
                f"  2. Verify the {provider} API is operational\n"
                f"  3. Review the error message for specific details"
            )

        if retry_count > 0:
            message += f" (after {retry_count} retries)"

        super().__init__(
            message,
            suggestion=suggestion,
            context={"provider": provider, "error": error, "retries": retry_count},
        )


class JSONParseError(HelpfulError):
    """JSON parsing failed with helpful context."""

    def __init__(
        self, source: str, preview: str, original_error: Optional[Exception] = None
    ) -> None:
        message = f"Failed to parse JSON from {source}"

        suggestion = (
            f"Response preview: {preview[:200]}...\n\n"
            f"Common causes:\n"
            f"  1. LLM returned text instead of JSON\n"
            f"  2. JSON is malformed (missing quotes, commas, etc.)\n"
            f"  3. Response contains markdown code blocks\n\n"
            f"Fix: Ensure your prompt clearly requests JSON format"
        )

        super().__init__(
            message,
            suggestion=suggestion,
            original_error=original_error,
            context={"source": source, "preview": preview},
        )


# =============================================================================
# SWARM-SPECIFIC ERRORS
# =============================================================================


class SwarmNotFoundError(HelpfulError):
    """Swarm not found with suggestions."""

    def __init__(self, swarm_name: str, available_swarms: Optional[List[str]] = None) -> None:
        message = f"Swarm '{swarm_name}' not found"

        suggestion = "Available swarms:\n"
        if available_swarms:
            for s in available_swarms:
                suggestion += f"  - {s}\n"
        else:
            suggestion += "  Run: python -m Jotty.cli to see available swarms"

        suggestion += "\nOr check Jotty/CLAUDE.md for the full list"

        super().__init__(message, suggestion=suggestion)


class AgentFailedError(HelpfulError):
    """Agent execution failed with debugging tips."""

    def __init__(self, agent_name: str, task: str, error: str, trace: Optional[str] = None) -> None:
        message = f"Agent '{agent_name}' failed: {error}"

        suggestion = (
            "Debugging steps:\n"
            "  1. Check logs for detailed error messages\n"
            "  2. Verify agent has access to required tools\n"
            "  3. Ensure task is clear and achievable\n"
            "  4. Try running with a simpler version of the task"
        )

        if trace:
            suggestion += f"\n\nExecution trace:\n{trace[:500]}..."

        super().__init__(
            message, suggestion=suggestion, context={"agent": agent_name, "task": task}
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def suggest_fix(error: Exception, default_suggestion: str = "") -> str:
    """Generate helpful suggestion for any error."""

    error_str = str(error).lower()

    if "no module named" in error_str or "cannot import" in error_str:
        return "Run: pip install -r requirements.txt\nOr check the import path is correct"

    if "permission denied" in error_str:
        return "Check file permissions or run with appropriate access rights"

    if "no such file" in error_str:
        return "Verify the file path exists and is spelled correctly"

    if "connection" in error_str or "network" in error_str:
        return "Check your internet connection and try again"

    return default_suggestion or "Check the error message and logs for more details"


def format_error_with_context(error: Exception, operation: str, **context: Any) -> str:
    """Format error with helpful context."""

    lines = [
        f"❌ Error during: {operation}",
        f"   Type: {type(error).__name__}",
        f"   Message: {error}",
    ]

    if context:
        lines.append("   Context:")
        for key, value in context.items():
            lines.append(f"     - {key}: {value}")

    suggestion = suggest_fix(error)
    if suggestion:
        lines.append(f"\n💡 Suggestion: {suggestion}")

    return "\n".join(lines)
