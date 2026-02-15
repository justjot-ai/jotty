"""
Math LaTeX Renderer Validation

Validates Math LaTeX expressions using latex-validation library.
Falls back to QuickLaTeX API or structure-based validation.
"""

import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Try to import latex-validation library
# Note: openreview/latex-validation is a TypeScript/Node.js library
# It runs as an HTTP server or can be used via CLI
LATEX_VALIDATION_AVAILABLE = False
validate_latex = None

try:
    # Try Python package first (if someone creates a Python wrapper)
    try:
        from latex_validation import validate_latex  # type: ignore[import-not-found]

        LATEX_VALIDATION_AVAILABLE = True
        logger.info("latex-validation Python package available")
    except ImportError:
        # Try Node.js wrapper (HTTP server or CLI)
        try:
            from .latex_validator_wrapper import validate_latex_with_openreview

            LATEX_VALIDATION_AVAILABLE = True
            # Wrapper returns (is_valid, error_msg, metadata)
            validate_latex = lambda code: validate_latex_with_openreview(code)
            logger.info("latex-validation Node.js wrapper available (HTTP server or CLI)")
        except ImportError:
            LATEX_VALIDATION_AVAILABLE = False
            logger.debug("latex-validation not available, using fallback methods")
except Exception as e:
    LATEX_VALIDATION_AVAILABLE = False
    logger.debug(f"latex-validation not available: {e}, using fallback methods")

# Maximum safe URL size for GET requests
MAX_URL_SAFE_SIZE = 1500


def clean_latex_code(latex_code: str) -> str:
    """Remove markdown code fences and clean LaTeX code."""
    code = latex_code.strip()

    # Remove markdown fences
    if code.startswith("```"):
        lines = code.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines)

    return code.strip()


def validate_via_latex_validation_library(
    latex_code: str,
) -> Tuple[Optional[bool], str, Dict[str, Any]]:
    """
    Validate LaTeX using latex-validation library (preferred method).

    Supports:
    - Python package (if available)
    - Node.js wrapper (if Node.js and latex-validation installed)

    Args:
        latex_code: LaTeX code to validate

    Returns:
        Tuple of (is_valid, error_message, metadata)
        Returns (None, error, {}) if validation cannot be performed
    """
    if not LATEX_VALIDATION_AVAILABLE or validate_latex is None:
        return None, "latex-validation library not available", {}

    try:
        # Call validate_latex function (may be Python or Node.js wrapper)
        if callable(validate_latex):
            result = validate_latex(latex_code)
        else:
            return None, "validate_latex is not callable", {}

        metadata = {"validation_method": "latex-validation-library", "code_length": len(latex_code)}

        # Handle different result formats
        # Python package: may return bool or dict
        # Node.js wrapper: returns (is_valid, error_msg, metadata) or (None, error, {})
        if isinstance(result, tuple) and len(result) >= 2:
            # Node.js wrapper format: (is_valid, error_msg, metadata)
            is_valid, error_msg = result[0], result[1]
            if is_valid is None:
                # Validation cannot be performed (server not running, etc.)
                return None, error_msg, {}
            if len(result) > 2:
                metadata.update(result[2])
        elif isinstance(result, dict):
            is_valid = result.get("valid", False)
            error_msg = result.get("error", "") if not is_valid else ""
            if "warnings" in result:
                metadata["warnings"] = result["warnings"]
        elif isinstance(result, bool):
            is_valid = result
            error_msg = "" if is_valid else "LaTeX validation failed"
        elif isinstance(result, str):
            # String result - assume it's an error message if not empty
            is_valid = len(result.strip()) == 0
            error_msg = result if not is_valid else ""
        else:
            # Assume truthy means valid
            is_valid = bool(result)
            error_msg = "" if is_valid else "LaTeX validation failed"

        logger.debug(f"latex-validation library result: valid={is_valid}, error={error_msg[:50]}")
        return is_valid, error_msg, metadata

    except Exception as e:
        logger.warning(f"latex-validation library error: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        return None, f"latex-validation error: {e}", {}


def validate_via_renderer(
    latex_code: str, timeout: int = 5, use_post: bool = True, prefer_library: bool = True
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Validate Math LaTeX syntax by attempting to render it.

    Uses latex-validation library (preferred) or QuickLaTeX API.
    Handles HTTP 414 (URI Too Long) errors for large expressions.

    Args:
        latex_code: LaTeX code to validate
        timeout: Request timeout in seconds
        use_post: Use POST for large expressions (default: True)
        prefer_library: Use latex-validation library if available (default: True)

    Returns:
        Tuple of (is_valid, error_message, metadata)
    """
    code = clean_latex_code(latex_code)

    if not code:
        return False, "Empty LaTeX code", {}

    # Check for basic LaTeX structure
    has_math_delimiters = (
        "$" in code
        or "$$" in code
        or "\\[" in code
        or "\\(" in code
        or "\\begin{equation}" in code.lower()
        or "\\begin{align}" in code.lower()
    )

    if not has_math_delimiters:
        # Might be valid LaTeX without delimiters (e.g., just commands)
        # Check if it has LaTeX commands
        has_commands = bool(re.search(r"\\[a-zA-Z]+", code))
        if not has_commands:
            return False, "Missing math delimiters ($, $$, \\[, etc.) or LaTeX commands", {}

    metadata = {
        "code_length": len(code),
        "has_delimiters": has_math_delimiters,
        "method": None,
        "status_code": None,
    }

    # Try latex-validation library first (preferred)
    if prefer_library and LATEX_VALIDATION_AVAILABLE:
        library_result = validate_via_latex_validation_library(code)
        if library_result[0] is not None:  # Result is not None
            is_valid, error_msg, lib_metadata = library_result
            metadata.update(lib_metadata)
            if is_valid:
                return True, "", metadata
            # If library says invalid, still try renderer as fallback
            logger.debug(f"latex-validation library says invalid: {error_msg}, trying renderer")

    # Fallback to QuickLaTeX API (if library not available or failed)
    # Format: https://quicklatex.com/latex3.f?equation={encoded_latex}
    try:
        encoded = urllib.parse.quote(code, safe="")
        should_use_post = use_post and len(encoded) > MAX_URL_SAFE_SIZE

        if should_use_post:
            # Use POST request for large expressions
            metadata["method"] = "POST"
            url = "https://quicklatex.com/latex3.f"

            data = urllib.parse.urlencode({"equation": code}).encode("utf-8")
            req = urllib.request.Request(url, data=data)
            req.add_header("Content-Type", "application/x-www-form-urlencoded")
            req.add_header("User-Agent", "Jotty-MathLaTeXValidator/1.0")

            try:
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    metadata["status_code"] = response.getcode()
                    response_text = response.read().decode("utf-8")

                    # QuickLaTeX returns: 0 {image_url} {width} {height}
                    # 0 = success, non-zero = error
                    # Error codes: -1 = invalid LaTeX, -2 = timeout, etc.
                    if response_text.startswith("0"):
                        return True, "", metadata
                    else:
                        error_msg = (
                            response_text.split("\n")[0] if "\n" in response_text else response_text
                        )
                        # Error -1 might be API issue, fallback to structure validation
                        if error_msg.strip() == "-1":
                            logger.debug("QuickLaTeX returned -1, using structure-based validation")
                            return _structure_based_validation(code, metadata)
                        return False, f"QuickLaTeX error: {error_msg}", metadata
            except urllib.error.HTTPError as e:
                metadata["status_code"] = e.code
                if e.code == 414:
                    return _structure_based_validation(code, metadata)
                return False, f"HTTP {e.code}: {e.reason}", metadata
        else:
            # Use GET request (smaller expressions)
            metadata["method"] = "GET"
            url = f"https://quicklatex.com/latex3.f?equation={encoded}"

            req = urllib.request.Request(url)
            req.add_header("User-Agent", "Jotty-MathLaTeXValidator/1.0")

            try:
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    metadata["status_code"] = response.getcode()
                    response_text = response.read().decode("utf-8")

                    if response_text.startswith("0"):
                        return True, "", metadata
                    else:
                        error_msg = (
                            response_text.split("\n")[0] if "\n" in response_text else response_text
                        )
                        # Error -1 might be API issue, fallback to structure validation
                        if error_msg.strip() == "-1":
                            logger.debug("QuickLaTeX returned -1, using structure-based validation")
                            return _structure_based_validation(code, metadata)
                        return False, f"QuickLaTeX error: {error_msg}", metadata
            except urllib.error.HTTPError as e:
                metadata["status_code"] = e.code
                if e.code == 414:
                    # Try POST instead
                    logger.info("HTTP 414: URI Too Long, trying POST request")
                    return validate_via_renderer(latex_code, timeout=timeout, use_post=True)
                return False, f"HTTP {e.code}: {e.reason}", metadata

    except urllib.error.URLError as e:
        logger.warning(f"Network error validating LaTeX: {e}")
        # Fallback to structure-based validation
        return _structure_based_validation(code, metadata)
    except Exception as e:
        logger.warning(f"Error validating LaTeX: {e}")
        # Fallback to structure-based validation
        return _structure_based_validation(code, metadata)


def _structure_based_validation(
    code: str, metadata: Dict[str, Any]
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Fallback validation based on structure when renderer fails.

    Checks:
    - Has math delimiters or LaTeX commands
    - Balanced braces/parentheses
    - Valid LaTeX command syntax
    """
    errors = []

    # Check for math delimiters or LaTeX commands
    has_delimiters = (
        "$" in code or "$$" in code or "\\[" in code or "\\(" in code or "\\begin{" in code.lower()
    )
    has_commands = bool(re.search(r"\\[a-zA-Z]+", code))

    if not has_delimiters and not has_commands:
        errors.append("Missing math delimiters ($, $$, \\[, etc.) or LaTeX commands")

    # Check balanced braces
    open_braces = code.count("{")
    close_braces = code.count("}")
    if open_braces != close_braces:
        errors.append(f"Unbalanced braces: {open_braces} open, {close_braces} close")

    # Check balanced parentheses (in math context)
    open_parens = code.count("(")
    close_parens = code.count(")")
    if open_parens != close_parens:
        errors.append(f"Unbalanced parentheses: {open_parens} open, {close_parens} close")

    # Check for common LaTeX syntax errors
    # Unescaped special characters outside math mode
    if "$" not in code and "\\[" not in code:
        # Check for unescaped special chars that need escaping
        unescaped = re.findall(r"(?<!\\)[&%#]", code)
        if unescaped:
            errors.append(f"Unescaped special characters: {set(unescaped)}")

    # Check for valid LaTeX command syntax
    invalid_commands = re.findall(r"\\(?![a-zA-Z@]+)", code)
    if invalid_commands:
        errors.append("Invalid LaTeX command syntax (backslash not followed by valid command)")

    metadata["validation_method"] = "structure_based"
    metadata["errors"] = errors

    if errors:
        return False, "; ".join(errors), metadata
    else:
        return True, "", metadata


def validate_math_latex_syntax(
    latex_code: str, use_renderer: bool = True, prefer_library: bool = True
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Validate Math LaTeX syntax.

    Priority:
    1. latex-validation library (if available and prefer_library=True)
    2. QuickLaTeX API renderer
    3. Structure-based validation (fallback)

    Args:
        latex_code: LaTeX code to validate
        use_renderer: Use renderer/library (default: True) or structure-based only
        prefer_library: Use latex-validation library if available (default: True)

    Returns:
        Tuple of (is_valid, error_message, metadata)
    """
    if use_renderer:
        try:
            return validate_via_renderer(latex_code, timeout=5, prefer_library=prefer_library)
        except Exception as e:
            logger.warning(f"Renderer validation failed: {e}, falling back to structure check")
            code = clean_latex_code(latex_code)
            return _structure_based_validation(code, {})
    else:
        code = clean_latex_code(latex_code)
        return _structure_based_validation(code, {})
