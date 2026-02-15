"""
PlantUML Renderer Validation

Validates PlantUML diagrams by attempting to render them.
Handles HTTP 414 (URI Too Long) errors for large diagrams.
"""

import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Tuple

logger = logging.getLogger(__name__)

# Maximum safe URL size for GET requests (conservative estimate)
MAX_URL_SAFE_SIZE = 1500


def clean_plantuml_code(plantuml_code: str) -> str:
    """Remove markdown code fences from PlantUML code."""
    code = plantuml_code.strip()

    # Remove markdown fences
    if code.startswith("```"):
        lines = code.split("\n")
        # Remove first line if it's a fence
        if lines[0].startswith("```"):
            lines = lines[1:]
        # Remove last line if it's a fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines)

    return code.strip()


def validate_via_renderer(
    plantuml_code: str, timeout: int = 5, use_post: bool = True
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Validate PlantUML syntax by attempting to render it.

    Uses PlantUML server API to render diagrams.
    Handles HTTP 414 (URI Too Long) errors for large diagrams.

    Args:
        plantuml_code: PlantUML code to validate
        timeout: Request timeout in seconds
        use_post: Use POST for large diagrams (default: True)

    Returns:
        Tuple of (is_valid, error_message, metadata)
    """
    code = clean_plantuml_code(plantuml_code)

    if not code:
        return False, "Empty PlantUML code", {}

    # Check for basic PlantUML structure
    if "@startuml" not in code and "@start" not in code:
        return False, "Missing @startuml or @start tag", {}

    metadata = {
        "code_length": len(code),
        "lines": len(code.split("\n")),
        "method": None,
        "status_code": None,
    }

    # Encode PlantUML code
    try:
        encoded = urllib.parse.quote(code, safe="")
    except Exception as e:
        return False, f"Failed to encode PlantUML code: {e}", metadata

    # Determine if we should use POST (for large diagrams)
    should_use_post = use_post and len(encoded) > MAX_URL_SAFE_SIZE

    try:
        if should_use_post:
            # Use POST request for large diagrams
            metadata["method"] = "POST"
            url = "http://www.plantuml.com/plantuml/img"

            # POST request
            data = code.encode("utf-8")
            req = urllib.request.Request(url, data=data)
            req.add_header("Content-Type", "text/plain; charset=utf-8")
            req.add_header("User-Agent", "Jotty-PlantUMLValidator/1.0")

            try:
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    metadata["status_code"] = response.getcode()
                    if response.getcode() == 200:
                        return True, "", metadata
                    else:
                        return False, f"HTTP {response.getcode()}", metadata
            except urllib.error.HTTPError as e:
                metadata["status_code"] = e.code
                if e.code == 414:
                    # Still too long even with POST - use structure-based check
                    logger.warning("HTTP 414 even with POST, using structure-based validation")
                    return _structure_based_validation(code, metadata)
                return False, f"HTTP {e.code}: {e.reason}", metadata
        else:
            # Use GET request (smaller diagrams)
            metadata["method"] = "GET"
            url = f"http://www.plantuml.com/plantuml/img/{encoded}"

            req = urllib.request.Request(url)
            req.add_header("User-Agent", "Jotty-PlantUMLValidator/1.0")

            try:
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    metadata["status_code"] = response.getcode()
                    if response.getcode() == 200:
                        return True, "", metadata
                    else:
                        return False, f"HTTP {response.getcode()}", metadata
            except urllib.error.HTTPError as e:
                metadata["status_code"] = e.code
                if e.code == 414:
                    # URI Too Long - try POST instead
                    logger.info("HTTP 414: URI Too Long, trying POST request")
                    return validate_via_renderer(plantuml_code, timeout=timeout, use_post=True)
                return False, f"HTTP {e.code}: {e.reason}", metadata

    except urllib.error.URLError as e:
        logger.warning(f"Network error validating PlantUML: {e}")
        # Fallback to structure-based validation
        return _structure_based_validation(code, metadata)
    except Exception as e:
        logger.warning(f"Error validating PlantUML: {e}")
        # Fallback to structure-based validation
        return _structure_based_validation(code, metadata)


def _structure_based_validation(
    code: str, metadata: Dict[str, Any]
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Fallback validation based on structure when renderer fails.

    Checks:
    - Has @startuml/@start and @enduml/@end tags
    - Basic syntax structure
    - Balanced brackets/parentheses
    """
    errors = []

    # Check for start/end tags
    has_start = "@startuml" in code or "@start" in code.lower()
    has_end = "@enduml" in code or "@end" in code.lower()

    if not has_start:
        errors.append("Missing @startuml or @start tag")
    if not has_end:
        errors.append("Missing @enduml or @end tag")

    # Basic bracket balancing (simple check)
    open_braces = code.count("{")
    close_braces = code.count("}")
    if open_braces != close_braces:
        errors.append(f"Unbalanced braces: {open_braces} open, {close_braces} close")

    # Check for basic PlantUML syntax
    if has_start and has_end:
        # Extract content between tags
        start_idx = code.find("@startuml")
        if start_idx == -1:
            start_idx = code.lower().find("@start")
        end_idx = code.find("@enduml")
        if end_idx == -1:
            end_idx = code.lower().find("@end")

        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            content = code[start_idx:end_idx]
            if len(content.strip()) < 10:
                errors.append("PlantUML content too short")

    metadata["validation_method"] = "structure_based"
    metadata["errors"] = errors

    if errors:
        return False, "; ".join(errors), metadata
    else:
        return True, "", metadata


def validate_plantuml_syntax(
    plantuml_code: str, use_renderer: bool = True
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Validate PlantUML syntax.

    Args:
        plantuml_code: PlantUML code to validate
        use_renderer: Use renderer API (default: True) or structure-based only

    Returns:
        Tuple of (is_valid, error_message, metadata)
    """
    if use_renderer:
        try:
            return validate_via_renderer(plantuml_code, timeout=5)
        except Exception as e:
            logger.warning(f"Renderer validation failed: {e}, falling back to structure check")
            code = clean_plantuml_code(plantuml_code)
            return _structure_based_validation(code, {})
    else:
        code = clean_plantuml_code(plantuml_code)
        return _structure_based_validation(code, {})
