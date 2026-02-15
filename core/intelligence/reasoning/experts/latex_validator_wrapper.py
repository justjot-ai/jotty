"""
LaTeX Validator Wrapper

Wrapper for openreview/latex-validation library.
The library is TypeScript/Node.js and runs as an HTTP server.

Usage:
1. Start server: `node dist/main.js run-server` (listens on localhost:8080)
2. POST to: http://localhost:8080/latex/fragment
3. Body: {"latex": "your latex code"}

This wrapper can:
1. Use HTTP API if server is running
2. Start server automatically (if configured)
3. Fall back to other validators
"""

import json
import logging
import os
import subprocess
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Check if Node.js is available
NODE_AVAILABLE = False
try:
    result = subprocess.run(["node", "--version"], capture_output=True, timeout=2)
    if result.returncode == 0:
        NODE_AVAILABLE = True
        logger.debug(f"Node.js available: {result.stdout.decode().strip()}")
except Exception:
    pass

# Server URL (centralized default, overridable via env var)
try:
    from Jotty.core.infrastructure.foundation.config_defaults import DEFAULTS as _DEFAULTS

    _DEFAULT_LATEX_URL = _DEFAULTS.LATEX_VALIDATION_URL
except ImportError:
    _DEFAULT_LATEX_URL = "http://localhost:8080"
LATEX_VALIDATION_SERVER_URL = os.getenv("LATEX_VALIDATION_SERVER_URL", _DEFAULT_LATEX_URL)


def validate_via_http_server(
    latex_code: str, server_url: Optional[str] = None
) -> Tuple[Optional[bool], str, Dict[str, Any]]:
    """
    Validate LaTeX using openreview/latex-validation HTTP server.

    The server should be running: `node dist/main.js run-server`
    Listens on localhost:8080 by default.

    Args:
        latex_code: LaTeX code to validate
        server_url: Server URL (default: http://localhost:8080)

    Returns:
        Tuple of (is_valid, error_message, metadata)
        Returns (None, error, {}) if validation cannot be performed
    """
    if server_url is None:
        server_url = LATEX_VALIDATION_SERVER_URL

    try:
        # Prepare POST request
        url = f"{server_url}/latex/fragment"
        data = json.dumps({"latex": latex_code}).encode("utf-8")

        req = urllib.request.Request(url, data=data)
        req.add_header("Content-Type", "application/json")
        req.add_header("User-Agent", "Jotty-LaTeXValidator/1.0")

        # Make request
        with urllib.request.urlopen(req, timeout=10) as response:
            response_data = json.loads(response.read().decode("utf-8"))

            # Response format: {"status": "ok"} or {"status": "error", "message": "..."}
            status = response_data.get("status", "error")
            is_valid = status == "ok"
            error_msg = response_data.get("message", "") if not is_valid else ""

            metadata = {
                "validation_method": "latex-validation-http-server",
                "code_length": len(latex_code),
                "server_url": server_url,
            }

            return is_valid, error_msg, metadata

    except urllib.error.URLError as e:
        logger.debug(f"LaTeX validation server not available at {server_url}: {e}")
        return None, f"Server not available: {e}", {}
    except Exception as e:
        logger.warning(f"Error using LaTeX validation HTTP server: {e}")
        return None, f"HTTP server error: {e}", {}


def validate_via_node_cli(
    latex_code: str, latex_validation_path: Optional[str] = None
) -> Tuple[Optional[bool], str, Dict[str, Any]]:
    """
    Validate LaTeX using openreview/latex-validation CLI directly.

    Requires:
    - Node.js installed
    - latex-validation built (npm run build)
    - tectonic installed

    Args:
        latex_code: LaTeX code to validate
        latex_validation_path: Path to latex-validation directory

    Returns:
        Tuple of (is_valid, error_message, metadata)
    """
    if not NODE_AVAILABLE:
        return None, "Node.js not available", {}

    # Try to find latex-validation
    if latex_validation_path is None:
        possible_paths = [
            os.path.expanduser("~/latex-validation"),
            "/tmp/latex-validation",
            os.path.join(os.path.dirname(__file__), "../../../latex-validation"),
        ]
        for path in possible_paths:
            if os.path.exists(path) and os.path.exists(os.path.join(path, "dist", "main.js")):
                latex_validation_path = path
                break

    if latex_validation_path is None or not os.path.exists(latex_validation_path):
        return None, "latex-validation not found (need to build with 'npm run build')", {}

    try:
        # Use CLI: node dist/main.js validate --fragment "latex" --latex-packages resources/latex-packages.txt
        packages_file = os.path.join(latex_validation_path, "resources", "latex-packages.txt")
        if not os.path.exists(packages_file):
            return None, "latex-packages.txt not found", {}

        result = subprocess.run(
            [
                "node",
                "dist/main.js",
                "validate",
                "--fragment",
                latex_code,
                "--latex-packages",
                packages_file,
            ],
            cwd=latex_validation_path,
            capture_output=True,
            text=True,
            timeout=15,
        )

        if result.returncode == 0:
            # Success - LaTeX is valid
            return (
                True,
                "",
                {"validation_method": "latex-validation-cli", "code_length": len(latex_code)},
            )
        else:
            # Error - extract error message
            error_msg = result.stderr.strip() or result.stdout.strip()
            return (
                False,
                error_msg,
                {"validation_method": "latex-validation-cli", "code_length": len(latex_code)},
            )

    except subprocess.TimeoutExpired:
        return None, "Validation timeout", {}
    except Exception as e:
        logger.warning(f"Error using LaTeX validation CLI: {e}")
        return None, f"CLI error: {e}", {}


def validate_via_python_latex_validator(
    latex_code: str,
) -> Tuple[Optional[bool], str, Dict[str, Any]]:
    """
    Validate LaTeX using Python-based validators.

    Tries alternative Python LaTeX validation libraries:
    - pylatexenc (if available)
    - Custom validation logic

    Args:
        latex_code: LaTeX code to validate

    Returns:
        Tuple of (is_valid, error_message, metadata)
    """
    # Try pylatexenc if available
    try:
        from pylatexenc.latex2text import LatexNodes2Text

        # This validates by attempting to parse
        try:
            LatexNodes2Text().latex_to_text(latex_code)
            return True, "", {"validation_method": "pylatexenc"}
        except Exception as e:
            return False, f"pylatexenc error: {e}", {"validation_method": "pylatexenc"}
    except ImportError:
        pass

    # Fallback: return None to indicate no Python validator available
    return None, "No Python LaTeX validator available", {}


def validate_latex_with_openreview(
    latex_code: str, prefer_http: bool = True
) -> Tuple[Optional[bool], str, Dict[str, Any]]:
    """
    Validate LaTeX using openreview/latex-validation.

    Tries multiple methods in order:
    1. HTTP server (if server is running)
    2. Node.js CLI (if latex-validation is built)
    3. Python validators (if available)
    4. Returns None if none available

    Args:
        latex_code: LaTeX code to validate
        prefer_http: Prefer HTTP server method (default: True)

    Returns:
        Tuple of (is_valid, error_message, metadata)
        Returns (None, error, {}) if validation cannot be performed
    """
    # Try HTTP server first (most efficient if server is running)
    if prefer_http:
        result = validate_via_http_server(latex_code)
        if result[0] is not None:
            return result

    # Try CLI if HTTP server not available
    if NODE_AVAILABLE:
        result = validate_via_node_cli(latex_code)
        if result[0] is not None:
            return result

    # Try Python validators
    result = validate_via_python_latex_validator(latex_code)
    if result[0] is not None:
        return result

    # No validator available
    return (
        None,
        "No LaTeX validator available. Options: 1) Start HTTP server: 'node dist/main.js run-server', 2) Install Node.js and build latex-validation",
        {},
    )
