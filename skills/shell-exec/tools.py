"""
Shell Execution Skill

Execute shell commands and Python scripts.
Refactored to use Jotty core utilities.

WARNING: This skill executes arbitrary code. Use with caution.
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any

from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper


@tool_wrapper(required_params=['command'])
def execute_command_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a shell command and return the output.

    WARNING: This tool executes shell commands. Use with caution.

    Args:
        params: Dictionary containing:
            - command (str, required): Shell command to execute
            - timeout (int, optional): Timeout in seconds (default: 30)
            - working_directory (str, optional): Working directory
            - shell (bool, optional): Use shell execution (default: True)

    Returns:
        Dictionary with success, stdout, stderr, exit_code, command
    """
    timeout = params.get('timeout', 30)
    working_directory = params.get('working_directory')
    use_shell = params.get('shell', True)

    cwd = None
    if working_directory:
        cwd_path = Path(working_directory)
        if not cwd_path.exists() or not cwd_path.is_dir():
            return tool_error(f'Invalid working directory: {working_directory}')
        cwd = str(cwd_path)

    try:
        result = subprocess.run(
            params['command'],
            shell=use_shell,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd
        )

        return tool_response(
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.returncode,
            command=params['command']
        )

    except subprocess.TimeoutExpired:
        return tool_error(f'Command timed out after {timeout} seconds', command=params['command'])


@tool_wrapper(required_params=['script'])
def execute_script_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a Python script and return the output.

    WARNING: This tool executes Python code. Use with extreme caution.

    Args:
        params: Dictionary containing:
            - script (str, required): Python script code to execute
            - timeout (int, optional): Timeout in seconds (default: 30)

    Returns:
        Dictionary with success, output, exit_code
    """
    timeout = params.get('timeout', 30)

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(params['script'])
        temp_script = f.name

    try:
        result = subprocess.run(
            [sys.executable, temp_script],
            capture_output=True,
            text=True,
            timeout=timeout
        )

        return tool_response(
            output=result.stdout + result.stderr,
            exit_code=result.returncode,
            error=result.stderr if result.returncode != 0 else None
        )

    except subprocess.TimeoutExpired:
        return tool_error(f'Script timed out after {timeout} seconds')

    finally:
        try:
            os.unlink(temp_script)
        except Exception:
            pass


__all__ = ['execute_command_tool', 'execute_script_tool']
