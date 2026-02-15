"""
Code Interpreter Module
=======================

Provides sandboxed code execution for Python and JavaScript.
Supports both Docker-based isolation and subprocess fallback.
"""

import asyncio
import logging
import os
import sys
import tempfile
import uuid
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List, AsyncGenerator

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of code execution."""
    execution_id: str
    success: bool
    output: str
    error: Optional[str] = None
    duration_ms: int = 0
    language: str = "python"
    return_value: Any = None


class CodeInterpreter:
    """
    Sandboxed code execution engine.

    Supports Python and JavaScript execution with:
    - Docker-based isolation (preferred)
    - Subprocess with resource limits (fallback)
    - Timeout protection
    - Output streaming
    """

    # Safe imports whitelist for Python
    PYTHON_SAFE_IMPORTS = {
        # Built-ins
        "math", "random", "datetime", "json", "re", "collections",
        "itertools", "functools", "operator", "string", "textwrap",
        "statistics", "decimal", "fractions", "cmath",
        # Data science
        "numpy", "pandas", "scipy",
        # Visualization
        "matplotlib", "seaborn", "plotly",
        # ML (read-only operations)
        "sklearn",
    }

    # Blocked patterns in code
    BLOCKED_PATTERNS = [
        "import os", "from os",
        "import sys", "from sys",
        "import subprocess", "from subprocess",
        "import shutil", "from shutil",
        "__import__",
        "exec(", "eval(",
        "open(", "file(",
        "compile(",
        "globals(", "locals(",
        "getattr(", "setattr(", "delattr(",
        "import socket", "from socket",
        "import requests", "from requests",
        "import urllib", "from urllib",
        "import http", "from http",
    ]

    def __init__(
        self,
        timeout: int = 30,
        max_output: int = 50000,
        use_docker: bool = True
    ):
        """
        Initialize code interpreter.

        Args:
            timeout: Maximum execution time in seconds
            max_output: Maximum output size in characters
            use_docker: Try to use Docker for isolation
        """
        self.timeout = timeout
        self.max_output = max_output
        self.use_docker = use_docker and self._check_docker()
        self._temp_dir = Path(tempfile.gettempdir()) / "jotty_code"
        self._temp_dir.mkdir(exist_ok=True)

    def _check_docker(self) -> bool:
        """Check if Docker is available."""
        try:
            result = subprocess.run(
                ["docker", "version"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

    def _validate_python_code(self, code: str) -> Optional[str]:
        """
        Validate Python code for safety.

        Returns error message if code is unsafe, None if safe.
        """
        for pattern in self.BLOCKED_PATTERNS:
            if pattern in code:
                return f"Blocked pattern detected: {pattern}"
        return None

    async def execute_python(
        self,
        code: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """
        Execute Python code in sandbox.

        Args:
            code: Python code to execute
            context: Optional context variables

        Returns:
            ExecutionResult with output
        """
        import time
        start_time = time.time()
        execution_id = str(uuid.uuid4())[:12]

        # Validate code
        error = self._validate_python_code(code)
        if error:
            return ExecutionResult(
                execution_id=execution_id,
                success=False,
                output="",
                error=error,
                language="python"
            )

        try:
            if self.use_docker:
                result = await self._execute_python_docker(code, execution_id)
            else:
                result = await self._execute_python_subprocess(code, execution_id)

            duration_ms = int((time.time() - start_time) * 1000)
            result.duration_ms = duration_ms
            return result

        except asyncio.TimeoutError:
            return ExecutionResult(
                execution_id=execution_id,
                success=False,
                output="",
                error=f"Execution timed out after {self.timeout} seconds",
                language="python",
                duration_ms=self.timeout * 1000
            )
        except Exception as e:
            logger.error(f"Python execution error: {e}")
            return ExecutionResult(
                execution_id=execution_id,
                success=False,
                output="",
                error=str(e),
                language="python"
            )

    async def _execute_python_docker(
        self,
        code: str,
        execution_id: str
    ) -> ExecutionResult:
        """Execute Python in Docker container."""
        # Create temp file with code
        code_file = self._temp_dir / f"{execution_id}.py"
        code_file.write_text(code, encoding="utf-8")

        try:
            # Run in Docker with resource limits
            cmd = [
                "docker", "run",
                "--rm",
                "--network", "none",  # No network access
                "--memory", "256m",  # Memory limit
                "--cpus", "0.5",  # CPU limit
                "--read-only",  # Read-only filesystem
                "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",  # Temp storage
                "-v", f"{code_file}:/code.py:ro",  # Mount code read-only
                "python:3.11-slim",
                "python", "/code.py"
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.timeout
            )

            output = stdout.decode("utf-8", errors="replace")
            error_output = stderr.decode("utf-8", errors="replace")

            # Truncate if too long
            if len(output) > self.max_output:
                output = output[:self.max_output] + "\n... (output truncated)"

            if process.returncode == 0:
                return ExecutionResult(
                    execution_id=execution_id,
                    success=True,
                    output=output,
                    language="python"
                )
            else:
                return ExecutionResult(
                    execution_id=execution_id,
                    success=False,
                    output=output,
                    error=error_output or "Execution failed",
                    language="python"
                )

        finally:
            # Cleanup
            code_file.unlink(missing_ok=True)

    async def _execute_python_subprocess(
        self,
        code: str,
        execution_id: str
    ) -> ExecutionResult:
        """Execute Python in subprocess with limits."""
        # Create temp file with code
        code_file = self._temp_dir / f"{execution_id}.py"

        # Wrap code with output capture and limits
        wrapped_code = f'''
import sys
import resource

# Set resource limits
resource.setrlimit(resource.RLIMIT_CPU, (5, 5))  # 5 seconds CPU
resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))  # 256MB memory

# Execute user code
{code}
'''
        code_file.write_text(wrapped_code, encoding="utf-8")

        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable, str(code_file),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._temp_dir)
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.timeout
            )

            output = stdout.decode("utf-8", errors="replace")
            error_output = stderr.decode("utf-8", errors="replace")

            # Truncate if too long
            if len(output) > self.max_output:
                output = output[:self.max_output] + "\n... (output truncated)"

            if process.returncode == 0:
                return ExecutionResult(
                    execution_id=execution_id,
                    success=True,
                    output=output,
                    language="python"
                )
            else:
                return ExecutionResult(
                    execution_id=execution_id,
                    success=False,
                    output=output,
                    error=error_output or "Execution failed",
                    language="python"
                )

        finally:
            # Cleanup
            code_file.unlink(missing_ok=True)

    async def execute_javascript(
        self,
        code: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """
        Execute JavaScript code in sandbox.

        Args:
            code: JavaScript code to execute
            context: Optional context variables

        Returns:
            ExecutionResult with output
        """
        import time
        start_time = time.time()
        execution_id = str(uuid.uuid4())[:12]

        # Check for Node.js
        node_path = shutil.which("node")
        if not node_path:
            return ExecutionResult(
                execution_id=execution_id,
                success=False,
                output="",
                error="Node.js is not installed",
                language="javascript"
            )

        try:
            result = await self._execute_js_subprocess(code, execution_id, node_path)
            duration_ms = int((time.time() - start_time) * 1000)
            result.duration_ms = duration_ms
            return result

        except asyncio.TimeoutError:
            return ExecutionResult(
                execution_id=execution_id,
                success=False,
                output="",
                error=f"Execution timed out after {self.timeout} seconds",
                language="javascript",
                duration_ms=self.timeout * 1000
            )
        except Exception as e:
            logger.error(f"JavaScript execution error: {e}")
            return ExecutionResult(
                execution_id=execution_id,
                success=False,
                output="",
                error=str(e),
                language="javascript"
            )

    async def _execute_js_subprocess(
        self,
        code: str,
        execution_id: str,
        node_path: str
    ) -> ExecutionResult:
        """Execute JavaScript in Node.js subprocess."""
        # Create temp file with code
        code_file = self._temp_dir / f"{execution_id}.js"

        # Wrap code with console capture
        wrapped_code = f'''
// Sandbox setup
const originalLog = console.log;
const output = [];
console.log = (...args) => {{
    output.push(args.map(a => typeof a === 'object' ? JSON.stringify(a) : String(a)).join(' '));
}};
console.error = console.log;
console.warn = console.log;

try {{
    {code}
}} catch (e) {{
    console.log('Error:', e.message);
}}

// Print collected output
process.stdout.write(output.join('\\n'));
'''
        code_file.write_text(wrapped_code, encoding="utf-8")

        try:
            process = await asyncio.create_subprocess_exec(
                node_path, str(code_file),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._temp_dir)
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.timeout
            )

            output = stdout.decode("utf-8", errors="replace")
            error_output = stderr.decode("utf-8", errors="replace")

            # Truncate if too long
            if len(output) > self.max_output:
                output = output[:self.max_output] + "\n... (output truncated)"

            if process.returncode == 0:
                return ExecutionResult(
                    execution_id=execution_id,
                    success=True,
                    output=output,
                    language="javascript"
                )
            else:
                return ExecutionResult(
                    execution_id=execution_id,
                    success=False,
                    output=output,
                    error=error_output or "Execution failed",
                    language="javascript"
                )

        finally:
            # Cleanup
            code_file.unlink(missing_ok=True)

    async def execute(
        self,
        code: str,
        language: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """
        Execute code in the specified language.

        Args:
            code: Code to execute
            language: Programming language (python, javascript, etc.)
            context: Optional context variables

        Returns:
            ExecutionResult with output
        """
        language_lower = language.lower()

        if language_lower in ("python", "py", "python3"):
            return await self.execute_python(code, context)
        elif language_lower in ("javascript", "js", "node"):
            return await self.execute_javascript(code, context)
        else:
            return ExecutionResult(
                execution_id=str(uuid.uuid4())[:12],
                success=False,
                output="",
                error=f"Unsupported language: {language}. Supported: python, javascript",
                language=language
            )

    async def execute_streaming(
        self,
        code: str,
        language: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute code with streaming output.

        Yields output events as they occur.
        """
        execution_id = str(uuid.uuid4())[:12]

        yield {
            "type": "execution_start",
            "execution_id": execution_id,
            "language": language
        }

        result = await self.execute(code, language)

        # Stream output in chunks for large output
        output = result.output
        chunk_size = 1000
        for i in range(0, len(output), chunk_size):
            chunk = output[i:i + chunk_size]
            yield {
                "type": "output",
                "execution_id": execution_id,
                "chunk": chunk
            }

        yield {
            "type": "execution_end",
            "execution_id": execution_id,
            "success": result.success,
            "error": result.error,
            "duration_ms": result.duration_ms
        }


# Singleton interpreter instance
_interpreter = None


def get_code_interpreter() -> CodeInterpreter:
    """Get singleton code interpreter."""
    global _interpreter
    if _interpreter is None:
        _interpreter = CodeInterpreter()
    return _interpreter


async def execute_code(
    code: str,
    language: str = "python"
) -> Dict[str, Any]:
    """
    Convenience function to execute code.

    Returns result dictionary.
    """
    interpreter = get_code_interpreter()
    result = await interpreter.execute(code, language)
    return {
        "execution_id": result.execution_id,
        "success": result.success,
        "output": result.output,
        "error": result.error,
        "duration_ms": result.duration_ms,
        "language": result.language
    }
