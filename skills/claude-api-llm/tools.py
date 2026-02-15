"""
Claude API LLM Skill
=====================

Structured code generation and text output using Anthropic API tool_use.
Eliminates preamble/fence issues by forcing structured JSON responses.

Tools:
    - generate_code_tool: Code generation with lint-gating via forced tool_use
    - generate_text_tool: Drop-in replacement for claude-cli-llm text generation
    - agentic_generate_tool: Multi-step agentic mode with tool loop
    - structured_output_tool: JSON schema-constrained output
"""

import ast
import difflib
import json
import logging
import os
import subprocess
import time as _time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from Jotty.core.utils.skill_status import SkillStatus
from Jotty.core.utils.tool_helpers import tool_response, tool_error, tool_wrapper

logger = logging.getLogger(__name__)

status = SkillStatus("claude-api-llm")


# =============================================================================
# CLAUDE API CLIENT (Singleton)
# =============================================================================

class ClaudeAPIClient:
    """Reusable Anthropic API client with tool_use support.

    Singleton wrapping anthropic.Anthropic() using shared client kwargs.
    Handles cost tracking and rate-limit retry.
    """

    _instance: Optional["ClaudeAPIClient"] = None

    def __init__(self):
        self._client = None
        self._model = None

    @classmethod
    def get_instance(cls) -> "ClaudeAPIClient":
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = ClaudeAPIClient()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None

    def _ensure_client(self):
        """Lazy-init the Anthropic client."""
        if self._client is not None:
            return

        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")

        # DSPy loads dotenv which populates ANTHROPIC_API_KEY from .env files.
        # Import it before reading client kwargs to ensure the key is available.
        try:
            import dspy  # noqa: F401 — side-effect: loads dotenv
        except ImportError:
            pass

        from Jotty.core.foundation.anthropic_client_kwargs import get_anthropic_client_kwargs
        from Jotty.core.foundation.config_defaults import MODEL_SONNET

        kwargs = get_anthropic_client_kwargs()
        self._client = anthropic.Anthropic(**kwargs)
        self._model = MODEL_SONNET

    @property
    def client(self):
        """Get the initialized Anthropic client."""
        self._ensure_client()
        return self._client

    @property
    def model(self) -> str:
        """Get the configured model name."""
        self._ensure_client()
        return self._model

    def call_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        tool_choice: Optional[Dict[str, Any]] = None,
        max_tokens: int = 4096,
        system: Optional[str] = None,
    ) -> Any:
        """Make an API call with tool definitions.

        Args:
            messages: Conversation messages
            tools: Tool definitions for the API
            tool_choice: Tool choice constraint (e.g., force a specific tool)
            max_tokens: Maximum response tokens
            system: Optional system prompt

        Returns:
            API response object
        """
        return self._call_with_retry(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
            system=system,
        )

    def call_messages(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 4096,
        system: Optional[str] = None,
    ) -> Any:
        """Make a standard messages API call (no tool_use).

        Args:
            messages: Conversation messages
            max_tokens: Maximum response tokens
            system: Optional system prompt

        Returns:
            API response object
        """
        return self._call_with_retry(
            messages=messages,
            max_tokens=max_tokens,
            system=system,
        )

    def stream_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        tool_choice: Optional[Dict[str, Any]] = None,
        max_tokens: int = 4096,
        system: Optional[str] = None,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Any:
        """Stream API response with tool_use support.

        Uses Anthropic's messages.stream() to yield text tokens in real-time
        while still supporting tool_use blocks in the final message.

        Args:
            messages: Conversation messages
            tools: Tool definitions for the API
            tool_choice: Tool choice constraint
            max_tokens: Maximum response tokens
            system: Optional system prompt
            on_token: Optional callback(str) called for each text token

        Returns:
            Final Message object (same shape as call_with_tools)
        """
        self._ensure_client()

        request = {
            "model": self._model,
            "messages": messages,
            "tools": tools,
            "max_tokens": max_tokens,
        }
        if tool_choice:
            request["tool_choice"] = tool_choice
        if system:
            request["system"] = system

        with self._client.messages.stream(**request) as stream:
            for text in stream.text_stream:
                if on_token:
                    on_token(text)
            response = stream.get_final_message()

        self._track_cost(response)
        return response

    def _call_with_retry(
        self,
        max_retries: int = 3,
        base_delay: float = 8.0,
        **api_kwargs,
    ) -> Any:
        """Call API with exponential backoff on rate-limit errors."""
        self._ensure_client()

        # Build request kwargs
        request = {"model": self._model, **api_kwargs}
        # Remove None values
        request = {k: v for k, v in request.items() if v is not None}

        for attempt in range(max_retries + 1):
            try:
                response = self._client.messages.create(**request)
                self._track_cost(response)
                return response
            except Exception as e:
                err_str = str(e)
                is_rate_limit = (
                    "429" in err_str
                    or "RateLimit" in err_str
                    or "rate limit" in err_str.lower()
                    or "Too Many Requests" in err_str
                )
                if is_rate_limit and attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    logger.info(
                        "Rate limited (attempt %d/%d), retrying in %.0fs...",
                        attempt + 1, max_retries, delay,
                    )
                    _time.sleep(delay)
                else:
                    raise

    def _track_cost(self, response) -> None:
        """Track API call cost via CostTracker."""
        try:
            from Jotty.core.foundation.direct_anthropic_lm import get_cost_tracker
            usage = getattr(response, "usage", None)
            if usage:
                tracker = get_cost_tracker()
                tracker.record_call(
                    model=self._model,
                    input_tokens=getattr(usage, "input_tokens", 0),
                    output_tokens=getattr(usage, "output_tokens", 0),
                    provider="anthropic-api",
                )
        except Exception:
            pass  # Cost tracking is best-effort


# =============================================================================
# LINT GATING
# =============================================================================

class LintGate:
    """Syntax validation for generated code."""

    @staticmethod
    def validate_python(code: str) -> Optional[str]:
        """Validate Python code via ast.parse.

        Returns None if valid, error message string if invalid.
        """
        try:
            ast.parse(code)
            return None
        except SyntaxError as e:
            return f"SyntaxError at line {e.lineno}: {e.msg}"

    @staticmethod
    def validate(code: str, language: str) -> Optional[str]:
        """Validate code for the given language.

        Returns None if valid, error message if invalid.
        Currently supports: python.
        """
        if language.lower() in ("python", "py"):
            return LintGate.validate_python(code)
        # Other languages: no validation yet, pass through
        return None


# =============================================================================
# TOOL 1: generate_code_tool
# =============================================================================

@tool_wrapper()
def generate_code_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate code using Anthropic API tool_use for structured output.

    Uses forced tool_choice to get clean code without preamble or fences.
    Includes lint-gating: validates syntax and retries on failure.

    Args:
        params: Dictionary containing:
            - prompt (str, required): Code generation prompt
            - language (str, optional): Target language (default: "python")
            - filename (str, optional): Suggested filename

    Returns:
        Dictionary with:
            - success (bool): Whether generation succeeded
            - code (str): Clean generated code
            - language (str): Language of the code
            - filename (str): Suggested filename
            - lint_passed (bool): Whether code passed syntax validation
    """
    status.set_callback(params.pop("_status_callback", None))

    prompt = params.get("prompt")
    if not prompt:
        return tool_error("prompt parameter is required")

    language = params.get("language", "python")
    filename = params.get("filename", "")

    status.emit("Generating", f"Generating {language} code via API tool_use...")

    # Define the create_file tool for forced structured output
    create_file_tool = {
        "name": "create_file",
        "description": "Create a file with the specified code content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The complete code content for the file. Must be clean, valid code with no markdown fences or conversational text.",
                },
                "language": {
                    "type": "string",
                    "description": "Programming language of the code.",
                },
                "filename": {
                    "type": "string",
                    "description": "Suggested filename for the code.",
                },
            },
            "required": ["code", "language", "filename"],
        },
    }

    messages = [{"role": "user", "content": prompt}]

    max_lint_retries = 2
    last_lint_error = None

    try:
        api = ClaudeAPIClient.get_instance()

        for lint_attempt in range(max_lint_retries + 1):
            # On retry, include the lint error as feedback
            if lint_attempt > 0 and last_lint_error:
                messages.append({
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "retry", "name": "create_file", "input": _last_input}],
                })
                messages.append({
                    "role": "user",
                    "content": f"The code has a syntax error: {last_lint_error}. Please fix it and try again.",
                })

            response = api.call_with_tools(
                messages=messages,
                tools=[create_file_tool],
                tool_choice={"type": "tool", "name": "create_file"},
                system="You are a code generation assistant. Generate clean, valid code. No explanations, no markdown fences — just the code via the create_file tool.",
            )

            # Extract tool_use block from response
            tool_input = _extract_tool_input(response)
            if tool_input is None:
                return tool_error("Model did not return a tool_use block")

            code = tool_input.get("code", "")
            resp_language = tool_input.get("language", language)
            resp_filename = tool_input.get("filename", filename)
            _last_input = tool_input

            # Lint gate
            lint_error = LintGate.validate(code, resp_language)
            if lint_error is None:
                status.emit("Done", f"Code generated ({resp_language}), lint passed")
                return tool_response(
                    code=code,
                    language=resp_language,
                    filename=resp_filename,
                    lint_passed=True,
                    model=api.model,
                    provider="anthropic-api",
                )

            last_lint_error = lint_error
            logger.warning("Lint failed (attempt %d/%d): %s", lint_attempt + 1, max_lint_retries + 1, lint_error)

        # All lint retries exhausted — return code with lint_passed=False
        status.emit("Warning", "Code generated but lint validation failed")
        return tool_response(
            code=code,
            language=resp_language,
            filename=resp_filename,
            lint_passed=False,
            lint_error=last_lint_error,
            model=api.model,
            provider="anthropic-api",
        )

    except Exception as e:
        logger.error("generate_code_tool error: %s", e, exc_info=True)
        return tool_error(f"Code generation failed: {e}")


# =============================================================================
# TOOL 2: generate_text_tool
# =============================================================================

@tool_wrapper()
def generate_text_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate text using Anthropic API (drop-in replacement for claude-cli-llm).

    Args:
        params: Dictionary containing:
            - prompt (str, required): Text generation prompt
            - max_tokens (int, optional): Maximum tokens (default: 4096)

    Returns:
        Dictionary with:
            - success (bool): Whether generation succeeded
            - text (str): Generated text
            - model (str): Model used
            - provider (str): Always "anthropic-api"
    """
    status.set_callback(params.pop("_status_callback", None))

    prompt = params.get("prompt")
    if not prompt:
        return tool_error("prompt parameter is required")

    max_tokens = int(params.get("max_tokens", 4096))

    status.emit("Generating", "Generating text via Anthropic API...")

    try:
        api = ClaudeAPIClient.get_instance()
        response = api.call_messages(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )

        text = _extract_text(response)
        status.emit("Done", "Text generated")

        return tool_response(
            text=text,
            model=api.model,
            provider="anthropic-api",
        )

    except Exception as e:
        logger.error("generate_text_tool error: %s", e, exc_info=True)
        return tool_error(f"Text generation failed: {e}")


# =============================================================================
# TOOL 3: agentic_generate_tool
# =============================================================================

class AgenticToolExecutor:
    """Executes tool calls from the agentic loop.

    Maps internal tool names to actual skill implementations
    (file-operations, terminal-session) with lint-gating for code writes.
    """

    SANDBOX_ENV_STRIP_PATTERNS = ('SECRET', 'KEY', 'TOKEN', 'PASSWORD', 'CREDENTIAL')

    def __init__(self, working_directory: str = "/tmp", sandbox_level: str = "sandboxed"):
        self.working_directory = working_directory
        self._sandbox_level = sandbox_level
        self.files_created: List[str] = []
        self.execution_output: List[str] = []
        self.tool_call_history: List[Dict[str, Any]] = []

    def execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """Execute a tool call and return the result as a string."""
        result = None
        try:
            if tool_name == "write_file":
                result = self._write_file(tool_input)
            elif tool_name == "execute_command":
                result = self._execute_command(tool_input)
            elif tool_name == "read_file":
                result = self._read_file(tool_input)
            elif tool_name == "edit_file":
                result = self._edit_file(tool_input)
            elif tool_name == "search_replace":
                result = self._search_replace(tool_input)
            else:
                result = {"success": False, "error": f"Unknown tool: {tool_name}"}
        except Exception as e:
            result = {"success": False, "error": str(e)}

        self.tool_call_history.append({
            "tool": tool_name,
            "input": tool_input,
            "result": result,
        })

        return json.dumps(result, default=str)

    def _write_file(self, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """Write a file with lint-gating for code files."""
        path = tool_input.get("path", "")
        content = tool_input.get("content", "")

        if not path:
            return {"success": False, "error": "path is required"}

        # Resolve relative paths against working directory
        file_path = Path(path)
        if not file_path.is_absolute():
            file_path = Path(self.working_directory) / file_path

        # Lint gate for Python files
        if file_path.suffix.lower() == ".py":
            lint_error = LintGate.validate_python(content)
            if lint_error:
                return {"success": False, "error": f"Lint failed: {lint_error}. Fix the code and try again."}

        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        self.files_created.append(str(file_path))

        return {"success": True, "path": str(file_path), "bytes_written": len(content)}

    def _execute_command(self, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a shell command, routing through SandboxManager when available."""
        command = tool_input.get("command", "")
        if not command:
            return {"success": False, "error": "command is required"}

        try:
            from Jotty.core.orchestration.sandbox_manager import (
                SandboxManager, TrustLevel,
            )
            trust_map = {
                "trusted": TrustLevel.TRUSTED,
                "sandboxed": TrustLevel.SANDBOXED,
                "dangerous": TrustLevel.DANGEROUS,
            }
            trust = trust_map.get(self._sandbox_level, TrustLevel.SANDBOXED)

            # Wrap command as Python subprocess call for SandboxManager
            code = (
                "import subprocess, sys\n"
                f"r = subprocess.run({command!r}, shell=True, capture_output=True, "
                f"text=True, cwd={self.working_directory!r})\n"
                "print(r.stdout)\n"
                "print(r.stderr, file=sys.stderr)\n"
                "sys.exit(r.returncode)"
            )

            import asyncio
            try:
                loop = asyncio.get_running_loop()
                # Already inside an async context — fall back to sync with sanitization
                return self._execute_command_sync(command, tool_input)
            except RuntimeError:
                # No running loop — we can use asyncio.run
                sandbox = SandboxManager()
                result = asyncio.run(
                    sandbox.execute_sandboxed(code, trust)
                )
                self.execution_output.append(result.stdout or "")
                return {
                    "success": result.success,
                    "output": (result.stdout or "")[:10000],
                    "exit_code": result.exit_code,
                    "sandbox": result.sandbox_type,
                }
        except ImportError:
            return self._execute_command_sync(command, tool_input)

    def _execute_command_sync(self, command: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """Sync fallback with environment variable sanitization."""
        timeout = int(tool_input.get("timeout", 30))

        env = os.environ.copy()
        for key in list(env.keys()):
            if any(s in key.upper() for s in self.SANDBOX_ENV_STRIP_PATTERNS):
                del env[key]

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.working_directory,
                env=env,
            )
            output = result.stdout
            if result.stderr:
                output += f"\nSTDERR:\n{result.stderr}" if output else result.stderr

            self.execution_output.append(output)

            return {
                "success": result.returncode == 0,
                "output": output[:10000],
                "exit_code": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": f"Command timed out after {timeout}s"}

    def _read_file(self, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """Read a file's contents."""
        path = tool_input.get("path", "")
        if not path:
            return {"success": False, "error": "path is required"}

        file_path = Path(path)
        if not file_path.is_absolute():
            file_path = Path(self.working_directory) / file_path

        if not file_path.exists():
            return {"success": False, "error": f"File not found: {file_path}"}

        content = file_path.read_text(encoding="utf-8")
        return {"success": True, "content": content[:20000], "path": str(file_path)}

    def _edit_file(self, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """Edit a file by replacing old_text with new_text."""
        path = tool_input.get("path", "")
        old_text = tool_input.get("old_text", "")
        new_text = tool_input.get("new_text", "")

        if not path:
            return {"success": False, "error": "path is required"}
        if not old_text:
            return {"success": False, "error": "old_text is required"}

        file_path = Path(path)
        if not file_path.is_absolute():
            file_path = Path(self.working_directory) / file_path

        if not file_path.exists():
            return {"success": False, "error": f"File not found: {file_path}"}

        content = file_path.read_text(encoding="utf-8")
        if old_text not in content:
            return {"success": False, "error": "old_text not found in file"}

        new_content = content.replace(old_text, new_text, 1)

        # Lint gate for Python files
        if file_path.suffix.lower() == ".py":
            lint_error = LintGate.validate_python(new_content)
            if lint_error:
                return {"success": False, "error": f"Edit would produce invalid syntax: {lint_error}"}

        file_path.write_text(new_content, encoding="utf-8")
        return {"success": True, "path": str(file_path)}

    def _search_replace(self, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """Apply search/replace edits with fuzzy matching and diff preview."""
        path = tool_input.get("path", "")
        edits = tool_input.get("edits", [])

        if not path:
            return {"success": False, "error": "path is required"}
        if not edits:
            return {"success": False, "error": "edits array is required"}

        file_path = Path(path)
        if not file_path.is_absolute():
            file_path = Path(self.working_directory) / file_path

        if not file_path.exists():
            return {"success": False, "error": f"File not found: {file_path}"}

        original = file_path.read_text(encoding="utf-8")
        content = original
        applied = 0

        for edit in edits:
            search = edit.get("search", "")
            replace = edit.get("replace", "")
            if not search:
                continue

            # Try exact match first
            if search in content:
                content = content.replace(search, replace, 1)
                applied += 1
                continue

            # Fuzzy match: find best matching line range
            lines = content.splitlines(keepends=True)
            search_lines = search.splitlines()
            best_start, best_ratio = self._fuzzy_find(lines, search_lines)
            if best_ratio >= 0.6:
                replace_lines = replace.splitlines(keepends=True)
                if replace and not replace.endswith("\n"):
                    replace_lines = replace.splitlines(keepends=False)
                    replace_lines = [l + "\n" for l in replace_lines]
                lines[best_start:best_start + len(search_lines)] = replace_lines
                content = "".join(lines)
                applied += 1

        if applied == 0:
            return {"success": False, "error": "No edits could be applied", "applied": 0, "total": len(edits)}

        # Lint gate for Python files
        if file_path.suffix.lower() == ".py":
            lint_error = LintGate.validate_python(content)
            if lint_error:
                return {"success": False, "error": f"Edit would produce invalid syntax: {lint_error}"}

        # Generate diff preview
        diff_lines = list(difflib.unified_diff(
            original.splitlines(keepends=True),
            content.splitlines(keepends=True),
            fromfile=str(file_path),
            tofile=str(file_path),
        ))
        diff_preview = "".join(diff_lines[:50])

        file_path.write_text(content, encoding="utf-8")
        return {"success": True, "applied": applied, "total": len(edits), "diff": diff_preview}

    @staticmethod
    def _fuzzy_find(lines: List[str], search_lines: List[str]) -> tuple:
        """Find the best fuzzy match for search_lines within lines.

        Returns:
            (best_start_index, best_ratio) — ratio >= 0.6 is a usable match.
        """
        if not search_lines or not lines:
            return (-1, 0.0)

        search_block = "".join(search_lines)
        best_start = -1
        best_ratio = 0.0
        window = len(search_lines)

        for i in range(len(lines) - window + 1):
            candidate = "".join(lines[i:i + window])
            ratio = difflib.SequenceMatcher(None, search_block, candidate).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_start = i

        return (best_start, best_ratio)


# Agentic tool definitions for the Claude API
AGENTIC_TOOL_DEFINITIONS = [
    {
        "name": "write_file",
        "description": "Write content to a file. For Python files, content must be syntactically valid.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to write to"},
                "content": {"type": "string", "description": "File content to write"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "execute_command",
        "description": "Execute a shell command and return stdout/stderr.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
                "timeout": {"type": "integer", "description": "Timeout in seconds (default: 30)"},
            },
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to read"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "edit_file",
        "description": "Edit a file by replacing old_text with new_text.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to edit"},
                "old_text": {"type": "string", "description": "Text to find and replace"},
                "new_text": {"type": "string", "description": "Replacement text"},
            },
            "required": ["path", "old_text", "new_text"],
        },
    },
    {
        "name": "search_replace",
        "description": (
            "Edit a file using search/replace blocks. Supports fuzzy matching — "
            "the search text doesn't need to be exact. Shows a diff preview."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to edit"},
                "edits": {
                    "type": "array",
                    "description": "List of search/replace edits to apply",
                    "items": {
                        "type": "object",
                        "properties": {
                            "search": {"type": "string", "description": "Text to find (fuzzy match)"},
                            "replace": {"type": "string", "description": "Replacement text"},
                        },
                        "required": ["search", "replace"],
                    },
                },
            },
            "required": ["path", "edits"],
        },
    },
]


class ContextBuilder:
    """Builds file context for the agentic loop's system prompt.

    Discovers relevant project files and builds a summary containing
    directory tree + key file contents within a token budget.
    """

    SKIP_DIRS = {
        '.git', '__pycache__', 'node_modules', '.venv', 'venv',
        'env', '.env', 'dist', 'build', '.pytest_cache', '.mypy_cache',
        '.tox', '.eggs', '*.egg-info',
    }
    CODE_EXTENSIONS = {
        '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.go', '.rs',
        '.cpp', '.h', '.hpp', '.css', '.html', '.json', '.yaml', '.yml',
        '.toml', '.md', '.sh',
    }
    MAX_CONTEXT_CHARS = 50000  # ~12K tokens
    MAX_TREE_LINES = 200
    MAX_FILES = 20
    MAX_DEPTH = 3
    MAX_FILE_CHARS = 5000
    PRIORITY_PATTERNS = [
        'README', 'setup.py', 'pyproject.toml', 'package.json',
        'main.py', 'app.py', 'index.', '__init__.py',
    ]

    @classmethod
    def build_context(cls, working_directory: str, extensions: Optional[set] = None) -> str:
        """Build a context summary of the project directory.

        Returns a string suitable for injection into the system prompt,
        containing directory tree + key file contents (within token budget).
        """
        extensions = extensions or cls.CODE_EXTENSIONS
        root = Path(working_directory)
        if not root.exists():
            return ""

        tree_lines = []
        file_list = []

        for dirpath, dirnames, filenames in os.walk(root):
            # Prune skipped directories in-place
            dirnames[:] = [d for d in dirnames if d not in cls.SKIP_DIRS]
            rel = os.path.relpath(dirpath, root)
            depth = rel.count(os.sep) if rel != '.' else 0
            if depth > cls.MAX_DEPTH:
                continue
            indent = "  " * depth
            tree_lines.append(f"{indent}{os.path.basename(dirpath)}/")
            for f in sorted(filenames):
                if Path(f).suffix in extensions:
                    tree_lines.append(f"{indent}  {f}")
                    file_list.append(os.path.join(dirpath, f))

        tree = "\n".join(tree_lines[:cls.MAX_TREE_LINES])

        # Include small key files (README, config, main entry points)
        context_parts = [f"PROJECT STRUCTURE:\n{tree}\n"]
        budget = cls.MAX_CONTEXT_CHARS - len(tree)

        # Prioritize: README > config > small source files
        prioritized = sorted(
            file_list,
            key=lambda f: next(
                (i for i, p in enumerate(cls.PRIORITY_PATTERNS)
                 if p in os.path.basename(f)),
                999,
            ),
        )

        for fpath in prioritized[:cls.MAX_FILES]:
            if budget <= 0:
                break
            try:
                content = Path(fpath).read_text(encoding='utf-8')
                if len(content) > cls.MAX_FILE_CHARS:
                    content = content[:cls.MAX_FILE_CHARS] + "\n... (truncated)"
                rel = os.path.relpath(fpath, root)
                entry = f"\n--- {rel} ---\n{content}\n"
                if len(entry) <= budget:
                    context_parts.append(entry)
                    budget -= len(entry)
            except (IOError, UnicodeDecodeError):
                continue

        return "\n".join(context_parts)


@tool_wrapper()
def agentic_generate_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Multi-step agentic code generation with tool loop.

    Sends prompt to Claude with tool definitions (write_file, execute_command,
    read_file, edit_file, search_replace). Claude autonomously writes files,
    runs commands, and iterates until done.

    Args:
        params: Dictionary containing:
            - prompt (str, required): Task prompt for agentic execution
            - tools (list, optional): Tool names to enable
            - working_directory (str, optional): Working directory (default: "/tmp")
            - max_tool_rounds (int, optional): Max tool-use rounds (default: 5)
            - stream (bool, optional): Enable token-level streaming (default: False)
            - sandbox_level (str, optional): "trusted", "sandboxed", or "dangerous" (default: "sandboxed")
            - include_context (bool, optional): Auto-discover project files for context (default: True)

    Returns:
        Dictionary with:
            - success (bool): Whether execution succeeded
            - response (str): Final response text
            - tool_calls (list): Tool call history
            - files_created (list): Paths of files created
            - execution_output (str): Combined execution output
    """
    status.set_callback(params.pop("_status_callback", None))

    prompt = params.get("prompt")
    if not prompt:
        return tool_error("prompt parameter is required")

    working_dir = params.get("working_directory", "/tmp")
    max_rounds = int(params.get("max_tool_rounds", 5))
    use_stream = bool(params.get("stream", False))
    sandbox_level = params.get("sandbox_level", "sandboxed")
    include_context = bool(params.get("include_context", True))

    # Filter tool definitions based on requested tools
    requested_tools = params.get("tools")
    if requested_tools:
        tool_defs = [t for t in AGENTIC_TOOL_DEFINITIONS if t["name"] in requested_tools]
    else:
        tool_defs = AGENTIC_TOOL_DEFINITIONS

    status.emit("Starting", "Starting agentic generation loop...")

    executor = AgenticToolExecutor(working_directory=working_dir, sandbox_level=sandbox_level)
    messages = [{"role": "user", "content": prompt}]

    # Build system prompt with optional project context
    system_parts = [
        "You are a code generation and execution assistant. "
        "Use the provided tools to write files, execute commands, and complete the task. "
        "For Python files, ensure the code is syntactically valid."
    ]
    if include_context:
        project_context = ContextBuilder.build_context(working_dir)
        if project_context:
            system_parts.append(f"\n\nCURRENT PROJECT CONTEXT:\n{project_context}")
    system_prompt = "".join(system_parts)

    try:
        api = ClaudeAPIClient.get_instance()

        for round_num in range(max_rounds):
            status.emit("Round", f"Tool loop round {round_num + 1}/{max_rounds}...")

            if use_stream:
                response = api.stream_with_tools(
                    messages=messages,
                    tools=tool_defs,
                    system=system_prompt,
                    on_token=lambda t: status.emit("Token", t),
                )
            else:
                response = api.call_with_tools(
                    messages=messages,
                    tools=tool_defs,
                    system=system_prompt,
                )

            # Check stop reason
            stop_reason = getattr(response, "stop_reason", "end_turn")

            # Process content blocks
            assistant_content = []
            tool_results = []
            has_tool_use = False

            for block in response.content:
                if block.type == "text":
                    assistant_content.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    has_tool_use = True
                    assistant_content.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })

                    # Execute the tool
                    status.emit("Tool", f"Executing {block.name}...")
                    result_str = executor.execute_tool(block.name, block.input)

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_str,
                    })

            # Add assistant response to messages
            messages.append({"role": "assistant", "content": assistant_content})

            if not has_tool_use or stop_reason == "end_turn":
                # Model is done — extract final text
                final_text = _extract_text(response)
                status.emit("Done", f"Agentic loop complete ({round_num + 1} rounds)")
                return tool_response(
                    response=final_text,
                    tool_calls=executor.tool_call_history,
                    files_created=executor.files_created,
                    execution_output="\n".join(executor.execution_output),
                    rounds=round_num + 1,
                    model=api.model,
                    provider="anthropic-api",
                )

            # Feed tool results back for next round
            messages.append({"role": "user", "content": tool_results})

        # Max rounds reached
        status.emit("Warning", f"Max rounds ({max_rounds}) reached")
        return tool_response(
            response="Max tool rounds reached",
            tool_calls=executor.tool_call_history,
            files_created=executor.files_created,
            execution_output="\n".join(executor.execution_output),
            rounds=max_rounds,
            model=api.model,
            provider="anthropic-api",
        )

    except Exception as e:
        logger.error("agentic_generate_tool error: %s", e, exc_info=True)
        return tool_error(
            f"Agentic generation failed: {e}",
            tool_calls=executor.tool_call_history,
            files_created=executor.files_created,
        )


# =============================================================================
# TOOL 4: structured_output_tool
# =============================================================================

@tool_wrapper()
def structured_output_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate structured JSON output using API tool_use with schema enforcement.

    Args:
        params: Dictionary containing:
            - prompt (str, required): Prompt for structured output
            - schema (dict, optional): JSON schema for the output structure

    Returns:
        Dictionary with:
            - success (bool): Whether generation succeeded
            - data (dict): Parsed JSON conforming to schema
            - model (str): Model used
    """
    status.set_callback(params.pop("_status_callback", None))

    prompt = params.get("prompt")
    if not prompt:
        return tool_error("prompt parameter is required")

    schema = params.get("schema")
    if not schema:
        # Default schema: freeform JSON object
        schema = {
            "type": "object",
            "properties": {
                "result": {"type": "string", "description": "The result"},
            },
        }

    status.emit("Generating", "Generating structured output via API tool_use...")

    # Define a tool whose input_schema matches the user's schema
    extract_tool = {
        "name": "extract_data",
        "description": "Extract structured data according to the specified schema.",
        "input_schema": schema,
    }

    try:
        api = ClaudeAPIClient.get_instance()
        response = api.call_with_tools(
            messages=[{"role": "user", "content": prompt}],
            tools=[extract_tool],
            tool_choice={"type": "tool", "name": "extract_data"},
            system="Extract the requested information into the structured format using the extract_data tool.",
        )

        tool_input = _extract_tool_input(response)
        if tool_input is None:
            return tool_error("Model did not return structured data")

        status.emit("Done", "Structured output generated")
        result = tool_response(model=api.model, provider="anthropic-api")
        result["data"] = tool_input
        return result

    except Exception as e:
        logger.error("structured_output_tool error: %s", e, exc_info=True)
        return tool_error(f"Structured output failed: {e}")


# =============================================================================
# HELPERS
# =============================================================================

def _extract_tool_input(response) -> Optional[Dict[str, Any]]:
    """Extract the input dict from the first tool_use block in a response."""
    for block in getattr(response, "content", []):
        if getattr(block, "type", None) == "tool_use":
            return block.input
    return None


def _extract_text(response) -> str:
    """Extract concatenated text from all text blocks in a response."""
    texts = []
    for block in getattr(response, "content", []):
        if getattr(block, "type", None) == "text":
            texts.append(block.text)
    return "\n".join(texts) if texts else ""


__all__ = [
    "generate_code_tool",
    "generate_text_tool",
    "agentic_generate_tool",
    "structured_output_tool",
    "ClaudeAPIClient",
    "LintGate",
    "AgenticToolExecutor",
    "ContextBuilder",
]
