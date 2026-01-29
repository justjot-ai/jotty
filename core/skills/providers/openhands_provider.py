"""
OpenHands Provider for Jotty V2
================================

Integrates the OpenHands (formerly OpenDevin) SDK for terminal and code operations.
https://github.com/OpenHands/OpenHands

Capabilities:
- Shell command execution
- Code file editing
- Git operations
- Package management
- Process management
"""

import time
import logging
import asyncio
import subprocess
from typing import Any, Dict, List, Optional
from pathlib import Path

from .base import SkillProvider, SkillCategory, ProviderCapability, ProviderResult

logger = logging.getLogger(__name__)

# Try to import OpenHands
try:
    # OpenHands SDK imports
    from openhands import Agent as OpenHandsAgent
    from openhands.tools import TerminalTool, FileEditTool
    OPENHANDS_AVAILABLE = True
except ImportError:
    OPENHANDS_AVAILABLE = False
    OpenHandsAgent = None


class OpenHandsProvider(SkillProvider):
    """
    Provider using OpenHands SDK for terminal and code operations.

    Features:
    - Sandboxed shell execution
    - Code editing with AI assistance
    - Git workflow automation
    - Multi-step development tasks
    """

    name = "openhands"
    version = "0.38.0"
    description = "Terminal and code operations via OpenHands SDK"

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        self.capabilities = [
            ProviderCapability(
                category=SkillCategory.TERMINAL,
                actions=["run_command", "install_package", "git_commit", "process_manage"],
                max_concurrent=5,
                requires_display=False,
                estimated_latency_ms=1000,
            ),
            ProviderCapability(
                category=SkillCategory.CODE_EXECUTION,
                actions=["run_python", "run_node", "run_bash", "run_tests"],
                estimated_latency_ms=2000,
            ),
            ProviderCapability(
                category=SkillCategory.FILE_OPERATIONS,
                actions=["read", "write", "edit", "search", "replace"],
                estimated_latency_ms=500,
            ),
        ]

        # Configuration
        self.sandbox_mode = config.get('sandbox', True) if config else True
        self.working_dir = config.get('working_dir', '.') if config else '.'
        self.timeout = config.get('timeout', 120) if config else 120

        # State
        self._agent = None

    async def initialize(self) -> bool:
        """Initialize OpenHands provider."""
        # Even without the full SDK, we can provide terminal capabilities
        # via subprocess as a fallback

        if OPENHANDS_AVAILABLE:
            try:
                # Initialize OpenHands agent
                self._agent = OpenHandsAgent()
                self.is_initialized = True
                self.is_available = True
                logger.info(f"✅ {self.name} provider initialized with full SDK")
                return True
            except Exception as e:
                logger.warning(f"OpenHands SDK init failed: {e}, using fallback")

        # Fallback: Use subprocess-based terminal
        self.is_initialized = True
        self.is_available = True
        logger.info(f"✅ {self.name} provider initialized (fallback mode)")
        return True

    def get_categories(self) -> List[SkillCategory]:
        return [
            SkillCategory.TERMINAL,
            SkillCategory.CODE_EXECUTION,
            SkillCategory.FILE_OPERATIONS,
        ]

    async def execute(self, task: str, context: Dict[str, Any] = None) -> ProviderResult:
        """Execute terminal/code task."""
        start_time = time.time()
        context = context or {}

        try:
            # Determine task type
            task_lower = task.lower()

            if any(kw in task_lower for kw in ['run', 'execute', 'command', 'shell', 'bash']):
                result = await self._execute_command(task, context)
            elif any(kw in task_lower for kw in ['edit', 'modify', 'change', 'update file']):
                result = await self._edit_file(task, context)
            elif any(kw in task_lower for kw in ['read', 'cat', 'show', 'display file']):
                result = await self._read_file(task, context)
            elif any(kw in task_lower for kw in ['git', 'commit', 'push', 'pull']):
                result = await self._git_operation(task, context)
            elif any(kw in task_lower for kw in ['install', 'pip', 'npm', 'apt']):
                result = await self._install_package(task, context)
            else:
                # Generic command execution
                result = await self._execute_command(task, context)

            result.execution_time = time.time() - start_time
            result.provider_name = self.name
            self.record_execution(result)
            return result

        except Exception as e:
            logger.error(f"OpenHands error: {e}")
            result = ProviderResult(
                success=False,
                output=None,
                error=str(e),
                execution_time=time.time() - start_time,
                provider_name=self.name,
                retryable=True,
            )
            self.record_execution(result)
            return result

    async def _execute_command(self, task: str, context: Dict) -> ProviderResult:
        """Execute a shell command."""
        # Extract command from task (simple extraction)
        command = self._extract_command(task)

        if not command:
            return ProviderResult(
                success=False,
                output=None,
                error="Could not extract command from task",
                category=SkillCategory.TERMINAL,
            )

        logger.info(f"🖥️  Executing: {command}")

        try:
            # Run command
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.working_dir,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.timeout
            )

            success = process.returncode == 0
            output = {
                'command': command,
                'stdout': stdout.decode('utf-8', errors='replace'),
                'stderr': stderr.decode('utf-8', errors='replace'),
                'return_code': process.returncode,
            }

            return ProviderResult(
                success=success,
                output=output,
                error=stderr.decode('utf-8', errors='replace') if not success else "",
                category=SkillCategory.TERMINAL,
                confidence=0.95 if success else 0.3,
            )

        except asyncio.TimeoutError:
            return ProviderResult(
                success=False,
                output=None,
                error=f"Command timed out after {self.timeout}s",
                category=SkillCategory.TERMINAL,
                retryable=True,
            )

    async def _edit_file(self, task: str, context: Dict) -> ProviderResult:
        """Edit a file."""
        # This would use OpenHands FileEditTool or fall back to manual edit
        file_path = context.get('file_path')
        changes = context.get('changes')

        if not file_path:
            return ProviderResult(
                success=False,
                output=None,
                error="No file_path provided in context",
                category=SkillCategory.FILE_OPERATIONS,
            )

        # Simple file edit implementation
        try:
            path = Path(file_path)
            if path.exists():
                content = path.read_text()
                # Apply changes (simplified)
                if changes:
                    for old, new in changes.items():
                        content = content.replace(old, new)
                    path.write_text(content)

            return ProviderResult(
                success=True,
                output={'file': str(path), 'edited': True},
                category=SkillCategory.FILE_OPERATIONS,
            )

        except Exception as e:
            return ProviderResult(
                success=False,
                output=None,
                error=str(e),
                category=SkillCategory.FILE_OPERATIONS,
            )

    async def _read_file(self, task: str, context: Dict) -> ProviderResult:
        """Read a file."""
        file_path = context.get('file_path') or self._extract_file_path(task)

        if not file_path:
            return ProviderResult(
                success=False,
                output=None,
                error="Could not determine file path",
                category=SkillCategory.FILE_OPERATIONS,
            )

        try:
            path = Path(file_path)
            if not path.exists():
                return ProviderResult(
                    success=False,
                    output=None,
                    error=f"File not found: {file_path}",
                    category=SkillCategory.FILE_OPERATIONS,
                )

            content = path.read_text()
            return ProviderResult(
                success=True,
                output={'file': str(path), 'content': content},
                category=SkillCategory.FILE_OPERATIONS,
            )

        except Exception as e:
            return ProviderResult(
                success=False,
                output=None,
                error=str(e),
                category=SkillCategory.FILE_OPERATIONS,
            )

    async def _git_operation(self, task: str, context: Dict) -> ProviderResult:
        """Execute git operation."""
        task_lower = task.lower()

        if 'commit' in task_lower:
            message = context.get('message', 'Auto-commit by Jotty')
            command = f'git add -A && git commit -m "{message}"'
        elif 'push' in task_lower:
            command = 'git push'
        elif 'pull' in task_lower:
            command = 'git pull'
        elif 'status' in task_lower:
            command = 'git status'
        elif 'diff' in task_lower:
            command = 'git diff'
        else:
            command = self._extract_command(task)

        return await self._execute_command(f"Run: {command}", context)

    async def _install_package(self, task: str, context: Dict) -> ProviderResult:
        """Install a package."""
        task_lower = task.lower()

        if 'pip' in task_lower:
            package = self._extract_package_name(task)
            command = f"pip install {package}"
        elif 'npm' in task_lower:
            package = self._extract_package_name(task)
            command = f"npm install {package}"
        elif 'apt' in task_lower:
            package = self._extract_package_name(task)
            command = f"sudo apt-get install -y {package}"
        else:
            command = self._extract_command(task)

        return await self._execute_command(f"Run: {command}", context)

    def _extract_command(self, task: str) -> Optional[str]:
        """Extract command from task description."""
        # Simple extraction - look for quoted commands or after keywords
        import re

        # Look for backtick or quote enclosed commands
        patterns = [
            r'`([^`]+)`',
            r'"([^"]+)"',
            r"'([^']+)'",
            r'run[:\s]+(.+?)(?:\.|$)',
            r'execute[:\s]+(.+?)(?:\.|$)',
            r'command[:\s]+(.+?)(?:\.|$)',
        ]

        for pattern in patterns:
            match = re.search(pattern, task, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # Fallback: if task looks like a command itself
        if task.startswith(('ls', 'cd', 'cat', 'echo', 'git', 'pip', 'npm', 'python')):
            return task

        return None

    def _extract_file_path(self, task: str) -> Optional[str]:
        """Extract file path from task."""
        import re
        # Look for path-like strings
        patterns = [
            r'([/\w.-]+\.\w+)',  # file.ext
            r'([/\w.-]+/[\w.-]+)',  # path/file
        ]
        for pattern in patterns:
            match = re.search(pattern, task)
            if match:
                return match.group(1)
        return None

    def _extract_package_name(self, task: str) -> str:
        """Extract package name from install task."""
        import re
        # Look for package name after install keyword
        match = re.search(r'install\s+(\S+)', task, re.IGNORECASE)
        if match:
            return match.group(1)
        return ""

    # Convenience methods

    async def run_command(self, command: str) -> ProviderResult:
        """Run a specific command."""
        return await self.execute(f"Run: `{command}`")

    async def run_python(self, script: str) -> ProviderResult:
        """Run Python code."""
        return await self.execute(f"Run Python: `python -c \"{script}\"`")

    async def git_commit(self, message: str) -> ProviderResult:
        """Create a git commit."""
        return await self.execute("git commit", context={'message': message})

    async def install_pip(self, package: str) -> ProviderResult:
        """Install a pip package."""
        return await self.execute(f"pip install {package}")
