"""
Coding Swarm - Workspace Manager
==================================

Wraps SwarmTerminal for CodingSwarm validation.
Non-blocking: falls back to string-only mode if SwarmTerminal is unavailable.
"""

import logging
from typing import Optional, List

logger = logging.getLogger(__name__)


class WorkspaceManager:
    """Wraps SwarmTerminal for CodingSwarm validation.

    Non-blocking: falls back to string-only mode if SwarmTerminal is unavailable.
    Lazy-loads all dependencies on first use.
    """

    def __init__(self):
        self._terminal = None
        self._workspace_dir: Optional[str] = None
        self._initialized = False

    def _ensure_init(self) -> bool:
        """Lazy-load SwarmTerminal and create temp workspace directory."""
        if self._initialized:
            return self._terminal is not None
        self._initialized = True
        try:
            from ...orchestration.v2.swarm_terminal import SwarmTerminal
            import tempfile
            self._workspace_dir = tempfile.mkdtemp(prefix="codingswarm_")
            self._terminal = SwarmTerminal(auto_fix=False, max_fix_attempts=1)
            return True
        except Exception as e:
            logger.debug(f"WorkspaceManager: SwarmTerminal unavailable: {e}")
            return False

    @property
    def available(self) -> bool:
        """Whether terminal-based validation is available."""
        return self._ensure_init()

    @property
    def workspace_dir(self) -> Optional[str]:
        """Return workspace directory path, or None if unavailable."""
        self._ensure_init()
        return self._workspace_dir

    async def write_file(self, filename: str, content: str) -> 'CommandResult':
        """Write a file to the workspace. Returns CommandResult."""
        if not self._ensure_init():
            from ...orchestration.v2.swarm_terminal import CommandResult
            return CommandResult(success=False, command="write_file", output="", error="WorkspaceManager unavailable")
        import os
        filepath = os.path.join(self._workspace_dir, filename)
        return await self._terminal.write_file(filepath, content)

    async def bash(self, command: str, timeout: int = 30) -> 'CommandResult':
        """Execute a bash command in the workspace."""
        if not self._ensure_init():
            from ...orchestration.v2.swarm_terminal import CommandResult
            return CommandResult(success=False, command=command, output="", error="WorkspaceManager unavailable")
        return await self._terminal.execute(command, timeout=timeout, working_dir=self._workspace_dir)

    async def syntax_check(self, filename: str, language: str = "python") -> 'CommandResult':
        """Run syntax check on a file in the workspace."""
        if not self._ensure_init():
            from ...orchestration.v2.swarm_terminal import CommandResult
            return CommandResult(success=False, command="syntax_check", output="", error="WorkspaceManager unavailable")
        import os
        filepath = os.path.join(self._workspace_dir, filename)
        if language == "python":
            return await self._terminal.execute(f"python3 -m py_compile {filepath}", timeout=15, working_dir=self._workspace_dir)
        return await self._terminal.execute(f"cat {filepath}", timeout=5, working_dir=self._workspace_dir)

    async def run_python(self, filename: str, timeout: int = 30) -> 'CommandResult':
        """Run a Python file in the workspace."""
        if not self._ensure_init():
            from ...orchestration.v2.swarm_terminal import CommandResult
            return CommandResult(success=False, command="run_python", output="", error="WorkspaceManager unavailable")
        import os
        filepath = os.path.join(self._workspace_dir, filename)
        return await self._terminal.execute(f"python3 {filepath}", timeout=timeout, working_dir=self._workspace_dir)

    async def run_tests(self, test_filename: Optional[str] = None, timeout: int = 60) -> 'CommandResult':
        """Run pytest on test files in the workspace."""
        if not self._ensure_init():
            from ...orchestration.v2.swarm_terminal import CommandResult
            return CommandResult(success=False, command="run_tests", output="", error="WorkspaceManager unavailable")
        if test_filename:
            import os
            filepath = os.path.join(self._workspace_dir, test_filename)
            cmd = f"python3 -m pytest {filepath} -v --tb=short"
        else:
            cmd = f"python3 -m pytest {self._workspace_dir} -v --tb=short"
        return await self._terminal.execute(cmd, timeout=timeout, working_dir=self._workspace_dir)

    async def pip_install(self, packages: List[str]) -> 'CommandResult':
        """Install pip packages."""
        if not self._ensure_init() or not packages:
            from ...orchestration.v2.swarm_terminal import CommandResult
            return CommandResult(success=not packages, command="pip_install", output="" if not packages else "", error="" if not packages else "WorkspaceManager unavailable")
        pkg_str = " ".join(packages)
        return await self._terminal.execute(f"pip install {pkg_str}", timeout=60, working_dir=self._workspace_dir)

    def cleanup(self):
        """Remove workspace directory. Safe to call multiple times."""
        if self._workspace_dir:
            try:
                import shutil
                shutil.rmtree(self._workspace_dir, ignore_errors=True)
            except Exception:
                pass
            self._workspace_dir = None
