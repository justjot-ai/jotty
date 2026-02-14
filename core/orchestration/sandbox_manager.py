"""
SandboxManager - Secure Code Execution with Trust Levels
=========================================================

Routes execution based on trust level with multiple sandbox backends:
- E2B (Firecracker microVMs) - most isolated, cloud-based
- Docker - container isolation, self-hosted
- Subprocess - basic isolation, always available fallback

Trust Levels:
- TRUSTED: Execute directly (built-in Jotty skills, verified packages)
- SANDBOXED: Execute in basic sandbox (community packages)
- DANGEROUS: Execute in isolated sandbox only (untrusted/generated code)

Usage:
    from core.orchestration.sandbox_manager import SandboxManager, TrustLevel

    sandbox = SandboxManager()
    result = await sandbox.execute_sandboxed(
        code="print('Hello')",
        trust_level=TrustLevel.SANDBOXED
    )
"""

import os
import sys
import logging
import asyncio
import tempfile
import subprocess
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TrustLevel(Enum):
    """Trust level for code/provider execution."""
    TRUSTED = "trusted"      # Execute directly - verified safe
    SANDBOXED = "sandboxed"  # Execute in basic sandbox
    DANGEROUS = "dangerous"  # Execute in isolated sandbox only


class SandboxType(Enum):
    """Available sandbox backends."""
    NONE = "none"           # Direct execution (trusted only)
    E2B = "e2b"             # E2B Firecracker microVM
    DOCKER = "docker"       # Docker container
    SUBPROCESS = "subprocess"  # Restricted subprocess (fallback)


@dataclass
class SandboxConfig:
    """Configuration for a sandbox execution."""
    sandbox_type: SandboxType
    timeout: int = 120  # seconds
    memory_limit: str = "512m"
    cpu_limit: float = 1.0
    network_enabled: bool = False
    filesystem_access: List[str] = field(default_factory=list)
    environment: Dict[str, str] = field(default_factory=dict)


@dataclass
class SandboxResult:
    """Result from sandbox execution."""
    success: bool
    output: Any
    error: str = ""
    execution_time: float = 0.0
    sandbox_type: str = ""
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class SandboxManager:
    """
    Manages sandboxed code execution with multiple backends.

    Auto-detects available backends and routes execution based on trust level.
    Priority: E2B > Docker > Subprocess

    Attributes:
        e2b_available: Whether E2B is installed and configured
        docker_available: Whether Docker is available
        default_sandbox: Fallback sandbox type when preferred not available
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize SandboxManager.

        Args:
            config: Optional configuration dict with keys:
                - e2b_api_key: E2B API key (or use E2B_API_KEY env var)
                - docker_image: Default Docker image
                - timeout: Default timeout in seconds
        """
        self.config = config or {}
        self.e2b_available = False
        self.docker_available = False
        self.e2b_api_key = None
        self.docker_image = self.config.get('docker_image', 'python:3.11-slim')
        self.default_timeout = self.config.get('timeout', 120)

        # Initialize backends
        self._init_backends()

    def _init_backends(self) -> Any:
        """Check availability of E2B and Docker backends."""
        # Check E2B
        try:
            import e2b_code_interpreter
            self.e2b_api_key = self.config.get('e2b_api_key') or os.getenv('E2B_API_KEY')
            if self.e2b_api_key:
                self.e2b_available = True
                logger.info(" E2B sandbox available")
            else:
                logger.debug("E2B installed but no API key configured")
        except ImportError:
            logger.debug("E2B not installed (pip install e2b-code-interpreter)")

        # Check Docker
        try:
            result = subprocess.run(
                ['docker', 'info'],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                self.docker_available = True
                logger.info(" Docker sandbox available")
            else:
                logger.debug("Docker not running or accessible")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.debug("Docker not installed or not running")

        # Log fallback
        if not self.e2b_available and not self.docker_available:
            logger.info("ℹ Using subprocess sandbox (fallback)")

    def get_sandbox_config(self, trust_level: TrustLevel) -> SandboxConfig:
        """
        Get appropriate sandbox configuration for trust level.

        Args:
            trust_level: The trust level of the code to execute

        Returns:
            SandboxConfig with appropriate settings
        """
        if trust_level == TrustLevel.TRUSTED:
            return SandboxConfig(
                sandbox_type=SandboxType.NONE,
                timeout=self.default_timeout,
                network_enabled=True,
            )

        elif trust_level == TrustLevel.SANDBOXED:
            # Use Docker if available, else subprocess
            if self.docker_available:
                return SandboxConfig(
                    sandbox_type=SandboxType.DOCKER,
                    timeout=self.default_timeout,
                    memory_limit="512m",
                    cpu_limit=1.0,
                    network_enabled=False,
                )
            else:
                return SandboxConfig(
                    sandbox_type=SandboxType.SUBPROCESS,
                    timeout=min(self.default_timeout, 60),
                    network_enabled=False,
                )

        else:  # DANGEROUS
            # Use E2B if available, else Docker, else subprocess with strict limits
            if self.e2b_available:
                return SandboxConfig(
                    sandbox_type=SandboxType.E2B,
                    timeout=self.default_timeout,
                    memory_limit="256m",
                    cpu_limit=0.5,
                    network_enabled=False,
                )
            elif self.docker_available:
                return SandboxConfig(
                    sandbox_type=SandboxType.DOCKER,
                    timeout=60,
                    memory_limit="256m",
                    cpu_limit=0.5,
                    network_enabled=False,
                )
            else:
                return SandboxConfig(
                    sandbox_type=SandboxType.SUBPROCESS,
                    timeout=30,  # Very short for dangerous code
                    network_enabled=False,
                )

    async def execute_sandboxed(
        self,
        code: str,
        trust_level: TrustLevel,
        context: Optional[Dict[str, Any]] = None,
        language: str = "python"
    ) -> SandboxResult:
        """
        Execute code in appropriate sandbox based on trust level.

        Args:
            code: Code to execute
            trust_level: Trust level determining sandbox type
            context: Optional context with variables/files to inject
            language: Programming language (default: python)

        Returns:
            SandboxResult with output and execution info
        """
        import time
        start_time = time.time()
        context = context or {}

        config = self.get_sandbox_config(trust_level)
        logger.info(f" Executing with {config.sandbox_type.value} sandbox (trust: {trust_level.value})")

        try:
            if config.sandbox_type == SandboxType.NONE:
                result = await self._execute_direct(code, config, context, language)
            elif config.sandbox_type == SandboxType.E2B:
                result = await self._execute_e2b(code, config, context, language)
            elif config.sandbox_type == SandboxType.DOCKER:
                result = await self._execute_docker(code, config, context, language)
            else:
                result = await self._execute_subprocess(code, config, context, language)

            result.execution_time = time.time() - start_time
            result.sandbox_type = config.sandbox_type.value
            return result

        except Exception as e:
            logger.error(f"Sandbox execution failed: {e}")
            return SandboxResult(
                success=False,
                output=None,
                error=str(e),
                execution_time=time.time() - start_time,
                sandbox_type=config.sandbox_type.value,
            )

    async def _execute_direct(
        self,
        code: str,
        config: SandboxConfig,
        context: Dict[str, Any],
        language: str
    ) -> SandboxResult:
        """Execute code directly (trusted code only)."""
        if language != "python":
            return SandboxResult(
                success=False,
                output=None,
                error=f"Direct execution only supports Python, got {language}",
            )

        try:
            # Create execution namespace with context
            namespace = dict(context)

            # Execute code
            exec(code, namespace)

            # Extract result (look for 'result' variable)
            output = namespace.get('result', namespace.get('output', None))

            return SandboxResult(
                success=True,
                output=output,
                metadata={'namespace_keys': list(namespace.keys())},
            )

        except Exception as e:
            return SandboxResult(
                success=False,
                output=None,
                error=str(e),
            )

    async def _execute_e2b(
        self,
        code: str,
        config: SandboxConfig,
        context: Dict[str, Any],
        language: str
    ) -> SandboxResult:
        """Execute code in E2B Firecracker microVM."""
        if not self.e2b_available:
            logger.warning("E2B not available, falling back to subprocess")
            return await self._execute_subprocess(code, config, context, language)

        try:
            from e2b_code_interpreter import Sandbox

            async with Sandbox(api_key=self.e2b_api_key) as sandbox:
                # Set timeout
                sandbox.timeout = config.timeout

                # Inject context variables
                if context:
                    context_code = self._build_context_code(context)
                    await sandbox.run_code(context_code)

                # Execute main code
                execution = await sandbox.run_code(code)

                # Collect output
                stdout_parts = []
                stderr_parts = []

                for log in execution.logs:
                    if log.type == 'stdout':
                        stdout_parts.append(log.line)
                    elif log.type == 'stderr':
                        stderr_parts.append(log.line)

                stdout = '\n'.join(stdout_parts)
                stderr = '\n'.join(stderr_parts)

                # Check for errors
                if execution.error:
                    return SandboxResult(
                        success=False,
                        output=None,
                        error=execution.error.message,
                        stdout=stdout,
                        stderr=stderr,
                        exit_code=1,
                    )

                # Get result
                result = execution.results[0] if execution.results else None
                output = result.text if result else stdout

                return SandboxResult(
                    success=True,
                    output=output,
                    stdout=stdout,
                    stderr=stderr,
                    exit_code=0,
                    metadata={'results_count': len(execution.results)},
                )

        except Exception as e:
            logger.error(f"E2B execution failed: {e}")
            # Fallback to subprocess
            return await self._execute_subprocess(code, config, context, language)

    async def _execute_docker(
        self,
        code: str,
        config: SandboxConfig,
        context: Dict[str, Any],
        language: str
    ) -> SandboxResult:
        """Execute code in Docker container."""
        if not self.docker_available:
            logger.warning("Docker not available, falling back to subprocess")
            return await self._execute_subprocess(code, config, context, language)

        try:
            # Create temp file with code
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.py' if language == 'python' else '.js',
                delete=False
            ) as f:
                # Prepend context
                if context:
                    f.write(self._build_context_code(context))
                    f.write('\n')
                f.write(code)
                code_file = f.name

            try:
                # Build Docker command
                docker_cmd = [
                    'docker', 'run',
                    '--rm',  # Remove container after execution
                    '--network=none' if not config.network_enabled else '',
                    f'--memory={config.memory_limit}',
                    f'--cpus={config.cpu_limit}',
                    '-v', f'{code_file}:/code/script.py:ro',
                    '-w', '/code',
                    self.docker_image,
                    'python', 'script.py'
                ]

                # Filter empty args
                docker_cmd = [arg for arg in docker_cmd if arg]

                # Run with timeout
                process = await asyncio.create_subprocess_exec(
                    *docker_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=config.timeout
                    )
                except asyncio.TimeoutError:
                    process.kill()
                    return SandboxResult(
                        success=False,
                        output=None,
                        error=f"Execution timed out after {config.timeout}s",
                        exit_code=-1,
                    )

                stdout_str = stdout.decode('utf-8', errors='replace')
                stderr_str = stderr.decode('utf-8', errors='replace')

                return SandboxResult(
                    success=process.returncode == 0,
                    output=stdout_str,
                    error=stderr_str if process.returncode != 0 else "",
                    stdout=stdout_str,
                    stderr=stderr_str,
                    exit_code=process.returncode or 0,
                )

            finally:
                # Cleanup temp file
                Path(code_file).unlink(missing_ok=True)

        except Exception as e:
            logger.error(f"Docker execution failed: {e}")
            return await self._execute_subprocess(code, config, context, language)

    async def _execute_subprocess(
        self,
        code: str,
        config: SandboxConfig,
        context: Dict[str, Any],
        language: str
    ) -> SandboxResult:
        """Execute code in restricted subprocess (fallback)."""
        if language not in ('python', 'python3'):
            return SandboxResult(
                success=False,
                output=None,
                error=f"Subprocess sandbox only supports Python, got {language}",
            )

        try:
            # Create temp file with code
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.py',
                delete=False
            ) as f:
                # Prepend context
                if context:
                    f.write(self._build_context_code(context))
                    f.write('\n')
                f.write(code)
                code_file = f.name

            try:
                # Build restricted environment
                env = os.environ.copy()
                # Remove potentially dangerous env vars
                for key in list(env.keys()):
                    if any(s in key.upper() for s in ['SECRET', 'KEY', 'TOKEN', 'PASSWORD', 'CREDENTIAL']):
                        del env[key]

                # Run with timeout (start_new_session for clean kill)
                process = await asyncio.create_subprocess_exec(
                    sys.executable, code_file,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env,
                    start_new_session=True,
                )

                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=config.timeout
                    )
                except asyncio.TimeoutError:
                    # Kill entire process group for clean teardown
                    import signal as _sig
                    try:
                        os.killpg(os.getpgid(process.pid), _sig.SIGKILL)
                    except (ProcessLookupError, OSError):
                        process.kill()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=2)
                    except asyncio.TimeoutError:
                        pass
                    return SandboxResult(
                        success=False,
                        output=None,
                        error=f"Execution timed out after {config.timeout}s",
                        exit_code=-1,
                    )

                stdout_str = stdout.decode('utf-8', errors='replace')
                stderr_str = stderr.decode('utf-8', errors='replace')

                return SandboxResult(
                    success=process.returncode == 0,
                    output=stdout_str,
                    error=stderr_str if process.returncode != 0 else "",
                    stdout=stdout_str,
                    stderr=stderr_str,
                    exit_code=process.returncode or 0,
                )

            finally:
                # Cleanup temp file
                Path(code_file).unlink(missing_ok=True)

        except Exception as e:
            logger.error(f"Subprocess execution failed: {e}")
            return SandboxResult(
                success=False,
                output=None,
                error=str(e),
            )

    def _build_context_code(self, context: Dict[str, Any]) -> str:
        """Build Python code to inject context variables."""
        lines = ["# Injected context"]
        for key, value in context.items():
            if isinstance(value, str):
                lines.append(f'{key} = """{value}"""')
            elif isinstance(value, (int, float, bool, type(None))):
                lines.append(f'{key} = {value!r}')
            elif isinstance(value, (list, dict)):
                import json
                lines.append(f'{key} = {json.dumps(value)}')
            # Skip complex types
        return '\n'.join(lines)

    def get_available_backends(self) -> List[str]:
        """Get list of available sandbox backends."""
        backends = ['subprocess']  # Always available
        if self.docker_available:
            backends.append('docker')
        if self.e2b_available:
            backends.append('e2b')
        return backends

    def get_status(self) -> Dict[str, Any]:
        """Get sandbox manager status."""
        return {
            'e2b_available': self.e2b_available,
            'docker_available': self.docker_available,
            'available_backends': self.get_available_backends(),
            'default_timeout': self.default_timeout,
            'docker_image': self.docker_image,
        }


# =============================================================================
# Convenience Functions
# =============================================================================

_sandbox_manager: Optional[SandboxManager] = None


def get_sandbox_manager(config: Optional[Dict[str, Any]] = None) -> SandboxManager:
    """Get singleton SandboxManager instance."""
    global _sandbox_manager
    if _sandbox_manager is None:
        _sandbox_manager = SandboxManager(config)
    return _sandbox_manager


async def execute_safely(
    code: str,
    trust_level: TrustLevel = TrustLevel.SANDBOXED,
    context: Optional[Dict[str, Any]] = None
) -> SandboxResult:
    """
    Convenience function to execute code safely.

    Args:
        code: Code to execute
        trust_level: Trust level (default: SANDBOXED)
        context: Optional context variables

    Returns:
        SandboxResult
    """
    manager = get_sandbox_manager()
    return await manager.execute_sandboxed(code, trust_level, context)
