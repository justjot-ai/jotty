"""
SwarmTerminal - Intelligent Terminal Agent for Swarm
====================================================

Intelligent terminal that:
1. Executes commands with smart error detection
2. Searches web for solutions on failure
3. Applies fixes automatically
4. Can write new skills when needed
5. Learns from successful fixes

Integrates terminal-session, web-search, and skill generation.
"""

import os
import re
import json
import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import DSPy for intelligent reasoning
try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False


@dataclass
class CommandResult:
    """Result of a command execution."""
    success: bool
    command: str
    output: str
    error: str = ""
    exit_code: int = 0
    fix_applied: bool = False
    fix_description: str = ""


@dataclass
class ErrorSolution:
    """Solution found for an error."""
    error_pattern: str
    solution: str
    source: str  # 'web', 'cache', 'llm'
    confidence: float
    commands: List[str] = field(default_factory=list)


class ErrorPatternMatcher:
    """Matches errors to known patterns and solutions."""

    # Common error patterns and their fix strategies
    KNOWN_PATTERNS = {
        # Package/dependency errors
        r"ModuleNotFoundError: No module named '(\w+)'": {
            "type": "missing_module",
            "fix_template": "pip install {module}",
        },
        r"command not found: (\w+)": {
            "type": "missing_command",
            "fix_template": "apt-get install {command} || brew install {command}",
        },
        r"Permission denied": {
            "type": "permission",
            "fix_template": "sudo {original_command}",
        },
        r"Connection refused": {
            "type": "connection",
            "fix_template": "# Check if service is running",
        },
        r"No space left on device": {
            "type": "disk_space",
            "fix_template": "df -h && du -sh * | sort -hr | head -20",
        },
        r"ECONNREFUSED|ETIMEDOUT|ENOTFOUND": {
            "type": "network",
            "fix_template": "ping -c 3 8.8.8.8 && curl -I https://google.com",
        },
        r"npm ERR!": {
            "type": "npm_error",
            "fix_template": "rm -rf node_modules package-lock.json && npm install",
        },
        r"pip.*error|Could not find a version": {
            "type": "pip_error",
            "fix_template": "pip install --upgrade pip && pip cache purge",
        },
        r"git.*fatal": {
            "type": "git_error",
            "fix_template": "git status && git stash",
        },
        r"docker.*error|Cannot connect to the Docker daemon": {
            "type": "docker_error",
            "fix_template": "sudo systemctl start docker",
        },
    }

    @classmethod
    def match_error(cls, error_text: str) -> Optional[Dict[str, Any]]:
        """Match error text to known patterns."""
        error_lower = error_text.lower()
        for pattern, fix_info in cls.KNOWN_PATTERNS.items():
            match = re.search(pattern, error_text, re.IGNORECASE)
            if match:
                result = fix_info.copy()
                result["match"] = match
                result["groups"] = match.groups() if match.groups() else ()
                return result
        return None


class CommandAnalyzerSignature(dspy.Signature):
    """Analyze command failure and suggest fixes."""
    command: str = dspy.InputField(desc="The command that failed")
    error_output: str = dspy.InputField(desc="Error output from command")
    system_info: str = dspy.InputField(desc="System information (OS, shell)")

    error_type: str = dspy.OutputField(desc="Type of error: package, permission, network, config, syntax, other")
    root_cause: str = dspy.OutputField(desc="Brief root cause analysis")
    fix_commands: List[str] = dspy.OutputField(desc="List of commands to fix the issue")
    confidence: float = dspy.OutputField(desc="Confidence in fix (0.0-1.0)")


class SkillGeneratorSignature(dspy.Signature):
    """Generate a new skill to solve a problem."""
    problem_description: str = dspy.InputField(desc="Problem that needs a skill")
    error_context: str = dspy.InputField(desc="Error context and failed attempts")
    available_tools: str = dspy.InputField(desc="Available tools/APIs")

    skill_name: str = dspy.OutputField(desc="Name for the new skill (kebab-case)")
    skill_description: str = dspy.OutputField(desc="Description of what the skill does")
    skill_code: str = dspy.OutputField(desc="Python code for tools.py")
    dependencies: List[str] = dspy.OutputField(desc="Required pip packages")


class SwarmTerminal:
    """
    Intelligent terminal for swarm operations.

    Features:
    - Execute commands with automatic error handling
    - Search web for solutions on failure
    - Apply fixes automatically (with approval)
    - Learn from successful fixes
    - Generate new skills when needed
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        auto_fix: bool = True,
        max_fix_attempts: int = 3,
        skills_dir: Optional[str] = None
    ):
        """
        Initialize SwarmTerminal.

        Args:
            config: Jotty configuration
            auto_fix: Automatically apply fixes (default: True)
            max_fix_attempts: Maximum fix attempts per command
            skills_dir: Directory for generated skills
        """
        self.config = config
        self.auto_fix = auto_fix
        self.max_fix_attempts = max_fix_attempts

        # Skills directory
        if skills_dir:
            self.skills_dir = Path(skills_dir)
        else:
            # Default to Jotty skills directory
            current_file = Path(__file__).resolve()
            jotty_root = current_file.parent.parent.parent.parent
            self.skills_dir = jotty_root / "skills"

        # Session management
        self._session_id: Optional[str] = None
        self._session_manager = None

        # Fix cache (learned fixes) with persistence
        self._fix_cache: Dict[str, ErrorSolution] = {}
        self._fix_history: List[Dict[str, Any]] = []

        # Fix persistence directory
        self._fix_db_path = Path.home() / '.jotty' / 'fix_database.json'
        self._load_fix_database()

        # Sandbox guard for untrusted code execution
        self._sandbox = None
        try:
            from .sandbox_manager import SandboxManager
            self._sandbox = SandboxManager()
            logger.debug("SandboxManager attached to SwarmTerminal")
        except Exception as e:
            logger.debug(f"SandboxManager unavailable: {e}")

        # DSPy modules for intelligent analysis
        if DSPY_AVAILABLE:
            self.command_analyzer = dspy.ChainOfThought(CommandAnalyzerSignature)
            self.skill_generator = dspy.ChainOfThought(SkillGeneratorSignature)
        else:
            self.command_analyzer = None
            self.skill_generator = None

        # Load web search skill
        self._web_search = None
        self._load_web_search()

        # Load file operations skill (safer than shell echo/cat for file writing)
        self._write_file = None
        self._read_file = None
        self._load_file_operations()

        logger.info(" SwarmTerminal initialized (intelligent terminal agent)")

    def _load_web_search(self):
        """Load web search capability."""
        try:
            from Jotty.core.skills import get_skills_registry
            registry = get_skills_registry()
            skill = registry.get_skill('web-search')
            if skill and skill.tools:
                self._web_search = skill.tools.get('search_web_tool')
        except Exception:
            pass

        # Fallback: direct import
        if not self._web_search:
            try:
                import sys
                skills_path = str(self.skills_dir)
                if skills_path not in sys.path:
                    sys.path.insert(0, skills_path)
                from web_search.tools import search_web_tool, search_and_scrape_tool
                self._web_search = search_web_tool
                self._web_scrape = search_and_scrape_tool
            except ImportError:
                logger.debug("Web search not available for error resolution")

    def _load_file_operations(self):
        """Load file operations for safe file writing (instead of shell echo/cat)."""
        try:
            from Jotty.core.registry.skills_registry import get_skills_registry
            registry = get_skills_registry()
            registry.init()  # Ensure registry is initialized
            skill = registry.get_skill('file-operations')
            if skill and skill.tools:
                self._write_file = skill.tools.get('write_file_tool')
                self._read_file = skill.tools.get('read_file_tool')
                logger.debug("File operations tools loaded for SwarmTerminal")
        except Exception as e:
            logger.debug(f"Could not load file-operations from registry: {e}")

        # Fallback: direct import from skills directory
        if not self._write_file:
            try:
                import sys
                import importlib.util
                tools_path = self.skills_dir / "file-operations" / "tools.py"
                if tools_path.exists():
                    spec = importlib.util.spec_from_file_location("file_operations_tools", tools_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    self._write_file = getattr(module, 'write_file_tool', None)
                    self._read_file = getattr(module, 'read_file_tool', None)
                    if self._write_file:
                        logger.debug("File operations loaded via direct import")
            except Exception as e:
                logger.debug(f"File operations fallback failed: {e}")

    def _load_fix_database(self) -> None:
        """Load fix database from disk."""
        if not self._fix_db_path.exists():
            return

        try:
            with open(self._fix_db_path) as f:
                data = json.load(f)

            for hash_, fix_data in data.items():
                self._fix_cache[hash_] = ErrorSolution(
                    error_pattern=fix_data['error_pattern'],
                    solution=fix_data['solution_description'],
                    source=fix_data.get('source', 'database'),
                    confidence=fix_data.get('success_count', 1) / max(fix_data.get('success_count', 1) + fix_data.get('fail_count', 0), 1),
                    commands=fix_data.get('solution_commands', [])
                )

            logger.info(f"Loaded {len(self._fix_cache)} fixes from database")

        except Exception as e:
            logger.warning(f"Could not load fix database: {e}")

    def save_fix_database(self) -> None:
        """Save fix database to disk."""
        try:
            self._fix_db_path.parent.mkdir(parents=True, exist_ok=True)

            data = {}
            for hash_, fix in self._fix_cache.items():
                data[hash_] = {
                    'error_pattern': fix.error_pattern,
                    'solution_description': fix.solution,
                    'solution_commands': fix.commands,
                    'source': fix.source,
                    'success_count': int(fix.confidence * 10),  # Approximate
                    'fail_count': int((1 - fix.confidence) * 10) if fix.confidence < 1 else 0
                }

            # Also save from fix history
            for entry in self._fix_history:
                error = entry.get('error', '')
                if error:
                    import hashlib
                    normalized = re.sub(r'\d+', 'N', error.lower())
                    normalized = re.sub(r'/[\w/.-]+', '/PATH', normalized)
                    hash_ = hashlib.md5(normalized.encode()).hexdigest()
                    if hash_ not in data:
                        data[hash_] = {
                            'error_pattern': error[:500],
                            'solution_description': entry.get('description', ''),
                            'solution_commands': entry.get('commands', []),
                            'source': entry.get('source', 'terminal'),
                            'success_count': 1 if entry.get('success', True) else 0,
                            'fail_count': 0 if entry.get('success', True) else 1
                        }

            with open(self._fix_db_path, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved {len(data)} fixes to database")

        except Exception as e:
            logger.warning(f"Could not save fix database: {e}")

    def record_fix(self, error: str, commands: List[str], description: str, source: str, success: bool) -> None:
        """Record a fix for future use."""
        self._fix_history.append({
            'error': error,
            'commands': commands,
            'description': description,
            'source': source,
            'success': success,
            'timestamp': datetime.now().isoformat()
        })

        # Auto-save periodically
        if len(self._fix_history) % 5 == 0:
            self.save_fix_database()

    async def _get_session(self) -> str:
        """Get or create terminal session."""
        if self._session_id:
            return self._session_id

        try:
            # Try to use terminal-session skill
            from Jotty.core.skills import get_skills_registry
            registry = get_skills_registry()
            skill = registry.get_skill('terminal-session')
            if skill and skill.tools:
                create_session = skill.tools.get('terminal_create_session_tool')
                if create_session:
                    result = await create_session({})
                    if result.get('success'):
                        self._session_id = result.get('session_id')
                        self._session_manager = skill.tools
                        return self._session_id
        except Exception as e:
            logger.debug(f"Terminal session skill not available: {e}")

        # Fallback: use subprocess directly
        self._session_id = "subprocess"
        return self._session_id

    async def execute(
        self,
        command: str,
        timeout: int = 60,
        auto_fix: Optional[bool] = None,
        working_dir: Optional[str] = None
    ) -> CommandResult:
        """
        Execute command with intelligent error handling.

        Args:
            command: Shell command to execute
            timeout: Command timeout in seconds
            auto_fix: Override auto_fix setting
            working_dir: Working directory

        Returns:
            CommandResult with output and any applied fixes
        """
        auto_fix = auto_fix if auto_fix is not None else self.auto_fix

        # Execute command
        result = await self._execute_raw(command, timeout, working_dir)

        # If successful, return
        if result.success:
            return result

        # Error handling pipeline
        logger.info(f" Command failed: {command}")
        logger.debug(f"   Error: {result.error[:200]}")

        if not auto_fix:
            return result

        # Try to fix
        for attempt in range(self.max_fix_attempts):
            logger.info(f" Fix attempt {attempt + 1}/{self.max_fix_attempts}")

            # Find solution
            solution = await self._find_solution(command, result.error)

            if not solution:
                logger.warning("   No solution found")
                break

            logger.info(f"   Found solution ({solution.source}): {solution.solution[:100]}")

            # Apply fix commands
            for fix_cmd in solution.commands:
                logger.info(f"   Applying: {fix_cmd}")
                fix_result = await self._execute_raw(fix_cmd, timeout, working_dir)
                if not fix_result.success:
                    logger.warning(f"   Fix command failed: {fix_result.error[:100]}")

            # Retry original command
            result = await self._execute_raw(command, timeout, working_dir)
            if result.success:
                result.fix_applied = True
                result.fix_description = solution.solution

                # Cache successful fix
                self._cache_fix(result.error, solution)
                logger.info(f" Fixed: {solution.solution}")
                return result

        return result

    async def execute_sandboxed(
        self,
        code: str,
        language: str = "python",
        timeout: int = 30,
    ) -> 'CommandResult':
        """
        Execute code in a sandbox (if SandboxManager is available).

        Falls back to _execute_raw if sandbox is unavailable.

        Args:
            code: Code string to execute
            language: Programming language (default: python)
            timeout: Execution timeout in seconds

        Returns:
            CommandResult
        """
        if self._sandbox:
            try:
                from .sandbox_manager import TrustLevel
                # Override default timeout for this call
                old_timeout = self._sandbox.default_timeout
                self._sandbox.default_timeout = timeout
                try:
                    sb_result = await self._sandbox.execute_sandboxed(
                        code=code,
                        trust_level=TrustLevel.SANDBOXED,
                        language=language,
                    )
                finally:
                    self._sandbox.default_timeout = old_timeout
                out_text = getattr(sb_result, 'stdout', '') or str(sb_result.output or '')
                err_text = getattr(sb_result, 'stderr', '') or sb_result.error or ''
                return CommandResult(
                    command=f"[sandbox:{language}] {code[:80]}...",
                    success=sb_result.success,
                    output=out_text,
                    error=err_text,
                )
            except Exception as e:
                logger.warning(f"Sandbox execution failed, falling back: {e}")

        # Fallback: run via _execute_raw (less isolated, but with timeout)
        if language == "python":
            cmd = f'python3 -c {repr(code)}'
        else:
            cmd = code
        return await self._execute_raw(cmd, timeout=timeout)

    async def _execute_raw(
        self,
        command: str,
        timeout: int = 60,
        working_dir: Optional[str] = None
    ) -> CommandResult:
        """Execute command without error handling."""
        import subprocess
        import asyncio
        import signal

        def _run_sync():
            proc = None
            try:
                proc = subprocess.Popen(
                    command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=working_dir,
                    start_new_session=True,
                )
                stdout, stderr = proc.communicate(timeout=timeout)
                return CommandResult(
                    success=proc.returncode == 0,
                    command=command,
                    output=stdout.decode(errors='replace') if stdout else "",
                    error=stderr.decode(errors='replace') if stderr else "",
                    exit_code=proc.returncode,
                )
            except subprocess.TimeoutExpired:
                if proc:
                    # Kill entire process group (shell + children)
                    try:
                        pgid = os.getpgid(proc.pid)
                        os.killpg(pgid, signal.SIGKILL)
                    except (ProcessLookupError, OSError):
                        try:
                            proc.kill()
                        except OSError:
                            pass
                    # Close pipes explicitly (don't drain — may block)
                    for pipe in (proc.stdout, proc.stderr, proc.stdin):
                        if pipe:
                            try:
                                pipe.close()
                            except OSError:
                                pass
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        pass
                return CommandResult(
                    success=False,
                    command=command,
                    output="",
                    error=f"Command timed out after {timeout}s",
                    exit_code=-1,
                )
            except Exception as e:
                return CommandResult(
                    success=False,
                    command=command,
                    output="",
                    error=str(e),
                    exit_code=-1,
                )

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _run_sync)

    async def write_file(self, path: str, content: str, mode: str = 'w') -> CommandResult:
        """
        Write content to file using file-operations tool (safer than shell commands).

        This is preferred over echo/cat shell commands because:
        1. No shell escaping issues
        2. No special character problems
        3. Atomic write operation
        4. Proper error handling
        """
        if self._write_file:
            try:
                result = self._write_file({'path': path, 'content': content, 'mode': mode})
                if result.get('success'):
                    return CommandResult(
                        success=True,
                        command=f"write_file({path})",
                        output=f"Written {result.get('bytes_written', len(content))} bytes to {path}",
                        exit_code=0
                    )
                else:
                    return CommandResult(
                        success=False,
                        command=f"write_file({path})",
                        output="",
                        error=result.get('error', 'Unknown error'),
                        exit_code=1
                    )
            except Exception as e:
                logger.warning(f"write_file tool failed, falling back to shell: {e}")

        # Fallback to shell (escape content properly)
        import shlex
        escaped_content = content.replace("'", "'\\''")
        command = f"printf '%s' '{escaped_content}' > {shlex.quote(path)}"
        return await self._execute_raw(command)

    async def read_file(self, path: str) -> CommandResult:
        """Read file content using file-operations tool."""
        if self._read_file:
            try:
                result = self._read_file({'path': path})
                if result.get('success'):
                    return CommandResult(
                        success=True,
                        command=f"read_file({path})",
                        output=result.get('content', ''),
                        exit_code=0
                    )
                else:
                    return CommandResult(
                        success=False,
                        command=f"read_file({path})",
                        output="",
                        error=result.get('error', 'Unknown error'),
                        exit_code=1
                    )
            except Exception as e:
                logger.warning(f"read_file tool failed, falling back to shell: {e}")

        # Fallback to shell
        import shlex
        return await self._execute_raw(f"cat {shlex.quote(path)}")

    async def _find_solution(
        self,
        command: str,
        error: str
    ) -> Optional[ErrorSolution]:
        """Find solution for error using multiple strategies."""

        # Strategy 1: Check cache
        cache_key = self._error_cache_key(error)
        if cache_key in self._fix_cache:
            logger.debug("   Using cached solution")
            return self._fix_cache[cache_key]

        # Strategy 2: Pattern matching
        pattern_match = ErrorPatternMatcher.match_error(error)
        if pattern_match:
            fix_template = pattern_match.get("fix_template", "")
            groups = pattern_match.get("groups", ())

            # Substitute captured groups
            if groups and "{module}" in fix_template:
                fix_template = fix_template.format(module=groups[0])
            elif groups and "{command}" in fix_template:
                fix_template = fix_template.format(command=groups[0])
            elif "{original_command}" in fix_template:
                fix_template = fix_template.format(original_command=command)

            return ErrorSolution(
                error_pattern=error[:100],
                solution=f"Pattern match: {pattern_match['type']}",
                source="pattern",
                confidence=0.8,
                commands=[fix_template]
            )

        # Strategy 3: Web search for solution
        if self._web_search:
            web_solution = await self._search_web_for_solution(command, error)
            if web_solution:
                return web_solution

        # Strategy 4: LLM analysis
        if self.command_analyzer:
            llm_solution = await self._analyze_with_llm(command, error)
            if llm_solution:
                return llm_solution

        return None

    async def _search_web_for_solution(
        self,
        command: str,
        error: str
    ) -> Optional[ErrorSolution]:
        """Search web for solution to error."""
        if not self._web_search:
            return None

        try:
            # Extract key error message
            error_lines = error.strip().split('\n')
            key_error = error_lines[-1] if error_lines else error[:100]

            # Search query
            query = f"fix {key_error} linux terminal"

            result = self._web_search({'query': query, 'max_results': 5})
            if not result.get('success') or not result.get('results'):
                return None

            # Extract commands from search results
            commands = []
            solution_text = ""

            for item in result['results'][:3]:
                snippet = item.get('snippet', '')
                solution_text += snippet + " "

                # Extract commands from snippet (look for code-like patterns)
                code_patterns = re.findall(
                    r'`([^`]+)`|(?:run|execute|try|use):\s*([^\n.]+)',
                    snippet,
                    re.IGNORECASE
                )
                for groups in code_patterns:
                    cmd = groups[0] or groups[1]
                    if cmd and len(cmd) > 3 and not cmd.startswith('http'):
                        commands.append(cmd.strip())

            if commands:
                return ErrorSolution(
                    error_pattern=key_error[:100],
                    solution=solution_text[:200],
                    source="web",
                    confidence=0.6,
                    commands=commands[:3]  # Limit to 3 commands
                )

        except Exception as e:
            logger.debug(f"Web search failed: {e}")

        return None

    async def _analyze_with_llm(
        self,
        command: str,
        error: str
    ) -> Optional[ErrorSolution]:
        """Use LLM to analyze error and suggest fix."""
        if not self.command_analyzer:
            return None

        try:
            import platform
            system_info = f"OS: {platform.system()} {platform.release()}, Shell: bash"

            result = self.command_analyzer(
                command=command,
                error_output=error[:1000],  # Limit error length
                system_info=system_info
            )

            fix_commands = result.fix_commands
            if isinstance(fix_commands, str):
                # Parse if returned as string
                fix_commands = [cmd.strip() for cmd in fix_commands.split('\n') if cmd.strip()]

            if fix_commands:
                confidence = result.confidence
                if isinstance(confidence, str):
                    try:
                        confidence = float(re.search(r'[\d.]+', confidence).group())
                    except (ValueError, AttributeError):
                        confidence = 0.5

                return ErrorSolution(
                    error_pattern=error[:100],
                    solution=f"{result.error_type}: {result.root_cause}",
                    source="llm",
                    confidence=min(1.0, max(0.0, confidence)),
                    commands=fix_commands[:5]
                )

        except Exception as e:
            logger.debug(f"LLM analysis failed: {e}")

        return None

    def _error_cache_key(self, error: str) -> str:
        """Generate cache key from error."""
        # Normalize error for caching
        normalized = re.sub(r'\d+', 'N', error.lower())
        normalized = re.sub(r'[/\\][\w./\\]+', 'PATH', normalized)
        return normalized[:200]

    def _cache_fix(self, error: str, solution: ErrorSolution):
        """Cache successful fix."""
        cache_key = self._error_cache_key(error)
        self._fix_cache[cache_key] = solution
        self._fix_history.append({
            "timestamp": datetime.now().isoformat(),
            "error": error[:200],
            "solution": solution.solution,
            "commands": solution.commands
        })

    async def generate_skill(
        self,
        problem: str,
        error_context: str = "",
        skill_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a new skill to solve a problem.

        Args:
            problem: Problem description
            error_context: Error context from failed attempts
            skill_name: Optional skill name (auto-generated if not provided)

        Returns:
            Dict with skill info or None if failed
        """
        if not self.skill_generator:
            logger.warning("Skill generation requires DSPy")
            return None

        try:
            # Get available tools for context
            available_tools = "requests, subprocess, json, re, os, pathlib, write_file(path, content), read_file(path)"

            result = self.skill_generator(
                problem_description=problem,
                error_context=error_context[:500],
                available_tools=available_tools
            )

            # Use provided name or generated name
            final_name = skill_name or result.skill_name
            final_name = re.sub(r'[^a-z0-9-]', '-', final_name.lower())

            # Create skill directory
            skill_dir = self.skills_dir / final_name
            skill_dir.mkdir(parents=True, exist_ok=True)

            # Write SKILL.md
            skill_md = f"""# {final_name}

{result.skill_description}

## Description

Auto-generated skill to solve: {problem[:100]}

## Dependencies

{chr(10).join(f'- {dep}' for dep in (result.dependencies or []))}

## Generated

- Date: {datetime.now().isoformat()}
- Source: SwarmTerminal auto-generation
"""
            (skill_dir / "SKILL.md").write_text(skill_md)

            # Write tools.py
            skill_code = result.skill_code
            if not skill_code.startswith('"""'):
                skill_code = f'"""\n{final_name} - Auto-generated skill\n"""\n\n{skill_code}'
            (skill_dir / "tools.py").write_text(skill_code)

            # Install dependencies
            if result.dependencies:
                for dep in result.dependencies:
                    await self.execute(f"pip install {dep}", auto_fix=False)

            logger.info(f" Generated new skill: {final_name}")

            return {
                "name": final_name,
                "path": str(skill_dir),
                "description": result.skill_description,
                "dependencies": result.dependencies
            }

        except Exception as e:
            logger.error(f"Skill generation failed: {e}", exc_info=True)
            return None

    async def diagnose_system(self) -> Dict[str, Any]:
        """Run system diagnostics."""
        diagnostics = {
            "timestamp": datetime.now().isoformat(),
            "checks": {}
        }

        # Check internet connectivity
        net_result = await self.execute("ping -c 1 8.8.8.8", timeout=10, auto_fix=False)
        diagnostics["checks"]["internet"] = net_result.success

        # Check disk space
        disk_result = await self.execute("df -h /", timeout=5, auto_fix=False)
        diagnostics["checks"]["disk"] = disk_result.output if disk_result.success else disk_result.error

        # Check memory
        mem_result = await self.execute("free -h", timeout=5, auto_fix=False)
        diagnostics["checks"]["memory"] = mem_result.output if mem_result.success else "N/A"

        # Check Python environment
        py_result = await self.execute("python3 --version && pip --version", timeout=5, auto_fix=False)
        diagnostics["checks"]["python"] = py_result.output if py_result.success else py_result.error

        # Check common tools
        for tool in ["git", "node", "npm", "docker"]:
            tool_result = await self.execute(f"which {tool}", timeout=5, auto_fix=False)
            diagnostics["checks"][tool] = "installed" if tool_result.success else "not found"

        return diagnostics

    def get_fix_history(self) -> List[Dict[str, Any]]:
        """Get history of applied fixes."""
        return self._fix_history.copy()

    def clear_cache(self) -> None:
        """Clear fix cache."""
        self._fix_cache.clear()
        logger.info("Fix cache cleared")


# Convenience function
def create_swarm_terminal(**kwargs) -> SwarmTerminal:
    """Create SwarmTerminal instance."""
    return SwarmTerminal(**kwargs)
