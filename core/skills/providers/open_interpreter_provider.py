"""
Open Interpreter Provider for Jotty V2
=======================================

Integrates Open Interpreter for natural language code execution.
https://github.com/openinterpreter/open-interpreter

Capabilities:
- Execute Python, JavaScript, Shell code
- Data analysis and visualization
- File manipulation
- System control
- Natural language to code conversion
"""

import time
import logging
import asyncio
from typing import Any, Dict, List, Optional

from .base import SkillProvider, SkillCategory, ProviderCapability, ProviderResult

logger = logging.getLogger(__name__)

# Try to import Open Interpreter
try:
    from interpreter import Interpreter
    INTERPRETER_AVAILABLE = True
except ImportError:
    INTERPRETER_AVAILABLE = False
    Interpreter = None


class OpenInterpreterProvider(SkillProvider):
    """
    Provider using Open Interpreter for code execution.

    Features:
    - Natural language to code conversion
    - Multi-language support (Python, JS, Shell)
    - Streaming output
    - Safe mode for sandboxed execution
    """

    name = "open-interpreter"
    version = "0.4.0"
    description = "Natural language code execution via Open Interpreter"

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)

        self.capabilities = [
            ProviderCapability(
                category=SkillCategory.CODE_EXECUTION,
                actions=["run_python", "run_js", "run_shell", "analyze_data", "create_chart"],
                max_concurrent=3,
                estimated_latency_ms=3000,
            ),
            ProviderCapability(
                category=SkillCategory.TERMINAL,
                actions=["execute_command", "file_operations", "system_info"],
                estimated_latency_ms=1000,
            ),
            ProviderCapability(
                category=SkillCategory.DATA_EXTRACTION,
                actions=["parse_data", "transform", "analyze", "visualize"],
                estimated_latency_ms=5000,
            ),
        ]

        # Configuration
        self.model = config.get('model', 'gpt-4') if config else 'gpt-4'
        self.safe_mode = config.get('safe_mode', False) if config else False
        self.auto_run = config.get('auto_run', True) if config else True
        self.timeout = config.get('timeout', 120) if config else 120

        # Interpreter instance
        self._interpreter = None

    async def initialize(self) -> bool:
        """Initialize Open Interpreter."""
        if not INTERPRETER_AVAILABLE:
            logger.warning("Open Interpreter not installed. Run: pip install open-interpreter")
            # Try fallback with direct code execution
            self.is_initialized = True
            self.is_available = True
            logger.info(f" {self.name} provider initialized (fallback mode)")
            return True

        try:
            # Create interpreter instance
            self._interpreter = Interpreter()
            self._interpreter.llm.model = self.model
            self._interpreter.auto_run = self.auto_run
            self._interpreter.safe_mode = self.safe_mode

            self.is_initialized = True
            self.is_available = True
            logger.info(f" {self.name} provider initialized (model: {self.model})")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Open Interpreter: {e}")
            # Still mark as available for fallback
            self.is_initialized = True
            self.is_available = True
            return True

    def get_categories(self) -> List[SkillCategory]:
        return [
            SkillCategory.CODE_EXECUTION,
            SkillCategory.TERMINAL,
            SkillCategory.DATA_EXTRACTION,
        ]

    async def execute(self, task: str, context: Dict[str, Any] = None) -> ProviderResult:
        """Execute code task via natural language."""
        start_time = time.time()
        context = context or {}

        try:
            if INTERPRETER_AVAILABLE and self._interpreter:
                return await self._execute_interpreter(task, context)
            else:
                return await self._execute_fallback(task, context)

        except Exception as e:
            logger.error(f"Open Interpreter error: {e}")
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

    async def _execute_interpreter(self, task: str, context: Dict) -> ProviderResult:
        """Execute using Open Interpreter."""
        start_time = time.time()

        try:
            logger.info(f" Open Interpreter executing: {task[:50]}...")

            # Run in thread pool to avoid blocking
            loop = asyncio.get_running_loop()

            def run_interpreter():
                messages = self._interpreter.chat(task, display=False)
                return messages

            messages = await asyncio.wait_for(
                loop.run_in_executor(None, run_interpreter),
                timeout=self.timeout
            )

            # Extract results
            output = {
                'task': task,
                'messages': messages,
                'code_executed': [],
                'outputs': [],
            }

            # Parse messages for code and outputs
            for msg in messages:
                if msg.get('type') == 'code':
                    output['code_executed'].append({
                        'language': msg.get('format', 'python'),
                        'code': msg.get('content', ''),
                    })
                elif msg.get('type') == 'console':
                    output['outputs'].append(msg.get('content', ''))

            execution_time = time.time() - start_time

            result = ProviderResult(
                success=True,
                output=output,
                execution_time=execution_time,
                provider_name=self.name,
                category=SkillCategory.CODE_EXECUTION,
                confidence=0.9,
            )

            self.record_execution(result)
            logger.info(f" Open Interpreter completed in {execution_time:.2f}s")
            return result

        except asyncio.TimeoutError:
            return ProviderResult(
                success=False,
                output=None,
                error=f"Task timed out after {self.timeout}s",
                execution_time=time.time() - start_time,
                provider_name=self.name,
                retryable=True,
            )

    async def _execute_fallback(self, task: str, context: Dict) -> ProviderResult:
        """Execute using direct code execution fallback."""
        start_time = time.time()

        try:
            task_lower = task.lower()

            # Determine what kind of execution
            if 'python' in task_lower or context.get('language') == 'python':
                return await self._run_python(task, context)
            elif 'shell' in task_lower or 'bash' in task_lower:
                return await self._run_shell(task, context)
            elif 'javascript' in task_lower or 'js' in task_lower:
                return await self._run_javascript(task, context)
            else:
                # Default to Python
                return await self._run_python(task, context)

        except Exception as e:
            return ProviderResult(
                success=False,
                output=None,
                error=str(e),
                execution_time=time.time() - start_time,
                provider_name=self.name,
            )

    async def _run_python(self, task: str, context: Dict) -> ProviderResult:
        """Run Python code."""
        code = context.get('code')

        if not code:
            # Try to extract code from task
            code = self._extract_code(task, 'python')

        if not code:
            return ProviderResult(
                success=False,
                output=None,
                error="No Python code to execute. Use context={'code': 'print(1)'}",
                category=SkillCategory.CODE_EXECUTION,
            )

        try:
            # Execute Python code
            local_vars = {}
            exec(code, {'__builtins__': __builtins__}, local_vars)

            return ProviderResult(
                success=True,
                output={
                    'language': 'python',
                    'code': code,
                    'result': local_vars,
                },
                category=SkillCategory.CODE_EXECUTION,
            )

        except Exception as e:
            return ProviderResult(
                success=False,
                output={'code': code},
                error=str(e),
                category=SkillCategory.CODE_EXECUTION,
            )

    async def _run_shell(self, task: str, context: Dict) -> ProviderResult:
        """Run shell command."""
        command = context.get('command') or self._extract_code(task, 'shell')

        if not command:
            return ProviderResult(
                success=False,
                output=None,
                error="No shell command to execute",
                category=SkillCategory.TERMINAL,
            )

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            return ProviderResult(
                success=process.returncode == 0,
                output={
                    'language': 'shell',
                    'command': command,
                    'stdout': stdout.decode(),
                    'stderr': stderr.decode(),
                    'return_code': process.returncode,
                },
                error=stderr.decode() if process.returncode != 0 else "",
                category=SkillCategory.TERMINAL,
            )

        except Exception as e:
            return ProviderResult(
                success=False,
                output={'command': command},
                error=str(e),
                category=SkillCategory.TERMINAL,
            )

    async def _run_javascript(self, task: str, context: Dict) -> ProviderResult:
        """Run JavaScript code via Node.js."""
        code = context.get('code') or self._extract_code(task, 'javascript')

        if not code:
            return ProviderResult(
                success=False,
                output=None,
                error="No JavaScript code to execute",
                category=SkillCategory.CODE_EXECUTION,
            )

        try:
            # Run via node
            process = await asyncio.create_subprocess_exec(
                'node', '-e', code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            return ProviderResult(
                success=process.returncode == 0,
                output={
                    'language': 'javascript',
                    'code': code,
                    'stdout': stdout.decode(),
                    'stderr': stderr.decode(),
                },
                error=stderr.decode() if process.returncode != 0 else "",
                category=SkillCategory.CODE_EXECUTION,
            )

        except Exception as e:
            return ProviderResult(
                success=False,
                output={'code': code},
                error=str(e),
                category=SkillCategory.CODE_EXECUTION,
            )

    def _extract_code(self, task: str, language: str) -> Optional[str]:
        """Extract code from task description."""
        import re

        # Look for code blocks
        patterns = [
            rf'```{language}\n?(.*?)```',
            rf'```\n?(.*?)```',
            r'`([^`]+)`',
        ]

        for pattern in patterns:
            match = re.search(pattern, task, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return None

    # Convenience methods

    async def run_code(self, code: str, language: str = 'python') -> ProviderResult:
        """Run code in specified language."""
        return await self.execute(f"Run {language} code", context={'code': code, 'language': language})

    async def analyze_data(self, data_path: str) -> ProviderResult:
        """Analyze a data file."""
        return await self.execute(f"Analyze the data in {data_path} and provide insights")

    async def create_chart(self, data: Dict, chart_type: str = 'bar') -> ProviderResult:
        """Create a chart from data."""
        return await self.execute(
            f"Create a {chart_type} chart from the provided data",
            context={'data': data}
        )

    def reset(self):
        """Reset the interpreter state."""
        if self._interpreter:
            self._interpreter.reset()
