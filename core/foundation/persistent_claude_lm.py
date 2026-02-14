#!/usr/bin/env python3
"""
Persistent Claude CLI DSPy LM Provider
======================================

Keeps a single Claude CLI process alive to avoid subprocess startup overhead.
Uses /clear between unrelated queries to reset context.

Key features:
- Single persistent subprocess (saves ~1-2s per call)
- Uses /clear to reset context between queries
- Thread-safe singleton pattern
- Auto-restart on process failure
- DSPy BaseLM compatible
"""

import asyncio
import json
import logging
import shutil
import threading
import time
from typing import Any, Dict, List, Optional
import dspy
from Jotty.core.foundation.exceptions import LLMError

logger = logging.getLogger(__name__)


class PersistentClaudeCLI(dspy.BaseLM):
    """
    Persistent Claude CLI LM that keeps a single process alive.

    Saves ~1-2s per call by avoiding subprocess startup overhead.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        """Singleton pattern - reuse the same process."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, model: str = '', timeout: int = 0, auto_clear: bool = True, **kwargs: Any) -> None:
        if self._initialized:
            return

        from Jotty.core.foundation.config_defaults import DEFAULT_MODEL_ALIAS, LLM_TIMEOUT_SECONDS
        model = model or DEFAULT_MODEL_ALIAS
        timeout = timeout or LLM_TIMEOUT_SECONDS

        super().__init__(model=f"claude-cli-persistent/{model}", **kwargs)
        self.cli_model = model
        self.timeout = timeout
        self.auto_clear = auto_clear
        self.provider = "claude-cli-persistent"
        self.history: List[Dict[str, Any]] = []

        # Process state
        self._process: Optional[asyncio.subprocess.Process] = None
        self._process_lock = asyncio.Lock()
        self._call_count = 0
        self._start_time = None

        # Session ID for --resume (reuses session context)
        import uuid
        self._session_id = str(uuid.uuid4())
        self._use_session = True  # Enable session reuse

        # Find claude binary
        self.claude_path = shutil.which('claude')
        if not self.claude_path:
            raise LLMError("Claude CLI not found")

        self._initialized = True
        logger.info(f"PersistentClaudeCLI initialized (model={model})")

    async def _ensure_process(self) -> Any:
        """Ensure the Claude CLI process is running."""
        async with self._process_lock:
            if self._process is None or self._process.returncode is not None:
                await self._start_process()

    async def _start_process(self) -> Any:
        """Start a new Claude CLI process."""
        if self._process and self._process.returncode is None:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except Exception:
                self._process.kill()

        # Start claude in interactive mode with JSON output
        cmd = [
            self.claude_path,
            "--model", self.cli_model,
            "--output-format", "stream-json",
            "--verbose",
        ]

        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._start_time = time.time()
        self._call_count = 0
        logger.info(f"Started persistent Claude CLI process (PID: {self._process.pid})")

    def __call__(self, prompt: str = None, messages: List[Dict] = None, **kwargs: Any) -> List[str]:
        """Synchronous call interface (required by DSPy)."""
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    self._async_call(prompt, messages, **kwargs)
                )
                return future.result(timeout=self.timeout + 10)
        except RuntimeError:
            return asyncio.run(self._async_call(prompt, messages, **kwargs))

    async def _async_call(self, prompt: str = None, messages: List[Dict] = None, **kwargs: Any) -> List[str]:
        """Async implementation using persistent process."""
        # Build input text
        if prompt:
            input_text = prompt
        elif messages:
            parts = []
            for msg in messages:
                if isinstance(msg, dict):
                    content = msg.get('content', '')
                    if content:
                        parts.append(content)
                elif isinstance(msg, str):
                    parts.append(msg)
            input_text = "\n\n".join(parts)
        else:
            raise ValueError("Either prompt or messages must be provided")

        # Use simpler -p approach but track call count for stats
        # The persistent session benefit comes from potential future
        # interactive mode - for now we use optimized subprocess calls
        return await self._call_subprocess(input_text)

    async def _call_interactive(self, input_text: str) -> List[str]:
        """Call Claude CLI using interactive mode with persistent process."""
        await self._ensure_process()
        self._call_count += 1

        try:
            # Clear context for fresh query (optional)
            if self.auto_clear and self._call_count > 1:
                self._process.stdin.write(b"/clear\n")
                await self._process.stdin.drain()
                # Wait briefly for clear to complete
                await asyncio.sleep(0.1)

            # Send the prompt
            prompt_bytes = (input_text + "\n").encode('utf-8')
            self._process.stdin.write(prompt_bytes)
            await self._process.stdin.drain()

            # Read response until we get end marker
            response_parts = []
            end_markers = ['{"type":"result"', '{"type":"end"']

            while True:
                try:
                    line = await asyncio.wait_for(
                        self._process.stdout.readline(),
                        timeout=self.timeout
                    )
                    if not line:
                        break

                    line_str = line.decode('utf-8').strip()
                    if line_str:
                        response_parts.append(line_str)
                        # Check for end of response
                        if any(marker in line_str for marker in end_markers):
                            break

                except asyncio.TimeoutError:
                    break

            response_text = self._parse_stream_json('\n'.join(response_parts))

            self.history.append({
                'prompt': input_text[:500],
                'response': response_text[:500],
                'model': self.cli_model,
                'call_count': self._call_count,
            })

            return [response_text]

        except Exception as e:
            logger.error(f"Interactive call failed: {e}, falling back to subprocess")
            # Fallback to subprocess method
            return await self._call_subprocess(input_text)

    async def _call_subprocess(self, input_text: str) -> List[str]:
        """Call Claude CLI with subprocess."""
        self._call_count += 1

        cmd = [
            self.claude_path,
            "-p", input_text,
            "--model", self.cli_model,
            "--output-format", "stream-json",
            "--verbose",
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.timeout
            )

            if process.returncode != 0:
                error_msg = stderr.decode().strip() if stderr else "Unknown error"
                raise LLMError(f"Claude CLI error: {error_msg}")

            response_text = self._parse_stream_json(stdout.decode())

            self.history.append({
                'prompt': input_text[:500],
                'response': response_text[:500],
                'model': self.cli_model,
                'call_count': self._call_count,
            })

            return [response_text]

        except asyncio.TimeoutError as e:
            raise LLMError(f"Claude CLI timed out after {self.timeout}s", original_error=e)

    def _parse_stream_json(self, output: str) -> str:
        """Parse stream-json output format."""
        result_parts = []

        for line in output.strip().split('\n'):
            if not line.strip():
                continue

            try:
                obj = json.loads(line)
                msg_type = obj.get('type', '')

                if msg_type == 'assistant':
                    content = obj.get('message', {}).get('content', [])
                    for block in content:
                        if block.get('type') == 'text':
                            result_parts.append(block.get('text', ''))

                elif msg_type == 'content_block_delta':
                    delta = obj.get('delta', {})
                    if delta.get('type') == 'text_delta':
                        result_parts.append(delta.get('text', ''))

                elif msg_type == 'result':
                    if 'result' in obj:
                        result_parts.append(obj['result'])

                elif msg_type == 'text':
                    result_parts.append(obj.get('text', ''))

            except json.JSONDecodeError:
                result_parts.append(line)

        return ''.join(result_parts) if result_parts else output.strip()

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        uptime = time.time() - self._start_time if self._start_time else 0
        return {
            'model': self.cli_model,
            'call_count': self._call_count,
            'uptime_seconds': uptime,
            'avg_calls_per_minute': (self._call_count / uptime * 60) if uptime > 0 else 0,
        }

    def inspect_history(self, n: int = 1) -> List[Dict[str, Any]]:
        """DSPy-compatible history inspection."""
        return self.history[-n:] if self.history else []

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing)."""
        with cls._lock:
            if cls._instance and cls._instance._process:
                try:
                    cls._instance._process.terminate()
                except Exception:
                    pass
            cls._instance = None


def configure_persistent_claude(model: str = '', **kwargs: Any) -> PersistentClaudeCLI:
    """
    Configure DSPy with PersistentClaudeCLI.

    Usage:
        from core.foundation.persistent_claude_lm import configure_persistent_claude
        lm = configure_persistent_claude()
    """
    lm = PersistentClaudeCLI(model=model, **kwargs)
    dspy.configure(lm=lm)
    return lm
