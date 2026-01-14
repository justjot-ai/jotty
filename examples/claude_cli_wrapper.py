"""
Claude CLI Wrapper for DSPy
Enables DSPy to use Claude via Claude Code CLI
"""
import subprocess
import json
import os
from typing import Any, Dict, List, Optional


class ClaudeCLILM:
    """Language Model wrapper for Claude CLI"""

    def __init__(self, model: str = "sonnet", max_tokens: int = 8000):
        """
        Initialize Claude CLI wrapper

        Args:
            model: Claude model (sonnet, opus, haiku)
            max_tokens: Maximum tokens per request
        """
        self.model = model
        self.max_tokens = max_tokens
        self.claude_bin = self._find_claude_cli()

    def _find_claude_cli(self) -> str:
        """Find Claude CLI binary"""
        # Check common locations
        paths = [
            "/usr/local/bin/claude",
            os.path.expanduser("~/.local/bin/claude"),
        ]

        for path in paths:
            if os.path.exists(path):
                return path

        # Try which
        try:
            result = subprocess.run(
                ["which", "claude"],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            raise FileNotFoundError(
                "Claude CLI not found. Install: npm install -g @anthropic-ai/claude-code"
            )

    def __call__(self, prompt: str, **kwargs) -> str:
        """
        Call Claude CLI with prompt

        Args:
            prompt: The prompt to send to Claude
            **kwargs: Additional arguments (temperature, etc.)

        Returns:
            Claude's response text
        """
        # Build command
        cmd = [
            self.claude_bin,
            "--no-interactive",
            "--model", self.model,
        ]

        # Add prompt
        # Note: Claude CLI expects stdin or --file
        # For DSPy, we'll use stdin
        try:
            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=120,  # 2 minute timeout
            )

            if result.returncode != 0:
                raise RuntimeError(f"Claude CLI error: {result.stderr}")

            return result.stdout.strip()

        except subprocess.TimeoutExpired:
            raise RuntimeError("Claude CLI timed out after 2 minutes")
        except Exception as e:
            raise RuntimeError(f"Claude CLI execution failed: {e}")

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate response (compatible with DSPy's expected interface)

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Max tokens to generate

        Returns:
            Dict with 'choices' containing the response
        """
        # Convert messages to single prompt
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        prompt = "\n\n".join(prompt_parts)

        # Call Claude
        response_text = self(prompt, temperature=temperature, max_tokens=max_tokens)

        # Format as expected by DSPy
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    }
                }
            ],
            "usage": {
                "total_tokens": len(response_text.split())  # Rough estimate
            }
        }
