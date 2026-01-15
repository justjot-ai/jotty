#!/usr/bin/env python3
"""
OpenCode DSPy LM Provider
Integrates OpenCode as a DSPy Language Model provider
Supports multiple models via OpenCode (GLM, Claude, GPT, etc.)

OpenCode is like an AI SDK - it provides access to multiple models.
This should be in Jotty behind DSPy LM abstraction, not in supervisor.
"""
import os
import subprocess
import json
from typing import Iterator, Optional
import dspy
from dspy.clients.base_lm import BaseLM

REMOTE_HOST = os.getenv("OPENCODE_REMOTE_HOST", "cmd.prod.ancillary")
OPENCODE_BIN = os.getenv("OPENCODE_BIN", "opencode")


class OpenCodeLM(BaseLM):
    """
    DSPy Language Model wrapper for OpenCode
    
    OpenCode is an AI SDK that provides access to multiple models (GLM, Claude, GPT, etc.).
    This integrates OpenCode as a DSPy LM provider, making it available through DSPy's
    standard LM interface.
    
    Architecture:
    - Supervisor → Jotty → DSPy → OpenCodeLM (this class)
    - OpenCodeLM handles remote execution (ARM → x86) automatically
    - Supports multiple models via OpenCode's model selection
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        remote_host: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize OpenCode LM provider
        
        Args:
            model: Model name (e.g., 'glm-4', 'claude-3.5-sonnet', etc.)
                   If None, uses OpenCode's default free model
            remote_host: Remote host for execution (default: cmd.prod.ancillary)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments passed to BaseLM
        """
        # BaseLM expects model name
        model_name = model or "opencode-free"
        super().__init__(model_name, **kwargs)
        
        self.model = model
        self.remote_host = remote_host or REMOTE_HOST
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Detect architecture
        arch = os.uname().machine if hasattr(os, 'uname') else 'unknown'
        self.use_remote = arch in ('aarch64', 'arm64')
    
    def forward(self, prompt: str = None, messages: list = None, **kwargs):
        """
        DSPy BaseLM required method
        Returns OpenAI-compatible response format
        
        Args:
            prompt: Single prompt string
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional arguments
        
        Returns:
            OpenAI-compatible response dict
        """
        # Build prompt from messages or use prompt
        if messages:
            # Extract last user message or combine all
            prompt_text = messages[-1].get('content', '') if messages else ''
        else:
            prompt_text = prompt or ''
        
        if not prompt_text:
            raise ValueError("Either prompt or messages must be provided")
        
        # Get response from OpenCode
        response_text = ''
        for chunk in self._stream(prompt_text, **kwargs):
            response_text += chunk
        
        # Return OpenAI-compatible format
        return {
            'choices': [
                {
                    'message': {
                        'role': 'assistant',
                        'content': response_text
                    },
                    'finish_reason': 'stop'
                }
            ],
            'usage': {
                'prompt_tokens': len(prompt_text.split()),
                'completion_tokens': len(response_text.split()),
                'total_tokens': len(prompt_text.split()) + len(response_text.split())
            }
        }
    
    def __call__(self, prompt: str = None, messages: list = None, **kwargs):
        """
        Generate response from OpenCode (non-streaming)
        Implements BaseLM interface
        
        Args:
            prompt: Input prompt
            messages: List of message dicts
            **kwargs: Additional arguments
        
        Returns:
            List of response strings (DSPy format)
        """
        result = self.forward(prompt=prompt, messages=messages, **kwargs)
        # Return in DSPy format (list of strings or dicts)
        return [result['choices'][0]['message']['content']]
    
    def _stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """
        Stream response from OpenCode
        
        Args:
            prompt: Input prompt
            **kwargs: Additional arguments
        
        Yields:
            Text chunks
        """
        # Prepare OpenCode command
        model_flag = f'--model {self.model}' if self.model else ''
        
        if self.use_remote:
            # Remote execution via SSH
            import shlex
            escaped_prompt = shlex.quote(prompt)
            opencode_cmd = f"opencode run {escaped_prompt} --format json {model_flag}".strip()
            ssh_cmd = f"bash -lc '{opencode_cmd}'"
            cmd = ['ssh', self.remote_host, ssh_cmd]
        else:
            # Local execution
            import shlex
            escaped_prompt = shlex.quote(prompt)
            opencode_cmd = f"{OPENCODE_BIN} run {escaped_prompt} --format json {model_flag}".strip()
            cmd = ['bash', '-c', opencode_cmd]
        
        # Execute OpenCode
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Stream response
        for line in iter(process.stdout.readline, ''):
            if not line:
                break
                
            line = line.strip()
            if not line:
                continue
            
            # Parse OpenCode JSON output
            try:
                data = json.loads(line)
                if data.get('type') == 'text' and 'part' in data:
                    part = data['part']
                    if 'text' in part:
                        yield part['text']
            except json.JSONDecodeError:
                # Skip non-JSON lines
                continue
        
        process.wait()
        
        if process.returncode != 0:
            error = process.stderr.read() if process.stderr else 'Unknown error'
            raise RuntimeError(f'OpenCode execution failed: {error}')
    
    
    def __repr__(self) -> str:
        return f"OpenCodeLM(model={self.model or 'default'}, remote={self.use_remote})"
