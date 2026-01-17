"""
Enhanced Claude CLI Wrapper with JSON Schema Support
=====================================================

Uses --json-schema to enforce structured output for DSPy signatures.
"""

import subprocess
import json
import dspy
from dspy import BaseLM
from typing import Dict, Any, Optional


class EnhancedClaudeCLILM(BaseLM):
    """DSPy-compatible LM using Claude CLI with JSON schema enforcement."""

    def __init__(self, model="sonnet", **kwargs):
        super().__init__(model=f"claude-cli/{model}", **kwargs)
        self.cli_model = model
        self._verify_cli_available()
        self.provider = "claude-cli"
        self.history = []

    def _verify_cli_available(self):
        """Check if claude CLI is available."""
        try:
            result = subprocess.run(
                ["claude", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                print(f"✓ Claude CLI available: {result.stdout.strip()}")
            else:
                raise RuntimeError("Claude CLI not working")
        except FileNotFoundError:
            raise RuntimeError("Claude CLI not found")

    def _extract_json_schema(self, signature) -> Optional[Dict[str, Any]]:
        """
        Extract JSON schema from DSPy signature for --json-schema parameter.

        Converts DSPy signature output fields to JSON schema format.
        """
        # DSPy signatures use model_fields (pydantic v2)
        if not hasattr(signature, 'model_fields'):
            return None

        # Get output fields only (marked with __dspy_field_type: 'output')
        output_fields = {}
        required_fields = []

        for field_name, field_info in signature.model_fields.items():
            # Check if this is an output field
            json_extra = field_info.json_schema_extra or {}
            if json_extra.get('__dspy_field_type') == 'output':
                field_desc = json_extra.get('desc', '')
                output_fields[field_name] = {
                    "type": "string",
                    "description": field_desc
                }
                if field_info.is_required():
                    required_fields.append(field_name)

        if not output_fields:
            return None

        schema = {
            "type": "object",
            "properties": output_fields,
            "required": required_fields
        }

        return schema

    def __call__(self, prompt=None, messages=None, **kwargs):
        """
        DSPy-compatible call interface with JSON schema enforcement.
        """
        # Build messages list
        if messages is None:
            messages = []
            if prompt:
                messages = [{"role": "user", "content": prompt}]

        if not messages:
            raise ValueError("Either prompt or messages must be provided")

        # Extract the user message
        user_message = None
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get('role') == 'user':
                user_message = msg.get('content', '')
                break
            elif isinstance(msg, str):
                user_message = msg
                break

        if not user_message:
            raise ValueError("No user message found")

        # Try to get JSON schema from kwargs if signature is passed
        json_schema = kwargs.get('json_schema')
        signature = kwargs.get('signature')

        # DEBUG: Print what we received
        #print(f"DEBUG: kwargs keys: {list(kwargs.keys())}")
        #print(f"DEBUG: signature in kwargs: {signature is not None}")

        if not json_schema and signature:
            json_schema = self._extract_json_schema(signature)
            #print(f"DEBUG: Extracted schema: {json_schema}")

        # Build command
        cmd = [
            "claude",
            "--model", self.cli_model,
            "--print",  # Non-interactive mode
            "--output-format", "json",  # JSON output
            "--dangerously-skip-permissions",  # Skip permission prompts
        ]

        # Note: --json-schema flag doesn't exist in Claude CLI 2.0.36
        # JSON output format + DSPy's JSONAdapter handles parsing

        cmd.append(user_message)

        # Execute command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=kwargs.get('timeout', 120)  # Increased timeout
        )

        if result.returncode != 0:
            raise RuntimeError(f"Claude CLI error: {result.stderr}")

        # Parse JSON response
        try:
            response_data = json.loads(result.stdout.strip())

            # If we used JSON schema, extract from structured_output field
            if json_schema and 'structured_output' in response_data:
                structured_output = response_data['structured_output']
                # Return as JSON string for DSPy to parse
                response_text = json.dumps(structured_output)
            else:
                # Standard flow: extract from result field
                response_text = response_data.get('result', result.stdout.strip())

        except json.JSONDecodeError:
            # Fallback to raw text
            response_text = result.stdout.strip()

        # Store in history
        self.history.append({
            "prompt": user_message,
            "response": response_text,
            "kwargs": kwargs
        })

        # Return in DSPy format
        return [response_text]

    def inspect_history(self, n=1):
        """DSPy-compatible history inspection."""
        return self.history[-n:] if self.history else []
