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
        if not hasattr(signature, '__annotations__'):
            return None

        # Get output fields from signature
        output_fields = {}
        required_fields = []

        for field_name, field in signature.__dict__.items():
            if hasattr(field, 'json_schema_extra'):
                # This is a DSPy OutputField
                field_desc = getattr(field, 'desc', '')
                output_fields[field_name] = {
                    "type": "string",
                    "description": field_desc
                }
                required_fields.append(field_name)

        # Fallback: parse from field annotations
        if not output_fields and hasattr(signature, '__annotations__'):
            for field_name in signature.__annotations__:
                if field_name.startswith('_'):
                    continue
                # Check if this is an output field
                field_obj = getattr(signature, field_name, None)
                if field_obj and hasattr(field_obj, '__class__'):
                    if 'Output' in field_obj.__class__.__name__:
                        field_desc = getattr(field_obj, 'desc', '')
                        output_fields[field_name] = {
                            "type": "string",
                            "description": field_desc
                        }
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

        if not json_schema and signature:
            json_schema = self._extract_json_schema(signature)

        # Build command
        cmd = [
            "claude",
            "--model", self.cli_model,
            "--print",  # Non-interactive mode
            "--output-format", "json",  # JSON output
        ]

        # Add JSON schema if available (enforces structured output)
        if json_schema:
            cmd.extend(["--json-schema", json.dumps(json_schema)])

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
