"""
Cursor CLI DSPy LM Provider
============================

Part of Jotty multi-agent framework.
Uses Cursor Agent CLI with JSON output support.
"""

import subprocess
import json
import os
import dspy
from dspy import BaseLM
from typing import Dict, Any, Optional


class CursorCLILM(BaseLM):
    """DSPy-compatible LM using Cursor CLI."""

    def __init__(self, model="sonnet-4", **kwargs):
        super().__init__(model=f"cursor-cli/{model}", **kwargs)
        self.cli_model = model
        self.history = []

    def _extract_json_schema(self, signature):
        """
        Extract JSON schema from DSPy signature.
        Cursor doesn't have native JSON schema support, so we'll enforce it in the prompt.
        """
        if not hasattr(signature, 'output_fields'):
            return None

        output_fields = {}
        required_fields = []

        for name, field in signature.output_fields.items():
            field_type = "string"  # Default to string
            if hasattr(field, 'annotation'):
                if field.annotation == int:
                    field_type = "integer"
                elif field.annotation == float:
                    field_type = "number"
                elif field.annotation == bool:
                    field_type = "boolean"

            output_fields[name] = {"type": field_type}
            required_fields.append(name)

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
        DSPy-compatible call interface.
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

        # If we have a JSON schema, add it to the prompt
        if json_schema:
            schema_prompt = f"\n\nIMPORTANT: Respond with ONLY valid JSON matching this schema:\n{json.dumps(json_schema, indent=2)}\n\nDo not include any text before or after the JSON."
            user_message = user_message + schema_prompt

        # Build command
        cmd = [
            "cursor-agent",
            "--model", self.cli_model,
            "--print",  # Non-interactive mode
            "--output-format", "json",  # JSON output
        ]

        cmd.append(user_message)

        # Execute command
        # Note: Cursor CLI doesn't have the same OAuth token issue as Claude
        # It uses credentials from auth.json automatically
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=kwargs.get('timeout', 120)  # Increased timeout
        )

        if result.returncode != 0:
            # Extract error message
            error_msg = result.stderr.strip() or result.stdout.strip() or "Unknown error"
            raise RuntimeError(f"Cursor CLI error: {error_msg}")

        # Parse JSON response
        try:
            response_data = json.loads(result.stdout.strip())

            # Cursor CLI returns different JSON format than Claude
            # Extract the actual response text
            if isinstance(response_data, dict):
                # Try common response fields
                response_text = (
                    response_data.get('response') or
                    response_data.get('message') or
                    response_data.get('content') or
                    response_data.get('result') or
                    json.dumps(response_data)  # Fallback to full JSON
                )
            else:
                response_text = str(response_data)

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
