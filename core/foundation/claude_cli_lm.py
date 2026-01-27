"""
Claude CLI DSPy LM Provider
============================

Part of Jotty multi-agent framework.
Uses Claude CLI with JSON schema support for structured output.
Automatically loads skills from Jotty's skills registry.
"""

import subprocess
import json
import os
import dspy
from dspy import BaseLM
from typing import Dict, Any, Optional


class ClaudeCLILM(BaseLM):
    """DSPy-compatible LM using Claude CLI with JSON schema enforcement."""

    def __init__(self, model="sonnet", enable_skills=True, **kwargs):
        """
        Initialize Claude CLI LM.
        
        Args:
            model: Claude model (sonnet, opus, haiku)
            enable_skills: Whether to enable Jotty skills (default: True)
            **kwargs: Additional arguments
        """
        super().__init__(model=f"claude-cli/{model}", **kwargs)
        self.cli_model = model
        self.enable_skills = enable_skills
        self._verify_cli_available()
        self.provider = "claude-cli"
        self.history = []
        
        # Load skills if enabled
        if self.enable_skills:
            try:
                from ..registry.skills_registry import get_skills_registry
                self.skills_registry = get_skills_registry()
                self.skills_registry.init()
                self.skills = self.skills_registry.list_skills()
                print(f"✓ Loaded {len(self.skills)} skills for Claude CLI")
            except Exception as e:
                print(f"⚠️  Failed to load skills: {e}")
                self.skills_registry = None
                self.skills = []
        else:
            self.skills_registry = None
            self.skills = []

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
        if signature is None:
            return None
            
        # Handle both class and instance
        if isinstance(signature, type):
            signature_class = signature
        else:
            signature_class = signature.__class__ if hasattr(signature, '__class__') else None
        
        if signature_class is None:
            return None

        # Get output fields from signature
        output_fields = {}
        required_fields = []

        # Try to get fields from signature class
        if hasattr(signature_class, '__dict__'):
            for field_name, field in signature_class.__dict__.items():
                if field_name.startswith('_'):
                    continue
                # Check if this is a DSPy OutputField
                if hasattr(field, '__class__'):
                    class_name = field.__class__.__name__
                    if 'Output' in class_name or 'Field' in class_name:
                        field_desc = getattr(field, 'desc', '')
                        # Determine type from field name or description
                        field_type = "string"
                        if 'confidence' in field_name.lower() or 'float' in str(type(field)):
                            field_type = "number"
                        elif 'int' in str(type(field)):
                            field_type = "integer"
                        elif 'bool' in str(type(field)):
                            field_type = "boolean"
                        
                        output_fields[field_name] = {
                            "type": field_type,
                            "description": field_desc
                        }
                        required_fields.append(field_name)

        # Fallback: parse from annotations
        if not output_fields and hasattr(signature_class, '__annotations__'):
            for field_name, field_type in signature_class.__annotations__.items():
                if field_name.startswith('_'):
                    continue
                field_obj = getattr(signature_class, field_name, None)
                if field_obj:
                    field_desc = getattr(field_obj, 'desc', '')
                    # Map Python types to JSON types
                    json_type = "string"
                    if 'float' in str(field_type) or 'float' in str(field_obj):
                        json_type = "number"
                    elif 'int' in str(field_type):
                        json_type = "integer"
                    elif 'bool' in str(field_type):
                        json_type = "boolean"
                    
                    output_fields[field_name] = {
                        "type": json_type,
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
        
        # Also check if signature is in the prompt context (DSPy sometimes passes it differently)
        if not signature:
            # Try to extract from messages
            for msg in messages:
                if isinstance(msg, dict) and 'signature' in msg:
                    signature = msg['signature']
                    break

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
        
        # Note: Claude CLI automatically discovers skills in ~/.claude/skills/
        # Skills are available via /skillname syntax (e.g., /last30days)
        # No need to explicitly pass them via --tools parameter
        # The skills registry ensures skills are loaded and available

        cmd.append(user_message)

        # Execute command
        # IMPORTANT: Unset ANTHROPIC_API_KEY if it's an OAuth token
        # OAuth tokens (sk-ant-oat01-*) don't work with --print mode
        env = os.environ.copy()
        api_key = env.get('ANTHROPIC_API_KEY', '')
        if api_key.startswith('sk-ant-oat'):
            # Remove OAuth token - let Claude use credentials file
            env.pop('ANTHROPIC_API_KEY', None)

        # Get timeout from kwargs or use default (30s for code gen, 60s for planning, 120s for others)
        timeout = kwargs.get('timeout', 120)
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout
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
