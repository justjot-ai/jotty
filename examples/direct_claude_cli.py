"""
Direct Claude CLI calls with JSON schema enforcement.

Bypasses DSPy for maximum control over JSON schema enforcement.
"""

import subprocess
import json
from typing import Dict, Any, Optional


def call_claude_with_schema(
    prompt: str,
    output_schema: Dict[str, Any],
    model: str = "sonnet",
    timeout: int = 120
) -> Dict[str, Any]:
    """
    Call Claude CLI with JSON schema enforcement.

    Args:
        prompt: The prompt to send
        output_schema: JSON schema for output validation
        model: Claude model (sonnet, opus, haiku)
        timeout: Timeout in seconds

    Returns:
        Dict with structured output matching schema
    """
    cmd = [
        "claude",
        "--model", model,
        "--print",
        "--output-format", "json",
        "--json-schema", json.dumps(output_schema),
        prompt
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout
    )

    if result.returncode != 0:
        raise RuntimeError(f"Claude CLI error: {result.stderr}")

    # Parse JSON response
    response_data = json.loads(result.stdout.strip())

    # Extract structured output
    if 'structured_output' in response_data:
        return response_data['structured_output']
    else:
        # Fallback: try to parse result field as JSON
        result_text = response_data.get('result', '{}')
        try:
            return json.loads(result_text)
        except json.JSONDecodeError:
            return {"result": result_text}


# Example schemas for Jotty signatures
PLANNER_SCHEMA = {
    "type": "object",
    "properties": {
        "plan": {
            "type": "string",
            "description": "Step-by-step plan to accomplish the goal"
        },
        "required_agents": {
            "type": "string",
            "description": "List of agent types needed (comma-separated)"
        }
    },
    "required": ["plan", "required_agents"]
}

EXECUTOR_SCHEMA = {
    "type": "object",
    "properties": {
        "result": {
            "type": "string",
            "description": "Result of task execution"
        }
    },
    "required": ["result"]
}

COMPLEXITY_SCHEMA = {
    "type": "object",
    "properties": {
        "complexity_score": {
            "type": "string",
            "description": "Complexity score (1-5)"
        },
        "should_spawn": {
            "type": "string",
            "description": "Should spawn sub-agents? (yes/no)"
        },
        "reasoning": {
            "type": "string",
            "description": "Reasoning for the decision"
        }
    },
    "required": ["complexity_score", "should_spawn", "reasoning"]
}
