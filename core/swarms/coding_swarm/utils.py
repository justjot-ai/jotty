"""
Coding Swarm - Utilities
=========================

Shared helper functions: code fence stripping, component extraction,
progress callbacks, and DSPy streaming wrapper.
"""

import asyncio
import re
import logging
from typing import Dict, List

import dspy

logger = logging.getLogger(__name__)


# =============================================================================
# CODE UTILITIES
# =============================================================================

def _strip_code_fences(code: str) -> str:
    """Strip markdown code fences from LLM-generated code.

    Handles: ```python ... ```, ```py ... ```, ``` ... ```
    Also strips leading/trailing whitespace and handles nested fences.
    """
    if not code or not isinstance(code, str):
        return code or ""
    stripped = code.strip()
    # Match opening fence with optional language tag
    if re.match(r'^```\w*\s*\n', stripped):
        # Remove opening fence line
        stripped = re.sub(r'^```\w*\s*\n', '', stripped, count=1)
        # Remove closing fence (last line)
        stripped = re.sub(r'\n```\s*$', '', stripped)
    return stripped.strip()


def _extract_components_from_text(file_structure: str, architecture: str = "", interfaces: str = "") -> List[Dict[str, str]]:
    """Extract components from file structure, architecture, or interfaces text.

    This is a fallback when JSON parsing fails. It looks for:
    1. Python files (.py) - each becomes a component
    2. Class names mentioned in the text
    3. Directory/module names

    Returns a list of component dicts with 'name' and 'responsibility' keys.
    """
    components = []
    seen_names = set()

    # Combine all text sources
    all_text = f"{file_structure}\n{architecture}\n{interfaces}"

    # Pattern 1: Extract .py files (e.g., "main.py", "game.py", "board.py")
    py_files = re.findall(r'(\w+)\.py\b', all_text)
    for name in py_files:
        if name not in seen_names and name not in ('__init__', '__main__'):
            seen_names.add(name)
            # Try to find a description near the file mention
            desc_match = re.search(rf'{name}\.py[^\n]*?(?:#|:|-|–)\s*([^\n]+)', all_text)
            responsibility = desc_match.group(1).strip() if desc_match else f"{name} module"
            components.append({'name': name, 'responsibility': responsibility})

    # Pattern 2: Extract class names (e.g., "class Board", "TicTacToeGame", "GameController")
    class_patterns = [
        r'\bclass\s+(\w+)',  # "class ClassName"
        r'(\w+(?:Game|Board|Display|Controller|View|Model|Service|Handler|Manager|Engine))\b',  # Common class suffixes
    ]
    for pattern in class_patterns:
        for match in re.finditer(pattern, all_text):
            name = match.group(1)
            if name not in seen_names and len(name) > 2:
                seen_names.add(name)
                components.append({'name': name, 'responsibility': f"{name} component"})

    # Pattern 3: Extract directory names from tree structure (e.g., "├── src/", "│   ├── models/")
    dir_matches = re.findall(r'[├└│─\s]+(\w+)/', all_text)
    for name in dir_matches:
        if name not in seen_names and name not in ('src', 'lib', 'dist', 'build', 'node_modules', '__pycache__'):
            seen_names.add(name)
            components.append({'name': name, 'responsibility': f"{name} directory"})

    return components


# =============================================================================
# PROGRESS / STREAMING
# =============================================================================

# Global callbacks for TUI integration (set by generate(), reset on completion)
_active_progress_callback = None
_active_trace_callback = None


def _progress(phase: str, agent: str, message: str):
    """Print live progress to console."""
    print(f"  [{phase}] {agent}: {message}", flush=True)
    if _active_progress_callback is not None:
        try:
            _active_progress_callback(phase, agent, message)
        except Exception:
            pass  # Never let callback errors break the pipeline


async def _stream_call(module, phase: str, agent: str, listener_field: str = "reasoning",
                       timeout: float = 90.0, max_retries: int = 3, **kwargs):
    """Call a DSPy module with streaming, forwarding reasoning tokens to _progress().

    Args:
        module: DSPy ChainOfThought module to call
        phase: Phase name for progress messages (e.g. "Phase 1")
        agent: Agent name for progress messages (e.g. "Architect")
        listener_field: Output field to stream (default: "reasoning")
        timeout: Timeout in seconds per attempt (default: 90s)
        max_retries: Max retry attempts on timeout (default: 3)
        **kwargs: Arguments to pass to the module

    Returns:
        dspy.Prediction result
    """
    from dspy.streaming import streamify, StreamListener

    async def _do_call():
        listener = StreamListener(listener_field)
        streaming_module = streamify(module, stream_listeners=[listener])

        result = None
        last_text = ""
        async for chunk in streaming_module(**kwargs):
            if isinstance(chunk, dspy.Prediction):
                result = chunk
            elif isinstance(chunk, str):
                new_text = chunk[len(last_text):]
                if new_text.strip():
                    display = chunk.strip()[-80:]
                    _progress(phase, agent, f"  ...{display}")
                last_text = chunk

        if result is None:
            result = module(**kwargs)

        return result

    # Retry with timeout
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            return await asyncio.wait_for(_do_call(), timeout=timeout)
        except asyncio.TimeoutError:
            last_error = asyncio.TimeoutError(f"Timeout after {timeout}s")
            # Increase timeout for next attempt
            timeout = min(timeout * 1.5, 180.0)
            print(f"⏱️ Attempt {attempt}/{max_retries}: Timeout after {timeout/1.5:.0f}s", flush=True)
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

    # All retries exhausted - raise last error
    raise last_error
