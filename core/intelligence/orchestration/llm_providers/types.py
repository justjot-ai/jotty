"""
Executor data types shared across LLM providers and ChatExecutor.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolResult:
    """Result from a tool execution."""

    tool_name: str
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class LLMExecutionResult:
    """Result from unified execution."""

    success: bool
    content: str
    tool_results: List[ToolResult] = field(default_factory=list)
    sections: List[Dict[str, Any]] = field(default_factory=list)
    output_path: Optional[str] = None
    output_format: str = "text"
    error: Optional[str] = None
    steps_taken: List[str] = field(default_factory=list)
    usage: Optional[Dict[str, Any]] = None


@dataclass
class StreamEvent:
    """Event emitted during streaming execution."""

    type: str  # 'text', 'tool_start', 'tool_end', 'section', 'complete', 'error'
    data: Any


@dataclass
class LLMResponse:
    """Unified response from any LLM provider."""

    content: List[Any]  # Content blocks (text, tool_use)
    stop_reason: str  # 'end_turn', 'tool_use', 'stop', etc.
    usage: Optional[Dict[str, int]] = None


@dataclass
class ToolUseBlock:
    """Unified tool use block."""

    id: str
    name: str
    input: Dict[str, Any]
    type: str = "tool_use"


@dataclass
class TextBlock:
    """Unified text block."""

    text: str
    type: str = "text"
