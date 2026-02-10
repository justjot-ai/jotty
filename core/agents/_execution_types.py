"""
Shared execution types - breaks circular dependencies.

This module provides ExecutionStep, ExecutionStepSchema, TaskType, and
ExecutionResult in a dependency-free location so that both agentic_planner.py
and auto_agent.py can import them without circular imports.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

# =============================================================================
# TASK TYPE (Enum)
# =============================================================================


class TaskType(Enum):
    """Inferred task types."""
    RESEARCH = "research"
    COMPARISON = "comparison"
    CREATION = "creation"
    COMMUNICATION = "communication"
    ANALYSIS = "analysis"
    AUTOMATION = "automation"
    UNKNOWN = "unknown"


# =============================================================================
# EXECUTION STEP (Dataclass)
# =============================================================================


@dataclass
class ExecutionStep:
    """A step in the execution plan."""
    skill_name: str
    tool_name: str
    params: Dict[str, Any]
    description: str
    depends_on: List[int] = field(default_factory=list)
    output_key: str = ""
    optional: bool = False
    verification: str = ""
    fallback_skill: str = ""


# =============================================================================
# EXECUTION STEP SCHEMA (Pydantic, optional)
# =============================================================================

try:
    from pydantic import BaseModel, Field, model_validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False


if PYDANTIC_AVAILABLE:
    class ExecutionStepSchema(BaseModel):
        """Schema for execution plan steps - accepts common LLM field name variations."""
        skill_name: str = Field(default="", description="Skill name from available_skills")
        tool_name: str = Field(default="", description="Tool name from that skill's tools list")
        params: Dict[str, Any] = Field(default_factory=dict, description="Tool parameters")
        description: str = Field(default="", description="What this step does")
        depends_on: List[int] = Field(default_factory=list, description="Indices of steps this depends on")
        output_key: str = Field(default="", description="Key to store output under")
        optional: bool = Field(default=False, description="Whether step is optional")
        verification: str = Field(default="", description="How to confirm this step succeeded")
        fallback_skill: str = Field(default="", description="Alternative skill if this one fails")

        model_config = {"extra": "allow"}

        @model_validator(mode='before')
        @classmethod
        def normalize_field_names(cls, data: Dict[str, Any]) -> Dict[str, Any]:
            """Normalize common LLM field name variations to expected names."""
            if not isinstance(data, dict):
                return data

            if 'skill_name' not in data or not data.get('skill_name'):
                skill = data.get('skill', '')
                if not skill:
                    skills_used = data.get('skills_used', [])
                    if skills_used and isinstance(skills_used, list):
                        skill = skills_used[0]
                data['skill_name'] = skill

            if 'tool_name' not in data or not data.get('tool_name'):
                tool = data.get('tool', '')
                if not tool:
                    tools_used = data.get('tools_used', [])
                    if tools_used and isinstance(tools_used, list):
                        tool = tools_used[0]
                if not tool:
                    action = data.get('action', '')
                    if action and isinstance(action, str):
                        tool_match = re.search(r'\b([a-z_]+_tool)\b', action)
                        if tool_match:
                            tool = tool_match.group(1)
                data['tool_name'] = tool

            if 'params' not in data or not data.get('params'):
                data['params'] = (
                    data.get('parameters') or
                    data.get('tool_input') or
                    data.get('tool_params') or
                    data.get('inputs') or
                    data.get('input') or
                    {}
                )

            return data
else:
    ExecutionStepSchema = None  # type: ignore[assignment,misc]


# =============================================================================
# EXECUTION RESULT (Dataclass)
# =============================================================================

try:
    from Jotty.core.utils.context_utils import strip_enrichment_context
except ImportError:
    def strip_enrichment_context(text: str) -> str:
        return text


def _clean_for_display(text: str) -> str:
    """Remove internal enrichment context from text for user-facing display."""
    return strip_enrichment_context(text) if text else text


@dataclass
class ExecutionResult:
    """Result of task execution."""
    success: bool
    task: str
    task_type: TaskType
    skills_used: List[str]
    steps_executed: int
    outputs: Dict[str, Any]
    final_output: Any
    errors: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    stopped_early: bool = False

    @property
    def artifacts(self) -> List[Dict[str, Any]]:
        """Extract all created files/artifacts from execution outputs.

        Scans outputs for file paths (from file-operations, shell-exec, etc.)
        Returns list of {path, type, size_bytes, step} dicts.
        """
        found = []
        for step_name, step_data in (self.outputs or {}).items():
            if not isinstance(step_data, dict):
                continue
            # file-operations returns {path, bytes_written}
            if 'path' in step_data and step_data.get('success', True):
                found.append({
                    'path': step_data['path'],
                    'type': 'file',
                    'size_bytes': step_data.get('bytes_written', 0),
                    'step': step_name,
                })
            # shell-exec may create files (check stdout for file paths)
            if 'stdout' in step_data:
                stdout = str(step_data.get('stdout', ''))
                for m in re.finditer(
                    r'(?:saved?|wrot?e?|created?|output)\s+(?:to\s+)?["\']?([^\s"\']+\.\w{1,5})',
                    stdout, re.I,
                ):
                    fpath = m.group(1)
                    if len(fpath) > 3 and ('/' in fpath or '.' in fpath):
                        found.append({
                            'path': fpath, 'type': 'file',
                            'size_bytes': 0, 'step': step_name,
                        })
        return found

    @property
    def summary(self) -> str:
        """Human-readable summary of what was done and what was produced."""
        parts = []
        status = "completed successfully" if self.success else "failed"
        parts.append(f"Task {status} in {self.execution_time:.1f}s ({self.steps_executed} steps)")

        if self.skills_used:
            parts.append(f"Skills used: {', '.join(self.skills_used)}")

        artifacts = self.artifacts
        if artifacts:
            parts.append("Files created:")
            for a in artifacts:
                size = f" ({a['size_bytes']} bytes)" if a.get('size_bytes') else ""
                parts.append(f"  → {a['path']}{size}")

        if self.errors:
            parts.append(f"Errors: {'; '.join(self.errors[:3])}")

        if self.final_output and isinstance(self.final_output, str):
            clean = self.final_output.strip()[:300]
            if clean:
                parts.append(f"Output: {clean}")

        return '\n'.join(parts)


__all__ = [
    'TaskType',
    'ExecutionStep',
    'ExecutionStepSchema',
    'ExecutionResult',
    '_clean_for_display',
]
