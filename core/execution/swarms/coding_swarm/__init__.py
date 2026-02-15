"""
Coding Swarm Package
=====================

Production-grade swarm for code generation, refactoring, debugging,
architecture design, and code review automation.

Usage:
    from core.swarms.coding_swarm import CodingSwarm, code

    swarm = CodingSwarm()
    result = await swarm.generate("Create a REST API for user management")
"""

from .agents import (
    ArchitectAgent,
    DebuggerAgent,
    DeveloperAgent,
    DocWriterAgent,
    OptimizerAgent,
    TestWriterAgent,
)
from .swarm import CodingSwarm, code, code_sync
from .types import CodeLanguage, CodeOutput, CodeStyle, CodingConfig, CodingResult, EditMode

__all__ = [
    "CodingSwarm",
    "CodingConfig",
    "CodingResult",
    "CodeOutput",
    "CodeLanguage",
    "CodeStyle",
    "EditMode",
    "code",
    "code_sync",
    "ArchitectAgent",
    "DeveloperAgent",
    "DebuggerAgent",
    "OptimizerAgent",
    "TestWriterAgent",
    "DocWriterAgent",
]
