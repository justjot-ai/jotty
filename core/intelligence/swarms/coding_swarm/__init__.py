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

from .types import (
    CodeLanguage, CodeStyle, EditMode,
    CodingConfig, CodingResult, CodeOutput,
)
from .agents import (
    ArchitectAgent, DeveloperAgent, DebuggerAgent,
    OptimizerAgent, TestWriterAgent, DocWriterAgent,
)
from .swarm import CodingSwarm, code, code_sync

__all__ = [
    'CodingSwarm', 'CodingConfig', 'CodingResult', 'CodeOutput',
    'CodeLanguage', 'CodeStyle', 'EditMode',
    'code', 'code_sync',
    'ArchitectAgent', 'DeveloperAgent', 'DebuggerAgent',
    'OptimizerAgent', 'TestWriterAgent', 'DocWriterAgent',
]
