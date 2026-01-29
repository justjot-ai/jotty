"""
CLI Commands Module
===================

Slash commands for Jotty CLI.
"""

from .base import BaseCommand, CommandResult, CommandRegistry
from .run import RunCommand
from .agents import AgentsCommand
from .skills import SkillsCommand
from .swarm import SwarmCommand
from .learn import LearnCommand
from .memory import MemoryCommand
from .config_cmd import ConfigCommand
from .stats import StatsCommand
from .plan import PlanCommand
from .git import GitCommand
from .help_cmd import HelpCommand
from .tools import ToolsCommand

__all__ = [
    "BaseCommand",
    "CommandResult",
    "CommandRegistry",
    "RunCommand",
    "AgentsCommand",
    "SkillsCommand",
    "SwarmCommand",
    "LearnCommand",
    "MemoryCommand",
    "ConfigCommand",
    "StatsCommand",
    "PlanCommand",
    "GitCommand",
    "HelpCommand",
    "ToolsCommand",
]


def register_all_commands(registry: CommandRegistry):
    """Register all built-in commands."""
    commands = [
        RunCommand(),
        AgentsCommand(),
        SkillsCommand(),
        SwarmCommand(),
        LearnCommand(),
        MemoryCommand(),
        ConfigCommand(),
        StatsCommand(),
        PlanCommand(),
        GitCommand(),
        HelpCommand(),
        ToolsCommand(),
    ]

    for cmd in commands:
        registry.register(cmd)
