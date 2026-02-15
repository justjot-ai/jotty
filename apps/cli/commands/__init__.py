from typing import Any

"""
CLI Commands Module
===================

Slash commands for Jotty CLI.
"""

from .agents import AgentsCommand
from .backtest_report import BacktestReportCommand, BatchBacktestReportCommand
from .base import BaseCommand, CommandRegistry, CommandResult
from .browse import BrowseCommand
from .config_cmd import ConfigCommand
from .export import ExportCommand
from .gateway import GatewayCommand
from .git import GitCommand

# WhatsApp moved to apps/whatsapp/
# from .whatsapp_cmd import WhatsAppCommand  # TODO: Import from apps.whatsapp.command
from .heartbeat_cmd import HeartbeatCommand, RemindCommand
from .help_cmd import HelpCommand
from .justjot import JustJotCommand
from .learn import LearnCommand
from .memory import MemoryCommand
from .ml import MLCommand
from .mlflow_cmd import MLflowCommand
from .model_chat import ModelChatCommand
from .plan import PlanCommand
from .preview import PreviewCommand
from .research import ResearchCommand
from .resume import ResumeCommand
from .run import RunCommand
from .sdk_cmd import SDKCommand
from .skills import SkillsCommand
from .stats import StatsCommand
from .stock_ml import StockMLCommand
from .supervisor_cmd import SupervisorCommand
from .swarm import SwarmCommand
from .swimlane import SwimlaneCommand
from .task_queue import TaskCommand
from .telegram_bot import TelegramCommand
from .tools import ToolsCommand
from .web_server import WebServerCommand
from .workflow import WorkflowCommand

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
    "JustJotCommand",
    "ResumeCommand",
    "ExportCommand",
    "MLCommand",
    "MLflowCommand",
    "StockMLCommand",
    "PreviewCommand",
    "BrowseCommand",
    "ResearchCommand",
    "WorkflowCommand",
    "TelegramCommand",
    "WebServerCommand",
    "ModelChatCommand",
    "GatewayCommand",
    # "WhatsAppCommand",  # Moved to apps/whatsapp/
    "HeartbeatCommand",
    "RemindCommand",
    "TaskCommand",
    "SupervisorCommand",
    "SwimlaneCommand",
    "BacktestReportCommand",
    "BatchBacktestReportCommand",
    "SDKCommand",
]


def register_all_commands(registry: CommandRegistry) -> Any:
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
        JustJotCommand(),
        ResumeCommand(),
        ExportCommand(),
        MLCommand(),
        MLflowCommand(),
        StockMLCommand(),
        PreviewCommand(),
        BrowseCommand(),
        ResearchCommand(),
        WorkflowCommand(),
        TelegramCommand(),
        WebServerCommand(),
        ModelChatCommand(),
        GatewayCommand(),
        # WhatsAppCommand(),  # Moved to apps/whatsapp/
        HeartbeatCommand(),
        RemindCommand(),
        TaskCommand(),
        SupervisorCommand(),
        SwimlaneCommand(),
        BacktestReportCommand(),
        BatchBacktestReportCommand(),
        SDKCommand(),
    ]

    for cmd in commands:
        registry.register(cmd)
