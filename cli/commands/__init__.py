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
from .justjot import JustJotCommand
from .resume import ResumeCommand
from .export import ExportCommand
from .ml import MLCommand
from .mlflow_cmd import MLflowCommand
from .stock_ml import StockMLCommand
from .preview import PreviewCommand
from .browse import BrowseCommand
from .research import ResearchCommand
from .workflow import WorkflowCommand
from .telegram_bot import TelegramCommand
from .web_server import WebServerCommand
from .model_chat import ModelChatCommand
from .gateway import GatewayCommand
from .whatsapp_cmd import WhatsAppCommand
from .heartbeat_cmd import HeartbeatCommand, RemindCommand
from .task_queue import TaskCommand
from .supervisor_cmd import SupervisorCommand
from .swimlane import SwimlaneCommand
from .backtest_report import BacktestReportCommand, BatchBacktestReportCommand

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
    "WhatsAppCommand",
    "HeartbeatCommand",
    "RemindCommand",
    "TaskCommand",
    "SupervisorCommand",
    "SwimlaneCommand",
    "BacktestReportCommand",
    "BatchBacktestReportCommand",
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
        WhatsAppCommand(),
        HeartbeatCommand(),
        RemindCommand(),
        TaskCommand(),
        SupervisorCommand(),
        SwimlaneCommand(),
        BacktestReportCommand(),
        BatchBacktestReportCommand(),
    ]

    for cmd in commands:
        registry.register(cmd)
