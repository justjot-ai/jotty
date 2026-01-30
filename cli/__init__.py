"""
Jotty CLI - Claude Code-like Interactive CLI for Jotty SDK
===========================================================

Provides an interactive REPL with slash commands, agent/swarm exposure,
learning integration, and rich terminal UI.

Usage:
    python -m Jotty.cli               # Interactive REPL
    python -m Jotty.cli run "task"    # Single command mode
    jotty                             # Shell entry point (if installed)

Key Features:
- /run <goal> - Execute tasks with SwarmManager
- /agents - List and manage agents
- /skills - Discover 120+ skills
- /swarm - Swarm intelligence status
- /learn - Training and warmup
- /memory - Hierarchical memory inspection
- /stats - Learning metrics
- /tools - Execute skills directly
"""

# Suppress HuggingFace/BERT warnings early
import os as _os
# HF_TOKEN should be set via environment variable, not hardcoded
_os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
_os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
_os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
_os.environ.setdefault('TQDM_DISABLE', '1')

import warnings as _warnings
_warnings.filterwarnings('ignore', message='.*unauthenticated.*')
_warnings.filterwarnings('ignore', message='.*huggingface.*')

import logging as _logging
for _name in ['safetensors', 'sentence_transformers', 'transformers', 'huggingface_hub']:
    _logging.getLogger(_name).setLevel(_logging.ERROR)

from .app import JottyCLI
from .repl.engine import REPLEngine
from .repl.session import SessionManager
from .commands.base import BaseCommand, CommandResult, CommandRegistry
from .ui.renderer import RichRenderer
from .config.loader import ConfigLoader
from .config.schema import CLIConfig

__all__ = [
    "JottyCLI",
    "REPLEngine",
    "SessionManager",
    "BaseCommand",
    "CommandResult",
    "CommandRegistry",
    "RichRenderer",
    "ConfigLoader",
    "CLIConfig",
]

__version__ = "1.0.0"
