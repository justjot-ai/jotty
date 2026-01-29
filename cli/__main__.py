"""
Jotty CLI Entry Point
=====================

Usage:
    python -m Jotty.cli                    # Interactive REPL
    python -m Jotty.cli run "search for AI news"  # Single command
    python -m Jotty.cli --help             # Show help
"""

# CRITICAL: Suppress HuggingFace/BERT warnings BEFORE any other imports
import os
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import warnings
warnings.filterwarnings('ignore', message='.*unauthenticated.*')
warnings.filterwarnings('ignore', message='.*huggingface.*')
warnings.filterwarnings('ignore', category=FutureWarning)

import logging as _logging
for _logger_name in ['safetensors', 'sentence_transformers', 'transformers', 'huggingface_hub']:
    _logging.getLogger(_logger_name).setLevel(_logging.ERROR)

import sys
import asyncio
import argparse
from pathlib import Path


def main():
    """Main entry point for Jotty CLI."""
    parser = argparse.ArgumentParser(
        prog="jotty",
        description="Jotty CLI - Interactive Multi-Agent AI Assistant",
        epilog="For more info: https://github.com/yourusername/jotty"
    )

    parser.add_argument(
        "-v", "--version",
        action="store_true",
        help="Show version and exit"
    )

    parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="Path to config file (default: ~/.jotty/config.yaml)"
    )

    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    # Subcommands for single execution
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # run command
    run_parser = subparsers.add_parser("run", help="Execute a task")
    run_parser.add_argument("goal", nargs="+", help="Task goal in natural language")

    # skills command
    skills_parser = subparsers.add_parser("skills", help="List available skills")
    skills_parser.add_argument("--category", "-c", help="Filter by category")

    # agents command
    agents_parser = subparsers.add_parser("agents", help="List agents")

    # warmup command
    warmup_parser = subparsers.add_parser("warmup", help="Run DrZero warmup")
    warmup_parser.add_argument(
        "--episodes", "-e",
        type=int,
        default=10,
        help="Number of warmup episodes"
    )

    args = parser.parse_args()

    # Handle version
    if args.version:
        from . import __version__
        from .. import __version__ as jotty_version
        print(f"Jotty CLI v{__version__}")
        print(f"Jotty SDK v{jotty_version}")
        return 0

    # Configure logging
    if args.debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    # Import JottyCLI (delayed to speed up help/version)
    from .app import JottyCLI

    # Create CLI instance
    cli = JottyCLI(
        config_path=args.config,
        no_color=args.no_color,
        debug=args.debug
    )

    # Handle subcommands (single execution mode)
    if args.command == "run":
        goal = " ".join(args.goal)
        return asyncio.run(cli.run_once(f"/run {goal}"))

    elif args.command == "skills":
        cmd = "/skills"
        if args.category:
            cmd += f" --category {args.category}"
        return asyncio.run(cli.run_once(cmd))

    elif args.command == "agents":
        return asyncio.run(cli.run_once("/agents"))

    elif args.command == "warmup":
        return asyncio.run(cli.run_once(f"/learn warmup {args.episodes}"))

    # Interactive REPL mode (default)
    try:
        asyncio.run(cli.run_interactive())
        return 0
    except KeyboardInterrupt:
        print("\nGoodbye!")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
