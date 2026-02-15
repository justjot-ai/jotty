from typing import Any

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

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import warnings

warnings.filterwarnings("ignore", message=".*unauthenticated.*")
warnings.filterwarnings("ignore", message=".*huggingface.*")
warnings.filterwarnings("ignore", category=FutureWarning)

import logging as _logging

for _logger_name in ["safetensors", "sentence_transformers", "transformers", "huggingface_hub"]:
    _logging.getLogger(_logger_name).setLevel(_logging.ERROR)

import argparse
import asyncio
import sys


async def _list_commands(cli: Any, args: Any) -> Any:
    """List all available slash commands."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    # Get all registered commands
    commands = cli.command_registry.list_commands()

    # Group by category
    categories = {}
    for cmd in commands:
        cat = cmd.get("category", "general")
        if args.category and cat != args.category:
            continue
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(cmd)

    # Create table
    table = Table(title="Available Slash Commands", show_header=True, header_style="bold cyan")
    table.add_column("Command", style="green", width=20)
    table.add_column("Aliases", style="dim", width=15)
    table.add_column("Category", style="yellow", width=12)
    table.add_column("Description", width=45)

    # Sort categories
    cat_order = ["ml", "research", "system", "general", "tools", "memory", "config"]
    sorted_cats = sorted(
        categories.keys(), key=lambda x: cat_order.index(x) if x in cat_order else 99
    )

    for cat in sorted_cats:
        cmds = sorted(categories[cat], key=lambda x: x["name"])
        for cmd in cmds:
            aliases = ", ".join(cmd.get("aliases", []))
            table.add_row(f"/{cmd['name']}", aliases, cat, cmd.get("description", "")[:45])

    console.print(table)
    console.print(f"\n[dim]Total: {len(commands)} commands[/dim]")
    console.print(
        "[dim]Use 'jotty' without args for interactive mode, then /help <command> for details[/dim]"
    )

    return 0


async def _run_stock_ml(cli: Any, args: Any) -> Any:
    """Run stock ML command with args."""
    # Build command string
    cmd = "/stock-ml"

    if args.list:
        cmd += " --list"
    elif args.leaderboard:
        cmd += " --leaderboard"
    elif args.sets:
        cmd += " --sets"
    elif args.sweep:
        cmd += " --sweep"
        if args.stocks:
            cmd += f" --stocks {args.stocks}"
        if args.target:
            cmd += f" --target {args.target}"
    elif args.symbol:
        cmd += f" {args.symbol}"
        cmd += f" --target {args.target}"
        cmd += f" --timeframe {args.timeframe}"
        cmd += f" --years {args.years}"
        if args.compare:
            cmd += " --compare"
        if args.backtest:
            cmd += " --backtest"
        if args.fundamentals:
            cmd += " --fundamentals"
        if args.wc:
            cmd += " --wc"
    else:
        # No symbol provided, show help
        from rich.console import Console

        console = Console()
        console.print("[bold cyan]Stock ML - Machine Learning for Stock Prediction[/bold cyan]\n")
        console.print("[bold]Usage:[/bold]")
        console.print("  jotty stock-ml RELIANCE                    # Predict next day up/down")
        console.print("  jotty stock-ml RELIANCE --target next_30d_up   # 30-day prediction")
        console.print("  jotty stock-ml RELIANCE --backtest             # With backtesting")
        console.print("  jotty stock-ml --sweep --stocks nifty_bank     # Multi-stock sweep")
        console.print("  jotty stock-ml --leaderboard                   # Show sweep results")
        console.print("  jotty stock-ml --list                          # List available stocks")
        console.print("  jotty stock-ml --sets                          # Show stock sets")
        console.print("\n[bold]Targets:[/bold]")
        console.print("  next_1d_up, next_5d_up, next_30d_up     # Classification (up/down)")
        console.print("  return_5d, return_10d, return_30d       # Regression (return %)")
        console.print("\n[bold]Stock Sets:[/bold]")
        console.print("  nifty50, nifty_bank, nifty_it, nifty_pharma, nifty_auto, top10, top20")
        return 0

    return await cli.run_once(cmd)


def main() -> Any:
    """Main entry point for Jotty CLI."""
    parser = argparse.ArgumentParser(
        prog="jotty",
        description="Jotty CLI - Interactive Multi-Agent AI Assistant",
        epilog="""
Slash Commands (interactive mode):
  Enter interactive mode (run 'jotty' without args) for 30+ slash commands:
  /stock-ml, /ml, /research, /git, /browse, /mlflow, /swarm, /plan, and more.

  Use 'jotty commands' to list all available slash commands.

For more info: https://github.com/yourusername/jotty
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("-v", "--version", action="store_true", help="Show version and exit")

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Path to config file (default: ~/.jotty/config.yaml)",
    )

    parser.add_argument("--no-color", action="store_true", help="Disable colored output")

    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    # Subcommands for single execution
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # run command
    run_parser = subparsers.add_parser("run", help="Execute a task")
    run_parser.add_argument("goal", nargs="+", help="Task goal in natural language")

    # skills command
    skills_parser = subparsers.add_parser("skills", help="List available skills")
    skills_parser.add_argument("--category", "-c", help="Filter by category")

    # agents command
    subparsers.add_parser("agents", help="List agents")

    # warmup command
    warmup_parser = subparsers.add_parser("warmup", help="Run DrZero warmup")
    warmup_parser.add_argument(
        "--episodes", "-e", type=int, default=10, help="Number of warmup episodes"
    )

    # commands - list all slash commands
    commands_parser = subparsers.add_parser("commands", help="List all available slash commands")
    commands_parser.add_argument("--category", "-c", help="Filter by category")

    # stock-ml - direct access to stock ML pipeline
    stockml_parser = subparsers.add_parser(
        "stock-ml", help="Stock market ML prediction", aliases=["sml", "stockml"]
    )
    stockml_parser.add_argument("symbol", nargs="?", help="Stock symbol (e.g., RELIANCE, TCS)")
    stockml_parser.add_argument(
        "--target", "-t", default="next_1d_up", help="Target type (next_Nd_up, return_Nd)"
    )
    stockml_parser.add_argument(
        "--timeframe", "--tf", default="day", help="Timeframe (day, 15minute, etc.)"
    )
    stockml_parser.add_argument("--years", "-y", type=int, default=5, help="Years of data")
    stockml_parser.add_argument("--sweep", action="store_true", help="Run multi-stock sweep")
    stockml_parser.add_argument("--stocks", help="Stock set for sweep (nifty50, nifty_bank, top10)")
    stockml_parser.add_argument("--compare", action="store_true", help="Compare all targets")
    stockml_parser.add_argument(
        "--leaderboard", "--lb", action="store_true", help="Show sweep leaderboard"
    )
    stockml_parser.add_argument("--list", action="store_true", help="List available stocks")
    stockml_parser.add_argument("--sets", action="store_true", help="Show stock sets")
    stockml_parser.add_argument("--backtest", "--bt", action="store_true", help="Run backtesting")
    stockml_parser.add_argument(
        "--fundamentals", "--fund", action="store_true", help="Include fundamental features"
    )
    stockml_parser.add_argument(
        "--wc", "--world-class", action="store_true", help="Run world-class comprehensive backtest"
    )

    # ml - general ML command
    ml_parser = subparsers.add_parser("ml", help="Machine learning on any dataset")
    ml_parser.add_argument("dataset", nargs="?", help="Dataset path or name")
    ml_parser.add_argument("--target", "-t", help="Target column name")
    ml_parser.add_argument("--problem", "-p", help="Problem type (classification/regression)")

    # research - stock research
    research_parser = subparsers.add_parser("research", help="Comprehensive stock research")
    research_parser.add_argument("ticker", nargs="?", help="Stock ticker symbol")
    research_parser.add_argument("--pages", type=int, default=10, help="Target report pages")

    # mlflow - MLflow dashboard
    mlflow_parser = subparsers.add_parser("mlflow", help="MLflow experiment tracking")
    mlflow_parser.add_argument("--ui", action="store_true", help="Launch MLflow UI")
    mlflow_parser.add_argument("--experiments", action="store_true", help="List experiments")

    args = parser.parse_args()

    # Handle version
    if args.version:
        from .. import __version__ as jotty_version
        from . import __version__

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
    cli = JottyCLI(config_path=args.config, no_color=args.no_color, debug=args.debug)

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

    elif args.command == "commands":
        return asyncio.run(_list_commands(cli, args))

    elif args.command in ("stock-ml", "sml", "stockml"):
        return asyncio.run(_run_stock_ml(cli, args))

    elif args.command == "ml":
        cmd = "/ml"
        if args.dataset:
            cmd += f" {args.dataset}"
        if args.target:
            cmd += f" --target {args.target}"
        if args.problem:
            cmd += f" --problem {args.problem}"
        return asyncio.run(cli.run_once(cmd))

    elif args.command == "research":
        cmd = "/research"
        if args.ticker:
            cmd += f" {args.ticker}"
        cmd += f" --pages {args.pages}"
        return asyncio.run(cli.run_once(cmd))

    elif args.command == "mlflow":
        cmd = "/mlflow"
        if args.ui:
            cmd += " --ui"
        if args.experiments:
            cmd += " --experiments"
        return asyncio.run(cli.run_once(cmd))

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
