"""
Backtest Report Command
=======================

Generate comprehensive ML backtest PDF reports via Jotty CLI.

Usage:
    /backtest-report RELIANCE                     # Generate report with defaults
    /backtest-report RELIANCE --template citadel  # Use Citadel template
    /backtest-report RELIANCE --target next_5d_up # Specific target
    /backtest-report --list-templates             # List available templates

Templates:
    quantitative (default), two_sigma, renaissance, aqr, man_group, citadel
"""

from pathlib import Path
from typing import TYPE_CHECKING

from .base import BaseCommand, CommandResult, ParsedArgs

if TYPE_CHECKING:
    from ..app import JottyCLI


class BacktestReportCommand(BaseCommand):
    """Generate comprehensive ML backtest PDF reports."""

    name = "backtest-report"
    aliases = ["btr", "bt-report", "mlreport"]
    description = "Generate world-class ML backtest PDF reports"
    usage = "/backtest-report <symbol> [--template <name>] [--target <type>]"
    category = "ml"

    # Available templates
    TEMPLATES = {
        "quantitative": "Default quantitative research style",
        "two_sigma": "Two Sigma - clean, data-driven",
        "renaissance": "Renaissance Technologies - mathematical precision",
        "aqr": "AQR Capital - academic rigor",
        "man_group": "Man Group - institutional quality",
        "citadel": "Citadel - executive focus",
    }

    async def execute(self, args: ParsedArgs, cli: "JottyCLI") -> CommandResult:
        """Execute backtest report generation."""

        # List templates
        if "list-templates" in args.flags or "templates" in args.flags:
            return self._list_templates(cli)

        # Parse arguments
        symbol = args.positional[0].upper() if args.positional else None
        template = args.flags.get("template", args.flags.get("t", "quantitative"))
        target_type = args.flags.get("target", "next_5d_up")
        timeframe = args.flags.get("timeframe", args.flags.get("tf", "day"))
        years = int(args.flags.get("years", args.flags.get("y", "3")))

        if not symbol:
            cli.renderer.error("Stock symbol required.")
            cli.renderer.info("")
            cli.renderer.info("Usage: /backtest-report <SYMBOL> [options]")
            cli.renderer.info("")
            cli.renderer.info("Options:")
            cli.renderer.info("  --template <name>    Template style (default: quantitative)")
            cli.renderer.info("  --target <type>      Target type (default: next_5d_up)")
            cli.renderer.info("  --timeframe <tf>     Data timeframe (default: day)")
            cli.renderer.info("  --years <n>          Years of data (default: 3)")
            cli.renderer.info("")
            cli.renderer.info("Examples:")
            cli.renderer.info("  /backtest-report RELIANCE")
            cli.renderer.info("  /backtest-report TCS --template citadel")
            cli.renderer.info("  /backtest-report HDFCBANK --target next_10d_up")
            cli.renderer.info("")
            cli.renderer.info("Templates: " + ", ".join(self.TEMPLATES.keys()))
            return CommandResult.fail("Symbol required")

        # Validate template
        template_key = template.lower().replace(" ", "_")
        if template_key not in self.TEMPLATES:
            cli.renderer.error(f"Unknown template: {template}")
            cli.renderer.info(f"Available: {', '.join(self.TEMPLATES.keys())}")
            return CommandResult.fail("Invalid template")

        cli.renderer.header(f"ML Backtest Report: {symbol}")
        cli.renderer.info(f"Template: {template_key}")
        cli.renderer.info(f"Target: {target_type}")
        cli.renderer.info("")

        try:
            # Use stock_ml command to run the full pipeline with report generation
            from .stock_ml import StockMLCommand

            stock_ml = StockMLCommand()

            # Create args for stock_ml
            ml_args = ParsedArgs(
                positional=[symbol],
                flags={
                    "timeframe": timeframe,
                    "target": target_type,
                    "years": str(years),
                    "backtest": True,
                    "report": True,
                    "template": template_key,
                },
            )

            result = await stock_ml.execute(ml_args, cli)

            if result.success:
                data = result.data or {}
                report_paths = data.get("report_paths", {})

                if report_paths:
                    cli.renderer.info("")
                    cli.renderer.header("Report Generated Successfully")
                    cli.renderer.info("")

                    if report_paths.get("markdown"):
                        cli.renderer.info(f"Markdown: {report_paths['markdown']}")
                    if report_paths.get("pdf"):
                        cli.renderer.info(f"PDF: {report_paths['pdf']}")

                    # Summary
                    backtest = data.get("backtest", {})
                    if backtest:
                        strat = backtest.get("strategy", {})
                        bnh = backtest.get("bnh", {})

                        cli.renderer.info("")
                        cli.renderer.info("Performance Summary:")
                        cli.renderer.info(
                            f"  Strategy Return:  {strat.get('total_return', 0):+.2f}%"
                        )
                        cli.renderer.info(f"  Benchmark Return: {bnh.get('total_return', 0):+.2f}%")
                        cli.renderer.info(
                            f"  Alpha:            {backtest.get('outperformance', 0):+.2f}%"
                        )
                        cli.renderer.info(f"  Sharpe Ratio:     {strat.get('sharpe', 0):.2f}")
                        cli.renderer.info(
                            f"  Max Drawdown:     {strat.get('max_drawdown', 0):.2f}%"
                        )

                return result
            else:
                return result

        except Exception as e:
            cli.renderer.error(f"Report generation failed: {e}")
            import traceback

            traceback.print_exc()
            return CommandResult.fail(str(e))

    def _list_templates(self, cli: "JottyCLI") -> CommandResult:
        """List available report templates."""
        cli.renderer.header("Available Backtest Report Templates")
        cli.renderer.info("")

        cli.renderer.info("┌───────────────────┬────────────────────────────────────────────┐")
        cli.renderer.info("│     Template      │              Description                   │")
        cli.renderer.info("├───────────────────┼────────────────────────────────────────────┤")

        for name, desc in self.TEMPLATES.items():
            cli.renderer.info(f"│ {name:<17} │ {desc:<42} │")

        cli.renderer.info("└───────────────────┴────────────────────────────────────────────┘")

        cli.renderer.info("")
        cli.renderer.info("Usage: /backtest-report <SYMBOL> --template <name>")

        return CommandResult.ok(data=self.TEMPLATES)


class BatchBacktestReportCommand(BaseCommand):
    """Generate backtest reports for multiple stocks."""

    name = "batch-backtest"
    aliases = ["bb", "batch-bt"]
    description = "Generate backtest reports for multiple stocks"
    usage = "/batch-backtest --stocks top10 [--template <name>]"
    category = "ml"

    # Stock sets (same as stock_ml)
    STOCK_SETS = {
        "top5": ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"],
        "top10": [
            "RELIANCE",
            "TCS",
            "HDFCBANK",
            "INFY",
            "ICICIBANK",
            "HINDUNILVR",
            "SBIN",
            "BHARTIARTL",
            "ITC",
            "KOTAKBANK",
        ],
        "nifty_bank": ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK"],
        "nifty_it": ["TCS", "INFY", "WIPRO", "HCLTECH", "TECHM"],
        "nifty_pharma": ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "AUROPHARMA"],
    }

    async def execute(self, args: ParsedArgs, cli: "JottyCLI") -> CommandResult:
        """Execute batch backtest report generation."""

        # Get stock set
        stock_set = args.flags.get("stocks", args.flags.get("s", "top5"))
        template = args.flags.get("template", args.flags.get("t", "quantitative"))
        target_type = args.flags.get("target", "next_5d_up")

        # Get stocks
        if stock_set in self.STOCK_SETS:
            stocks = self.STOCK_SETS[stock_set]
        else:
            # Try to parse as comma-separated list
            stocks = [s.strip().upper() for s in stock_set.split(",")]

        cli.renderer.header(f"Batch Backtest Reports: {len(stocks)} stocks")
        cli.renderer.info(f"Stocks: {', '.join(stocks[:5])}{'...' if len(stocks) > 5 else ''}")
        cli.renderer.info(f"Template: {template}")
        cli.renderer.info(f"Target: {target_type}")
        cli.renderer.info("")

        results = []

        for i, symbol in enumerate(stocks, 1):
            cli.renderer.info(f"[{i}/{len(stocks)}] Processing {symbol}...")

            try:
                # Create backtest report command
                from .stock_ml import StockMLCommand

                stock_ml = StockMLCommand()

                ml_args = ParsedArgs(
                    positional=[symbol],
                    flags={
                        "timeframe": "day",
                        "target": target_type,
                        "years": "3",
                        "backtest": True,
                        "report": True,
                        "template": template,
                    },
                )

                result = await stock_ml.execute(ml_args, cli)

                if result.success:
                    data = result.data or {}
                    report_paths = data.get("report_paths", {})
                    backtest = data.get("backtest", {})

                    results.append(
                        {
                            "symbol": symbol,
                            "status": "success",
                            "pdf_path": report_paths.get("pdf"),
                            "total_return": backtest.get("strategy", {}).get("total_return", 0),
                            "sharpe": backtest.get("strategy", {}).get("sharpe", 0),
                            "outperformance": backtest.get("outperformance", 0),
                        }
                    )
                    cli.renderer.info(
                        f" {symbol}: Return={backtest.get('strategy', {}).get('total_return', 0):+.1f}%, Sharpe={backtest.get('strategy', {}).get('sharpe', 0):.2f}"
                    )
                else:
                    results.append({"symbol": symbol, "status": "failed"})
                    cli.renderer.info(f" {symbol}: Failed")

            except Exception as e:
                results.append({"symbol": symbol, "status": "error", "error": str(e)})
                cli.renderer.info(f" {symbol}: Error - {e}")

        # Summary
        successful = [r for r in results if r["status"] == "success"]

        cli.renderer.info("")
        cli.renderer.header("Batch Summary")
        cli.renderer.info(f"Completed: {len(successful)}/{len(stocks)}")

        if successful:
            cli.renderer.info("")
            cli.renderer.info(
                "┌─────────────┬─────────────┬─────────┬─────────────────────────────────┐"
            )
            cli.renderer.info(
                "│   Symbol    │   Return    │  Sharpe │              PDF                │"
            )
            cli.renderer.info(
                "├─────────────┼─────────────┼─────────┼─────────────────────────────────┤"
            )

            for r in sorted(successful, key=lambda x: -x.get("total_return", 0)):
                pdf_display = Path(r.get("pdf_path", "")).name if r.get("pdf_path") else "-"
                cli.renderer.info(
                    f"│ {r['symbol']:<11} │ {r.get('total_return', 0):>+9.1f}% │ {r.get('sharpe', 0):>7.2f} │ {pdf_display:<31} │"
                )

            cli.renderer.info(
                "└─────────────┴─────────────┴─────────┴─────────────────────────────────┘"
            )

        return CommandResult.ok(data={"results": results})
