"""
MLflow Command
==============

Manage MLflow experiments and models.

Usage:
    /mlflow list                    # List recent runs
    /mlflow runs --experiment myexp # List runs in experiment
    /mlflow best --metric accuracy  # Get best run by metric
    /mlflow load <run_id>           # Load model from run
    /mlflow compare <id1> <id2>     # Compare runs
    /mlflow ui                      # Launch MLflow UI
"""

from typing import TYPE_CHECKING, Dict, Any, Optional, List
import asyncio

from .base import BaseCommand, CommandResult, ParsedArgs
from .ml import MLCommand  # Import to access MLflow state

if TYPE_CHECKING:
    from ..app import JottyCLI


class MLflowCommand(BaseCommand):
    """Manage MLflow experiments and models."""

    name = "mlflow"
    aliases = ["mlf", "runs"]
    description = "Manage MLflow experiments, runs, and models"
    usage = "/mlflow <subcommand> [options]"
    category = "ml"

    def _get_default_experiment(self, args: ParsedArgs) -> str:
        """Get experiment name from args or saved state."""
        # First check if explicitly provided
        if "experiment" in args.flags or "exp" in args.flags:
            return args.flags.get("experiment", args.flags.get("exp"))

        # Otherwise load from saved state
        state = MLCommand.load_mlflow_state()
        return state.get("experiment_name", "jotty_ml")

    def _get_saved_state(self) -> Dict[str, Any]:
        """Get saved MLflow state."""
        return MLCommand.load_mlflow_state()

    async def execute(self, args: ParsedArgs, cli: "JottyCLI") -> CommandResult:
        """Execute MLflow command."""
        subcommand = args.positional[0] if args.positional else "list"

        # Initialize MLflow tracker with saved state
        from Jotty.core.skills.ml import MLflowTrackerSkill

        state = self._get_saved_state()
        tracker = MLflowTrackerSkill()
        await tracker.init(
            tracking_uri=state.get("tracking_uri"),
            experiment_name=self._get_default_experiment(args)
        )

        if tracker._mlflow is None:
            cli.renderer.error("MLflow not installed. Install with: pip install mlflow")
            return CommandResult.fail("MLflow not available")

        if subcommand == "list" or subcommand == "runs":
            return await self._list_runs(args, cli, tracker)
        elif subcommand == "best":
            return await self._get_best_run(args, cli, tracker)
        elif subcommand == "load":
            return await self._load_model(args, cli, tracker)
        elif subcommand == "compare":
            return await self._compare_runs(args, cli, tracker)
        elif subcommand == "ui":
            return await self._launch_ui(args, cli, tracker)
        elif subcommand == "experiments":
            return await self._list_experiments(args, cli, tracker)
        elif subcommand == "status":
            return await self._show_status(args, cli, tracker)
        elif subcommand == "help":
            self._show_help(cli)
            return CommandResult.ok()
        else:
            cli.renderer.error(f"Unknown subcommand: {subcommand}")
            self._show_help(cli)
            return CommandResult.fail(f"Unknown subcommand: {subcommand}")

    async def _list_runs(self, args: ParsedArgs, cli: "JottyCLI", tracker) -> CommandResult:
        """List runs in an experiment."""
        experiment = self._get_default_experiment(args)
        max_results = int(args.flags.get("limit", args.flags.get("n", "10")))

        cli.renderer.header(f"MLflow Runs - {experiment}")

        runs = await tracker.list_runs(
            experiment_name=experiment,
            max_results=max_results
        )

        if not runs:
            cli.renderer.info("No runs found.")
            return CommandResult.ok(data=[])

        # Display runs in a table format
        cli.renderer.info(f"{'Run ID':<36} {'Score':<10} {'Status':<10} {'Start Time':<20}")
        cli.renderer.info("-" * 80)

        for run in runs:
            run_id = run.get('run_id', run.get('run_uuid', 'N/A'))[:36]
            score = run.get('metrics.best_score', run.get('metrics.accuracy', 'N/A'))
            if isinstance(score, float):
                score = f"{score:.4f}"
            status = run.get('status', 'N/A')
            start_time = run.get('start_time', 'N/A')
            if isinstance(start_time, (int, float)):
                import datetime
                start_time = datetime.datetime.fromtimestamp(start_time/1000).strftime('%Y-%m-%d %H:%M')

            cli.renderer.info(f"{run_id:<36} {str(score):<10} {status:<10} {str(start_time):<20}")

        return CommandResult.ok(data=runs)

    async def _get_best_run(self, args: ParsedArgs, cli: "JottyCLI", tracker) -> CommandResult:
        """Get the best run by metric."""
        experiment = self._get_default_experiment(args)
        metric = args.flags.get("metric", args.flags.get("m", "best_score"))
        ascending = args.flags.get("ascending", "false").lower() == "true"

        cli.renderer.header(f"Best Run - {experiment}")

        best = await tracker.get_best_run(
            experiment_name=experiment,
            metric=metric,
            ascending=ascending
        )

        if not best:
            cli.renderer.info("No runs found.")
            return CommandResult.ok(data=None)

        run_id = best.get('run_id', best.get('run_uuid', 'N/A'))
        cli.renderer.info(f"Run ID: {run_id}")
        cli.renderer.info(f"Score ({metric}): {best.get(f'metrics.{metric}', 'N/A')}")
        cli.renderer.info(f"Status: {best.get('status', 'N/A')}")

        # Show all metrics
        cli.renderer.info("")
        cli.renderer.subheader("Metrics")
        for key, value in best.items():
            if key.startswith('metrics.'):
                metric_name = key.replace('metrics.', '')
                if isinstance(value, float):
                    cli.renderer.info(f"  {metric_name}: {value:.4f}")
                else:
                    cli.renderer.info(f"  {metric_name}: {value}")

        # Show params
        cli.renderer.info("")
        cli.renderer.subheader("Parameters")
        for key, value in best.items():
            if key.startswith('params.'):
                param_name = key.replace('params.', '')
                cli.renderer.info(f"  {param_name}: {value}")

        return CommandResult.ok(data=best)

    async def _load_model(self, args: ParsedArgs, cli: "JottyCLI", tracker) -> CommandResult:
        """Load a model from a run."""
        if len(args.positional) < 2:
            cli.renderer.error("Run ID required. Usage: /mlflow load <run_id>")
            return CommandResult.fail("Run ID required")

        run_id = args.positional[1]
        model_name = args.flags.get("model", args.flags.get("m", "best_ensemble"))

        cli.renderer.info(f"Loading model from run: {run_id}")

        model = await tracker.load_model(run_id=run_id, model_name=model_name)

        if model is None:
            cli.renderer.error(f"Failed to load model from run {run_id}")
            return CommandResult.fail("Model load failed")

        cli.renderer.success(f"Model loaded successfully!")
        cli.renderer.info(f"Type: {type(model).__name__}")

        # Store in CLI context for later use
        return CommandResult.ok(data={'model': model, 'run_id': run_id})

    async def _compare_runs(self, args: ParsedArgs, cli: "JottyCLI", tracker) -> CommandResult:
        """Compare multiple runs."""
        run_ids = args.positional[1:] if len(args.positional) > 1 else []

        if len(run_ids) < 2:
            cli.renderer.error("At least 2 run IDs required. Usage: /mlflow compare <id1> <id2> ...")
            return CommandResult.fail("Need at least 2 run IDs")

        cli.renderer.header("Run Comparison")

        comparison = await tracker.compare_runs(run_ids)

        if comparison.empty:
            cli.renderer.info("No comparison data available.")
            return CommandResult.ok(data=None)

        # Display comparison
        cli.renderer.info(comparison.to_string())

        return CommandResult.ok(data=comparison.to_dict('records'))

    async def _launch_ui(self, args: ParsedArgs, cli: "JottyCLI", tracker) -> CommandResult:
        """Launch MLflow UI."""
        import subprocess
        import os

        port = args.flags.get("port", args.flags.get("p", "5000"))
        tracking_uri = tracker._tracking_uri or "mlruns"

        cli.renderer.info(f"Launching MLflow UI on port {port}...")
        cli.renderer.info(f"Tracking URI: {tracking_uri}")
        cli.renderer.info(f"Open http://localhost:{port} in your browser")

        try:
            # Launch in background
            process = subprocess.Popen(
                ["mlflow", "ui", "--port", str(port), "--backend-store-uri", tracking_uri],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            cli.renderer.success(f"MLflow UI started (PID: {process.pid})")
            return CommandResult.ok(data={'pid': process.pid, 'port': port})

        except FileNotFoundError:
            cli.renderer.error("MLflow CLI not found. Install with: pip install mlflow")
            return CommandResult.fail("MLflow CLI not available")
        except Exception as e:
            cli.renderer.error(f"Failed to launch MLflow UI: {e}")
            return CommandResult.fail(str(e))

    async def _list_experiments(self, args: ParsedArgs, cli: "JottyCLI", tracker) -> CommandResult:
        """List all experiments."""
        cli.renderer.header("MLflow Experiments")

        try:
            experiments = tracker._mlflow.search_experiments()

            for exp in experiments:
                cli.renderer.info(f"  {exp.experiment_id}: {exp.name}")
                cli.renderer.info(f"     Artifact Location: {exp.artifact_location}")
                cli.renderer.info("")

            return CommandResult.ok(data=[{'id': e.experiment_id, 'name': e.name} for e in experiments])

        except Exception as e:
            cli.renderer.error(f"Failed to list experiments: {e}")
            return CommandResult.fail(str(e))

    async def _show_status(self, args: ParsedArgs, cli: "JottyCLI", tracker) -> CommandResult:
        """Show current MLflow status and saved state."""
        state = self._get_saved_state()

        cli.renderer.header("MLflow Status")
        cli.renderer.info("")
        cli.renderer.info(f"Current Experiment: {state.get('experiment_name', 'jotty_ml')}")
        cli.renderer.info(f"Last Run ID:        {state.get('last_run_id', 'None')}")
        cli.renderer.info(f"Tracking URI:       {state.get('tracking_uri') or tracker._tracking_uri or 'mlruns (local)'}")
        cli.renderer.info("")

        # Show tip
        cli.renderer.info("Tip: The experiment name is remembered from your last /ml --mlflow run.")
        cli.renderer.info("     Override with: /mlflow list --experiment <name>")

        return CommandResult.ok(data=state)

    def _show_help(self, cli: "JottyCLI"):
        """Show command help."""
        state = self._get_saved_state()
        current_exp = state.get('experiment_name', 'jotty_ml')

        cli.renderer.info("")
        cli.renderer.info(f"Current experiment: {current_exp}")
        cli.renderer.info("")
        cli.renderer.info("Available subcommands:")
        cli.renderer.info("  list, runs    - List runs in an experiment")
        cli.renderer.info("  best          - Get the best run by metric")
        cli.renderer.info("  load <id>     - Load model from a run")
        cli.renderer.info("  compare       - Compare multiple runs")
        cli.renderer.info("  experiments   - List all experiments")
        cli.renderer.info("  status        - Show current MLflow status")
        cli.renderer.info("  ui            - Launch MLflow UI")
        cli.renderer.info("  help          - Show this help")
        cli.renderer.info("")
        cli.renderer.info("Options:")
        cli.renderer.info("  --experiment  - Experiment name (default: jotty_ml)")
        cli.renderer.info("  --metric      - Metric to optimize (default: best_score)")
        cli.renderer.info("  --limit       - Max results to show (default: 10)")

    def get_completions(self, partial: str) -> list:
        """Get completions."""
        subcommands = ["list", "runs", "best", "load", "compare", "experiments", "status", "ui", "help"]
        flags = ["--experiment", "--metric", "--limit", "--model", "--port", "--ascending"]
        all_completions = subcommands + flags
        return [s for s in all_completions if s.startswith(partial)]
