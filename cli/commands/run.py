"""
Run Command
===========

Execute tasks with SwarmManager.
"""

import time
from typing import TYPE_CHECKING
from .base import BaseCommand, CommandResult, ParsedArgs

if TYPE_CHECKING:
    from ..app import JottyCLI


class RunCommand(BaseCommand):
    """Execute tasks with SwarmManager."""

    name = "run"
    aliases = ["r", "do"]
    description = "Execute a task using SwarmManager with zero-config intelligence"
    usage = "/run <goal> [--agent <name>] [--verbose]"
    category = "execution"

    async def execute(self, args: ParsedArgs, cli: "JottyCLI") -> CommandResult:
        """Execute task with SwarmManager."""
        if not args.positional and not args.raw.strip():
            return CommandResult.fail("Usage: /run <goal>")

        # Get goal from args
        goal = args.raw.strip()
        if goal.startswith("run "):
            goal = goal[4:].strip()
        elif goal.startswith("r ") or goal.startswith("do "):
            goal = goal[2:].strip()

        # Remove flags from goal
        for flag in ["--agent", "-a", "--verbose", "-v"]:
            if flag in goal:
                parts = goal.split(flag)
                goal = parts[0].strip()

        if not goal:
            return CommandResult.fail("Please specify a task goal")

        # Get options
        verbose = args.flags.get("verbose", False) or args.flags.get("v", False)
        agent_name = args.flags.get("agent") or args.flags.get("a")

        # Zero-config swarm mode - system decides single vs multi-agent
        cli.renderer.info("Swarm mode: zero-config (system decides single/multi-agent)")

        start_time = time.time()

        try:
            # Initialize swarm if needed
            swarm = await cli.get_swarm_manager()

            # Status callback for streaming progress
            def status_callback(stage: str, detail: str = ""):
                """Stream progress updates to the CLI."""
                if detail:
                    cli.renderer.print(f"  [cyan]→[/cyan] {stage}: {detail}")
                else:
                    cli.renderer.print(f"  [cyan]→[/cyan] {stage}")

            # Add to conversation history
            cli.session.add_message("user", goal)

            # Execute with full swarm intelligence
            result = await swarm.run(
                goal,
                status_callback=status_callback
            )

            elapsed = time.time() - start_time

            # Add result to history
            cli.session.add_message("assistant", str(result.output) if result.output else "Task completed")

            # Display result with clear success/failure
            cli.renderer.newline()

            if result.success:
                cli.renderer.success(f"Task completed in {elapsed:.1f}s")

                # Extract and display file paths + summary from output
                output = result.output if hasattr(result, 'output') else result
                file_paths = []
                summary = {}

                # Handle ExecutionResult (from AutoAgent/SwarmManager)
                if hasattr(output, 'outputs') and hasattr(output, 'skills_used'):
                    outputs_dict = output.outputs or {}
                    seen_paths = set()
                    for step_key, step_result in outputs_dict.items():
                        if isinstance(step_result, dict):
                            for key in ['pdf_path', 'md_path', 'output_path', 'file_path', 'image_path']:
                                if key in step_result and step_result[key]:
                                    path = step_result[key]
                                    if path not in seen_paths:
                                        file_paths.append((key.replace('_', ' ').title(), path))
                                        seen_paths.add(path)
                            for key in ['success', 'ticker', 'company_name', 'word_count', 'telegram_sent']:
                                if key in step_result and step_result[key]:
                                    summary[key] = step_result[key]

                    # Add skills and steps to summary
                    if output.skills_used:
                        summary['skills'] = ', '.join(output.skills_used)
                    if output.steps_executed:
                        summary['steps'] = output.steps_executed

                elif isinstance(output, dict):
                    for key in ['pdf_path', 'md_path', 'output_path', 'file_path']:
                        if key in output and output[key]:
                            file_paths.append((key.replace('_', ' ').title(), output[key]))
                    for key in ['success', 'ticker', 'company_name', 'word_count', 'telegram_sent']:
                        if key in output and output[key]:
                            summary[key] = output[key]

                elif isinstance(output, str):
                    import re
                    path_matches = re.findall(r'(/[\w/\-_.]+\.(pdf|md|txt|html|json|csv|png|jpg|docx))', output)
                    for match in path_matches:
                        file_paths.append(('Output', match[0]))

                # Display file paths prominently
                if file_paths:
                    cli.renderer.newline()
                    cli.renderer.print("[bold green]📁 Generated Files:[/bold green]")
                    for label, path in file_paths:
                        cli.renderer.print(f"   {label}: [cyan]{path}[/cyan]")

                # Show summary
                if summary:
                    cli.renderer.newline()
                    cli.renderer.panel(
                        "\n".join([f"• {k}: {v}" for k, v in summary.items()]),
                        title="Summary",
                        style="green"
                    )
                elif not file_paths:
                    # Fallback: show final_output or string repr (truncated)
                    final = getattr(output, 'final_output', None) or str(output)
                    final_str = str(final)[:500]
                    if len(str(final)) > 500:
                        final_str += "..."
                    if final_str.strip():
                        cli.renderer.newline()
                        cli.renderer.panel(final_str, title="Output", style="green")

                if verbose and hasattr(result, "trajectory") and result.trajectory:
                    cli.renderer.panel(
                        "\n".join([f"• {step}" for step in result.trajectory[:10]]),
                        title="Trajectory",
                        style="dim"
                    )

                return CommandResult.ok(data=result)
            else:
                cli.renderer.error(f"Task failed after {elapsed:.1f}s")

                # Show error details
                error_msg = getattr(result, 'error', None)
                if not error_msg and hasattr(result, 'alerts') and result.alerts:
                    error_msg = "; ".join(result.alerts[:3])

                if error_msg:
                    cli.renderer.panel(error_msg, title="Error Details", style="red")

                return CommandResult.fail(error_msg or "Unknown error", data=result)

        except Exception as e:
            elapsed = time.time() - start_time
            cli.renderer.error(f"Execution error after {elapsed:.1f}s: {e}")
            if cli.config.debug:
                import traceback
                traceback.print_exc()
            return CommandResult.fail(str(e))

    def get_completions(self, partial: str) -> list:
        """Get task completions (example tasks)."""
        examples = [
            "search for latest AI news",
            "summarize this document",
            "analyze data in file.csv",
            "write a Python script",
            "research machine learning trends",
        ]
        return [e for e in examples if e.startswith(partial)]
