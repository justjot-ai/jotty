"""
Plan Command
============

Task planning and decomposition.
"""

from typing import TYPE_CHECKING
from .base import BaseCommand, CommandResult, ParsedArgs

if TYPE_CHECKING:
    from ..app import JottyCLI


class PlanCommand(BaseCommand):
    """Task planning and decomposition."""

    name = "plan"
    aliases = ["p"]
    description = "Plan mode for task decomposition and analysis"
    usage = "/plan <task> [--execute] [--verbose]"
    category = "execution"

    async def execute(self, args: ParsedArgs, cli: "JottyCLI") -> CommandResult:
        """Execute plan command."""
        if not args.positional and not args.raw.strip():
            return CommandResult.fail("Usage: /plan <task description>")

        # Get task from args
        task = args.raw.strip()
        if task.startswith("plan "):
            task = task[5:].strip()
        elif task.startswith("p "):
            task = task[2:].strip()

        # Remove flags from task
        for flag in ["--execute", "-e", "--verbose", "-v"]:
            if flag in task:
                parts = task.split(flag)
                task = parts[0].strip()

        if not task:
            return CommandResult.fail("Please specify a task to plan")

        execute = args.flags.get("execute", False) or args.flags.get("e", False)
        verbose = args.flags.get("verbose", False) or args.flags.get("v", False)

        return await self._create_plan(task, execute, verbose, cli)

    async def _create_plan(
        self,
        task: str,
        execute: bool,
        verbose: bool,
        cli: "JottyCLI"
    ) -> CommandResult:
        """Create task plan."""
        try:
            swarm = await cli.get_swarm_manager()

            cli.renderer.info(f"Planning: {task}")

            # Use intent parser to analyze task
            async with await cli.renderer.progress.spinner_async(
                "Analyzing task...",
                style="cyan"
            ):
                task_graph = swarm.swarm_intent_parser.parse(task)

            # Build plan info
            plan = {
                "Task Type": task_graph.task_type.value if hasattr(task_graph.task_type, "value") else str(task_graph.task_type),
                "Workflow": task_graph.workflow,
                "Operations": task_graph.operations,
                "Requirements": task_graph.requirements,
                "Integrations": task_graph.integrations,
            }

            # Get best agent recommendation
            best_agent = swarm.get_best_agent_for_task(task)
            if best_agent:
                plan["Recommended Agent"] = best_agent

            # Get swarm wisdom
            if verbose:
                wisdom = swarm.get_swarm_wisdom(task)
                if wisdom:
                    plan["Swarm Wisdom"] = wisdom[:500] + "..." if len(wisdom) > 500 else wisdom

            cli.renderer.tree(plan, title="Task Plan")

            # Execute if requested
            if execute:
                cli.renderer.info("Executing plan...")
                result = await swarm.run(task)

                if result.success:
                    cli.renderer.success("Plan executed successfully")
                    cli.renderer.result(result, title="Execution Result")
                else:
                    cli.renderer.error(f"Execution failed: {result.error}")

                return CommandResult.ok(data={"plan": plan, "result": result})

            return CommandResult.ok(data=plan)

        except Exception as e:
            cli.renderer.error(f"Planning failed: {e}")
            if cli.config.debug:
                import traceback
                traceback.print_exc()
            return CommandResult.fail(str(e))

    def get_completions(self, partial: str) -> list:
        """Get task completions."""
        examples = [
            "research machine learning trends",
            "analyze data in file.csv",
            "create a Python web scraper",
            "summarize this document",
        ]
        return [e for e in examples if e.startswith(partial)]
