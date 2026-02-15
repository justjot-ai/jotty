"""
Learn Command
=============

Learning, warmup, and curriculum management.
"""

from typing import TYPE_CHECKING

from .base import BaseCommand, CommandResult, ParsedArgs

if TYPE_CHECKING:
    from ..app import JottyCLI


class LearnCommand(BaseCommand):
    """Learning and training management."""

    name = "learn"
    aliases = ["l"]
    description = "Manage learning: warmup, curriculum, and training"
    usage = "/learn [warmup <episodes>|status|recommend|save|load <path>]"
    category = "learning"

    async def execute(self, args: ParsedArgs, cli: "JottyCLI") -> CommandResult:
        """Execute learn command."""
        subcommand = args.positional[0] if args.positional else "status"

        if subcommand == "warmup":
            episodes = int(args.positional[1]) if len(args.positional) > 1 else 10
            verbose = args.flags.get("verbose", False) or args.flags.get("v", False)
            return await self._warmup(episodes, verbose, cli)
        elif subcommand == "status":
            return await self._show_status(cli)
        elif subcommand == "recommend":
            return await self._show_recommendation(cli)
        elif subcommand == "save":
            path = args.positional[1] if len(args.positional) > 1 else None
            return await self._save_learnings(path, cli)
        elif subcommand == "load":
            path = args.positional[1] if len(args.positional) > 1 else None
            return await self._load_learnings(path, cli)
        else:
            return await self._show_status(cli)

    async def _warmup(self, episodes: int, verbose: bool, cli: "JottyCLI") -> CommandResult:
        """Run DrZero warmup."""
        try:
            swarm = await cli.get_swarm_manager()

            cli.renderer.info(f"Starting warmup: {episodes} episodes")

            # Run warmup with progress
            async with await cli.renderer.progress.spinner_async(
                f"Running warmup ({episodes} episodes)...", style="cyan"
            ) as spinner:
                stats = await swarm.warmup(num_episodes=episodes, verbose=verbose)

            # Display results
            cli.renderer.success("Warmup complete!")

            result_info = {
                "Episodes Run": stats["episodes_run"],
                "Successes": stats["successes"],
                "Failures": stats["failures"],
                "Success Rate": f"{stats['success_rate']:.1%}",
            }

            if stats.get("agent_improvements"):
                result_info["Agent Improvements"] = {
                    agent: f"{rate:.1%}" for agent, rate in stats["agent_improvements"].items()
                }

            if stats.get("task_type_results"):
                result_info["Task Types"] = {
                    task_type: f"{r['success']}/{r['total']}"
                    for task_type, r in stats["task_type_results"].items()
                }

            cli.renderer.tree(result_info, title="Warmup Results")
            return CommandResult.ok(data=stats)

        except Exception as e:
            cli.renderer.error(f"Warmup failed: {e}")
            return CommandResult.fail(str(e))

    async def _show_status(self, cli: "JottyCLI") -> CommandResult:
        """Show learning status."""
        try:
            swarm = await cli.get_swarm_manager()

            status = {
                "Episodes": swarm.episode_count,
                "Learning Enabled": cli.config.swarm.enable_learning,
            }

            # Q-learning stats
            if hasattr(swarm, "learning_manager"):
                q_summary = swarm.learning_manager.get_q_table_summary()
                status["Q-Table Size"] = q_summary.get("size", 0)
                status["Avg Q-Value"] = f"{q_summary.get('avg_value', 0):.4f}"

            # Transfer learning stats
            if hasattr(swarm, "transfer_learning"):
                tl = swarm.transfer_learning
                status["Transfer Learning"] = {
                    "Experiences": len(tl.experiences),
                    "Role Profiles": len(tl.role_profiles),
                }

            # Curriculum stats
            if hasattr(swarm, "swarm_intelligence"):
                curriculum = swarm.swarm_intelligence.curriculum_generator
                status["Curriculum"] = {
                    "Task Templates": len(curriculum.task_templates),
                    "Episodes Generated": curriculum.episodes_generated,
                }

            cli.renderer.tree(status, title="Learning Status")
            return CommandResult.ok(data=status)

        except Exception as e:
            cli.renderer.error(f"Failed to get learning status: {e}")
            return CommandResult.fail(str(e))

    async def _show_recommendation(self, cli: "JottyCLI") -> CommandResult:
        """Show warmup recommendation."""
        try:
            swarm = await cli.get_swarm_manager()

            recommendation = swarm.get_warmup_recommendation()

            if recommendation["should_warmup"]:
                cli.renderer.warning(f"Warmup recommended: {recommendation['reason']}")
                cli.renderer.info(f"Suggested episodes: {recommendation['recommended_episodes']}")

                if recommendation.get("weak_areas"):
                    cli.renderer.panel(
                        "\n".join([f"• {area}" for area in recommendation["weak_areas"]]),
                        title="Weak Areas",
                        style="yellow",
                    )
            else:
                cli.renderer.success("Learning state is healthy - no warmup needed")

            return CommandResult.ok(data=recommendation)

        except Exception as e:
            cli.renderer.error(f"Failed to get recommendation: {e}")
            return CommandResult.fail(str(e))

    async def _save_learnings(self, path: str, cli: "JottyCLI") -> CommandResult:
        """Save learnings to file."""
        try:
            swarm = await cli.get_swarm_manager()

            # Use internal save method
            swarm._auto_save_learnings()

            cli.renderer.success("Learnings saved successfully")
            return CommandResult.ok()

        except Exception as e:
            cli.renderer.error(f"Failed to save learnings: {e}")
            return CommandResult.fail(str(e))

    async def _load_learnings(self, path: str, cli: "JottyCLI") -> CommandResult:
        """Load learnings from file."""
        try:
            swarm = await cli.get_swarm_manager()

            # Use internal load method
            swarm._auto_load_learnings()

            cli.renderer.success("Learnings loaded successfully")
            return CommandResult.ok()

        except Exception as e:
            cli.renderer.error(f"Failed to load learnings: {e}")
            return CommandResult.fail(str(e))

    def get_completions(self, partial: str) -> list:
        """Get subcommand completions."""
        subcommands = ["warmup", "status", "recommend", "save", "load"]
        return [s for s in subcommands if s.startswith(partial)]
