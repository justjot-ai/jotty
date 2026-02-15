"""
Stats Command
=============

Learning statistics and metrics.
"""

from typing import TYPE_CHECKING, Any, Dict

from .base import BaseCommand, CommandResult, ParsedArgs

if TYPE_CHECKING:
    from ..app import JottyCLI


class StatsCommand(BaseCommand):
    """Learning statistics and metrics."""

    name = "stats"
    aliases = ["st"]
    description = "View learning statistics, Q-values, and episode history"
    usage = "/stats [summary|q-table|episodes|agents]"
    category = "learning"

    async def execute(self, args: ParsedArgs, cli: "JottyCLI") -> CommandResult:
        """Execute stats command."""
        subcommand = args.positional[0] if args.positional else "summary"

        if subcommand == "summary":
            return await self._show_summary(cli)
        elif subcommand in ["q-table", "qtable", "q"]:
            return await self._show_q_table(cli)
        elif subcommand == "episodes":
            limit = int(args.positional[1]) if len(args.positional) > 1 else 10
            return await self._show_episodes(limit, cli)
        elif subcommand == "agents":
            return await self._show_agent_stats(cli)
        else:
            return await self._show_summary(cli)

    async def _show_summary(self, cli: "JottyCLI") -> CommandResult:
        """Show learning summary."""
        try:
            swarm = await cli.get_swarm_manager()

            stats: Dict[str, Any] = {
                "Episodes": swarm.episode_count,
            }

            # Q-learning stats
            if hasattr(swarm, "learning_manager"):
                q_summary = swarm.learning_manager.get_q_table_summary()
                stats.update(
                    {
                        "Q-Table Size": q_summary.get("size", 0),
                        "Avg Q-Value": q_summary.get("avg_value", 0),
                        "Max Q-Value": q_summary.get("max_value", 0),
                        "Min Q-Value": q_summary.get("min_value", 0),
                    }
                )

            # Adaptive learning rate
            if hasattr(swarm, "learning_manager") and hasattr(
                swarm.learning_manager, "adaptive_lr"
            ):
                alr = swarm.learning_manager.adaptive_lr
                if alr:
                    stats["Adaptive Alpha"] = getattr(alr, "current_alpha", 0.1)

            # Swarm intelligence
            if hasattr(swarm, "swarm_intelligence"):
                si = swarm.swarm_intelligence
                profiles = si.agent_profiles

                total_tasks = sum(p.total_tasks for p in profiles.values())
                # Calculate successes from task_success dict: sum of success counts
                total_success = sum(
                    sum(s for s, t in p.task_success.values()) for p in profiles.values()
                )

                stats.update(
                    {
                        "Total Tasks": total_tasks,
                        "Total Successes": total_success,
                        "Overall Success Rate": total_success / max(1, total_tasks),
                    }
                )

            # Credit weights
            if hasattr(swarm, "credit_weights"):
                stats["Credit Weights"] = swarm.credit_weights.to_dict()

            table = cli.renderer.tables.stats_table(stats)
            cli.renderer.tables.print_table(table)

            return CommandResult.ok(data=stats)

        except Exception as e:
            cli.renderer.error(f"Failed to get stats: {e}")
            return CommandResult.fail(str(e))

    async def _show_q_table(self, cli: "JottyCLI") -> CommandResult:
        """Show Q-table entries."""
        try:
            swarm = await cli.get_swarm_manager()

            if not hasattr(swarm, "learning_manager"):
                cli.renderer.warning("Learning manager not available")
                return CommandResult.fail("Learning manager not available")

            q_learner = swarm.learning_manager.q_learner

            if not hasattr(q_learner, "q_table") or not q_learner.q_table:
                cli.renderer.info("Q-table is empty")
                return CommandResult.ok(data={})

            # Get top entries
            entries = []
            for key, value in list(q_learner.q_table.items())[:20]:
                entries.append(
                    {
                        "state_action": str(key)[:60],
                        "q_value": f"{value:.4f}",
                    }
                )

            cli.renderer.panel(
                "\n".join([f"{e['state_action']}: {e['q_value']}" for e in entries]),
                title=f"Q-Table ({len(q_learner.q_table)} entries)",
                style="cyan",
            )

            return CommandResult.ok(data=entries)

        except Exception as e:
            cli.renderer.error(f"Failed to get Q-table: {e}")
            return CommandResult.fail(str(e))

    async def _show_episodes(self, limit: int, cli: "JottyCLI") -> CommandResult:
        """Show episode history."""
        try:
            swarm = await cli.get_swarm_manager()

            if not hasattr(swarm, "learning_manager"):
                cli.renderer.warning("Learning manager not available")
                return CommandResult.fail("Learning manager not available")

            lm = swarm.learning_manager

            if not hasattr(lm, "episode_history") or not lm.episode_history:
                cli.renderer.info("No episode history")
                return CommandResult.ok(data=[])

            episodes = lm.episode_history[-limit:]

            cli.renderer.panel(
                "\n".join(
                    [
                        f"Episode {i+1}: reward={ep.get('reward', 0):.2f}, success={ep.get('success', False)}"
                        for i, ep in enumerate(episodes)
                    ]
                ),
                title=f"Recent Episodes (last {len(episodes)})",
                style="blue",
            )

            return CommandResult.ok(data=episodes)

        except Exception as e:
            cli.renderer.error(f"Failed to get episodes: {e}")
            return CommandResult.fail(str(e))

    async def _show_agent_stats(self, cli: "JottyCLI") -> CommandResult:
        """Show per-agent statistics."""
        try:
            swarm = await cli.get_swarm_manager()

            if not hasattr(swarm, "swarm_intelligence"):
                cli.renderer.warning("Swarm intelligence not available")
                return CommandResult.fail("Swarm intelligence not available")

            profiles = swarm.swarm_intelligence.agent_profiles

            if not profiles:
                cli.renderer.info("No agent statistics yet")
                return CommandResult.ok(data={})

            agent_stats = {}
            for name, profile in profiles.items():
                # Calculate successes from task_success dict
                successes = sum(s for s, t in profile.task_success.values())
                success_rate = successes / max(1, profile.total_tasks)
                agent_stats[name] = {
                    "Total Tasks": profile.total_tasks,
                    "Successes": successes,
                    "Success Rate": f"{success_rate:.1%}",
                    "Trust Score": f"{profile.trust_score:.2f}",
                    "Specialization": (
                        str(profile.specialization.value)
                        if hasattr(profile.specialization, "value")
                        else str(profile.specialization)
                    ),
                    "Avg Execution Time": f"{profile.avg_execution_time:.2f}s",
                }

            cli.renderer.tree(agent_stats, title="Agent Statistics")
            return CommandResult.ok(data=agent_stats)

        except Exception as e:
            cli.renderer.error(f"Failed to get agent stats: {e}")
            return CommandResult.fail(str(e))

    def get_completions(self, partial: str) -> list:
        """Get subcommand completions."""
        subcommands = ["summary", "q-table", "episodes", "agents"]
        return [s for s in subcommands if s.startswith(partial)]
