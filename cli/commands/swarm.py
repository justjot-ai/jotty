"""
Swarm Command
=============

Swarm intelligence status and management.
"""

from typing import TYPE_CHECKING, Dict, Any
from .base import BaseCommand, CommandResult, ParsedArgs

if TYPE_CHECKING:
    from ..app import JottyCLI


class SwarmCommand(BaseCommand):
    """Swarm intelligence status and management."""

    name = "swarm"
    aliases = ["sw"]
    description = "View swarm intelligence status, consensus, and routing"
    usage = "/swarm [status|routing|consensus|providers]"
    category = "swarm"

    async def execute(self, args: ParsedArgs, cli: "JottyCLI") -> CommandResult:
        """Execute swarm command."""
        subcommand = args.positional[0] if args.positional else "status"

        if subcommand == "status":
            return await self._show_status(cli)
        elif subcommand == "routing":
            return await self._show_routing(cli)
        elif subcommand == "consensus":
            return await self._show_consensus(cli)
        elif subcommand == "providers":
            return await self._show_providers(cli)
        else:
            return await self._show_status(cli)

    async def _show_status(self, cli: "JottyCLI") -> CommandResult:
        """Show swarm intelligence status."""
        try:
            swarm = await cli.get_swarm_manager()

            status: Dict[str, Any] = {
                "Mode": swarm.mode,
                "Agents": len(swarm.agents),
                "Episodes": swarm.episode_count,
            }

            # Swarm intelligence stats
            if hasattr(swarm, "swarm_intelligence"):
                si = swarm.swarm_intelligence
                profiles = si.agent_profiles

                status["Agent Profiles"] = len(profiles)

                # Aggregate stats
                total_tasks = sum(p.total_tasks for p in profiles.values())
                # Calculate successes from task_success dict
                total_success = sum(
                    sum(s for s, t in p.task_success.values())
                    for p in profiles.values()
                )

                status["Total Tasks"] = total_tasks
                status["Success Rate"] = f"{total_success/max(1, total_tasks):.1%}"

                # Specializations
                specs = {p.agent_name: str(p.specialization) for p in profiles.values() if p.specialization}
                if specs:
                    status["Specializations"] = specs

            # Learning stats
            if hasattr(swarm, "learning_manager"):
                q_summary = swarm.learning_manager.get_q_table_summary()
                status["Q-Table Size"] = q_summary.get("size", 0)

            # Provider status
            if hasattr(swarm, "provider_registry") and swarm.provider_registry:
                summary = swarm.get_provider_summary()
                status["Providers"] = summary.get("available", False)

            cli.renderer.tree(status, title="Swarm Intelligence Status")
            return CommandResult.ok(data=status)

        except Exception as e:
            cli.renderer.error(f"Failed to get swarm status: {e}")
            return CommandResult.fail(str(e))

    async def _show_routing(self, cli: "JottyCLI") -> CommandResult:
        """Show task routing information."""
        try:
            swarm = await cli.get_swarm_manager()

            if not hasattr(swarm, "swarm_intelligence"):
                cli.renderer.warning("Swarm intelligence not available")
                return CommandResult.fail("Swarm intelligence not available")

            si = swarm.swarm_intelligence
            profiles = si.agent_profiles

            routing_info = {}

            for agent_name, profile in profiles.items():
                task_success = {}
                for task_type, (success, total) in profile.task_success.items():
                    if total > 0:
                        task_success[task_type] = f"{success}/{total} ({success/total:.0%})"

                if task_success:
                    routing_info[agent_name] = task_success

            if not routing_info:
                cli.renderer.info("No routing data yet. Run some tasks first.")
                return CommandResult.ok(data={})

            cli.renderer.tree(routing_info, title="Task Routing (Agent -> Task Type Success)")
            return CommandResult.ok(data=routing_info)

        except Exception as e:
            cli.renderer.error(f"Failed to get routing info: {e}")
            return CommandResult.fail(str(e))

    async def _show_consensus(self, cli: "JottyCLI") -> CommandResult:
        """Show consensus mechanism info."""
        try:
            swarm = await cli.get_swarm_manager()

            if not hasattr(swarm, "swarm_intelligence"):
                cli.renderer.warning("Swarm intelligence not available")
                return CommandResult.fail("Swarm intelligence not available")

            si = swarm.swarm_intelligence

            consensus_info = {
                "Consensus Cache Size": len(si.consensus_cache),
                "Recent Decisions": [],
            }

            # Get recent consensus decisions from cache
            for key, decision in list(si.consensus_cache.items())[:5]:
                consensus_info["Recent Decisions"].append({
                    "Query": key[:50] + "..." if len(key) > 50 else key,
                    "Decision": decision.get("decision", "unknown"),
                    "Confidence": f"{decision.get('confidence', 0):.2f}",
                })

            cli.renderer.tree(consensus_info, title="Swarm Consensus")
            return CommandResult.ok(data=consensus_info)

        except Exception as e:
            cli.renderer.error(f"Failed to get consensus info: {e}")
            return CommandResult.fail(str(e))

    async def _show_providers(self, cli: "JottyCLI") -> CommandResult:
        """Show provider registry info."""
        try:
            swarm = await cli.get_swarm_manager()

            if not hasattr(swarm, "provider_registry") or not swarm.provider_registry:
                cli.renderer.warning("Provider registry not available")
                return CommandResult.fail("Provider registry not available")

            summary = swarm.get_provider_summary()

            if not summary.get("available"):
                cli.renderer.warning("Providers not configured")
                return CommandResult.fail("Providers not configured")

            cli.renderer.tree(summary, title="Provider Registry")
            return CommandResult.ok(data=summary)

        except Exception as e:
            cli.renderer.error(f"Failed to get provider info: {e}")
            return CommandResult.fail(str(e))

    def get_completions(self, partial: str) -> list:
        """Get subcommand completions."""
        subcommands = ["status", "routing", "consensus", "providers"]
        return [s for s in subcommands if s.startswith(partial)]
