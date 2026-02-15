"""
Agents Command
==============

List and manage agents.
"""

from typing import TYPE_CHECKING, List, Dict, Any
from .base import BaseCommand, CommandResult, ParsedArgs

if TYPE_CHECKING:
    from ..app import JottyCLI


class AgentsCommand(BaseCommand):
    """List and manage agents."""

    name = "agents"
    aliases = ["a"]
    description = "List available agents, their specializations and status"
    usage = "/agents [list|info <name>|specializations]"
    category = "swarm"

    async def execute(self, args: ParsedArgs, cli: "JottyCLI") -> CommandResult:
        """Execute agents command."""
        subcommand = args.positional[0] if args.positional else "list"

        if subcommand == "list":
            return await self._list_agents(cli)
        elif subcommand == "info" and len(args.positional) > 1:
            return await self._agent_info(args.positional[1], cli)
        elif subcommand == "specializations":
            return await self._show_specializations(cli)
        else:
            return await self._list_agents(cli)

    async def _list_agents(self, cli: "JottyCLI") -> CommandResult:
        """List all agents."""
        try:
            swarm = await cli.get_swarm_manager()

            # Get agent information
            agents_data: List[Dict[str, Any]] = []

            for agent_config in swarm.agents:
                agent_info = {
                    "name": agent_config.name,
                    "type": type(agent_config.agent).__name__,
                    "specialization": "general",
                    "success_rate": 0.0,
                }

                # Get specialization from swarm intelligence
                if hasattr(swarm, "swarm_intelligence"):
                    profile = swarm.swarm_intelligence.agent_profiles.get(agent_config.name)
                    if profile:
                        spec = getattr(profile, 'specialization', None)
                        agent_info["specialization"] = spec.value if hasattr(spec, 'value') else (spec or "general")
                        # Calculate success rate from task_success dict
                        if hasattr(profile, 'task_success') and profile.task_success:
                            total_success = sum(s for s, t in profile.task_success.values())
                            total_tasks = sum(t for s, t in profile.task_success.values())
                            if total_tasks > 0:
                                agent_info["success_rate"] = total_success / total_tasks
                        elif hasattr(profile, 'total_tasks') and profile.total_tasks > 0:
                            agent_info["success_rate"] = getattr(profile, 'trust_score', 0.5)

                agents_data.append(agent_info)

            # Render table
            table = cli.renderer.tables.agents_table(agents_data)
            cli.renderer.tables.print_table(table)

            cli.renderer.info(f"Total: {len(agents_data)} agents")
            return CommandResult.ok(data=agents_data)

        except Exception as e:
            cli.renderer.error(f"Failed to list agents: {e}")
            return CommandResult.fail(str(e))

    async def _agent_info(self, name: str, cli: "JottyCLI") -> CommandResult:
        """Show detailed agent info."""
        try:
            swarm = await cli.get_swarm_manager()

            # Find agent
            agent_config = None
            for ac in swarm.agents:
                if ac.name == name:
                    agent_config = ac
                    break

            if not agent_config:
                cli.renderer.error(f"Agent not found: {name}")
                return CommandResult.fail(f"Agent not found: {name}")

            # Build info
            info = {
                "Name": agent_config.name,
                "Type": type(agent_config.agent).__name__,
                "Is Executor": agent_config.is_executor,
                "Is Critical": agent_config.is_critical,
                "Enable Architect": agent_config.enable_architect,
                "Enable Auditor": agent_config.enable_auditor,
            }

            # Get learning stats
            if hasattr(swarm, "swarm_intelligence"):
                profile = swarm.swarm_intelligence.agent_profiles.get(name)
                if profile:
                    info["Total Tasks"] = getattr(profile, 'total_tasks', 0)
                    # Calculate successful tasks from task_success dict
                    if hasattr(profile, 'task_success') and profile.task_success:
                        info["Successful Tasks"] = sum(s for s, t in profile.task_success.values())
                    info["Trust Score"] = f"{getattr(profile, 'trust_score', 0.5):.2f}"
                    spec = getattr(profile, 'specialization', None)
                    info["Specialization"] = spec.value if hasattr(spec, 'value') else (spec or "general")
                    info["Avg Execution Time"] = f"{getattr(profile, 'avg_execution_time', 0.0):.2f}s"

            # Get capabilities
            if agent_config.capabilities:
                info["Capabilities"] = ", ".join(agent_config.capabilities)

            # Display as tree
            cli.renderer.tree(info, title=f"Agent: {name}")
            return CommandResult.ok(data=info)

        except Exception as e:
            cli.renderer.error(f"Failed to get agent info: {e}")
            return CommandResult.fail(str(e))

    async def _show_specializations(self, cli: "JottyCLI") -> CommandResult:
        """Show agent specializations."""
        try:
            swarm = await cli.get_swarm_manager()

            if not hasattr(swarm, "swarm_intelligence"):
                cli.renderer.warning("Swarm intelligence not available")
                return CommandResult.fail("Swarm intelligence not available")

            specs = swarm.get_agent_specializations()

            if not specs:
                cli.renderer.info("No specializations learned yet")
                return CommandResult.ok(data={})

            cli.renderer.panel(
                "\n".join([f"• {agent}: {spec}" for agent, spec in specs.items()]),
                title="Agent Specializations",
                style="magenta"
            )

            return CommandResult.ok(data=specs)

        except Exception as e:
            cli.renderer.error(f"Failed to get specializations: {e}")
            return CommandResult.fail(str(e))

    def get_completions(self, partial: str) -> list:
        """Get subcommand completions."""
        subcommands = ["list", "info", "specializations"]
        return [s for s in subcommands if s.startswith(partial)]
