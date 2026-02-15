"""
Memory Command
==============

Inspect hierarchical memory.
"""

from typing import TYPE_CHECKING

from .base import BaseCommand, CommandResult, ParsedArgs

if TYPE_CHECKING:
    from ..app import JottyCLI


class MemoryCommand(BaseCommand):
    """Inspect hierarchical memory."""

    name = "memory"
    aliases = ["m"]
    description = "Inspect and query hierarchical memory system"
    usage = "/memory [status|query <text>|levels|clear]"
    category = "memory"

    async def execute(self, args: ParsedArgs, cli: "JottyCLI") -> CommandResult:
        """Execute memory command."""
        subcommand = args.positional[0] if args.positional else "status"

        if subcommand == "status":
            return await self._show_status(cli)
        elif subcommand == "query" and len(args.positional) > 1:
            query = " ".join(args.positional[1:])
            return await self._query_memory(query, cli)
        elif subcommand == "levels":
            return await self._show_levels(cli)
        elif subcommand == "clear":
            confirm = args.flags.get("confirm", False) or args.flags.get("y", False)
            return await self._clear_memory(confirm, cli)
        else:
            return await self._show_status(cli)

    async def _show_status(self, cli: "JottyCLI") -> CommandResult:
        """Show memory status."""
        try:
            swarm = await cli.get_swarm_manager()
            memory = swarm.swarm_memory

            status = {
                "Agent Name": memory.agent_name,
                "Total Entries": len(memory.memories),
            }

            # Count by level
            level_counts = {}
            for mem in memory.memories:
                level = str(mem.level.name) if hasattr(mem.level, "name") else str(mem.level)
                level_counts[level] = level_counts.get(level, 0) + 1

            status["By Level"] = level_counts

            # Recent entries
            recent = memory.memories[-5:] if memory.memories else []
            status["Recent Entries"] = len(recent)

            cli.renderer.tree(status, title="Memory Status")
            return CommandResult.ok(data=status)

        except Exception as e:
            cli.renderer.error(f"Failed to get memory status: {e}")
            return CommandResult.fail(str(e))

    async def _query_memory(self, query: str, cli: "JottyCLI") -> CommandResult:
        """Query memory for relevant entries."""
        try:
            swarm = await cli.get_swarm_manager()
            memory = swarm.swarm_memory

            cli.renderer.info(f"Querying memory: {query}")

            # Retrieve relevant memories
            results = await memory.retrieve(query=query, top_k=10, context={"goal": query})

            if not results:
                cli.renderer.warning("No relevant memories found")
                return CommandResult.ok(data=[])

            # Format results
            memory_data = []
            for mem in results:
                memory_data.append(
                    {
                        "level": (
                            str(mem.level.name) if hasattr(mem.level, "name") else str(mem.level)
                        ),
                        "content": str(mem.content)[:100],
                        "relevance_score": getattr(mem, "relevance_score", 0.0),
                    }
                )

            table = cli.renderer.tables.memory_table(memory_data)
            cli.renderer.tables.print_table(table)

            return CommandResult.ok(data=memory_data)

        except Exception as e:
            cli.renderer.error(f"Memory query failed: {e}")
            return CommandResult.fail(str(e))

    async def _show_levels(self, cli: "JottyCLI") -> CommandResult:
        """Show memory level descriptions."""
        try:
            # Memory levels (SDK uses string levels)
            memory_levels = ["EPISODIC", "SEMANTIC", "PROCEDURAL", "META", "CAUSAL"]

            levels_info = {}
            for level_name in memory_levels:
                levels_info[level_name] = {
                    "Value": level_name.lower(),
                    "Description": self._get_level_description(level_name),
                }

            cli.renderer.tree(levels_info, title="Memory Levels")
            return CommandResult.ok(data=levels_info)

        except Exception as e:
            cli.renderer.error(f"Failed to get memory levels: {e}")
            return CommandResult.fail(str(e))

    def _get_level_description(self, level_name: str) -> str:
        """Get description for memory level."""
        descriptions = {
            "EPISODIC": "Raw experiences with fast decay",
            "SEMANTIC": "Abstracted patterns with slow decay",
            "PROCEDURAL": "Action sequences with medium decay",
            "META": "Learning wisdom with no decay",
            "CAUSAL": "Why things work with no decay",
        }
        return descriptions.get(level_name, "Unknown level")

    async def _clear_memory(self, confirm: bool, cli: "JottyCLI") -> CommandResult:
        """Clear memory."""
        if not confirm:
            cli.renderer.warning("This will clear all memory. Use --confirm or -y to proceed.")
            return CommandResult.fail("Confirmation required")

        try:
            swarm = await cli.get_swarm_manager()
            memory = swarm.swarm_memory

            count = len(memory.memories)
            memory.memories.clear()

            cli.renderer.success(f"Cleared {count} memory entries")
            return CommandResult.ok()

        except Exception as e:
            cli.renderer.error(f"Failed to clear memory: {e}")
            return CommandResult.fail(str(e))

    def get_completions(self, partial: str) -> list:
        """Get subcommand completions."""
        subcommands = ["status", "query", "levels", "clear"]
        return [s for s in subcommands if s.startswith(partial)]
