"""
Tools Command
=============

Execute skills directly as tools.
"""

from typing import TYPE_CHECKING, Dict, Any
from .base import BaseCommand, CommandResult, ParsedArgs

if TYPE_CHECKING:
    from ..app import JottyCLI


class ToolsCommand(BaseCommand):
    """Execute skills directly as tools."""

    name = "tools"
    aliases = ["t", "tool"]
    description = "List and execute Jotty skills as tools"
    usage = "/tools [list|<tool-name> <args>]"
    category = "tools"

    async def execute(self, args: ParsedArgs, cli: "JottyCLI") -> CommandResult:
        """Execute tools command."""
        if not args.positional:
            return await self._list_tools(cli)

        subcommand = args.positional[0]

        if subcommand == "list":
            category = args.flags.get("category") or args.flags.get("c")
            return await self._list_tools(cli, category)
        else:
            # Execute tool
            tool_name = subcommand
            tool_args = " ".join(args.positional[1:]) if len(args.positional) > 1 else ""
            return await self._execute_tool(tool_name, tool_args, args.flags, cli)

    async def _list_tools(self, cli: "JottyCLI", category: str = None) -> CommandResult:
        """List available tools."""
        try:
            registry = cli.get_skills_registry()

            if not registry.initialized:
                with cli.renderer.progress.spinner("Loading skills..."):
                    registry.init()

            # Get all tools from skills
            tools = registry.get_registered_tools()

            if category:
                # Filter by category
                tools = {k: v for k, v in tools.items() if category.lower() in k.lower()}

            if not tools:
                cli.renderer.info("No tools found")
                return CommandResult.ok(data=[])

            # Format as table
            tool_list = [
                {
                    "name": name,
                    "description": f"Tool: {name}",
                    "tools": [name],
                }
                for name in sorted(tools.keys())[:50]
            ]

            table = cli.renderer.tables.skills_table(tool_list)
            cli.renderer.tables.print_table(table)

            cli.renderer.info(f"Total: {len(tools)} tools. Use /tools <name> <args> to execute.")
            return CommandResult.ok(data=list(tools.keys()))

        except Exception as e:
            cli.renderer.error(f"Failed to list tools: {e}")
            return CommandResult.fail(str(e))

    async def _execute_tool(
        self,
        tool_name: str,
        tool_args: str,
        flags: Dict[str, Any],
        cli: "JottyCLI"
    ) -> CommandResult:
        """Execute a specific tool."""
        try:
            registry = cli.get_skills_registry()

            if not registry.initialized:
                registry.init()

            tools = registry.get_registered_tools()

            # Find tool
            if tool_name not in tools:
                # Try partial match
                matches = [t for t in tools.keys() if tool_name.lower() in t.lower()]
                if len(matches) == 1:
                    tool_name = matches[0]
                elif len(matches) > 1:
                    cli.renderer.warning(f"Multiple tools match '{tool_name}': {matches[:5]}")
                    return CommandResult.fail(f"Ambiguous tool name: {tool_name}")
                else:
                    cli.renderer.error(f"Tool not found: {tool_name}")
                    return CommandResult.fail(f"Tool not found: {tool_name}")

            tool = tools[tool_name]

            cli.renderer.info(f"Executing: {tool_name}")

            # Build params from args and flags
            params = {
                "query": tool_args,
                "input": tool_args,
                **flags
            }

            # Execute tool
            async with await cli.renderer.progress.spinner_async(
                f"Running {tool_name}...",
                style="cyan"
            ):
                if callable(tool):
                    result = await tool(params)
                elif hasattr(tool, "execute"):
                    result = await tool.execute(params)
                else:
                    result = {"error": "Tool is not executable"}

            # Display result
            if isinstance(result, dict):
                success = result.get("success", True)
                if success:
                    cli.renderer.success(f"Tool {tool_name} completed")
                    cli.renderer.result(result, title=f"Tool: {tool_name}")
                else:
                    cli.renderer.error(f"Tool failed: {result.get('error', 'Unknown error')}")
                return CommandResult.ok(data=result) if success else CommandResult.fail(result.get("error", "Failed"))
            else:
                cli.renderer.success(f"Tool {tool_name} completed")
                cli.renderer.panel(str(result)[:500], title=f"Output: {tool_name}")
                return CommandResult.ok(data={"output": result})

        except Exception as e:
            cli.renderer.error(f"Tool execution failed: {e}")
            if cli.config.debug:
                import traceback
                traceback.print_exc()
            return CommandResult.fail(str(e))

    def get_completions(self, partial: str) -> list:
        """Get tool completions."""
        # Common tool operations
        common = ["list", "web-search", "shell-exec", "file-read", "file-write"]
        return [c for c in common if c.startswith(partial)]
