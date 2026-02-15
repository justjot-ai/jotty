"""
Skills Command
==============

List and search skills.
"""

from typing import TYPE_CHECKING, Any, Dict, List

from .base import BaseCommand, CommandResult, ParsedArgs

if TYPE_CHECKING:
    from ..app import JottyCLI


class SkillsCommand(BaseCommand):
    """List and search skills."""

    name = "skills"
    aliases = ["s"]
    description = "List, search, and get info about available skills"
    usage = "/skills [list|search <query>|info <name>|categories] [--category <cat>]"
    category = "skills"

    async def execute(self, args: ParsedArgs, cli: "JottyCLI") -> CommandResult:
        """Execute skills command."""
        subcommand = args.positional[0] if args.positional else "list"

        if subcommand == "list":
            category = args.flags.get("category") or args.flags.get("c")
            return await self._list_skills(cli, category=category)
        elif subcommand == "search" and len(args.positional) > 1:
            query = " ".join(args.positional[1:])
            return await self._search_skills(query, cli)
        elif subcommand == "info" and len(args.positional) > 1:
            return await self._skill_info(args.positional[1], cli)
        elif subcommand == "categories":
            return await self._list_categories(cli)
        else:
            return await self._list_skills(cli)

    async def _list_skills(
        self, cli: "JottyCLI", category: str = None, limit: int = 50
    ) -> CommandResult:
        """List all skills."""
        try:
            registry = cli.get_skills_registry()

            # Initialize if needed
            if not registry.initialized:
                with cli.renderer.progress.spinner("Loading skills..."):
                    registry.init()

            # Get skills
            skills = registry.list_skills()

            # Filter by category if specified
            if category:
                skills = [s for s in skills if category.lower() in s.get("name", "").lower()]

            # Limit display
            total = len(skills)
            skills = skills[:limit]

            # Render table
            table = cli.renderer.tables.skills_table(skills)
            cli.renderer.tables.print_table(table)

            if total > limit:
                cli.renderer.info(f"Showing {limit} of {total} skills. Use --category to filter.")
            else:
                cli.renderer.info(f"Total: {total} skills")

            return CommandResult.ok(data=skills)

        except Exception as e:
            cli.renderer.error(f"Failed to list skills: {e}")
            return CommandResult.fail(str(e))

    async def _search_skills(self, query: str, cli: "JottyCLI") -> CommandResult:
        """Search skills by query."""
        try:
            registry = cli.get_skills_registry()

            if not registry.initialized:
                registry.init()

            skills = registry.list_skills()

            # Search in name and description
            query_lower = query.lower()
            matches = []

            for skill in skills:
                name = skill.get("name", "").lower()
                desc = skill.get("description", "").lower()
                tools = [t.lower() for t in skill.get("tools", [])]

                if (
                    query_lower in name
                    or query_lower in desc
                    or any(query_lower in t for t in tools)
                ):
                    matches.append(skill)

            if not matches:
                cli.renderer.warning(f"No skills found matching: {query}")
                return CommandResult.ok(data=[])

            table = cli.renderer.tables.skills_table(matches)
            cli.renderer.tables.print_table(table)

            cli.renderer.info(f"Found {len(matches)} matching skills")
            return CommandResult.ok(data=matches)

        except Exception as e:
            cli.renderer.error(f"Search failed: {e}")
            return CommandResult.fail(str(e))

    async def _skill_info(self, name: str, cli: "JottyCLI") -> CommandResult:
        """Show detailed skill info."""
        try:
            registry = cli.get_skills_registry()

            if not registry.initialized:
                registry.init()

            skill = registry.get_skill(name)

            if not skill:
                cli.renderer.error(f"Skill not found: {name}")
                return CommandResult.fail(f"Skill not found: {name}")

            # Build info
            info = {
                "Name": skill.name,
                "Description": skill.description,
                "Tools": list(skill.tools.keys()),
                "Metadata": skill.metadata,
            }

            cli.renderer.tree(info, title=f"Skill: {name}")
            return CommandResult.ok(data=info)

        except Exception as e:
            cli.renderer.error(f"Failed to get skill info: {e}")
            return CommandResult.fail(str(e))

    async def _list_categories(self, cli: "JottyCLI") -> CommandResult:
        """List skill categories."""
        try:
            registry = cli.get_skills_registry()

            if not registry.initialized:
                registry.init()

            # Extract categories from skill names
            categories: Dict[str, int] = {}
            skills = registry.list_skills()

            for skill in skills:
                name = skill.get("name", "")
                # Try to extract category from name (e.g., "web-search" -> "web")
                if "-" in name:
                    cat = name.split("-")[0]
                else:
                    cat = "general"

                categories[cat] = categories.get(cat, 0) + 1

            # Sort by count
            sorted_cats = sorted(categories.items(), key=lambda x: x[1], reverse=True)

            cli.renderer.panel(
                "\n".join([f"• {cat}: {count} skills" for cat, count in sorted_cats]),
                title="Skill Categories",
                style="blue",
            )

            return CommandResult.ok(data=dict(sorted_cats))

        except Exception as e:
            cli.renderer.error(f"Failed to list categories: {e}")
            return CommandResult.fail(str(e))

    def get_completions(self, partial: str) -> list:
        """Get subcommand completions."""
        subcommands = ["list", "search", "info", "categories"]
        return [s for s in subcommands if s.startswith(partial)]
