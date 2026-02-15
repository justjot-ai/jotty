"""
Plugin Base Class
=================

Base class for CLI plugins.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, List, Dict, Any

if TYPE_CHECKING:
    from ..app import JottyCLI
    from ..commands.base import BaseCommand


@dataclass
class PluginInfo:
    """Plugin metadata."""
    name: str
    version: str
    description: str
    author: Optional[str] = None
    commands: Optional[List[str]] = None
    requires: Optional[List[str]] = None


class PluginBase(ABC):
    """
    Base class for CLI plugins.

    Plugins can:
    - Register new commands
    - Hook into CLI lifecycle
    - Extend functionality
    """

    @property
    @abstractmethod
    def info(self) -> PluginInfo:
        """Get plugin information."""
        pass

    def on_load(self, cli: 'JottyCLI') -> Any:
        """
        Called when plugin is loaded.

        Args:
            cli: JottyCLI instance
        """
        pass

    def on_unload(self, cli: 'JottyCLI') -> Any:
        """
        Called when plugin is unloaded.

        Args:
            cli: JottyCLI instance
        """
        pass

    def get_commands(self) -> List["BaseCommand"]:
        """
        Get commands provided by this plugin.

        Returns:
            List of BaseCommand instances
        """
        return []

    def on_input(self, text: str, cli: "JottyCLI") -> Optional[str]:
        """
        Hook for processing input before handling.

        Args:
            text: User input
            cli: JottyCLI instance

        Returns:
            Modified input or None to use original
        """
        return None

    def on_output(self, result: Any, cli: "JottyCLI") -> Optional[Any]:
        """
        Hook for processing output before display.

        Args:
            result: Command result
            cli: JottyCLI instance

        Returns:
            Modified result or None to use original
        """
        return None


class SkillPlugin(PluginBase):
    """
    Plugin generated from a Jotty skill.

    Auto-exposes skill as CLI tool.
    """

    def __init__(self, skill_definition: Any) -> None:
        """
        Initialize from skill definition.

        Args:
            skill_definition: SkillDefinition instance
        """
        self.skill = skill_definition

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name=f"skill-{self.skill.name}",
            version="1.0.0",
            description=self.skill.description,
            commands=[self.skill.name],
        )

    def get_commands(self) -> List["BaseCommand"]:
        """Create command from skill."""
        from ..commands.base import BaseCommand, CommandResult, ParsedArgs

        skill = self.skill

        class SkillCommand(BaseCommand):
            name = skill.name
            aliases = []
            description = skill.description
            usage = f"/{skill.name} <args>"
            category = "skills"

            async def execute(self, args: ParsedArgs, cli: "JottyCLI") -> CommandResult:
                params = {
                    "query": args.raw,
                    "input": args.raw,
                    **args.flags
                }

                # Get first tool from skill
                tools = skill.tools
                if not tools:
                    return CommandResult.fail("Skill has no tools")

                tool = list(tools.values())[0]

                try:
                    if callable(tool):
                        result = await tool(params)
                    elif hasattr(tool, "execute"):
                        result = await tool.execute(params)
                    else:
                        return CommandResult.fail("Tool is not executable")

                    return CommandResult.ok(data=result)

                except Exception as e:
                    return CommandResult.fail(str(e))

        return [SkillCommand()]
