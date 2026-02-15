"""
Config Command
==============

Configuration management.
"""

from typing import TYPE_CHECKING, Any

from .base import BaseCommand, CommandResult, ParsedArgs

if TYPE_CHECKING:
    from ..app import JottyCLI


class ConfigCommand(BaseCommand):
    """Configuration management."""

    name = "config"
    aliases = ["c"]
    description = "View and modify CLI configuration"
    usage = "/config [show|get <key>|set <key> <value>|reset|path]"
    category = "system"

    async def execute(self, args: ParsedArgs, cli: "JottyCLI") -> CommandResult:
        """Execute config command."""
        subcommand = args.positional[0] if args.positional else "show"

        if subcommand == "show":
            return await self._show_config(cli)
        elif subcommand == "get" and len(args.positional) > 1:
            return await self._get_config(args.positional[1], cli)
        elif subcommand == "set" and len(args.positional) > 2:
            return await self._set_config(args.positional[1], args.positional[2], cli)
        elif subcommand == "reset":
            return await self._reset_config(cli)
        elif subcommand == "path":
            return await self._show_path(cli)
        else:
            return await self._show_config(cli)

    async def _show_config(self, cli: "JottyCLI") -> CommandResult:
        """Show current configuration."""
        try:
            config_dict = cli.config.to_dict()

            # Filter out empty/None values for cleaner display
            def clean_dict(d: Any) -> Any:
                if isinstance(d, dict):
                    return {
                        k: clean_dict(v)
                        for k, v in d.items()
                        if v is not None and v != "" and v != []
                    }
                return d

            clean_config = clean_dict(config_dict)

            cli.renderer.tree(clean_config, title="Current Configuration")
            return CommandResult.ok(data=config_dict)

        except Exception as e:
            cli.renderer.error(f"Failed to show config: {e}")
            return CommandResult.fail(str(e))

    async def _get_config(self, key: str, cli: "JottyCLI") -> CommandResult:
        """Get specific config value."""
        try:
            value = cli.config_loader.get(key)

            if value is None:
                cli.renderer.warning(f"Config key not found: {key}")
                return CommandResult.fail(f"Key not found: {key}")

            cli.renderer.info(f"{key} = {value}")
            return CommandResult.ok(data={"key": key, "value": value})

        except Exception as e:
            cli.renderer.error(f"Failed to get config: {e}")
            return CommandResult.fail(str(e))

    async def _set_config(self, key: str, value: str, cli: "JottyCLI") -> CommandResult:
        """Set config value."""
        try:
            # Parse value
            if value.lower() == "true":
                parsed_value = True
            elif value.lower() == "false":
                parsed_value = False
            elif value.isdigit():
                parsed_value = int(value)
            else:
                try:
                    parsed_value = float(value)
                except ValueError:
                    parsed_value = value

            cli.config_loader.set(key, parsed_value)
            cli.config_loader.save()

            cli.renderer.success(f"Set {key} = {parsed_value}")
            return CommandResult.ok()

        except Exception as e:
            cli.renderer.error(f"Failed to set config: {e}")
            return CommandResult.fail(str(e))

    async def _reset_config(self, cli: "JottyCLI") -> CommandResult:
        """Reset to default configuration."""
        try:
            cli.config_loader.create_default_config(force=True)
            cli.renderer.success("Configuration reset to defaults")
            return CommandResult.ok()

        except Exception as e:
            cli.renderer.error(f"Failed to reset config: {e}")
            return CommandResult.fail(str(e))

    async def _show_path(self, cli: "JottyCLI") -> CommandResult:
        """Show config file path."""
        try:
            path = cli.config_loader.default_config_file
            exists = path.exists()

            cli.renderer.info(f"Config path: {path}")
            cli.renderer.info(f"Exists: {exists}")

            return CommandResult.ok(data={"path": str(path), "exists": exists})

        except Exception as e:
            cli.renderer.error(f"Failed to get config path: {e}")
            return CommandResult.fail(str(e))

    def get_completions(self, partial: str) -> list:
        """Get subcommand completions."""
        subcommands = ["show", "get", "set", "reset", "path"]
        return [s for s in subcommands if s.startswith(partial)]
