"""
Plugin Loader
=============

Discovers and loads CLI plugins.
"""

import importlib.util
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .base import PluginBase, PluginInfo, SkillPlugin

if TYPE_CHECKING:
    from ..app import JottyCLI

logger = logging.getLogger(__name__)


class PluginLoader:
    """
    Plugin discovery and loading.

    Discovers plugins from:
    - ~/.jotty/plugins/
    - Built-in skill plugins
    """

    def __init__(self, plugin_dir: Optional[str] = None) -> None:
        """
        Initialize plugin loader.

        Args:
            plugin_dir: Custom plugin directory
        """
        self.plugin_dir = Path(plugin_dir or "~/.jotty/plugins").expanduser()
        self.loaded_plugins: Dict[str, PluginBase] = {}

    def discover(self) -> List[PluginInfo]:
        """
        Discover available plugins.

        Returns:
            List of PluginInfo for available plugins
        """
        plugins = []

        # Check plugin directory
        if self.plugin_dir.exists():
            for path in self.plugin_dir.glob("*.py"):
                try:
                    info = self._get_plugin_info(path)
                    if info:
                        plugins.append(info)
                except Exception as e:
                    logger.warning(f"Failed to load plugin {path}: {e}")

        return plugins

    def _get_plugin_info(self, path: Path) -> Optional[PluginInfo]:
        """Get plugin info from file without fully loading."""
        try:
            spec = importlib.util.spec_from_file_location(path.stem, path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Look for plugin class
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, PluginBase)
                        and attr is not PluginBase
                    ):
                        instance = attr()
                        return instance.info

        except Exception as e:
            logger.debug(f"Could not get plugin info from {path}: {e}")

        return None

    def load(self, name: str, cli: "JottyCLI") -> Optional[PluginBase]:
        """
        Load a plugin by name.

        Args:
            name: Plugin name
            cli: JottyCLI instance

        Returns:
            Loaded plugin or None
        """
        if name in self.loaded_plugins:
            return self.loaded_plugins[name]

        # Try file-based plugin
        plugin_file = self.plugin_dir / f"{name}.py"
        if plugin_file.exists():
            plugin = self._load_from_file(plugin_file, cli)
            if plugin:
                self.loaded_plugins[name] = plugin
                return plugin

        return None

    def _load_from_file(self, path: Path, cli: "JottyCLI") -> Optional[PluginBase]:
        """Load plugin from Python file."""
        try:
            spec = importlib.util.spec_from_file_location(path.stem, path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Find and instantiate plugin class
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, PluginBase)
                        and attr is not PluginBase
                    ):
                        plugin = attr()
                        plugin.on_load(cli)

                        # Register commands
                        for cmd in plugin.get_commands():
                            cli.command_registry.register(cmd)

                        logger.info(f"Loaded plugin: {plugin.info.name}")
                        return plugin

        except Exception as e:
            logger.error(f"Failed to load plugin from {path}: {e}")

        return None

    def load_skill_plugins(self, cli: "JottyCLI") -> Any:
        """
        Load all skills as plugins.

        Args:
            cli: JottyCLI instance
        """
        try:
            registry = cli.get_skills_registry()

            if not registry.initialized:
                registry.init()

            # Create plugins from skills
            for skill in registry.loaded_skills.values():
                try:
                    plugin = SkillPlugin(skill)
                    plugin.on_load(cli)

                    # Register commands
                    for cmd in plugin.get_commands():
                        cli.command_registry.register(cmd)

                    self.loaded_plugins[plugin.info.name] = plugin

                except Exception as e:
                    logger.debug(f"Could not create plugin for skill {skill.name}: {e}")

            logger.info(f"Loaded {len(self.loaded_plugins)} skill plugins")

        except Exception as e:
            logger.warning(f"Failed to load skill plugins: {e}")

    def unload(self, name: str, cli: "JottyCLI") -> Any:
        """
        Unload a plugin.

        Args:
            name: Plugin name
            cli: JottyCLI instance
        """
        if name in self.loaded_plugins:
            plugin = self.loaded_plugins[name]
            plugin.on_unload(cli)
            del self.loaded_plugins[name]
            logger.info(f"Unloaded plugin: {name}")

    def list_loaded(self) -> List[PluginInfo]:
        """List loaded plugins."""
        return [p.info for p in self.loaded_plugins.values()]
