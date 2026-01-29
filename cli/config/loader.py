"""
Configuration Loader
====================

Loads CLI configuration from YAML files.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import yaml

from .schema import CLIConfig
from .defaults import DEFAULT_CONFIG_YAML, DEFAULT_CONFIG_DIR, DEFAULT_CONFIG_FILE

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Configuration loader for Jotty CLI.

    Loads from:
    1. Custom path (if provided)
    2. ~/.jotty/config.yaml
    3. Default configuration
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize config loader.

        Args:
            config_path: Optional custom config file path
        """
        self.config_path = config_path
        self._config: Optional[CLIConfig] = None

    @property
    def config_dir(self) -> Path:
        """Get configuration directory."""
        return Path(os.path.expanduser(DEFAULT_CONFIG_DIR))

    @property
    def default_config_file(self) -> Path:
        """Get default config file path."""
        return self.config_dir / DEFAULT_CONFIG_FILE

    def load(self) -> CLIConfig:
        """
        Load configuration.

        Priority:
        1. Custom path
        2. ~/.jotty/config.yaml
        3. Default config

        Returns:
            CLIConfig instance
        """
        if self._config is not None:
            return self._config

        # Try custom path first
        if self.config_path:
            config_file = Path(self.config_path)
            if config_file.exists():
                self._config = self._load_from_file(config_file)
                logger.info(f"Loaded config from: {config_file}")
                return self._config
            else:
                logger.warning(f"Config file not found: {config_file}")

        # Try default path
        if self.default_config_file.exists():
            self._config = self._load_from_file(self.default_config_file)
            logger.info(f"Loaded config from: {self.default_config_file}")
            return self._config

        # Use defaults
        logger.info("Using default configuration")
        self._config = CLIConfig.default()
        return self._config

    def _load_from_file(self, path: Path) -> CLIConfig:
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML file

        Returns:
            CLIConfig instance
        """
        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f) or {}
            return CLIConfig.from_dict(data)
        except yaml.YAMLError as e:
            logger.error(f"YAML parse error in {path}: {e}")
            return CLIConfig.default()
        except Exception as e:
            logger.error(f"Error loading config from {path}: {e}")
            return CLIConfig.default()

    def save(self, config: Optional[CLIConfig] = None, path: Optional[Path] = None):
        """
        Save configuration to file.

        Args:
            config: Configuration to save (default: current config)
            path: Path to save to (default: default config file)
        """
        config = config or self._config or CLIConfig.default()
        path = path or self.default_config_file

        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(path, "w") as f:
                yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)
            logger.info(f"Saved config to: {path}")
        except Exception as e:
            logger.error(f"Error saving config to {path}: {e}")

    def ensure_config_dir(self):
        """Ensure config directory exists."""
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def create_default_config(self, force: bool = False):
        """
        Create default config file if it doesn't exist.

        Args:
            force: Overwrite existing file
        """
        if self.default_config_file.exists() and not force:
            logger.info(f"Config file already exists: {self.default_config_file}")
            return

        self.ensure_config_dir()

        with open(self.default_config_file, "w") as f:
            f.write(DEFAULT_CONFIG_YAML)

        logger.info(f"Created default config: {self.default_config_file}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key path.

        Args:
            key: Dot-separated key path (e.g., "provider.use_unified")
            default: Default value if key not found

        Returns:
            Configuration value
        """
        config = self.load()
        parts = key.split(".")

        value = config
        for part in parts:
            if hasattr(value, part):
                value = getattr(value, part)
            elif isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default

        return value

    def set(self, key: str, value: Any):
        """
        Set configuration value by key path.

        Args:
            key: Dot-separated key path
            value: Value to set
        """
        config = self.load()
        parts = key.split(".")

        # Navigate to parent
        obj = config
        for part in parts[:-1]:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                logger.warning(f"Config key not found: {key}")
                return

        # Set value
        if hasattr(obj, parts[-1]):
            setattr(obj, parts[-1], value)
            self._config = config
        else:
            logger.warning(f"Config key not found: {key}")


# Singleton loader
_loader: Optional[ConfigLoader] = None


def get_config_loader(config_path: Optional[str] = None) -> ConfigLoader:
    """Get singleton config loader."""
    global _loader
    if _loader is None or config_path is not None:
        _loader = ConfigLoader(config_path)
    return _loader
