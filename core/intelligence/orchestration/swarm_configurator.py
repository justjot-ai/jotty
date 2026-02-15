"""
SwarmConfigurator - Smart Configuration Management

Handles configuration with minimal user prompts.
Follows DRY: Reuses existing credential management and validation.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ConfigurationResult:
    """Result of configuration attempt."""

    service: str
    success: bool
    config: Dict[str, Any]
    requires_user_input: bool
    prompts: List[str] = None
    error: Optional[str] = None


class SwarmConfigurator:
    """
    Smart configuration manager for services and APIs.

    DRY Principle: Reuses existing credential storage and validation.
    """

    def __init__(self, config: Any = None) -> None:
        """
        Initialize SwarmConfigurator.

        Args:
            config: Optional SwarmConfig
        """
        self.config = config
        self._config_store: Dict[str, Dict[str, Any]] = {}
        self._config_path = Path.home() / ".jotty" / "configs.json"
        self._load_configs()

    def _load_configs(self) -> Any:
        """Load saved configurations (DRY: reuse existing storage)."""
        if self._config_path.exists():
            try:
                with open(self._config_path, "r") as f:
                    self._config_store = json.load(f)
                logger.debug(f" Loaded {len(self._config_store)} saved configurations")
            except Exception as e:
                logger.warning(f" Failed to load configs: {e}")
                self._config_store = {}
        else:
            self._config_store = {}

    def _save_configs(self) -> Any:
        """Save configurations to disk."""
        try:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._config_path, "w") as f:
                json.dump(self._config_store, f, indent=2)
            logger.debug(" Saved configurations")
        except Exception as e:
            logger.warning(f" Failed to save configs: {e}")

    async def configure(
        self,
        service: str,
        config_template: Optional[Dict[str, Any]] = None,
        auto_detect: bool = True,
    ) -> ConfigurationResult:
        """
        Configure a service with minimal prompts.

        Args:
            service: Service name (e.g., "reddit", "notion", "slack")
            config_template: Optional template for required config keys
            auto_detect: Try to auto-detect from environment variables

        Returns:
            ConfigurationResult
        """
        logger.info(f" SwarmConfigurator: Configuring '{service}'")

        # Check if already configured
        if service in self._config_store:
            logger.info(f" {service} already configured")
            return ConfigurationResult(
                service=service,
                success=True,
                config=self._config_store[service],
                requires_user_input=False,
            )

        # Try auto-detection from environment (DRY: reuse env vars)
        if auto_detect:
            auto_config = self._auto_detect_config(service)
            if auto_config:
                self._config_store[service] = auto_config
                self._save_configs()
                logger.info(f" Auto-configured {service} from environment")
                return ConfigurationResult(
                    service=service, success=True, config=auto_config, requires_user_input=False
                )

        # Determine required config keys
        required_keys = config_template or self._get_default_config_template(service)

        # Check what we have vs what we need
        missing_keys = []
        partial_config = {}

        for key in required_keys:
            # Try environment variable first (DRY: reuse env)
            env_key = f"{service.upper()}_{key.upper()}"
            import os

            env_value = os.getenv(env_key)
            if env_value:
                partial_config[key] = env_value
            else:
                missing_keys.append(key)

        # If all keys found, success
        if not missing_keys:
            self._config_store[service] = partial_config
            self._save_configs()
            return ConfigurationResult(
                service=service, success=True, config=partial_config, requires_user_input=False
            )

        # Need user input for missing keys
        prompts = [f"Please provide {key} for {service}" for key in missing_keys]

        return ConfigurationResult(
            service=service,
            success=False,
            config=partial_config,
            requires_user_input=True,
            prompts=prompts,
        )

    def _auto_detect_config(self, service: str) -> Optional[Dict[str, Any]]:
        """Auto-detect configuration from environment variables."""
        import os

        # Common environment variable patterns
        env_patterns = {
            "reddit": ["REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT"],
            "notion": ["NOTION_API_KEY", "NOTION_DATABASE_ID"],
            "slack": ["SLACK_BOT_TOKEN", "SLACK_CHANNEL"],
            "openai": ["OPENAI_API_KEY"],
            "anthropic": ["ANTHROPIC_API_KEY"],
            "morph": ["MORPH_API_KEY", "MORPH_WORKSPACE_ID"],
        }

        patterns = env_patterns.get(service.lower(), [])
        config = {}

        for pattern in patterns:
            value = os.getenv(pattern)
            if value:
                # Convert pattern to config key
                key = pattern.replace(f"{service.upper()}_", "").lower()
                config[key] = value

        # Return config if we have at least one key
        if config:
            return config

        return None

    def _get_default_config_template(self, service: str) -> List[str]:
        """Get default config template for service."""
        templates = {
            "reddit": ["client_id", "client_secret", "user_agent"],
            "notion": ["api_key", "database_id"],
            "slack": ["bot_token", "channel"],
            "openai": ["api_key"],
            "anthropic": ["api_key"],
            "github": ["token"],
            "twitter": ["api_key", "api_secret", "access_token", "access_token_secret"],
            "morph": ["api_key", "workspace_id"],  # Optional for cloud deployment
        }

        return templates.get(service.lower(), ["api_key"])

    def get_config(self, service: str) -> Optional[Dict[str, Any]]:
        """Get saved configuration for service."""
        return self._config_store.get(service)

    def set_config(self, service: str, config: Dict[str, Any]) -> None:
        """Manually set configuration."""
        self._config_store[service] = config
        self._save_configs()
        logger.info(f" Saved configuration for {service}")
