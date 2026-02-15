"""
WhatsApp Providers
==================

Dual-provider WhatsApp support following OpenClaw patterns.
- Baileys: Open-source, free, like OpenClaw
- Business API: Official Meta WhatsApp Business API
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class WhatsAppConfig:
    """WhatsApp skill configuration."""

    # Baileys config (open-source, like OpenClaw)
    baileys_session_path: Optional[str] = None
    baileys_host: Optional[str] = None  # If using baileys-api server
    baileys_port: int = 3000

    # Business API config
    business_phone_id: Optional[str] = None
    business_token: Optional[str] = None

    # Provider selection
    default_provider: str = "auto"  # "baileys", "business", "auto"

    def __post_init__(self):
        """Load from environment if not set."""
        self.baileys_session_path = self.baileys_session_path or os.getenv("BAILEYS_SESSION_PATH")
        self.baileys_host = self.baileys_host or os.getenv("BAILEYS_HOST")
        self.business_phone_id = self.business_phone_id or os.getenv("WHATSAPP_PHONE_ID")
        self.business_token = (
            self.business_token or os.getenv("WHATSAPP_TOKEN") or os.getenv("WHATSAPP_ACCESS_TOKEN")
        )

    @property
    def has_baileys(self) -> bool:
        """Check if Baileys is available."""
        if self.baileys_host:
            return True
        if self.baileys_session_path:
            from pathlib import Path

            return Path(self.baileys_session_path).exists()
        return False

    @property
    def has_business_api(self) -> bool:
        """Check if Business API is available."""
        return bool(self.business_phone_id and self.business_token)


# Global config
_config: Optional[WhatsAppConfig] = None


def get_config() -> WhatsAppConfig:
    """Get WhatsApp configuration singleton."""
    global _config
    if _config is None:
        _config = WhatsAppConfig()
    return _config


def get_provider(provider: str = "auto"):
    """
    Get WhatsApp provider based on selection and availability.

    Provider selection (like OpenClaw):
    1. If Baileys available: Prefer Baileys (free, open-source)
    2. If Business API available: Use Business API
    3. Raise error if neither available
    """
    from .baileys import BaileysProvider
    from .business_api import BusinessAPIProvider

    config = get_config()

    if provider == "auto":
        # Prefer Baileys (like OpenClaw)
        if config.has_baileys:
            return BaileysProvider()
        elif config.has_business_api:
            return BusinessAPIProvider()
        else:
            raise RuntimeError(
                "No WhatsApp provider available. "
                "Set BAILEYS_SESSION_PATH or WHATSAPP_PHONE_ID/WHATSAPP_TOKEN"
            )

    elif provider == "baileys":
        if not config.has_baileys:
            raise RuntimeError("Baileys not available. Set BAILEYS_SESSION_PATH or BAILEYS_HOST")
        return BaileysProvider()

    elif provider == "business":
        if not config.has_business_api:
            raise RuntimeError(
                "Business API not available. Set WHATSAPP_PHONE_ID and WHATSAPP_TOKEN"
            )
        return BusinessAPIProvider()

    else:
        raise ValueError(f"Unknown WhatsApp provider: {provider}")


__all__ = ["get_provider", "get_config", "WhatsAppConfig"]
