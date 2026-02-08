"""
INTERFACE Layer - External Entry Points
"""
# Use absolute imports to avoid relative import complexity
from Jotty.cli.app import JottyCLI
from Jotty.cli.gateway import UnifiedGateway, ChannelRouter, ChannelType, MessageEvent, start_gateway

try:
    from Jotty.sdk.generated.python.jotty_api_client import Client as SDKClient
except ImportError:
    SDKClient = None

__all__ = ["JottyCLI", "UnifiedGateway", "ChannelRouter", "ChannelType", "MessageEvent", "start_gateway", "SDKClient"]
