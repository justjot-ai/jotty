"""
Trust Layer
===========

Security layer for gateway message handling.
Implements allowlist-based trust with pairing codes (like OpenClaw DM security).

Features:
- User allowlist stored in ~/.jotty/allowed_users.json
- 6-digit pairing codes for new DMs
- Auto-expire codes after 5 minutes
- Per-channel trust policies
"""

import json
import logging
import os
import random
import string
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Set

from .channels import ChannelType

logger = logging.getLogger(__name__)


class TrustPolicy(Enum):
    """Trust policies for channels."""

    OPEN = "open"  # Accept all messages (no trust required)
    PAIRING = "pairing"  # Require pairing code for new users
    ALLOWLIST = "allowlist"  # Only accept from allowlist (strict)


@dataclass
class PairingCode:
    """Temporary pairing code for user verification."""

    code: str
    user_id: str
    channel: ChannelType
    created_at: datetime
    expires_at: datetime

    def is_expired(self) -> bool:
        return datetime.now() > self.expires_at


@dataclass
class TrustedUser:
    """A trusted user entry."""

    user_id: str
    channel: str  # ChannelType.value
    user_name: str = ""
    added_at: str = ""
    added_by: str = "system"


class TrustManager:
    """
    Manages user trust and authorization for gateway messages.

    Like OpenClaw's DM pairing security:
    - New users must provide a pairing code to be trusted
    - Trusted users are stored in allowlist
    - Per-channel policies control trust requirements
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.config_path = Path(config_path or os.path.expanduser("~/.jotty/allowed_users.json"))
        self._allowlist: Dict[str, Set[str]] = {}  # channel -> set of user_ids
        self._pending_codes: Dict[str, PairingCode] = {}  # code -> PairingCode
        self._user_pending: Dict[str, str] = {}  # user_key -> code (for lookup)
        self._policies: Dict[str, TrustPolicy] = {}  # channel -> policy

        # Default policies
        self._default_policies = {
            ChannelType.WEBSOCKET.value: TrustPolicy.OPEN,
            ChannelType.HTTP.value: TrustPolicy.OPEN,
            ChannelType.TELEGRAM.value: TrustPolicy.PAIRING,
            ChannelType.SLACK.value: TrustPolicy.OPEN,  # Slack has its own auth
            ChannelType.DISCORD.value: TrustPolicy.PAIRING,
            ChannelType.WHATSAPP.value: TrustPolicy.PAIRING,
        }

        self._load_allowlist()

    def _load_allowlist(self) -> None:
        """Load allowlist from config file."""
        try:
            if self.config_path.exists():
                data = json.loads(self.config_path.read_text())
                self._allowlist = {
                    channel: set(users) for channel, users in data.get("users", {}).items()
                }
                self._policies = {
                    channel: TrustPolicy(policy)
                    for channel, policy in data.get("policies", {}).items()
                }
                logger.info(f"Loaded trust allowlist from {self.config_path}")
            else:
                self._allowlist = {}
                self._policies = {}
        except Exception as e:
            logger.error(f"Failed to load allowlist: {e}")
            self._allowlist = {}
            self._policies = {}

    def _save_allowlist(self) -> None:
        """Save allowlist to config file."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "users": {channel: list(users) for channel, users in self._allowlist.items()},
                "policies": {channel: policy.value for channel, policy in self._policies.items()},
                "updated_at": datetime.now().isoformat(),
            }
            self.config_path.write_text(json.dumps(data, indent=2))
            logger.debug(f"Saved trust allowlist to {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to save allowlist: {e}")

    def get_policy(self, channel: ChannelType) -> TrustPolicy:
        """Get trust policy for a channel."""
        channel_key = channel.value if isinstance(channel, ChannelType) else str(channel)
        return self._policies.get(
            channel_key, self._default_policies.get(channel_key, TrustPolicy.PAIRING)
        )

    def set_policy(self, channel: ChannelType, policy: TrustPolicy) -> None:
        """Set trust policy for a channel."""
        channel_key = channel.value if isinstance(channel, ChannelType) else str(channel)
        self._policies[channel_key] = policy
        self._save_allowlist()

    def is_allowed(self, channel: ChannelType, user_id: str) -> bool:
        """
        Check if user is allowed to send messages on this channel.

        Args:
            channel: The channel type
            user_id: User identifier

        Returns:
            True if user is allowed, False otherwise
        """
        policy = self.get_policy(channel)

        # Open policy - everyone allowed
        if policy == TrustPolicy.OPEN:
            return True

        # Check allowlist
        channel_key = channel.value if isinstance(channel, ChannelType) else str(channel)
        channel_users = self._allowlist.get(channel_key, set())

        return user_id in channel_users

    def add_to_allowlist(
        self, channel: ChannelType, user_id: str, user_name: str = "", added_by: str = "system"
    ) -> None:
        """Add user to allowlist."""
        channel_key = channel.value if isinstance(channel, ChannelType) else str(channel)

        if channel_key not in self._allowlist:
            self._allowlist[channel_key] = set()

        self._allowlist[channel_key].add(user_id)
        self._save_allowlist()
        logger.info(f"Added user {user_id} to {channel_key} allowlist")

    def remove_from_allowlist(self, channel: ChannelType, user_id: str) -> bool:
        """Remove user from allowlist."""
        channel_key = channel.value if isinstance(channel, ChannelType) else str(channel)

        if channel_key in self._allowlist and user_id in self._allowlist[channel_key]:
            self._allowlist[channel_key].discard(user_id)
            self._save_allowlist()
            logger.info(f"Removed user {user_id} from {channel_key} allowlist")
            return True

        return False

    def generate_pairing_code(
        self, channel: ChannelType, user_id: str, expires_minutes: int = 5
    ) -> str:
        """
        Generate a 6-digit pairing code for a user.

        Args:
            channel: Channel type
            user_id: User identifier
            expires_minutes: Code expiration time in minutes

        Returns:
            6-digit pairing code
        """
        # Clean up expired codes first
        self._cleanup_expired_codes()

        # Check if user already has a pending code
        user_key = f"{channel.value}:{user_id}"
        if user_key in self._user_pending:
            existing_code = self._user_pending[user_key]
            if existing_code in self._pending_codes:
                pairing = self._pending_codes[existing_code]
                if not pairing.is_expired():
                    return pairing.code

        # Generate new 6-digit code
        code = "".join(random.choices(string.digits, k=6))

        # Ensure uniqueness
        while code in self._pending_codes:
            code = "".join(random.choices(string.digits, k=6))

        now = datetime.now()
        pairing = PairingCode(
            code=code,
            user_id=user_id,
            channel=channel,
            created_at=now,
            expires_at=now + timedelta(minutes=expires_minutes),
        )

        self._pending_codes[code] = pairing
        self._user_pending[user_key] = code

        logger.info(f"Generated pairing code for user {user_id} on {channel.value}")
        return code

    def verify_pairing_code(self, code: str, channel: ChannelType, user_id: str) -> bool:
        """
        Verify a pairing code and add user to allowlist if valid.

        Args:
            code: The pairing code to verify
            channel: Channel type
            user_id: User identifier

        Returns:
            True if code was valid and user was added to allowlist
        """
        self._cleanup_expired_codes()

        # Clean the code (remove spaces, etc.)
        code = code.strip().replace(" ", "").replace("-", "")

        pairing = self._pending_codes.get(code)

        if not pairing:
            logger.debug(f"Invalid pairing code: {code}")
            return False

        if pairing.is_expired():
            logger.debug(f"Expired pairing code: {code}")
            del self._pending_codes[code]
            return False

        if pairing.channel != channel or pairing.user_id != user_id:
            logger.debug(
                f"Pairing code mismatch: expected {pairing.channel.value}:{pairing.user_id}"
            )
            return False

        # Code is valid - add user to allowlist
        self.add_to_allowlist(channel, user_id, added_by="pairing")

        # Clean up the code
        del self._pending_codes[code]
        user_key = f"{channel.value}:{user_id}"
        self._user_pending.pop(user_key, None)

        logger.info(f"Verified pairing code for user {user_id} on {channel.value}")
        return True

    def _cleanup_expired_codes(self) -> None:
        """Remove expired pairing codes."""
        expired = [code for code, pairing in self._pending_codes.items() if pairing.is_expired()]

        for code in expired:
            pairing = self._pending_codes.pop(code, None)
            if pairing:
                user_key = f"{pairing.channel.value}:{pairing.user_id}"
                self._user_pending.pop(user_key, None)

    def check_message(self, channel: ChannelType, user_id: str, content: str) -> Dict[str, Any]:
        """
        Check if a message should be processed.

        This method handles the full trust flow:
        1. If user is allowed, return proceed=True
        2. If message contains a valid pairing code, verify and return proceed=True
        3. If user needs pairing, generate code and return proceed=False with pairing message

        Args:
            channel: Channel type
            user_id: User identifier
            content: Message content (might contain pairing code)

        Returns:
            Dict with:
                - proceed (bool): Whether to process the message
                - response (str, optional): Response to send to user
                - pairing_required (bool): Whether pairing is needed
        """
        policy = self.get_policy(channel)

        # Open policy - always proceed
        if policy == TrustPolicy.OPEN:
            return {"proceed": True, "pairing_required": False}

        # Check if already allowed
        if self.is_allowed(channel, user_id):
            return {"proceed": True, "pairing_required": False}

        # Check if message contains a pairing code
        # Look for 6-digit number in message
        import re

        code_match = re.search(r"\b(\d{6})\b", content)

        if code_match:
            code = code_match.group(1)
            if self.verify_pairing_code(code, channel, user_id):
                return {
                    "proceed": True,
                    "pairing_required": False,
                    "response": "Pairing successful! You can now send messages.",
                }

        # User needs to pair - generate code
        if policy == TrustPolicy.PAIRING:
            code = self.generate_pairing_code(channel, user_id)
            return {
                "proceed": False,
                "pairing_required": True,
                "response": f"Please reply with this pairing code to verify: {code}\n(Expires in 5 minutes)",
            }

        # Strict allowlist - just reject
        return {
            "proceed": False,
            "pairing_required": False,
            "response": "You are not authorized to send messages. Contact an administrator.",
        }

    @property
    def stats(self) -> Dict[str, Any]:
        """Get trust manager statistics."""
        return {
            "total_trusted_users": sum(len(users) for users in self._allowlist.values()),
            "pending_pairing_codes": len(self._pending_codes),
            "channels": list(self._allowlist.keys()),
            "policies": {channel: policy.value for channel, policy in self._policies.items()},
        }
