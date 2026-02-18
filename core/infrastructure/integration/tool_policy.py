"""
Tool Access Policy — Cascading Deny-Wins Permission Chain
=========================================================

Cascading permission chain: global defaults -> channel overrides -> group overrides.
"deny" at any level wins (most restrictive policy applies).

Loads optional tool_policies.yaml from workspace root.

Usage:
    from Jotty.core.infrastructure.integration.tool_policy import get_policy_chain

    chain = get_policy_chain("/workspace")
    allowed = chain.evaluate("web-search", channel="telegram", group="students")
    # True if allowed, False if denied at any level
"""

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_POLICY_FILENAME = "tool_policies.yaml"


@dataclass
class ToolPolicy:
    """Single tool permission entry."""

    tool_name: str
    allowed: bool
    scope: str  # "global", "channel:{name}", "group:{name}"


@dataclass
class PolicyChain:
    """
    Cascading permission chain with deny-wins semantics.

    Resolution order:
    1. Check group-level policy (most specific)
    2. Check channel-level policy
    3. Check global-level policy
    4. Default: allow (no policy = allow)

    If ANY level says deny, the tool is denied.
    """

    # Nested dict: scope -> tool_name -> allowed
    _policies: Dict[str, Dict[str, bool]] = field(default_factory=dict)

    def add_policy(self, tool_name: str, allowed: bool, scope: str = "global") -> None:
        """Add a policy entry."""
        if scope not in self._policies:
            self._policies[scope] = {}
        self._policies[scope][tool_name] = allowed

    def evaluate(
        self,
        tool_name: str,
        channel: Optional[str] = None,
        group: Optional[str] = None,
    ) -> bool:
        """
        Evaluate whether a tool is allowed. Deny wins at any level.

        Args:
            tool_name: The tool to check
            channel: Optional channel context (e.g., "telegram")
            group: Optional group context (e.g., "students")

        Returns:
            True if allowed, False if denied at any level
        """
        # Check each applicable scope — deny at ANY level blocks
        scopes_to_check = ["global"]
        if channel:
            scopes_to_check.append(f"channel:{channel}")
        if group:
            scopes_to_check.append(f"group:{group}")

        for scope in scopes_to_check:
            scope_policies = self._policies.get(scope, {})

            # Check exact tool name
            if tool_name in scope_policies:
                if not scope_policies[tool_name]:
                    return False  # Deny wins

            # Check wildcard "*" (applies to all tools in scope)
            if "*" in scope_policies:
                if not scope_policies["*"]:
                    return False

        return True  # No deny found → allow


def load_policy_chain(workspace_dir: str) -> PolicyChain:
    """
    Load tool policies from workspace tool_policies.yaml.

    Expected YAML format:
        global:
          web-search: true
          dangerous-tool: false
        channel:telegram:
          file-write: false
        group:students:
          code-exec: false

    Args:
        workspace_dir: Path to workspace root

    Returns:
        PolicyChain with loaded policies (empty if no file)
    """
    chain = PolicyChain()
    filepath = Path(workspace_dir) / _POLICY_FILENAME

    if not filepath.is_file():
        return chain

    try:
        import yaml

        content = filepath.read_text(encoding="utf-8")
        data = yaml.safe_load(content)

        if not isinstance(data, dict):
            return chain

        for scope, tools in data.items():
            if not isinstance(tools, dict):
                continue
            for tool_name, allowed in tools.items():
                chain.add_policy(
                    tool_name=str(tool_name),
                    allowed=bool(allowed),
                    scope=str(scope),
                )

        logger.info(f"Loaded tool policies from {filepath}")
    except ImportError:
        logger.debug("PyYAML not installed, tool policies disabled")
    except Exception as e:
        logger.warning(f"Failed to load tool policies from {filepath}: {e}")

    return chain


# =============================================================================
# Singleton with workspace-based caching
# =============================================================================

_lock = threading.Lock()
_instances: Dict[str, PolicyChain] = {}


def get_policy_chain(workspace_dir: str = "") -> PolicyChain:
    """Get or create PolicyChain for a workspace."""
    if not workspace_dir:
        return PolicyChain()  # Empty chain = allow all

    if workspace_dir not in _instances:
        with _lock:
            if workspace_dir not in _instances:
                _instances[workspace_dir] = load_policy_chain(workspace_dir)
    return _instances[workspace_dir]
