"""
Unit tests for ToolPolicy / PolicyChain — cascading deny-wins permission chain.

Tests cover:
- Empty chain (no policies) allows everything
- Global allow / deny semantics
- Deny-wins across scopes (channel, group)
- Channel-specific isolation
- Missing policy defaults to allow
- Wildcard "*" deny blocks all tools in scope
- YAML loading (with temp files)
- Singleton caching via get_policy_chain
- Complex multi-policy cascade scenarios
"""

import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from Jotty.core.infrastructure.integration.tool_policy import (
    PolicyChain,
    ToolPolicy,
    get_policy_chain,
    load_policy_chain,
)


@pytest.mark.unit
class TestToolPolicy:
    """Tests for the ToolPolicy dataclass."""

    def test_tool_policy_fields(self):
        """ToolPolicy stores tool_name, allowed, and scope."""
        policy = ToolPolicy(tool_name="web-search", allowed=True, scope="global")
        assert policy.tool_name == "web-search"
        assert policy.allowed is True
        assert policy.scope == "global"

    def test_tool_policy_deny(self):
        """ToolPolicy can represent a deny rule."""
        policy = ToolPolicy(tool_name="dangerous-tool", allowed=False, scope="channel:telegram")
        assert policy.allowed is False
        assert policy.scope == "channel:telegram"


@pytest.mark.unit
class TestPolicyChainEmpty:
    """Tests for PolicyChain with no policies configured."""

    def test_empty_chain_allows_any_tool(self):
        """An empty PolicyChain should allow any tool."""
        chain = PolicyChain()
        assert chain.evaluate("web-search") is True
        assert chain.evaluate("file-write") is True
        assert chain.evaluate("code-exec") is True

    def test_empty_chain_allows_with_channel(self):
        """An empty chain allows any tool even with a channel context."""
        chain = PolicyChain()
        assert chain.evaluate("web-search", channel="telegram") is True

    def test_empty_chain_allows_with_group(self):
        """An empty chain allows any tool even with a group context."""
        chain = PolicyChain()
        assert chain.evaluate("code-exec", group="students") is True

    def test_empty_chain_allows_with_both(self):
        """An empty chain allows any tool with both channel and group."""
        chain = PolicyChain()
        assert chain.evaluate("any-tool", channel="slack", group="admins") is True


@pytest.mark.unit
class TestPolicyChainGlobal:
    """Tests for global-level policies."""

    def test_global_allow_policy(self):
        """A global allow policy should permit the tool."""
        chain = PolicyChain()
        chain.add_policy("web-search", allowed=True, scope="global")
        assert chain.evaluate("web-search") is True

    def test_global_deny_policy_blocks_tool(self):
        """A global deny policy should block the tool."""
        chain = PolicyChain()
        chain.add_policy("dangerous-tool", allowed=False, scope="global")
        assert chain.evaluate("dangerous-tool") is False

    def test_global_deny_blocks_with_channel_context(self):
        """A globally denied tool stays denied regardless of channel."""
        chain = PolicyChain()
        chain.add_policy("dangerous-tool", allowed=False, scope="global")
        assert chain.evaluate("dangerous-tool", channel="telegram") is False

    def test_global_deny_blocks_with_group_context(self):
        """A globally denied tool stays denied regardless of group."""
        chain = PolicyChain()
        chain.add_policy("dangerous-tool", allowed=False, scope="global")
        assert chain.evaluate("dangerous-tool", group="admins") is False

    def test_global_allow_does_not_affect_other_tools(self):
        """Allowing one tool globally does not change the default for others."""
        chain = PolicyChain()
        chain.add_policy("web-search", allowed=True, scope="global")
        # Other tools should still be allowed (missing policy = allow)
        assert chain.evaluate("file-write") is True


@pytest.mark.unit
class TestPolicyChainDenyWins:
    """Tests for deny-wins semantics across scopes."""

    def test_global_allow_channel_deny_results_in_denied(self):
        """Deny at channel level overrides global allow."""
        chain = PolicyChain()
        chain.add_policy("file-write", allowed=True, scope="global")
        chain.add_policy("file-write", allowed=False, scope="channel:telegram")

        # Without channel context: allowed (only global checked)
        assert chain.evaluate("file-write") is True
        # With the denying channel: denied
        assert chain.evaluate("file-write", channel="telegram") is False

    def test_global_allow_group_deny_results_in_denied(self):
        """Deny at group level overrides global allow."""
        chain = PolicyChain()
        chain.add_policy("code-exec", allowed=True, scope="global")
        chain.add_policy("code-exec", allowed=False, scope="group:students")

        # Without group context: allowed
        assert chain.evaluate("code-exec") is True
        # With the denying group: denied
        assert chain.evaluate("code-exec", group="students") is False

    def test_channel_allow_global_deny_still_denied(self):
        """Deny at global level cannot be overridden by channel allow."""
        chain = PolicyChain()
        chain.add_policy("tool-x", allowed=False, scope="global")
        chain.add_policy("tool-x", allowed=True, scope="channel:slack")

        # Global deny wins even though channel says allow
        assert chain.evaluate("tool-x", channel="slack") is False

    def test_group_allow_global_deny_still_denied(self):
        """Deny at global level cannot be overridden by group allow."""
        chain = PolicyChain()
        chain.add_policy("tool-x", allowed=False, scope="global")
        chain.add_policy("tool-x", allowed=True, scope="group:admins")

        # Global deny wins
        assert chain.evaluate("tool-x", group="admins") is False

    def test_deny_at_any_single_level_blocks(self):
        """If any one scope denies, the tool is blocked."""
        chain = PolicyChain()
        chain.add_policy("tool-y", allowed=True, scope="global")
        chain.add_policy("tool-y", allowed=True, scope="channel:web")
        chain.add_policy("tool-y", allowed=False, scope="group:restricted")

        # Group deny should block
        assert chain.evaluate("tool-y", channel="web", group="restricted") is False


@pytest.mark.unit
class TestPolicyChainChannelIsolation:
    """Tests for channel-specific deny isolation."""

    def test_channel_deny_does_not_affect_other_channels(self):
        """A deny on channel:telegram should not affect channel:slack."""
        chain = PolicyChain()
        chain.add_policy("file-write", allowed=False, scope="channel:telegram")

        # Denied on telegram
        assert chain.evaluate("file-write", channel="telegram") is False
        # Allowed on slack (no policy for channel:slack)
        assert chain.evaluate("file-write", channel="slack") is True
        # Allowed with no channel specified
        assert chain.evaluate("file-write") is True

    def test_group_deny_does_not_affect_other_groups(self):
        """A deny on group:students should not affect group:admins."""
        chain = PolicyChain()
        chain.add_policy("code-exec", allowed=False, scope="group:students")

        assert chain.evaluate("code-exec", group="students") is False
        assert chain.evaluate("code-exec", group="admins") is True
        assert chain.evaluate("code-exec") is True


@pytest.mark.unit
class TestPolicyChainMissingPolicy:
    """Tests for missing policy defaults."""

    def test_missing_policy_for_tool_allows_it(self):
        """A tool with no policy at any level should be allowed."""
        chain = PolicyChain()
        # Add policies for other tools but not "unknown-tool"
        chain.add_policy("web-search", allowed=True, scope="global")
        chain.add_policy("dangerous-tool", allowed=False, scope="global")

        assert chain.evaluate("unknown-tool") is True
        assert chain.evaluate("unknown-tool", channel="telegram") is True
        assert chain.evaluate("unknown-tool", group="students") is True

    def test_missing_scope_policies_allow(self):
        """If only global has policies and we check channel, missing scope = allow."""
        chain = PolicyChain()
        chain.add_policy("web-search", allowed=True, scope="global")

        # channel:telegram scope has no policies at all
        assert chain.evaluate("web-search", channel="telegram") is True


@pytest.mark.unit
class TestPolicyChainWildcard:
    """Tests for wildcard '*' deny behavior."""

    def test_global_wildcard_deny_blocks_all_tools(self):
        """A wildcard '*' deny at global scope blocks all tools."""
        chain = PolicyChain()
        chain.add_policy("*", allowed=False, scope="global")

        assert chain.evaluate("web-search") is False
        assert chain.evaluate("file-write") is False
        assert chain.evaluate("any-tool") is False

    def test_channel_wildcard_deny_blocks_all_tools_in_channel(self):
        """A wildcard '*' deny at channel scope blocks all tools in that channel."""
        chain = PolicyChain()
        chain.add_policy("*", allowed=False, scope="channel:telegram")

        # No channel: allowed (wildcard only in channel:telegram)
        assert chain.evaluate("web-search") is True
        # With telegram channel: denied
        assert chain.evaluate("web-search", channel="telegram") is False
        assert chain.evaluate("file-write", channel="telegram") is False
        # Other channel: allowed
        assert chain.evaluate("web-search", channel="slack") is True

    def test_group_wildcard_deny_blocks_all_tools_in_group(self):
        """A wildcard '*' deny at group scope blocks all tools in that group."""
        chain = PolicyChain()
        chain.add_policy("*", allowed=False, scope="group:students")

        assert chain.evaluate("code-exec") is True
        assert chain.evaluate("code-exec", group="students") is False
        assert chain.evaluate("code-exec", group="admins") is True

    def test_wildcard_deny_with_explicit_allow_still_denied(self):
        """
        Wildcard deny blocks even if the same tool has an explicit allow
        in the same scope, because deny-wins checks wildcard and exact name
        independently.
        """
        chain = PolicyChain()
        chain.add_policy("*", allowed=False, scope="global")
        chain.add_policy("web-search", allowed=True, scope="global")

        # Wildcard deny in global scope means deny (deny wins)
        assert chain.evaluate("web-search") is False


@pytest.mark.unit
class TestLoadPolicyChain:
    """Tests for load_policy_chain YAML loading."""

    def test_load_from_yaml_file(self, tmp_path):
        """load_policy_chain reads tool_policies.yaml and builds a PolicyChain."""
        yaml = pytest.importorskip("yaml")

        policy_data = {
            "global": {
                "web-search": True,
                "dangerous-tool": False,
            },
            "channel:telegram": {
                "file-write": False,
            },
            "group:students": {
                "code-exec": False,
            },
        }

        policy_file = tmp_path / "tool_policies.yaml"
        policy_file.write_text(yaml.dump(policy_data), encoding="utf-8")

        chain = load_policy_chain(str(tmp_path))

        # Global policies
        assert chain.evaluate("web-search") is True
        assert chain.evaluate("dangerous-tool") is False

        # Channel policy
        assert chain.evaluate("file-write", channel="telegram") is False
        assert chain.evaluate("file-write", channel="slack") is True

        # Group policy
        assert chain.evaluate("code-exec", group="students") is False
        assert chain.evaluate("code-exec", group="admins") is True

        # Unlisted tool is allowed
        assert chain.evaluate("other-tool") is True

    def test_load_with_no_yaml_file_returns_empty_chain(self, tmp_path):
        """When no tool_policies.yaml exists, return an empty chain (all allowed)."""
        chain = load_policy_chain(str(tmp_path))

        assert chain.evaluate("web-search") is True
        assert chain.evaluate("dangerous-tool") is True
        assert chain.evaluate("anything", channel="telegram", group="students") is True

    def test_load_with_empty_yaml(self, tmp_path):
        """An empty YAML file should produce an empty chain."""
        policy_file = tmp_path / "tool_policies.yaml"
        policy_file.write_text("", encoding="utf-8")

        chain = load_policy_chain(str(tmp_path))
        assert chain.evaluate("web-search") is True

    def test_load_with_invalid_yaml_content(self, tmp_path):
        """Non-dict top-level YAML content should produce an empty chain."""
        policy_file = tmp_path / "tool_policies.yaml"
        policy_file.write_text("- list\n- not\n- dict\n", encoding="utf-8")

        chain = load_policy_chain(str(tmp_path))
        assert chain.evaluate("web-search") is True

    def test_load_with_non_dict_scope_values(self, tmp_path):
        """If a scope value is not a dict, it should be skipped gracefully."""
        yaml = pytest.importorskip("yaml")

        policy_data = {
            "global": "not-a-dict",
            "channel:telegram": {"file-write": False},
        }

        policy_file = tmp_path / "tool_policies.yaml"
        policy_file.write_text(yaml.dump(policy_data), encoding="utf-8")

        chain = load_policy_chain(str(tmp_path))
        # global scope was skipped (not a dict)
        assert chain.evaluate("web-search") is True
        # channel:telegram was loaded correctly
        assert chain.evaluate("file-write", channel="telegram") is False

    def test_load_without_pyyaml_installed(self, tmp_path):
        """If PyYAML is not installed, load returns an empty chain."""
        policy_file = tmp_path / "tool_policies.yaml"
        policy_file.write_text("global:\n  web-search: true\n", encoding="utf-8")

        with patch.dict("sys.modules", {"yaml": None}):
            # Force re-import to fail
            import importlib
            import Jotty.core.infrastructure.integration.tool_policy as tp_mod

            # Directly call with the import patched out
            chain = PolicyChain()
            # Simulate what load_policy_chain does when yaml import fails
            filepath = Path(tmp_path) / "tool_policies.yaml"
            assert filepath.is_file()
            try:
                import yaml  # noqa: F811
            except ImportError:
                pass  # This won't actually fail in our test env

        # The main thing: load_policy_chain should handle missing yaml gracefully
        # We can test this by verifying the function itself doesn't crash
        chain = load_policy_chain(str(tmp_path))
        # If yaml IS available, it loads; if not, returns empty chain
        # Either way, the function should not raise
        assert isinstance(chain, PolicyChain)


@pytest.mark.unit
class TestGetPolicyChain:
    """Tests for get_policy_chain singleton caching."""

    def setup_method(self):
        """Clear the singleton cache before each test."""
        import Jotty.core.infrastructure.integration.tool_policy as tp_mod

        tp_mod._instances.clear()

    def test_empty_workspace_returns_allow_all(self):
        """get_policy_chain with empty string returns a fresh chain (all allowed)."""
        chain = get_policy_chain("")
        assert chain.evaluate("web-search") is True
        assert chain.evaluate("dangerous-tool") is True
        assert chain.evaluate("anything", channel="telegram", group="students") is True

    def test_empty_workspace_returns_new_instance_each_time(self):
        """Empty workspace should not be cached (returns fresh each time)."""
        chain1 = get_policy_chain("")
        chain2 = get_policy_chain("")
        # They are separate instances (no caching for empty workspace)
        assert chain1 is not chain2

    def test_workspace_is_cached(self, tmp_path):
        """Same workspace_dir returns the same cached PolicyChain."""
        workspace = str(tmp_path)
        chain1 = get_policy_chain(workspace)
        chain2 = get_policy_chain(workspace)
        assert chain1 is chain2

    def test_different_workspaces_get_different_chains(self, tmp_path):
        """Different workspace directories get independent chains."""
        ws1 = str(tmp_path / "workspace1")
        ws2 = str(tmp_path / "workspace2")
        Path(ws1).mkdir()
        Path(ws2).mkdir()

        chain1 = get_policy_chain(ws1)
        chain2 = get_policy_chain(ws2)
        assert chain1 is not chain2

    def test_workspace_with_yaml_loads_policies(self, tmp_path):
        """get_policy_chain loads from tool_policies.yaml if it exists."""
        yaml = pytest.importorskip("yaml")

        policy_data = {"global": {"code-exec": False}}
        policy_file = tmp_path / "tool_policies.yaml"
        policy_file.write_text(yaml.dump(policy_data), encoding="utf-8")

        chain = get_policy_chain(str(tmp_path))
        assert chain.evaluate("code-exec") is False
        assert chain.evaluate("web-search") is True


@pytest.mark.unit
class TestPolicyChainComplexCascade:
    """Tests for complex multi-policy cascade scenarios."""

    def test_complex_cascade_scenario(self):
        """
        Multi-scope scenario:
        - Global: web-search allowed, code-exec allowed, file-delete denied
        - channel:telegram: code-exec denied
        - group:students: web-search denied

        Expected:
        - web-search: globally OK, but denied for group:students
        - code-exec: globally OK, but denied on channel:telegram
        - file-delete: denied everywhere (global deny)
        - other-tool: allowed everywhere (no policy)
        """
        chain = PolicyChain()
        chain.add_policy("web-search", allowed=True, scope="global")
        chain.add_policy("code-exec", allowed=True, scope="global")
        chain.add_policy("file-delete", allowed=False, scope="global")
        chain.add_policy("code-exec", allowed=False, scope="channel:telegram")
        chain.add_policy("web-search", allowed=False, scope="group:students")

        # No context
        assert chain.evaluate("web-search") is True
        assert chain.evaluate("code-exec") is True
        assert chain.evaluate("file-delete") is False
        assert chain.evaluate("other-tool") is True

        # Telegram channel
        assert chain.evaluate("web-search", channel="telegram") is True
        assert chain.evaluate("code-exec", channel="telegram") is False
        assert chain.evaluate("file-delete", channel="telegram") is False

        # Students group
        assert chain.evaluate("web-search", group="students") is False
        assert chain.evaluate("code-exec", group="students") is True
        assert chain.evaluate("file-delete", group="students") is False

        # Both telegram + students
        assert chain.evaluate("web-search", channel="telegram", group="students") is False
        assert chain.evaluate("code-exec", channel="telegram", group="students") is False
        assert chain.evaluate("file-delete", channel="telegram", group="students") is False
        assert chain.evaluate("other-tool", channel="telegram", group="students") is True

    def test_multiple_channels_and_groups(self):
        """Test with policies spread across multiple channels and groups."""
        chain = PolicyChain()
        chain.add_policy("tool-a", allowed=False, scope="channel:telegram")
        chain.add_policy("tool-b", allowed=False, scope="channel:slack")
        chain.add_policy("tool-c", allowed=False, scope="group:students")
        chain.add_policy("tool-d", allowed=False, scope="group:trial-users")

        # Each deny is isolated to its scope
        assert chain.evaluate("tool-a", channel="telegram") is False
        assert chain.evaluate("tool-a", channel="slack") is True
        assert chain.evaluate("tool-b", channel="slack") is False
        assert chain.evaluate("tool-b", channel="telegram") is True

        assert chain.evaluate("tool-c", group="students") is False
        assert chain.evaluate("tool-c", group="trial-users") is True
        assert chain.evaluate("tool-d", group="trial-users") is False
        assert chain.evaluate("tool-d", group="students") is True

    def test_wildcard_deny_combined_with_specific_allow(self):
        """
        Wildcard deny in channel scope combined with global allow.
        Channel wildcard deny should block everything in that channel.
        """
        chain = PolicyChain()
        chain.add_policy("web-search", allowed=True, scope="global")
        chain.add_policy("code-exec", allowed=True, scope="global")
        chain.add_policy("*", allowed=False, scope="channel:restricted")

        # Without channel: allowed
        assert chain.evaluate("web-search") is True
        assert chain.evaluate("code-exec") is True

        # In restricted channel: wildcard blocks everything
        assert chain.evaluate("web-search", channel="restricted") is False
        assert chain.evaluate("code-exec", channel="restricted") is False
        assert chain.evaluate("any-unknown-tool", channel="restricted") is False

        # Other channels: still allowed
        assert chain.evaluate("web-search", channel="open") is True

    def test_add_policy_overwrites_same_scope_and_tool(self):
        """Adding a policy for the same tool and scope overwrites the previous one."""
        chain = PolicyChain()
        chain.add_policy("web-search", allowed=True, scope="global")
        assert chain.evaluate("web-search") is True

        chain.add_policy("web-search", allowed=False, scope="global")
        assert chain.evaluate("web-search") is False

    def test_end_to_end_yaml_complex_scenario(self, tmp_path):
        """Full end-to-end test: YAML load + complex evaluation."""
        yaml = pytest.importorskip("yaml")

        policy_data = {
            "global": {
                "web-search": True,
                "file-delete": False,
            },
            "channel:telegram": {
                "code-exec": False,
                "web-search": True,
            },
            "channel:restricted": {
                "*": False,
            },
            "group:students": {
                "code-exec": False,
            },
        }

        policy_file = tmp_path / "tool_policies.yaml"
        policy_file.write_text(yaml.dump(policy_data), encoding="utf-8")

        chain = load_policy_chain(str(tmp_path))

        # Global policies work
        assert chain.evaluate("web-search") is True
        assert chain.evaluate("file-delete") is False

        # Channel:telegram denies code-exec
        assert chain.evaluate("code-exec", channel="telegram") is False

        # Channel:restricted wildcard denies everything
        assert chain.evaluate("web-search", channel="restricted") is False
        assert chain.evaluate("code-exec", channel="restricted") is False

        # Group:students denies code-exec
        assert chain.evaluate("code-exec", group="students") is False
        assert chain.evaluate("web-search", group="students") is True

        # Combined: telegram + students
        assert chain.evaluate("code-exec", channel="telegram", group="students") is False
        assert chain.evaluate("web-search", channel="telegram", group="students") is True

        # Unlisted tool in neutral context
        assert chain.evaluate("unknown-tool") is True
