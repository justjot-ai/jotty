"""
Tests for the NO_REPLY sentinel and suppress_output feature.

Covers:
- NO_REPLY sentinel identity and is_no_reply() checks
- JottyMessage.suppress_output field (default, explicit True)
- Round-trip serialization via to_dict() / from_dict()
"""

import pytest

from Jotty.core.interface.interfaces.message import (
    NO_REPLY,
    InterfaceType,
    JottyMessage,
    is_no_reply,
)


# ---------------------------------------------------------------------------
# Helper: build a minimal JottyMessage for tests
# ---------------------------------------------------------------------------
def _make_message(**overrides) -> JottyMessage:
    defaults = {
        "content": "hello",
        "interface": InterfaceType.CLI,
        "user_id": "test_user",
        "session_id": "test_session",
    }
    defaults.update(overrides)
    return JottyMessage(**defaults)


# ===========================================================================
# NO_REPLY sentinel tests
# ===========================================================================


class TestNoReplySentinel:
    """Tests for the NO_REPLY module-level sentinel object."""

    @pytest.mark.unit
    def test_no_reply_is_not_none(self):
        """NO_REPLY must be a distinct sentinel, not None."""
        assert NO_REPLY is not None

    @pytest.mark.unit
    def test_no_reply_is_not_false(self):
        """NO_REPLY must not be False."""
        assert NO_REPLY is not False

    @pytest.mark.unit
    def test_no_reply_identity(self):
        """The sentinel is the same object when referenced twice."""
        assert NO_REPLY is NO_REPLY

    @pytest.mark.unit
    def test_different_object_is_not_no_reply(self):
        """A fresh object() must not be confused with NO_REPLY."""
        other = object()
        assert other is not NO_REPLY


# ===========================================================================
# is_no_reply() function tests
# ===========================================================================


class TestIsNoReply:
    """Tests for the is_no_reply() helper function."""

    @pytest.mark.unit
    def test_is_no_reply_with_sentinel(self):
        """is_no_reply(NO_REPLY) returns True."""
        assert is_no_reply(NO_REPLY) is True

    @pytest.mark.unit
    def test_is_no_reply_with_none(self):
        """is_no_reply(None) returns False."""
        assert is_no_reply(None) is False

    @pytest.mark.unit
    def test_is_no_reply_with_string(self):
        """is_no_reply('some string') returns False."""
        assert is_no_reply("some string") is False

    @pytest.mark.unit
    def test_is_no_reply_with_false(self):
        """is_no_reply(False) returns False."""
        assert is_no_reply(False) is False


# ===========================================================================
# JottyMessage.suppress_output field tests
# ===========================================================================


class TestSuppressOutput:
    """Tests for the suppress_output field on JottyMessage."""

    @pytest.mark.unit
    def test_default_suppress_output_is_false(self):
        """A new JottyMessage defaults suppress_output to False."""
        msg = _make_message()
        assert msg.suppress_output is False

    @pytest.mark.unit
    def test_suppress_output_set_true(self):
        """suppress_output can be explicitly set to True."""
        msg = _make_message(suppress_output=True)
        assert msg.suppress_output is True

    @pytest.mark.unit
    def test_to_dict_includes_suppress_output(self):
        """to_dict() must include the suppress_output key."""
        msg = _make_message(suppress_output=True)
        d = msg.to_dict()
        assert "suppress_output" in d
        assert d["suppress_output"] is True

    @pytest.mark.unit
    def test_to_dict_suppress_output_false(self):
        """to_dict() includes suppress_output even when False."""
        msg = _make_message()
        d = msg.to_dict()
        assert "suppress_output" in d
        assert d["suppress_output"] is False

    @pytest.mark.unit
    def test_from_dict_with_suppress_output_true(self):
        """from_dict() reads suppress_output=True correctly."""
        data = {
            "content": "test",
            "interface": "cli",
            "user_id": "u1",
            "session_id": "s1",
            "suppress_output": True,
        }
        msg = JottyMessage.from_dict(data)
        assert msg.suppress_output is True

    @pytest.mark.unit
    def test_from_dict_without_suppress_output_defaults_false(self):
        """from_dict() defaults suppress_output to False when key is absent."""
        data = {
            "content": "test",
            "interface": "cli",
            "user_id": "u1",
            "session_id": "s1",
        }
        msg = JottyMessage.from_dict(data)
        assert msg.suppress_output is False
