"""
Tests for SessionKey dataclass and JottyMessage.session_key property.

SessionKey provides structured session identifiers with format
{channel}:{user_id}:{thread_id}, replacing opaque session_id strings.
"""

import pytest

from Jotty.core.interface.interfaces.message import (
    InterfaceType,
    JottyMessage,
    SessionKey,
)


@pytest.mark.unit
class TestSessionKeyBuild:
    """Tests for SessionKey.build() classmethod."""

    def test_build_with_all_fields(self):
        key = SessionKey.build("telegram", "user123", "thread1")
        assert key.channel == "telegram"
        assert key.user_id == "user123"
        assert key.thread_id == "thread1"

    def test_build_with_default_thread_id(self):
        key = SessionKey.build("cli", "user456")
        assert key.channel == "cli"
        assert key.user_id == "user456"
        assert key.thread_id == "default"

    def test_build_returns_session_key_instance(self):
        key = SessionKey.build("web", "u1")
        assert isinstance(key, SessionKey)


@pytest.mark.unit
class TestSessionKeyRaw:
    """Tests for the raw property."""

    def test_raw_three_parts(self):
        key = SessionKey(channel="telegram", user_id="user123", thread_id="thread1")
        assert key.raw == "telegram:user123:thread1"

    def test_raw_default_thread(self):
        key = SessionKey(channel="cli", user_id="user456")
        assert key.raw == "cli:user456:default"


@pytest.mark.unit
class TestSessionKeyParse:
    """Tests for SessionKey.parse() classmethod."""

    def test_parse_three_parts(self):
        key = SessionKey.parse("telegram:user123:thread1")
        assert key.channel == "telegram"
        assert key.user_id == "user123"
        assert key.thread_id == "thread1"

    def test_parse_two_parts(self):
        key = SessionKey.parse("cli:user456")
        assert key.channel == "cli"
        assert key.user_id == "user456"
        assert key.thread_id == "default"

    def test_parse_legacy_plain_string(self):
        key = SessionKey.parse("some_legacy_session_id")
        assert key.channel == "unknown"
        assert key.user_id == "some_legacy_session_id"
        assert key.thread_id == "default"

    def test_parse_empty_string(self):
        key = SessionKey.parse("")
        assert key.channel == "unknown"
        assert key.user_id == "unknown"
        assert key.thread_id == "default"

    def test_parse_three_parts_with_colons_in_thread(self):
        """Colon within thread_id is preserved because split uses maxsplit=2."""
        key = SessionKey.parse("telegram:user1:thread:with:colons")
        assert key.channel == "telegram"
        assert key.user_id == "user1"
        assert key.thread_id == "thread:with:colons"

    def test_parse_roundtrip(self):
        """Parsing a raw string should reconstruct the original key."""
        original = SessionKey.build("api", "user789", "sess42")
        roundtripped = SessionKey.parse(original.raw)
        assert roundtripped.channel == original.channel
        assert roundtripped.user_id == original.user_id
        assert roundtripped.thread_id == original.thread_id


@pytest.mark.unit
class TestSessionKeyEquality:
    """Tests for __eq__ with SessionKey and str."""

    def test_equal_session_keys(self):
        key1 = SessionKey.build("telegram", "user1", "thread1")
        key2 = SessionKey.build("telegram", "user1", "thread1")
        assert key1 == key2

    def test_unequal_session_keys_different_channel(self):
        key1 = SessionKey.build("telegram", "user1", "thread1")
        key2 = SessionKey.build("cli", "user1", "thread1")
        assert key1 != key2

    def test_unequal_session_keys_different_user(self):
        key1 = SessionKey.build("telegram", "user1", "thread1")
        key2 = SessionKey.build("telegram", "user2", "thread1")
        assert key1 != key2

    def test_unequal_session_keys_different_thread(self):
        key1 = SessionKey.build("telegram", "user1", "thread1")
        key2 = SessionKey.build("telegram", "user1", "thread2")
        assert key1 != key2

    def test_equal_to_raw_string(self):
        key = SessionKey.build("telegram", "user123", "thread1")
        assert key == "telegram:user123:thread1"

    def test_equal_to_raw_string_reverse(self):
        """String == SessionKey also works (via SessionKey.__eq__)."""
        key = SessionKey.build("telegram", "user123", "thread1")
        # Python tries str.__eq__ first, which returns NotImplemented for
        # non-str, then Python tries key.__eq__. But since str doesn't know
        # about SessionKey, this direction relies on reflected __eq__.
        assert "telegram:user123:thread1" == key

    def test_not_equal_to_different_string(self):
        key = SessionKey.build("telegram", "user123", "thread1")
        assert key != "cli:user123:thread1"

    def test_not_equal_to_unrelated_type(self):
        key = SessionKey.build("telegram", "user1")
        assert key != 42
        assert key != None  # noqa: E711
        assert key != ["telegram", "user1"]


@pytest.mark.unit
class TestSessionKeyHash:
    """Tests for __hash__ consistency."""

    def test_equal_keys_same_hash(self):
        key1 = SessionKey.build("telegram", "user1", "thread1")
        key2 = SessionKey.build("telegram", "user1", "thread1")
        assert hash(key1) == hash(key2)

    def test_different_keys_likely_different_hash(self):
        key1 = SessionKey.build("telegram", "user1", "thread1")
        key2 = SessionKey.build("cli", "user2", "thread2")
        # Not guaranteed, but extremely unlikely to collide
        assert hash(key1) != hash(key2)

    def test_usable_as_dict_key(self):
        key = SessionKey.build("telegram", "user1", "thread1")
        d = {key: "value"}
        assert d[key] == "value"

    def test_usable_in_set(self):
        key1 = SessionKey.build("telegram", "user1", "thread1")
        key2 = SessionKey.build("telegram", "user1", "thread1")
        key3 = SessionKey.build("cli", "user2")
        s = {key1, key2, key3}
        assert len(s) == 2

    def test_hash_matches_raw_string_hash(self):
        key = SessionKey.build("telegram", "user1", "thread1")
        assert hash(key) == hash("telegram:user1:thread1")


@pytest.mark.unit
class TestSessionKeyStr:
    """Tests for __str__ returning raw."""

    def test_str_returns_raw(self):
        key = SessionKey.build("telegram", "user123", "thread1")
        assert str(key) == "telegram:user123:thread1"

    def test_str_default_thread(self):
        key = SessionKey.build("cli", "user456")
        assert str(key) == "cli:user456:default"

    def test_fstring_uses_str(self):
        key = SessionKey.build("api", "u1", "t1")
        assert f"session={key}" == "session=api:u1:t1"


@pytest.mark.unit
class TestSessionKeyFrozen:
    """Tests that SessionKey is immutable (frozen dataclass)."""

    def test_cannot_set_channel(self):
        key = SessionKey.build("telegram", "user1")
        with pytest.raises(AttributeError):
            key.channel = "cli"  # type: ignore[misc]

    def test_cannot_set_user_id(self):
        key = SessionKey.build("telegram", "user1")
        with pytest.raises(AttributeError):
            key.user_id = "user2"  # type: ignore[misc]

    def test_cannot_set_thread_id(self):
        key = SessionKey.build("telegram", "user1")
        with pytest.raises(AttributeError):
            key.thread_id = "other"  # type: ignore[misc]


@pytest.mark.unit
class TestJottyMessageSessionKey:
    """Tests for JottyMessage.session_key property."""

    def test_session_key_from_structured_session_id(self):
        msg = JottyMessage(
            content="hello",
            interface=InterfaceType.TELEGRAM,
            user_id="u1",
            session_id="telegram:user123:thread1",
        )
        sk = msg.session_key
        assert isinstance(sk, SessionKey)
        assert sk.channel == "telegram"
        assert sk.user_id == "user123"
        assert sk.thread_id == "thread1"

    def test_session_key_from_two_part_session_id(self):
        msg = JottyMessage(
            content="hello",
            interface=InterfaceType.CLI,
            user_id="u1",
            session_id="cli:user456",
        )
        sk = msg.session_key
        assert sk.channel == "cli"
        assert sk.user_id == "user456"
        assert sk.thread_id == "default"

    def test_session_key_from_legacy_session_id(self):
        msg = JottyMessage(
            content="hello",
            interface=InterfaceType.CLI,
            user_id="u1",
            session_id="tg_12345",
        )
        sk = msg.session_key
        assert sk.channel == "unknown"
        assert sk.user_id == "tg_12345"
        assert sk.thread_id == "default"

    def test_session_key_from_empty_session_id(self):
        msg = JottyMessage(
            content="hello",
            interface=InterfaceType.WEB,
            user_id="u1",
            session_id="",
        )
        sk = msg.session_key
        assert sk.channel == "unknown"
        assert sk.user_id == "unknown"
        assert sk.thread_id == "default"

    def test_session_key_raw_matches_session_id(self):
        session_id = "sdk:user789:sess42"
        msg = JottyMessage(
            content="test",
            interface=InterfaceType.SDK,
            user_id="u1",
            session_id=session_id,
        )
        assert msg.session_key.raw == session_id
        assert str(msg.session_key) == session_id
