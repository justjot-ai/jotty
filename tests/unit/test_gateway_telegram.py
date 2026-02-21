"""
Tests for Telegram webhook: channel_post extraction, dedupe, TTL cleanup.
"""

import asyncio
import json
import time
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

import httpx


def _make_gateway():
    """Create a UnifiedGateway with mocked dependencies."""
    with patch("Jotty.apps.cli.gateway.server.FASTAPI_AVAILABLE", True):
        from Jotty.apps.cli.gateway.server import UnifiedGateway

        gw = UnifiedGateway(enable_trust=False)
        gw.router = MagicMock()
        gw.router.handle_message = AsyncMock(return_value="ok")
        gw.router.active_sessions = 0
        gw.router.stats = {}
        gw.router._responders = {}
        return gw


def _make_client(app):
    """Create an httpx AsyncClient for testing against the ASGI app."""
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.unit
class TestTelegramChannelPost:
    """Test that channel_post and edited_channel_post are handled."""

    @pytest.mark.asyncio
    async def test_channel_post_extracted(self):
        gw = _make_gateway()
        app = gw.create_app()

        async with _make_client(app) as client:
            payload = {
                "update_id": 1,
                "channel_post": {
                    "message_id": 42,
                    "chat": {"id": -1001234, "title": "My Channel", "type": "channel"},
                    "text": "Hello from channel",
                    "date": 1708000000,
                },
            }
            resp = await client.post("/webhook/telegram", json=payload)
            assert resp.status_code == 200
            assert resp.json()["ok"] is True

        # Wait for async task
        await asyncio.sleep(0.1)
        gw.router.handle_message.assert_called()
        event = gw.router.handle_message.call_args[0][0]
        assert event.content == "Hello from channel"
        assert event.channel_id == str(-1001234)
        # channel_post has no "from", should fall back to chat info
        assert event.user_id == str(-1001234)
        assert event.user_name == "My Channel"

    @pytest.mark.asyncio
    async def test_edited_channel_post_extracted(self):
        gw = _make_gateway()
        app = gw.create_app()

        async with _make_client(app) as client:
            payload = {
                "update_id": 2,
                "edited_channel_post": {
                    "message_id": 43,
                    "chat": {"id": -1001234, "title": "Edits", "type": "channel"},
                    "text": "Edited text",
                    "date": 1708000000,
                },
            }
            resp = await client.post("/webhook/telegram", json=payload)
            assert resp.status_code == 200
            assert resp.json()["ok"] is True

    @pytest.mark.asyncio
    async def test_regular_message_still_works(self):
        gw = _make_gateway()
        app = gw.create_app()

        async with _make_client(app) as client:
            payload = {
                "update_id": 3,
                "message": {
                    "message_id": 99,
                    "chat": {"id": 555, "type": "private"},
                    "from": {"id": 111, "first_name": "Alice"},
                    "text": "Hi bot",
                    "date": 1708000000,
                },
            }
            resp = await client.post("/webhook/telegram", json=payload)
            assert resp.status_code == 200
            assert resp.json()["ok"] is True

        await asyncio.sleep(0.1)
        gw.router.handle_message.assert_called()
        event = gw.router.handle_message.call_args[0][0]
        assert event.content == "Hi bot"
        assert event.user_id == "111"
        assert event.user_name == "Alice"


@pytest.mark.unit
class TestTelegramDedupe:
    """Test message deduplication."""

    @pytest.mark.asyncio
    async def test_duplicate_message_skipped(self):
        gw = _make_gateway()
        app = gw.create_app()

        async with _make_client(app) as client:
            payload = {
                "update_id": 10,
                "message": {
                    "message_id": 100,
                    "chat": {"id": 555, "type": "private"},
                    "from": {"id": 111, "first_name": "Alice"},
                    "text": "Hello",
                    "date": 1708000000,
                },
            }

            # First request — processed
            resp1 = await client.post("/webhook/telegram", json=payload)
            assert resp1.json()["ok"] is True
            await asyncio.sleep(0.1)
            assert gw.router.handle_message.call_count == 1

            # Second identical request — deduplicated
            resp2 = await client.post("/webhook/telegram", json=payload)
            assert resp2.json()["ok"] is True
            await asyncio.sleep(0.1)
            # Should still be 1 call (second was deduplicated)
            assert gw.router.handle_message.call_count == 1

    @pytest.mark.asyncio
    async def test_empty_update_returns_ok(self):
        gw = _make_gateway()
        app = gw.create_app()

        async with _make_client(app) as client:
            payload = {"update_id": 20}
            resp = await client.post("/webhook/telegram", json=payload)
            assert resp.status_code == 200
            assert resp.json()["ok"] is True

    @pytest.mark.asyncio
    async def test_no_text_message_returns_ok(self):
        gw = _make_gateway()
        app = gw.create_app()

        async with _make_client(app) as client:
            payload = {
                "update_id": 21,
                "message": {
                    "message_id": 200,
                    "chat": {"id": 555, "type": "private"},
                    "from": {"id": 111, "first_name": "Alice"},
                    "photo": [{"file_id": "abc"}],
                    "date": 1708000000,
                },
            }
            resp = await client.post("/webhook/telegram", json=payload)
            assert resp.status_code == 200
            assert resp.json()["ok"] is True

        # No text = no processing
        gw.router.handle_message.assert_not_called()


@pytest.mark.unit
class TestTelegramMissingFrom:
    """Test fallback when 'from' field is missing (channel posts)."""

    @pytest.mark.asyncio
    async def test_missing_from_uses_chat_info(self):
        gw = _make_gateway()
        app = gw.create_app()

        async with _make_client(app) as client:
            payload = {
                "update_id": 30,
                "channel_post": {
                    "message_id": 300,
                    "chat": {"id": -1009999, "title": "Test Channel", "type": "channel"},
                    "text": "Channel message",
                    "date": 1708000000,
                },
            }
            resp = await client.post("/webhook/telegram", json=payload)
            assert resp.json()["ok"] is True

        await asyncio.sleep(0.1)
        event = gw.router.handle_message.call_args[0][0]
        assert event.user_id == str(-1009999)  # Falls back to chat.id
        assert event.user_name == "Test Channel"  # Falls back to chat.title


@pytest.mark.unit
class TestSlackDedupe:
    """Test Slack webhook deduplication."""

    @pytest.mark.asyncio
    async def test_slack_duplicate_skipped(self):
        gw = _make_gateway()
        app = gw.create_app()

        async with _make_client(app) as client:
            payload = {
                "type": "event_callback",
                "event": {
                    "type": "message",
                    "channel": "C123",
                    "user": "U456",
                    "text": "Hello Slack",
                    "ts": "1708000000.000001",
                },
            }

            r1 = await client.post("/webhook/slack", json=payload)
            assert r1.json()["ok"] is True
            await asyncio.sleep(0.1)
            assert gw.router.handle_message.call_count == 1

            # Same ts → deduplicated
            r2 = await client.post("/webhook/slack", json=payload)
            assert r2.json()["ok"] is True
            await asyncio.sleep(0.1)
            assert gw.router.handle_message.call_count == 1

    @pytest.mark.asyncio
    async def test_slack_different_ts_not_deduped(self):
        gw = _make_gateway()
        app = gw.create_app()

        async with _make_client(app) as client:
            payload1 = {
                "type": "event_callback",
                "event": {
                    "type": "message",
                    "channel": "C123",
                    "user": "U456",
                    "text": "First",
                    "ts": "1708000000.000001",
                },
            }
            payload2 = {
                "type": "event_callback",
                "event": {
                    "type": "message",
                    "channel": "C123",
                    "user": "U456",
                    "text": "Second",
                    "ts": "1708000000.000002",
                },
            }

            await client.post("/webhook/slack", json=payload1)
            await asyncio.sleep(0.1)
            await client.post("/webhook/slack", json=payload2)
            await asyncio.sleep(0.1)
            assert gw.router.handle_message.call_count == 2


@pytest.mark.unit
class TestDiscordDedupe:
    """Test Discord webhook deduplication."""

    @pytest.mark.asyncio
    async def test_discord_duplicate_skipped(self):
        gw = _make_gateway()
        app = gw.create_app()

        async with _make_client(app) as client:
            payload = {
                "type": 2,
                "id": "disc-123",
                "channel_id": "ch-1",
                "member": {"user": {"id": "u1", "username": "Bob"}},
                "data": {"options": [{"value": "test command"}]},
            }

            r1 = await client.post("/webhook/discord", json=payload)
            assert r1.status_code == 200
            await asyncio.sleep(0.1)
            assert gw.router.handle_message.call_count == 1

            # Same interaction id → deduplicated
            r2 = await client.post("/webhook/discord", json=payload)
            assert r2.status_code == 200
            await asyncio.sleep(0.1)
            assert gw.router.handle_message.call_count == 1


@pytest.mark.unit
class TestWhatsAppDedupe:
    """Test WhatsApp webhook deduplication."""

    @pytest.mark.asyncio
    async def test_whatsapp_duplicate_skipped(self):
        gw = _make_gateway()
        app = gw.create_app()

        async with _make_client(app) as client:
            payload = {
                "entry": [
                    {
                        "changes": [
                            {
                                "value": {
                                    "messages": [
                                        {
                                            "type": "text",
                                            "from": "1234567890",
                                            "id": "wa-msg-1",
                                            "text": {"body": "Hello WhatsApp"},
                                        }
                                    ],
                                    "contacts": [{"profile": {"name": "Test User"}}],
                                },
                            }
                        ],
                    }
                ],
            }

            r1 = await client.post("/webhook/whatsapp", json=payload)
            assert r1.json()["status"] == "ok"
            await asyncio.sleep(0.1)
            assert gw.router.handle_message.call_count == 1

            # Same message id → deduplicated
            r2 = await client.post("/webhook/whatsapp", json=payload)
            assert r2.json()["status"] == "ok"
            await asyncio.sleep(0.1)
            assert gw.router.handle_message.call_count == 1

    @pytest.mark.asyncio
    async def test_whatsapp_different_msg_not_deduped(self):
        gw = _make_gateway()
        app = gw.create_app()

        async with _make_client(app) as client:

            def _make_wa(msg_id, text):
                return {
                    "entry": [
                        {
                            "changes": [
                                {
                                    "value": {
                                        "messages": [
                                            {
                                                "type": "text",
                                                "from": "1234567890",
                                                "id": msg_id,
                                                "text": {"body": text},
                                            }
                                        ],
                                        "contacts": [{"profile": {"name": "Test"}}],
                                    },
                                }
                            ],
                        }
                    ],
                }

            await client.post("/webhook/whatsapp", json=_make_wa("wa-1", "First"))
            await asyncio.sleep(0.1)
            await client.post("/webhook/whatsapp", json=_make_wa("wa-2", "Second"))
            await asyncio.sleep(0.1)
            assert gw.router.handle_message.call_count == 2
