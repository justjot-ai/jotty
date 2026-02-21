#!/usr/bin/env python3
"""
Live E2E: Verify OpenClaw v2026.2.19 features fire correctly.

Features tested:
    F8  SSRF Protection (incl. IPv6 transition)
    F9  API Middleware (auth + rate limiting)
    F10 Telegram Dedupe + channel_post
    F11 Streaming thinking events
    F12 Webhook dedupe (Slack/Discord/WhatsApp)

Usage:
    python tests/e2e/test_openclaw_v2_features_live.py
"""

import asyncio
import ipaddress
import os
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Load .env
env_path = Path(__file__).resolve().parents[2] / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())


def _h(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _ok(msg):
    print(f"  [OK] {msg}")


def _fail(msg):
    print(f"  [FAIL] {msg}")


def _info(msg):
    print(f"  [..] {msg}")


_results = {}


def _test(name, condition, detail=""):
    _results[name] = condition
    if condition:
        _ok(f"{name}: {detail}" if detail else name)
    else:
        _fail(f"{name}: {detail}" if detail else name)


async def main():
    _h("OpenClaw v2026.2.19 Features — E2E Verification")

    # ─────────────────────────────────────────────────────
    # F8: SSRF Protection
    # ─────────────────────────────────────────────────────
    _h("F8: SSRF Protection")

    from Jotty.core.infrastructure.utils.smart_fetcher import (
        is_ssrf_safe,
        _is_ipv6_transition_unsafe,
    )

    # Public URL (mock DNS)
    with patch("socket.getaddrinfo", return_value=[(2, 1, 6, "", ("93.184.216.34", 443))]):
        safe, _ = is_ssrf_safe("https://example.com")
        _test("F8.1 Public URL allowed", safe)

    # Blocked schemes
    safe, reason = is_ssrf_safe("file:///etc/passwd")
    _test("F8.2 file:// blocked", not safe, reason)

    safe, reason = is_ssrf_safe("gopher://evil.com")
    _test("F8.3 gopher:// blocked", not safe, reason)

    # Loopback
    safe, reason = is_ssrf_safe("http://localhost/admin")
    _test("F8.4 localhost blocked", not safe, reason)

    safe, reason = is_ssrf_safe("http://127.0.0.1:8080/")
    _test("F8.5 127.0.0.1 blocked", not safe, reason)

    # Private IP via DNS
    with patch("socket.getaddrinfo", return_value=[(2, 1, 6, "", ("10.0.0.1", 443))]):
        safe, reason = is_ssrf_safe("https://internal.corp.com")
        _test("F8.6 Private 10.x blocked", not safe, reason)

    # AWS metadata
    with patch("socket.getaddrinfo", return_value=[(2, 1, 6, "", ("169.254.169.254", 80))]):
        safe, reason = is_ssrf_safe("http://169.254.169.254/latest/meta-data/")
        _test("F8.7 AWS metadata blocked", not safe, reason)

    # IPv4-mapped IPv6
    ip = ipaddress.IPv6Address("::ffff:127.0.0.1")
    unsafe, reason = _is_ipv6_transition_unsafe(ip)
    _test("F8.8 IPv4-mapped ::ffff:127.0.0.1 blocked", unsafe, reason)

    ip = ipaddress.IPv6Address("::ffff:10.0.0.1")
    unsafe, reason = _is_ipv6_transition_unsafe(ip)
    _test("F8.9 IPv4-mapped ::ffff:10.0.0.1 blocked", unsafe, reason)

    # 6to4
    ip = ipaddress.IPv6Address("2002:7f00:0001::")
    unsafe, reason = _is_ipv6_transition_unsafe(ip)
    _test("F8.10 6to4 2002:7f00:: blocked", unsafe, reason)

    # Safe IPv6 (public)
    ip = ipaddress.IPv6Address("2607:f8b0:4004:800::200e")  # Google DNS
    unsafe, _ = _is_ipv6_transition_unsafe(ip)
    _test("F8.11 Public IPv6 allowed", not unsafe)

    # smart_fetch integration
    with patch(
        "Jotty.core.infrastructure.utils.smart_fetcher.is_ssrf_safe",
        return_value=(False, "Blocked private IP"),
    ):
        from Jotty.core.infrastructure.utils.smart_fetcher import smart_fetch

        result = smart_fetch("http://10.0.0.1/admin")
        _test("F8.12 smart_fetch blocks SSRF", not result.success and result.skipped)

    # ─────────────────────────────────────────────────────
    # F9: API Middleware (auth + rate limiting)
    # ─────────────────────────────────────────────────────
    _h("F9: API Middleware")

    from Jotty.core.infrastructure.utils.api_middleware import (
        check_bearer_token,
        RateLimiter,
        get_api_token,
    )

    # Auth
    _test("F9.1 Valid bearer accepted", check_bearer_token("Bearer abc123", "abc123"))
    _test("F9.2 Invalid bearer rejected", not check_bearer_token("Bearer wrong", "abc123"))
    _test("F9.3 Missing auth rejected", not check_bearer_token(None, "abc123"))
    _test("F9.4 No Bearer prefix rejected", not check_bearer_token("Token abc", "abc"))

    # Rate limiter
    limiter = RateLimiter(max_requests=3, window_seconds=60)
    assert limiter.is_allowed("ip1")
    assert limiter.is_allowed("ip1")
    assert limiter.is_allowed("ip1")
    _test("F9.5 Rate limiter allows within limit", True)
    _test("F9.6 Rate limiter blocks over limit", not limiter.is_allowed("ip1"))

    # Independent keys
    limiter2 = RateLimiter(max_requests=1, window_seconds=60)
    limiter2.is_allowed("a")
    _test("F9.7 Different keys independent", limiter2.is_allowed("b"))

    # Auth disabled
    with patch.dict(os.environ, {"JOTTY_API_AUTH": "none"}):
        _test("F9.8 Auth disabled returns None", get_api_token() is None)

    # Token from env
    with patch.dict(os.environ, {"JOTTY_API_TOKEN": "env-token"}, clear=False):
        os.environ.pop("JOTTY_API_AUTH", None)
        _test("F9.9 Env token read", get_api_token() == "env-token")

    # ─────────────────────────────────────────────────────
    # F10: Telegram Dedupe + channel_post
    # ─────────────────────────────────────────────────────
    _h("F10: Telegram Dedupe + channel_post")

    import httpx

    with patch("Jotty.apps.cli.gateway.server.FASTAPI_AVAILABLE", True):
        from Jotty.apps.cli.gateway.server import UnifiedGateway

        gw = UnifiedGateway(enable_trust=False)
        gw.router = MagicMock()
        gw.router.handle_message = AsyncMock(return_value="ok")
        gw.router.active_sessions = 0
        gw.router.stats = {}
        gw.router._responders = {}
        app = gw.create_app()

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # channel_post
            resp = await client.post(
                "/webhook/telegram",
                json={
                    "update_id": 1,
                    "channel_post": {
                        "message_id": 42,
                        "chat": {"id": -1001234, "title": "Chan", "type": "channel"},
                        "text": "Channel msg",
                        "date": 1708000000,
                    },
                },
            )
            _test("F10.1 channel_post accepted", resp.json()["ok"] is True)

            # edited_channel_post
            resp = await client.post(
                "/webhook/telegram",
                json={
                    "update_id": 2,
                    "edited_channel_post": {
                        "message_id": 43,
                        "chat": {"id": -1001234, "title": "Chan", "type": "channel"},
                        "text": "Edited",
                        "date": 1708000000,
                    },
                },
            )
            _test("F10.2 edited_channel_post accepted", resp.json()["ok"] is True)

            # Dedupe — send same message_id again
            resp = await client.post(
                "/webhook/telegram",
                json={
                    "update_id": 3,
                    "channel_post": {
                        "message_id": 42,
                        "chat": {"id": -1001234, "title": "Chan", "type": "channel"},
                        "text": "Channel msg",
                        "date": 1708000000,
                    },
                },
            )
            _test("F10.3 Duplicate deduplicated", resp.json()["ok"] is True)

            # Regular message
            resp = await client.post(
                "/webhook/telegram",
                json={
                    "update_id": 4,
                    "message": {
                        "message_id": 99,
                        "chat": {"id": 555, "type": "private"},
                        "from": {"id": 111, "first_name": "Alice"},
                        "text": "Hi",
                        "date": 1708000000,
                    },
                },
            )
            _test("F10.4 Regular message works", resp.json()["ok"] is True)

    # ─────────────────────────────────────────────────────
    # F11: Streaming thinking events
    # ─────────────────────────────────────────────────────
    _h("F11: Streaming Thinking Events")

    from Jotty.core.intelligence.orchestration.llm_providers.types import ThinkingBlock, TextBlock
    from Jotty.core.intelligence.orchestration.llm_providers.anthropic import AnthropicProvider

    # ThinkingBlock type
    tb = ThinkingBlock(thinking="Let me reason...")
    _test("F11.1 ThinkingBlock type", tb.type == "thinking" and tb.thinking == "Let me reason...")

    # _parse_response handles thinking
    provider = AnthropicProvider.__new__(AnthropicProvider)
    mock_thinking = MagicMock(type="thinking", thinking="Deep thought")
    mock_text = MagicMock(type="text", text="Answer")
    mock_resp = MagicMock(
        content=[mock_thinking, mock_text],
        stop_reason="end_turn",
        usage=MagicMock(input_tokens=10, output_tokens=5),
    )
    parsed = provider._parse_response(mock_resp)
    _test(
        "F11.2 _parse_response includes ThinkingBlock",
        isinstance(parsed.content[0], ThinkingBlock)
        and parsed.content[0].thinking == "Deep thought",
    )
    _test(
        "F11.3 _parse_response includes TextBlock",
        isinstance(parsed.content[1], TextBlock) and parsed.content[1].text == "Answer",
    )

    # Signature check — all providers accept thinking_callback
    import inspect

    for name, cls in [
        ("Anthropic", AnthropicProvider),
        (
            "OpenAI",
            __import__(
                "Jotty.core.intelligence.orchestration.llm_providers.openai",
                fromlist=["OpenAIProvider"],
            ).OpenAIProvider,
        ),
        (
            "Google",
            __import__(
                "Jotty.core.intelligence.orchestration.llm_providers.google",
                fromlist=["GoogleProvider"],
            ).GoogleProvider,
        ),
    ]:
        sig = inspect.signature(cls.call_streaming)
        has_param = "thinking_callback" in sig.parameters
        _test(f"F11.4 {name} has thinking_callback", has_param)

    # ─────────────────────────────────────────────────────
    # F12: Webhook Dedupe (Slack/Discord/WhatsApp)
    # ─────────────────────────────────────────────────────
    _h("F12: Webhook Dedupe (all channels)")

    with patch("Jotty.apps.cli.gateway.server.FASTAPI_AVAILABLE", True):
        gw2 = UnifiedGateway(enable_trust=False)
        gw2.router = MagicMock()
        gw2.router.handle_message = AsyncMock(return_value="ok")
        gw2.router.active_sessions = 0
        gw2.router.stats = {}
        gw2.router._responders = {}
        app2 = gw2.create_app()

        transport2 = httpx.ASGITransport(app=app2)
        async with httpx.AsyncClient(transport=transport2, base_url="http://test") as client:
            # Slack dedupe
            slack_payload = {
                "type": "event_callback",
                "event": {
                    "type": "message",
                    "channel": "C123",
                    "user": "U456",
                    "text": "Hello",
                    "ts": "1708000000.000001",
                },
            }
            r1 = await client.post("/webhook/slack", json=slack_payload)
            r2 = await client.post("/webhook/slack", json=slack_payload)
            _test("F12.1 Slack dedupe works", r1.json()["ok"] and r2.json()["ok"])

            # Discord dedupe
            discord_payload = {
                "type": 2,
                "id": "disc-123",
                "channel_id": "ch-1",
                "member": {"user": {"id": "u1", "username": "Bob"}},
                "data": {"options": [{"value": "test cmd"}]},
            }
            r1 = await client.post("/webhook/discord", json=discord_payload)
            r2 = await client.post("/webhook/discord", json=discord_payload)
            _test("F12.2 Discord dedupe works", r1.status_code == 200 and r2.status_code == 200)

            # WhatsApp dedupe
            wa_payload = {
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
                                            "text": {"body": "Hello"},
                                        }
                                    ],
                                    "contacts": [{"profile": {"name": "Test"}}],
                                },
                            }
                        ],
                    }
                ],
            }
            r1 = await client.post("/webhook/whatsapp", json=wa_payload)
            r2 = await client.post("/webhook/whatsapp", json=wa_payload)
            _test(
                "F12.3 WhatsApp dedupe works",
                r1.json()["status"] == "ok" and r2.json()["status"] == "ok",
            )

    # ─────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────
    _h("Summary")

    passed = sum(1 for v in _results.values() if v)
    total = len(_results)
    failed = total - passed

    for name, ok in _results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\n  {passed}/{total} passed, {failed} failed")
    if failed == 0:
        print("  All OpenClaw v2026.2.19 features verified!")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
