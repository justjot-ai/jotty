#!/usr/bin/env python3
"""
Telegram long-polling bot that processes messages through Jotty.
No public URL needed — pulls updates from Telegram API directly.

Usage: python telegram_poll.py
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Load .env
env_path = Path(__file__).resolve().parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("telegram-poll")

TOKEN = os.environ.get("TELEGRAM_TOKEN")
if not TOKEN:
    print("TELEGRAM_TOKEN not set")
    sys.exit(1)

API = f"https://api.telegram.org/bot{TOKEN}"


async def send_message(chat_id: int, text: str, client: httpx.AsyncClient):
    """Send a message back to Telegram."""
    # Telegram max message length is 4096
    for i in range(0, len(text), 4096):
        chunk = text[i : i + 4096]
        await client.post(
            f"{API}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": chunk,
                "parse_mode": "Markdown",
            },
        )


_jotty = None


def _get_jotty():
    """Get or create singleton Jotty instance with DSPy LM configured."""
    global _jotty
    if _jotty is None:
        # Configure DSPy LM first — many skills depend on it
        try:
            from Jotty.core.infrastructure.foundation.unified_lm_provider import configure_dspy_lm

            configure_dspy_lm()
            log.info("DSPy LM configured")
        except Exception as e:
            log.warning(f"DSPy LM config failed: {e}")

        from Jotty import Jotty

        _jotty = Jotty(log_level="WARNING")
    return _jotty


async def process_message(text: str) -> str:
    """Process message through Jotty — auto tier selection."""
    try:
        jotty = _get_jotty()
        # Let Jotty auto-select tier based on intent classification
        result = await jotty.run(text)
        if result.success and result.output:
            return str(result.output)
        return f"(Jotty returned no output: success={result.success})"
    except Exception as e:
        log.error(f"Jotty error: {e}", exc_info=True)
        return f"Error: {e}"


async def main():
    log.info("Starting Telegram polling bot with Jotty...")
    log.info(f"Bot token: {TOKEN[:12]}...")

    offset = 0
    async with httpx.AsyncClient(timeout=60) as client:
        # Get bot info
        me = (await client.get(f"{API}/getMe")).json()
        bot_name = me["result"]["username"]
        log.info(f"Bot: @{bot_name}")

        while True:
            try:
                resp = await client.get(
                    f"{API}/getUpdates",
                    params={
                        "offset": offset,
                        "timeout": 30,
                        "allowed_updates": '["message","channel_post"]',
                    },
                )
                data = resp.json()

                if not data.get("ok"):
                    log.error(f"getUpdates error: {data}")
                    await asyncio.sleep(5)
                    continue

                for update in data.get("result", []):
                    offset = update["update_id"] + 1

                    # Extract message from update
                    msg = update.get("message") or update.get("channel_post")
                    if not msg or not msg.get("text"):
                        continue

                    chat_id = msg["chat"]["id"]
                    text = msg["text"]
                    user = msg.get("from", {})
                    user_name = user.get("first_name", msg["chat"].get("title", "Unknown"))

                    log.info(f"[{user_name}] {text}")

                    # Send "typing" indicator
                    await client.post(
                        f"{API}/sendChatAction",
                        json={
                            "chat_id": chat_id,
                            "action": "typing",
                        },
                    )

                    # Process through Jotty
                    response = await process_message(text)
                    log.info(f"[Bot] {response[:100]}...")

                    # Send response
                    await send_message(chat_id, response, client)

            except httpx.ReadTimeout:
                continue  # Normal for long polling
            except KeyboardInterrupt:
                log.info("Shutting down...")
                break
            except Exception as e:
                log.error(f"Poll error: {e}", exc_info=True)
                await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(main())
