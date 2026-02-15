#!/usr/bin/env python3
"""
Simple test to verify bot can handle commands
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


async def test_command_handling():
    """Test command handling logic."""
    from apps.shared import ChatInterface
    from apps.shared.renderers import TelegramMessageRenderer, TelegramStatusRenderer
    from apps.telegram.bot_migrated import TelegramBotMigrated

    messages_sent = []

    def mock_send(text: str):
        """Mock send function."""
        messages_sent.append(text)
        print(f"📤 Bot would send:\n{text}\n")

    # Create chat interface
    chat = ChatInterface(
        message_renderer=TelegramMessageRenderer(mock_send),
        status_renderer=TelegramStatusRenderer(mock_send),
        input_handler=None,
    )

    # Create bot
    bot = TelegramBotMigrated()

    # Test each command
    commands = [
        "/start",
        "/help",
        "/status",
        "/clear",
    ]

    for cmd in commands:
        print(f"\n{'='*60}")
        print(f"Testing command: {cmd}")
        print("=" * 60)
        messages_sent.clear()

        try:
            await bot._handle_command(cmd, chat, mock_send)
            if messages_sent:
                print(f"✅ Command '{cmd}' sent {len(messages_sent)} message(s)")
            else:
                print(f"⚠️ Command '{cmd}' sent no messages")
        except Exception as e:
            print(f"❌ Command '{cmd}' failed: {e}")
            import traceback

            traceback.print_exc()

    print(f"\n{'='*60}")
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_command_handling())
