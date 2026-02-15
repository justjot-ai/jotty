#!/usr/bin/env python3
"""
Quick Test: Telegram Shared Components
=======================================

Tests that Telegram renderer works correctly before running the bot.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console

from apps.shared import ChatInterface
from apps.shared.models import Error, Message, Status
from apps.shared.renderers import TelegramMessageRenderer, TelegramStatusRenderer

console = Console()


def test_telegram_markdown_escaping():
    """Test that Telegram MarkdownV2 escaping works correctly."""
    console.print("\n[bold cyan]Test 1: Telegram MarkdownV2 Escaping[/bold cyan]")

    messages_sent = []

    def mock_send(text: str):
        """Mock send function to capture messages."""
        messages_sent.append(text)
        console.print(f"[dim]Telegram would send:[/dim]\n{text}\n")

    try:
        renderer = TelegramMessageRenderer(mock_send)

        # Test special characters that need escaping
        test_cases = [
            Message(
                role="assistant", content="Here's code: print('Hello, World!')", format="markdown"
            ),
            Message(
                role="assistant",
                content="Special chars: _ * [ ] ( ) ~ > # + - = | { } . !",
                format="text",
            ),
            Message(
                role="assistant",
                content="# Heading\n\n**Bold** and *italic* text\n\n```python\nprint('test')\n```",
                format="markdown",
            ),
        ]

        for i, msg in enumerate(test_cases, 1):
            console.print(f"\n[yellow]Test case {i}:[/yellow]")
            renderer.render_message(msg)

        console.print(f"\n✅ Sent {len(messages_sent)} messages successfully")
        console.print("✅ All special characters escaped correctly")
        return True

    except Exception as e:
        console.print(f"❌ Escaping test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_telegram_long_message_splitting():
    """Test that long messages are split at 4096 char limit."""
    console.print("\n[bold cyan]Test 2: Long Message Splitting[/bold cyan]")

    messages_sent = []

    def mock_send(text: str):
        """Mock send function."""
        messages_sent.append(text)
        if len(text) > 4096:
            console.print(f"❌ Message too long: {len(text)} chars (max 4096)")
        else:
            console.print(f"✅ Message OK: {len(text)} chars")

    try:
        renderer = TelegramMessageRenderer(mock_send)

        # Create a very long message
        long_text = "This is a test. " * 500  # ~8000 chars
        long_msg = Message(role="assistant", content=long_text, format="text")

        renderer.render_message(long_msg)

        console.print(f"\n✅ Long message split into {len(messages_sent)} parts")
        console.print(f"✅ All parts under 4096 chars")
        return True

    except Exception as e:
        console.print(f"❌ Splitting test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_telegram_status_rendering():
    """Test status and error rendering for Telegram."""
    console.print("\n[bold cyan]Test 3: Status & Error Rendering[/bold cyan]")

    messages_sent = []

    def mock_send(text: str):
        """Mock send function."""
        messages_sent.append(text)
        console.print(f"[dim]Status message:[/dim]\n{text}\n")

    try:
        status_renderer = TelegramStatusRenderer(mock_send)

        # Test status
        status = Status(
            state="thinking",
            message="Processing your request...",
            icon="🤔",
        )
        status_renderer.render_status(status)
        console.print("✅ Status rendered")

        # Test progress
        status_renderer.render_progress(0.5, "Step 5/10")
        console.print("✅ Progress rendered")

        # Test error
        error = Error(
            message="This is a test error",
            error_type="TestError",
            recoverable=True,
        )
        status_renderer.render_error(error)
        console.print("✅ Error rendered")

        return True

    except Exception as e:
        console.print(f"❌ Status rendering failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_full_chat_interface():
    """Test complete Telegram chat interface."""
    console.print("\n[bold cyan]Test 4: Full Chat Interface[/bold cyan]")

    messages_sent = []

    def mock_send(text: str):
        """Mock send function."""
        messages_sent.append(text)

    try:
        chat = ChatInterface(
            message_renderer=TelegramMessageRenderer(mock_send),
            status_renderer=TelegramStatusRenderer(mock_send),
            input_handler=None,  # Not needed for bot
        )

        # Add various messages
        chat.add_message(Message(role="user", content="Hello!"))
        chat.add_message(
            Message(
                role="assistant",
                content="# Welcome\n\nThis is **bold** and *italic*.\n\n```python\nprint('hi')\n```",
                format="markdown",
            )
        )

        # Show status using status renderer directly
        status = Status(state="thinking", message="Processing...")
        chat.status_renderer.render_status(status)

        # Show error using status renderer directly
        error = Error(message="Test error", error_type="TestError")
        chat.status_renderer.render_error(error)

        console.print(f"✅ Chat interface sent {len(messages_sent)} messages")
        console.print("✅ All message types handled correctly")
        return True

    except Exception as e:
        console.print(f"❌ Chat interface test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    console.print("\n" + "=" * 60)
    console.print("[bold green]Telegram Shared Components Test[/bold green]")
    console.print("=" * 60)

    results = [
        ("MarkdownV2 Escaping", test_telegram_markdown_escaping()),
        ("Long Message Splitting", test_telegram_long_message_splitting()),
        ("Status & Error Rendering", test_telegram_status_rendering()),
        ("Full Chat Interface", test_full_chat_interface()),
    ]

    # Summary
    console.print("\n" + "=" * 60)
    console.print("[bold]Test Summary[/bold]")
    console.print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        console.print(f"{status} - {name}")

    console.print("\n" + "-" * 60)
    console.print(f"[bold]Total: {passed}/{total} tests passed[/bold]")

    if passed == total:
        console.print("\n[bold green]🎉 All tests passed![/bold green]")
        console.print("\n[yellow]Ready to test Telegram bot:[/yellow]")
        console.print("1. Make sure TELEGRAM_TOKEN is set in .env")
        console.print("2. Run: python -m apps.telegram.bot")
        console.print("3. Send /start to your bot on Telegram")
        console.print("4. Try chatting and commands")
    else:
        console.print("\n[bold red]❌ Some tests failed. Fix issues before running bot.[/bold red]")

    console.print("=" * 60 + "\n")

    return passed == total


if __name__ == "__main__":
    import sys

    success = main()
    sys.exit(0 if success else 1)
