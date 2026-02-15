"""
Telegram Bot Entry Point
========================

Standalone entry point for running Jotty Telegram bot.

Usage:
    python -m Jotty.telegram
    python -m Jotty.telegram --debug
"""

import argparse
import logging
import sys


def setup_logging(debug: bool = False):
    """Configure logging."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run Jotty Telegram Bot")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug logging")
    parser.add_argument("--token", help="Telegram bot token (overrides .env)")
    parser.add_argument("--chat-ids", help="Comma-separated allowed chat IDs")

    args = parser.parse_args()

    setup_logging(args.debug)

    logger = logging.getLogger(__name__)

    # Parse allowed chat IDs
    allowed_chat_ids = None
    if args.chat_ids:
        allowed_chat_ids = [int(x.strip()) for x in args.chat_ids.split(",")]

    try:
        from .bot import TelegramBotHandler

        bot = TelegramBotHandler(token=args.token, allowed_chat_ids=allowed_chat_ids)

        logger.info("Starting Jotty Telegram Bot...")
        bot.run()

    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")
        sys.exit(0)
    except ImportError as e:
        logger.error(f"Missing dependency: {e}\n" "Install with: pip install python-telegram-bot")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to start bot: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
