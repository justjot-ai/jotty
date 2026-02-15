#!/usr/bin/env python3
"""
Jotty CLI - Main Entry Point
=============================

Clean entry point for Jotty command-line interface.

This CLI is an APPLICATION built on the Jotty SDK, following
clean architecture principles used by world-class companies
(Google, Amazon, Stripe, GitHub, etc.).

Architecture:
    CLI (apps/cli/) → SDK (sdk/) → Core (core/)

Usage:
    python -m Jotty.apps.cli
    python -m Jotty.apps.cli.main
"""

import sys


def main():
    """Main entry point for Jotty CLI."""
    try:
        from Jotty.apps.cli.__main__ import main as cli_main

        # Run the CLI
        cli_main()

    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error starting Jotty CLI: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
