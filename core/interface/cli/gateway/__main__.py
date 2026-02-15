"""
Jotty Gateway Entry Point
=========================

Run with: python -m Jotty.cli.gateway [--port 8766] [--host 0.0.0.0]

For jotty.justjot.ai deployment on cmd.dev.
"""

from . import main

if __name__ == "__main__":
    main()
