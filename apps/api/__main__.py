"""
Web Server Entry Point
======================

Standalone entry point for running Jotty Web Server.

Usage:
    python -m Jotty.web
    python -m Jotty.web --port 8080
    python -m Jotty.web --host 0.0.0.0 --port 8080 --debug
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
    parser = argparse.ArgumentParser(description="Run Jotty Web Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument(
        "--port", "-p", type=int, default=8080, help="Port to bind to (default: 8080)"
    )
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug mode")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (development)")
    parser.add_argument(
        "--workers", "-w", type=int, default=1, help="Number of workers (default: 1)"
    )

    args = parser.parse_args()

    setup_logging(args.debug)

    logger = logging.getLogger(__name__)

    try:
        import uvicorn

        logger.info(f"Starting Jotty Web Server on http://{args.host}:{args.port}")
        logger.info("Press Ctrl+C to stop.")

        uvicorn.run(
            "Jotty.web.api:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers if not args.reload else 1,
            log_level="debug" if args.debug else "info",
        )

    except KeyboardInterrupt:
        logger.info("Server stopped by user.")
        sys.exit(0)
    except ImportError as e:
        logger.error(
            f"Missing dependency: {e}\n" "Install with: pip install fastapi uvicorn websockets"
        )
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
