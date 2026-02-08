"""
FastAPI Web API
===============

REST and WebSocket API for Jotty Web UI.
Thin wiring layer — routes are in web/routes/, business logic in web/jotty_api.py.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def create_app() -> "FastAPI":
    """Create FastAPI application with all route groups."""
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
    from fastapi.middleware.cors import CORSMiddleware

    from .jotty_api import JottyAPI
    from .routes import (
        register_system_routes,
        register_chat_routes,
        register_sessions_routes,
        register_document_routes,
        register_agui_routes,
        register_tools_routes,
        register_sharing_routes,
        register_voice_routes,
    )

    app = FastAPI(
        title="Jotty API",
        description="Jotty AI Assistant API",
        version="1.0.0"
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API handler
    api = JottyAPI()

    # Register all route groups
    register_system_routes(app, api)
    register_chat_routes(app, api)
    register_sessions_routes(app, api)
    register_document_routes(app, api)
    register_agui_routes(app, api)
    register_tools_routes(app, api)
    register_sharing_routes(app, api)
    register_voice_routes(app, api)

    # Mount static files
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    return app


# Create app instance for uvicorn
app = create_app()
