"""
Jotty API Module
================

REST API for n8n and external automation.
"""

from .server import JottyAPIServer, start_api_server

__all__ = ["JottyAPIServer", "start_api_server"]
