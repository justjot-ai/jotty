"""
PlanMyInvesting API Client
==========================

Shared HTTP client for all PMI skill packs.
Extends BaseAPIClient with Bearer token auth against the PMI REST API.
"""

import logging
import os
from typing import Any, Dict, Optional

from Jotty.core.infrastructure.utils.api_client import BaseAPIClient
from Jotty.core.infrastructure.utils.env_loader import load_jotty_env

load_jotty_env()
logger = logging.getLogger(__name__)


class PlanMyInvestingClient(BaseAPIClient):
    """PlanMyInvesting REST API client with Bearer token auth."""

    AUTH_PREFIX = "Bearer"
    TOKEN_ENV_VAR = "PMI_API_TOKEN"
    TOKEN_CONFIG_PATH = ""
    CONTENT_TYPE = "application/json"
    DEFAULT_TIMEOUT = 30

    def __init__(self, token: Optional[str] = None, base_url: Optional[str] = None):
        super().__init__(token or os.getenv("PMI_API_TOKEN"))
        self.BASE_URL = (base_url or os.getenv("PMI_API_URL", "http://localhost:5000")).rstrip("/")

    def get(
        self, endpoint: str, params: Optional[Dict] = None, timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """Convenience GET request."""
        return self._make_request(endpoint, method="GET", params=params, timeout=timeout)

    def post(
        self, endpoint: str, data: Optional[Dict] = None, timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """Convenience POST request."""
        return self._make_request(endpoint, method="POST", json_data=data, timeout=timeout)

    def put(
        self, endpoint: str, data: Optional[Dict] = None, timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """Convenience PUT request."""
        return self._make_request(endpoint, method="PUT", json_data=data, timeout=timeout)

    def delete(self, endpoint: str, timeout: Optional[int] = None) -> Dict[str, Any]:
        """Convenience DELETE request."""
        return self._make_request(endpoint, method="DELETE", timeout=timeout)


__all__ = ["PlanMyInvestingClient"]
