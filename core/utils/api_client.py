"""
Base API Client
===============

Reusable base class for HTTP API clients.
Reduces duplication across Discord, Slack, WhatsApp, etc.

Usage:
    class DiscordClient(BaseAPIClient):
        BASE_URL = "https://discord.com/api/v10"
        AUTH_PREFIX = "Bot"
        TOKEN_ENV_VAR = "DISCORD_BOT_TOKEN"
        TOKEN_CONFIG_PATH = ".config/discord/token"
"""

import os
import logging
import requests
from pathlib import Path
from typing import Dict, Any, Optional
from abc import ABC

logger = logging.getLogger(__name__)


class BaseAPIClient(ABC):
    """
    Base class for HTTP API clients.

    Subclasses should define:
    - BASE_URL: API base URL
    - AUTH_PREFIX: Authorization prefix ("Bearer", "Bot", etc.)
    - TOKEN_ENV_VAR: Environment variable name for token
    - TOKEN_CONFIG_PATH: Optional config file path (relative to home)
    """

    BASE_URL: str = ""
    AUTH_PREFIX: str = "Bearer"
    TOKEN_ENV_VAR: str = ""
    TOKEN_CONFIG_PATH: str = ""
    CONTENT_TYPE: str = "application/json"
    DEFAULT_TIMEOUT: int = 30

    def __init__(self, token: Optional[str] = None):
        """
        Initialize API client.

        Args:
            token: Optional token. If not provided, loads from env/config.
        """
        self.token = token or self._get_token()

    def _get_token(self) -> Optional[str]:
        """Get token from environment or config file."""
        # Try environment variable first
        if self.TOKEN_ENV_VAR:
            token = os.getenv(self.TOKEN_ENV_VAR)
            if token:
                return token

        # Try config file
        if self.TOKEN_CONFIG_PATH:
            config_path = Path.home() / self.TOKEN_CONFIG_PATH
            if config_path.exists():
                return config_path.read_text().strip()

        return None

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        headers = {"Content-Type": self.CONTENT_TYPE}
        if self.token:
            headers["Authorization"] = f"{self.AUTH_PREFIX} {self.token}"
        return headers

    def _build_url(self, endpoint: str) -> str:
        """Build full URL from endpoint."""
        if endpoint.startswith("http"):
            return endpoint
        base = self.BASE_URL.rstrip("/")
        endpoint = endpoint.lstrip("/")
        return f"{base}/{endpoint}"

    def _make_request(
        self,
        endpoint: str,
        method: str = "POST",
        json_data: Optional[Dict] = None,
        params: Optional[Dict] = None,
        files: Optional[Dict] = None,
        data: Optional[Dict] = None,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Make HTTP request to API.

        Args:
            endpoint: API endpoint
            method: HTTP method (GET, POST, PUT, DELETE)
            json_data: JSON body data
            params: Query parameters
            files: Files for multipart upload
            data: Form data
            timeout: Request timeout in seconds

        Returns:
            Dict with 'success' key and response data or 'error'
        """
        url = self._build_url(endpoint)
        timeout = timeout or self.DEFAULT_TIMEOUT

        try:
            if files:
                # For file uploads, don't include Content-Type header
                headers = {"Authorization": f"{self.AUTH_PREFIX} {self.token}"}
                response = requests.post(
                    url, headers=headers, files=files, data=data, timeout=timeout
                )
            elif method.upper() == "GET":
                response = requests.get(
                    url, headers=self._get_headers(), params=params or json_data, timeout=timeout
                )
            elif method.upper() == "POST":
                response = requests.post(
                    url, headers=self._get_headers(), json=json_data, timeout=timeout
                )
            elif method.upper() == "PUT":
                response = requests.put(
                    url, headers=self._get_headers(), json=json_data, timeout=timeout
                )
            elif method.upper() == "DELETE":
                response = requests.delete(
                    url, headers=self._get_headers(), timeout=timeout
                )
            elif method.upper() == "PATCH":
                response = requests.patch(
                    url, headers=self._get_headers(), json=json_data, timeout=timeout
                )
            else:
                return {"success": False, "error": f"Unsupported HTTP method: {method}"}

            return self._handle_response(response)

        except requests.exceptions.Timeout:
            return {"success": False, "error": f"Request timeout after {timeout}s"}
        except requests.exceptions.ConnectionError as e:
            return {"success": False, "error": f"Connection error: {e}"}
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
        """
        Handle API response.

        Override this method for custom response handling.
        """
        # No content success
        if response.status_code == 204:
            return {"success": True}

        # Rate limiting
        if response.status_code == 429:
            try:
                data = response.json()
                retry_after = data.get("retry_after", 1)
                return {
                    "success": False,
                    "error": f"Rate limited. Retry after {retry_after}s",
                    "retry_after": retry_after
                }
            except Exception:
                return {"success": False, "error": "Rate limited"}

        # Error responses
        if response.status_code >= 400:
            try:
                error_data = response.json()
                error_msg = (
                    error_data.get("error", {}).get("message") or
                    error_data.get("message") or
                    error_data.get("error") or
                    f"HTTP {response.status_code}"
                )
                return {
                    "success": False,
                    "error": error_msg,
                    "status_code": response.status_code
                }
            except Exception:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text[:200]}"
                }

        # Success
        try:
            result = response.json()
            # Handle APIs that return {"ok": false} style errors
            if isinstance(result, dict) and result.get("ok") is False:
                return {
                    "success": False,
                    "error": result.get("error", "Unknown error")
                }
            return {"success": True, **result}
        except Exception:
            return {"success": True, "text": response.text}

    @property
    def is_configured(self) -> bool:
        """Check if client has valid token."""
        return bool(self.token)

    def require_token(self) -> Dict[str, Any]:
        """Return error dict if token not configured."""
        if not self.token:
            return {
                "success": False,
                "error": f"Token required. Set {self.TOKEN_ENV_VAR} env var or provide token parameter"
            }
        return None
