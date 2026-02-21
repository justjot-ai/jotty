"""
Shared API Middleware: Auth + Rate Limiting
===========================================

DRY utilities used by both apps/api/ and apps/cli/gateway/.

Auth:
    - Bearer token from JOTTY_API_TOKEN env var, or auto-generated to ~/.jotty/api_token
    - Disabled when JOTTY_API_AUTH=none
    - Constant-time comparison via secrets.compare_digest

Rate Limiting:
    - Sliding-window in-memory rate limiter
    - Configurable max_requests / window_seconds
    - FastAPI middleware factory
"""

import logging
import os
import secrets
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# =========================================================================
# AUTH
# =========================================================================

_TOKEN_DIR = Path.home() / ".jotty"
_TOKEN_FILE = _TOKEN_DIR / "api_token"


def get_api_token() -> Optional[str]:
    """
    Get the API bearer token.

    Resolution order:
    1. JOTTY_API_AUTH=none → returns None (auth disabled)
    2. JOTTY_API_TOKEN env var → use that value
    3. ~/.jotty/api_token file → read from file
    4. Auto-generate → write to ~/.jotty/api_token and return

    Returns:
        Token string, or None if auth is explicitly disabled.
    """
    if os.getenv("JOTTY_API_AUTH", "").lower() == "none":
        return None

    # Env var takes priority
    env_token = os.getenv("JOTTY_API_TOKEN")
    if env_token:
        return env_token

    # Read from file
    if _TOKEN_FILE.exists():
        token = _TOKEN_FILE.read_text().strip()
        if token:
            return token

    # Auto-generate
    token = secrets.token_urlsafe(32)
    _TOKEN_DIR.mkdir(parents=True, exist_ok=True)
    _TOKEN_FILE.write_text(token)
    _TOKEN_FILE.chmod(0o600)
    logger.info(f"Generated API token at {_TOKEN_FILE}")
    return token


def check_bearer_token(authorization: Optional[str], expected: str) -> bool:
    """
    Validate an Authorization header against the expected token.

    Uses constant-time comparison to prevent timing attacks.

    Args:
        authorization: The Authorization header value (e.g. "Bearer abc123")
        expected: The expected token value

    Returns:
        True if valid, False otherwise
    """
    if not authorization:
        return False
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return False
    return secrets.compare_digest(parts[1], expected)


# =========================================================================
# RATE LIMITING
# =========================================================================


class RateLimiter:
    """
    Sliding-window in-memory rate limiter.

    Tracks request timestamps per key (typically client IP).
    Thread-safe for asyncio (single-threaded event loop).
    """

    def __init__(self, max_requests: int = 60, window_seconds: int = 60) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Dict[str, List[float]] = {}

    def is_allowed(self, key: str) -> bool:
        """
        Check if a request from `key` is within the rate limit.

        Automatically cleans expired entries for the key.

        Args:
            key: Rate limit key (e.g. client IP)

        Returns:
            True if allowed, False if rate limited
        """
        now = time.time()
        cutoff = now - self.window_seconds

        if key in self._requests:
            # Prune expired entries
            self._requests[key] = [t for t in self._requests[key] if t > cutoff]
        else:
            self._requests[key] = []

        if len(self._requests[key]) >= self.max_requests:
            return False

        self._requests[key].append(now)
        return True

    def cleanup(self) -> int:
        """
        Remove keys with no recent requests. Returns number of keys removed.
        Call periodically to prevent unbounded memory growth.
        """
        now = time.time()
        cutoff = now - self.window_seconds
        stale = [k for k, v in self._requests.items() if not v or v[-1] < cutoff]
        for k in stale:
            del self._requests[k]
        return len(stale)


# Shared default limiter: 60 req/min
_default_limiter = RateLimiter(max_requests=60, window_seconds=60)


def get_rate_limiter() -> RateLimiter:
    """Get the shared default RateLimiter (60 req/min)."""
    return _default_limiter


def make_rate_limit_middleware(
    limiter: Optional[RateLimiter] = None,
    path_prefixes: Optional[List[str]] = None,
) -> Callable:
    """
    Create a FastAPI middleware function for rate limiting.

    Args:
        limiter: RateLimiter instance (defaults to shared 60/min)
        path_prefixes: URL path prefixes to rate-limit (default: ["/api/"])

    Returns:
        An async middleware function for FastAPI app.add_middleware()
    """
    _limiter = limiter or get_rate_limiter()
    _prefixes = path_prefixes or ["/api/"]

    async def middleware(request: Any, call_next: Any) -> Any:
        path = request.url.path
        if any(path.startswith(p) for p in _prefixes):
            client_ip = request.client.host if request.client else "unknown"
            if not _limiter.is_allowed(client_ip):
                from starlette.responses import JSONResponse

                return JSONResponse(
                    {"success": False, "error": "Rate limit exceeded"},
                    status_code=429,
                )
        return await call_next(request)

    return middleware


def make_auth_middleware(
    skip_paths: Optional[Set[str]] = None,
) -> Callable:
    """
    Create a FastAPI middleware function for bearer token auth.

    Auth is skipped for paths in skip_paths. If JOTTY_API_AUTH=none,
    the middleware passes everything through.

    Args:
        skip_paths: Set of path prefixes to skip auth on

    Returns:
        An async middleware function for FastAPI app.add_middleware()
    """
    _skip = skip_paths or {"/health", "/docs", "/openapi.json", "/redoc", "/static"}

    async def middleware(request: Any, call_next: Any) -> Any:
        token = get_api_token()
        if token is None:
            # Auth disabled
            return await call_next(request)

        path = request.url.path
        if any(path.startswith(s) for s in _skip):
            return await call_next(request)

        auth_header = request.headers.get("authorization")
        if not check_bearer_token(auth_header, token):
            from starlette.responses import JSONResponse

            return JSONResponse(
                {"success": False, "error": "Unauthorized"},
                status_code=401,
            )
        return await call_next(request)

    return middleware
