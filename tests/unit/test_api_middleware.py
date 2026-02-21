"""
Tests for shared API middleware: auth + rate limiting.
"""

import os
import pytest
from unittest.mock import patch, AsyncMock, MagicMock


@pytest.mark.unit
class TestGetApiToken:
    """Test get_api_token() resolution."""

    @patch.dict(os.environ, {"JOTTY_API_AUTH": "none"}, clear=False)
    def test_auth_disabled_returns_none(self):
        from Jotty.core.infrastructure.utils.api_middleware import get_api_token

        assert get_api_token() is None

    @patch.dict(os.environ, {"JOTTY_API_TOKEN": "my-secret-token"}, clear=False)
    def test_env_var_token(self):
        from Jotty.core.infrastructure.utils.api_middleware import get_api_token

        # Clear JOTTY_API_AUTH if set
        os.environ.pop("JOTTY_API_AUTH", None)
        assert get_api_token() == "my-secret-token"

    @patch.dict(os.environ, {}, clear=False)
    def test_auto_generates_token(self, tmp_path):
        from Jotty.core.infrastructure.utils import api_middleware

        # Clear env vars
        os.environ.pop("JOTTY_API_TOKEN", None)
        os.environ.pop("JOTTY_API_AUTH", None)

        # Redirect token file to tmp
        token_file = tmp_path / "api_token"
        with (
            patch.object(api_middleware, "_TOKEN_FILE", token_file),
            patch.object(api_middleware, "_TOKEN_DIR", tmp_path),
        ):
            token = api_middleware.get_api_token()
            assert token is not None
            assert len(token) > 20  # urlsafe(32) is ~43 chars
            # Should persist
            assert token_file.read_text().strip() == token

    @patch.dict(os.environ, {}, clear=False)
    def test_reads_existing_token_file(self, tmp_path):
        from Jotty.core.infrastructure.utils import api_middleware

        os.environ.pop("JOTTY_API_TOKEN", None)
        os.environ.pop("JOTTY_API_AUTH", None)

        token_file = tmp_path / "api_token"
        token_file.write_text("existing-token-value")
        with (
            patch.object(api_middleware, "_TOKEN_FILE", token_file),
            patch.object(api_middleware, "_TOKEN_DIR", tmp_path),
        ):
            assert api_middleware.get_api_token() == "existing-token-value"


@pytest.mark.unit
class TestCheckBearerToken:
    """Test check_bearer_token()."""

    def test_valid_bearer_token(self):
        from Jotty.core.infrastructure.utils.api_middleware import check_bearer_token

        assert check_bearer_token("Bearer abc123", "abc123") is True

    def test_invalid_bearer_token(self):
        from Jotty.core.infrastructure.utils.api_middleware import check_bearer_token

        assert check_bearer_token("Bearer wrong", "abc123") is False

    def test_missing_authorization(self):
        from Jotty.core.infrastructure.utils.api_middleware import check_bearer_token

        assert check_bearer_token(None, "abc123") is False

    def test_no_bearer_prefix(self):
        from Jotty.core.infrastructure.utils.api_middleware import check_bearer_token

        assert check_bearer_token("Token abc123", "abc123") is False

    def test_empty_authorization(self):
        from Jotty.core.infrastructure.utils.api_middleware import check_bearer_token

        assert check_bearer_token("", "abc123") is False

    def test_bearer_case_insensitive(self):
        from Jotty.core.infrastructure.utils.api_middleware import check_bearer_token

        assert check_bearer_token("bearer abc123", "abc123") is True


@pytest.mark.unit
class TestRateLimiter:
    """Test RateLimiter class."""

    def test_allows_within_limit(self):
        from Jotty.core.infrastructure.utils.api_middleware import RateLimiter

        limiter = RateLimiter(max_requests=3, window_seconds=60)
        assert limiter.is_allowed("ip1") is True
        assert limiter.is_allowed("ip1") is True
        assert limiter.is_allowed("ip1") is True

    def test_blocks_over_limit(self):
        from Jotty.core.infrastructure.utils.api_middleware import RateLimiter

        limiter = RateLimiter(max_requests=2, window_seconds=60)
        assert limiter.is_allowed("ip1") is True
        assert limiter.is_allowed("ip1") is True
        assert limiter.is_allowed("ip1") is False

    def test_different_keys_independent(self):
        from Jotty.core.infrastructure.utils.api_middleware import RateLimiter

        limiter = RateLimiter(max_requests=1, window_seconds=60)
        assert limiter.is_allowed("ip1") is True
        assert limiter.is_allowed("ip2") is True
        assert limiter.is_allowed("ip1") is False
        assert limiter.is_allowed("ip2") is False

    @patch("time.time")
    def test_expired_entries_pruned(self, mock_time):
        from Jotty.core.infrastructure.utils.api_middleware import RateLimiter

        limiter = RateLimiter(max_requests=1, window_seconds=10)

        # First request at t=100
        mock_time.return_value = 100.0
        assert limiter.is_allowed("ip1") is True

        # Second request at t=100 — blocked
        assert limiter.is_allowed("ip1") is False

        # Third request at t=111 — old entry expired, allowed
        mock_time.return_value = 111.0
        assert limiter.is_allowed("ip1") is True

    def test_cleanup_removes_stale_keys(self):
        from Jotty.core.infrastructure.utils.api_middleware import RateLimiter

        limiter = RateLimiter(max_requests=10, window_seconds=1)
        limiter.is_allowed("ip1")
        limiter.is_allowed("ip2")

        # Force all entries to be old
        import time

        time.sleep(1.1)

        removed = limiter.cleanup()
        assert removed == 2


@pytest.mark.unit
class TestMakeRateLimitMiddleware:
    """Test the FastAPI middleware factory."""

    @pytest.mark.asyncio
    async def test_allows_non_matching_path(self):
        from Jotty.core.infrastructure.utils.api_middleware import (
            RateLimiter,
            make_rate_limit_middleware,
        )

        limiter = RateLimiter(max_requests=0, window_seconds=60)  # Block everything
        mw = make_rate_limit_middleware(limiter=limiter, path_prefixes=["/api/"])

        request = MagicMock()
        request.url.path = "/health"
        request.client.host = "1.2.3.4"

        call_next = AsyncMock(return_value="ok")
        result = await mw(request, call_next)
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_blocks_rate_limited_path(self):
        from Jotty.core.infrastructure.utils.api_middleware import (
            RateLimiter,
            make_rate_limit_middleware,
        )

        limiter = RateLimiter(max_requests=0, window_seconds=60)
        mw = make_rate_limit_middleware(limiter=limiter, path_prefixes=["/api/"])

        request = MagicMock()
        request.url.path = "/api/chat"
        request.client.host = "1.2.3.4"

        call_next = AsyncMock(return_value="ok")
        result = await mw(request, call_next)
        assert result.status_code == 429


@pytest.mark.unit
class TestMakeAuthMiddleware:
    """Test the FastAPI auth middleware factory."""

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"JOTTY_API_AUTH": "none"}, clear=False)
    async def test_auth_disabled_passes_through(self):
        from Jotty.core.infrastructure.utils.api_middleware import make_auth_middleware

        mw = make_auth_middleware()
        request = MagicMock()
        request.url.path = "/api/chat"
        request.headers = {}

        call_next = AsyncMock(return_value="ok")
        result = await mw(request, call_next)
        assert result == "ok"

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"JOTTY_API_TOKEN": "secret123"}, clear=False)
    async def test_auth_rejects_missing_token(self):
        os.environ.pop("JOTTY_API_AUTH", None)
        from Jotty.core.infrastructure.utils.api_middleware import make_auth_middleware

        mw = make_auth_middleware()
        request = MagicMock()
        request.url.path = "/api/chat"
        request.headers = {}

        call_next = AsyncMock(return_value="ok")
        result = await mw(request, call_next)
        assert result.status_code == 401

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"JOTTY_API_TOKEN": "secret123"}, clear=False)
    async def test_auth_accepts_valid_token(self):
        os.environ.pop("JOTTY_API_AUTH", None)
        from Jotty.core.infrastructure.utils.api_middleware import make_auth_middleware

        mw = make_auth_middleware()
        request = MagicMock()
        request.url.path = "/api/chat"
        request.headers = {"authorization": "Bearer secret123"}

        call_next = AsyncMock(return_value="ok")
        result = await mw(request, call_next)
        assert result == "ok"

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"JOTTY_API_TOKEN": "secret123"}, clear=False)
    async def test_auth_skips_health_endpoint(self):
        os.environ.pop("JOTTY_API_AUTH", None)
        from Jotty.core.infrastructure.utils.api_middleware import make_auth_middleware

        mw = make_auth_middleware()
        request = MagicMock()
        request.url.path = "/health"
        request.headers = {}

        call_next = AsyncMock(return_value="ok")
        result = await mw(request, call_next)
        assert result == "ok"
