"""
Tests for Smart HTTP Fetcher (core/utils/smart_fetcher.py)
===========================================================

Comprehensive unit tests covering:
- FetchResult construction and __slots__
- _random_headers() variety and structure
- _try_archive_org() success/failure paths
- ProxyRotator: rotation, failure marking, cache refresh, available_count
- smart_fetch() 3-level escalation chain (direct -> archive -> proxy)
- Domain blocking: UNFETCHABLE, AGGRESSIVELY_BLOCKED, ALWAYS_BLOCKED
- Budget exhaustion at each escalation boundary
- _blocked_cache runtime caching
- get_proxy_rotator() singleton

All tests are fast (< 1s), offline, no real HTTP requests.
"""

import time
import pytest
from unittest.mock import Mock, MagicMock, patch, PropertyMock

try:
    from Jotty.core.utils.smart_fetcher import (
        FetchResult,
        ProxyRotator,
        smart_fetch,
        get_proxy_rotator,
        _random_headers,
        _try_archive_org,
        USER_AGENTS,
        ALWAYS_BLOCKED_DOMAINS,
        AGGRESSIVELY_BLOCKED_DOMAINS,
        UNFETCHABLE_DOMAINS,
        _blocked_cache,
    )
    SMART_FETCHER_AVAILABLE = True
except ImportError:
    SMART_FETCHER_AVAILABLE = False

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(not SMART_FETCHER_AVAILABLE, reason="smart_fetcher not importable"),
]


# =============================================================================
# Helpers
# =============================================================================

def _mock_response(status_code=200, text="<html>OK</html>", json_data=None):
    """Create a mock requests.Response object."""
    resp = Mock()
    resp.status_code = status_code
    resp.text = text
    if json_data is not None:
        resp.json.return_value = json_data
    else:
        resp.json.return_value = {}
    return resp


def _clear_blocked_cache():
    """Clear the module-level _blocked_cache before tests that depend on it."""
    _blocked_cache.clear()


# =============================================================================
# TestFetchResult
# =============================================================================

class TestFetchResult:
    """Tests for the FetchResult data class."""

    def test_default_construction(self):
        """FetchResult() with no args produces correct defaults."""
        result = FetchResult()
        assert result.success is False
        assert result.response is None
        assert result.content == ''
        assert result.status_code == 0
        assert result.used_proxy is False
        assert result.error == ''
        assert result.skipped is False
        assert result.source == 'direct'

    def test_custom_construction(self):
        """FetchResult with all args set stores them correctly."""
        resp = Mock()
        result = FetchResult(
            success=True,
            response=resp,
            content="<html>body</html>",
            status_code=200,
            used_proxy=True,
            error='',
            skipped=False,
            source='proxy',
        )
        assert result.success is True
        assert result.response is resp
        assert result.content == "<html>body</html>"
        assert result.status_code == 200
        assert result.used_proxy is True
        assert result.source == 'proxy'

    def test_slots_defined(self):
        """FetchResult uses __slots__ for memory efficiency."""
        assert hasattr(FetchResult, '__slots__')
        expected_slots = {'success', 'response', 'content', 'status_code',
                          'used_proxy', 'error', 'skipped', 'source'}
        assert set(FetchResult.__slots__) == expected_slots

    def test_no_dict(self):
        """FetchResult instances have no __dict__ thanks to __slots__."""
        result = FetchResult()
        assert not hasattr(result, '__dict__')

    def test_cannot_add_arbitrary_attributes(self):
        """__slots__ prevents adding arbitrary attributes."""
        result = FetchResult()
        with pytest.raises(AttributeError):
            result.nonexistent_attr = "boom"

    def test_source_options(self):
        """FetchResult supports all documented source types."""
        for source in ('direct', 'google_cache', 'archive_org', 'proxy'):
            result = FetchResult(source=source)
            assert result.source == source

    def test_skipped_with_error(self):
        """Skipped results carry an error message."""
        result = FetchResult(skipped=True, error="Domain blocked")
        assert result.skipped is True
        assert "blocked" in result.error.lower()


# =============================================================================
# TestRandomHeaders
# =============================================================================

class TestRandomHeaders:
    """Tests for _random_headers() function."""

    def test_returns_dict(self):
        """_random_headers returns a dict."""
        headers = _random_headers()
        assert isinstance(headers, dict)

    def test_required_keys_present(self):
        """Headers include all expected browser-like keys."""
        headers = _random_headers()
        expected_keys = {
            'User-Agent', 'Accept', 'Accept-Language',
            'Accept-Encoding', 'DNT', 'Connection',
            'Upgrade-Insecure-Requests',
        }
        assert expected_keys == set(headers.keys())

    def test_user_agent_from_list(self):
        """User-Agent comes from the USER_AGENTS list."""
        headers = _random_headers()
        assert headers['User-Agent'] in USER_AGENTS

    def test_variety_over_calls(self):
        """Multiple calls should eventually produce different User-Agent values."""
        agents_seen = set()
        for _ in range(100):
            h = _random_headers()
            agents_seen.add(h['User-Agent'])
        # With 5 UAs and 100 draws, we should see at least 2 distinct
        assert len(agents_seen) >= 2

    def test_dnt_is_set(self):
        """DNT (Do Not Track) header is set to '1'."""
        headers = _random_headers()
        assert headers['DNT'] == '1'

    def test_accept_encoding_includes_gzip(self):
        """Accept-Encoding includes gzip."""
        headers = _random_headers()
        assert 'gzip' in headers['Accept-Encoding']


# =============================================================================
# TestConstants
# =============================================================================

class TestConstants:
    """Tests for module-level constants."""

    def test_user_agents_is_nonempty_list(self):
        """USER_AGENTS is a non-empty list of strings."""
        assert isinstance(USER_AGENTS, list)
        assert len(USER_AGENTS) >= 3
        for ua in USER_AGENTS:
            assert isinstance(ua, str)
            assert 'Mozilla' in ua

    def test_always_blocked_contains_linkedin(self):
        """LinkedIn is in ALWAYS_BLOCKED_DOMAINS."""
        assert 'linkedin.com' in ALWAYS_BLOCKED_DOMAINS

    def test_aggressively_blocked_contains_reddit_and_medium(self):
        """Reddit and Medium are in AGGRESSIVELY_BLOCKED_DOMAINS."""
        assert 'reddit.com' in AGGRESSIVELY_BLOCKED_DOMAINS
        assert 'medium.com' in AGGRESSIVELY_BLOCKED_DOMAINS

    def test_unfetchable_contains_facebook_instagram(self):
        """Facebook and Instagram are in UNFETCHABLE_DOMAINS."""
        assert 'facebook.com' in UNFETCHABLE_DOMAINS
        assert 'instagram.com' in UNFETCHABLE_DOMAINS

    def test_domain_sets_are_disjoint(self):
        """The three domain sets have no overlap."""
        assert ALWAYS_BLOCKED_DOMAINS.isdisjoint(AGGRESSIVELY_BLOCKED_DOMAINS)
        assert ALWAYS_BLOCKED_DOMAINS.isdisjoint(UNFETCHABLE_DOMAINS)
        assert AGGRESSIVELY_BLOCKED_DOMAINS.isdisjoint(UNFETCHABLE_DOMAINS)


# =============================================================================
# TestTryArchiveOrg
# =============================================================================

class TestTryArchiveOrg:
    """Tests for _try_archive_org() Wayback Machine helper."""

    @patch('Jotty.core.utils.smart_fetcher.requests.get')
    def test_success_returns_html(self, mock_get):
        """Returns HTML content when archive.org has a valid snapshot."""
        snapshot_html = "<html>" + "x" * 600 + "</html>"
        api_response = _mock_response(200, json_data={
            'archived_snapshots': {
                'closest': {
                    'url': 'https://web.archive.org/web/20250101/http://example.com',
                    'status': '200',
                    'timestamp': '20250101120000',
                }
            }
        })
        snapshot_response = _mock_response(200, text=snapshot_html)
        mock_get.side_effect = [api_response, snapshot_response]

        result = _try_archive_org("http://example.com")
        assert result == snapshot_html

    @patch('Jotty.core.utils.smart_fetcher.requests.get')
    def test_no_snapshot_returns_none(self, mock_get):
        """Returns None when no snapshot exists."""
        api_response = _mock_response(200, json_data={
            'archived_snapshots': {}
        })
        mock_get.return_value = api_response

        result = _try_archive_org("http://example.com/never-archived")
        assert result is None

    @patch('Jotty.core.utils.smart_fetcher.requests.get')
    def test_api_non_200_returns_none(self, mock_get):
        """Returns None when archive.org API returns non-200."""
        mock_get.return_value = _mock_response(503)

        result = _try_archive_org("http://example.com")
        assert result is None

    @patch('Jotty.core.utils.smart_fetcher.requests.get')
    def test_snapshot_status_not_200_returns_none(self, mock_get):
        """Returns None when snapshot status is not '200'."""
        api_response = _mock_response(200, json_data={
            'archived_snapshots': {
                'closest': {
                    'url': 'https://web.archive.org/web/20250101/http://example.com',
                    'status': '404',
                }
            }
        })
        mock_get.return_value = api_response

        result = _try_archive_org("http://example.com")
        assert result is None

    @patch('Jotty.core.utils.smart_fetcher.requests.get')
    def test_short_content_returns_none(self, mock_get):
        """Returns None if snapshot content is under 500 chars (noise)."""
        api_response = _mock_response(200, json_data={
            'archived_snapshots': {
                'closest': {
                    'url': 'https://web.archive.org/web/20250101/http://example.com',
                    'status': '200',
                }
            }
        })
        short_response = _mock_response(200, text="short")
        mock_get.side_effect = [api_response, short_response]

        result = _try_archive_org("http://example.com")
        assert result is None

    @patch('Jotty.core.utils.smart_fetcher.requests.get')
    def test_exception_returns_none(self, mock_get):
        """Returns None on any exception (no crash)."""
        mock_get.side_effect = Exception("network error")

        result = _try_archive_org("http://example.com")
        assert result is None

    @patch('Jotty.core.utils.smart_fetcher.requests.get')
    def test_snapshot_fetch_non_200_returns_none(self, mock_get):
        """Returns None when snapshot URL returns non-200."""
        api_response = _mock_response(200, json_data={
            'archived_snapshots': {
                'closest': {
                    'url': 'https://web.archive.org/web/20250101/http://example.com',
                    'status': '200',
                }
            }
        })
        bad_snapshot = _mock_response(500, text="x" * 600)
        mock_get.side_effect = [api_response, bad_snapshot]

        result = _try_archive_org("http://example.com")
        assert result is None


# =============================================================================
# TestProxyRotator
# =============================================================================

class TestProxyRotator:
    """Tests for the ProxyRotator class."""

    def test_init_defaults(self):
        """ProxyRotator initializes with empty state."""
        rotator = ProxyRotator()
        assert rotator._proxies == []
        assert rotator._index == 0
        assert rotator._failed == set()
        assert rotator._last_fetch == 0.0
        assert rotator._cache_duration == 3600

    def test_init_custom_cache_duration(self):
        """ProxyRotator accepts custom cache_duration."""
        rotator = ProxyRotator(cache_duration=120)
        assert rotator._cache_duration == 120

    def test_available_count_empty(self):
        """available_count is 0 when no proxies loaded."""
        rotator = ProxyRotator()
        assert rotator.available_count == 0

    def test_available_count_with_proxies(self):
        """available_count reflects proxies minus failed."""
        rotator = ProxyRotator()
        rotator._proxies = ["http://1.1.1.1:8080", "http://2.2.2.2:8080", "http://3.3.3.3:8080"]
        assert rotator.available_count == 3

    def test_available_count_after_failures(self):
        """available_count decreases as proxies are marked failed."""
        rotator = ProxyRotator()
        rotator._proxies = ["http://1.1.1.1:8080", "http://2.2.2.2:8080", "http://3.3.3.3:8080"]
        rotator.mark_failed("http://1.1.1.1:8080")
        assert rotator.available_count == 2
        rotator.mark_failed("http://2.2.2.2:8080")
        assert rotator.available_count == 1

    def test_available_count_never_negative(self):
        """available_count does not go below 0."""
        rotator = ProxyRotator()
        rotator._proxies = ["http://1.1.1.1:8080"]
        rotator.mark_failed("http://1.1.1.1:8080")
        rotator.mark_failed("http://extra.fake:9999")
        assert rotator.available_count == 0

    def test_mark_failed_adds_to_set(self):
        """mark_failed adds the proxy URL to the _failed set."""
        rotator = ProxyRotator()
        rotator.mark_failed("http://1.1.1.1:8080")
        assert "http://1.1.1.1:8080" in rotator._failed

    def test_mark_failed_idempotent(self):
        """Marking the same proxy failed twice does not raise."""
        rotator = ProxyRotator()
        rotator.mark_failed("http://1.1.1.1:8080")
        rotator.mark_failed("http://1.1.1.1:8080")
        assert len(rotator._failed) == 1

    @patch('Jotty.core.utils.smart_fetcher.requests.get')
    def test_get_proxy_fetches_when_empty(self, mock_get):
        """get_proxy triggers _fetch_proxies when proxy list is empty."""
        mock_get.return_value = _mock_response(200, text="1.1.1.1:8080\n2.2.2.2:3128\n")
        rotator = ProxyRotator()
        proxy = rotator.get_proxy()
        assert proxy is not None
        assert 'http' in proxy
        assert 'https' in proxy

    @patch('Jotty.core.utils.smart_fetcher.requests.get')
    def test_get_proxy_rotates(self, mock_get):
        """Successive get_proxy calls rotate through available proxies."""
        rotator = ProxyRotator()
        rotator._proxies = ["http://1.1.1.1:8080", "http://2.2.2.2:8080"]
        rotator._last_fetch = time.time()  # prevent re-fetch

        p1 = rotator.get_proxy()
        p2 = rotator.get_proxy()
        assert p1['http'] == "http://1.1.1.1:8080"
        assert p2['http'] == "http://2.2.2.2:8080"

    @patch('Jotty.core.utils.smart_fetcher.requests.get')
    def test_get_proxy_skips_failed(self, mock_get):
        """get_proxy skips proxies in the failed set."""
        rotator = ProxyRotator()
        rotator._proxies = ["http://1.1.1.1:8080", "http://2.2.2.2:8080"]
        rotator._last_fetch = time.time()
        rotator.mark_failed("http://1.1.1.1:8080")

        proxy = rotator.get_proxy()
        assert proxy['http'] == "http://2.2.2.2:8080"

    def test_get_proxy_resets_failed_when_all_exhausted(self):
        """When all proxies are failed, _failed is cleared and first proxy returned."""
        rotator = ProxyRotator()
        rotator._proxies = ["http://1.1.1.1:8080", "http://2.2.2.2:8080"]
        rotator._last_fetch = time.time()
        rotator.mark_failed("http://1.1.1.1:8080")
        rotator.mark_failed("http://2.2.2.2:8080")

        proxy = rotator.get_proxy()
        assert proxy is not None
        assert proxy['http'] == "http://1.1.1.1:8080"
        assert len(rotator._failed) == 0

    @patch('Jotty.core.utils.smart_fetcher.requests.get')
    def test_get_proxy_returns_none_when_no_sources(self, mock_get):
        """get_proxy returns None when no proxy sources respond."""
        mock_get.side_effect = Exception("all down")
        rotator = ProxyRotator()
        proxy = rotator.get_proxy()
        assert proxy is None

    @patch('Jotty.core.utils.smart_fetcher.requests.get')
    def test_fetch_proxies_geonode_parsing(self, mock_get):
        """_fetch_proxies correctly parses geonode JSON format."""
        geonode_response = _mock_response(200, json_data={
            'data': [
                {'ip': '10.0.0.1', 'port': '8080'},
                {'ip': '10.0.0.2', 'port': '3128'},
            ]
        })
        mock_get.return_value = geonode_response
        rotator = ProxyRotator()
        proxies = rotator._fetch_proxies()
        assert 'http://10.0.0.1:8080' in proxies
        assert 'http://10.0.0.2:3128' in proxies

    @patch('Jotty.core.utils.smart_fetcher.requests.get')
    def test_fetch_proxies_plaintext_parsing(self, mock_get):
        """_fetch_proxies parses ip:port plaintext format."""
        # First source (geonode) fails, second source returns plaintext
        geonode_fail = _mock_response(500)
        plaintext_resp = _mock_response(200, text="5.5.5.5:8080\n6.6.6.6:3128\n")
        mock_get.side_effect = [geonode_fail, plaintext_resp]

        rotator = ProxyRotator()
        proxies = rotator._fetch_proxies()
        assert 'http://5.5.5.5:8080' in proxies
        assert 'http://6.6.6.6:3128' in proxies

    @patch('Jotty.core.utils.smart_fetcher.requests.get')
    def test_fetch_proxies_skips_invalid_lines(self, mock_get):
        """_fetch_proxies skips lines containing 'invalid' or 'error'."""
        geonode_fail = _mock_response(500)
        resp = _mock_response(200, text="invalid proxy\nerror occurred\n5.5.5.5:8080\n")
        mock_get.side_effect = [geonode_fail, resp]

        rotator = ProxyRotator()
        proxies = rotator._fetch_proxies()
        assert len(proxies) == 1
        assert 'http://5.5.5.5:8080' in proxies

    @patch('Jotty.core.utils.smart_fetcher.requests.get')
    def test_fetch_proxies_limits_to_20(self, mock_get):
        """_fetch_proxies returns at most 20 proxies."""
        lines = "\n".join(f"1.1.1.{i}:8080" for i in range(30))
        geonode_fail = _mock_response(500)
        resp = _mock_response(200, text=lines)
        mock_get.side_effect = [geonode_fail, resp]

        rotator = ProxyRotator()
        proxies = rotator._fetch_proxies()
        assert len(proxies) <= 20

    @patch('Jotty.core.utils.smart_fetcher.requests.get')
    def test_cache_expiry_triggers_refetch(self, mock_get):
        """get_proxy refetches when cache has expired."""
        mock_get.return_value = _mock_response(200, text="9.9.9.9:1234\n")
        rotator = ProxyRotator(cache_duration=1)
        rotator._proxies = ["http://old.proxy:8080"]
        rotator._last_fetch = time.time() - 10  # expired

        proxy = rotator.get_proxy()
        mock_get.assert_called()  # refetch happened
        # After refetch, failed set should be cleared
        assert len(rotator._failed) == 0

    def test_proxy_sources_defined(self):
        """PROXY_SOURCES class attribute is a non-empty list of tuples."""
        assert len(ProxyRotator.PROXY_SOURCES) >= 1
        for url, source in ProxyRotator.PROXY_SOURCES:
            assert url.startswith("http")
            assert isinstance(source, str)


# =============================================================================
# TestGetProxyRotator
# =============================================================================

class TestGetProxyRotator:
    """Tests for the get_proxy_rotator() singleton."""

    def test_returns_proxy_rotator(self):
        """get_proxy_rotator() returns a ProxyRotator instance."""
        rotator = get_proxy_rotator()
        assert isinstance(rotator, ProxyRotator)

    def test_singleton_returns_same_instance(self):
        """Multiple calls return the exact same instance."""
        r1 = get_proxy_rotator()
        r2 = get_proxy_rotator()
        assert r1 is r2


# =============================================================================
# TestSmartFetchDomainBlocking
# =============================================================================

class TestSmartFetchDomainBlocking:
    """Tests for domain blocking logic in smart_fetch."""

    def setup_method(self):
        """Clear _blocked_cache before each test."""
        _clear_blocked_cache()

    def test_unfetchable_facebook_skipped(self):
        """Facebook URLs are immediately skipped."""
        result = smart_fetch("https://www.facebook.com/some-post")
        assert result.success is False
        assert result.skipped is True
        assert "login" in result.error.lower() or "cannot" in result.error.lower()

    def test_unfetchable_instagram_skipped(self):
        """Instagram URLs are immediately skipped."""
        result = smart_fetch("https://www.instagram.com/p/abc123")
        assert result.success is False
        assert result.skipped is True

    @patch('Jotty.core.utils.smart_fetcher.get_proxy_rotator')
    @patch('Jotty.core.utils.smart_fetcher._try_archive_org', return_value=None)
    @patch('Jotty.core.utils.smart_fetcher.requests.request')
    def test_always_blocked_skips_direct(self, mock_request, mock_archive, mock_get_rotator):
        """LinkedIn (ALWAYS_BLOCKED) skips direct request, goes to archive then proxy."""
        mock_rotator = Mock()
        mock_rotator.get_proxy.return_value = None  # no proxies, so proxy level exits
        mock_get_rotator.return_value = mock_rotator

        result = smart_fetch("https://www.linkedin.com/in/someone")
        # archive.org should have been tried
        mock_archive.assert_called_once()
        # Direct requests.request should NOT have been called (proxy also got no proxies)
        mock_request.assert_not_called()

    @patch('Jotty.core.utils.smart_fetcher.get_proxy_rotator')
    @patch('Jotty.core.utils.smart_fetcher._try_archive_org', return_value=None)
    @patch('Jotty.core.utils.smart_fetcher.requests.request')
    def test_aggressively_blocked_skips_direct_and_proxy(self, mock_request, mock_archive, mock_rotator):
        """Reddit (AGGRESSIVELY_BLOCKED) skips direct and proxy, only tries archive."""
        result = smart_fetch("https://www.reddit.com/r/python")
        # Direct request should NOT have been called
        mock_request.assert_not_called()
        # Archive was tried
        mock_archive.assert_called_once()
        # Proxy rotator should NOT have been used
        mock_rotator.return_value.get_proxy.assert_not_called()
        # Result should reflect failure
        assert result.success is False
        assert result.status_code == 403

    @patch('Jotty.core.utils.smart_fetcher._try_archive_org')
    def test_aggressively_blocked_medium(self, mock_archive):
        """Medium (AGGRESSIVELY_BLOCKED) is handled correctly."""
        mock_archive.return_value = None
        result = smart_fetch("https://medium.com/@author/article-slug")
        assert result.success is False
        mock_archive.assert_called_once()

    def test_blocked_cache_hit_skipped(self):
        """Domains in _blocked_cache are skipped immediately."""
        _blocked_cache.add("cached-blocked.com")
        result = smart_fetch("https://cached-blocked.com/page")
        assert result.success is False
        assert result.skipped is True
        assert "blocked" in result.error.lower()
        _blocked_cache.discard("cached-blocked.com")


# =============================================================================
# TestSmartFetchDirectSuccess
# =============================================================================

class TestSmartFetchDirectSuccess:
    """Tests for smart_fetch Level 1 (direct request) success paths."""

    def setup_method(self):
        _clear_blocked_cache()

    @patch('Jotty.core.utils.smart_fetcher.requests.request')
    def test_direct_200_success(self, mock_request):
        """Direct 200 response returns success with source='direct'."""
        mock_request.return_value = _mock_response(200, text="<html>Hello</html>")
        result = smart_fetch("https://example.com/page")
        assert result.success is True
        assert result.content == "<html>Hello</html>"
        assert result.status_code == 200
        assert result.source == 'direct'
        assert result.used_proxy is False

    @patch('Jotty.core.utils.smart_fetcher.requests.request')
    def test_direct_301_success(self, mock_request):
        """Redirect (3xx) is treated as success (< 400)."""
        mock_request.return_value = _mock_response(301, text="redirected")
        result = smart_fetch("https://example.com/old-page")
        assert result.success is True
        assert result.status_code == 301

    @patch('Jotty.core.utils.smart_fetcher.requests.request')
    def test_direct_uses_custom_headers(self, mock_request):
        """Custom headers are passed through to the request."""
        mock_request.return_value = _mock_response(200, text="ok")
        custom = {"Authorization": "Bearer token123"}
        smart_fetch("https://api.example.com/data", headers=custom)
        call_kwargs = mock_request.call_args
        assert call_kwargs[1]['headers'] == custom

    @patch('Jotty.core.utils.smart_fetcher.requests.request')
    def test_direct_uses_method(self, mock_request):
        """HTTP method is passed through (e.g., POST)."""
        mock_request.return_value = _mock_response(200, text="ok")
        smart_fetch("https://api.example.com/data", method='POST')
        call_args = mock_request.call_args
        assert call_args[0][0] == 'POST'


# =============================================================================
# TestSmartFetchDirectFailureNoEscalation
# =============================================================================

class TestSmartFetchDirectFailureNoEscalation:
    """Tests for direct failures that do NOT trigger escalation."""

    def setup_method(self):
        _clear_blocked_cache()

    @patch('Jotty.core.utils.smart_fetcher.requests.request')
    def test_direct_404_no_escalation(self, mock_request):
        """404 does not trigger escalation (non-blocking error)."""
        mock_request.return_value = _mock_response(404, text="Not Found")
        result = smart_fetch("https://example.com/missing")
        assert result.success is False
        assert result.status_code == 404
        assert "404" in result.error

    @patch('Jotty.core.utils.smart_fetcher.requests.request')
    def test_direct_500_no_escalation(self, mock_request):
        """500 does not trigger escalation."""
        mock_request.return_value = _mock_response(500, text="Server Error")
        result = smart_fetch("https://example.com/broken")
        assert result.success is False
        assert result.status_code == 500


# =============================================================================
# TestSmartFetchEscalation
# =============================================================================

class TestSmartFetchEscalation:
    """Tests for smart_fetch escalation chain: direct -> archive -> proxy."""

    def setup_method(self):
        _clear_blocked_cache()

    @patch('Jotty.core.utils.smart_fetcher._try_archive_org')
    @patch('Jotty.core.utils.smart_fetcher.requests.request')
    def test_403_escalates_to_archive(self, mock_request, mock_archive):
        """403 triggers escalation to archive.org."""
        mock_request.return_value = _mock_response(403)
        archive_html = "<html>" + "archived content" * 50 + "</html>"
        mock_archive.return_value = archive_html

        result = smart_fetch("https://blocked-site.com/article")
        assert result.success is True
        assert result.source == 'archive_org'
        assert result.content == archive_html
        mock_archive.assert_called_once()

    @patch('Jotty.core.utils.smart_fetcher._try_archive_org')
    @patch('Jotty.core.utils.smart_fetcher.requests.request')
    def test_429_escalates_to_archive(self, mock_request, mock_archive):
        """429 (rate limited) triggers escalation."""
        mock_request.return_value = _mock_response(429)
        mock_archive.return_value = None

        result = smart_fetch("https://rate-limited.com/page")
        mock_archive.assert_called_once()

    @patch('Jotty.core.utils.smart_fetcher._try_archive_org')
    @patch('Jotty.core.utils.smart_fetcher.requests.request')
    def test_451_escalates_to_archive(self, mock_request, mock_archive):
        """451 (unavailable for legal reasons) triggers escalation."""
        mock_request.return_value = _mock_response(451)
        mock_archive.return_value = None

        result = smart_fetch("https://censored.com/page")
        mock_archive.assert_called_once()

    @patch('Jotty.core.utils.smart_fetcher.get_proxy_rotator')
    @patch('Jotty.core.utils.smart_fetcher._try_archive_org', return_value=None)
    @patch('Jotty.core.utils.smart_fetcher.requests.request')
    def test_full_escalation_to_proxy(self, mock_request, mock_archive, mock_get_rotator):
        """When direct=403 and archive=None, escalation reaches proxy."""
        direct_resp = _mock_response(403)
        proxy_resp = _mock_response(200, text="proxy content")
        mock_request.side_effect = [direct_resp, proxy_resp]

        mock_rotator = Mock()
        mock_rotator.get_proxy.return_value = {'http': 'http://proxy:8080', 'https': 'http://proxy:8080'}
        mock_get_rotator.return_value = mock_rotator

        result = smart_fetch("https://hard-to-reach.com/page")
        assert result.success is True
        assert result.source == 'proxy'
        assert result.used_proxy is True
        assert result.content == "proxy content"

    @patch('Jotty.core.utils.smart_fetcher.get_proxy_rotator')
    @patch('Jotty.core.utils.smart_fetcher._try_archive_org', return_value=None)
    @patch('Jotty.core.utils.smart_fetcher.requests.request')
    def test_all_levels_fail(self, mock_request, mock_archive, mock_get_rotator):
        """When all 3 levels fail, result is failure and domain cached."""
        mock_request.return_value = _mock_response(403)

        mock_rotator = Mock()
        mock_rotator.get_proxy.return_value = {'http': 'http://proxy:8080', 'https': 'http://proxy:8080'}
        mock_get_rotator.return_value = mock_rotator

        result = smart_fetch("https://totally-blocked.com/page")
        assert result.success is False
        assert "totally-blocked.com" in _blocked_cache

    @patch('Jotty.core.utils.smart_fetcher._try_archive_org', return_value=None)
    @patch('Jotty.core.utils.smart_fetcher.requests.request')
    def test_timeout_escalates(self, mock_request, mock_archive):
        """requests.Timeout triggers escalation."""
        import requests as req_lib
        mock_request.side_effect = req_lib.Timeout("read timeout")

        result = smart_fetch("https://slow-site.com/page")
        mock_archive.assert_called_once()

    @patch('Jotty.core.utils.smart_fetcher._try_archive_org', return_value=None)
    @patch('Jotty.core.utils.smart_fetcher.requests.request')
    def test_connection_error_escalates(self, mock_request, mock_archive):
        """requests.RequestException triggers escalation."""
        import requests as req_lib
        mock_request.side_effect = req_lib.ConnectionError("DNS resolution failed")

        result = smart_fetch("https://unreachable.com/page")
        mock_archive.assert_called_once()


# =============================================================================
# TestSmartFetchBudget
# =============================================================================

class TestSmartFetchBudget:
    """Tests for total_budget enforcement in smart_fetch."""

    def setup_method(self):
        _clear_blocked_cache()

    @patch('Jotty.core.utils.smart_fetcher.time.time')
    @patch('Jotty.core.utils.smart_fetcher.requests.request')
    def test_budget_exhausted_after_direct(self, mock_request, mock_time):
        """Budget exhaustion after direct attempt stops escalation."""
        # time.time() called: _fetch_start, _budget_remaining in min(), _budget_exhausted check
        # Simulate: start=100, direct call uses ~21s -> budget gone
        call_count = [0]
        def time_side_effect():
            call_count[0] += 1
            if call_count[0] <= 2:
                return 100.0  # start + first budget check
            return 121.0  # 21 seconds elapsed (> 20s budget)
        mock_time.side_effect = time_side_effect

        mock_request.return_value = _mock_response(403)

        result = smart_fetch("https://example.com/page", total_budget=20.0)
        assert result.success is False
        assert "budget" in result.error.lower() or "Budget" in result.error

    @patch('Jotty.core.utils.smart_fetcher.time.time')
    @patch('Jotty.core.utils.smart_fetcher._try_archive_org', return_value=None)
    @patch('Jotty.core.utils.smart_fetcher.requests.request')
    def test_budget_exhausted_before_proxy(self, mock_request, mock_archive, mock_time):
        """Budget exhaustion after archive attempt prevents proxy."""
        call_count = [0]
        def time_side_effect():
            call_count[0] += 1
            if call_count[0] <= 4:
                return 100.0  # start + direct + budget checks before archive
            return 121.0  # expired by time we check before proxy
        mock_time.side_effect = time_side_effect

        mock_request.return_value = _mock_response(403)

        result = smart_fetch("https://example.com/page", total_budget=20.0)
        assert result.success is False
        assert "budget" in result.error.lower() or "Budget" in result.error


# =============================================================================
# TestSmartFetchProxyDetails
# =============================================================================

class TestSmartFetchProxyDetails:
    """Tests for proxy-level behavior within smart_fetch."""

    def setup_method(self):
        _clear_blocked_cache()

    @patch('Jotty.core.utils.smart_fetcher.get_proxy_rotator')
    @patch('Jotty.core.utils.smart_fetcher._try_archive_org', return_value=None)
    @patch('Jotty.core.utils.smart_fetcher.requests.request')
    def test_proxy_403_marks_failed(self, mock_request, mock_archive, mock_get_rotator):
        """Proxy returning 403 triggers mark_failed on the rotator."""
        direct_resp = _mock_response(403)
        proxy_resp = _mock_response(403)
        mock_request.side_effect = [direct_resp, proxy_resp]

        mock_rotator = Mock()
        mock_rotator.get_proxy.return_value = {'http': 'http://bad-proxy:8080', 'https': 'http://bad-proxy:8080'}
        mock_get_rotator.return_value = mock_rotator

        result = smart_fetch("https://hard.com/page", max_proxy_attempts=1)
        mock_rotator.mark_failed.assert_called_once_with('http://bad-proxy:8080')

    @patch('Jotty.core.utils.smart_fetcher.get_proxy_rotator')
    @patch('Jotty.core.utils.smart_fetcher._try_archive_org', return_value=None)
    @patch('Jotty.core.utils.smart_fetcher.requests.request')
    def test_proxy_exception_marks_failed(self, mock_request, mock_archive, mock_get_rotator):
        """Exception during proxy request triggers mark_failed."""
        direct_resp = _mock_response(403)
        mock_request.side_effect = [direct_resp, Exception("proxy died")]

        mock_rotator = Mock()
        mock_rotator.get_proxy.return_value = {'http': 'http://dead-proxy:8080', 'https': 'http://dead-proxy:8080'}
        mock_get_rotator.return_value = mock_rotator

        result = smart_fetch("https://hard.com/page", max_proxy_attempts=1)
        mock_rotator.mark_failed.assert_called_once_with('http://dead-proxy:8080')

    @patch('Jotty.core.utils.smart_fetcher.get_proxy_rotator')
    @patch('Jotty.core.utils.smart_fetcher._try_archive_org', return_value=None)
    @patch('Jotty.core.utils.smart_fetcher.requests.request')
    def test_proxy_no_proxies_available(self, mock_request, mock_archive, mock_get_rotator):
        """When rotator returns None, proxy level is skipped."""
        mock_request.return_value = _mock_response(403)

        mock_rotator = Mock()
        mock_rotator.get_proxy.return_value = None
        mock_get_rotator.return_value = mock_rotator

        result = smart_fetch("https://hard.com/page")
        assert result.success is False

    @patch('Jotty.core.utils.smart_fetcher.get_proxy_rotator')
    @patch('Jotty.core.utils.smart_fetcher._try_archive_org', return_value=None)
    @patch('Jotty.core.utils.smart_fetcher.requests.request')
    def test_proxy_non_403_error_returns_immediately(self, mock_request, mock_archive, mock_get_rotator):
        """Proxy returning 500 returns immediately (no retry)."""
        direct_resp = _mock_response(403)
        proxy_resp = _mock_response(500, text="Server Error")
        mock_request.side_effect = [direct_resp, proxy_resp]

        mock_rotator = Mock()
        mock_rotator.get_proxy.return_value = {'http': 'http://proxy:8080', 'https': 'http://proxy:8080'}
        mock_get_rotator.return_value = mock_rotator

        result = smart_fetch("https://hard.com/page", max_proxy_attempts=3)
        assert result.success is False
        assert result.status_code == 500
        assert result.used_proxy is True
        assert result.source == 'proxy'
        assert "500" in result.error

    @patch('Jotty.core.utils.smart_fetcher.get_proxy_rotator')
    @patch('Jotty.core.utils.smart_fetcher._try_archive_org', return_value=None)
    @patch('Jotty.core.utils.smart_fetcher.requests.request')
    def test_max_proxy_attempts_respected(self, mock_request, mock_archive, mock_get_rotator):
        """max_proxy_attempts limits the number of proxy tries."""
        direct_resp = _mock_response(403)
        proxy_resp_1 = _mock_response(429)
        proxy_resp_2 = _mock_response(429)
        proxy_resp_3 = _mock_response(429)
        mock_request.side_effect = [direct_resp, proxy_resp_1, proxy_resp_2, proxy_resp_3]

        mock_rotator = Mock()
        mock_rotator.get_proxy.return_value = {'http': 'http://proxy:8080', 'https': 'http://proxy:8080'}
        mock_get_rotator.return_value = mock_rotator

        result = smart_fetch("https://hard.com/page", max_proxy_attempts=2)
        # direct + 2 proxy attempts = 3 total requests.request calls
        assert mock_request.call_count == 3


# =============================================================================
# TestSmartFetchBlockedCacheIntegration
# =============================================================================

class TestSmartFetchBlockedCacheIntegration:
    """Tests for _blocked_cache integration within smart_fetch."""

    def setup_method(self):
        _clear_blocked_cache()

    def teardown_method(self):
        _clear_blocked_cache()

    @patch('Jotty.core.utils.smart_fetcher.get_proxy_rotator')
    @patch('Jotty.core.utils.smart_fetcher._try_archive_org', return_value=None)
    @patch('Jotty.core.utils.smart_fetcher.requests.request')
    def test_domain_added_to_cache_on_total_failure(self, mock_request, mock_archive, mock_get_rotator):
        """Domain is added to _blocked_cache when all levels fail."""
        mock_request.return_value = _mock_response(403)

        mock_rotator = Mock()
        mock_rotator.get_proxy.return_value = {'http': 'http://proxy:8080', 'https': 'http://proxy:8080'}
        mock_get_rotator.return_value = mock_rotator

        smart_fetch("https://failsite.com/page")
        assert "failsite.com" in _blocked_cache

    @patch('Jotty.core.utils.smart_fetcher.requests.request')
    def test_second_call_to_cached_domain_skips_all(self, mock_request):
        """After caching, second call skips all levels."""
        _blocked_cache.add("already-failed.com")
        result = smart_fetch("https://already-failed.com/another-page")
        assert result.success is False
        assert result.skipped is True
        mock_request.assert_not_called()

    @patch('Jotty.core.utils.smart_fetcher._try_archive_org', return_value=None)
    @patch('Jotty.core.utils.smart_fetcher.requests.request')
    def test_aggressively_blocked_added_to_cache(self, mock_request, mock_archive):
        """Aggressively blocked domains get added to _blocked_cache after failure."""
        result = smart_fetch("https://reddit.com/r/python/comments/abc")
        assert "reddit.com" in _blocked_cache


# =============================================================================
# TestSmartFetchEdgeCases
# =============================================================================

class TestSmartFetchEdgeCases:
    """Edge case and boundary tests for smart_fetch."""

    def setup_method(self):
        _clear_blocked_cache()

    @patch('Jotty.core.utils.smart_fetcher.requests.request')
    def test_url_with_special_characters(self, mock_request):
        """URLs with query parameters and fragments work correctly."""
        mock_request.return_value = _mock_response(200, text="ok")
        result = smart_fetch("https://example.com/search?q=hello+world&page=2#results")
        assert result.success is True

    @patch('Jotty.core.utils.smart_fetcher.requests.request')
    def test_default_method_is_get(self, mock_request):
        """Default HTTP method is GET."""
        mock_request.return_value = _mock_response(200, text="ok")
        smart_fetch("https://example.com/page")
        call_args = mock_request.call_args
        assert call_args[0][0] == 'GET'

    @patch('Jotty.core.utils.smart_fetcher.requests.request')
    def test_timeout_parameter_passed(self, mock_request):
        """timeout parameter is passed to requests."""
        mock_request.return_value = _mock_response(200, text="ok")
        smart_fetch("https://example.com/page", timeout=5)
        call_kwargs = mock_request.call_args[1]
        # timeout should be min(5, budget_remaining)
        assert call_kwargs['timeout'] <= 5

    @patch('Jotty.core.utils.smart_fetcher._try_archive_org')
    @patch('Jotty.core.utils.smart_fetcher.requests.request')
    def test_archive_org_success_for_always_blocked(self, mock_request, mock_archive):
        """ALWAYS_BLOCKED domains succeed if archive.org has content."""
        archive_html = "<html>" + "linkedin content" * 100 + "</html>"
        mock_archive.return_value = archive_html

        result = smart_fetch("https://www.linkedin.com/in/someone")
        assert result.success is True
        assert result.source == 'archive_org'
        # Direct request was skipped
        mock_request.assert_not_called()

    @patch('Jotty.core.utils.smart_fetcher._try_archive_org')
    def test_aggressively_blocked_success_from_archive(self, mock_archive):
        """AGGRESSIVELY_BLOCKED domains can succeed via archive.org."""
        archive_html = "<html>" + "medium article" * 100 + "</html>"
        mock_archive.return_value = archive_html

        result = smart_fetch("https://medium.com/@author/article")
        assert result.success is True
        assert result.source == 'archive_org'

    @patch('Jotty.core.utils.smart_fetcher.requests.request')
    def test_allow_redirects_enabled(self, mock_request):
        """allow_redirects is True in direct requests."""
        mock_request.return_value = _mock_response(200, text="ok")
        smart_fetch("https://example.com/page")
        call_kwargs = mock_request.call_args[1]
        assert call_kwargs['allow_redirects'] is True

    @patch('Jotty.core.utils.smart_fetcher.requests.request')
    def test_subdomain_matching_for_unfetchable(self, mock_request):
        """Subdomain matching uses 'in' check for unfetchable domains."""
        # m.facebook.com contains 'facebook.com'
        result = smart_fetch("https://m.facebook.com/post/123")
        assert result.success is False
        assert result.skipped is True
        mock_request.assert_not_called()
