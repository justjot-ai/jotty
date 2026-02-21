"""
Tests for SSRF protection in smart_fetcher.py and http-client tools.py.
"""

import ipaddress
import pytest
from unittest.mock import patch, MagicMock


class TestIsSSRFSafe:
    """Test is_ssrf_safe() for various URL patterns."""

    def _is_ssrf_safe(self, url):
        from Jotty.core.infrastructure.utils.smart_fetcher import is_ssrf_safe

        return is_ssrf_safe(url)

    # ---- Safe URLs ----

    @patch("socket.getaddrinfo")
    def test_public_http_url_is_safe(self, mock_getaddrinfo):
        mock_getaddrinfo.return_value = [
            (2, 1, 6, "", ("93.184.216.34", 443)),
        ]
        safe, reason = self._is_ssrf_safe("https://example.com/page")
        assert safe is True
        assert reason == ""

    @patch("socket.getaddrinfo")
    def test_public_http_url_is_safe_http(self, mock_getaddrinfo):
        mock_getaddrinfo.return_value = [
            (2, 1, 6, "", ("93.184.216.34", 80)),
        ]
        safe, reason = self._is_ssrf_safe("http://example.com/page")
        assert safe is True
        assert reason == ""

    # ---- Blocked schemes ----

    def test_file_scheme_blocked(self):
        safe, reason = self._is_ssrf_safe("file:///etc/passwd")
        assert safe is False
        assert "scheme" in reason.lower()

    def test_gopher_scheme_blocked(self):
        safe, reason = self._is_ssrf_safe("gopher://evil.com")
        assert safe is False
        assert "scheme" in reason.lower()

    def test_ftp_scheme_blocked(self):
        safe, reason = self._is_ssrf_safe("ftp://internal/file")
        assert safe is False
        assert "scheme" in reason.lower()

    def test_empty_scheme_blocked(self):
        safe, reason = self._is_ssrf_safe("://noscheme")
        assert safe is False

    # ---- Loopback hostnames ----

    def test_localhost_blocked(self):
        safe, reason = self._is_ssrf_safe("http://localhost/admin")
        assert safe is False
        assert "loopback" in reason.lower()

    def test_127_0_0_1_blocked(self):
        safe, reason = self._is_ssrf_safe("http://127.0.0.1:8080/")
        assert safe is False
        assert "loopback" in reason.lower()

    def test_ipv6_loopback_blocked(self):
        safe, reason = self._is_ssrf_safe("http://[::1]/")
        assert safe is False
        assert "loopback" in reason.lower()

    def test_0_0_0_0_blocked(self):
        safe, reason = self._is_ssrf_safe("http://0.0.0.0/")
        assert safe is False
        assert "loopback" in reason.lower()

    # ---- Private IPs (via DNS resolution) ----

    @patch("socket.getaddrinfo")
    def test_private_10_x_blocked(self, mock_getaddrinfo):
        mock_getaddrinfo.return_value = [
            (2, 1, 6, "", ("10.0.0.1", 443)),
        ]
        safe, reason = self._is_ssrf_safe("https://internal.corp.com")
        assert safe is False
        assert "private" in reason.lower()

    @patch("socket.getaddrinfo")
    def test_private_172_16_blocked(self, mock_getaddrinfo):
        mock_getaddrinfo.return_value = [
            (2, 1, 6, "", ("172.16.0.1", 443)),
        ]
        safe, reason = self._is_ssrf_safe("https://staging.example.com")
        assert safe is False
        assert "private" in reason.lower()

    @patch("socket.getaddrinfo")
    def test_private_192_168_blocked(self, mock_getaddrinfo):
        mock_getaddrinfo.return_value = [
            (2, 1, 6, "", ("192.168.1.1", 443)),
        ]
        safe, reason = self._is_ssrf_safe("https://router.local")
        assert safe is False
        assert "private" in reason.lower()

    # ---- Link-local / AWS metadata ----

    @patch("socket.getaddrinfo")
    def test_link_local_169_254_blocked(self, mock_getaddrinfo):
        mock_getaddrinfo.return_value = [
            (2, 1, 6, "", ("169.254.169.254", 80)),
        ]
        safe, reason = self._is_ssrf_safe("http://169.254.169.254/latest/meta-data/")
        assert safe is False
        # 169.254.x is both link-local and private; either reason is acceptable
        assert "link-local" in reason.lower() or "private" in reason.lower()

    # ---- DNS resolution failure ----

    @patch("socket.getaddrinfo", side_effect=__import__("socket").gaierror)
    def test_dns_failure_blocked(self, mock_getaddrinfo):
        safe, reason = self._is_ssrf_safe("https://nonexistent.invalid")
        assert safe is False
        assert "dns" in reason.lower()

    # ---- No hostname ----

    def test_no_hostname_blocked(self):
        safe, reason = self._is_ssrf_safe("http:///path")
        assert safe is False

    # ---- Resolved loopback IP via DNS ----

    @patch("socket.getaddrinfo")
    def test_dns_resolves_to_loopback_blocked(self, mock_getaddrinfo):
        mock_getaddrinfo.return_value = [
            (2, 1, 6, "", ("127.0.0.1", 443)),
        ]
        safe, reason = self._is_ssrf_safe("https://evil-redirect.com")
        assert safe is False
        assert "loopback" in reason.lower()


@pytest.mark.unit
class TestIPv6TransitionBlocking:
    """Test _is_ipv6_transition_unsafe() for IPv6 transition mechanisms."""

    def _check(self, ip_str):
        from Jotty.core.infrastructure.utils.smart_fetcher import _is_ipv6_transition_unsafe

        return _is_ipv6_transition_unsafe(ipaddress.IPv6Address(ip_str))

    # ---- IPv4-mapped IPv6 (::ffff:x.x.x.x) ----

    def test_ipv4_mapped_loopback_blocked(self):
        unsafe, reason = self._check("::ffff:127.0.0.1")
        assert unsafe is True
        assert "ipv4-mapped" in reason.lower()

    def test_ipv4_mapped_private_10_blocked(self):
        unsafe, reason = self._check("::ffff:10.0.0.1")
        assert unsafe is True
        assert "ipv4-mapped" in reason.lower()

    def test_ipv4_mapped_private_192_blocked(self):
        unsafe, reason = self._check("::ffff:192.168.1.1")
        assert unsafe is True
        assert "ipv4-mapped" in reason.lower()

    def test_ipv4_mapped_metadata_blocked(self):
        unsafe, reason = self._check("::ffff:169.254.169.254")
        assert unsafe is True
        assert "ipv4-mapped" in reason.lower()

    def test_ipv4_mapped_public_allowed(self):
        unsafe, _ = self._check("::ffff:93.184.216.34")
        assert unsafe is False

    # ---- 6to4 (2002::/16) ----

    def test_6to4_loopback_blocked(self):
        # 2002:7f00:0001:: embeds 127.0.0.1
        unsafe, reason = self._check("2002:7f00:0001::")
        assert unsafe is True
        assert "6to4" in reason.lower()

    def test_6to4_private_10_blocked(self):
        # 2002:0a00:0001:: embeds 10.0.0.1
        unsafe, reason = self._check("2002:0a00:0001::")
        assert unsafe is True
        assert "6to4" in reason.lower()

    def test_6to4_public_allowed(self):
        # 2002:5db8:d822:: embeds 93.184.216.34 (public)
        unsafe, _ = self._check("2002:5db8:d822::")
        assert unsafe is False

    # ---- Teredo (2001:0000::/32) ----

    def test_teredo_loopback_blocked(self):
        # Teredo XORs the IPv4 in last 4 bytes with 0xFF
        # 127.0.0.1 XOR 0xFF = 0x80, 0xFF, 0xFF, 0xFE
        unsafe, reason = self._check("2001:0000:0000:0000:0000:0000:80ff:fffe")
        assert unsafe is True
        assert "teredo" in reason.lower()

    def test_teredo_private_blocked(self):
        # 10.0.0.1 XOR 0xFF = 0xF5, 0xFF, 0xFF, 0xFE
        unsafe, reason = self._check("2001:0000:0000:0000:0000:0000:f5ff:fffe")
        assert unsafe is True
        assert "teredo" in reason.lower()

    # ---- Public IPv6 (not transition) ----

    def test_public_ipv6_allowed(self):
        # Google DNS
        unsafe, _ = self._check("2607:f8b0:4004:800::200e")
        assert unsafe is False

    # ---- Integration: is_ssrf_safe with IPv6 transition ----

    @patch("socket.getaddrinfo")
    def test_is_ssrf_safe_blocks_ipv4_mapped(self, mock_getaddrinfo):
        from Jotty.core.infrastructure.utils.smart_fetcher import is_ssrf_safe

        # Return an IPv4-mapped IPv6 address
        mock_getaddrinfo.return_value = [
            (10, 1, 6, "", ("::ffff:10.0.0.1", 443, 0, 0)),
        ]
        safe, reason = is_ssrf_safe("https://sneaky.example.com")
        assert safe is False
        assert "ipv4-mapped" in reason.lower() or "private" in reason.lower()


@pytest.mark.unit
class TestSmartFetchSSRF:
    """Test that smart_fetch() calls is_ssrf_safe and blocks unsafe URLs."""

    @patch("Jotty.core.infrastructure.utils.smart_fetcher.is_ssrf_safe")
    def test_smart_fetch_blocks_ssrf(self, mock_ssrf_check):
        from Jotty.core.infrastructure.utils.smart_fetcher import smart_fetch

        mock_ssrf_check.return_value = (False, "Blocked private IP: 10.0.0.1")
        result = smart_fetch("http://10.0.0.1/admin")
        assert result.success is False
        assert result.skipped is True
        assert "SSRF" in result.error

    @patch("Jotty.core.infrastructure.utils.smart_fetcher.is_ssrf_safe")
    @patch("requests.request")
    def test_smart_fetch_allows_safe_url(self, mock_request, mock_ssrf_check):
        from Jotty.core.infrastructure.utils.smart_fetcher import smart_fetch

        mock_ssrf_check.return_value = (True, "")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "OK"
        mock_resp.headers = {"Content-Type": "text/html"}
        mock_request.return_value = mock_resp

        result = smart_fetch("https://example.com")
        assert result.success is True


@pytest.mark.unit
class TestHTTPClientSSRF:
    """Test that http-client tools.py blocks SSRF URLs."""

    @patch("Jotty.core.infrastructure.utils.smart_fetcher.is_ssrf_safe")
    def test_http_client_blocks_ssrf(self, mock_ssrf_check):
        from importlib import import_module

        tools = import_module("Jotty.skills.http-client.tools")
        mock_ssrf_check.return_value = (False, "Blocked loopback hostname: localhost")

        result = tools._make_http_request("GET", "http://localhost/admin", {})
        assert result.get("success") is False
        assert "SSRF" in result.get("error", "")
