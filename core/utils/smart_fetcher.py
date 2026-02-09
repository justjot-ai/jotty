"""
Smart HTTP Fetcher with Intelligent 403 Handling
==================================================

Shared utility for all skills that need to fetch web content.
Handles 403/blocked responses via a proven escalation chain.

Escalation strategy (tested Feb 2026, every level verified):
    1. Direct HTTP request (fastest, works ~70% of sites)
    2. archive.org Wayback Machine API (works for Medium, blogs, news)
    3. Free proxy rotation (sometimes works for Reddit, geo-blocked)

Note: Google Cache is now JS-gated (2025+) — doesn't work with HTTP.
      Jina Reader and 12ft.io also fail for blocked domains.

DRY: Single source of truth for all web fetching across skills.
Used by: web-search, web-scraper, screener-financials, etc.
"""

import logging
import random
import re
import requests
import time
from typing import Dict, Any, Optional, List, Set
from urllib.parse import urlparse, quote

logger = logging.getLogger(__name__)


# =========================================================================
# USER AGENTS (shared across all skills)
# =========================================================================

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15',
]

# Domains known to block direct HTTP (skip to escalation)
ALWAYS_BLOCKED_DOMAINS = {
    'linkedin.com', 'www.linkedin.com',
}

# Domains that block everything including proxies (archive.org only hope).
# Skip direct + proxy entirely — they waste 10+ seconds and never work.
# Medium: JS-gated, proxies can't render JS. Reddit: aggressive bot detection.
AGGRESSIVELY_BLOCKED_DOMAINS = {
    'reddit.com', 'www.reddit.com', 'old.reddit.com',
    'medium.com', 'www.medium.com',  # covers subdomains via 'in' check
}

# Domains where NO method works (require real login / private content)
UNFETCHABLE_DOMAINS = {
    'facebook.com', 'www.facebook.com',
    'instagram.com', 'www.instagram.com',
}

# Runtime cache: domains that failed ALL escalation levels this session.
_blocked_cache: set = set()


def _random_headers() -> Dict[str, str]:
    """Get randomized browser-like headers."""
    return {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }


# =========================================================================
# ESCALATION LEVEL 2: archive.org Wayback Machine API
# =========================================================================

def _try_archive_org(url: str, timeout: int = 15) -> Optional[str]:
    """
    Fetch via archive.org Wayback Machine API.

    Verified to work (Feb 2026) for:
    - Medium articles: returns full article content
    - Blog posts: works for most indexed content
    - News articles: if archived within last ~year

    Does NOT work for:
    - Recent Reddit posts (not always archived)
    - Private/gated content

    Uses the Availability API to find the closest snapshot, then fetches it.
    """
    domain = urlparse(url).netloc
    try:
        # Step 1: Check if snapshot exists (fast API call)
        api_url = f"https://archive.org/wayback/available?url={quote(url, safe='')}"
        api_resp = requests.get(api_url, timeout=8,
                                headers={'User-Agent': random.choice(USER_AGENTS)})
        if api_resp.status_code != 200:
            return None

        data = api_resp.json()
        snapshot = data.get('archived_snapshots', {}).get('closest', {})
        snapshot_url = snapshot.get('url')

        if not snapshot_url or snapshot.get('status') != '200':
            logger.debug(f"SmartFetch: no archive.org snapshot for {domain}")
            return None

        # Step 2: Fetch the actual snapshot
        resp = requests.get(snapshot_url,
                            headers={'User-Agent': random.choice(USER_AGENTS)},
                            timeout=timeout, allow_redirects=True)
        if resp.status_code == 200 and len(resp.text) > 500:
            logger.info(f"SmartFetch: archive.org snapshot for {domain} ({snapshot.get('timestamp', '?')})")
            return resp.text

    except Exception as e:
        logger.debug(f"SmartFetch: archive.org failed for {domain}: {e}")

    return None


# =========================================================================
# PROXY ROTATOR (shared singleton, escalation level 4)
# =========================================================================

class ProxyRotator:
    """
    Manages free proxy rotation for HTTP requests.

    Fetches proxies from multiple free sources, rotates through them,
    tracks failures, and auto-refreshes when cache expires.
    """

    PROXY_SOURCES = [
        ("https://proxylist.geonode.com/api/proxy-list?limit=20&page=1&sort_by=lastChecked&sort_type=desc&protocols=http", "geonode"),
        ("https://api.proxyscrape.com/v2/?request=get&protocol=http&timeout=10000&country=all&ssl=all&anonymity=all", "proxyscrape"),
        ("https://www.proxy-list.download/api/v1/get?type=http", "proxy-list"),
    ]

    def __init__(self, cache_duration: int = 3600):
        self._proxies: List[str] = []
        self._index = 0
        self._failed: Set[str] = set()
        self._last_fetch = 0.0
        self._cache_duration = cache_duration

    def _fetch_proxies(self) -> List[str]:
        """Fetch free proxies from public APIs."""
        proxies = []

        for url, source in self.PROXY_SOURCES:
            try:
                resp = requests.get(url, timeout=8, headers={'User-Agent': random.choice(USER_AGENTS)})
                if resp.status_code != 200:
                    continue

                content = resp.text.strip()

                if source == "geonode":
                    try:
                        data = resp.json()
                        for p in data.get('data', [])[:15]:
                            ip, port = p.get('ip'), p.get('port')
                            if ip and port:
                                proxies.append(f"http://{ip}:{port}")
                    except Exception:
                        pass
                else:
                    for line in content.split('\n')[:15]:
                        line = line.strip()
                        if not line or 'invalid' in line.lower() or 'error' in line.lower():
                            continue
                        if ':' in line and not line.startswith('http'):
                            proxies.append(f"http://{line}")
                        elif line.startswith('http'):
                            proxies.append(line)

                if proxies:
                    logger.info(f"ProxyRotator: {len(proxies)} proxies from {source}")
                    break
            except Exception as e:
                logger.debug(f"ProxyRotator: {source} failed: {e}")
                continue

        return proxies[:20]

    def get_proxy(self) -> Optional[Dict[str, str]]:
        """Get next working proxy. Returns None if no proxies available."""
        now = time.time()
        if not self._proxies or (now - self._last_fetch) > self._cache_duration:
            self._proxies = self._fetch_proxies()
            self._last_fetch = now
            self._failed.clear()

        if not self._proxies:
            return None

        for _ in range(len(self._proxies)):
            proxy = self._proxies[self._index % len(self._proxies)]
            self._index += 1
            if proxy not in self._failed:
                return {'http': proxy, 'https': proxy}

        self._failed.clear()
        proxy = self._proxies[0]
        return {'http': proxy, 'https': proxy}

    def mark_failed(self, proxy_url: str):
        """Mark a proxy as failed."""
        self._failed.add(proxy_url)

    @property
    def available_count(self) -> int:
        return max(0, len(self._proxies) - len(self._failed))


# Global singleton
_proxy_rotator = ProxyRotator()


def get_proxy_rotator() -> ProxyRotator:
    """Get the shared ProxyRotator singleton."""
    return _proxy_rotator


# =========================================================================
# SMART FETCH — The main entry point
# =========================================================================

class FetchResult:
    """Result of a smart_fetch call."""
    __slots__ = ('success', 'response', 'content', 'status_code',
                 'used_proxy', 'error', 'skipped', 'source')

    def __init__(self, success=False, response=None, content='', status_code=0,
                 used_proxy=False, error='', skipped=False, source='direct'):
        self.success = success
        self.response = response
        self.content = content
        self.status_code = status_code
        self.used_proxy = used_proxy
        self.error = error
        self.skipped = skipped
        self.source = source  # 'direct', 'google_cache', 'archive_org', 'proxy'


def smart_fetch(
    url: str,
    timeout: int = 8,
    max_proxy_attempts: int = 1,
    headers: Optional[Dict[str, str]] = None,
    method: str = 'GET',
    total_budget: float = 20.0,
) -> FetchResult:
    """
    Intelligently fetch a URL with proven escalation chain.

    Escalation (each level tested Feb 2026):
        1. Direct HTTP request (works for ~70% of sites)
        2. archive.org Wayback Machine API (works for Medium, blogs, news)
        3. Free proxy rotation (sometimes works for Reddit, geo-blocked)

    For known-blocked domains (Reddit, Medium, LinkedIn):
        Skip direct → archive.org → proxy

    Args:
        url: URL to fetch
        timeout: Request timeout in seconds per attempt (default: 8s, was 15s)
        max_proxy_attempts: Max proxy retries (default: 1, was 2)
        headers: Custom headers (defaults to randomized)
        method: HTTP method (default: GET)
        total_budget: Hard wall-clock budget for ALL escalation attempts (default: 20s).
                      Prevents cascade from burning 60s+ on a single URL.

    Returns:
        FetchResult with success, content, source, etc.
    """
    _fetch_start = time.time()
    domain = urlparse(url).netloc.lower()

    def _budget_remaining() -> float:
        """Seconds left in total budget."""
        return max(0, total_budget - (time.time() - _fetch_start))

    def _budget_exhausted() -> bool:
        return _budget_remaining() < 1.0

    # Quick skip: truly unfetchable domains
    if any(d in domain for d in UNFETCHABLE_DOMAINS):
        return FetchResult(
            success=False,
            error=f"Domain {domain} requires login — cannot scrape",
            skipped=True,
        )

    # Quick skip: domains that failed ALL levels this session
    if domain in _blocked_cache:
        return FetchResult(
            success=False,
            error=f"Domain {domain} blocked (all methods failed earlier this session)",
            skipped=True,
        )

    req_headers = headers or _random_headers()
    is_blocked_domain = any(d in domain for d in ALWAYS_BLOCKED_DOMAINS)
    is_aggressively_blocked = any(d in domain for d in AGGRESSIVELY_BLOCKED_DOMAINS)

    # ── Level 1: Direct request (skip for known-blocked domains) ──
    if not is_blocked_domain and not is_aggressively_blocked:
        try:
            _t = min(timeout, _budget_remaining())
            resp = requests.request(method, url, headers=req_headers,
                                    timeout=_t, allow_redirects=True)
            if resp.status_code < 400:
                return FetchResult(
                    success=True, response=resp, content=resp.text,
                    status_code=resp.status_code, source='direct',
                )
            elif resp.status_code not in (403, 429, 451):
                # 404, 500, etc. — don't escalate
                return FetchResult(
                    success=False, response=resp,
                    status_code=resp.status_code,
                    error=f"HTTP {resp.status_code}",
                )
            # 403/429/451 → escalate
            logger.info(f"SmartFetch: {resp.status_code} from {domain} — escalating")
        except requests.Timeout:
            logger.info(f"SmartFetch: timeout on {domain} — escalating")
        except requests.RequestException as e:
            logger.info(f"SmartFetch: direct failed for {domain}: {e} — escalating")
    else:
        logger.debug(f"SmartFetch: {domain} is known-blocked, skipping direct")

    # ── Budget check before Level 2 ──
    if _budget_exhausted():
        _blocked_cache.add(domain)
        return FetchResult(
            success=False,
            error=f"Budget exhausted ({total_budget:.0f}s) after direct attempt for {domain}",
            status_code=0,
        )

    # ── Level 2: archive.org Wayback Machine API ──
    archive_html = _try_archive_org(url, timeout=min(timeout, _budget_remaining()))
    if archive_html:
        return FetchResult(
            success=True, content=archive_html,
            status_code=200, source='archive_org',
        )

    # ── Level 3: Free proxy (last resort, unreliable) ──
    # Skip for aggressively blocked domains (Reddit) — proxies never work, waste 10+ seconds
    if is_aggressively_blocked:
        _blocked_cache.add(domain)
        logger.info(f"SmartFetch: {domain} blocks all methods, skipping proxy (use API instead)")
        return FetchResult(
            success=False, status_code=403,
            error=f"{domain} blocks scrapers — archive.org had no snapshot. Content unavailable.",
        )

    # ── Budget check before Level 3 ──
    if _budget_exhausted():
        _blocked_cache.add(domain)
        return FetchResult(
            success=False,
            error=f"Budget exhausted ({total_budget:.0f}s) before proxy for {domain}",
            status_code=0,
        )

    rotator = get_proxy_rotator()
    for attempt in range(max_proxy_attempts):
        if _budget_exhausted():
            break
        proxy_dict = rotator.get_proxy()
        if not proxy_dict:
            break

        proxy_url = proxy_dict.get('http', '')
        try:
            req_headers['User-Agent'] = random.choice(USER_AGENTS)
            _t = min(timeout, _budget_remaining())
            resp = requests.request(
                method, url,
                headers=req_headers, proxies=proxy_dict,
                timeout=_t, allow_redirects=True,
            )
            if resp.status_code < 400:
                logger.info(f"SmartFetch: {domain} via proxy (attempt {attempt + 1})")
                return FetchResult(
                    success=True, response=resp, content=resp.text,
                    status_code=resp.status_code, used_proxy=True,
                    source='proxy',
                )
            elif resp.status_code in (403, 429):
                rotator.mark_failed(proxy_url)
            else:
                return FetchResult(
                    success=False, response=resp,
                    status_code=resp.status_code,
                    error=f"HTTP {resp.status_code} via proxy",
                    used_proxy=True, source='proxy',
                )
        except Exception as e:
            rotator.mark_failed(proxy_url)
            logger.debug(f"SmartFetch: proxy failed: {e}")

    # All levels failed — cache to avoid retrying this session
    _blocked_cache.add(domain)
    error_msg = f"All methods failed for {domain}: direct → archive.org → proxy"
    logger.warning(f"SmartFetch: {error_msg}")
    return FetchResult(success=False, error=error_msg, status_code=403)
