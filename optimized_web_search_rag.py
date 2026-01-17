#!/usr/bin/env python3
"""
Optimized Web Search RAG Tool
=============================
Implements multiple strategies to avoid CAPTCHA and rate limiting issues.
"""

import json
import logging
import requests
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import hashlib
from pathlib import Path
from bs4 import BeautifulSoup
import time
import random
from urllib.parse import quote_plus, urlparse
import os
from fake_useragent import UserAgent
import cloudscraper

logger = logging.getLogger(__name__)

class OptimizedWebSearchRAG:
    """Optimized web search tool with anti-CAPTCHA strategies"""

    def __init__(self, cache_dir: str = "content/web_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize with rotating user agents
        try:
            self.ua = UserAgent()
        except:
            self.ua = None
            logger.warning("fake_useragent not available, using static user agents")

        # Create cloudscraper session for anti-bot protection
        self.scraper = cloudscraper.create_scraper()

        # Rate limiting configuration
        self.last_request_time = {}
        self.min_delay = 3  # Minimum delay between requests to same domain
        self.random_delay = (2, 5)  # Random additional delay range

        # Search providers with better alternatives
        self.search_providers = {
            'searx': self._search_searx_instances,  # Use public Searx instances
            'google_scholar': self._search_google_scholar,  # Academic content
            'bing': self._search_bing_api,  # Use Bing API if available
            'brave': self._search_brave,  # Brave Search API
            'duckduckgo_api': self._search_duckduckgo_api,  # Use API instead of scraping
        }

        # Fallback order - prioritize APIs over scraping
        self.fallback_order = ['searx', 'brave', 'bing', 'google_scholar']

        # Cache duration
        self.cache_duration = timedelta(days=7)

        # Session management
        self.session = requests.Session()
        self.session.headers.update(self._get_random_headers())

    def _get_random_headers(self) -> Dict[str, str]:
        """Generate random realistic headers"""
        if self.ua:
            user_agent = self.ua.random
        else:
            user_agents = [
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15'
            ]
            user_agent = random.choice(user_agents)

        headers = {
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
        }

        # Randomly add some headers
        if random.random() > 0.5:
            headers['Referer'] = 'https://www.google.com/'

        return headers

    def _respect_rate_limit(self, domain: str):
        """Implement intelligent rate limiting"""
        current_time = time.time()

        if domain in self.last_request_time:
            elapsed = current_time - self.last_request_time[domain]
            if elapsed < self.min_delay:
                sleep_time = self.min_delay - elapsed + random.uniform(*self.random_delay)
                logger.info(f"Rate limiting: waiting {sleep_time:.1f}s before next request to {domain}")
                time.sleep(sleep_time)

        self.last_request_time[domain] = time.time()

    def _search_searx_instances(self, query: str, num_results: int) -> List[Dict]:
        """Search using public Searx instances (no CAPTCHA)"""
        # List of public Searx instances
        searx_instances = [
            'https://searx.be',
            'https://search.bus-hit.me',
            'https://searx.tiekoetter.com',
            'https://searx.work',
        ]

        for instance in searx_instances:
            try:
                logger.info(f"Trying Searx instance: {instance}")

                params = {
                    'q': query,
                    'format': 'json',
                    'engines': 'google,bing,duckduckgo',
                    'pageno': 1,
                    'safesearch': 0
                }

                # Respect rate limit
                domain = urlparse(instance).netloc
                self._respect_rate_limit(domain)

                response = self.scraper.get(
                    f"{instance}/search",
                    params=params,
                    timeout=10,
                    headers=self._get_random_headers()
                )

                if response.status_code == 200:
                    data = response.json()
                    results = []

                    for result in data.get('results', [])[:num_results]:
                        results.append({
                            'title': result.get('title', ''),
                            'url': result.get('url', ''),
                            'snippet': result.get('content', '')
                        })

                    if results:
                        logger.info(f"Successfully got {len(results)} results from {instance}")
                        return results

            except Exception as e:
                logger.warning(f"Searx instance {instance} failed: {e}")
                continue

        return []

    def _search_brave(self, query: str, num_results: int) -> List[Dict]:
        """Search using Brave Search API (requires API key but no CAPTCHA)"""
        api_key = os.getenv('BRAVE_SEARCH_API_KEY')
        if not api_key:
            return []

        try:
            headers = {
                'Accept': 'application/json',
                'X-Subscription-Token': api_key
            }

            params = {
                'q': query,
                'count': num_results
            }

            response = requests.get(
                'https://api.search.brave.com/res/v1/web/search',
                headers=headers,
                params=params,
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                results = []

                for result in data.get('web', {}).get('results', [])[:num_results]:
                    results.append({
                        'title': result.get('title', ''),
                        'url': result.get('url', ''),
                        'snippet': result.get('description', '')
                    })

                return results

        except Exception as e:
            logger.warning(f"Brave search failed: {e}")

        return []

    def _search_duckduckgo_api(self, query: str, num_results: int) -> List[Dict]:
        """Use DuckDuckGo Instant Answer API (limited but no CAPTCHA)"""
        try:
            # DuckDuckGo Instant Answer API
            params = {
                'q': query,
                'format': 'json',
                'no_html': 1,
                'skip_disambig': 1
            }

            response = requests.get(
                'https://api.duckduckgo.com/',
                params=params,
                headers=self._get_random_headers(),
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                results = []

                # Extract from RelatedTopics
                for topic in data.get('RelatedTopics', [])[:num_results]:
                    if isinstance(topic, dict) and 'FirstURL' in topic:
                        results.append({
                            'title': topic.get('Text', '').split(' - ')[0],
                            'url': topic.get('FirstURL', ''),
                            'snippet': topic.get('Text', '')
                        })

                # Also check Abstract
                if data.get('AbstractURL'):
                    results.insert(0, {
                        'title': data.get('Heading', query),
                        'url': data.get('AbstractURL', ''),
                        'snippet': data.get('Abstract', '')
                    })

                return results[:num_results]

        except Exception as e:
            logger.warning(f"DuckDuckGo API search failed: {e}")

        return []

    def _search_google_scholar(self, query: str, num_results: int) -> List[Dict]:
        """Search Google Scholar for academic content (less likely to trigger CAPTCHA)"""
        try:
            from scholarly import scholarly

            results = []
            search_query = scholarly.search_pubs(query)

            for i, result in enumerate(search_query):
                if i >= num_results:
                    break

                results.append({
                    'title': result.get('bib', {}).get('title', ''),
                    'url': result.get('pub_url', result.get('eprint_url', '')),
                    'snippet': result.get('bib', {}).get('abstract', '')[:200] + '...'
                })

                # Rate limit
                time.sleep(random.uniform(1, 3))

            return results

        except Exception as e:
            logger.warning(f"Google Scholar search failed: {e}")
            return []

    def _search_bing_api(self, query: str, num_results: int) -> List[Dict]:
        """Use Bing Search API (requires API key but reliable)"""
        api_key = os.getenv('BING_SEARCH_API_KEY')
        if not api_key:
            return []

        try:
            headers = {
                'Ocp-Apim-Subscription-Key': api_key
            }

            params = {
                'q': query,
                'count': num_results,
                'textDecorations': False,
                'textFormat': 'HTML'
            }

            response = requests.get(
                'https://api.bing.microsoft.com/v7.0/search',
                headers=headers,
                params=params,
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                results = []

                for result in data.get('webPages', {}).get('value', [])[:num_results]:
                    results.append({
                        'title': result.get('name', ''),
                        'url': result.get('url', ''),
                        'snippet': result.get('snippet', '')
                    })

                return results

        except Exception as e:
            logger.warning(f"Bing API search failed: {e}")

        return []

    def search_and_extract(self, query: str, num_results: int = 5,
                          provider: str = 'searx') -> List[Dict[str, str]]:
        """Search and extract with anti-CAPTCHA optimizations"""
        logger.info(f"Optimized search for: {query}")

        # Check cache first (with longer duration)
        cache_key = self._get_cache_key(query, provider)
        cached_results = self._load_from_cache(cache_key)
        if cached_results:
            logger.info(f"Using {len(cached_results)} cached results")
            return cached_results

        # Try search providers in order
        results = []
        tried_providers = []

        # Start with requested provider
        if provider in self.search_providers:
            try:
                results = self.search_providers[provider](query, num_results)
                if results:
                    logger.info(f"Got {len(results)} results from {provider}")
                    self._save_to_cache(cache_key, results)
                    return results
            except Exception as e:
                logger.warning(f"Provider {provider} failed: {e}")
            tried_providers.append(provider)

        # Try fallback providers
        for fallback in self.fallback_order:
            if fallback in tried_providers:
                continue

            try:
                logger.info(f"Trying fallback provider: {fallback}")
                results = self.search_providers[fallback](query, num_results)
                if results:
                    logger.info(f"Got {len(results)} results from {fallback}")
                    self._save_to_cache(cache_key, results)
                    return results
            except Exception as e:
                logger.warning(f"Fallback {fallback} failed: {e}")

        logger.error("All search providers failed")
        return []

    def _get_cache_key(self, query: str, provider: str) -> str:
        """Generate cache key"""
        return hashlib.md5(f"{query}:{provider}".encode()).hexdigest()

    def _load_from_cache(self, cache_key: str) -> Optional[List[Dict]]:
        """Load from cache if not expired"""
        cache_file = self.cache_dir / f"{cache_key}.json"

        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached = json.load(f)

                # Check if cache is still valid
                cached_time = datetime.fromisoformat(cached['timestamp'])
                if datetime.now() - cached_time < self.cache_duration:
                    return cached['results']

            except Exception as e:
                logger.warning(f"Cache read error: {e}")

        return None

    def _save_to_cache(self, cache_key: str, results: List[Dict]):
        """Save results to cache"""
        cache_file = self.cache_dir / f"{cache_key}.json"

        try:
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'results': results
            }

            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)

        except Exception as e:
            logger.warning(f"Cache write error: {e}")


# Drop-in replacement for WebSearchRAG
class WebSearchRAG(OptimizedWebSearchRAG):
    """Backward compatible wrapper"""
    pass
