#!/usr/bin/env python3
"""
Standard Browser Session/Cookie Extractor

Extracts cookies and session data from browsers using standard methods:
- Browser DevTools Protocol (CDP) - Chrome/Edge/Opera standard
- Netscape cookie format - Universal format
- JSON cookie format - Modern standard
- Browser storage files - Direct database access

Supports multiple browsers:
- Chrome/Chromium (via CDP or SQLite)
- Firefox (via JSON)
- Edge (via CDP or SQLite)
- Opera (via CDP or SQLite)

Usage:
    # Extract from Chrome profile
    python3 scripts/extract_browser_session.py --browser chrome --profile-dir ~/.config/google-chrome
    
    # Extract from Playwright profile
    python3 scripts/extract_browser_session.py --browser playwright --profile-dir ~/.notebooklm_browser
    
    # Export in Netscape format (universal)
    python3 scripts/extract_browser_session.py --profile-dir ~/.notebooklm_browser --format netscape
    
    # Export for Docker environment variable
    python3 scripts/extract_browser_session.py --profile-dir ~/.notebooklm_browser --format docker-env
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse
import base64
import os

try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


def extract_chrome_cookies_sqlite(profile_dir: Path, domain: str = None) -> List[Dict[str, Any]]:
    """
    Extract cookies from Chrome/Chromium SQLite database.
    
    Standard Chrome cookie storage:
    - Cookies database: {profile_dir}/Default/Cookies
    - Uses SQLite with encrypted values (needs key from Local State)
    """
    cookies_db = profile_dir / 'Default' / 'Cookies'
    
    if not cookies_db.exists():
        # Try alternative locations
        for alt_path in ['Cookies', 'Network/Cookies']:
            alt_db = profile_dir / alt_path
            if alt_db.exists():
                cookies_db = alt_db
                break
        else:
            return []
    
    cookies = []
    try:
        conn = sqlite3.connect(str(cookies_db))
        cursor = conn.cursor()
        
        query = "SELECT name, value, host_key, path, expires_utc, is_secure, is_httponly, same_site FROM cookies"
        if domain:
            query += f" WHERE host_key LIKE '%{domain}%'"
        
        cursor.execute(query)
        
        for row in cursor.fetchall():
            name, value, host_key, path, expires_utc, is_secure, is_httponly, same_site = row
            
            # Chrome stores encrypted values - for most cases, we can use as-is
            # For encrypted values, would need to decrypt using key from Local State
            cookie = {
                'name': name,
                'value': value,  # May be encrypted
                'domain': host_key,
                'path': path or '/',
                'expires': expires_utc / 1000000 if expires_utc else None,
                'secure': bool(is_secure),
                'httpOnly': bool(is_httponly),
                'sameSite': 'None' if same_site == 0 else ('Lax' if same_site == 1 else 'Strict')
            }
            cookies.append(cookie)
        
        conn.close()
    except Exception as e:
        print(f"⚠️  SQLite extraction failed: {e}", file=sys.stderr)
    
    return cookies


def extract_firefox_cookies_json(profile_dir: Path, domain: str = None) -> List[Dict[str, Any]]:
    """
    Extract cookies from Firefox JSON storage.
    
    Standard Firefox cookie storage:
    - cookies.sqlite: SQLite database
    - Or cookies.json: JSON format (newer versions)
    """
    cookies_file = profile_dir / 'cookies.sqlite'
    
    if not cookies_file.exists():
        cookies_file = profile_dir / 'cookies.json'
        if cookies_file.exists():
            # JSON format
            try:
                with open(cookies_file, 'r') as f:
                    data = json.load(f)
                    cookies = []
                    for cookie in data.get('cookies', []):
                        if domain and domain not in cookie.get('domain', ''):
                            continue
                        cookies.append({
                            'name': cookie.get('name'),
                            'value': cookie.get('value'),
                            'domain': cookie.get('domain'),
                            'path': cookie.get('path', '/'),
                            'expires': cookie.get('expiry'),
                            'secure': cookie.get('secure', False),
                            'httpOnly': cookie.get('httpOnly', False),
                            'sameSite': cookie.get('sameSite', 'Lax')
                        })
                    return cookies
            except Exception as e:
                print(f"⚠️  JSON extraction failed: {e}", file=sys.stderr)
        return []
    
    # SQLite format (older Firefox)
    cookies = []
    try:
        conn = sqlite3.connect(str(cookies_file))
        cursor = conn.cursor()
        
        query = "SELECT name, value, host, path, expiry, isSecure, isHttpOnly FROM moz_cookies"
        if domain:
            query += f" WHERE host LIKE '%{domain}%'"
        
        cursor.execute(query)
        
        for row in cursor.fetchall():
            name, value, host, path, expiry, is_secure, is_httponly = row
            cookies.append({
                'name': name,
                'value': value,
                'domain': host,
                'path': path or '/',
                'expires': expiry,
                'secure': bool(is_secure),
                'httpOnly': bool(is_httponly),
                'sameSite': 'Lax'
            })
        
        conn.close()
    except Exception as e:
        print(f"⚠️  Firefox SQLite extraction failed: {e}", file=sys.stderr)
    
    return cookies


async def extract_cookies_cdp(profile_dir: Path, domain: str = None) -> List[Dict[str, Any]]:
    """
    Extract cookies using Browser DevTools Protocol (CDP).
    
    CDP is the standard protocol used by Chrome, Edge, Opera, and other Chromium-based browsers.
    This is the most reliable method as it uses the browser's native API.
    """
    if not PLAYWRIGHT_AVAILABLE:
        return []
    
    from playwright.async_api import async_playwright
    
    cookies = []
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch_persistent_context(
                user_data_dir=str(profile_dir),
                headless=True
            )
            
            pages = browser.pages
            if pages:
                page = pages[0]
            else:
                page = await browser.new_page()
            
            # Navigate to domain to load cookies
            target_url = f"https://{domain}" if domain else "https://notebooklm.google.com"
            await page.goto(target_url, wait_until='networkidle', timeout=10000)
            await page.wait_for_timeout(1000)
            
            # Get cookies via CDP
            cdp_session = await page.context.new_cdp_session(page)
            cdp_cookies = await cdp_session.send('Network.getAllCookies')
            
            for cookie in cdp_cookies.get('cookies', []):
                if domain and domain not in cookie.get('domain', ''):
                    continue
                cookies.append({
                    'name': cookie.get('name'),
                    'value': cookie.get('value'),
                    'domain': cookie.get('domain'),
                    'path': cookie.get('path', '/'),
                    'expires': cookie.get('expires'),
                    'secure': cookie.get('secure', False),
                    'httpOnly': cookie.get('httpOnly', False),
                    'sameSite': cookie.get('sameSite', 'None')
                })
            
            await browser.close()
    except Exception as e:
        print(f"⚠️  CDP extraction failed: {e}", file=sys.stderr)
    
    return cookies


def format_netscape_cookies(cookies: List[Dict[str, Any]], output_file: Path):
    """
    Export cookies in Netscape format (universal standard).
    
    Format:
    # Netscape HTTP Cookie File
    domain	flag	path	secure	expiration	name	value
    """
    with open(output_file, 'w') as f:
        f.write("# Netscape HTTP Cookie File\n")
        f.write("# This is a generated file! Do not edit.\n\n")
        
        for cookie in cookies:
            domain = cookie.get('domain', '')
            # Remove leading dot
            if domain.startswith('.'):
                domain = domain[1:]
            
            flag = 'TRUE' if domain.startswith('.') else 'FALSE'
            path = cookie.get('path', '/')
            secure = 'TRUE' if cookie.get('secure') else 'FALSE'
            expires = str(int(cookie.get('expires', 0))) if cookie.get('expires') else '0'
            name = cookie.get('name', '')
            value = cookie.get('value', '')
            
            f.write(f"{domain}\t{flag}\t{path}\t{secure}\t{expires}\t{name}\t{value}\n")


def format_docker_env(cookies: List[Dict[str, Any]]) -> str:
    """Format cookies as Docker environment variable."""
    return json.dumps(cookies)


def format_curl_cookies(cookies: List[Dict[str, Any]]) -> str:
    """Format cookies for curl command."""
    cookie_pairs = [f"{c['name']}={c['value']}" for c in cookies]
    return "; ".join(cookie_pairs)


def format_har(cookies: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Format cookies in HAR (HTTP Archive) format.
    
    HAR is a standard JSON format for HTTP transactions.
    """
    return {
        'log': {
            'version': '1.2',
            'creator': {'name': 'Browser Session Extractor', 'version': '1.0'},
            'entries': [{
                'request': {'cookies': cookies},
                'response': {}
            }]
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description='Extract browser session/cookies using standard methods',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported Browsers:
  - chrome/chromium: Chrome/Chromium (SQLite or CDP)
  - firefox: Firefox (JSON or SQLite)
  - edge: Microsoft Edge (SQLite or CDP)
  - playwright: Playwright profile (CDP)
  - auto: Auto-detect (default)

Supported Formats:
  - json: JSON format (default)
  - netscape: Netscape cookie file format (universal)
  - docker-env: Docker environment variable format
  - curl: curl cookie format
  - har: HTTP Archive format

Examples:
  # Extract from Chrome profile
  python3 scripts/extract_browser_session.py --browser chrome --profile-dir ~/.config/google-chrome
  
  # Extract in Netscape format (universal)
  python3 scripts/extract_browser_session.py --profile-dir ~/.notebooklm_browser --format netscape
  
  # Extract for specific domain
  python3 scripts/extract_browser_session.py --profile-dir ~/.notebooklm_browser --domain google.com
  
  # Export for Docker
  python3 scripts/extract_browser_session.py --profile-dir ~/.notebooklm_browser --format docker-env
        """
    )
    
    parser.add_argument('--browser', 
                       choices=['chrome', 'chromium', 'firefox', 'edge', 'playwright', 'auto'],
                       default='auto',
                       help='Browser type (default: auto-detect)')
    parser.add_argument('--profile-dir', required=True, help='Browser profile directory')
    parser.add_argument('--domain', help='Filter cookies by domain (e.g., google.com)')
    parser.add_argument('--format',
                       choices=['json', 'netscape', 'docker-env', 'curl', 'har'],
                       default='json',
                       help='Output format (default: json)')
    parser.add_argument('--output', help='Output file (default: stdout or cookies.{format})')
    parser.add_argument('--method',
                       choices=['cdp', 'sqlite', 'auto'],
                       default='auto',
                       help='Extraction method (default: auto)')
    
    args = parser.parse_args()
    
    profile_dir = Path(args.profile_dir)
    if not profile_dir.exists():
        print(f"❌ Profile directory not found: {profile_dir}")
        sys.exit(1)
    
    print(f"🔍 Extracting cookies from: {profile_dir}")
    print(f"🌐 Browser: {args.browser}")
    print(f"📋 Method: {args.method}")
    if args.domain:
        print(f"🎯 Domain filter: {args.domain}")
    
    cookies = []
    
    # Auto-detect browser if needed
    browser = args.browser
    if browser == 'auto':
        if 'chrome' in str(profile_dir).lower() or 'chromium' in str(profile_dir).lower():
            browser = 'chrome'
        elif 'firefox' in str(profile_dir).lower():
            browser = 'firefox'
        elif 'edge' in str(profile_dir).lower():
            browser = 'edge'
        else:
            browser = 'playwright'  # Default to Playwright/CDP
    
    # Extract cookies based on method
    if args.method == 'cdp' or (args.method == 'auto' and browser in ['chrome', 'chromium', 'edge', 'playwright']):
        print("📡 Using CDP (Browser DevTools Protocol)...")
        import asyncio
        cookies = asyncio.run(extract_cookies_cdp(profile_dir, args.domain))
    
    if not cookies and browser in ['chrome', 'chromium', 'edge']:
        print("💾 Trying SQLite extraction...")
        cookies = extract_chrome_cookies_sqlite(profile_dir, args.domain)
    
    if not cookies and browser == 'firefox':
        print("💾 Extracting from Firefox storage...")
        cookies = extract_firefox_cookies_json(profile_dir, args.domain)
    
    if not cookies:
        print("❌ No cookies found or extraction failed")
        sys.exit(1)
    
    print(f"✅ Extracted {len(cookies)} cookies")
    
    # Format output
    if args.format == 'json':
        output = json.dumps(cookies, indent=2)
    elif args.format == 'netscape':
        output_file = Path(args.output) if args.output else Path('cookies.txt')
        format_netscape_cookies(cookies, output_file)
        print(f"💾 Saved Netscape format to: {output_file}")
        sys.exit(0)
    elif args.format == 'docker-env':
        output = format_docker_env(cookies)
        print("\n📋 Docker environment variable:")
        print(f"export NOTEBOOKLM_COOKIES='{output}'")
        sys.exit(0)
    elif args.format == 'curl':
        output = format_curl_cookies(cookies)
    elif args.format == 'har':
        output = json.dumps(format_har(cookies), indent=2)
    
    # Write output
    if args.output:
        output_file = Path(args.output)
        output_file.write_text(output)
        print(f"💾 Saved to: {output_file}")
    else:
        print("\n" + output)


if __name__ == '__main__':
    main()
