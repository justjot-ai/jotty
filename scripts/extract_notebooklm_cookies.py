#!/usr/bin/env python3
"""
Extract NotebookLM Cookies from Browser Profile

This script extracts cookies from an authenticated NotebookLM browser profile
for use in Docker environments via environment variable.

Usage:
    python3 scripts/extract_notebooklm_cookies.py [--profile-dir /path/to/profile]
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

try:
    from playwright.async_api import async_playwright
except ImportError:
    print("❌ Playwright not installed!")
    print("Install with: pip install playwright && playwright install chromium")
    sys.exit(1)


async def extract_cookies(profile_dir: str):
    """
    Extract cookies from browser profile.
    
    Args:
        profile_dir: Directory containing browser profile
    """
    profile_path = Path(profile_dir)
    
    if not profile_path.exists():
        print(f"❌ Profile directory not found: {profile_path}")
        print("Run auth_notebooklm.py first to create a profile")
        sys.exit(1)
    
    print(f"🍪 Extracting cookies from: {profile_path}")
    
    try:
        async with async_playwright() as p:
            # Launch browser with existing profile
            browser = await p.chromium.launch_persistent_context(
                user_data_dir=str(profile_path),
                headless=True  # Don't need to show browser
            )
            
            # Get page
            pages = browser.pages
            if pages:
                page = pages[0]
            else:
                page = await browser.new_page()
            
            # Navigate to NotebookLM to refresh cookies
            await page.goto('https://notebooklm.google.com', wait_until='networkidle', timeout=30000)
            await page.wait_for_timeout(2000)
            
            # Get cookies
            cookies = await page.context.cookies()
            
            await browser.close()
            
            if not cookies:
                print("⚠️  No cookies found. Profile may not be authenticated.")
                sys.exit(1)
            
            # Save cookies to file
            cookies_file = profile_path / 'cookies.json'
            with open(cookies_file, 'w') as f:
                json.dump(cookies, f, indent=2)
            
            print(f"✅ Extracted {len(cookies)} cookies")
            print(f"💾 Saved to: {cookies_file}")
            print()
            print("To use in Docker, set environment variable:")
            print(f"  export NOTEBOOKLM_COOKIES='{json.dumps(cookies)}'")
            print()
            print("Or in docker-compose.yml:")
            print(f"  environment:")
            print(f"    - NOTEBOOKLM_COOKIES='{json.dumps(cookies)}'")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Extract cookies from NotebookLM browser profile'
    )
    parser.add_argument(
        '--profile-dir',
        default=str(Path.home() / '.notebooklm_browser'),
        help='Directory containing browser profile (default: ~/.notebooklm_browser)'
    )
    
    args = parser.parse_args()
    
    asyncio.run(extract_cookies(args.profile_dir))


if __name__ == '__main__':
    main()
