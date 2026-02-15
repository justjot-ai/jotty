#!/usr/bin/env python3
"""
Standalone NotebookLM Authentication Script

This script authenticates with NotebookLM outside Docker and saves
the browser profile for use in Docker containers.

Usage:
    python3 scripts/auth_notebooklm.py [--profile-dir /path/to/profile]

The script will:
1. Open a browser window
2. Navigate to NotebookLM
3. Wait for you to sign in manually
4. Save the authenticated browser profile
"""

import argparse
import asyncio
import sys
from pathlib import Path

try:
    from playwright.async_api import async_playwright
except ImportError:
    print("❌ Playwright not installed!")
    print("Install with: pip install playwright && playwright install chromium")
    sys.exit(1)


async def authenticate_notebooklm(profile_dir: str):
    """
    Authenticate with NotebookLM and save browser profile.

    Args:
        profile_dir: Directory to save browser profile
    """
    profile_path = Path(profile_dir)
    profile_path.mkdir(parents=True, exist_ok=True)

    print("🔐 NotebookLM Authentication")
    print(f"📁 Profile directory: {profile_path}")
    print()
    print("Steps:")
    print("1. A browser window will open")
    print("2. Sign in to your Google account")
    print("3. Wait for NotebookLM to load")
    print("4. Press Enter here when done")
    print()

    try:
        async with async_playwright() as p:
            # Launch browser with persistent context
            browser = await p.chromium.launch_persistent_context(
                user_data_dir=str(profile_path),
                headless=False,  # Show browser for authentication
                viewport={"width": 1920, "height": 1080},
            )

            # Get or create page
            pages = browser.pages
            if pages:
                page = pages[0]
            else:
                page = await browser.new_page()

            # Navigate to NotebookLM
            print("🌐 Opening NotebookLM...")
            await page.goto(
                "https://notebooklm.google.com", wait_until="networkidle", timeout=30000
            )

            print("✅ Browser opened!")
            print()
            print("👉 Please sign in to your Google account in the browser window")
            print("👉 Wait for NotebookLM to fully load")
            print("👉 Then press Enter here to save the authentication...")
            print()

            # Wait for user to press Enter
            input()

            # Check if we're signed in
            await page.wait_for_timeout(2000)
            page_content = await page.content()

            sign_in_indicators = ["Sign in", "sign in", "Sign In"]
            is_signed_in = not any(indicator in page_content for indicator in sign_in_indicators)

            if is_signed_in:
                print("✅ Authentication successful!")
            else:
                print("⚠️  Warning: May not be fully authenticated")
                print("   Check the browser window to confirm")

            # Save cookies for reference
            cookies = await page.context.cookies()
            cookies_file = profile_path / "cookies.json"
            import json

            with open(cookies_file, "w") as f:
                json.dump(cookies, f, indent=2)
            print(f"💾 Cookies saved to: {cookies_file}")

            await browser.close()

            print()
            print("✅ Browser profile saved!")
            print(f"📁 Profile location: {profile_path}")
            print()
            print("To use in Docker:")
            print("  1. Copy this directory to your Docker host")
            print(f"  2. Mount it: -v {profile_path}:/app/.notebooklm_browser")
            print("  3. Set env: NOTEBOOKLM_USER_DATA_DIR=/app/.notebooklm_browser")
            print()
            print("Or set cookies in Docker:")
            print(f"  export NOTEBOOKLM_COOKIES=$(cat {cookies_file})")

    except KeyboardInterrupt:
        print("\n❌ Authentication cancelled")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Authenticate with NotebookLM and save browser profile"
    )
    parser.add_argument(
        "--profile-dir",
        default=str(Path.home() / ".notebooklm_browser"),
        help="Directory to save browser profile (default: ~/.notebooklm_browser)",
    )

    args = parser.parse_args()

    asyncio.run(authenticate_notebooklm(args.profile_dir))


if __name__ == "__main__":
    main()
