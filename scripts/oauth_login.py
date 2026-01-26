#!/usr/bin/env python3
"""
Standalone OAuth Login Script

Automate OAuth login flows for services like NotebookLM.
Can be used independently or integrated with Jotty skills.

Usage:
    # With credentials as arguments
    python3 scripts/oauth_login.py --provider google --email user@example.com --password pass
    
    # With environment variables (more secure)
    export OAUTH_EMAIL=user@example.com
    export OAUTH_PASSWORD=password
    python3 scripts/oauth_login.py --provider google
    
    # For specific service
    python3 scripts/oauth_login.py --provider google --service-url https://notebooklm.google.com
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path to import skill
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from skills.oauth_automation.tools import oauth_login_tool
except ImportError:
    print("❌ Could not import oauth_automation skill")
    print("Make sure you're running from the Jotty directory")
    sys.exit(1)


async def main():
    parser = argparse.ArgumentParser(
        description='Automate OAuth login flows',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic login
  python3 scripts/oauth_login.py --provider google --email user@example.com --password pass
  
  # Using environment variables (more secure)
  export OAUTH_EMAIL=user@example.com
  export OAUTH_PASSWORD=password
  python3 scripts/oauth_login.py --provider google
  
  # For specific service with custom profile
  python3 scripts/oauth_login.py --provider google --service-url https://notebooklm.google.com --profile-dir /tmp/oauth_profile
  
  # With 2FA code
  python3 scripts/oauth_login.py --provider google --email user@example.com --password pass --2fa-code 123456
        """
    )
    
    parser.add_argument('--provider', required=True, choices=['google'], help='OAuth provider')
    parser.add_argument('--email', help='Email address (or use OAUTH_EMAIL env var)')
    parser.add_argument('--password', help='Password (or use OAUTH_PASSWORD env var)')
    parser.add_argument('--profile-dir', help='Browser profile directory (default: ~/.oauth_browser)')
    parser.add_argument('--service-url', default='https://notebooklm.google.com', help='Target service URL')
    parser.add_argument('--headless', action='store_true', default=False, help='Run in headless mode')
    parser.add_argument('--no-headless', dest='headless', action='store_false', help='Show browser window')
    parser.add_argument('--2fa-code', dest='two_factor_code', help='2FA verification code')
    
    args = parser.parse_args()
    
    # Get credentials
    email = args.email or os.getenv('OAUTH_EMAIL')
    password = args.password or os.getenv('OAUTH_PASSWORD')
    
    if not email or not password:
        print("❌ Email and password required")
        print("   Provide via --email/--password or OAUTH_EMAIL/OAUTH_PASSWORD env vars")
        sys.exit(1)
    
    print(f"🔐 OAuth Login")
    print(f"   Provider: {args.provider}")
    print(f"   Email: {email}")
    print(f"   Service: {args.service_url}")
    print(f"   Headless: {args.headless}")
    print()
    
    result = await oauth_login_tool({
        'provider': args.provider,
        'email': email,
        'password': password,
        'profile_dir': args.profile_dir,
        'headless': args.headless,
        'service_url': args.service_url,
        'two_factor_code': args.two_factor_code
    })
    
    if result.get('success'):
        print("✅ OAuth login successful!")
        print(f"   Profile: {result.get('profile_dir')}")
        print()
        print("To use this profile:")
        print(f"   export OAUTH_PROFILE_DIR={result.get('profile_dir')}")
        print(f"   # Or mount in Docker: -v {result.get('profile_dir')}:/app/.oauth_browser")
    else:
        print(f"❌ OAuth login failed: {result.get('error')}")
        if result.get('needs_2fa'):
            print()
            print("💡 2FA required. Run with:")
            print("   python3 scripts/oauth_login.py --provider google --email ... --password ... --2fa-code <code>")
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
