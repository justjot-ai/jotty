"""
OAuth Automation Skill

Automates OAuth login flows for services like NotebookLM, Google services, etc.
Supports headless/Docker environments with credential-based authentication.
"""
import asyncio
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import tempfile

logger = logging.getLogger(__name__)


async def oauth_login_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Automate OAuth login flow.
    
    Args:
        params: Dictionary containing:
            - provider (str, required): OAuth provider ('google', 'microsoft', etc.)
            - email (str, required): Email address (or use OAUTH_EMAIL env var)
            - password (str, required): Password (or use OAUTH_PASSWORD env var)
            - profile_dir (str, optional): Browser profile directory
            - headless (bool, optional): Run in headless mode (default: True)
            - service_url (str, optional): Target service URL
            - two_factor_code (str, optional): 2FA code if needed
    
    Returns:
        Dictionary with:
            - success (bool): Whether login succeeded
            - profile_dir (str): Browser profile directory
            - authenticated (bool): Whether authenticated
            - error (str, optional): Error message if failed
    """
    try:
        # Check if Playwright is available
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            return {
                'success': False,
                'error': 'Playwright not available. Install with: pip install playwright && playwright install chromium'
            }
        
        # Get credentials from params or environment
        provider = params.get('provider', 'google').lower()
        email = params.get('email') or os.getenv('OAUTH_EMAIL')
        password = params.get('password') or os.getenv('OAUTH_PASSWORD')
        
        if not email or not password:
            return {
                'success': False,
                'error': 'Email and password required (provide via params or OAUTH_EMAIL/OAUTH_PASSWORD env vars)'
            }
        
        # Determine profile directory
        profile_dir = params.get('profile_dir') or os.getenv('OAUTH_PROFILE_DIR')
        if not profile_dir:
            profile_dir = str(Path.home() / '.oauth_browser')
        
        profile_path = Path(profile_dir)
        profile_path.mkdir(parents=True, exist_ok=True)
        
        headless = params.get('headless', True)
        service_url = params.get('service_url', 'https://notebooklm.google.com')
        two_factor_code = params.get('two_factor_code')
        
        logger.info(f"Starting OAuth login for {provider} ({email})")
        
        # Provider-specific login
        if provider == 'google':
            result = await _google_oauth_login(
                email=email,
                password=password,
                profile_dir=profile_path,
                headless=headless,
                service_url=service_url,
                two_factor_code=two_factor_code
            )
        else:
            return {
                'success': False,
                'error': f'Provider {provider} not yet supported. Supported: google'
            }
        
        if result.get('success'):
            logger.info(f"✅ OAuth login successful. Profile: {profile_path}")
            return {
                'success': True,
                'profile_dir': str(profile_path),
                'authenticated': True,
                'provider': provider,
                'email': email
            }
        else:
            return result
            
    except Exception as e:
        logger.error(f"OAuth login error: {e}", exc_info=True)
        return {
            'success': False,
            'error': f'OAuth login failed: {str(e)}'
        }


async def _google_oauth_login(
    email: str,
    password: str,
    profile_dir: Path,
    headless: bool,
    service_url: str,
    two_factor_code: Optional[str] = None
) -> Dict[str, Any]:
    """
    Handle Google OAuth login flow.
    
    Google OAuth flow:
    1. Navigate to service (e.g., NotebookLM)
    2. Click "Sign in" -> redirects to accounts.google.com
    3. Enter email
    4. Enter password
    5. Handle 2FA if needed
    6. Grant permissions if needed
    7. Redirect back to service
    """
    from playwright.async_api import async_playwright
    
    try:
        async with async_playwright() as p:
            # Launch browser
            browser = await p.chromium.launch_persistent_context(
                user_data_dir=str(profile_dir),
                headless=headless,
                viewport={'width': 1920, 'height': 1080},
                args=['--disable-blink-features=AutomationControlled']
            )
            
            pages = browser.pages
            if pages:
                page = pages[0]
            else:
                page = await browser.new_page()
            
            # Navigate to service
            logger.info(f"Navigating to {service_url}...")
            await page.goto(service_url, wait_until='networkidle', timeout=30000)
            await page.wait_for_timeout(2000)
            
            # Check if already signed in
            page_content = await page.content()
            sign_in_indicators = ['Sign in', 'sign in', 'Sign In']
            is_signed_in = not any(indicator in page_content for indicator in sign_in_indicators)
            
            if is_signed_in:
                logger.info("Already signed in!")
                await browser.close()
                return {
                    'success': True,
                    'message': 'Already authenticated'
                }
            
            # Click sign in button
            logger.info("Looking for sign in button...")
            sign_in_selectors = [
                'text=Sign in',
                'a:has-text("Sign in")',
                'button:has-text("Sign in")',
                '[aria-label*="Sign in"]',
                'text=Sign in with Google'
            ]
            
            sign_in_clicked = False
            for selector in sign_in_selectors:
                try:
                    sign_in_btn = await page.query_selector(selector)
                    if sign_in_btn and await sign_in_btn.is_visible():
                        await sign_in_btn.click()
                        await page.wait_for_timeout(2000)
                        logger.info(f"✅ Clicked: {selector}")
                        sign_in_clicked = True
                        break
                except Exception as e:
                    logger.debug(f"Selector {selector} failed: {e}")
                    continue
            
            if not sign_in_clicked:
                # Try direct navigation to Google sign in
                logger.info("Direct navigation to Google sign in...")
                await page.goto('https://accounts.google.com/signin', wait_until='networkidle')
                await page.wait_for_timeout(2000)
            
            # Wait for email input
            logger.info("Entering email...")
            await page.wait_for_timeout(2000)
            
            email_selectors = [
                'input[type="email"]',
                'input[name="identifier"]',
                'input[id="identifierId"]',
                '#identifierId',
                'input[aria-label*="email"]'
            ]
            
            email_entered = False
            for selector in email_selectors:
                try:
                    email_input = await page.query_selector(selector)
                    if email_input and await email_input.is_visible():
                        await email_input.fill(email)
                        await page.wait_for_timeout(1000)
                        logger.info("✅ Email entered")
                        email_entered = True
                        break
                except Exception as e:
                    logger.debug(f"Email selector {selector} failed: {e}")
                    continue
            
            if not email_entered:
                await browser.close()
                return {
                    'success': False,
                    'error': 'Could not find email input field'
                }
            
            # Click Next button
            logger.info("Clicking Next...")
            next_selectors = [
                'button:has-text("Next")',
                'button:has-text("Continue")',
                '#identifierNext',
                'button[type="button"]:has-text("Next")'
            ]
            
            for selector in next_selectors:
                try:
                    next_btn = await page.query_selector(selector)
                    if next_btn and await next_btn.is_visible():
                        await next_btn.click()
                        await page.wait_for_timeout(3000)
                        logger.info("✅ Clicked Next")
                        break
                except:
                    continue
            
            # Wait for password input
            logger.info("Entering password...")
            await page.wait_for_timeout(2000)
            
            password_selectors = [
                'input[type="password"]',
                'input[name="password"]',
                'input[name="Passwd"]',
                '#password',
                'input[aria-label*="password"]'
            ]
            
            password_entered = False
            for selector in password_selectors:
                try:
                    password_input = await page.query_selector(selector)
                    if password_input and await password_input.is_visible():
                        await password_input.fill(password)
                        await page.wait_for_timeout(1000)
                        logger.info("✅ Password entered")
                        password_entered = True
                        break
                except Exception as e:
                    logger.debug(f"Password selector {selector} failed: {e}")
                    continue
            
            if not password_entered:
                await browser.close()
                return {
                    'success': False,
                    'error': 'Could not find password input field'
                }
            
            # Click Next/Submit
            logger.info("Submitting password...")
            for selector in next_selectors:
                try:
                    submit_btn = await page.query_selector(selector)
                    if submit_btn and await submit_btn.is_visible():
                        await submit_btn.click()
                        await page.wait_for_timeout(5000)
                        logger.info("✅ Password submitted")
                        break
                except:
                    continue
            
            # Check for 2FA
            await page.wait_for_timeout(3000)
            page_content = await page.content()
            
            two_factor_indicators = [
                'verification code',
                'two-factor',
                '2-step',
                'Enter the code',
                'phone number'
            ]
            
            needs_2fa = any(indicator.lower() in page_content.lower() for indicator in two_factor_indicators)
            
            if needs_2fa:
                logger.info("⚠️  2FA detected")
                
                if two_factor_code:
                    logger.info("Entering 2FA code...")
                    code_selectors = [
                        'input[type="tel"]',
                        'input[name="totpPin"]',
                        'input[aria-label*="code"]',
                        'input[aria-label*="verification"]'
                    ]
                    
                    for selector in code_selectors:
                        try:
                            code_input = await page.query_selector(selector)
                            if code_input and await code_input.is_visible():
                                await code_input.fill(two_factor_code)
                                await page.wait_for_timeout(1000)
                                
                                # Click Verify/Next
                                for next_sel in next_selectors:
                                    try:
                                        verify_btn = await page.query_selector(next_sel)
                                        if verify_btn and await verify_btn.is_visible():
                                            await verify_btn.click()
                                            await page.wait_for_timeout(3000)
                                            break
                                    except:
                                        continue
                                break
                        except:
                            continue
                else:
                    if not headless:
                        logger.info("Please enter 2FA code in the browser window...")
                        input("Press Enter after entering 2FA code...")
                    else:
                        await browser.close()
                        return {
                            'success': False,
                            'error': '2FA required. Provide two_factor_code parameter or run with headless=False',
                            'needs_2fa': True
                        }
            
            # Wait for redirect/authentication
            logger.info("Waiting for authentication to complete...")
            await page.wait_for_timeout(5000)
            
            # Check if we're back at the service or still on Google
            current_url = page.url
            
            if 'accounts.google.com' in current_url:
                # Might need to grant permissions or handle consent
                logger.info("Checking for permission/consent screens...")
                await page.wait_for_timeout(2000)
                
                # Look for "Allow" or "Continue" buttons
                allow_selectors = [
                    'button:has-text("Allow")',
                    'button:has-text("Continue")',
                    'button:has-text("Accept")'
                ]
                
                for selector in allow_selectors:
                    try:
                        allow_btn = await page.query_selector(selector)
                        if allow_btn and await allow_btn.is_visible():
                            await allow_btn.click()
                            await page.wait_for_timeout(3000)
                            logger.info("✅ Permissions granted")
                            break
                    except:
                        continue
            
            # Final check - navigate back to service
            if service_url not in current_url:
                logger.info(f"Navigating back to {service_url}...")
                await page.goto(service_url, wait_until='networkidle', timeout=30000)
                await page.wait_for_timeout(3000)
            
            # Verify authentication
            final_content = await page.content()
            is_authenticated = not any(indicator in final_content for indicator in sign_in_indicators)
            
            if is_authenticated:
                logger.info("✅ Authentication successful!")
                await browser.close()
                return {
                    'success': True,
                    'message': 'OAuth login successful',
                    'profile_dir': str(profile_dir)
                }
            else:
                await browser.close()
                return {
                    'success': False,
                    'error': 'Authentication may have failed - still seeing sign in page'
                }
                
    except Exception as e:
        logger.error(f"Google OAuth login error: {e}", exc_info=True)
        return {
            'success': False,
            'error': f'Google OAuth login failed: {str(e)}'
        }


__all__ = ['oauth_login_tool']
