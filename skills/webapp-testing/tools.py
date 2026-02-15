"""
Web Application Testing Skill - Test local web applications with Playwright.

Supports verifying frontend functionality, debugging UI behavior,
capturing screenshots, and viewing browser logs.
"""
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import os

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import tool_response, tool_error, async_tool_wrapper

# Status emitter for progress updates
status = SkillStatus("webapp-testing")


logger = logging.getLogger(__name__)

try:
    from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logger.warning("playwright not available, webapp testing will be limited")


@async_tool_wrapper()
async def test_webapp_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Test a local web application.
    
    Args:
        params:
            - app_url (str): URL of web application
            - test_type (str, optional): Type of test
            - actions (list, optional): Actions to perform
            - screenshot_path (str, optional): Screenshot path
            - headless (bool, optional): Headless mode
            - wait_for_networkidle (bool, optional): Wait for network idle
    
    Returns:
        Dictionary with test results
    """
    status.set_callback(params.pop('_status_callback', None))

    app_url = params.get('app_url', '')
    test_type = params.get('test_type', 'full')
    actions = params.get('actions', [])
    screenshot_path = params.get('screenshot_path', None)
    headless = params.get('headless', True)
    wait_for_networkidle = params.get('wait_for_networkidle', True)
    
    if not app_url:
        return {
            'success': False,
            'error': 'app_url is required'
        }
    
    if not PLAYWRIGHT_AVAILABLE:
        return {
            'success': False,
            'error': 'playwright not available. Install with: pip install playwright && playwright install chromium'
        }
    
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=headless)
            page = await browser.new_page()
            
            # Capture console logs
            console_logs = []
            
            def handle_console(msg):
                console_logs.append({
                    'type': msg.type,
                    'text': msg.text,
                    'timestamp': datetime.now().isoformat()
                })
            
            page.on('console', handle_console)
            
            # Navigate to app
            await page.goto(app_url)
            
            # Wait for network idle (critical for dynamic apps)
            if wait_for_networkidle:
                await page.wait_for_load_state('networkidle')
            
            test_results = {
                'url': app_url,
                'title': await page.title(),
                'actions_performed': []
            }
            
            # Perform actions
            if test_type in ['interaction', 'full'] and actions:
                for action in actions:
                    action_type = action.get('type', '')
                    selector = action.get('selector', '')
                    
                    try:
                        if action_type == 'click':
                            await page.click(selector)
                            test_results['actions_performed'].append(f"Clicked: {selector}")
                        
                        elif action_type == 'fill':
                            value = action.get('value', '')
                            await page.fill(selector, value)
                            test_results['actions_performed'].append(f"Filled {selector} with: {value}")
                        
                        elif action_type == 'select':
                            value = action.get('value', '')
                            await page.select_option(selector, value)
                            test_results['actions_performed'].append(f"Selected {value} in {selector}")
                        
                        elif action_type == 'wait':
                            timeout = action.get('timeout', 5000)
                            await page.wait_for_selector(selector, timeout=timeout)
                            test_results['actions_performed'].append(f"Waited for: {selector}")
                        
                        # Wait a bit between actions
                        await page.wait_for_timeout(500)
                        
                    except Exception as e:
                        logger.warning(f"Action failed: {action_type} on {selector}: {e}")
                        test_results['actions_performed'].append(f"Failed: {action_type} on {selector}")
            
            # Take screenshot if requested
            final_screenshot_path = None
            if test_type in ['screenshot', 'full'] or screenshot_path:
                if not screenshot_path:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    screenshot_path = f"screenshot_{timestamp}.png"
                
                screenshot_file = Path(screenshot_path)
                screenshot_file.parent.mkdir(parents=True, exist_ok=True)
                
                await page.screenshot(path=str(screenshot_file), full_page=True)
                final_screenshot_path = str(screenshot_file)
            
            # Get page content for validation
            if test_type in ['validation', 'full']:
                content = await page.content()
                test_results['page_content_length'] = len(content)
                test_results['has_content'] = len(content) > 0
            
            await browser.close()
            
            return {
                'success': True,
                'screenshot_path': final_screenshot_path,
                'console_logs': console_logs[:50],  # Limit logs
                'test_results': test_results
            }
        
    except Exception as e:
        logger.error(f"Webapp testing failed: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }
