"""
Browser Automation Skill
========================

Full browser automation using Playwright for:
- Navigating to URLs with JS rendering
- Taking screenshots
- Filling forms
- Clicking elements
- Extracting data from dynamic pages
- Executing JavaScript
- Handling authentication
- Downloading files

Inspired by OpenClaw's browser automation capabilities.
"""
import os
import asyncio
import base64
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import Playwright
try:
    from playwright.async_api import async_playwright, Browser, Page, BrowserContext
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logger.warning("Playwright not installed. Run: pip install playwright && playwright install chromium")


class BrowserSession:
    """Manages a browser session for reuse across multiple operations."""

    _instance: Optional["BrowserSession"] = None
    _browser: Optional["Browser"] = None
    _context: Optional["BrowserContext"] = None
    _page: Optional["Page"] = None
    _playwright = None

    @classmethod
    async def get_instance(cls) -> "BrowserSession":
        """Get or create browser session singleton."""
        if cls._instance is None:
            cls._instance = BrowserSession()
        return cls._instance

    async def get_page(self, new_context: bool = False) -> "Page":
        """Get a browser page, creating browser if needed."""
        if not PLAYWRIGHT_AVAILABLE:
            raise ImportError("Playwright not installed")

        if self._playwright is None:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=True,
                args=['--no-sandbox', '--disable-dev-shm-usage']
            )

        if self._context is None or new_context:
            if self._context:
                await self._context.close()
            self._context = await self._browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            )
            self._page = await self._context.new_page()

        return self._page

    async def close(self):
        """Close browser session."""
        if self._context:
            await self._context.close()
            self._context = None
            self._page = None
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None
        BrowserSession._instance = None


async def browser_navigate_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Navigate to a URL and optionally wait for content.

    Args:
        params: Dictionary containing:
            - url (str, required): URL to navigate to
            - wait_for (str, optional): CSS selector to wait for
            - wait_timeout (int, optional): Timeout in ms (default: 30000)
            - screenshot (bool, optional): Take screenshot after load (default: False)
            - extract_text (bool, optional): Extract page text (default: True)

    Returns:
        Dictionary with:
            - success (bool): Whether navigation succeeded
            - url (str): Final URL after navigation
            - title (str): Page title
            - text (str): Extracted text content
            - screenshot_base64 (str, optional): Screenshot if requested
            - error (str, optional): Error message if failed
    """
    if not PLAYWRIGHT_AVAILABLE:
        return {'success': False, 'error': 'Playwright not installed. Run: pip install playwright && playwright install chromium'}

    try:
        url = params.get('url')
        if not url:
            return {'success': False, 'error': 'url parameter is required'}

        wait_for = params.get('wait_for')
        wait_timeout = params.get('wait_timeout', 30000)
        take_screenshot = params.get('screenshot', False)
        extract_text = params.get('extract_text', True)

        session = await BrowserSession.get_instance()
        page = await session.get_page()

        # Navigate
        logger.info(f"Navigating to: {url}")
        await page.goto(url, wait_until='networkidle', timeout=wait_timeout)

        # Wait for specific element if requested
        if wait_for:
            await page.wait_for_selector(wait_for, timeout=wait_timeout)

        # Get page info
        title = await page.title()
        final_url = page.url

        result = {
            'success': True,
            'url': final_url,
            'title': title
        }

        # Extract text
        if extract_text:
            text = await page.evaluate('() => document.body.innerText')
            result['text'] = text[:50000]  # Limit size
            result['text_length'] = len(text)

        # Take screenshot
        if take_screenshot:
            screenshot_bytes = await page.screenshot(full_page=False)
            result['screenshot_base64'] = base64.b64encode(screenshot_bytes).decode('utf-8')

        return result

    except Exception as e:
        logger.error(f"Browser navigation error: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}


async def browser_screenshot_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Take a screenshot of the current page or a specific URL.

    Args:
        params: Dictionary containing:
            - url (str, optional): URL to screenshot (uses current page if not provided)
            - full_page (bool, optional): Capture full scrollable page (default: False)
            - selector (str, optional): Screenshot specific element
            - save_path (str, optional): Save to file path
            - format (str, optional): 'png' or 'jpeg' (default: 'png')
            - quality (int, optional): JPEG quality 0-100 (default: 80)

    Returns:
        Dictionary with:
            - success (bool): Whether screenshot succeeded
            - screenshot_base64 (str): Base64 encoded image
            - saved_path (str, optional): Path if saved to file
            - width (int): Image width
            - height (int): Image height
            - error (str, optional): Error message if failed
    """
    if not PLAYWRIGHT_AVAILABLE:
        return {'success': False, 'error': 'Playwright not installed'}

    try:
        url = params.get('url')
        full_page = params.get('full_page', False)
        selector = params.get('selector')
        save_path = params.get('save_path')
        img_format = params.get('format', 'png')
        quality = params.get('quality', 80)

        session = await BrowserSession.get_instance()
        page = await session.get_page()

        # Navigate if URL provided
        if url:
            await page.goto(url, wait_until='networkidle')

        # Screenshot options
        screenshot_opts = {
            'full_page': full_page,
            'type': img_format
        }

        if img_format == 'jpeg':
            screenshot_opts['quality'] = quality

        # Take screenshot
        if selector:
            element = await page.query_selector(selector)
            if not element:
                return {'success': False, 'error': f'Element not found: {selector}'}
            screenshot_bytes = await element.screenshot(**screenshot_opts)
        else:
            screenshot_bytes = await page.screenshot(**screenshot_opts)

        result = {
            'success': True,
            'screenshot_base64': base64.b64encode(screenshot_bytes).decode('utf-8'),
            'format': img_format,
            'size_bytes': len(screenshot_bytes)
        }

        # Save to file if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_bytes(screenshot_bytes)
            result['saved_path'] = str(save_path)

        return result

    except Exception as e:
        logger.error(f"Screenshot error: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}


async def browser_fill_form_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fill form fields on the current page.

    Args:
        params: Dictionary containing:
            - url (str, optional): Navigate to URL first
            - fields (dict, required): Map of selector -> value to fill
            - submit_selector (str, optional): Selector for submit button
            - wait_after_submit (str, optional): Selector to wait for after submit
            - screenshot (bool, optional): Take screenshot after filling

    Returns:
        Dictionary with:
            - success (bool): Whether form was filled
            - fields_filled (int): Number of fields filled
            - submitted (bool): Whether form was submitted
            - url_after (str): URL after submission
            - error (str, optional): Error message if failed
    """
    if not PLAYWRIGHT_AVAILABLE:
        return {'success': False, 'error': 'Playwright not installed'}

    try:
        url = params.get('url')
        fields = params.get('fields', {})
        submit_selector = params.get('submit_selector')
        wait_after_submit = params.get('wait_after_submit')
        take_screenshot = params.get('screenshot', False)

        if not fields:
            return {'success': False, 'error': 'fields parameter is required'}

        session = await BrowserSession.get_instance()
        page = await session.get_page()

        # Navigate if URL provided
        if url:
            await page.goto(url, wait_until='networkidle')

        # Fill each field
        fields_filled = 0
        for selector, value in fields.items():
            try:
                element = await page.query_selector(selector)
                if element:
                    # Check element type
                    tag_name = await element.evaluate('el => el.tagName.toLowerCase()')
                    input_type = await element.evaluate('el => el.type || ""')

                    if tag_name == 'select':
                        await element.select_option(value)
                    elif input_type == 'checkbox':
                        if value:
                            await element.check()
                        else:
                            await element.uncheck()
                    elif input_type == 'radio':
                        await element.check()
                    elif input_type == 'file':
                        await element.set_input_files(value)
                    else:
                        await element.fill(value)

                    fields_filled += 1
                    logger.info(f"Filled field: {selector}")
            except Exception as e:
                logger.warning(f"Could not fill {selector}: {e}")

        result = {
            'success': True,
            'fields_filled': fields_filled,
            'submitted': False
        }

        # Submit if requested
        if submit_selector:
            submit_btn = await page.query_selector(submit_selector)
            if submit_btn:
                await submit_btn.click()
                result['submitted'] = True

                # Wait for navigation or element
                if wait_after_submit:
                    await page.wait_for_selector(wait_after_submit, timeout=30000)
                else:
                    await page.wait_for_load_state('networkidle')

                result['url_after'] = page.url

        # Screenshot if requested
        if take_screenshot:
            screenshot_bytes = await page.screenshot()
            result['screenshot_base64'] = base64.b64encode(screenshot_bytes).decode('utf-8')

        return result

    except Exception as e:
        logger.error(f"Form fill error: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}


async def browser_click_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Click an element on the page.

    Args:
        params: Dictionary containing:
            - selector (str, required): CSS selector for element to click
            - url (str, optional): Navigate to URL first
            - wait_for (str, optional): Selector to wait for after click
            - double_click (bool, optional): Double click instead (default: False)
            - right_click (bool, optional): Right click instead (default: False)
            - screenshot (bool, optional): Take screenshot after click

    Returns:
        Dictionary with:
            - success (bool): Whether click succeeded
            - url_after (str): URL after click
            - error (str, optional): Error message if failed
    """
    if not PLAYWRIGHT_AVAILABLE:
        return {'success': False, 'error': 'Playwright not installed'}

    try:
        selector = params.get('selector')
        url = params.get('url')
        wait_for = params.get('wait_for')
        double_click = params.get('double_click', False)
        right_click = params.get('right_click', False)
        take_screenshot = params.get('screenshot', False)

        if not selector:
            return {'success': False, 'error': 'selector parameter is required'}

        session = await BrowserSession.get_instance()
        page = await session.get_page()

        # Navigate if URL provided
        if url:
            await page.goto(url, wait_until='networkidle')

        # Find and click element
        element = await page.query_selector(selector)
        if not element:
            return {'success': False, 'error': f'Element not found: {selector}'}

        if double_click:
            await element.dblclick()
        elif right_click:
            await element.click(button='right')
        else:
            await element.click()

        # Wait for navigation or element
        if wait_for:
            await page.wait_for_selector(wait_for, timeout=30000)
        else:
            await asyncio.sleep(1)  # Brief wait for any JS effects

        result = {
            'success': True,
            'url_after': page.url
        }

        # Screenshot if requested
        if take_screenshot:
            screenshot_bytes = await page.screenshot()
            result['screenshot_base64'] = base64.b64encode(screenshot_bytes).decode('utf-8')

        return result

    except Exception as e:
        logger.error(f"Click error: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}


async def browser_extract_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract data from the current page using selectors.

    Args:
        params: Dictionary containing:
            - url (str, optional): Navigate to URL first
            - selectors (dict, required): Map of name -> selector for data extraction
            - extract_all (bool, optional): Extract all matches vs first (default: False)
            - attributes (dict, optional): Map of name -> attribute to extract (default: innerText)

    Returns:
        Dictionary with:
            - success (bool): Whether extraction succeeded
            - data (dict): Extracted data by name
            - error (str, optional): Error message if failed
    """
    if not PLAYWRIGHT_AVAILABLE:
        return {'success': False, 'error': 'Playwright not installed'}

    try:
        url = params.get('url')
        selectors = params.get('selectors', {})
        extract_all = params.get('extract_all', False)
        attributes = params.get('attributes', {})

        if not selectors:
            return {'success': False, 'error': 'selectors parameter is required'}

        session = await BrowserSession.get_instance()
        page = await session.get_page()

        # Navigate if URL provided
        if url:
            await page.goto(url, wait_until='networkidle')

        data = {}

        for name, selector in selectors.items():
            try:
                attr = attributes.get(name, 'innerText')

                if extract_all:
                    elements = await page.query_selector_all(selector)
                    values = []
                    for el in elements:
                        if attr == 'innerText':
                            val = await el.inner_text()
                        elif attr == 'innerHTML':
                            val = await el.inner_html()
                        elif attr == 'href':
                            val = await el.get_attribute('href')
                        else:
                            val = await el.get_attribute(attr)
                        values.append(val)
                    data[name] = values
                else:
                    element = await page.query_selector(selector)
                    if element:
                        if attr == 'innerText':
                            data[name] = await element.inner_text()
                        elif attr == 'innerHTML':
                            data[name] = await element.inner_html()
                        else:
                            data[name] = await element.get_attribute(attr)
                    else:
                        data[name] = None
            except Exception as e:
                logger.warning(f"Could not extract {name}: {e}")
                data[name] = None

        return {
            'success': True,
            'data': data,
            'url': page.url
        }

    except Exception as e:
        logger.error(f"Extract error: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}


async def browser_execute_js_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute JavaScript on the current page.

    Args:
        params: Dictionary containing:
            - script (str, required): JavaScript code to execute
            - url (str, optional): Navigate to URL first
            - args (list, optional): Arguments to pass to the script

    Returns:
        Dictionary with:
            - success (bool): Whether execution succeeded
            - result: Return value from script
            - error (str, optional): Error message if failed
    """
    if not PLAYWRIGHT_AVAILABLE:
        return {'success': False, 'error': 'Playwright not installed'}

    try:
        script = params.get('script')
        url = params.get('url')
        args = params.get('args', [])

        if not script:
            return {'success': False, 'error': 'script parameter is required'}

        session = await BrowserSession.get_instance()
        page = await session.get_page()

        # Navigate if URL provided
        if url:
            await page.goto(url, wait_until='networkidle')

        # Execute script
        if args:
            result = await page.evaluate(script, args)
        else:
            result = await page.evaluate(script)

        return {
            'success': True,
            'result': result,
            'url': page.url
        }

    except Exception as e:
        logger.error(f"JS execution error: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}


async def browser_wait_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Wait for an element or condition.

    Args:
        params: Dictionary containing:
            - selector (str, optional): CSS selector to wait for
            - state (str, optional): 'visible', 'hidden', 'attached', 'detached'
            - timeout (int, optional): Timeout in ms (default: 30000)
            - url (str, optional): Navigate to URL first

    Returns:
        Dictionary with:
            - success (bool): Whether wait completed
            - found (bool): Whether element was found
            - error (str, optional): Error message if failed
    """
    if not PLAYWRIGHT_AVAILABLE:
        return {'success': False, 'error': 'Playwright not installed'}

    try:
        selector = params.get('selector')
        state = params.get('state', 'visible')
        timeout = params.get('timeout', 30000)
        url = params.get('url')

        if not selector:
            return {'success': False, 'error': 'selector parameter is required'}

        session = await BrowserSession.get_instance()
        page = await session.get_page()

        # Navigate if URL provided
        if url:
            await page.goto(url, wait_until='networkidle')

        # Wait for element
        await page.wait_for_selector(selector, state=state, timeout=timeout)

        return {
            'success': True,
            'found': True,
            'selector': selector,
            'state': state
        }

    except Exception as e:
        logger.error(f"Wait error: {e}", exc_info=True)
        return {'success': False, 'error': str(e), 'found': False}


async def browser_close_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Close the browser session.

    Args:
        params: Empty dictionary (no parameters needed)

    Returns:
        Dictionary with:
            - success (bool): Whether close succeeded
    """
    try:
        session = await BrowserSession.get_instance()
        await session.close()
        return {'success': True, 'message': 'Browser session closed'}
    except Exception as e:
        return {'success': False, 'error': str(e)}


async def browser_pdf_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate PDF from current page or URL.

    Args:
        params: Dictionary containing:
            - url (str, optional): Navigate to URL first
            - save_path (str, required): Path to save PDF
            - format (str, optional): Paper format 'A4', 'Letter', etc
            - landscape (bool, optional): Landscape orientation
            - print_background (bool, optional): Include background graphics

    Returns:
        Dictionary with:
            - success (bool): Whether PDF was generated
            - saved_path (str): Path to saved PDF
            - size_bytes (int): PDF file size
            - error (str, optional): Error message if failed
    """
    if not PLAYWRIGHT_AVAILABLE:
        return {'success': False, 'error': 'Playwright not installed'}

    try:
        url = params.get('url')
        save_path = params.get('save_path')
        paper_format = params.get('format', 'A4')
        landscape = params.get('landscape', False)
        print_background = params.get('print_background', True)

        if not save_path:
            return {'success': False, 'error': 'save_path parameter is required'}

        session = await BrowserSession.get_instance()
        page = await session.get_page()

        # Navigate if URL provided
        if url:
            await page.goto(url, wait_until='networkidle')

        # Generate PDF
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        pdf_bytes = await page.pdf(
            format=paper_format,
            landscape=landscape,
            print_background=print_background
        )

        save_path.write_bytes(pdf_bytes)

        return {
            'success': True,
            'saved_path': str(save_path),
            'size_bytes': len(pdf_bytes),
            'url': page.url
        }

    except Exception as e:
        logger.error(f"PDF generation error: {e}", exc_info=True)
        return {'success': False, 'error': str(e)}


__all__ = [
    'browser_navigate_tool',
    'browser_screenshot_tool',
    'browser_fill_form_tool',
    'browser_click_tool',
    'browser_extract_tool',
    'browser_execute_js_tool',
    'browser_wait_tool',
    'browser_close_tool',
    'browser_pdf_tool'
]
