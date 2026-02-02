# Browser Automation

Full browser automation with Playwright and Selenium backends.

## Description

Comprehensive browser automation supporting navigation, screenshots, form filling, clicking, data extraction, JavaScript execution, and PDF generation. Supports both Playwright (async, faster) and Selenium with CDP (for Electron embedding).

## Features

- Navigate to URLs with JS rendering
- Take screenshots (full page or elements)
- Fill forms (text, select, checkbox, file upload)
- Click elements (single, double, right-click)
- Extract data using CSS selectors
- Execute JavaScript
- Wait for elements/conditions
- Generate PDFs from pages
- Connect via Chrome DevTools Protocol (CDP)

## Backends

### Playwright (Default)
- Async execution
- Faster page loads
- Better for headless automation
- Install: `pip install playwright && playwright install chromium`

### Selenium with CDP
- For Electron app embedding
- Remote browser debugging
- Connect to existing browser instances
- Install: `pip install selenium`
- Set `BROWSER_BACKEND=selenium` or use `cdp_url` parameter

## Tools

- `browser_navigate_tool`: Navigate to URL
- `browser_screenshot_tool`: Take screenshots
- `browser_fill_form_tool`: Fill form fields
- `browser_click_tool`: Click elements
- `browser_extract_tool`: Extract data
- `browser_execute_js_tool`: Run JavaScript
- `browser_wait_tool`: Wait for elements
- `browser_close_tool`: Close Playwright session
- `browser_pdf_tool`: Generate PDF
- `browser_connect_cdp_tool`: Connect via CDP
- `browser_close_selenium_tool`: Close Selenium session

## Usage

```python
# Basic navigation (Playwright)
result = await browser_navigate_tool({
    'url': 'https://example.com',
    'screenshot': True,
    'extract_text': True
})

# Selenium with CDP (for Electron)
result = browser_connect_cdp_tool({
    'cdp_url': 'localhost:9222',
    'test_navigate': 'https://example.com'
})

# Screenshot with Selenium backend
result = await browser_navigate_tool({
    'url': 'https://example.com',
    'backend': 'selenium',
    'screenshot': True
})
```

## Environment Variables

- `BROWSER_BACKEND`: Default backend ('playwright' or 'selenium')

## Dependencies

- playwright (default backend)
- selenium (alternative backend for CDP)
