# Browser Automation - API Reference

## Contents

### Tools

| Function | Description |
|----------|-------------|
| [`browser_navigate_tool`](#browser_navigate_tool) | Navigate to a URL and optionally wait for content. |
| [`browser_screenshot_tool`](#browser_screenshot_tool) | Take a screenshot of the current page or a specific URL. |
| [`browser_fill_form_tool`](#browser_fill_form_tool) | Fill form fields on the current page. |
| [`browser_click_tool`](#browser_click_tool) | Click an element on the page. |
| [`browser_extract_tool`](#browser_extract_tool) | Extract data from the current page using selectors. |
| [`browser_execute_js_tool`](#browser_execute_js_tool) | Execute JavaScript on the current page. |
| [`browser_wait_tool`](#browser_wait_tool) | Wait for an element or condition. |
| [`browser_close_tool`](#browser_close_tool) | Close the browser session. |
| [`browser_pdf_tool`](#browser_pdf_tool) | Generate PDF from current page or URL. |
| [`browser_connect_cdp_tool`](#browser_connect_cdp_tool) | Connect to browser via Chrome DevTools Protocol (CDP). |
| [`browser_close_selenium_tool`](#browser_close_selenium_tool) | Close Selenium browser session. |
| [`browser_save_cookies_tool`](#browser_save_cookies_tool) | Save browser cookies to file for session persistence. |
| [`browser_load_cookies_tool`](#browser_load_cookies_tool) | Load cookies from file to restore a browser session. |
| [`browser_reset_tool`](#browser_reset_tool) | Reset browser state: clear cookies, cache, and create fresh context. |
| [`browser_accessibility_tree_tool`](#browser_accessibility_tree_tool) | Extract accessibility tree via CDP for reliable element discovery. |
| [`browser_dom_structure_tool`](#browser_dom_structure_tool) | Get DOM structure with visibility, bounds, and interactive state. |
| [`browser_cdp_click_tool`](#browser_cdp_click_tool) | Click at exact page coordinates via CDP. |

### Helper Functions

| Function | Description |
|----------|-------------|
| [`get_instance`](#get_instance) | Get or create browser session singleton. |
| [`get_page`](#get_page) | Get a browser page, creating browser if needed. |
| [`close`](#close) | Close browser session. |
| [`get_driver`](#get_driver) | Get Selenium WebDriver, creating if needed. |
| [`navigate`](#navigate) | Navigate to URL using Selenium. |
| [`screenshot`](#screenshot) | Take screenshot using Selenium. |
| [`click`](#click) | Click element using Selenium. |
| [`fill`](#fill) | Fill form field using Selenium. |
| [`extract_text`](#extract_text) | Extract text using Selenium. |
| [`execute_script`](#execute_script) | Execute JavaScript using Selenium. |
| [`get_accessibility_tree`](#get_accessibility_tree) | Extract accessibility tree via CDP for element discovery. |
| [`get_dom_structure`](#get_dom_structure) | Get DOM structure with visibility, bounds, and interactive state. |
| [`cdp_click`](#cdp_click) | Coordinate-based click via CDP Input. |

---

## `browser_navigate_tool`

Navigate to a URL and optionally wait for content.

**Parameters:**

- **url** (`str, required`): URL to navigate to
- **wait_for** (`str, optional`): CSS selector to wait for
- **wait_timeout** (`int, optional`): Timeout in ms (default: 30000)
- **screenshot** (`bool, optional`): Take screenshot after load (default: False)
- **extract_text** (`bool, optional`): Extract page text (default: True)
- **backend** (`str, optional`): 'playwright' or 'selenium' (default: auto)
- **cdp_url** (`str, optional`): CDP URL for Selenium (enables remote debugging)

**Returns:** Dictionary with: - success (bool): Whether navigation succeeded - url (str): Final URL after navigation - title (str): Page title - text (str): Extracted text content - screenshot_base64 (str, optional): Screenshot if requested - backend (str): Which backend was used - error (str, optional): Error message if failed

---

## `browser_screenshot_tool`

Take a screenshot of the current page or a specific URL.

**Parameters:**

- **url** (`str, optional`): URL to screenshot (uses current page if not provided)
- **full_page** (`bool, optional`): Capture full scrollable page (default: False)
- **selector** (`str, optional`): Screenshot specific element
- **save_path** (`str, optional`): Save to file path
- **format** (`str, optional`): 'png' or 'jpeg' (default: 'png')
- **quality** (`int, optional`): JPEG quality 0-100 (default: 80)

**Returns:** Dictionary with: - success (bool): Whether screenshot succeeded - screenshot_base64 (str): Base64 encoded image - saved_path (str, optional): Path if saved to file - width (int): Image width - height (int): Image height - error (str, optional): Error message if failed

---

## `browser_fill_form_tool`

Fill form fields on the current page.

**Parameters:**

- **url** (`str, optional`): Navigate to URL first
- **fields** (`dict, required`): Map of selector -> value to fill
- **submit_selector** (`str, optional`): Selector for submit button
- **wait_after_submit** (`str, optional`): Selector to wait for after submit
- **screenshot** (`bool, optional`): Take screenshot after filling

**Returns:** Dictionary with: - success (bool): Whether form was filled - fields_filled (int): Number of fields filled - submitted (bool): Whether form was submitted - url_after (str): URL after submission - error (str, optional): Error message if failed

---

## `browser_click_tool`

Click an element on the page.

**Parameters:**

- **selector** (`str, required`): CSS selector for element to click
- **url** (`str, optional`): Navigate to URL first
- **wait_for** (`str, optional`): Selector to wait for after click
- **double_click** (`bool, optional`): Double click instead (default: False)
- **right_click** (`bool, optional`): Right click instead (default: False)
- **screenshot** (`bool, optional`): Take screenshot after click

**Returns:** Dictionary with: - success (bool): Whether click succeeded - url_after (str): URL after click - error (str, optional): Error message if failed

---

## `browser_extract_tool`

Extract data from the current page using selectors.

**Parameters:**

- **url** (`str, optional`): Navigate to URL first
- **selectors** (`dict, required`): Map of name -> selector for data extraction
- **extract_all** (`bool, optional`): Extract all matches vs first (default: False)
- **attributes** (`dict, optional`): Map of name -> attribute to extract (default: innerText)

**Returns:** Dictionary with: - success (bool): Whether extraction succeeded - data (dict): Extracted data by name - error (str, optional): Error message if failed

---

## `browser_execute_js_tool`

Execute JavaScript on the current page.

**Parameters:**

- **script** (`str, required`): JavaScript code to execute
- **url** (`str, optional`): Navigate to URL first
- **args** (`list, optional`): Arguments to pass to the script

**Returns:** Dictionary with: - success (bool): Whether execution succeeded - result: Return value from script - error (str, optional): Error message if failed

---

## `browser_wait_tool`

Wait for an element or condition.

**Parameters:**

- **selector** (`str, optional`): CSS selector to wait for
- **state** (`str, optional`): 'visible', 'hidden', 'attached', 'detached'
- **timeout** (`int, optional`): Timeout in ms (default: 30000)
- **url** (`str, optional`): Navigate to URL first

**Returns:** Dictionary with: - success (bool): Whether wait completed - found (bool): Whether element was found - error (str, optional): Error message if failed

---

## `browser_close_tool`

Close the browser session.

**Parameters:**

- **params** (`Dict[str, Any]`)

**Returns:** Dictionary with: - success (bool): Whether close succeeded

---

## `browser_pdf_tool`

Generate PDF from current page or URL.

**Parameters:**

- **url** (`str, optional`): Navigate to URL first
- **save_path** (`str, required`): Path to save PDF
- **format** (`str, optional`): Paper format 'A4', 'Letter', etc
- **landscape** (`bool, optional`): Landscape orientation
- **print_background** (`bool, optional`): Include background graphics

**Returns:** Dictionary with: - success (bool): Whether PDF was generated - saved_path (str): Path to saved PDF - size_bytes (int): PDF file size - error (str, optional): Error message if failed

---

## `browser_connect_cdp_tool`

Connect to browser via Chrome DevTools Protocol (CDP).  Useful for connecting to Electron apps or existing browser instances with remote debugging enabled.  Start browser with: chrome --remote-debugging-port=9222

**Parameters:**

- **cdp_url** (`str, required`): CDP URL (e.g., 'localhost:9222')
- **test_navigate** (`str, optional`): URL to navigate to for testing

**Returns:** Dictionary with: - success (bool): Whether connection succeeded - cdp_url (str): CDP URL connected to - title (str): Page title if test_navigate provided - error (str, optional): Error message if failed

---

## `browser_close_selenium_tool`

Close Selenium browser session.

**Parameters:**

- **params** (`Dict[str, Any]`)

**Returns:** Dictionary with: - success (bool): Whether close succeeded

---

## `browser_save_cookies_tool`

Save browser cookies to file for session persistence.

**Parameters:**

- **save_path** (`str, optional`): Path to save cookies (default: ~/jotty/browser/cookies.json)
- **name** (`str, optional`): Session name for organizing multiple sessions

**Returns:** Dictionary with success, saved_path, cookie_count

---

## `browser_load_cookies_tool`

Load cookies from file to restore a browser session.

**Parameters:**

- **load_path** (`str, optional`): Path to cookies file
- **name** (`str, optional`): Session name to load (default: 'default')

**Returns:** Dictionary with success, loaded_path, cookie_count

---

## `browser_reset_tool`

Reset browser state: clear cookies, cache, and create fresh context.

**Parameters:**

- **keep_browser** (`bool, optional`): Keep browser open, just new context (default: True)

**Returns:** Dictionary with success, message

---

## `browser_accessibility_tree_tool`

Extract accessibility tree via CDP for reliable element discovery.  More reliable than CSS selectors — provides roles, names, and states of all accessible elements on the page.

**Parameters:**

- **max_depth** (`int, optional`): Tree depth limit (default: 5)
- **cdp_url** (`str, optional`): CDP URL for remote browser

**Returns:** Dictionary with success, nodes (list of accessible elements), total count

---

## `browser_dom_structure_tool`

Get DOM structure with visibility, bounds, and interactive state.  Lightweight alternative to full accessibility tree — uses JS evaluation for fast structural inspection.

**Parameters:**

- **selector** (`str, optional`): Root CSS selector (default: "body")
- **max_depth** (`int, optional`): Tree depth limit (default: 3)
- **cdp_url** (`str, optional`): CDP URL for remote browser

**Returns:** Dictionary with success, tree (nested DOM structure)

---

## `browser_cdp_click_tool`

Click at exact page coordinates via CDP.  Fallback for when CSS selectors fail — uses Chrome DevTools Protocol to dispatch mouse events at precise coordinates.

**Parameters:**

- **x** (`int, required`): X coordinate
- **y** (`int, required`): Y coordinate
- **cdp_url** (`str, optional`): CDP URL for remote browser

**Returns:** Dictionary with success, x, y coordinates clicked

---

## `get_instance`

Get or create browser session singleton.

**Returns:** `'BrowserSession'`

---

## `get_page`

Get a browser page, creating browser if needed.

**Parameters:**

- **new_context** (`bool`)

**Returns:** `'Page'`

---

## `close`

Close browser session.

---

## `get_driver`

Get Selenium WebDriver, creating if needed.

**Parameters:**

- **cdp_url** (`Optional[str]`)
- **headless** (`bool`)

**Returns:** Selenium WebDriver instance

---

## `navigate`

Navigate to URL using Selenium.

**Parameters:**

- **url** (`str`)
- **wait_for** (`Optional[str]`)
- **timeout** (`int`)

**Returns:** `Dict[str, Any]`

---

## `screenshot`

Take screenshot using Selenium.

**Parameters:**

- **selector** (`Optional[str]`)
- **full_page** (`bool`)

**Returns:** `bytes`

---

## `click`

Click element using Selenium.

**Parameters:**

- **selector** (`str`)
- **wait_after** (`Optional[str]`)
- **timeout** (`int`)

---

## `fill`

Fill form field using Selenium.

**Parameters:**

- **selector** (`str`)
- **value** (`str`)

---

## `extract_text`

Extract text using Selenium.

**Parameters:**

- **selector** (`Optional[str]`)

**Returns:** `str`

---

## `execute_script`

Execute JavaScript using Selenium.

**Parameters:**

- **script** (`str`)

**Returns:** `Any`

---

## `get_accessibility_tree`

Extract accessibility tree via CDP for element discovery.  Returns a structured tree of accessible elements with roles, names, and states — more reliable than CSS selectors for finding interactive elements.

**Parameters:**

- **max_depth** (`int`)

**Returns:** `Dict[str, Any]`

---

## `get_dom_structure`

Get DOM structure with visibility, bounds, and interactive state.  Uses lightweight JS evaluation instead of full CDP tree — faster and works across all Selenium configurations.

**Parameters:**

- **selector** (`str`)
- **max_depth** (`int`)

**Returns:** `Dict[str, Any]`

---

## `cdp_click`

Coordinate-based click via CDP Input.dispatchMouseEvent.  Fallback when CSS selectors fail — clicks at exact page coordinates using Chrome DevTools Protocol.

**Parameters:**

- **x** (`int`)
- **y** (`int`)

**Returns:** `Dict[str, Any]`
