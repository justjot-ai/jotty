# Android Automation - API Reference

## Contents

### Tools

| Function | Description |
|----------|-------------|
| [`device_connect_tool`](#device_connect_tool) | Connect to an Android device or emulator. |
| [`tap_tool`](#tap_tool) | Tap at exact screen coordinates. |
| [`long_press_tool`](#long_press_tool) | Long press at screen coordinates. |
| [`swipe_tool`](#swipe_tool) | Swipe from one point to another on screen. |
| [`type_text_tool`](#type_text_tool) | Type text into the currently focused input field. |
| [`press_key_tool`](#press_key_tool) | Press a device key (home, back, enter, recent, volume_up, etc. |
| [`screenshot_tool`](#screenshot_tool) | Capture a screenshot of the current screen. |
| [`launch_app_tool`](#launch_app_tool) | Launch an Android app by package name. |
| [`stop_app_tool`](#stop_app_tool) | Force stop an Android app. |
| [`get_ui_tree_tool`](#get_ui_tree_tool) | Dump the UI accessibility tree for the current screen. |
| [`find_element_tool`](#find_element_tool) | Find a UI element by text, resource ID, class, or description. |
| [`tap_element_tool`](#tap_element_tool) | Find a UI element and tap it. |
| [`wait_for_element_tool`](#wait_for_element_tool) | Wait for a UI element to appear on screen. |
| [`scroll_tool`](#scroll_tool) | Scroll the screen in a direction. |
| [`get_current_app_tool`](#get_current_app_tool) | Get info about the current foreground app. |
| [`device_info_tool`](#device_info_tool) | Get device hardware and software info. |

### Helper Functions

| Function | Description |
|----------|-------------|
| [`get_instance`](#get_instance) | Get or create device session singleton. |
| [`reset`](#reset) | Reset singleton (used in tests). |
| [`connect`](#connect) | Connect to an Android device. |
| [`device`](#device) | Get the connected device, connecting if needed. |
| [`screen_size`](#screen_size) | Get screen dimensions (width, height). |
| [`tap`](#tap) | Tap at exact coordinates. |
| [`long_press`](#long_press) | Long press at coordinates. |
| [`swipe`](#swipe) | Swipe from (x1,y1) to (x2,y2). |
| [`type_text`](#type_text) | Type text into the currently focused field. |
| [`press_key`](#press_key) | Press a device key (home, back, enter, recent, volume_up, etc. |
| [`screenshot`](#screenshot) | Capture screenshot as base64 PNG. |
| [`launch_app`](#launch_app) | Launch an app by package name. |
| [`stop_app`](#stop_app) | Force stop an app. |
| [`current_app`](#current_app) | Get the current foreground app info. |
| [`dump_ui_tree`](#dump_ui_tree) | Dump UI hierarchy as structured JSON tree. |
| [`find_element`](#find_element) | Find UI element by attributes and return its info + bounds. |
| [`tap_element`](#tap_element) | Find an element and tap it. |
| [`wait_for_element`](#wait_for_element) | Wait for an element to appear on screen. |
| [`scroll`](#scroll) | Scroll the screen in a direction. |
| [`get_device_info`](#get_device_info) | Get device info (model, serial, screen size, battery, etc. |
| [`parse`](#parse) | Parse UIAutomator XML hierarchy into structured tree. |

---

## `device_connect_tool`

Connect to an Android device or emulator.

**Parameters:**

- **serial** (`str, optional`): Device serial (e.g. 'emulator-5554')
- **url** (`str, optional`): Remote device URL

**Returns:** Dictionary with: - serial (str): Connected device serial - screen_width (int): Screen width in pixels - screen_height (int): Screen height in pixels - model (str): Device model name

---

## `tap_tool`

Tap at exact screen coordinates.

**Parameters:**

- **x** (`int, required`): X coordinate
- **y** (`int, required`): Y coordinate

**Returns:** Dictionary with: - x (int): X coordinate tapped - y (int): Y coordinate tapped - action (str): Action performed

---

## `long_press_tool`

Long press at screen coordinates.

**Parameters:**

- **x** (`int, required`): X coordinate
- **y** (`int, required`): Y coordinate
- **duration** (`float, optional`): Hold duration in seconds (default: 1.0)

**Returns:** Dictionary with: - x (int): X coordinate - y (int): Y coordinate - duration (float): Hold duration - action (str): Action performed

---

## `swipe_tool`

Swipe from one point to another on screen.

**Parameters:**

- **x1** (`int, required`): Start X coordinate
- **y1** (`int, required`): Start Y coordinate
- **x2** (`int, required`): End X coordinate
- **y2** (`int, required`): End Y coordinate
- **duration** (`float, optional`): Swipe duration in seconds (default: 0.5)

**Returns:** Dictionary with: - from (list): Start coordinates [x1, y1] - to (list): End coordinates [x2, y2] - duration (float): Swipe duration - action (str): Action performed

---

## `type_text_tool`

Type text into the currently focused input field.

**Parameters:**

- **text** (`str, required`): Text to type
- **clear_first** (`bool, optional`): Clear field before typing (default: false)

**Returns:** Dictionary with: - text (str): Text typed - cleared (bool): Whether field was cleared first - action (str): Action performed

---

## `press_key_tool`

Press a device key (home, back, enter, recent, volume_up, etc.).

**Parameters:**

- **key** (`str, required`): Key name (home, back, enter, recent, volume_up, volume_down, power, menu)

**Returns:** Dictionary with: - key (str): Key pressed - action (str): Action performed

---

## `screenshot_tool`

Capture a screenshot of the current screen.

**Parameters:**

- **scale** (`float, optional`): Scale factor, 0.5 = half resolution (default: 0.5)

**Returns:** Dictionary with: - image_base64 (str): Base64-encoded PNG screenshot - width (int): Image width - height (int): Image height - format (str): Image format

---

## `launch_app_tool`

Launch an Android app by package name.

**Parameters:**

- **package** (`str, required`): Package name (e.g. 'com.google.android.youtube')
- **activity** (`str, optional`): Specific activity to launch

**Returns:** Dictionary with: - package (str): Package launched - activity (str): Activity launched - action (str): Action performed

---

## `stop_app_tool`

Force stop an Android app.

**Parameters:**

- **package** (`str, required`): Package name to stop

**Returns:** Dictionary with: - package (str): Package stopped - action (str): Action performed

---

## `get_ui_tree_tool`

Dump the UI accessibility tree for the current screen.  Returns a structured tree with roles, names, bounds, and interactive states. Also provides a flat list of all interactive (clickable/focusable) elements with their center coordinates for easy tapping.

**Parameters:**

- **max_depth** (`int, optional`): Maximum tree depth (default: 10)

**Returns:** Dictionary with: - tree (dict): Nested UI tree with role, name, bounds, children - node_count (int): Total nodes parsed - interactive_elements (list): Flat list of tappable elements with center coords

---

## `find_element_tool`

Find a UI element by text, resource ID, class, or description.  Returns element info including bounds and center coordinates for tapping.

**Parameters:**

- **text** (`str, optional`): Element text content
- **resource_id** (`str, optional`): Android resource ID
- **class_name** (`str, optional`): Widget class name
- **description** (`str, optional`): Content description

**Returns:** Dictionary with: - found (bool): Whether element was found - text (str): Element text - class (str): Widget class name - bounds (dict): Element bounds {left, top, right, bottom} - center (dict): Center coordinates {x, y} for tapping - clickable (bool): Whether element is clickable - enabled (bool): Whether element is enabled

---

## `tap_element_tool`

Find a UI element and tap it. Combines find + tap in one call.

**Parameters:**

- **text** (`str, optional`): Element text to find and tap
- **resource_id** (`str, optional`): Resource ID to find and tap
- **description** (`str, optional`): Content description to find and tap

**Returns:** Dictionary with: - tapped (bool): Whether element was found and tapped - text (str): Element text - center (dict): Coordinates tapped {x, y}

---

## `wait_for_element_tool`

Wait for a UI element to appear on screen.

**Parameters:**

- **text** (`str, optional`): Element text to wait for
- **resource_id** (`str, optional`): Resource ID to wait for
- **timeout** (`float, optional`): Max wait seconds (default: 10)

**Returns:** Dictionary with: - found (bool): Whether element appeared within timeout - criteria (dict): Search criteria used - timeout (float): Timeout value used

---

## `scroll_tool`

Scroll the screen in a direction.

**Parameters:**

- **direction** (`str, optional`): 'up', 'down', 'left', 'right' (default: 'down')
- **steps** (`int, optional`): Scroll steps/speed (default: 10)

**Returns:** Dictionary with: - direction (str): Scroll direction - steps (int): Scroll steps - action (str): Action performed

---

## `get_current_app_tool`

Get info about the current foreground app.

**Parameters:**

- **params** (`Dict[str, Any]`)

**Returns:** Dictionary with: - package (str): Current app package name - activity (str): Current activity name

---

## `device_info_tool`

Get device hardware and software info.

**Parameters:**

- **params** (`Dict[str, Any]`)

**Returns:** Dictionary with: - serial (str): Device serial - model (str): Device model - brand (str): Device brand - sdk (int): Android SDK version - screen_width (int): Screen width - screen_height (int): Screen height

---

## `get_instance`

Get or create device session singleton.

**Returns:** `'AndroidDevice'`

---

## `reset`

Reset singleton (used in tests).

---

## `connect`

Connect to an Android device.

**Parameters:**

- **serial** (`Optional[str]`)
- **url** (`Optional[str]`)

**Returns:** The uiautomator2 device object.

---

## `device`

Get the connected device, connecting if needed.

---

## `screen_size`

Get screen dimensions (width, height).

**Returns:** `Tuple[int, int]`

---

## `tap`

Tap at exact coordinates.

**Parameters:**

- **x** (`int`)
- **y** (`int`)

**Returns:** `Dict[str, Any]`

---

## `long_press`

Long press at coordinates.

**Parameters:**

- **x** (`int`)
- **y** (`int`)
- **duration** (`float`)

**Returns:** `Dict[str, Any]`

---

## `swipe`

Swipe from (x1,y1) to (x2,y2).

**Parameters:**

- **x1** (`int`)
- **y1** (`int`)
- **x2** (`int`)
- **y2** (`int`)
- **duration** (`float`)

**Returns:** `Dict[str, Any]`

---

## `type_text`

Type text into the currently focused field.

**Parameters:**

- **text** (`str`)
- **clear_first** (`bool`)

**Returns:** `Dict[str, Any]`

---

## `press_key`

Press a device key (home, back, enter, recent, volume_up, etc.).

**Parameters:**

- **key** (`str`)

**Returns:** `Dict[str, Any]`

---

## `screenshot`

Capture screenshot as base64 PNG.

**Parameters:**

- **scale** (`float`)

**Returns:** Dict with base64 image and dimensions.

---

## `launch_app`

Launch an app by package name.

**Parameters:**

- **package** (`str`)
- **activity** (`Optional[str]`)
- **wait** (`bool`)

**Returns:** `Dict[str, Any]`

---

## `stop_app`

Force stop an app.

**Parameters:**

- **package** (`str`)

**Returns:** `Dict[str, Any]`

---

## `current_app`

Get the current foreground app info.

**Returns:** `Dict[str, Any]`

---

## `dump_ui_tree`

Dump UI hierarchy as structured JSON tree.  Parses UIAutomator2 XML hierarchy into a structured tree matching the format of browser_accessibility_tree_tool: {role, name, bounds, clickable, focusable, children}

**Parameters:**

- **max_depth** (`int`)

**Returns:** Dict with tree, node_count, and flat interactive_elements list.

---

## `find_element`

Find UI element by attributes and return its info + bounds.

**Parameters:**

- **text** (`Optional[str]`)
- **resource_id** (`Optional[str]`)
- **class_name** (`Optional[str]`)
- **description** (`Optional[str]`)
- **clickable** (`Optional[bool]`)

**Returns:** Dict with element info (bounds, text, class, resource_id, center).

---

## `tap_element`

Find an element and tap it.  Combines find + tap in one call for efficiency.

**Parameters:**

- **text** (`Optional[str]`)
- **resource_id** (`Optional[str]`)
- **description** (`Optional[str]`)

**Returns:** `Dict[str, Any]`

---

## `wait_for_element`

Wait for an element to appear on screen.

**Parameters:**

- **text** (`Optional[str]`)
- **resource_id** (`Optional[str]`)
- **timeout** (`float`)

**Returns:** `Dict[str, Any]`

---

## `scroll`

Scroll the screen in a direction.

**Parameters:**

- **direction** (`str`)
- **steps** (`int`)

**Returns:** `Dict[str, Any]`

---

## `get_device_info`

Get device info (model, serial, screen size, battery, etc.).

**Returns:** `Dict[str, Any]`

---

## `parse`

Parse UIAutomator XML hierarchy into structured tree.

**Parameters:**

- **xml_str** (`str`)
- **max_depth** (`int`)

**Returns:** Dict with: - tree (dict): Nested node tree - node_count (int): Total nodes parsed - interactive_elements (list): Flat list of clickable/editable elements
