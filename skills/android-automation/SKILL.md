---
name: android-automation
description: "Full Android device control: touch, swipe, type, screenshot, UI tree inspection, app lifecycle management, and element-based interaction. Designed for MAS-Bench hybrid mobile agent evaluation. Mirrors browser-automation architecture but targets Android."
---

# Android Automation

Mobile GUI automation for Android devices and emulators via ADB + uiautomator2.

## Description

Full Android device control: touch, swipe, type, screenshot, UI tree inspection, app lifecycle management, and element-based interaction. Designed for MAS-Bench hybrid mobile agent evaluation. Mirrors browser-automation architecture but targets Android.

## Type
base

## Capabilities
- automation
- mobile
- gui

## Executor Type
gui

## Use When
User wants to automate Android app interactions, take mobile screenshots, inspect mobile UI elements, or run hybrid GUI+API mobile tasks.

## Tools

### device_connect_tool
Connect to an Android device or emulator.

**Parameters:**
- `serial` (str, optional): Device serial (e.g. 'emulator-5554')
- `url` (str, optional): Remote device URL

**Returns:**
- `serial`, `screen_width`, `screen_height`, `model`, `brand`

### tap_tool
Tap at exact screen coordinates.

**Parameters:**
- `x` (int, required): X coordinate
- `y` (int, required): Y coordinate

**Returns:**
- `x`, `y`, `action`

### long_press_tool
Long press at screen coordinates.

**Parameters:**
- `x` (int, required): X coordinate
- `y` (int, required): Y coordinate
- `duration` (float, optional): Hold duration seconds (default: 1.0)

**Returns:**
- `x`, `y`, `duration`, `action`

### swipe_tool
Swipe from one point to another.

**Parameters:**
- `x1` (int, required): Start X
- `y1` (int, required): Start Y
- `x2` (int, required): End X
- `y2` (int, required): End Y
- `duration` (float, optional): Duration seconds (default: 0.5)

**Returns:**
- `from`, `to`, `duration`, `action`

### type_text_tool
Type text into focused input field.

**Parameters:**
- `text` (str, required): Text to type
- `clear_first` (bool, optional): Clear field first (default: false)

**Returns:**
- `text`, `cleared`, `action`

### press_key_tool
Press a device key.

**Parameters:**
- `key` (str, required): Key name (home, back, enter, recent, volume_up, volume_down, power, menu)

**Returns:**
- `key`, `action`

### screenshot_tool
Capture screenshot as base64 PNG.

**Parameters:**
- `scale` (float, optional): Scale factor (default: 0.5)

**Returns:**
- `image_base64`, `width`, `height`, `format`

### launch_app_tool
Launch an Android app.

**Parameters:**
- `package` (str, required): Package name (e.g. 'com.google.android.youtube')
- `activity` (str, optional): Specific activity

**Returns:**
- `package`, `activity`, `action`

### stop_app_tool
Force stop an app.

**Parameters:**
- `package` (str, required): Package name

**Returns:**
- `package`, `action`

### get_ui_tree_tool
Dump UI accessibility tree for current screen.

**Parameters:**
- `max_depth` (int, optional): Maximum depth (default: 10)

**Returns:**
- `tree` (dict), `node_count` (int), `interactive_elements` (list)

### find_element_tool
Find a UI element by text, resource ID, class, or description.

**Parameters:**
- `text` (str, optional): Element text
- `resource_id` (str, optional): Android resource ID
- `class_name` (str, optional): Widget class
- `description` (str, optional): Content description

**Returns:**
- `found`, `text`, `class`, `bounds`, `center`, `clickable`, `enabled`

### tap_element_tool
Find and tap a UI element (combined find + tap).

**Parameters:**
- `text` (str, optional): Element text to tap
- `resource_id` (str, optional): Resource ID to tap
- `description` (str, optional): Description to tap

**Returns:**
- `tapped`, `text`, `center`

### wait_for_element_tool
Wait for a UI element to appear.

**Parameters:**
- `text` (str, optional): Text to wait for
- `resource_id` (str, optional): Resource ID to wait for
- `timeout` (float, optional): Max wait seconds (default: 10)

**Returns:**
- `found`, `criteria`, `timeout`

### scroll_tool
Scroll the screen in a direction.

**Parameters:**
- `direction` (str, optional): up/down/left/right (default: down)
- `steps` (int, optional): Scroll speed (default: 10)

**Returns:**
- `direction`, `steps`, `action`

### get_current_app_tool
Get current foreground app info.

**Returns:**
- `package`, `activity`

### device_info_tool
Get device hardware and software info.

**Returns:**
- `serial`, `model`, `brand`, `sdk`, `screen_width`, `screen_height`

## Triggers
- "android automation"

## Category
workflow-automation
