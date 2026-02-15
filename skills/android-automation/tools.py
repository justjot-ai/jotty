"""
Android Automation Skill
========================

Mobile GUI automation via ADB + uiautomator2 for MAS-Bench hybrid agents.
Provides touch/swipe/type primitives, screenshot capture, UI tree parsing,
app lifecycle, and element-based interaction.

Mirrors browser-automation architecture but targets Android devices/emulators.

Requires:
    pip install uiautomator2 Pillow

Device connection:
    - USB: adb devices → auto-detected
    - Emulator: ANDROID_SERIAL=emulator-5554
    - Remote: ANDROID_DEVICE_URL=http://192.168.1.x:7912
"""

import atexit
import base64
import io
import json
import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

from Jotty.core.infrastructure.utils.skill_status import SkillStatus
from Jotty.core.infrastructure.utils.tool_helpers import (
    async_tool_wrapper,
    tool_error,
    tool_response,
)

logger = logging.getLogger(__name__)
status = SkillStatus("android-automation")

# Lazy import uiautomator2
U2_AVAILABLE = False
try:
    import uiautomator2 as u2

    U2_AVAILABLE = True
except ImportError:
    logger.info("uiautomator2 not installed. Run: pip install uiautomator2")

# Optional Pillow for screenshot processing
PIL_AVAILABLE = False
try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    pass


# =============================================================================
# AndroidDevice — Singleton session manager (mirrors BrowserSession pattern)
# =============================================================================


class AndroidDevice:
    """Manages a connected Android device/emulator for reuse across tool calls.

    Singleton pattern ensures only one active device connection at a time.
    Wraps uiautomator2 for high-level Android interaction.
    """

    _instance: Optional["AndroidDevice"] = None
    _device = None  # u2.Device
    _serial: Optional[str] = None
    _screen_size: Optional[Tuple[int, int]] = None

    @classmethod
    def get_instance(cls) -> "AndroidDevice":
        """Get or create device session singleton."""
        if cls._instance is None:
            cls._instance = AndroidDevice()
            atexit.register(cls._cleanup_sync)
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset singleton (used in tests)."""
        if cls._instance and cls._device:
            try:
                cls._device = None
            except Exception:
                pass
        cls._instance = None
        cls._device = None
        cls._serial = None
        cls._screen_size = None

    @classmethod
    def _cleanup_sync(cls):
        """Cleanup on interpreter exit."""
        cls.reset()

    def connect(self, serial: Optional[str] = None, url: Optional[str] = None):
        """Connect to an Android device.

        Args:
            serial: Device serial (e.g. 'emulator-5554'). Auto-detected if None.
            url: Remote device URL (e.g. 'http://192.168.1.x:7912').

        Returns:
            The uiautomator2 device object.
        """
        if not U2_AVAILABLE:
            raise RuntimeError("uiautomator2 not installed. Run: pip install uiautomator2")

        if self._device is not None:
            return self._device

        target = (
            url
            or serial
            or os.environ.get("ANDROID_DEVICE_URL")
            or os.environ.get("ANDROID_SERIAL")
            or None  # u2.connect() auto-detects
        )

        logger.info(f"Connecting to Android device: {target or 'auto-detect'}")
        self._device = u2.connect(target)
        self._serial = target or "auto"

        # Cache screen size
        info = self._device.info
        self._screen_size = (
            info.get("displayWidth", 1080),
            info.get("displayHeight", 2400),
        )
        logger.info(
            f"Connected: {self._device.serial} ({self._screen_size[0]}x{self._screen_size[1]})"
        )
        return self._device

    @property
    def device(self):
        """Get the connected device, connecting if needed."""
        if self._device is None:
            self.connect()
        return self._device

    @property
    def screen_size(self) -> Tuple[int, int]:
        """Get screen dimensions (width, height)."""
        if self._screen_size is None:
            _ = self.device  # triggers connect
        return self._screen_size

    # -- Core Actions --

    def tap(self, x: int, y: int) -> Dict[str, Any]:
        """Tap at exact coordinates."""
        self.device.click(x, y)
        return {"x": x, "y": y, "action": "tap"}

    def long_press(self, x: int, y: int, duration: float = 1.0) -> Dict[str, Any]:
        """Long press at coordinates."""
        self.device.long_click(x, y, duration=duration)
        return {"x": x, "y": y, "duration": duration, "action": "long_press"}

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration: float = 0.5) -> Dict[str, Any]:
        """Swipe from (x1,y1) to (x2,y2)."""
        self.device.swipe(x1, y1, x2, y2, duration=duration)
        return {"from": [x1, y1], "to": [x2, y2], "duration": duration, "action": "swipe"}

    def type_text(self, text: str, clear_first: bool = False) -> Dict[str, Any]:
        """Type text into the currently focused field."""
        if clear_first:
            self.device.clear_text()
        self.device.send_keys(text)
        return {"text": text, "cleared": clear_first, "action": "type"}

    def press_key(self, key: str) -> Dict[str, Any]:
        """Press a device key (home, back, enter, recent, volume_up, etc.)."""
        self.device.press(key)
        return {"key": key, "action": "press"}

    def screenshot(self, scale: float = 0.5) -> Dict[str, Any]:
        """Capture screenshot as base64 PNG.

        Args:
            scale: Scale factor (0.5 = half resolution for faster transfer).

        Returns:
            Dict with base64 image and dimensions.
        """
        img = self.device.screenshot()

        if scale != 1.0 and PIL_AVAILABLE and img:
            w, h = img.size
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {
            "image_base64": b64,
            "width": img.size[0],
            "height": img.size[1],
            "format": "png",
        }

    def launch_app(
        self, package: str, activity: Optional[str] = None, wait: bool = True
    ) -> Dict[str, Any]:
        """Launch an app by package name.

        Args:
            package: Android package name (e.g. 'com.google.android.youtube').
            activity: Optional activity to launch.
            wait: Wait for app to launch.
        """
        if activity:
            self.device.app_start(package, activity=activity, wait=wait)
        else:
            self.device.app_start(package, wait=wait)
        return {"package": package, "activity": activity, "action": "launch"}

    def stop_app(self, package: str) -> Dict[str, Any]:
        """Force stop an app."""
        self.device.app_stop(package)
        return {"package": package, "action": "stop"}

    def current_app(self) -> Dict[str, Any]:
        """Get the current foreground app info."""
        info = self.device.app_current()
        return {
            "package": info.get("package", ""),
            "activity": info.get("activity", ""),
        }

    # -- UI Tree --

    def dump_ui_tree(self, max_depth: int = 10) -> Dict[str, Any]:
        """Dump UI hierarchy as structured JSON tree.

        Parses UIAutomator2 XML hierarchy into a structured tree matching
        the format of browser_accessibility_tree_tool:
        {role, name, bounds, clickable, focusable, children}

        Args:
            max_depth: Maximum tree depth to parse.

        Returns:
            Dict with tree, node_count, and flat interactive_elements list.
        """
        xml_str = self.device.dump_hierarchy()
        return UITreeParser.parse(xml_str, max_depth=max_depth)

    def find_element(
        self,
        text: Optional[str] = None,
        resource_id: Optional[str] = None,
        class_name: Optional[str] = None,
        description: Optional[str] = None,
        clickable: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Find UI element by attributes and return its info + bounds.

        Args:
            text: Element text content.
            resource_id: Android resource ID.
            class_name: Widget class name.
            description: Content description.
            clickable: Filter by clickability.

        Returns:
            Dict with element info (bounds, text, class, resource_id, center).
        """
        kwargs = {}
        if text is not None:
            kwargs["text"] = text
        if resource_id is not None:
            kwargs["resourceId"] = resource_id
        if class_name is not None:
            kwargs["className"] = class_name
        if description is not None:
            kwargs["description"] = description
        if clickable is not None:
            kwargs["clickable"] = clickable

        if not kwargs:
            return {"found": False, "error": "At least one search criteria required"}

        el = self.device(**kwargs)
        if not el.exists:
            return {"found": False, "criteria": kwargs}

        info = el.info
        bounds = info.get("bounds", {})
        center_x = (bounds.get("left", 0) + bounds.get("right", 0)) // 2
        center_y = (bounds.get("top", 0) + bounds.get("bottom", 0)) // 2

        return {
            "found": True,
            "text": info.get("text", ""),
            "class": info.get("className", ""),
            "resource_id": info.get("resourceId", ""),
            "description": info.get("contentDescription", ""),
            "bounds": bounds,
            "center": {"x": center_x, "y": center_y},
            "clickable": info.get("clickable", False),
            "focusable": info.get("focusable", False),
            "selected": info.get("selected", False),
            "checked": info.get("checked", False),
            "enabled": info.get("enabled", True),
        }

    def tap_element(
        self,
        text: Optional[str] = None,
        resource_id: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Find an element and tap it.

        Combines find + tap in one call for efficiency.
        """
        kwargs = {}
        if text is not None:
            kwargs["text"] = text
        if resource_id is not None:
            kwargs["resourceId"] = resource_id
        if description is not None:
            kwargs["description"] = description

        if not kwargs:
            return {"tapped": False, "error": "At least one search criteria required"}

        el = self.device(**kwargs)
        if not el.exists:
            return {"tapped": False, "criteria": kwargs, "error": "Element not found"}

        el.click()
        info = el.info
        bounds = info.get("bounds", {})
        center_x = (bounds.get("left", 0) + bounds.get("right", 0)) // 2
        center_y = (bounds.get("top", 0) + bounds.get("bottom", 0)) // 2

        return {
            "tapped": True,
            "text": info.get("text", ""),
            "center": {"x": center_x, "y": center_y},
        }

    def wait_for_element(
        self, text: Optional[str] = None, resource_id: Optional[str] = None, timeout: float = 10.0
    ) -> Dict[str, Any]:
        """Wait for an element to appear on screen.

        Args:
            text: Element text to wait for.
            resource_id: Resource ID to wait for.
            timeout: Max wait time in seconds.
        """
        kwargs = {}
        if text is not None:
            kwargs["text"] = text
        if resource_id is not None:
            kwargs["resourceId"] = resource_id

        if not kwargs:
            return {"found": False, "error": "At least one criteria required"}

        el = self.device(**kwargs)
        found = el.wait(timeout=timeout)

        return {
            "found": found,
            "criteria": kwargs,
            "timeout": timeout,
        }

    def scroll(self, direction: str = "down", steps: int = 10) -> Dict[str, Any]:
        """Scroll the screen in a direction.

        Args:
            direction: 'up', 'down', 'left', 'right'
            steps: Number of steps (controls speed).
        """
        w, h = self.screen_size
        cx, cy = w // 2, h // 2

        swipe_map = {
            "down": (cx, cy + h // 4, cx, cy - h // 4),
            "up": (cx, cy - h // 4, cx, cy + h // 4),
            "left": (cx + w // 4, cy, cx - w // 4, cy),
            "right": (cx - w // 4, cy, cx + w // 4, cy),
        }
        coords = swipe_map.get(direction, swipe_map["down"])
        self.device.swipe(*coords, steps=steps)
        return {"direction": direction, "steps": steps, "action": "scroll"}

    def get_device_info(self) -> Dict[str, Any]:
        """Get device info (model, serial, screen size, battery, etc.)."""
        info = self.device.info
        dev_info = self.device.device_info
        return {
            "serial": self._serial,
            "model": dev_info.get("model", ""),
            "brand": dev_info.get("brand", ""),
            "sdk": info.get("sdkInt", 0),
            "screen_width": info.get("displayWidth", 0),
            "screen_height": info.get("displayHeight", 0),
            "natural_orientation": info.get("naturalOrientation", True),
            "product_name": info.get("productName", ""),
        }


# =============================================================================
# UITreeParser — Convert UIAutomator XML hierarchy to structured JSON
# =============================================================================


class UITreeParser:
    """Parse UIAutomator2 XML hierarchy into structured accessibility tree.

    Output format matches browser_accessibility_tree_tool for uniformity:
    {role, name, bounds, clickable, focusable, children}
    """

    # Map Android widget class names to semantic roles
    _CLASS_TO_ROLE = {
        "android.widget.Button": "button",
        "android.widget.ImageButton": "button",
        "android.widget.EditText": "textbox",
        "android.widget.TextView": "text",
        "android.widget.ImageView": "image",
        "android.widget.CheckBox": "checkbox",
        "android.widget.RadioButton": "radio",
        "android.widget.Switch": "switch",
        "android.widget.ToggleButton": "switch",
        "android.widget.Spinner": "combobox",
        "android.widget.SeekBar": "slider",
        "android.widget.ProgressBar": "progressbar",
        "android.widget.ListView": "list",
        "android.widget.RecyclerView": "list",
        "android.widget.ScrollView": "scrollbar",
        "android.widget.HorizontalScrollView": "scrollbar",
        "android.widget.TabWidget": "tablist",
        "android.widget.ViewPager": "tabpanel",
        "android.widget.Toolbar": "toolbar",
        "android.webkit.WebView": "document",
        "android.widget.FrameLayout": "group",
        "android.widget.LinearLayout": "group",
        "android.widget.RelativeLayout": "group",
        "android.view.View": "generic",
    }

    @classmethod
    def parse(cls, xml_str: str, max_depth: int = 10) -> Dict[str, Any]:
        """Parse UIAutomator XML hierarchy into structured tree.

        Args:
            xml_str: Raw XML string from device.dump_hierarchy().
            max_depth: Maximum tree depth.

        Returns:
            Dict with:
                - tree (dict): Nested node tree
                - node_count (int): Total nodes parsed
                - interactive_elements (list): Flat list of clickable/editable elements
        """
        try:
            root = ET.fromstring(xml_str)
        except ET.ParseError as e:
            return {
                "tree": {},
                "node_count": 0,
                "interactive_elements": [],
                "error": f"XML parse error: {e}",
            }

        interactive = []
        node_count = [0]

        tree = cls._parse_node(root, 0, max_depth, interactive, node_count)

        return {
            "tree": tree,
            "node_count": node_count[0],
            "interactive_elements": interactive,
        }

    @classmethod
    def _parse_node(
        cls, node: ET.Element, depth: int, max_depth: int, interactive: list, node_count: list
    ) -> Dict[str, Any]:
        """Recursively parse an XML node into structured dict."""
        node_count[0] += 1

        attribs = node.attrib
        class_name = attribs.get("class", "")
        text = attribs.get("text", "")
        desc = attribs.get("content-desc", "")
        resource_id = attribs.get("resource-id", "")
        bounds_str = attribs.get("bounds", "")
        clickable = attribs.get("clickable", "false") == "true"
        focusable = attribs.get("focusable", "false") == "true"
        enabled = attribs.get("enabled", "true") == "true"
        selected = attribs.get("selected", "false") == "true"
        checked = attribs.get("checked", "false") == "true"
        scrollable = attribs.get("scrollable", "false") == "true"
        long_clickable = attribs.get("long-clickable", "false") == "true"

        # Parse bounds string "[x1,y1][x2,y2]"
        bounds = cls._parse_bounds(bounds_str)

        # Determine semantic role
        role = cls._CLASS_TO_ROLE.get(class_name, "generic")
        if not class_name:
            role = "group"

        # Compute display name (prefer text > content-desc > resource-id)
        name = text or desc or ""
        if not name and resource_id:
            # Extract readable name from resource ID (com.app:id/btn_submit -> btn_submit)
            name = resource_id.split("/")[-1] if "/" in resource_id else resource_id

        result = {
            "role": role,
            "name": name,
            "class": class_name,
            "bounds": bounds,
            "clickable": clickable,
            "focusable": focusable,
            "enabled": enabled,
            "scrollable": scrollable,
        }

        # Add optional fields only when true (saves space)
        if selected:
            result["selected"] = True
        if checked:
            result["checked"] = True
        if long_clickable:
            result["long_clickable"] = True
        if resource_id:
            result["resource_id"] = resource_id
        if desc and desc != name:
            result["description"] = desc

        # Track interactive elements in flat list
        is_interactive = (
            clickable
            or focusable
            or role in ("textbox", "button", "checkbox", "radio", "switch", "combobox", "slider")
        ) and enabled
        if is_interactive and bounds:
            center_x = (bounds["left"] + bounds["right"]) // 2
            center_y = (bounds["top"] + bounds["bottom"]) // 2
            interactive.append(
                {
                    "role": role,
                    "name": name,
                    "resource_id": resource_id,
                    "bounds": bounds,
                    "center": {"x": center_x, "y": center_y},
                    "clickable": clickable,
                    "class": class_name,
                }
            )

        # Recurse into children
        if depth < max_depth:
            children = []
            for child in node:
                child_node = cls._parse_node(child, depth + 1, max_depth, interactive, node_count)
                # Skip empty group nodes with no name and no interactive children
                if (
                    child_node.get("name")
                    or child_node.get("children")
                    or child_node.get("clickable")
                    or child_node.get("role") not in ("group", "generic")
                ):
                    children.append(child_node)
            if children:
                result["children"] = children

        return result

    @staticmethod
    def _parse_bounds(bounds_str: str) -> Dict[str, int]:
        """Parse UIAutomator bounds string '[x1,y1][x2,y2]' to dict."""
        if not bounds_str:
            return {}
        match = re.match(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", bounds_str)
        if match:
            return {
                "left": int(match.group(1)),
                "top": int(match.group(2)),
                "right": int(match.group(3)),
                "bottom": int(match.group(4)),
            }
        return {}


# =============================================================================
# Tool Functions (exposed to Jotty skill system)
# =============================================================================


@async_tool_wrapper()
async def device_connect_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Connect to an Android device or emulator.

    Args:
        params: Dictionary containing:
            - serial (str, optional): Device serial (e.g. 'emulator-5554')
            - url (str, optional): Remote device URL

    Returns:
        Dictionary with:
            - serial (str): Connected device serial
            - screen_width (int): Screen width in pixels
            - screen_height (int): Screen height in pixels
            - model (str): Device model name
    """
    status.set_callback(params.pop("_status_callback", None))
    status.emit("Connecting", "Connecting to Android device...")

    dev = AndroidDevice.get_instance()
    dev.connect(serial=params.get("serial"), url=params.get("url"))
    info = dev.get_device_info()

    return tool_response(
        serial=info.get("serial", ""),
        screen_width=info.get("screen_width", 0),
        screen_height=info.get("screen_height", 0),
        model=info.get("model", ""),
        brand=info.get("brand", ""),
    )


@async_tool_wrapper(required_params=["x", "y"])
async def tap_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tap at exact screen coordinates.

    Args:
        params: Dictionary containing:
            - x (int, required): X coordinate
            - y (int, required): Y coordinate

    Returns:
        Dictionary with:
            - x (int): X coordinate tapped
            - y (int): Y coordinate tapped
            - action (str): Action performed
    """
    status.set_callback(params.pop("_status_callback", None))
    dev = AndroidDevice.get_instance()
    result = dev.tap(int(params["x"]), int(params["y"]))
    return tool_response(**result)


@async_tool_wrapper(required_params=["x", "y"])
async def long_press_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Long press at screen coordinates.

    Args:
        params: Dictionary containing:
            - x (int, required): X coordinate
            - y (int, required): Y coordinate
            - duration (float, optional): Hold duration in seconds (default: 1.0)

    Returns:
        Dictionary with:
            - x (int): X coordinate
            - y (int): Y coordinate
            - duration (float): Hold duration
            - action (str): Action performed
    """
    status.set_callback(params.pop("_status_callback", None))
    dev = AndroidDevice.get_instance()
    result = dev.long_press(
        int(params["x"]),
        int(params["y"]),
        duration=float(params.get("duration", 1.0)),
    )
    return tool_response(**result)


@async_tool_wrapper(required_params=["x1", "y1", "x2", "y2"])
async def swipe_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Swipe from one point to another on screen.

    Args:
        params: Dictionary containing:
            - x1 (int, required): Start X coordinate
            - y1 (int, required): Start Y coordinate
            - x2 (int, required): End X coordinate
            - y2 (int, required): End Y coordinate
            - duration (float, optional): Swipe duration in seconds (default: 0.5)

    Returns:
        Dictionary with:
            - from (list): Start coordinates [x1, y1]
            - to (list): End coordinates [x2, y2]
            - duration (float): Swipe duration
            - action (str): Action performed
    """
    status.set_callback(params.pop("_status_callback", None))
    dev = AndroidDevice.get_instance()
    result = dev.swipe(
        int(params["x1"]),
        int(params["y1"]),
        int(params["x2"]),
        int(params["y2"]),
        duration=float(params.get("duration", 0.5)),
    )
    return tool_response(**result)


@async_tool_wrapper(required_params=["text"])
async def type_text_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Type text into the currently focused input field.

    Args:
        params: Dictionary containing:
            - text (str, required): Text to type
            - clear_first (bool, optional): Clear field before typing (default: false)

    Returns:
        Dictionary with:
            - text (str): Text typed
            - cleared (bool): Whether field was cleared first
            - action (str): Action performed
    """
    status.set_callback(params.pop("_status_callback", None))
    dev = AndroidDevice.get_instance()
    result = dev.type_text(
        params["text"],
        clear_first=params.get("clear_first", False),
    )
    return tool_response(**result)


@async_tool_wrapper(required_params=["key"])
async def press_key_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Press a device key (home, back, enter, recent, volume_up, etc.).

    Args:
        params: Dictionary containing:
            - key (str, required): Key name (home, back, enter, recent, volume_up, volume_down, power, menu)

    Returns:
        Dictionary with:
            - key (str): Key pressed
            - action (str): Action performed
    """
    status.set_callback(params.pop("_status_callback", None))
    dev = AndroidDevice.get_instance()
    result = dev.press_key(params["key"])
    return tool_response(**result)


@async_tool_wrapper()
async def screenshot_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Capture a screenshot of the current screen.

    Args:
        params: Dictionary containing:
            - scale (float, optional): Scale factor, 0.5 = half resolution (default: 0.5)

    Returns:
        Dictionary with:
            - image_base64 (str): Base64-encoded PNG screenshot
            - width (int): Image width
            - height (int): Image height
            - format (str): Image format
    """
    status.set_callback(params.pop("_status_callback", None))
    status.emit("Capturing", "Taking screenshot...")

    dev = AndroidDevice.get_instance()
    result = dev.screenshot(scale=float(params.get("scale", 0.5)))
    return tool_response(**result)


@async_tool_wrapper(required_params=["package"])
async def launch_app_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Launch an Android app by package name.

    Args:
        params: Dictionary containing:
            - package (str, required): Package name (e.g. 'com.google.android.youtube')
            - activity (str, optional): Specific activity to launch

    Returns:
        Dictionary with:
            - package (str): Package launched
            - activity (str): Activity launched
            - action (str): Action performed
    """
    status.set_callback(params.pop("_status_callback", None))
    status.emit("Launching", f"Starting {params['package']}...")

    dev = AndroidDevice.get_instance()
    result = dev.launch_app(
        params["package"],
        activity=params.get("activity"),
    )
    return tool_response(**result)


@async_tool_wrapper(required_params=["package"])
async def stop_app_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Force stop an Android app.

    Args:
        params: Dictionary containing:
            - package (str, required): Package name to stop

    Returns:
        Dictionary with:
            - package (str): Package stopped
            - action (str): Action performed
    """
    status.set_callback(params.pop("_status_callback", None))
    dev = AndroidDevice.get_instance()
    result = dev.stop_app(params["package"])
    return tool_response(**result)


@async_tool_wrapper()
async def get_ui_tree_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dump the UI accessibility tree for the current screen.

    Returns a structured tree with roles, names, bounds, and interactive states.
    Also provides a flat list of all interactive (clickable/focusable) elements
    with their center coordinates for easy tapping.

    Args:
        params: Dictionary containing:
            - max_depth (int, optional): Maximum tree depth (default: 10)

    Returns:
        Dictionary with:
            - tree (dict): Nested UI tree with role, name, bounds, children
            - node_count (int): Total nodes parsed
            - interactive_elements (list): Flat list of tappable elements with center coords
    """
    status.set_callback(params.pop("_status_callback", None))
    status.emit("Parsing", "Dumping UI hierarchy...")

    dev = AndroidDevice.get_instance()
    result = dev.dump_ui_tree(max_depth=int(params.get("max_depth", 10)))
    return tool_response(**result)


@async_tool_wrapper()
async def find_element_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Find a UI element by text, resource ID, class, or description.

    Returns element info including bounds and center coordinates for tapping.

    Args:
        params: Dictionary containing:
            - text (str, optional): Element text content
            - resource_id (str, optional): Android resource ID
            - class_name (str, optional): Widget class name
            - description (str, optional): Content description

    Returns:
        Dictionary with:
            - found (bool): Whether element was found
            - text (str): Element text
            - class (str): Widget class name
            - bounds (dict): Element bounds {left, top, right, bottom}
            - center (dict): Center coordinates {x, y} for tapping
            - clickable (bool): Whether element is clickable
            - enabled (bool): Whether element is enabled
    """
    status.set_callback(params.pop("_status_callback", None))
    dev = AndroidDevice.get_instance()
    result = dev.find_element(
        text=params.get("text"),
        resource_id=params.get("resource_id"),
        class_name=params.get("class_name"),
        description=params.get("description"),
    )
    return tool_response(**result)


@async_tool_wrapper()
async def tap_element_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Find a UI element and tap it. Combines find + tap in one call.

    Args:
        params: Dictionary containing:
            - text (str, optional): Element text to find and tap
            - resource_id (str, optional): Resource ID to find and tap
            - description (str, optional): Content description to find and tap

    Returns:
        Dictionary with:
            - tapped (bool): Whether element was found and tapped
            - text (str): Element text
            - center (dict): Coordinates tapped {x, y}
    """
    status.set_callback(params.pop("_status_callback", None))
    dev = AndroidDevice.get_instance()
    result = dev.tap_element(
        text=params.get("text"),
        resource_id=params.get("resource_id"),
        description=params.get("description"),
    )
    return tool_response(**result)


@async_tool_wrapper()
async def wait_for_element_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Wait for a UI element to appear on screen.

    Args:
        params: Dictionary containing:
            - text (str, optional): Element text to wait for
            - resource_id (str, optional): Resource ID to wait for
            - timeout (float, optional): Max wait seconds (default: 10)

    Returns:
        Dictionary with:
            - found (bool): Whether element appeared within timeout
            - criteria (dict): Search criteria used
            - timeout (float): Timeout value used
    """
    status.set_callback(params.pop("_status_callback", None))
    status.emit("Waiting", "Waiting for element...")

    dev = AndroidDevice.get_instance()
    result = dev.wait_for_element(
        text=params.get("text"),
        resource_id=params.get("resource_id"),
        timeout=float(params.get("timeout", 10)),
    )
    return tool_response(**result)


@async_tool_wrapper()
async def scroll_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Scroll the screen in a direction.

    Args:
        params: Dictionary containing:
            - direction (str, optional): 'up', 'down', 'left', 'right' (default: 'down')
            - steps (int, optional): Scroll steps/speed (default: 10)

    Returns:
        Dictionary with:
            - direction (str): Scroll direction
            - steps (int): Scroll steps
            - action (str): Action performed
    """
    status.set_callback(params.pop("_status_callback", None))
    dev = AndroidDevice.get_instance()
    result = dev.scroll(
        direction=params.get("direction", "down"),
        steps=int(params.get("steps", 10)),
    )
    return tool_response(**result)


@async_tool_wrapper()
async def get_current_app_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get info about the current foreground app.

    Args:
        params: Dictionary (no required params)

    Returns:
        Dictionary with:
            - package (str): Current app package name
            - activity (str): Current activity name
    """
    status.set_callback(params.pop("_status_callback", None))
    dev = AndroidDevice.get_instance()
    result = dev.current_app()
    return tool_response(**result)


@async_tool_wrapper()
async def device_info_tool(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get device hardware and software info.

    Args:
        params: Dictionary (no required params)

    Returns:
        Dictionary with:
            - serial (str): Device serial
            - model (str): Device model
            - brand (str): Device brand
            - sdk (int): Android SDK version
            - screen_width (int): Screen width
            - screen_height (int): Screen height
    """
    status.set_callback(params.pop("_status_callback", None))
    dev = AndroidDevice.get_instance()
    result = dev.get_device_info()
    return tool_response(**result)


__all__ = [
    "device_connect_tool",
    "tap_tool",
    "long_press_tool",
    "swipe_tool",
    "type_text_tool",
    "press_key_tool",
    "screenshot_tool",
    "launch_app_tool",
    "stop_app_tool",
    "get_ui_tree_tool",
    "find_element_tool",
    "tap_element_tool",
    "wait_for_element_tool",
    "scroll_tool",
    "get_current_app_tool",
    "device_info_tool",
]
