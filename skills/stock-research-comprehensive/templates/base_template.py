"""
Base Template System
====================

Abstract base class for all PDF templates with common functionality.
"""

import base64
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TemplateColors:
    """Color scheme for a template."""

    primary: str = "#2c5282"
    primary_dark: str = "#1a365d"
    primary_light: str = "#4299e1"
    secondary: str = "#718096"
    accent: str = "#38a169"
    success: str = "#38a169"
    warning: str = "#d69e2e"
    danger: str = "#e53e3e"
    text: str = "#2d3748"
    text_light: str = "#718096"
    background: str = "#ffffff"
    background_alt: str = "#f7fafc"
    border: str = "#e2e8f0"


@dataclass
class TemplateTypography:
    """Typography settings for a template."""

    heading_font: str = "'Inter', sans-serif"
    body_font: str = "'Inter', sans-serif"
    mono_font: str = "'Consolas', monospace"
    base_size: str = "10pt"
    h1_size: str = "24pt"
    h2_size: str = "16pt"
    h3_size: str = "12pt"
    line_height: str = "1.6"


@dataclass
class TemplateLayout:
    """Layout settings for a template."""

    page_size: str = "A4"
    margin_top: str = "2cm"
    margin_bottom: str = "2.5cm"
    margin_left: str = "1.5cm"
    margin_right: str = "1.5cm"
    header_height: str = "1cm"
    footer_height: str = "1cm"


class BaseTemplate(ABC):
    """Abstract base class for all templates."""

    def __init__(self) -> None:
        self.name: str = "Base Template"
        self.description: str = "Base template class"
        self.category: str = "general"
        self.colors = TemplateColors()
        self.typography = TemplateTypography()
        self.layout = TemplateLayout()

    @abstractmethod
    def get_css(self) -> str:
        """Return the CSS for this template."""
        pass

    @abstractmethod
    def get_html_wrapper(self, content: str, title: str) -> str:
        """Wrap content in HTML with this template's styling."""
        pass

    def embed_image(self, image_path: str) -> str:
        """Convert image to base64 for embedding in HTML."""
        try:
            path = Path(image_path)
            if not path.exists():
                logger.warning(f"Image not found: {image_path}")
                return ""

            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode("utf-8")

            ext = path.suffix.lower()
            mime_types = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".gif": "image/gif",
                ".svg": "image/svg+xml",
            }
            mime = mime_types.get(ext, "image/png")

            return f"data:{mime};base64,{data}"
        except Exception as e:
            logger.error(f"Failed to embed image {image_path}: {e}")
            return ""

    def create_image_tag(self, image_path: str, alt: str = "", width: str = "100%") -> str:
        """Create an img tag with embedded base64 image."""
        data_uri = self.embed_image(image_path)
        if not data_uri:
            return f'<p style="color:#999;">[Chart: {alt}]</p>'
        return f'<img src="{data_uri}" alt="{alt}" style="width:{width};max-width:100%;height:auto;display:block;margin:12pt auto;" />'

    def get_base_css(self) -> str:
        """Return common CSS that all templates share."""
        return f"""
/* Base Styles - Common to All Templates */
@page {{
    size: {self.layout.page_size};
    margin: {self.layout.margin_top} {self.layout.margin_right} {self.layout.margin_bottom} {self.layout.margin_left};
}}

* {{
    box-sizing: border-box;
}}

body {{
    font-family: {self.typography.body_font};
    font-size: {self.typography.base_size};
    line-height: {self.typography.line_height};
    color: {self.colors.text};
    background: {self.colors.background};
    margin: 0;
    padding: 0;
}}

/* Page Break Utilities */
.page-break {{
    page-break-before: always;
}}

.avoid-break {{
    page-break-inside: avoid;
}}

.keep-together {{
    page-break-inside: avoid;
}}

/* Tables */
table {{
    width: 100%;
    border-collapse: collapse;
    margin: 12pt 0;
    font-size: 9pt;
    page-break-inside: avoid;
}}

th, td {{
    padding: 6pt 8pt;
    border: 1px solid {self.colors.border};
    text-align: left;
}}

th {{
    background: {self.colors.primary};
    color: white;
    font-weight: 600;
}}

tr:nth-child(even) {{
    background: {self.colors.background_alt};
}}

td:not(:first-child), th:not(:first-child) {{
    text-align: right;
}}

/* Lists */
ul, ol {{
    margin: 8pt 0;
    padding-left: 20pt;
}}

li {{
    margin: 4pt 0;
}}

/* Horizontal Rules */
hr {{
    border: none;
    border-top: 1px solid {self.colors.border};
    margin: 16pt 0;
}}

/* Images */
img {{
    max-width: 100%;
    height: auto;
}}

/* Code Blocks */
pre, code {{
    font-family: {self.typography.mono_font};
    font-size: 8pt;
    background: {self.colors.background_alt};
    padding: 8pt;
    border-radius: 4pt;
    overflow-x: auto;
}}

/* Print */
@media print {{
    body {{
        -webkit-print-color-adjust: exact;
        print-color-adjust: exact;
    }}
}}
"""


class TemplateRegistry:
    """Registry for managing available templates."""

    _templates: Dict[str, BaseTemplate] = {}

    @classmethod
    def register(cls, template: BaseTemplate) -> None:
        """Register a template."""
        cls._templates[template.name.lower().replace(" ", "_")] = template

    @classmethod
    def get(cls, name: str) -> Optional[BaseTemplate]:
        """Get a template by name."""
        return cls._templates.get(name.lower().replace(" ", "_"))

    @classmethod
    def list_templates(cls, category: str = None) -> List[Dict[str, str]]:
        """List available templates."""
        templates = []
        for key, template in cls._templates.items():
            if category is None or template.category == category:
                templates.append(
                    {
                        "key": key,
                        "name": template.name,
                        "description": template.description,
                        "category": template.category,
                    }
                )
        return templates

    @classmethod
    def get_categories(cls) -> List[str]:
        """Get list of template categories."""
        return list(set(t.category for t in cls._templates.values()))
