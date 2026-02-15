"""
Section Schema Validator - Auto-syncs with JustJot.ai

Fetches section schemas from JustJot.ai API and provides validation/transformation.
Ensures Jotty always sends data in the exact format section renderers expect.

Usage:
    from core.ui.schema_validator import schema_registry

    content = schema_registry.validate_and_transform('kanban-board', {
        'columns': [{
            'items': [{'priority': 1, 'assignee': 'Alice'}]
        }]
    })
    # Returns: priority='low', assignee={'name': 'Alice'}
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)


class SectionSchemaRegistry:
    """
    Auto-syncing schema registry.

    Fetches schemas from JustJot.ai on startup, caches locally.
    Provides validation and transformation for all section types.
    """

    def __init__(self, api_url: str = None, lazy_load: bool = True) -> None:
        if api_url is None:
            import os

            try:
                from ..foundation.config_defaults import DEFAULTS as _DEFAULTS

                _base = os.getenv("JUSTJOT_API_URL", _DEFAULTS.JUSTJOT_API_URL)
            except ImportError:
                _base = os.getenv("JUSTJOT_API_URL", "http://localhost:3000")
            api_url = f"{_base}/api/sections/schemas"
        self.api_url = api_url
        self.schemas: Dict[str, Any] = {}
        self.cache_file = Path(__file__).parent / "section_schemas_cache.json"
        self._loaded = False

        # Lazy load by default (only load when first used)
        if not lazy_load:
            self._load_schemas()

    def _load_schemas(self) -> Any:
        """Fetch schemas from JustJot.ai API or load from cache."""
        try:
            # Try to fetch from API
            response = requests.get(self.api_url, timeout=10)
            if response.status_code == 200:
                catalog = response.json()

                # Fetch each schema (parallel would be better, but keeping it simple)
                for section_type in catalog.get("sections", []):
                    try:
                        schema_response = requests.get(
                            f"{self.api_url}?type={section_type}", timeout=5
                        )
                        if schema_response.status_code == 200:
                            self.schemas[section_type] = schema_response.json()
                    except Exception as e:
                        logger.warning(f"Failed to load schema for {section_type}: {e}")

                logger.info(f" Loaded {len(self.schemas)} section schemas from API")

                # Cache for offline use
                self._save_cache()
            else:
                logger.warning(f"API returned {response.status_code}, loading from cache")
                self._load_cache()

        except Exception as e:
            logger.warning(f"Failed to fetch schemas from API: {e}")
            self._load_cache()

    def _save_cache(self) -> Any:
        """Save schemas to local cache."""
        try:
            with open(self.cache_file, "w") as f:
                json.dump(self.schemas, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save schema cache: {e}")

    def _load_cache(self) -> Any:
        """Load schemas from local cache."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, "r") as f:
                    self.schemas = json.load(f)
                logger.info(f" Loaded {len(self.schemas)} schemas from cache")
            else:
                logger.warning("No schema cache found, using empty registry")
        except Exception as e:
            logger.error(f"Failed to load schema cache: {e}")

    def validate_and_transform(self, section_type: str, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate content against schema and apply transforms.

        Args:
            section_type: Section type (e.g., 'kanban-board')
            content: Content object to validate/transform

        Returns:
            Transformed content matching section renderer schema

        Example:
            >>> content = {
            ...     'columns': [{
            ...         'items': [
            ...             {'id': '1', 'title': 'Task', 'priority': 1, 'assignee': 'Alice'}
            ...         ]
            ...     }]
            ... }
            >>> result = registry.validate_and_transform('kanban-board', content)
            >>> result['columns'][0]['items'][0]['priority']
            'low'
            >>> result['columns'][0]['items'][0]['assignee']
            {'name': 'Alice'}
        """
        # Ensure schemas loaded on first use
        if not self._loaded:
            self._load_schemas()
            self._loaded = True

        schema = self.schemas.get(section_type)
        if not schema:
            logger.warning(f"No schema found for {section_type}, passing through")
            return content

        # Apply transforms
        transforms = schema.get("transforms", {})
        if transforms:
            content = self._apply_transforms(content, transforms)

        return content

    def _apply_transforms(
        self, content: Dict[str, Any], transforms: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply all transforms to content."""
        for field_path, transform_def in transforms.items():
            self._apply_transform(content, field_path, transform_def)
        return content

    def _apply_transform(self, content: Dict, field_path: str, transform_def: Dict) -> Any:
        """
        Apply a single transform.

        Supports nested paths with wildcards:
        - 'columns.*.items.*.priority' matches all items in all columns
        """
        parts = field_path.split(".")
        self._apply_transform_recursive(content, parts, transform_def)

    def _apply_transform_recursive(self, obj: Any, path_parts: list, transform_def: Dict) -> None:
        """Recursively apply transform to nested structures."""
        if not path_parts:
            return

        key = path_parts[0]
        remaining = path_parts[1:]

        if key == "*":
            # Wildcard: apply to all items in array
            if isinstance(obj, list):
                for item in obj:
                    self._apply_transform_recursive(item, remaining, transform_def)
            elif isinstance(obj, dict):
                for item in obj.values():
                    self._apply_transform_recursive(item, remaining, transform_def)
        elif remaining:
            # More path to traverse
            if isinstance(obj, dict) and key in obj:
                self._apply_transform_recursive(obj[key], remaining, transform_def)
        else:
            # Leaf node: apply transform
            if isinstance(obj, dict) and key in obj:
                transformed = self._transform_value(obj[key], transform_def)
                if transformed is not None:  # Only update if transform succeeded
                    obj[key] = transformed

    def _transform_value(self, value: Any, transform_def: Dict) -> Optional[Any]:
        """
        Transform a single value according to definition.

        Supported transforms:
        - number → enum: {from: 'number', to: 'enum', mapping: {...}}
        - string → object: {from: 'string', to: 'object', fields: {...}}
        """
        from_type = transform_def.get("from")
        to_type = transform_def.get("to")

        # Number to enum (e.g., 1 → 'low')
        if from_type == "number" and to_type == "enum":
            if isinstance(value, (int, float)):
                mapping = transform_def.get("mapping", {})
                return mapping.get(str(int(value)), value)

        # String to object (e.g., 'Alice' → {'name': 'Alice'})
        elif from_type == "string" and to_type == "object":
            if isinstance(value, str):
                fields = transform_def.get("fields", {})
                result = {}
                for field_name, field_value in fields.items():
                    if field_value == "$value":
                        result[field_name] = value
                    else:
                        result[field_name] = field_value
                return result
            elif isinstance(value, dict):
                # Already an object, pass through
                return value

        return None  # Transform not applicable

    def get_schema(self, section_type: str) -> Optional[Dict[str, Any]]:
        """Get full schema for a section type."""
        # Ensure schemas loaded on first use
        if not self._loaded:
            self._load_schemas()
            self._loaded = True

        return self.schemas.get(section_type)

    def list_sections(self) -> list:
        """List all available section types."""
        # Ensure schemas loaded on first use
        if not self._loaded:
            self._load_schemas()
            self._loaded = True

        return list(self.schemas.keys())


# Global instance (singleton pattern)
_registry_instance = None


def get_schema_registry() -> SectionSchemaRegistry:
    """Get or create global schema registry instance."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = SectionSchemaRegistry()
    return _registry_instance


# Convenience alias
schema_registry = get_schema_registry()
