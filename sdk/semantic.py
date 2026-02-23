"""
SDK bridge for semantic layer imports.

Re-exports semantic layer classes from skills/semantic-layer/ so that apps/
can import from Jotty.sdk.semantic instead of reaching into core.

Note: The skills/semantic-layer/ folder uses a hyphen, so we use importlib.
"""

import importlib
import os
import sys

_SEMANTIC_LAYER_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "skills", "semantic-layer"
)


def _import_data_loader():
    """Import data_loader from the semantic-layer skill (hyphenated package)."""
    spec = importlib.util.spec_from_file_location(
        "semantic_layer_data_loader",
        os.path.join(_SEMANTIC_LAYER_DIR, "query", "data_loader.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_dl = _import_data_loader()
DataLoaderFactory = _dl.DataLoaderFactory
OutputFormat = _dl.OutputFormat

__all__ = [
    "DataLoaderFactory",
    "OutputFormat",
]
