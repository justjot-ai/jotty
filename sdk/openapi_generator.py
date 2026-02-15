"""
OpenAPI Specification Generator for Jotty (SDK entrypoint)
==========================================================

Delegates to the canonical spec in core/api/openapi.py.
This file exists so `python sdk/openapi_generator.py` still works
and `generate_sdks.py` documentation remains valid.

The single source of truth for the OpenAPI spec is:
    core/api/openapi.py
"""

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Add parent to path so we can import core
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.api.openapi import generate_openapi_spec, save_openapi_spec

if __name__ == "__main__":
    output_file = Path("sdk/openapi.json")
    if len(sys.argv) > 1:
        output_file = Path(sys.argv[1])

    spec = generate_openapi_spec()
    save_openapi_spec(spec, output_file)

    logger.info("OpenAPI specification generated:")
    logger.info("   - File: %s", output_file)
    logger.info("   - Version: %s", spec["info"]["version"])
    logger.info("   - Endpoints: %d", len(spec["paths"]))
    logger.info("   - Schemas: %d", len(spec["components"]["schemas"]))
    logger.info("Next steps:")
    logger.info("   1. Review the spec: cat %s", output_file)
    logger.info("   2. Generate SDKs: python sdk/generate_sdks.py")
