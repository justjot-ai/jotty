"""
OpenAPI Specification Generator for Jotty (SDK entrypoint)
==========================================================

Delegates to the canonical spec in core/api/openapi.py.
This file exists so `python sdk/openapi_generator.py` still works
and `generate_sdks.py` documentation remains valid.

The single source of truth for the OpenAPI spec is:
    core/api/openapi.py
"""

import sys
from pathlib import Path

# Add parent to path so we can import core
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.api.openapi import generate_openapi_spec, save_openapi_spec


if __name__ == "__main__":
    output_file = Path("sdk/openapi.json")
    if len(sys.argv) > 1:
        output_file = Path(sys.argv[1])

    spec = generate_openapi_spec()
    save_openapi_spec(spec, output_file)

    print(f"\n📋 OpenAPI specification generated:")
    print(f"   - File: {output_file}")
    print(f"   - Version: {spec['info']['version']}")
    print(f"   - Endpoints: {len(spec['paths'])}")
    print(f"   - Schemas: {len(spec['components']['schemas'])}")
    print(f"\n💡 Next steps:")
    print(f"   1. Review the spec: cat {output_file}")
    print(f"   2. Generate SDKs: python sdk/generate_sdks.py")
