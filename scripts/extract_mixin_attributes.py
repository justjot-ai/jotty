#!/usr/bin/env python3
"""
Extract all self.* attribute accesses from mixin files.

This helps us build a comprehensive list of all attributes that mixins expect.
"""

import re
from pathlib import Path
from collections import defaultdict
from typing import Set, Dict

def extract_attributes_from_file(file_path: Path) -> Set[str]:
    """Extract all self.* attribute accesses from a file."""
    content = file_path.read_text()

    # Match self.attribute_name (but not method calls with parentheses immediately after)
    # Also match self._attribute_name
    pattern = r'self\.([a-zA-Z_][a-zA-Z0-9_]*)'

    attributes = set()
    for match in re.finditer(pattern, content):
        attr = match.group(1)
        # Skip common methods
        if attr not in ['__init__', '__class__', '__dict__']:
            attributes.add(attr)

    return attributes

def main():
    # Find all mixin files
    mixin_files = []
    for pattern in ['core/intelligence/**/*_mixin.py',
                    'core/intelligence/**/protocols/*.py']:
        mixin_files.extend(Path('.').glob(pattern))

    # Extract attributes from each
    all_attributes: Dict[str, Set[str]] = {}

    for file_path in sorted(mixin_files):
        attrs = extract_attributes_from_file(file_path)
        if attrs:
            all_attributes[str(file_path)] = attrs

    # Print results
    print("="*80)
    print("MIXIN ATTRIBUTE ANALYSIS")
    print("="*80)

    # Collect all unique attributes
    all_attrs_set = set()
    for attrs in all_attributes.values():
        all_attrs_set.update(attrs)

    print(f"\n📊 Found {len(all_attrs_set)} unique attributes across {len(all_attributes)} files\n")

    # Group by type (public vs private)
    public_attrs = sorted([a for a in all_attrs_set if not a.startswith('_')])
    private_attrs = sorted([a for a in all_attrs_set if a.startswith('_')])

    print(f"📋 PUBLIC ATTRIBUTES ({len(public_attrs)}):")
    for attr in public_attrs[:50]:  # Show first 50
        print(f"    {attr}")
    if len(public_attrs) > 50:
        print(f"    ... and {len(public_attrs) - 50} more")

    print(f"\n📋 PRIVATE ATTRIBUTES ({len(private_attrs)}):")
    for attr in private_attrs[:50]:  # Show first 50
        print(f"    {attr}")
    if len(private_attrs) > 50:
        print(f"    ... and {len(private_attrs) - 50} more")

    # Print by category
    print("\n" + "="*80)
    print("ATTRIBUTES BY CATEGORY")
    print("="*80)

    # Categorize
    categories = {
        'config': [],
        'learning': [],
        'memory': [],
        'orchestration': [],
        'report': [],
        'other': []
    }

    for attr in all_attrs_set:
        if 'config' in attr.lower():
            categories['config'].append(attr)
        elif 'learn' in attr.lower() or 'reward' in attr.lower() or 'episode' in attr.lower():
            categories['learning'].append(attr)
        elif 'memory' in attr.lower() or 'graph' in attr.lower() or 'node' in attr.lower():
            categories['memory'].append(attr)
        elif 'agent' in attr.lower() or 'swarm' in attr.lower() or 'handoff' in attr.lower():
            categories['orchestration'].append(attr)
        elif 'report' in attr.lower() or 'figure' in attr.lower() or 'section' in attr.lower() or 'chart' in attr.lower():
            categories['report'].append(attr)
        else:
            categories['other'].append(attr)

    for cat_name, attrs in sorted(categories.items()):
        if attrs:
            print(f"\n{cat_name.upper()} ({len(attrs)}):")
            for attr in sorted(attrs)[:20]:
                print(f"    {attr}")
            if len(attrs) > 20:
                print(f"    ... and {len(attrs) - 20} more")

if __name__ == '__main__':
    main()
