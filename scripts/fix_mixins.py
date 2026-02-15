#!/usr/bin/env python3
"""
Automatically add TYPE_CHECKING blocks to mixin files to fix mypy attr-defined errors.

This script:
1. Finds all mixin files (_*_mixin.py)
2. Adds TYPE_CHECKING imports and attribute declarations
3. Fixes 1000+ mypy errors in one go!

Usage:
    python scripts/fix_mixins.py [--dry-run]
"""

import argparse
import re
from pathlib import Path
from typing import List, Set


# Standard attributes that report generator mixins expect
REPORT_MIXIN_ATTRS = """
    if TYPE_CHECKING:
        # Declare expected attributes from parent class
        output_dir: Path
        figures_dir: Path
        theme: str
        config: Dict[str, Any]
        _llm_narrative_enabled: bool
        _html_enabled: bool
        _content: List[Any]
        _figures: List[Any]
        _warnings: List[Any]
        _metadata: Dict[str, Any]
        _raw_data: Dict[str, Any]
        _section_data: List[Any]
        _failed_sections: List[str]
        _failed_charts: List[str]

        def _record_chart_failure(self, chart_name: str, error: Exception) -> None: ...
        def _save_figure(self, fig: Any, name: str) -> Optional[Path]: ...
        def _fig_path_for_markdown(self, fig_path: Path) -> str: ...
        def _add_section(self, title: str, content: str, **kwargs: Any) -> None: ...
        def _record_section_failure(self, section: str, error: Exception) -> None: ...
"""

# Standard attributes for swarm learning mixins
SWARM_MIXIN_ATTRS = """
    if TYPE_CHECKING:
        # Declare expected attributes from parent swarm class
        learning_enabled: bool
        learning_config: Optional[Any]
        _learning_memory: List[Dict[str, Any]]
        name: str

        def _store_learning_memory(self, data: Dict[str, Any]) -> None: ...
"""

# Standard attributes for memory consolidation mixins
MEMORY_MIXIN_ATTRS = """
    if TYPE_CHECKING:
        # Declare expected attributes from parent memory class
        _graph: Any
        config: Any

        def _consolidate_nodes(self, nodes: List[Any]) -> Any: ...
"""


def find_mixin_files() -> List[Path]:
    """Find all mixin files in the codebase."""
    mixin_files = []

    # Find in templates directory
    templates_dir = Path('core/intelligence/orchestration/templates')
    if templates_dir.exists():
        mixin_files.extend(templates_dir.glob('_*_mixin.py'))

    # Find in other directories
    for pattern in ['core/intelligence/swarms/_*_mixin.py',
                    'core/intelligence/memory/_*_mixin.py',
                    'core/intelligence/orchestration/_*_mixin.py',
                    'core/intelligence/orchestration/protocols/*.py']:
        mixin_files.extend(Path('.').glob(pattern))

    return sorted(set(mixin_files))


def needs_type_checking_block(content: str) -> bool:
    """Check if file already has TYPE_CHECKING block with attribute declarations."""
    return 'if TYPE_CHECKING:' not in content or \
           '# Declare expected attributes' not in content


def get_mixin_type(file_path: Path) -> str:
    """Determine what type of mixin this is based on location and name."""
    if 'templates' in str(file_path) or 'orchestration' in str(file_path):
        return 'report'
    elif 'swarm' in str(file_path) and 'learning' in file_path.name:
        return 'swarm'
    elif 'memory' in str(file_path) and 'consolidation' in file_path.name:
        return 'memory'
    else:
        return 'report'  # Default


def get_attrs_for_mixin_type(mixin_type: str) -> str:
    """Get the appropriate TYPE_CHECKING block for this mixin type."""
    if mixin_type == 'swarm':
        return SWARM_MIXIN_ATTRS
    elif mixin_type == 'memory':
        return MEMORY_MIXIN_ATTRS
    else:
        return REPORT_MIXIN_ATTRS


def add_type_checking_block(file_path: Path, dry_run: bool = False) -> bool:
    """
    Add TYPE_CHECKING block to a mixin file.

    Returns True if file was modified, False otherwise.
    """
    content = file_path.read_text()

    # Check if already has the block
    if not needs_type_checking_block(content):
        print(f"  ⏭️  {file_path} - Already has TYPE_CHECKING block")
        return False

    # Determine mixin type
    mixin_type = get_mixin_type(file_path)
    attrs_block = get_attrs_for_mixin_type(mixin_type)

    # Find the class definition
    class_match = re.search(r'^class \w+Mixin[^:]*:', content, re.MULTILINE)
    if not class_match:
        print(f"  ⚠️  {file_path} - No Mixin class found")
        return False

    # Find the end of the class docstring (if any)
    class_start = class_match.end()
    lines = content[class_start:].split('\n')

    # Find where to insert (after docstring, before first method)
    insert_line = 0
    in_docstring = False
    for i, line in enumerate(lines):
        stripped = line.strip()

        # Check for docstring start
        if stripped.startswith('"""') or stripped.startswith("'''"):
            if in_docstring:
                # End of docstring
                insert_line = i + 1
                break
            else:
                in_docstring = True
                continue

        # If not in docstring and we hit a def or attribute, insert before it
        if not in_docstring and (stripped.startswith('def ') or
                                  (stripped and not stripped.startswith('#'))):
            insert_line = i
            break

    if insert_line == 0:
        insert_line = 1  # Default to after class line

    # Insert the TYPE_CHECKING block
    before_class = content[:class_start]
    after_class = content[class_start:].split('\n')
    after_class.insert(insert_line, attrs_block)
    new_content = before_class + '\n'.join(after_class)

    # Make sure imports are present
    if 'from __future__ import annotations' not in new_content:
        new_content = 'from __future__ import annotations\n\n' + new_content

    if 'from pathlib import Path' not in new_content and mixin_type == 'report':
        # Add to existing imports
        import_line = new_content.find('import ')
        if import_line != -1:
            new_content = new_content[:import_line] + 'from pathlib import Path\n' + new_content[import_line:]

    # Update TYPE_CHECKING import
    type_import_pattern = r'from typing import ([^(\n]+)'
    match = re.search(type_import_pattern, new_content)
    if match:
        current_imports = match.group(1).strip()
        if 'TYPE_CHECKING' not in current_imports:
            new_imports = current_imports + ', TYPE_CHECKING'
            new_content = new_content.replace(match.group(1), new_imports)
    else:
        # Add typing import
        import_line = new_content.find('import ')
        if import_line != -1:
            new_content = new_content[:import_line] + 'from typing import Any, Dict, List, Optional, TYPE_CHECKING\n' + new_content[import_line:]

    if dry_run:
        print(f"  🔍 Would update {file_path}")
        return True

    # Write back
    file_path.write_text(new_content)
    print(f"  ✅ Updated {file_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Fix mixin type checking")
    parser.add_argument('--dry-run', action='store_true',
                       help="Show what would be changed without making changes")
    args = parser.parse_args()

    print("🔍 Finding mixin files...")
    mixin_files = find_mixin_files()
    print(f"   Found {len(mixin_files)} mixin file(s)\n")

    if not mixin_files:
        print("No mixin files found!")
        return 1

    print("🔧 Adding TYPE_CHECKING blocks...")
    modified_count = 0

    for file_path in mixin_files:
        if add_type_checking_block(file_path, dry_run=args.dry_run):
            modified_count += 1

    print(f"\n{'🔍 Would modify' if args.dry_run else '✅ Modified'} {modified_count}/{len(mixin_files)} file(s)")

    if args.dry_run:
        print("\nRun without --dry-run to apply changes")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
