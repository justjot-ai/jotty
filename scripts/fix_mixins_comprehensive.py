#!/usr/bin/env python3
"""
Add comprehensive TYPE_CHECKING blocks to all mixin files.

This script analyzes each mixin file and adds ALL attributes it uses,
properly categorized by type (Report, Orchestration, Learning, Memory, Coding).
"""

import re
from pathlib import Path
from typing import Set, Dict, List

# Comprehensive attribute lists by category
REPORT_ATTRS = """
    # Report Generator attributes
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
    figures: List[Any]
    _report_config: Optional[Dict[str, Any]]
    _telegram_config: Optional[Dict[str, Any]]
    _context: Optional[Any]

    def _record_chart_failure(self, chart_name: str, error: Exception) -> None: ...
    def _save_figure(self, fig: Any, name: str) -> Optional[Path]: ...
    def _fig_path_for_markdown(self, fig_path: Path) -> str: ...
    def _add_section(self, title: str, content: str, **kwargs: Any) -> None: ...
    def _record_section_failure(self, section: str, error: Exception) -> None: ...
    def _chart_context(self, chart_name: str, **kwargs: Any) -> Any: ...
    def _build_report_story(self, **kwargs: Any) -> str: ...
    def generate_report(self, **kwargs: Any) -> Any: ...
    def init_telegram(self, **kwargs: Any) -> None: ...
"""

ORCHESTRATION_ATTRS = """
    # Orchestration attributes
    agents: List[Any]
    agent_profiles: Dict[str, Any]
    agent_name: Optional[str]
    config: Dict[str, Any]
    name: str
    history: List[Any]
    handoff_history: List[Any]
    pending_handoffs: List[Any]
    active_auctions: Dict[str, Any]
    agent_coalitions: Dict[str, Any]
    coalitions: Dict[str, Any]
    circuit_breakers: Dict[str, Any]
    gossip_inbox: List[Any]
    gossip_seen: Set[str]
    consensus_history: List[Any]
    morph_score_history: List[Any]
    morph_scorer: Optional[Any]

    def register_agent(self, agent: Any) -> None: ...
    def find_idle_agents(self) -> List[Any]: ...
    def find_overloaded_agents(self) -> List[Any]: ...
    def get_agent_load(self, agent_id: str) -> float: ...
    def get_available_agents(self) -> List[Any]: ...
    def initiate_handoff(self, from_agent: str, to_agent: str, **kwargs: Any) -> None: ...
    def auto_auction(self, task: Any) -> Any: ...
    def form_coalition(self, agents: List[str]) -> str: ...
    def check_circuit(self, operation: str) -> bool: ...
    def gossip_broadcast(self, message: Any) -> None: ...
    def get_swarm_health(self) -> Dict[str, Any]: ...
"""

LEARNING_ATTRS = """
    # Learning attributes
    learning_enabled: bool
    learning: Optional[Any]
    mas_learning: Optional[Any]
    learning_config: Optional[Any]
    _learning_state: Dict[str, Any]
    _learning_memory: List[Dict[str, Any]]
    name: str
    config: Dict[str, Any]

    def _store_learning_memory(self, data: Dict[str, Any]) -> None: ...
"""

MEMORY_ATTRS = """
    # Memory attributes
    _graph: Any
    collective_memory: Optional[Any]
    memories: List[Any]
    config: Any
    consolidation_count: int
    causal_extractor: Optional[Any]
    pattern_extractor: Optional[Any]
    causal_links: List[Any]

    def _consolidate_nodes(self, nodes: List[Any]) -> Any: ...
    def _cluster_episodic_memories(self, memories: List[Any]) -> List[Any]: ...
"""

def determine_mixin_category(file_path: Path, content: str) -> List[str]:
    """Determine what categories of attributes this mixin needs."""
    categories = []

    # Check file path
    path_str = str(file_path)

    if 'templates' in path_str or '_visualization_' in path_str or '_analysis_' in path_str or \
       '_rendering_' in path_str or '_report_' in path_str or '_interpretability_' in path_str or \
       '_drift_' in path_str or '_fairness_' in path_str or '_deployment_' in path_str or \
       '_mlflow_' in path_str or '_telegram_' in path_str or '_error_analysis_' in path_str or \
       '_world_class_' in path_str:
        categories.append('report')

    if 'protocols' in path_str or 'coordination' in path_str or 'lifecycle' in path_str or \
       'resilience' in path_str or 'routing' in path_str or '_morph_' in path_str or \
       '_session_' in path_str or '_consensus_' in path_str or '_ensemble_' in path_str:
        categories.append('orchestration')

    if '_learning_mixin' in path_str:
        categories.append('learning')

    if 'memory' in path_str and ('_consolidation_' in path_str or '_retrieval_' in path_str):
        categories.append('memory')

    if 'coding_swarm' in path_str or '_edit_' in path_str or '_review_' in path_str:
        categories.append('coding')

    # If no category detected, check content
    if not categories:
        if 'self.figures' in content or 'self.theme' in content or 'self.output_dir' in content:
            categories.append('report')
        if 'self.agents' in content or 'self.handoff' in content:
            categories.append('orchestration')

    # Default to report if nothing else matched
    if not categories:
        categories.append('report')

    return categories

def build_type_checking_block(categories: List[str]) -> str:
    """Build TYPE_CHECKING block with appropriate attributes."""
    blocks = []

    if 'report' in categories:
        blocks.append(REPORT_ATTRS)
    if 'orchestration' in categories:
        blocks.append(ORCHESTRATION_ATTRS)
    if 'learning' in categories:
        blocks.append(LEARNING_ATTRS)
    if 'memory' in categories:
        blocks.append(MEMORY_ATTRS)

    combined = '\n'.join(blocks)

    return f"""
    if TYPE_CHECKING:
        # Comprehensive attribute declarations for type checking
        # These are provided by parent class or other mixins in composition
{combined}
"""

def fix_mixin_file(file_path: Path, dry_run: bool = False) -> bool:
    """Add comprehensive TYPE_CHECKING block to mixin file."""
    content = file_path.read_text()

    # Skip if already has comprehensive block
    if '# Comprehensive attribute declarations' in content:
        print(f"  ⏭️  {file_path.name} - Already has comprehensive block")
        return False

    # Determine categories
    categories = determine_mixin_category(file_path, content)

    # Build TYPE_CHECKING block
    type_block = build_type_checking_block(categories)

    # Find class definition
    class_match = re.search(r'^(class \w+.*?:)\s*\n', content, re.MULTILINE)
    if not class_match:
        print(f"  ⚠️  {file_path.name} - No class found")
        return False

    # Find where to insert (after class definition, before first method/attribute)
    class_line_end = class_match.end()

    # Skip docstring if present
    after_class = content[class_line_end:]
    lines = after_class.split('\n')

    insert_idx = 0
    in_docstring = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        if stripped.startswith('"""') or stripped.startswith("'''"):
            if in_docstring:
                insert_idx = i + 1
                break
            else:
                in_docstring = True
                continue

        if not in_docstring and (stripped.startswith('def ') or stripped.startswith('if TYPE_CHECKING:') or
                                  (stripped and not stripped.startswith('#'))):
            insert_idx = i
            break

    # Remove old TYPE_CHECKING block if present
    old_type_checking = re.search(
        r'\n    if TYPE_CHECKING:.*?(?=\n    [a-z@]|\n[a-z]|$)',
        after_class,
        re.DOTALL
    )
    if old_type_checking:
        after_class = after_class[:old_type_checking.start()] + after_class[old_type_checking.end():]
        lines = after_class.split('\n')

    # Insert new block
    lines.insert(insert_idx, type_block)
    new_content = content[:class_line_end] + '\n'.join(lines)

    # Ensure imports
    if 'from __future__ import annotations' not in new_content:
        new_content = 'from __future__ import annotations\n\n' + new_content

    if 'from pathlib import Path' not in new_content:
        # Add after __future__
        new_content = new_content.replace(
            'from __future__ import annotations\n',
            'from __future__ import annotations\n\nfrom pathlib import Path\n'
        )

    # Update typing imports
    if 'from typing import' in new_content:
        # Add TYPE_CHECKING, Optional, Any, Dict, List, Set if not present
        needed = ['TYPE_CHECKING', 'Optional', 'Any', 'Dict', 'List', 'Set']
        typing_line = re.search(r'from typing import ([^\n]+)', new_content)
        if typing_line:
            current = typing_line.group(1)
            for imp in needed:
                if imp not in current:
                    current = current.rstrip() + f', {imp}'
            new_content = new_content.replace(typing_line.group(0), f'from typing import {current}')
    else:
        # Add typing import
        new_content = new_content.replace(
            'from pathlib import Path\n',
            'from pathlib import Path\nfrom typing import TYPE_CHECKING, Optional, Any, Dict, List, Set\n'
        )

    if dry_run:
        print(f"  🔍 Would update {file_path.name} (categories: {', '.join(categories)})")
        return True

    # Write back
    file_path.write_text(new_content)
    print(f"  ✅ Updated {file_path.name} (categories: {', '.join(categories)})")
    return True

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Add comprehensive TYPE_CHECKING to mixins")
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    # Find all mixin files
    mixin_patterns = [
        'core/intelligence/orchestration/templates/*_mixin.py',
        'core/intelligence/orchestration/*_mixin.py',
        'core/intelligence/orchestration/protocols/*.py',
        'core/intelligence/swarms/*_mixin.py',
        'core/intelligence/memory/*_mixin.py',
        'core/intelligence/swarms/coding_swarm/*_mixin.py',
    ]

    mixin_files = []
    for pattern in mixin_patterns:
        mixin_files.extend(Path('.').glob(pattern))

    mixin_files = sorted(set(mixin_files))

    print(f"🔍 Found {len(mixin_files)} mixin files")
    print()

    modified = 0
    for file_path in mixin_files:
        if fix_mixin_file(file_path, dry_run=args.dry_run):
            modified += 1

    print()
    print(f"{'🔍 Would modify' if args.dry_run else '✅ Modified'} {modified}/{len(mixin_files)} files")

    if args.dry_run:
        print("\nRun without --dry-run to apply changes")

if __name__ == '__main__':
    main()
