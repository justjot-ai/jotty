#!/usr/bin/env python3
"""
Update Agent Imports After Reorganization
==========================================

Maps old agent.base.* imports to new organized structure.
"""

import re
from pathlib import Path

# Mapping: old path → new path
IMPORT_MAPPINGS = {
    # Types
    'core.modes.agent.base._execution_types': 'core.modes.agent.types.execution_types',
    'core.modes.agent.base.dag_types': 'core.modes.agent.types.dag_types',
    'core.modes.agent.base.planner_signatures': 'core.modes.agent.types.planner_signatures',

    # Mixins
    'core.modes.agent.base._skill_selection_mixin': 'core.modes.agent.mixins.skill_selection',
    'core.modes.agent.base._plan_utils_mixin': 'core.modes.agent.mixins.plan_utils',
    'core.modes.agent.base._inference_mixin': 'core.modes.agent.mixins.inference',

    # Implementations
    'core.modes.agent.base.auto_agent': 'core.modes.agent.implementations.auto_agent',
    'core.modes.agent.base.autonomous_agent': 'core.modes.agent.implementations.autonomous_agent',
    'core.modes.agent.base.chat_assistant': 'core.modes.agent.implementations.chat_assistant',
    'core.modes.agent.base.chat_assistant_v2': 'core.modes.agent.implementations.chat_assistant_v2',
    'core.modes.agent.base.composite_agent': 'core.modes.agent.implementations.composite_agent',
    'core.modes.agent.base.domain_agent': 'core.modes.agent.implementations.domain_agent',
    'core.modes.agent.base.dspy_mcp_agent': 'core.modes.agent.implementations.dspy_mcp_agent',
    'core.modes.agent.base.meta_agent': 'core.modes.agent.implementations.meta_agent',
    'core.modes.agent.base.model_chat_agent': 'core.modes.agent.implementations.model_chat_agent',
    'core.modes.agent.base.skill_based_agent': 'core.modes.agent.implementations.skill_based_agent',
    'core.modes.agent.base.swarm_agent': 'core.modes.agent.implementations.swarm_agent',
    'core.modes.agent.base.task_breakdown_agent': 'core.modes.agent.implementations.task_breakdown_agent',
    'core.modes.agent.base.todo_creator_agent': 'core.modes.agent.implementations.todo_creator_agent',
    'core.modes.agent.base.validation_agent': 'core.modes.agent.implementations.validation_agent',

    # Executors
    'core.modes.agent.base.skill_plan_executor': 'core.modes.agent.executors.skill_plan_executor',
    'core.modes.agent.base.step_processors': 'core.modes.agent.executors.step_processors',

    # Planning
    'core.modes.agent.base.agentic_planner': 'core.modes.agent.planning.agentic_planner',
    'core.modes.agent.base.dag_agents': 'core.modes.agent.planning.dag_agents',

    # Tools
    'core.modes.agent.base.section_tools': 'core.modes.agent.tools.section_tools',
    'core.modes.agent.base.inspector': 'core.modes.agent.tools.inspector',
    'core.modes.agent.base.feedback_channel': 'core.modes.agent.tools.feedback_channel',
    'core.modes.agent.base.axon': 'core.modes.agent.tools.axon',
}

def update_file(file_path: Path) -> tuple[int, list[str]]:
    """Update imports in a single file."""
    try:
        content = file_path.read_text()
        original = content
        changes = []

        for old_path, new_path in IMPORT_MAPPINGS.items():
            # Handle both "from X import Y" and "import X"
            patterns = [
                (f'from {old_path}', f'from {new_path}'),
                (f'import {old_path}', f'import {new_path}'),
                # Also handle Jotty. prefix
                (f'from Jotty.{old_path}', f'from Jotty.{new_path}'),
                (f'import Jotty.{old_path}', f'import Jotty.{new_path}'),
            ]

            for old, new in patterns:
                if old in content:
                    content = content.replace(old, new)
                    changes.append(f"{old} → {new}")

        if content != original:
            file_path.write_text(content)
            return len(changes), changes
        return 0, []

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0, []

def main():
    """Update all Python files."""
    root = Path('.')
    files_updated = 0
    total_changes = 0

    print("Updating agent imports...\n")

    # Find all Python files
    for py_file in root.rglob('*.py'):
        # Skip certain directories
        if any(skip in str(py_file) for skip in ['__pycache__', '.backup', 'generated', '.git']):
            continue

        num_changes, changes = update_file(py_file)
        if num_changes > 0:
            files_updated += 1
            total_changes += num_changes
            print(f"✅ {py_file} ({num_changes} changes)")
            for change in changes[:3]:  # Show first 3 changes
                print(f"   {change}")
            if len(changes) > 3:
                print(f"   ... and {len(changes) - 3} more")

    print(f"\n{'='*60}")
    print(f"✅ Updated {files_updated} files")
    print(f"✅ Made {total_changes} import changes")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
