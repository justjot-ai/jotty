#!/usr/bin/env python3
"""
Update Agent Imports After Reorganization
==========================================

Maps old agent.base.* imports to new organized structure.
"""

from pathlib import Path

# Mapping: old path → new path
IMPORT_MAPPINGS = {
    # Types
    "core.intelligence.reasoning.agent.base._execution_types": "core.intelligence.reasoning.agent.types.execution_types",
    "core.intelligence.reasoning.agent.base.dag_types": "core.intelligence.reasoning.agent.types.dag_types",
    "core.intelligence.reasoning.agent.base.planner_signatures": "core.intelligence.reasoning.agent.types.planner_signatures",
    # Mixins
    "core.intelligence.reasoning.agent.base._skill_selection_mixin": "core.intelligence.reasoning.agent.mixins.skill_selection",
    "core.intelligence.reasoning.agent.base._plan_utils_mixin": "core.intelligence.reasoning.agent.mixins.plan_utils",
    "core.intelligence.reasoning.agent.base._inference_mixin": "core.intelligence.reasoning.agent.mixins.inference",
    # Agents (concrete implementations)
    "core.intelligence.reasoning.agent.base.auto_agent": "core.intelligence.reasoning.agent.agents.auto_agent",
    "core.intelligence.reasoning.agent.base.autonomous_agent": "core.intelligence.reasoning.agent.agents.autonomous_agent",
    "core.intelligence.reasoning.agent.base.chat_assistant": "core.intelligence.reasoning.agent.agents.chat_assistant",
    "core.intelligence.reasoning.agent.base.chat_assistant_v2": "core.intelligence.reasoning.agent.agents.chat_assistant_v2",
    "core.intelligence.reasoning.agent.base.composite_agent": "core.intelligence.reasoning.agent.agents.composite_agent",
    "core.intelligence.reasoning.agent.base.domain_agent": "core.intelligence.reasoning.agent.agents.domain_agent",
    "core.intelligence.reasoning.agent.base.dspy_mcp_agent": "core.intelligence.reasoning.agent.agents.dspy_mcp_agent",
    "core.intelligence.reasoning.agent.base.meta_agent": "core.intelligence.reasoning.agent.agents.meta_agent",
    "core.intelligence.reasoning.agent.base.model_chat_agent": "core.intelligence.reasoning.agent.agents.model_chat_agent",
    "core.intelligence.reasoning.agent.base.skill_based_agent": "core.intelligence.reasoning.agent.agents.skill_based_agent",
    "core.intelligence.reasoning.agent.base.swarm_agent": "core.intelligence.reasoning.agent.agents.swarm_agent",
    "core.intelligence.reasoning.agent.base.task_breakdown_agent": "core.intelligence.reasoning.agent.agents.task_breakdown_agent",
    "core.intelligence.reasoning.agent.base.todo_creator_agent": "core.intelligence.reasoning.agent.agents.todo_creator_agent",
    "core.intelligence.reasoning.agent.base.validation_agent": "core.intelligence.reasoning.agent.agents.validation_agent",
    # Executors
    "core.intelligence.reasoning.agent.base.skill_plan_executor": "core.intelligence.reasoning.agent.executors.skill_plan_executor",
    "core.intelligence.reasoning.agent.base.step_processors": "core.intelligence.reasoning.agent.executors.step_processors",
    # Planners
    "core.intelligence.reasoning.agent.base.agentic_planner": "core.intelligence.reasoning.agent.planners.agentic_planner",
    "core.intelligence.reasoning.agent.base.dag_agents": "core.intelligence.reasoning.agent.planners.dag_agents",
    # Tools
    "core.intelligence.reasoning.agent.base.section_tools": "core.intelligence.reasoning.agent.tools.section_tools",
    "core.intelligence.reasoning.agent.base.inspector": "core.intelligence.reasoning.agent.tools.inspector",
    "core.intelligence.reasoning.agent.base.feedback_channel": "core.intelligence.reasoning.agent.tools.feedback_channel",
    "core.intelligence.reasoning.agent.base.axon": "core.intelligence.reasoning.agent.tools.axon",
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
                (f"from {old_path}", f"from {new_path}"),
                (f"import {old_path}", f"import {new_path}"),
                # Also handle Jotty. prefix
                (f"from Jotty.{old_path}", f"from Jotty.{new_path}"),
                (f"import Jotty.{old_path}", f"import Jotty.{new_path}"),
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
    root = Path(".")
    files_updated = 0
    total_changes = 0

    print("Updating agent imports...\n")

    # Find all Python files
    for py_file in root.rglob("*.py"):
        # Skip certain directories
        if any(skip in str(py_file) for skip in ["__pycache__", ".backup", "generated", ".git"]):
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


if __name__ == "__main__":
    main()
