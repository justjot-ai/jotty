#!/usr/bin/env python3
"""
Auto-Migrate Skills to Base Architecture
=========================================

Automatically converts skills from old patterns to new BaseLLMSkill/BaseToolSkill.

Usage:
    # Migrate single skill
    python scripts/migrate_skills.py skills/calculator

    # Migrate all skills
    python scripts/migrate_skills.py --all

    # Dry run (preview changes)
    python scripts/migrate_skills.py skills/calculator --dry-run

    # Migrate skills with custom API clients
    python scripts/migrate_skills.py --llm-only
"""

import ast
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class SkillMigrator:
    """Automatically migrate skills to base architecture."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.stats = {
            "migrated": 0,
            "skipped": 0,
            "errors": 0,
            "lines_saved": 0,
        }

    def detect_skill_type(self, content: str) -> Tuple[str, Optional[str]]:
        """
        Detect if skill uses LLM and which provider.

        Returns:
            (skill_type, provider) where skill_type is 'llm' or 'tool'
        """
        # Check for LLM provider imports
        if re.search(r"import anthropic|from anthropic", content, re.IGNORECASE):
            return "llm", "anthropic"
        elif re.search(r"import openai|from openai", content, re.IGNORECASE):
            return "llm", "openai"
        elif re.search(r"import groq|from groq", content, re.IGNORECASE):
            return "llm", "groq"
        elif re.search(r"import dspy|from dspy", content, re.IGNORECASE):
            return "llm", "anthropic"  # Default to anthropic for dspy
        else:
            return "tool", None

    def extract_functions(self, content: str) -> List[Dict]:
        """Extract all @tool_wrapper decorated functions."""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return []

        functions = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check if has @tool_wrapper decorator
                has_tool_wrapper = any(
                    (isinstance(d, ast.Name) and d.id == "tool_wrapper")
                    or (
                        isinstance(d, ast.Call)
                        and isinstance(d.func, ast.Name)
                        and d.func.id == "tool_wrapper"
                    )
                    for d in node.decorator_list
                )

                if has_tool_wrapper:
                    # Extract required params from decorator
                    required_params = []
                    for decorator in node.decorator_list:
                        if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
                            if decorator.func.id == "tool_wrapper":
                                for keyword in decorator.keywords:
                                    if keyword.arg == "required_params":
                                        if isinstance(keyword.value, ast.List):
                                            required_params = [
                                                elt.s if isinstance(elt, ast.Str) else elt.value
                                                for elt in keyword.value.elts
                                            ]

                    functions.append(
                        {
                            "name": node.name,
                            "required_params": required_params,
                            "docstring": ast.get_docstring(node) or "",
                        }
                    )

        return functions

    def generate_tool_skill(self, skill_name: str, content: str, functions: List[Dict]) -> str:
        """Generate migrated tool skill code."""
        # Extract module docstring
        try:
            tree = ast.parse(content)
            module_docstring = ast.get_docstring(tree) or f"{skill_name} Skill"
        except Exception:
            module_docstring = f"{skill_name} Skill"

        # Extract imports (exclude old ones)
        imports = []
        for line in content.split("\n"):
            if line.strip().startswith(("import ", "from ")):
                # Skip old imports we're replacing
                if any(
                    x in line
                    for x in ["SkillStatus", "tool_error", "tool_response", "tool_wrapper"]
                ):
                    continue
                if any(x in line for x in ["anthropic", "openai", "groq"]):
                    continue
                imports.append(line)

        # Generate skill classes
        classes = []
        for func in functions:
            class_name = "".join(
                word.capitalize() for word in func["name"].replace("_tool", "").split("_")
            )
            if not class_name.endswith("Skill"):
                class_name += "Skill"

            required = ", ".join(f'"{p}"' for p in func["required_params"])

            class_code = f'''
class {class_name}(BaseToolSkill):
    """{func['docstring'] or 'TODO: Add docstring'}"""

    @validate_params(required=[{required}])
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        # TODO: Implement logic from {func['name']}
        # Extract params
        {chr(10).join(f"        {p} = params['{p}']" for p in func['required_params'])}

        # Your logic here
        result = None  # TODO: Implement

        return self.success(result=result)
'''
            classes.append(class_code)

        # Generate instances and wrappers
        instances = []
        wrappers = []
        exports = []

        for func in functions:
            instance_name = func["name"].replace("_tool", "")
            class_name = "".join(word.capitalize() for word in instance_name.split("_"))
            if not class_name.endswith("Skill"):
                class_name += "Skill"

            instances.append(f"{instance_name} = {class_name}('{instance_name}')")

            required = ", ".join(f'"{p}"' for p in func["required_params"])
            wrapper = f'''
@tool_wrapper(required_params=[{required}])
def {func['name']}(params: Dict[str, Any]) -> Dict[str, Any]:
    """{func['docstring'] or 'TODO: Add docstring'}"""
    return {instance_name}(params)
'''
            wrappers.append(wrapper)
            exports.append(f'"{func["name"]}"')

        # Combine everything
        migrated = f'''"""
{module_docstring}

Migrated to BaseToolSkill architecture.
"""

from typing import Any, Dict

from skills._base import BaseToolSkill, validate_params
from Jotty.core.infrastructure.utils.tool_helpers import tool_wrapper

{chr(10).join(imports)}

{''.join(classes)}

# Create skill instances
{chr(10).join(instances)}


# Backward compatibility wrappers
{''.join(wrappers)}

__all__ = [{', '.join(exports)}]
'''

        return migrated

    def generate_llm_skill(
        self, skill_name: str, content: str, functions: List[Dict], provider: str
    ) -> str:
        """Generate migrated LLM skill code."""
        # Extract module docstring
        try:
            tree = ast.parse(content)
            module_docstring = ast.get_docstring(tree) or f"{skill_name} Skill"
        except Exception:
            module_docstring = f"{skill_name} Skill"

        # Generate skill classes
        classes = []
        for func in functions:
            class_name = "".join(
                word.capitalize() for word in func["name"].replace("_tool", "").split("_")
            )
            if not class_name.endswith("Skill"):
                class_name += "Skill"

            required = ", ".join(f'"{p}"' for p in func["required_params"])

            class_code = f'''
class {class_name}(BaseLLMSkill):
    """{func['docstring'] or 'TODO: Add docstring'}"""

    def __init__(self):
        super().__init__("{func['name'].replace('_tool', '')}", provider="{provider}")

    @validate_params(required=[{required}])
    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        # TODO: Implement logic from {func['name']}
        # Extract params
        {chr(10).join(f"        {p} = params['{p}']" for p in func['required_params'])}

        # LLM call (replace with actual logic)
        result = self.call_lm(f"TODO: Implement prompt")

        return self.success(result=result)
'''
            classes.append(class_code)

        # Generate instances and wrappers
        instances = []
        wrappers = []
        exports = []

        for func in functions:
            instance_name = func["name"].replace("_tool", "")
            class_name = "".join(word.capitalize() for word in instance_name.split("_"))
            if not class_name.endswith("Skill"):
                class_name += "Skill"

            instances.append(f"{instance_name} = {class_name}()")

            required = ", ".join(f'"{p}"' for p in func["required_params"])
            wrapper = f'''
@tool_wrapper(required_params=[{required}])
def {func['name']}(params: Dict[str, Any]) -> Dict[str, Any]:
    """{func['docstring'] or 'TODO: Add docstring'}"""
    return {instance_name}(params)
'''
            wrappers.append(wrapper)
            exports.append(f'"{func["name"]}"')

        # Combine everything
        migrated = f'''"""
{module_docstring}

Migrated to BaseLLMSkill architecture.
Uses UnifiedLMProvider instead of custom {provider} client.
"""

from typing import Any, Dict

from skills._base import BaseLLMSkill, validate_params
from Jotty.core.infrastructure.utils.tool_helpers import tool_wrapper

{''.join(classes)}

# Create skill instances
{chr(10).join(instances)}


# Backward compatibility wrappers
{''.join(wrappers)}

__all__ = [{', '.join(exports)}]
'''

        return migrated

    def migrate_skill(self, skill_path: Path) -> bool:
        """
        Migrate a single skill.

        Args:
            skill_path: Path to skill directory

        Returns:
            True if migrated successfully
        """
        tools_file = skill_path / "tools.py"

        if not tools_file.exists():
            print(f"⚠️  No tools.py in {skill_path}")
            self.stats["skipped"] += 1
            return False

        # Read original
        original_content = tools_file.read_text()
        original_lines = len(original_content.split("\n"))

        # Detect skill type
        skill_type, provider = self.detect_skill_type(original_content)

        # Extract functions
        functions = self.extract_functions(original_content)

        if not functions:
            print(f"⚠️  No @tool_wrapper functions found in {skill_path}")
            self.stats["skipped"] += 1
            return False

        # Generate migrated code
        skill_name = skill_path.name

        if skill_type == "llm":
            migrated_content = self.generate_llm_skill(
                skill_name, original_content, functions, provider
            )
        else:
            migrated_content = self.generate_tool_skill(skill_name, original_content, functions)

        migrated_lines = len(migrated_content.split("\n"))
        lines_saved = original_lines - migrated_lines

        # Preview or write
        if self.dry_run:
            print(f"\n{'='*60}")
            print(f"📝 {skill_path}")
            print(f"Type: {skill_type.upper()}" + (f" ({provider})" if provider else ""))
            print(f"Functions: {', '.join(f['name'] for f in functions)}")
            print(f"Lines: {original_lines} → {migrated_lines} ({lines_saved:+d} lines)")
            print(f"{'='*60}\n")
        else:
            # Backup original
            backup_file = skill_path / "tools.py.bak"
            shutil.copy2(tools_file, backup_file)

            # Write migrated
            tools_file.write_text(migrated_content)

            print(f"✅ {skill_path} ({original_lines} → {migrated_lines} lines, {lines_saved:+d})")

        self.stats["migrated"] += 1
        self.stats["lines_saved"] += lines_saved

        return True

    def migrate_all_skills(self, skills_dir: Path = Path("skills"), llm_only: bool = False):
        """Migrate all skills in directory."""
        for skill_path in sorted(skills_dir.iterdir()):
            if not skill_path.is_dir():
                continue

            if skill_path.name.startswith(("_", ".")):
                continue

            tools_file = skill_path / "tools.py"
            if not tools_file.exists():
                continue

            # Check if LLM-only filter
            if llm_only:
                content = tools_file.read_text()
                skill_type, _ = self.detect_skill_type(content)
                if skill_type != "llm":
                    continue

            try:
                self.migrate_skill(skill_path)
            except Exception as e:
                print(f"❌ Error migrating {skill_path}: {e}")
                self.stats["errors"] += 1

    def print_summary(self):
        """Print migration summary."""
        print(f"\n{'='*60}")
        print("📊 MIGRATION SUMMARY")
        print(f"{'='*60}")
        print(f"✅ Migrated: {self.stats['migrated']}")
        print(f"⚠️  Skipped:  {self.stats['skipped']}")
        print(f"❌ Errors:   {self.stats['errors']}")
        print(f"📉 Lines saved: {self.stats['lines_saved']:+,d}")
        print(f"{'='*60}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Auto-migrate skills to base architecture")
    parser.add_argument(
        "skill_path", nargs="?", help="Path to skill directory (e.g., skills/calculator)"
    )
    parser.add_argument("--all", action="store_true", help="Migrate all skills")
    parser.add_argument(
        "--llm-only", action="store_true", help="Only migrate LLM skills (with custom clients)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview changes without modifying files"
    )

    args = parser.parse_args()

    migrator = SkillMigrator(dry_run=args.dry_run)

    if args.all or args.llm_only:
        print(f"🔄 Migrating {'LLM ' if args.llm_only else ''}skills...")
        migrator.migrate_all_skills(llm_only=args.llm_only)
    elif args.skill_path:
        skill_path = Path(args.skill_path)
        migrator.migrate_skill(skill_path)
    else:
        parser.print_help()
        return

    migrator.print_summary()


if __name__ == "__main__":
    main()
