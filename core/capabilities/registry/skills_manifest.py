"""
Skills Manifest - Discovery and categorization system for Jotty skills.

Provides:
- Category-based skill discovery
- Tag-based filtering
- Auto-discovery of new skills
- Metadata for authentication requirements
"""
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SkillInfo:
    """Information about a skill."""
    name: str
    category: str = "uncategorized"
    tags: List[str] = field(default_factory=list)
    description: str = ""
    icon: str = ""
    requires_auth: bool = False
    env_vars: List[str] = field(default_factory=list)
    requires_cli: List[str] = field(default_factory=list)
    is_discovered: bool = False  # True if auto-discovered (not in manifest)
    skill_type: str = "base"  # "base", "derived", or "composite"
    base_skills: List[str] = field(default_factory=list)  # Skills this depends on


@dataclass
class CategoryInfo:
    """Information about a category."""
    name: str
    description: str
    icon: str
    skills: List[str] = field(default_factory=list)


class SkillsManifest:
    """
    Skills manifest for discovery and categorization.

    Supports:
    - Loading manifest from YAML
    - Auto-discovering new skills not in manifest
    - Querying skills by category or tag
    - Getting skill metadata
    """

    def __init__(self, skills_dir: Optional[str] = None, manifest_path: Optional[str] = None) -> None:
        """
        Initialize skills manifest.

        Args:
            skills_dir: Directory containing skills
            manifest_path: Path to skills_manifest.yaml
        """
        if skills_dir is None:
            current_file = Path(__file__).resolve()
            repo_root = current_file.parent.parent.parent
            skills_dir = str(repo_root / "skills")

        self.skills_dir = Path(skills_dir)

        if manifest_path is None:
            manifest_path = self.skills_dir / "skills_manifest.yaml"

        self.manifest_path = Path(manifest_path)

        self.categories: Dict[str, CategoryInfo] = {}
        self.skills: Dict[str, SkillInfo] = {}
        self.tags: Dict[str, List[str]] = {}
        self.auto_discover = True

        self._load_manifest()
        if self.auto_discover:
            self._discover_new_skills()

    def _load_manifest(self) -> None:
        """Load manifest from YAML file."""
        if not self.manifest_path.exists():
            logger.warning(f"Manifest not found: {self.manifest_path}")
            return

        try:
            import yaml
        except ImportError:
            logger.error("PyYAML not installed. Run: pip install pyyaml")
            return

        try:
            with open(self.manifest_path, 'r') as f:
                data = yaml.safe_load(f)

            self.auto_discover = data.get('auto_discover', True)

            # Load categories
            for cat_name, cat_data in data.get('categories', {}).items():
                self.categories[cat_name] = CategoryInfo(
                    name=cat_name,
                    description=cat_data.get('description', ''),
                    icon=cat_data.get('icon', ''),
                    skills=cat_data.get('skills', [])
                )

                # Create skill entries for each skill in category
                for skill_name in cat_data.get('skills', []):
                    if skill_name not in self.skills:
                        self.skills[skill_name] = SkillInfo(
                            name=skill_name,
                            category=cat_name,
                            icon=cat_data.get('icon', '')
                        )
                    else:
                        self.skills[skill_name].category = cat_name
                        self.skills[skill_name].icon = cat_data.get('icon', '')

            # Load tags
            for tag_name, tag_data in data.get('tags', {}).items():
                tag_skills = tag_data.get('skills', [])
                self.tags[tag_name] = tag_skills

                # Add tags to skill info
                for skill_name in tag_skills:
                    if skill_name in self.skills:
                        if tag_name not in self.skills[skill_name].tags:
                            self.skills[skill_name].tags.append(tag_name)

            # Load skill metadata
            for skill_name, meta in data.get('skill_metadata', {}).items():
                if skill_name in self.skills:
                    self.skills[skill_name].requires_auth = meta.get('requires_auth', False)
                    self.skills[skill_name].env_vars = meta.get('env_vars', [])
                    self.skills[skill_name].requires_cli = meta.get('requires_cli', [])

            # Load skill type classifications
            for type_name, type_skills in data.get('skill_types', {}).items():
                if type_name in ('base', 'derived', 'composite'):
                    for entry in (type_skills or []):
                        if isinstance(entry, str):
                            skill_name = entry
                            base_skills_list = []
                        elif isinstance(entry, dict):
                            skill_name = entry.get('name', '')
                            base_skills_list = entry.get('base_skills', [])
                        else:
                            continue

                        if skill_name in self.skills:
                            self.skills[skill_name].skill_type = type_name
                            self.skills[skill_name].base_skills = base_skills_list

            logger.info(f"Loaded manifest: {len(self.categories)} categories, {len(self.skills)} skills")

        except Exception as e:
            logger.error(f"Failed to load manifest: {e}")

    def _discover_new_skills(self) -> None:
        """Auto-discover skills not in manifest."""
        if not self.skills_dir.exists():
            return

        discovered = []

        for item in self.skills_dir.iterdir():
            if item.is_dir() and not item.name.startswith(('.', '_')):
                skill_name = item.name
                tools_file = item / "tools.py"

                if tools_file.exists() and skill_name not in self.skills:
                    # New skill discovered
                    self.skills[skill_name] = SkillInfo(
                        name=skill_name,
                        category="uncategorized",
                        is_discovered=True
                    )
                    discovered.append(skill_name)

        if discovered:
            # Add uncategorized category if needed
            if "uncategorized" not in self.categories:
                self.categories["uncategorized"] = CategoryInfo(
                    name="uncategorized",
                    description="Newly discovered skills (not yet categorized)",
                    icon="🆕",
                    skills=discovered
                )
            else:
                self.categories["uncategorized"].skills.extend(discovered)

            logger.info(f"Auto-discovered {len(discovered)} new skills: {discovered}")

    def refresh(self) -> None:
        """Refresh manifest (reload + rediscover)."""
        self.categories.clear()
        self.skills.clear()
        self.tags.clear()
        self._load_manifest()
        if self.auto_discover:
            self._discover_new_skills()

    def get_categories(self) -> List[CategoryInfo]:
        """Get all categories."""
        return list(self.categories.values())

    def get_category(self, name: str) -> Optional[CategoryInfo]:
        """Get category by name."""
        return self.categories.get(name)

    def get_skills_by_category(self, category: str) -> List[SkillInfo]:
        """Get all skills in a category."""
        return [
            self.skills[name]
            for name in self.skills
            if self.skills[name].category == category
        ]

    def get_skills_by_tag(self, tag: str) -> List[SkillInfo]:
        """Get all skills with a specific tag."""
        skill_names = self.tags.get(tag, [])
        return [self.skills[name] for name in skill_names if name in self.skills]

    def get_skill(self, name: str) -> Optional[SkillInfo]:
        """Get skill info by name."""
        return self.skills.get(name)

    def get_all_skills(self) -> List[SkillInfo]:
        """Get all skills."""
        return list(self.skills.values())

    def get_uncategorized_skills(self) -> List[SkillInfo]:
        """Get skills that are not yet categorized."""
        return [s for s in self.skills.values() if s.is_discovered or s.category == "uncategorized"]

    def get_skills_by_type(self, skill_type: str) -> List[SkillInfo]:
        """Get all skills of a specific type (base, derived, composite)."""
        return [s for s in self.skills.values() if s.skill_type == skill_type]

    def get_type_summary(self) -> Dict[str, int]:
        """Get count of skills by type."""
        counts = {"base": 0, "derived": 0, "composite": 0}
        for skill in self.skills.values():
            if skill.skill_type in counts:
                counts[skill.skill_type] += 1
        return counts

    def search_skills(self, query: str) -> List[SkillInfo]:
        """Search skills by name, category, or tag."""
        query = query.lower()
        results = []

        for skill in self.skills.values():
            if (query in skill.name.lower() or
                query in skill.category.lower() or
                any(query in tag.lower() for tag in skill.tags)):
                results.append(skill)

        return results

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of manifest for agent discovery."""
        summary = {
            "total_skills": len(self.skills),
            "total_categories": len(self.categories),
            "categories": {}
        }

        for cat_name, cat_info in self.categories.items():
            summary["categories"][cat_name] = {
                "icon": cat_info.icon,
                "description": cat_info.description,
                "skill_count": len(cat_info.skills),
                "skills": cat_info.skills
            }

        return summary

    def get_discovery_prompt(self) -> str:
        """Generate a prompt for the agent to understand available skills."""
        lines = ["# Available Skills\n"]

        for cat_name, cat_info in self.categories.items():
            if cat_info.skills:
                lines.append(f"\n## {cat_info.icon} {cat_name.title()}")
                lines.append(f"*{cat_info.description}*\n")

                for skill_name in cat_info.skills:
                    skill = self.skills.get(skill_name)
                    if skill:
                        auth_note = " " if skill.requires_auth else ""
                        lines.append(f"- `{skill_name}`{auth_note}")

        # Add tags section
        lines.append("\n## Tags")
        for tag_name, tag_skills in self.tags.items():
            lines.append(f"- **{tag_name}**: {', '.join(tag_skills[:5])}{'...' if len(tag_skills) > 5 else ''}")

        return "\n".join(lines)

    def add_skill_to_category(self, skill_name: str, category: str) -> bool:
        """Add a skill to a category (updates manifest)."""
        if skill_name not in self.skills:
            return False

        if category not in self.categories:
            return False

        # Update in-memory
        old_category = self.skills[skill_name].category
        self.skills[skill_name].category = category
        self.skills[skill_name].is_discovered = False

        # Update category lists
        if old_category in self.categories:
            if skill_name in self.categories[old_category].skills:
                self.categories[old_category].skills.remove(skill_name)

        if skill_name not in self.categories[category].skills:
            self.categories[category].skills.append(skill_name)

        # Save manifest
        self._save_manifest()
        return True

    def _save_manifest(self) -> None:
        """Save manifest back to YAML file."""
        try:
            import yaml
        except ImportError:
            logger.error("PyYAML not installed")
            return

        try:
            # Read existing manifest to preserve structure
            with open(self.manifest_path, 'r') as f:
                data = yaml.safe_load(f)

            # Update categories
            for cat_name, cat_info in self.categories.items():
                if cat_name in data.get('categories', {}):
                    data['categories'][cat_name]['skills'] = cat_info.skills

            # Write back
            with open(self.manifest_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)

            logger.info("Manifest saved")

        except Exception as e:
            logger.error(f"Failed to save manifest: {e}")


# Singleton instance
_manifest_instance: Optional[SkillsManifest] = None


def get_skills_manifest(refresh: bool = False) -> SkillsManifest:
    """Get singleton manifest instance."""
    global _manifest_instance

    if _manifest_instance is None or refresh:
        _manifest_instance = SkillsManifest()

    return _manifest_instance
