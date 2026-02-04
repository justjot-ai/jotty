"""
Prompt Template Selector

Dynamically selects Architect and Auditor prompts based on task type.
Uses pattern matching to choose specialized templates for better validation.
"""

import re
import yaml
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, List

logger = logging.getLogger(__name__)


class PromptSelector:
    """
    Selects appropriate Architect and Auditor prompts based on task analysis.

    Usage:
        selector = PromptSelector()
        architect_path, auditor_path = selector.select_prompts(task_description)
    """

    def __init__(self, registry_path: str = None):
        """
        Initialize the prompt selector.

        Args:
            registry_path: Path to template_registry.yaml
        """
        self.base_path = Path(__file__).parent.parent.parent / "configs" / "prompts"

        if registry_path:
            self.registry_path = Path(registry_path)
        else:
            self.registry_path = self.base_path / "template_registry.yaml"

        self.registry = self._load_registry()
        self._compiled_patterns: Dict[str, List[re.Pattern]] = {}
        self._compile_patterns()

    def _load_registry(self) -> dict:
        """Load the template registry YAML."""
        try:
            if self.registry_path.exists():
                with open(self.registry_path, 'r') as f:
                    return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load prompt registry: {e}")

        # Return default config if file doesn't exist
        return {
            'defaults': {
                'architect': 'base_architect.md',
                'auditor': 'base_auditor.md'
            },
            'templates': {}
        }

    def _compile_patterns(self):
        """Pre-compile regex patterns for efficiency."""
        templates = self.registry.get('templates', {})

        for template_name, template_config in templates.items():
            patterns = template_config.get('patterns', [])
            self._compiled_patterns[template_name] = [
                re.compile(pattern, re.IGNORECASE)
                for pattern in patterns
            ]

    def detect_task_type(self, task: str) -> str:
        """
        Detect the task type from task description.

        Args:
            task: Task description

        Returns:
            Template name (e.g., 'code_generation', 'research')
        """
        task_lower = task.lower()

        # Check each template's patterns
        for template_name, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(task_lower):
                    logger.debug(f"Task matched template '{template_name}' via pattern: {pattern.pattern}")
                    return template_name

        return 'default'

    def select_prompts(self, task: str) -> Tuple[str, str]:
        """
        Select appropriate Architect and Auditor prompts for a task.

        Args:
            task: Task description

        Returns:
            Tuple of (architect_prompt_path, auditor_prompt_path)
        """
        task_type = self.detect_task_type(task)

        if task_type == 'default':
            defaults = self.registry.get('defaults', {})
            architect = defaults.get('architect', 'base_architect.md')
            auditor = defaults.get('auditor', 'base_auditor.md')
        else:
            template = self.registry.get('templates', {}).get(task_type, {})
            architect = template.get('architect', self.registry['defaults']['architect'])
            auditor = template.get('auditor', self.registry['defaults']['auditor'])

        # Build full paths
        architect_path = str(self.base_path / "architect" / architect)
        auditor_path = str(self.base_path / "auditor" / auditor)

        # Fallback to base if specific template doesn't exist
        if not Path(architect_path).exists():
            logger.warning(f"Architect template not found: {architect_path}, using base")
            architect_path = str(self.base_path / "architect" / "base_architect.md")

        if not Path(auditor_path).exists():
            logger.warning(f"Auditor template not found: {auditor_path}, using base")
            auditor_path = str(self.base_path / "auditor" / "base_auditor.md")

        logger.info(f"📋 Selected prompts for task type '{task_type}':")
        logger.info(f"   Architect: {Path(architect_path).name}")
        logger.info(f"   Auditor: {Path(auditor_path).name}")

        return architect_path, auditor_path

    def get_available_templates(self) -> Dict[str, str]:
        """Get list of available templates with descriptions."""
        descriptions = self.registry.get('descriptions', {})
        templates = {'default': 'Base templates for general tasks'}

        for template_name in self.registry.get('templates', {}).keys():
            templates[template_name] = descriptions.get(
                template_name,
                f"Specialized template for {template_name.replace('_', ' ')} tasks"
            )

        return templates


# Singleton instance for easy access
_selector_instance: Optional[PromptSelector] = None


def get_prompt_selector() -> PromptSelector:
    """Get the singleton prompt selector instance."""
    global _selector_instance
    if _selector_instance is None:
        _selector_instance = PromptSelector()
    return _selector_instance


def select_prompts_for_task(task: str) -> Tuple[str, str]:
    """
    Convenience function to select prompts for a task.

    Args:
        task: Task description

    Returns:
        Tuple of (architect_prompt_path, auditor_prompt_path)
    """
    return get_prompt_selector().select_prompts(task)
