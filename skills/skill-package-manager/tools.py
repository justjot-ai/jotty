"""
Skill Package Manager - Manage dependencies and virtual environments for skills.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def install_skill_dependencies(
    skill_name: str, requirements: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Install all required packages for a skill."""
    from .skill_dependency_manager import get_dependency_manager

    try:
        dep_manager = get_dependency_manager()

        # If specific requirements provided, use those
        # Otherwise extract from skill code
        if requirements:
            result = dep_manager.install_dependencies(skill_name, requirements)
        else:
            result = dep_manager.ensure_dependencies(skill_name)

        return {
            "success": result.get("success", False),
            "installed_packages": result.get("installed", []),
            "error": result.get("error"),
        }
    except Exception as e:
        logger.error(f"Failed to install dependencies for {skill_name}: {e}")
        return {
            "success": False,
            "installed_packages": [],
            "error": str(e),
        }


def create_skill_venv(skill_name: str, shared: bool = True) -> Dict[str, Any]:
    """Create an isolated virtual environment for a skill."""
    from .skill_venv_manager import get_venv_manager

    try:
        venv_manager = get_venv_manager(shared_venv=shared)
        venv_path = venv_manager.ensure_venv(skill_name)

        return {
            "success": True,
            "venv_path": str(venv_path),
        }
    except Exception as e:
        logger.error(f"Failed to create venv for {skill_name}: {e}")
        return {
            "success": False,
            "venv_path": "",
            "error": str(e),
        }


def check_skill_dependencies(skill_name: str) -> Dict[str, Any]:
    """Check if all dependencies for a skill are installed."""
    from .skill_dependency_manager import get_dependency_manager

    try:
        dep_manager = get_dependency_manager()
        result = dep_manager.check_dependencies(skill_name)

        return {
            "success": result.get("all_installed", False),
            "missing_packages": result.get("missing", []),
            "installed_packages": result.get("installed", []),
        }
    except Exception as e:
        logger.error(f"Failed to check dependencies for {skill_name}: {e}")
        return {
            "success": False,
            "missing_packages": [],
            "installed_packages": [],
            "error": str(e),
        }
