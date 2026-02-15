"""
Skill Package Management Tools

Tools that agents can use to install packages in skill venvs.
These tools are automatically registered and available to agents.
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def create_package_management_tools(venv_manager: Any) -> Dict[str, Any]:
    """
    Create package management tools for agents.

    Args:
        venv_manager: SkillVenvManager instance

    Returns:
        Dict mapping tool names to execute functions
    """

    def install_packages_tool(params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Install packages in skill venv.

        Args:
            params: Dictionary containing:
                - packages: List of package names (e.g., ['torch', 'diffusers'])
                - skill_name: Optional skill name (uses shared venv if not provided)
                - upgrade: Optional bool to upgrade packages

        Returns:
            Dict with success status and output
        """
        packages = params.get("packages", [])
        skill_name = params.get("skill_name")
        upgrade = params.get("upgrade", False)

        if not packages:
            return {
                "success": False,
                "error": "Missing required parameter: packages (list of package names)",
            }

        if not isinstance(packages, list):
            return {"success": False, "error": "packages must be a list of package names"}

        result = venv_manager.install_packages(
            packages=packages, skill_name=skill_name, upgrade=upgrade
        )

        return result

    def check_packages_tool(params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if packages are installed in skill venv.

        Args:
            params: Dictionary containing:
                - packages: List of package names to check
                - skill_name: Optional skill name

        Returns:
            Dict with installation status for each package
        """
        packages = params.get("packages", [])
        skill_name = params.get("skill_name")

        if not packages:
            return {"success": False, "error": "Missing required parameter: packages"}

        if not isinstance(packages, list):
            packages = [packages]

        installed = venv_manager.list_installed_packages(skill_name)
        installed_lower = [pkg.lower() for pkg in installed]

        status = {}
        for pkg in packages:
            status[pkg] = pkg.lower() in installed_lower

        return {"success": True, "packages": status, "all_installed": all(status.values())}

    def list_packages_tool(params: Dict[str, Any]) -> Dict[str, Any]:
        """
        List installed packages in skill venv.

        Args:
            params: Dictionary containing:
                - skill_name: Optional skill name

        Returns:
            Dict with list of installed packages
        """
        skill_name = params.get("skill_name")
        packages = venv_manager.list_installed_packages(skill_name)

        return {"success": True, "packages": packages, "count": len(packages)}

    return {
        "install_packages": install_packages_tool,
        "check_packages": check_packages_tool,
        "list_packages": list_packages_tool,
    }
