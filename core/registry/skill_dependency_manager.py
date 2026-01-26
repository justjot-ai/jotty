"""
Skill Dependency Manager

Manages dependencies for skills, automatically installing packages
when skills are loaded or used.

Integrates with SkillVenvManager to install packages in isolated venvs.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import re

logger = logging.getLogger(__name__)


class SkillDependencyManager:
    """
    Manages skill dependencies and auto-installation.
    
    Reads requirements from skills and installs them automatically.
    """
    
    def __init__(self, venv_manager=None):
        """
        Initialize dependency manager.
        
        Args:
            venv_manager: SkillVenvManager instance
        """
        from .skill_venv_manager import get_venv_manager
        self.venv_manager = venv_manager or get_venv_manager()
    
    def extract_requirements_from_code(self, code: str) -> List[str]:
        """
        Extract package requirements from Python code.
        
        Looks for:
        - import statements
        - from X import Y
        - Common patterns
        
        Args:
            code: Python code string
            
        Returns:
            List of package names
        """
        requirements = []
        
        # Common import -> package mappings
        import_map = {
            'torch': 'torch',
            'PIL': 'pillow',
            'Pillow': 'pillow',
            'numpy': 'numpy',
            'pandas': 'pandas',
            'requests': 'requests',
            'diffusers': 'diffusers',
            'transformers': 'transformers',
            'accelerate': 'accelerate',
            'pytz': 'pytz',
            'dateutil': 'python-dateutil',
            'psutil': 'psutil',
            'bs4': 'beautifulsoup4',
            'BeautifulSoup': 'beautifulsoup4',
            'html2text': 'html2text',
        }
        
        # Extract imports
        import_pattern = r'^(?:from|import)\s+(\w+)'
        for line in code.split('\n'):
            match = re.match(import_pattern, line.strip())
            if match:
                module = match.group(1)
                if module in import_map:
                    package = import_map[module]
                    if package not in requirements:
                        requirements.append(package)
        
        return requirements
    
    def check_and_install_dependencies(
        self,
        skill_name: str,
        tools_code: Optional[str] = None,
        requirements_file: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Check and install dependencies for a skill.
        
        Args:
            skill_name: Skill name
            tools_code: Python code from tools.py (for auto-detection)
            requirements_file: Path to requirements.txt file
            
        Returns:
            Dict with installation status
        """
        packages_to_install = []
        
        # Read requirements from file if exists
        if requirements_file and requirements_file.exists():
            with open(requirements_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Parse requirement line (handle version specifiers)
                        package = line.split('==')[0].split('>=')[0].split('<=')[0].split('~=')[0].strip()
                        packages_to_install.append(package)
        
        # Extract from code if provided
        if tools_code:
            code_requirements = self.extract_requirements_from_code(tools_code)
            for pkg in code_requirements:
                if pkg not in packages_to_install:
                    packages_to_install.append(pkg)
        
        if not packages_to_install:
            return {
                "success": True,
                "message": "No dependencies found",
                "packages": []
            }
        
        # Check which packages are already installed
        installed = self.venv_manager.list_installed_packages(skill_name)
        missing = [pkg for pkg in packages_to_install if pkg.lower() not in [i.lower() for i in installed]]
        
        if not missing:
            return {
                "success": True,
                "message": "All dependencies already installed",
                "packages": packages_to_install
            }
        
        # Install missing packages
        logger.info(f"Installing dependencies for {skill_name}: {missing}")
        result = self.venv_manager.install_packages(missing, skill_name)
        
        return {
            "success": result["success"],
            "message": result.get("error") or "Dependencies installed",
            "packages": packages_to_install,
            "installed": missing if result["success"] else [],
            "error": result.get("error")
        }
    
    def ensure_skill_dependencies(self, skill_name: str, skill_dir: Path) -> Dict[str, Any]:
        """
        Ensure skill dependencies are installed.
        
        Checks for requirements.txt or extracts from tools.py.
        
        Args:
            skill_name: Skill name
            skill_dir: Skill directory path
            
        Returns:
            Dict with installation status
        """
        tools_py = skill_dir / "tools.py"
        requirements_txt = skill_dir / "requirements.txt"
        
        tools_code = None
        if tools_py.exists():
            tools_code = tools_py.read_text()
        
        return self.check_and_install_dependencies(
            skill_name=skill_name,
            tools_code=tools_code,
            requirements_file=requirements_txt if requirements_txt.exists() else None
        )


# Singleton instance
_dependency_manager_instance: Optional[SkillDependencyManager] = None


def get_dependency_manager(venv_manager=None) -> SkillDependencyManager:
    """Get singleton dependency manager instance."""
    global _dependency_manager_instance
    if _dependency_manager_instance is None:
        _dependency_manager_instance = SkillDependencyManager(venv_manager)
    return _dependency_manager_instance
