"""
Skill Virtual Environment Manager

Manages virtual environments for skills, allowing agents to install
packages dynamically without affecting the main environment.

Architecture:
- Each skill can have its own venv (isolated)
- Or skills share a common venv (efficient)
- Agents can install packages via pip
- Automatic venv creation and activation
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import json

logger = logging.getLogger(__name__)


class SkillVenvManager:
    """
    Manages virtual environments for skills.
    
    Allows agents to install packages dynamically for skills.
    """
    
    def __init__(self, venv_base_dir: Optional[str] = None, shared_venv: bool = True) -> None:
        """
        Initialize venv manager.
        
        Args:
            venv_base_dir: Base directory for venvs (default: ~/jotty/venvs)
            shared_venv: If True, use one venv for all skills (efficient)
                       If False, create venv per skill (isolated)
        """
        if venv_base_dir is None:
            home = os.path.expanduser("~")
            venv_base_dir = os.path.join(home, "jotty", "venvs")
        
        self.venv_base_dir = Path(venv_base_dir)
        self.venv_base_dir.mkdir(parents=True, exist_ok=True)
        self.shared_venv = shared_venv
        
        if shared_venv:
            self.shared_venv_path = self.venv_base_dir / "shared"
        else:
            self.shared_venv_path = None
    
    def get_venv_path(self, skill_name: Optional[str] = None) -> Path:
        """
        Get venv path for a skill.
        
        Args:
            skill_name: Skill name (ignored if shared_venv=True)
            
        Returns:
            Path to venv directory
        """
        if self.shared_venv:
            return self.shared_venv_path
        else:
            if not skill_name:
                raise ValueError("skill_name required when shared_venv=False")
            return self.venv_base_dir / skill_name
    
    def create_venv(self, skill_name: Optional[str] = None, python: str = "python3") -> Path:
        """
        Create virtual environment for skill.
        
        Args:
            skill_name: Skill name (for isolated venvs)
            python: Python interpreter to use
            
        Returns:
            Path to created venv
        """
        venv_path = self.get_venv_path(skill_name)
        
        if venv_path.exists():
            logger.info(f"Venv already exists: {venv_path}")
            return venv_path
        
        logger.info(f"Creating venv: {venv_path}")
        
        try:
            # Use venv module
            result = subprocess.run(
                [python, "-m", "venv", str(venv_path)],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Failed to create venv: {result.stderr}")
            
            logger.info(f" Created venv: {venv_path}")
            return venv_path
            
        except Exception as e:
            logger.error(f"Failed to create venv: {e}")
            raise
    
    def get_pip_path(self, skill_name: Optional[str] = None) -> Path:
        """
        Get pip executable path for venv.
        
        Args:
            skill_name: Skill name
            
        Returns:
            Path to pip executable
        """
        venv_path = self.get_venv_path(skill_name)
        
        # venv/bin/pip (Linux/Mac) or venv/Scripts/pip.exe (Windows)
        if os.name == 'nt':
            pip_path = venv_path / "Scripts" / "pip.exe"
        else:
            pip_path = venv_path / "bin" / "pip"
        
        return pip_path
    
    def get_python_path(self, skill_name: Optional[str] = None) -> Path:
        """
        Get Python executable path for venv.
        
        Args:
            skill_name: Skill name
            
        Returns:
            Path to Python executable
        """
        venv_path = self.get_venv_path(skill_name)
        
        # venv/bin/python (Linux/Mac) or venv/Scripts/python.exe (Windows)
        if os.name == 'nt':
            python_path = venv_path / "Scripts" / "python.exe"
        else:
            python_path = venv_path / "bin" / "python"
        
        return python_path
    
    def install_packages(
        self,
        packages: List[str],
        skill_name: Optional[str] = None,
        upgrade: bool = False
    ) -> Dict[str, Any]:
        """
        Install packages in skill venv.
        
        Args:
            packages: List of package names (e.g., ['torch', 'diffusers'])
            skill_name: Skill name
            upgrade: Upgrade packages if already installed
            
        Returns:
            Dict with success status and output
        """
        venv_path = self.get_venv_path(skill_name)
        
        # Ensure venv exists
        if not venv_path.exists():
            self.create_venv(skill_name)
        
        pip_path = self.get_pip_path(skill_name)
        
        if not pip_path.exists():
            raise RuntimeError(f"pip not found in venv: {venv_path}")
        
        # Build pip install command
        cmd = [str(pip_path), "install"]
        
        if upgrade:
            cmd.append("--upgrade")
        
        cmd.extend(packages)
        
        logger.info(f"Installing packages in venv {venv_path}: {packages}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes for large packages
            )
            
            if result.returncode != 0:
                return {
                    "success": False,
                    "error": result.stderr,
                    "output": result.stdout
                }
            
            return {
                "success": True,
                "output": result.stdout,
                "packages": packages
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Installation timeout (exceeded 10 minutes)"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def list_installed_packages(self, skill_name: Optional[str] = None) -> List[str]:
        """
        List installed packages in venv.
        
        Args:
            skill_name: Skill name
            
        Returns:
            List of installed package names
        """
        venv_path = self.get_venv_path(skill_name)
        
        if not venv_path.exists():
            return []
        
        pip_path = self.get_pip_path(skill_name)
        
        if not pip_path.exists():
            return []
        
        try:
            result = subprocess.run(
                [str(pip_path), "list", "--format=json"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                return []
            
            packages = json.loads(result.stdout)
            return [pkg["name"] for pkg in packages]
            
        except Exception as e:
            logger.warning(f"Failed to list packages: {e}")
            return []
    
    def check_package_installed(self, package: str, skill_name: Optional[str] = None) -> bool:
        """
        Check if package is installed in venv.
        
        Args:
            package: Package name
            skill_name: Skill name
            
        Returns:
            True if installed
        """
        installed = self.list_installed_packages(skill_name)
        return package.lower() in [pkg.lower() for pkg in installed]
    
    def execute_in_venv(
        self,
        command: List[str],
        skill_name: Optional[str] = None,
        cwd: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute command in venv environment.
        
        Args:
            command: Command to execute
            skill_name: Skill name
            cwd: Working directory
            
        Returns:
            Dict with success status and output
        """
        venv_path = self.get_venv_path(skill_name)
        
        if not venv_path.exists():
            return {
                "success": False,
                "error": f"Venv not found: {venv_path}"
            }
        
        python_path = self.get_python_path(skill_name)
        
        # Execute with venv Python
        env = os.environ.copy()
        venv_bin = venv_path / ("Scripts" if os.name == 'nt' else "bin")
        env["PATH"] = str(venv_bin) + os.pathsep + env.get("PATH", "")
        
        try:
            result = subprocess.run(
                [str(python_path)] + command,
                capture_output=True,
                text=True,
                env=env,
                cwd=cwd,
                timeout=300
            )
            
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr if result.returncode != 0 else None,
                "returncode": result.returncode
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


# Singleton instance
_venv_manager_instance: Optional[SkillVenvManager] = None


def get_venv_manager(venv_base_dir: Optional[str] = None, shared_venv: bool = True) -> SkillVenvManager:
    """Get singleton venv manager instance."""
    global _venv_manager_instance
    if _venv_manager_instance is None:
        _venv_manager_instance = SkillVenvManager(venv_base_dir, shared_venv)
    return _venv_manager_instance
