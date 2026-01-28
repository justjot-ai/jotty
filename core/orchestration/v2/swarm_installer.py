"""
SwarmInstaller - Autonomous Dependency Installation

Automatically installs dependencies (pip, npm, etc.).
Follows DRY: Reuses existing package managers and skill registry.
"""
import logging
import subprocess
import sys
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class InstallationResult:
    """Result of installation attempt."""
    package: str
    success: bool
    method: str  # "pip", "npm", "skill", etc.
    version: Optional[str] = None
    error: Optional[str] = None


class SwarmInstaller:
    """
    Autonomous installer for dependencies and tools.
    
    DRY Principle: Reuses existing package managers and skill registry.
    """
    
    def __init__(self, config=None):
        """
        Initialize SwarmInstaller.
        
        Args:
            config: Optional JottyConfig
        """
        self.config = config
        self._skills_registry = None
        self._installed_packages: Dict[str, InstallationResult] = {}
    
    def _init_dependencies(self):
        """Lazy load dependencies (DRY: avoid circular imports)."""
        if self._skills_registry is None:
            from ...registry.skills_registry import get_skills_registry
            self._skills_registry = get_skills_registry()
            self._skills_registry.init()
    
    def _is_builtin_python_module(self, package: str) -> bool:
        """Check if package is a built-in Python module."""
        import sys
        builtin_modules = {
            'multiprocessing', 'threading', 'asyncio', 'json', 'os', 'sys',
            'datetime', 'time', 'random', 'math', 'collections', 'itertools',
            'functools', 'operator', 'copy', 'pickle', 'hashlib', 'base64',
            'urllib', 'http', 'socket', 'ssl', 'email', 'csv', 'xml', 'html',
            'sqlite3', 'dbm', 'zlib', 'gzip', 'bz2', 'lzma', 'tarfile', 'zipfile',
            'shutil', 'glob', 'fnmatch', 'pathlib', 'tempfile', 'io', 'codecs',
            'unicodedata', 'string', 're', 'difflib', 'textwrap', 'readline',
            'rlcompleter', 'struct', 'codecs', 'types', 'copyreg', 'pprint',
            'reprlib', 'enum', 'numbers', 'cmath', 'decimal', 'fractions', 'statistics',
            'array', 'weakref', 'gc', 'inspect', 'site', 'fpectl', 'atexit',
            'traceback', 'future_builtins', 'builtins', '__builtin__', 'warnings',
            'contextlib', 'abc', 'atexit', 'traceback', 'gc', 'inspect', 'site',
            'fpectl', 'pydoc', 'doctest', 'unittest', 'test', 'lib2to3', 'distutils',
            'ensurepip', 'venv', 'zipapp', 'faulthandler', 'pdb', 'profile',
            'pstats', 'timeit', 'trace', 'tracemalloc', 'curses', 'platform',
            'errno', 'ctypes', 'msilib', 'winreg', 'winsound', 'posix', 'pwd',
            'spwd', 'grp', 'crypt', 'termios', 'tty', 'pty', 'fcntl', 'pipes',
            'nis', 'syslog', 'sysv_ipc', 'mmap', 'select', 'selectors', 'asyncio',
            'signal', 'subprocess', 'sched', 'queue', 'threading', 'multiprocessing',
            'concurrent', 'multiprocessing', 'dummy_threading', '_thread', '_dummy_thread',
            'queue', 'queue', 'queue', 'queue', 'queue', 'queue', 'queue', 'queue'
        }
        return package.lower() in builtin_modules
    
    def _normalize_npm_package_name(self, package: str) -> str:
        """Normalize npm package name (lowercase, fix common issues)."""
        # npm package names must be lowercase
        normalized = package.lower()
        
        # Common fixes
        fixes = {
            'mongoose': 'mongoose',  # Mongoose -> mongoose
            'sequelize': 'sequelize',  # Already correct
        }
        
        return fixes.get(normalized, normalized)
    
    async def install(
        self,
        package: str,
        package_type: Optional[str] = None
    ) -> InstallationResult:
        """
        Install a package or tool automatically.
        
        Tries multiple methods: skill registry → pip → npm.
        Skips built-in Python modules automatically.
        
        Args:
            package: Package name (e.g., "praw", "notion-client", "web-search")
            package_type: Optional type hint ("pip", "npm", "skill")
            
        Returns:
            InstallationResult with success status, method used, version
            
        Example:
            result = await installer.install("praw")
            if result.success:
                print(f"Installed {result.package} via {result.method}")
        """
        self._init_dependencies()
        
        # Check if already installed
        if package in self._installed_packages:
            logger.info(f"✅ {package} already installed")
            return self._installed_packages[package]
        
        # Skip built-in Python modules
        if self._is_builtin_python_module(package):
            logger.info(f"ℹ️  Skipping built-in Python module: {package}")
            return InstallationResult(
                package=package,
                success=True,
                method="builtin",
                version="built-in"
            )
        
        logger.info(f"📦 SwarmInstaller: Installing '{package}'")
        
        # Try skill registry first (DRY: reuse existing skills)
        if package_type == "skill" or not package_type:
            skill_result = await self._try_install_skill(package)
            if skill_result.success:
                self._installed_packages[package] = skill_result
                return skill_result
        
        # Try pip (Python packages)
        if package_type in (None, "pip", "python"):
            pip_result = await self._install_with_pip(package)
            if pip_result.success:
                self._installed_packages[package] = pip_result
                return pip_result
        
        # Try npm (Node.js packages)
        if package_type in (None, "npm", "node"):
            npm_result = await self._install_with_npm(package)
            if npm_result.success:
                self._installed_packages[package] = npm_result
                return npm_result
        
        # If all methods failed
        error_msg = f"Failed to install {package} with any method"
        logger.error(f"❌ {error_msg}")
        result = InstallationResult(
            package=package,
            success=False,
            method="unknown",
            error=error_msg
        )
        self._installed_packages[package] = result
        return result
    
    async def _try_install_skill(self, skill_name: str) -> InstallationResult:
        """Try to install from skill registry (DRY: reuse existing skills)."""
        try:
            skill = self._skills_registry.get_skill(skill_name)
            if skill:
                logger.info(f"✅ Found skill '{skill_name}' in registry")
                return InstallationResult(
                    package=skill_name,
                    success=True,
                    method="skill",
                    version=getattr(skill, 'version', None)
                )
        except Exception as e:
            logger.debug(f"Skill registry check failed: {e}")
        
        return InstallationResult(
            package=skill_name,
            success=False,
            method="skill",
            error="Skill not found in registry"
        )
    
    async def _install_with_pip(self, package: str) -> InstallationResult:
        """Install Python package with pip."""
        try:
            logger.info(f"📦 Installing {package} with pip...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                # Try to get version
                version_result = subprocess.run(
                    [sys.executable, "-m", "pip", "show", package],
                    capture_output=True,
                    text=True
                )
                version = None
                if version_result.returncode == 0:
                    for line in version_result.stdout.split('\n'):
                        if line.startswith('Version:'):
                            version = line.split(':', 1)[1].strip()
                            break
                
                logger.info(f"✅ Installed {package} (version: {version or 'unknown'})")
                return InstallationResult(
                    package=package,
                    success=True,
                    method="pip",
                    version=version
                )
            else:
                error = result.stderr or result.stdout
                logger.error(f"❌ Pip installation failed: {error}")
                return InstallationResult(
                    package=package,
                    success=False,
                    method="pip",
                    error=error
                )
        except subprocess.TimeoutExpired:
            return InstallationResult(
                package=package,
                success=False,
                method="pip",
                error="Installation timeout"
            )
        except Exception as e:
            logger.error(f"❌ Pip installation error: {e}")
            return InstallationResult(
                package=package,
                success=False,
                method="pip",
                error=str(e)
            )
    
    async def _install_with_npm(self, package: str) -> InstallationResult:
        """Install Node.js package with npm."""
        try:
            # Normalize npm package name (must be lowercase)
            normalized_package = self._normalize_npm_package_name(package)
            if normalized_package != package:
                logger.info(f"📦 Normalizing npm package name: {package} -> {normalized_package}")
                package = normalized_package
            
            logger.info(f"📦 Installing {package} with npm...")
            result = subprocess.run(
                ["npm", "install", package],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                logger.info(f"✅ Installed {package} with npm")
                return InstallationResult(
                    package=package,
                    success=True,
                    method="npm"
                )
            else:
                error = result.stderr or result.stdout
                logger.error(f"❌ NPM installation failed: {error}")
                return InstallationResult(
                    package=package,
                    success=False,
                    method="npm",
                    error=error
                )
        except FileNotFoundError:
            return InstallationResult(
                package=package,
                success=False,
                method="npm",
                error="npm not found"
            )
        except subprocess.TimeoutExpired:
            return InstallationResult(
                package=package,
                success=False,
                method="npm",
                error="Installation timeout"
            )
        except Exception as e:
            logger.error(f"❌ NPM installation error: {e}")
            return InstallationResult(
                package=package,
                success=False,
                method="npm",
                error=str(e)
            )
    
    def is_installed(self, package: str) -> bool:
        """Check if package is already installed."""
        if package in self._installed_packages:
            return self._installed_packages[package].success
        
        # Check pip
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "show", package],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return True
        except Exception:
            pass
        
        # Check skill registry
        try:
            self._init_dependencies()
            skill = self._skills_registry.get_skill(package)
            if skill:
                return True
        except Exception:
            pass
        
        return False
