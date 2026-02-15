"""
Codebase Mixin - File discovery, diffing, git, and test utilities.
"""

import logging
import os
import re
import subprocess
from typing import Any, Dict, List, Optional

from .types import CodeLanguage
from .utils import _progress

logger = logging.getLogger(__name__)


class CodebaseMixin:
    def _discover_files(self, codebase_path: str, extensions: List[str] = None) -> Dict[str, str]:
        """Auto-discover source files from codebase path.

        Args:
            codebase_path: Root directory to scan
            extensions: File extensions to include (default: language-appropriate)

        Returns:
            Dict of {filepath: content} for discovered files
        """
        import fnmatch
        import os

        if extensions is None:
            # Determine extensions based on configured language
            lang_extensions = {
                CodeLanguage.PYTHON: [".py"],
                CodeLanguage.JAVASCRIPT: [".js", ".jsx", ".ts", ".tsx"],
                CodeLanguage.TYPESCRIPT: [".ts", ".tsx", ".js", ".jsx"],
                CodeLanguage.JAVA: [".java"],
                CodeLanguage.GO: [".go"],
                CodeLanguage.RUST: [".rs"],
                CodeLanguage.CPP: [".cpp", ".hpp", ".h", ".cc"],
            }
            extensions = lang_extensions.get(self.config.language, [".py"])

        # Directories to skip
        skip_dirs = {
            ".git",
            "__pycache__",
            "node_modules",
            ".venv",
            "venv",
            "env",
            ".env",
            "dist",
            "build",
            ".pytest_cache",
            ".mypy_cache",
        }

        discovered = {}
        codebase_path = os.path.abspath(codebase_path)

        for root, dirs, files in os.walk(codebase_path):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in skip_dirs]

            for filename in files:
                if any(filename.endswith(ext) for ext in extensions):
                    filepath = os.path.join(root, filename)
                    rel_path = os.path.relpath(filepath, codebase_path)
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            discovered[rel_path] = f.read()
                    except (IOError, UnicodeDecodeError):
                        continue

        return discovered

    def _generate_unified_diff(self, original: str, modified: str, filepath: str) -> str:
        """Generate unified diff between original and modified code.

        Args:
            original: Original file content
            modified: Modified file content
            filepath: Path to the file (for diff header)

        Returns:
            Unified diff string
        """
        import difflib

        original_lines = original.splitlines(keepends=True)
        modified_lines = modified.splitlines(keepends=True)

        diff = difflib.unified_diff(
            original_lines,
            modified_lines,
            fromfile=f"a/{filepath}",
            tofile=f"b/{filepath}",
            lineterm="",
        )
        return "".join(diff)

    def _analyze_import_graph(self, files: Dict[str, str]) -> Dict[str, List[str]]:
        """Analyze import dependencies between files.

        Args:
            files: Dict of {filepath: content}

        Returns:
            Dict of {filepath: [list of files it imports]}
        """
        import os
        import re

        graph = {}
        file_modules = {}

        # Build module name mapping
        for filepath in files.keys():
            # Convert filepath to module name (e.g., "src/utils/helpers.py" -> "src.utils.helpers")
            module = filepath.replace("/", ".").replace("\\", ".")
            if module.endswith(".py"):
                module = module[:-3]
            file_modules[module] = filepath
            # Also map just the filename
            basename = os.path.basename(filepath)
            if basename.endswith(".py"):
                file_modules[basename[:-3]] = filepath

        # Python import patterns
        import_pattern = re.compile(
            r"^(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))", re.MULTILINE
        )

        for filepath, content in files.items():
            imports = []
            for match in import_pattern.finditer(content):
                module = match.group(1) or match.group(2)
                if module:
                    # Check if this module maps to a local file
                    parts = module.split(".")
                    for i in range(len(parts), 0, -1):
                        partial = ".".join(parts[:i])
                        if partial in file_modules:
                            target = file_modules[partial]
                            if target != filepath:
                                imports.append(target)
                            break
            graph[filepath] = imports

        return graph

    def _get_edit_order(self, files: Dict[str, str], affected_files: List[str]) -> List[str]:
        """Determine optimal edit order based on dependency graph.

        Files that are imported by others should be edited first.

        Args:
            files: All files in codebase
            affected_files: Files that will be edited

        Returns:
            Sorted list of affected files (dependency order)
        """
        graph = self._analyze_import_graph(files)

        # Build reverse graph (who depends on me)
        reverse_graph = {fp: [] for fp in affected_files}
        for fp, deps in graph.items():
            if fp in affected_files:
                for dep in deps:
                    if dep in reverse_graph:
                        reverse_graph[dep].append(fp)

        # Topological sort (Kahn's algorithm)
        in_degree = {fp: 0 for fp in affected_files}
        for fp in affected_files:
            for dep in graph.get(fp, []):
                if dep in in_degree:
                    in_degree[fp] += 1

        queue = [fp for fp, deg in in_degree.items() if deg == 0]
        result = []

        while queue:
            fp = queue.pop(0)
            result.append(fp)
            for dependent in reverse_graph.get(fp, []):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        # Add any remaining (cycles) at the end
        result.extend([fp for fp in affected_files if fp not in result])
        return result

    def _is_test_file(self, filepath: str) -> bool:
        """Check if a file is a test file.

        Args:
            filepath: Path to check

        Returns:
            True if it appears to be a test file
        """
        import os

        # Normalize path separators
        normalized = filepath.replace("\\", "/")
        filename = os.path.basename(filepath)

        # Common test file patterns
        return (
            filename.startswith("test_")
            or filename.endswith("_test.py")
            or filename.endswith(".test.py")
            or filename.endswith(".spec.py")
            or (filename.startswith("test") and filename.endswith(".py"))
            or "/tests/" in normalized
            or normalized.startswith("tests/")
            or "/test/" in normalized
            or normalized.startswith("test/")
        )

    def _filter_test_files(self, files: Dict[str, str], preserve: bool = True) -> tuple:
        """Separate test files from source files.

        Args:
            files: All files
            preserve: If True, test files are returned separately

        Returns:
            (source_files, test_files) dicts
        """
        if not preserve:
            return files, {}

        source_files = {}
        test_files = {}

        for filepath, content in files.items():
            if self._is_test_file(filepath):
                test_files[filepath] = content
            else:
                source_files[filepath] = content

        return source_files, test_files

    async def _git_prepare_branch(self, codebase_path: str, requirements: str) -> Optional[str]:
        """Create a new git branch for the edit.

        Args:
            codebase_path: Path to git repository
            requirements: Requirements string (used to generate branch name)

        Returns:
            Branch name if created, None if git not available
        """
        import os
        import re
        import subprocess
        from datetime import datetime

        if not os.path.isdir(os.path.join(codebase_path, ".git")):
            return None

        try:
            # Generate branch name from requirements
            slug = re.sub(r"[^a-z0-9]+", "-", requirements.lower()[:40]).strip("-")
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            branch_name = f"{self.config.git_branch_prefix}/{slug}-{timestamp}"

            # Create and checkout branch
            subprocess.run(
                ["git", "checkout", "-b", branch_name],
                cwd=codebase_path,
                capture_output=True,
                check=True,
            )

            _progress("Git", "Integration", f"Created branch: {branch_name}")
            return branch_name

        except subprocess.CalledProcessError:
            return None
        except FileNotFoundError:
            return None

    async def _git_commit_changes(
        self, codebase_path: str, files: Dict[str, str], requirements: str
    ) -> bool:
        """Commit edited files to git.

        Args:
            codebase_path: Path to git repository
            files: Dict of {filepath: content} that were modified
            requirements: Requirements string (used for commit message)

        Returns:
            True if commit successful
        """
        import os
        import subprocess

        if not os.path.isdir(os.path.join(codebase_path, ".git")):
            return False

        try:
            # Stage all modified files
            for filepath in files.keys():
                full_path = os.path.join(codebase_path, filepath)
                subprocess.run(
                    ["git", "add", full_path], cwd=codebase_path, capture_output=True, check=True
                )

            # Create commit message
            commit_msg = f"Jotty edit: {requirements[:100]}"
            if len(requirements) > 100:
                commit_msg += "..."

            # Commit
            subprocess.run(
                ["git", "commit", "-m", commit_msg],
                cwd=codebase_path,
                capture_output=True,
                check=True,
            )

            _progress("Git", "Integration", f"Committed {len(files)} file(s)")
            return True

        except subprocess.CalledProcessError:
            return False
        except FileNotFoundError:
            return False

    def _detect_test_command(self, codebase_path: str) -> Optional[str]:
        """Auto-detect the test command for a codebase.

        Args:
            codebase_path: Root path of the codebase

        Returns:
            Test command string or None if not detected
        """
        import os

        # Check for common test configuration files (use python3 -m for reliability)
        checks = [
            ("pytest.ini", "python3 -m pytest"),
            ("setup.cfg", "python3 -m pytest"),
            ("pyproject.toml", "python3 -m pytest"),
            ("tox.ini", "tox"),
            ("Makefile", "make test"),
            ("package.json", "npm test"),
            ("Cargo.toml", "cargo test"),
            ("go.mod", "go test ./..."),
        ]

        for config_file, cmd in checks:
            if os.path.exists(os.path.join(codebase_path, config_file)):
                return cmd

        # Check for test directories
        if os.path.isdir(os.path.join(codebase_path, "tests")):
            return "python3 -m pytest tests/ -x"
        if os.path.isdir(os.path.join(codebase_path, "test")):
            return "python3 -m pytest test/ -x"

        # Look for test files
        for f in os.listdir(codebase_path):
            if f.startswith("test_") and f.endswith(".py"):
                return "python3 -m pytest -x"

        # Default to pytest for Python
        if self.config.language == CodeLanguage.PYTHON:
            return "python3 -m pytest -x"

        return None

    async def _run_tests(self, codebase_path: str, test_command: str = None) -> Dict[str, Any]:
        """Run tests in the codebase and return results.

        Args:
            codebase_path: Root path of the codebase
            test_command: Test command to run (auto-detect if None)

        Returns:
            Dict with 'success', 'output', 'passed', 'failed', 'errors'
        """
        import subprocess

        cmd = test_command or self.config.test_command or self._detect_test_command(codebase_path)
        if not cmd:
            return {
                "success": False,
                "output": "No test command detected",
                "passed": 0,
                "failed": 0,
                "errors": [],
            }

        _progress("Test", "Runner", f"Running: {cmd}")

        try:
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=codebase_path,
                capture_output=True,
                text=True,
                timeout=self.config.test_timeout,
            )

            output = result.stdout + "\n" + result.stderr
            success = result.returncode == 0

            # Parse test output for metrics
            import re

            passed = output.lower().count(" passed")
            failed = output.lower().count(" failed")
            errors = []

            # Also count collection/import errors (pytest shows these differently)
            collection_errors = len(
                re.findall(r"ImportError|ModuleNotFoundError|CollectionError", output)
            )
            if collection_errors > 0:
                failed += collection_errors

            # Extract error messages for feedback - focus on assertion details
            if not success:
                lines = output.split("\n")
                in_error = False
                current_error = []

                # First pass: find the key assertion error
                assertion_context = []
                for i, line in enumerate(lines):
                    if "AssertionError" in line or "assertEqual" in line or "assertIs" in line:
                        # Get context around assertion
                        start = max(0, i - 3)
                        end = min(len(lines), i + 5)
                        assertion_context.extend(lines[start:end])

                if assertion_context:
                    errors.append("=== KEY ASSERTION ===\n" + "\n".join(assertion_context))

                # Second pass: collect other errors
                for line in lines:
                    # Handle import errors and collection failures
                    if any(
                        x in line
                        for x in [
                            "FAILED",
                            "ERROR",
                            "AssertionError",
                            "ImportError",
                            "ModuleNotFoundError",
                            "CollectionError",
                            "no tests ran",
                        ]
                    ):
                        in_error = True
                    if in_error:
                        current_error.append(line)
                        if len(current_error) > 20:  # Limit error context
                            errors.append("\n".join(current_error))
                            current_error = []
                            in_error = False
                if current_error:
                    errors.append("\n".join(current_error))

                # If no specific failures found but tests didn't pass
                if failed == 0 and not success:
                    failed = 1  # At least one failure occurred
                    if not errors:
                        errors = [output[-2000:]]  # Include raw output for debugging

            status = "PASS" if success else f"FAIL ({failed} failure{'s' if failed != 1 else ''})"
            _progress("Test", "Runner", status)

            return {
                "success": success,
                "output": output[-5000:],  # Limit output size
                "passed": passed,
                "failed": failed,
                "errors": errors[:5],  # Limit to 5 errors
                "return_code": result.returncode,
            }

        except subprocess.TimeoutExpired:
            _progress("Test", "Runner", "TIMEOUT")
            return {
                "success": False,
                "output": "Test timeout exceeded",
                "passed": 0,
                "failed": 0,
                "errors": ["Timeout"],
            }
        except Exception as e:
            _progress("Test", "Runner", f"ERROR: {e}")
            return {
                "success": False,
                "output": str(e),
                "passed": 0,
                "failed": 0,
                "errors": [str(e)],
            }
