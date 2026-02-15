"""
Edit Mixin - Code editing pipeline with test-driven refinement.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List

import dspy

from . import utils as _coding_utils
from .agents import CodebaseAnalyzerAgent, EditPlannerAgent
from .signatures import TestFailureRefinementSignature
from .types import CodeOutput, CodingResult
from .utils import _progress, _stream_call, _strip_code_fences
from .workspace import WorkspaceManager

logger = logging.getLogger(__name__)


class EditMixin:
    async def _refine_from_test_failure(
        self,
        file_path: str,
        current_code: str,
        requirements: str,
        test_output: str,
        iteration: int,
        previous_attempts: List[str],
    ) -> Dict[str, Any]:
        """Refine code based on test failure feedback.

        Args:
            file_path: Path to the file being fixed
            current_code: Current code content
            requirements: Original requirements
            test_output: Test failure output
            iteration: Current iteration number
            previous_attempts: List of previous attempt summaries

        Returns:
            Dict with 'fixed_code', 'analysis', 'confidence'
        """
        if not hasattr(self, "_refinement_module") or self._refinement_module is None:
            self._refinement_module = dspy.ChainOfThought(TestFailureRefinementSignature)

        _progress(f"Iter {iteration}", "Refiner", f"Analyzing test failure for {file_path}...")

        try:
            result = await _stream_call(
                self._refinement_module,
                f"Iter {iteration}",
                "Refiner",
                current_code=current_code,
                file_path=file_path,
                original_requirements=requirements,
                test_output=test_output[-3000:],  # Limit context
                iteration=iteration,
                previous_attempts="\n".join(previous_attempts[-3:]) or "No previous attempts",
            )

            fixed_code = _strip_code_fences(str(result.fixed_code))
            confidence = str(result.confidence).upper()

            _progress(f"Iter {iteration}", "Refiner", f"Confidence: {confidence}")

            return {
                "fixed_code": fixed_code,
                "analysis": str(result.analysis),
                "strategy": str(result.fix_strategy),
                "confidence": confidence,
            }

        except Exception as e:
            logger.error(f"Refinement failed: {e}")
            return {
                "fixed_code": current_code,
                "analysis": f"Refinement error: {e}",
                "strategy": "",
                "confidence": "LOW",
            }

    async def _test_driven_edit_loop(
        self,
        codebase_path: str,
        edited_files: Dict[str, str],
        original_files: Dict[str, str],
        requirements: str,
        affected_files: List[str],
    ) -> tuple:
        """Run test-driven iteration loop until tests pass or max iterations.

        Args:
            codebase_path: Root path of codebase
            edited_files: Current edited file contents
            original_files: Original file contents (for diff)
            requirements: Original requirements
            affected_files: List of files that were edited

        Returns:
            (final_files, iteration_history)
        """
        import os

        max_iters = self.config.max_edit_iterations
        history = []
        current_files = edited_files.copy()

        _progress("Test", "Loop", f"Starting test-driven iteration (max {max_iters})")

        for iteration in range(1, max_iters + 1):
            # Write current files to disk
            for filepath, content in current_files.items():
                full_path = os.path.join(codebase_path, filepath)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, "w", encoding="utf-8") as f:
                    f.write(content)

            # Run tests
            test_result = await self._run_tests(codebase_path)

            if test_result["success"]:
                _progress("Test", "Loop", f"PASS on iteration {iteration}")
                history.append(
                    {
                        "iteration": iteration,
                        "success": True,
                        "passed": test_result["passed"],
                        "failed": test_result["failed"],
                    }
                )
                return current_files, history

            # Tests failed - refine
            history.append(
                {
                    "iteration": iteration,
                    "success": False,
                    "passed": test_result["passed"],
                    "failed": test_result["failed"],
                    "errors": test_result["errors"][:2],
                }
            )

            _progress(
                "Test",
                "Loop",
                f"Iteration {iteration}/{max_iters}: {test_result['failed']} failures",
            )

            # Refine each affected file
            previous_attempts = [
                f"Iter {h['iteration']}: {h.get('errors', ['?'])[:1]}" for h in history[:-1]
            ]

            for filepath in affected_files:
                if filepath not in current_files:
                    continue

                refine_result = await self._refine_from_test_failure(
                    file_path=filepath,
                    current_code=current_files[filepath],
                    requirements=requirements,
                    test_output=test_result["output"],
                    iteration=iteration,
                    previous_attempts=previous_attempts,
                )

                if refine_result["fixed_code"] != current_files[filepath]:
                    current_files[filepath] = refine_result["fixed_code"]
                    _progress(f"Iter {iteration}", "Refiner", f"Updated {filepath}")

        _progress("Test", "Loop", f"Max iterations ({max_iters}) reached")
        return current_files, history

    async def edit(
        self,
        requirements: str,
        target_files: Dict[str, str] = None,
        progress_callback: Any = None,
        codebase_path: str = None,
    ) -> CodingResult:
        """
        Edit existing code based on requirements.

        This is an alternative to generate() for modifying existing codebases.
        It analyzes existing code, plans surgical edits, and applies them.

        Features:
        - Auto-discovers files from codebase_path if target_files not provided
        - Analyzes import dependencies to determine optimal edit order
        - Preserves test files (unless preserve_tests=False)
        - Generates unified diffs (if output_diffs=True)
        - Git integration: creates branch and commits (if git_integration=True)

        Args:
            requirements: What changes to make
            target_files: Dict of {filepath: content} for files to edit (optional if codebase_path set)
            progress_callback: Optional callable(phase, agent, message)
            codebase_path: Root path of codebase (used for file discovery and git integration)

        Returns:
            CodingResult with edited files (and diffs if output_diffs=True)
        """
        _coding_utils._active_progress_callback = progress_callback
        start_time = datetime.now()

        config = self.config
        codebase_path = codebase_path or config.codebase_path

        # -----------------------------------------------------------------
        # FILE DISCOVERY: Auto-discover files if not provided
        # -----------------------------------------------------------------
        if target_files is None or len(target_files) == 0:
            if codebase_path and config.auto_discover_files:
                _progress("Phase 0", "FileDiscovery", f"Scanning {codebase_path}...")
                target_files = self._discover_files(codebase_path)
                _progress("Phase 0", "FileDiscovery", f"Found {len(target_files)} file(s)")
            else:
                return CodingResult(
                    success=False,
                    swarm_name=config.name,
                    domain=config.domain,
                    output={},
                    execution_time=0,
                    error="No target_files provided and codebase_path not set for auto-discovery",
                )

        # Store original files for diff generation
        original_files = {fp: content for fp, content in target_files.items()}

        # -----------------------------------------------------------------
        # TEST PRESERVATION: Separate test files if configured
        # -----------------------------------------------------------------
        test_files = {}
        if config.preserve_tests:
            target_files, test_files = self._filter_test_files(target_files, preserve=True)
            if test_files:
                _progress(
                    "Phase 0",
                    "TestPreservation",
                    f"Preserved {len(test_files)} test file(s) from editing",
                )

        # -----------------------------------------------------------------
        # GIT INTEGRATION: Create branch if enabled
        # -----------------------------------------------------------------
        git_branch = None
        if config.git_integration and codebase_path:
            git_branch = await self._git_prepare_branch(codebase_path, requirements)

        self._init_agents()

        # Initialize edit-specific agents
        if not hasattr(self, "_codebase_analyzer") or self._codebase_analyzer is None:
            self._codebase_analyzer = CodebaseAnalyzerAgent(
                self._memory, self._context, self._bus, self._agent_context("Architect")
            )
        if not hasattr(self, "_edit_planner") or self._edit_planner is None:
            self._edit_planner = EditPlannerAgent(
                self._memory, self._context, self._bus, self._agent_context("Developer")
            )

        config = self.config
        lang = config.language

        logger.info("=" * 60)
        logger.info("CodingSwarm EDIT MODE | %d file(s)", len(target_files))
        logger.info("=" * 60)

        try:
            # =================================================================
            # PHASE 0: CODEBASE ANALYSIS
            # =================================================================
            _progress("Phase 0", "CodebaseAnalyzer", f"Analyzing {len(target_files)} file(s)...")

            all_code = "\n\n".join(
                f"# FILE: {fp}\n{content}" for fp, content in target_files.items()
            )
            file_paths = list(target_files.keys())

            analysis = await self._codebase_analyzer.analyze(
                existing_code=all_code, file_paths=file_paths, requirements=requirements
            )

            if "error" in analysis:
                return CodingResult(
                    success=False,
                    swarm_name=self.config.name,
                    domain=self.config.domain,
                    output={},
                    execution_time=(datetime.now() - start_time).total_seconds(),
                    error=analysis["error"],
                )

            affected_files = analysis.get("affected_files", file_paths)
            # CRITICAL FIX: If analyzer returns empty, fall back to first few files
            # This ensures test-driven loop can refine even when analyzer misses
            if not affected_files:
                affected_files = file_paths[:3]  # Focus on top 3 most relevant files
            style_conventions = analysis.get("style_conventions", "")
            dependencies = analysis.get("dependencies", "")
            change_scope = analysis.get("change_scope", "moderate")

            _progress(
                "Phase 0",
                "CodebaseAnalyzer",
                f"Change scope: {change_scope}, affecting {len(affected_files)} file(s)",
            )

            # =================================================================
            # DEPENDENCY ANALYSIS: Determine optimal edit order
            # =================================================================
            if config.analyze_dependencies and len(affected_files) > 1:
                _progress("Phase 0.5", "DependencyAnalyzer", "Analyzing import graph...")
                import_graph = self._analyze_import_graph(target_files)
                affected_files = self._get_edit_order(target_files, affected_files)
                _progress(
                    "Phase 0.5",
                    "DependencyAnalyzer",
                    f"Edit order determined: {len(affected_files)} file(s)",
                )

            # =================================================================
            # PHASE 1: EDIT PLANNING (parallel for each affected file)
            # =================================================================
            _progress(
                "Phase 1", "EditPlanner", f"Planning edits for {len(affected_files)} file(s)..."
            )

            edited_files = {}
            edit_summaries = []

            async def _plan_and_apply_edit(file_path: str) -> Any:
                if file_path not in target_files:
                    return file_path, target_files.get(file_path, ""), {"skipped": True}

                existing_code = target_files[file_path]
                _progress("Phase 1", "EditPlanner", f"  Planning: {file_path}...")

                edit_result = await self._edit_planner.plan_edit(
                    existing_code=existing_code,
                    file_path=file_path,
                    requirements=requirements,
                    style_conventions=style_conventions,
                    dependencies=dependencies,
                )

                if "error" in edit_result:
                    return file_path, existing_code, edit_result

                edit_type = edit_result.get("edit_type", "patch")
                edits = edit_result.get("edits", [])

                if edit_type == "rewrite":
                    new_code = _strip_code_fences(edit_result.get("new_code", existing_code))
                    _progress("Phase 1", "EditPlanner", f"  {file_path}: REWRITE")
                else:
                    # Apply patch edits
                    new_code = existing_code
                    for edit in edits:
                        if isinstance(edit, dict) and "old" in edit and "new" in edit:
                            old_str = edit["old"]
                            new_str = edit["new"]
                            if old_str in new_code:
                                new_code = new_code.replace(old_str, new_str, 1)
                                _progress(
                                    "Phase 1",
                                    "EditPlanner",
                                    f"  {file_path}: patched '{old_str[:30]}...'",
                                )

                return file_path, new_code, edit_result

            # Run edit planning in parallel
            edit_tasks = [_plan_and_apply_edit(fp) for fp in affected_files]
            edit_results = await asyncio.gather(*edit_tasks, return_exceptions=True)

            for result in edit_results:
                if isinstance(result, Exception):
                    continue
                file_path, new_code, edit_info = result
                edited_files[file_path] = new_code
                if not edit_info.get("skipped"):
                    edit_summaries.append(
                        {
                            "file": file_path,
                            "type": edit_info.get("edit_type", "unknown"),
                            "num_edits": len(edit_info.get("edits", [])),
                        }
                    )

            # Include unchanged files
            for fp, content in target_files.items():
                if fp not in edited_files:
                    edited_files[fp] = content

            _progress("Phase 1", "EditPlanner", f"Done -- {len(edit_summaries)} file(s) modified")

            # =================================================================
            # PHASE 4.5: VALIDATION
            # =================================================================
            workspace = WorkspaceManager() if getattr(config, "enable_workspace", True) else None
            validation_metadata = {"validated": False, "fix_attempts": 0, "errors_fixed": []}

            if workspace and workspace.available:
                _progress("Phase 4.5", "Validator", "Validating edited code...")
                # Write and syntax check edited files
                for fname, content in edited_files.items():
                    if fname.endswith(".py"):
                        await workspace.write_file(fname, content)
                        check = await workspace.syntax_check(fname)
                        if check.success:
                            _progress("Phase 4.5", "Validator", f"Syntax OK: {fname}")
                            validation_metadata["validated"] = True
                        else:
                            _progress("Phase 4.5", "Validator", f"Syntax ERROR: {fname}")

            # =================================================================
            # TEST-DRIVEN ITERATION (if enabled)
            # =================================================================
            iteration_history = []
            if config.test_driven and codebase_path:
                _progress("Phase 3", "TestLoop", "Starting test-driven refinement...")

                # Merge test files temporarily for testing
                files_for_testing = edited_files.copy()
                files_for_testing.update(test_files)

                # Run test-driven loop
                refined_files, iteration_history = await self._test_driven_edit_loop(
                    codebase_path=codebase_path,
                    edited_files=files_for_testing,
                    original_files=original_files,
                    requirements=requirements,
                    affected_files=affected_files,
                )

                # Extract only source files (not test files) from refined result
                for fp in edited_files.keys():
                    if fp in refined_files:
                        edited_files[fp] = refined_files[fp]

                if iteration_history and iteration_history[-1].get("success"):
                    _progress("Phase 3", "TestLoop", "Tests passing!")
                else:
                    _progress("Phase 3", "TestLoop", "Max iterations reached, tests may still fail")

            # =================================================================
            # MERGE TEST FILES BACK
            # =================================================================
            if test_files:
                edited_files.update(test_files)
                _progress(
                    "Phase 2",
                    "TestPreservation",
                    f"Merged {len(test_files)} preserved test file(s)",
                )

            # =================================================================
            # GENERATE DIFFS (if enabled)
            # =================================================================
            diffs = {}
            if config.output_diffs:
                _progress("Phase 2", "DiffGenerator", "Generating unified diffs...")
                for filepath, new_content in edited_files.items():
                    old_content = original_files.get(filepath, "")
                    if old_content != new_content:
                        diff = self._generate_unified_diff(old_content, new_content, filepath)
                        if diff:
                            diffs[filepath] = diff
                _progress("Phase 2", "DiffGenerator", f"Generated {len(diffs)} diff(s)")

            # =================================================================
            # GIT COMMIT (if enabled)
            # =================================================================
            git_committed = False
            if config.git_integration and codebase_path and git_branch:
                # Write files to disk first
                import os

                modified_files = {
                    fp: content
                    for fp, content in edited_files.items()
                    if original_files.get(fp, "") != content
                }
                if modified_files:
                    for filepath, content in modified_files.items():
                        full_path = os.path.join(codebase_path, filepath)
                        os.makedirs(os.path.dirname(full_path), exist_ok=True)
                        with open(full_path, "w", encoding="utf-8") as f:
                            f.write(content)
                    git_committed = await self._git_commit_changes(
                        codebase_path, modified_files, requirements
                    )

            # =================================================================
            # BUILD RESULT
            # =================================================================
            exec_time = (datetime.now() - start_time).total_seconds()
            loc = sum(code.count("\n") + 1 for code in edited_files.values())

            code_output = CodeOutput(
                files=edited_files,
                main_file=affected_files[0] if affected_files else "",
                entry_point="",
                dependencies=[],
                tests=test_files,
                docs="",
                architecture=analysis.get("architecture_summary", ""),
            )

            result = CodingResult(
                success=True,
                swarm_name=self.config.name,
                domain=self.config.domain,
                output={"files": list(edited_files.keys()), "mode": "edit"},
                execution_time=exec_time,
                code=code_output,
                language=lang.value,
                loc=loc,
            )

            result.metadata["edit_summaries"] = edit_summaries
            result.metadata["change_scope"] = change_scope
            result.metadata["validation"] = validation_metadata
            result.metadata["diffs"] = diffs
            result.metadata["preserved_tests"] = list(test_files.keys()) if test_files else []
            result.metadata["git_branch"] = git_branch
            result.metadata["git_committed"] = git_committed
            result.metadata["test_iterations"] = iteration_history
            result.metadata["tests_passing"] = (
                iteration_history[-1].get("success", False) if iteration_history else None
            )

            logger.info("=" * 56)
            logger.info("EDIT DONE | %d file(s) modified | %d LOC", len(edit_summaries), loc)
            if diffs:
                logger.info(
                    "Diffs generated: %d | Tests preserved: %d", len(diffs), len(test_files)
                )
            if iteration_history:
                tests_status = "PASSING" if iteration_history[-1].get("success") else "FAILING"
                logger.info("Test iterations: %d | Tests: %s", len(iteration_history), tests_status)
            if git_branch:
                logger.info("Git branch: %s | Committed: %s", git_branch, git_committed)
            logger.info("Time: %.1fs", exec_time)
            logger.info("=" * 56)

            return result

        except Exception as e:
            logger.error(f"Edit mode error: {e}")
            import traceback

            traceback.print_exc()
            return CodingResult(
                success=False,
                swarm_name=self.config.name,
                domain=self.config.domain,
                output={},
                execution_time=(datetime.now() - start_time).total_seconds(),
                error=str(e),
            )

        finally:
            _coding_utils._active_progress_callback = None
            if workspace:
                workspace.cleanup()
