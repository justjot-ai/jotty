"""
Workspace Checkpoint — git-based save/restore (Cline checkpoint pattern).

KISS: uses git stash or shadow commits to snapshot workspace state.
No custom storage, no file copying. Git does the heavy lifting.

Usage:
    wc = WorkspaceCheckpoint("/path/to/project")
    cp_id = wc.save("before refactoring")
    # ... agent makes changes ...
    diff = wc.diff(cp_id)           # See what changed
    wc.restore(cp_id)               # Rollback everything
    wc.list_checkpoints()           # See all checkpoints
"""

import subprocess
import logging
import time
from typing import Optional, List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class WorkspaceCheckpoint:
    """Git-based workspace snapshots for agent rollback."""

    # Branch prefix for checkpoint refs (won't clutter normal branches)
    _REF_PREFIX = "refs/jotty-checkpoints/"

    def __init__(self, workspace_dir: str = "."):
        self.cwd = str(Path(workspace_dir).resolve())
        self._git_available = self._check_git()

    def _check_git(self) -> bool:
        """Check if workspace is a git repo."""
        try:
            r = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=self.cwd, capture_output=True, text=True, timeout=5,
            )
            return r.returncode == 0
        except Exception:
            return False

    def _run(self, *args: str, check: bool = True) -> str:
        """Run git command and return stdout."""
        r = subprocess.run(
            ["git"] + list(args),
            cwd=self.cwd, capture_output=True, text=True, timeout=30,
        )
        if check and r.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
        return r.stdout.strip()

    def save(self, label: str = "") -> str:
        """
        Save workspace state as a git checkpoint.

        Creates a temporary commit on a detached ref (doesn't affect branches).
        Returns checkpoint ID (short SHA).
        """
        if not self._git_available:
            raise RuntimeError("Not a git repository")

        # Stage everything (including untracked)
        self._run("add", "-A")

        # Create a tree object from the index
        tree = self._run("write-tree")

        # Get current HEAD (or empty tree if no commits)
        try:
            parent = self._run("rev-parse", "HEAD")
            commit = self._run(
                "commit-tree", tree, "-p", parent,
                "-m", f"jotty-checkpoint: {label or 'auto'} [{time.time():.0f}]"
            )
        except RuntimeError:
            commit = self._run(
                "commit-tree", tree,
                "-m", f"jotty-checkpoint: {label or 'auto'} [{time.time():.0f}]"
            )

        # Store as a named ref
        ref_name = f"{self._REF_PREFIX}{commit[:8]}"
        self._run("update-ref", ref_name, commit)

        # Reset index to not leave staged changes
        self._run("reset", check=False)

        logger.info(f"Workspace checkpoint saved: {commit[:8]} ({label})")
        return commit[:8]

    def restore(self, checkpoint_id: str) -> bool:
        """Restore workspace to a checkpoint state (tracked + untracked files)."""
        if not self._git_available:
            return False

        ref = f"{self._REF_PREFIX}{checkpoint_id}"
        try:
            full_sha = self._run("rev-parse", ref)
            # Checkout the tree from the checkpoint (restores tracked files)
            self._run("read-tree", "--reset", "-u", full_sha)
            # Remove untracked files that didn't exist at checkpoint time
            self._run("clean", "-fd", check=False)
            logger.info(f"Workspace restored to checkpoint: {checkpoint_id}")
            return True
        except RuntimeError as e:
            logger.error(f"Restore failed: {e}")
            return False

    def diff(self, checkpoint_id: str) -> str:
        """Show diff between current workspace and checkpoint."""
        if not self._git_available:
            return ""

        ref = f"{self._REF_PREFIX}{checkpoint_id}"
        try:
            return self._run("diff", ref, "--stat", check=False)
        except Exception:
            return ""

    def diff_full(self, checkpoint_id: str) -> str:
        """Full diff (not just stat)."""
        if not self._git_available:
            return ""
        ref = f"{self._REF_PREFIX}{checkpoint_id}"
        try:
            return self._run("diff", ref, check=False)
        except Exception:
            return ""

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all saved checkpoints."""
        if not self._git_available:
            return []

        try:
            refs = self._run(
                "for-each-ref", "--format=%(refname:short) %(objectname:short) %(subject)",
                self._REF_PREFIX, check=False,
            )
            checkpoints = []
            for line in refs.strip().split("\n"):
                if not line.strip():
                    continue
                parts = line.split(" ", 2)
                if len(parts) >= 2:
                    checkpoints.append({
                        'ref': parts[0],
                        'sha': parts[1],
                        'message': parts[2] if len(parts) > 2 else '',
                    })
            return checkpoints
        except Exception:
            return []

    def cleanup(self, keep_latest: int = 5):
        """Remove old checkpoints, keeping the N most recent."""
        cps = self.list_checkpoints()
        if len(cps) <= keep_latest:
            return
        for cp in cps[keep_latest:]:
            try:
                self._run("update-ref", "-d", f"{self._REF_PREFIX}{cp['sha']}", check=False)
            except Exception:
                pass
