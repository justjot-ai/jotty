"""
Git Command
===========

Git integration for repository operations.
"""

import subprocess
from typing import TYPE_CHECKING
from pathlib import Path
from .base import BaseCommand, CommandResult, ParsedArgs

if TYPE_CHECKING:
    from ..app import JottyCLI


class GitCommand(BaseCommand):
    """Git integration."""

    name = "git"
    aliases = []
    description = "Git integration for repository operations"
    usage = "/git [status|log|diff|branch|commit|push|pull|stash|checkout]"
    category = "tools"

    async def execute(self, args: ParsedArgs, cli: "JottyCLI") -> CommandResult:
        """Execute git command."""
        if not args.positional:
            return await self._git_status(cli)

        subcommand = args.positional[0]

        if subcommand == "status":
            return await self._git_status(cli)
        elif subcommand == "log":
            limit = int(args.positional[1]) if len(args.positional) > 1 else 10
            return await self._git_log(limit, cli)
        elif subcommand == "diff":
            return await self._git_diff(cli)
        elif subcommand == "branch":
            return await self._git_branch(cli)
        elif subcommand == "commit":
            if len(args.positional) > 1:
                message = " ".join(args.positional[1:])
            else:
                message = args.raw.replace("commit", "").strip()
            return await self._git_commit(message, cli)
        elif subcommand == "push":
            force = args.flags.get("force", False) or "-f" in args.positional
            return await self._git_push(cli, force)
        elif subcommand == "pull":
            rebase = args.flags.get("rebase", False)
            return await self._git_pull(cli, rebase)
        elif subcommand == "stash":
            action = args.positional[1] if len(args.positional) > 1 else "push"
            return await self._git_stash(cli, action)
        elif subcommand == "checkout":
            branch = args.positional[1] if len(args.positional) > 1 else None
            return await self._git_checkout(cli, branch, args.flags)
        elif subcommand == "fetch":
            return await self._git_fetch(cli)
        else:
            return await self._git_status(cli)

    def _run_git(self, args: list, cwd: Path = None) -> tuple:
        """Run git command."""
        try:
            result = subprocess.run(
                ["git"] + args,
                capture_output=True,
                text=True,
                cwd=cwd,
                timeout=30,
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Command timed out"
        except FileNotFoundError:
            return False, "", "Git not found"

    async def _git_status(self, cli: "JottyCLI") -> CommandResult:
        """Show git status."""
        success, stdout, stderr = self._run_git(["status", "--porcelain"])

        if not success:
            if "not a git repository" in stderr.lower():
                cli.renderer.warning("Not in a git repository")
            else:
                cli.renderer.error(f"Git error: {stderr}")
            return CommandResult.fail(stderr)

        if not stdout.strip():
            cli.renderer.success("Working directory clean")
            return CommandResult.ok(data={"clean": True})

        # Parse status
        lines = stdout.strip().split("\n")
        changes = {
            "modified": [],
            "added": [],
            "deleted": [],
            "untracked": [],
        }

        for line in lines:
            if not line:
                continue
            status = line[:2]
            file = line[3:]

            if "M" in status:
                changes["modified"].append(file)
            elif "A" in status:
                changes["added"].append(file)
            elif "D" in status:
                changes["deleted"].append(file)
            elif "?" in status:
                changes["untracked"].append(file)

        # Display
        output_lines = []
        for change_type, files in changes.items():
            if files:
                output_lines.append(f"{change_type.capitalize()}: {len(files)}")
                for f in files[:5]:
                    output_lines.append(f"  {f}")
                if len(files) > 5:
                    output_lines.append(f"  ... and {len(files) - 5} more")

        cli.renderer.panel(
            "\n".join(output_lines),
            title="Git Status",
            style="cyan"
        )

        return CommandResult.ok(data=changes)

    async def _git_log(self, limit: int, cli: "JottyCLI") -> CommandResult:
        """Show git log."""
        success, stdout, stderr = self._run_git([
            "log", f"-{limit}",
            "--oneline", "--decorate"
        ])

        if not success:
            cli.renderer.error(f"Git error: {stderr}")
            return CommandResult.fail(stderr)

        cli.renderer.panel(
            stdout.strip() or "No commits yet",
            title=f"Git Log (last {limit})",
            style="blue"
        )

        return CommandResult.ok(data={"log": stdout})

    async def _git_diff(self, cli: "JottyCLI") -> CommandResult:
        """Show git diff."""
        success, stdout, stderr = self._run_git(["diff", "--stat"])

        if not success:
            cli.renderer.error(f"Git error: {stderr}")
            return CommandResult.fail(stderr)

        if not stdout.strip():
            cli.renderer.info("No changes")
            return CommandResult.ok(data={"diff": ""})

        cli.renderer.code(stdout, language="diff", title="Git Diff")
        return CommandResult.ok(data={"diff": stdout})

    async def _git_branch(self, cli: "JottyCLI") -> CommandResult:
        """Show git branches."""
        success, stdout, stderr = self._run_git(["branch", "-a"])

        if not success:
            cli.renderer.error(f"Git error: {stderr}")
            return CommandResult.fail(stderr)

        cli.renderer.panel(
            stdout.strip() or "No branches",
            title="Git Branches",
            style="green"
        )

        return CommandResult.ok(data={"branches": stdout})

    async def _git_commit(self, message: str, cli: "JottyCLI") -> CommandResult:
        """Create git commit."""
        if not message:
            cli.renderer.error("Commit message required")
            return CommandResult.fail("Commit message required")

        # Stage all changes
        success, _, stderr = self._run_git(["add", "-A"])
        if not success:
            cli.renderer.error(f"Failed to stage: {stderr}")
            return CommandResult.fail(stderr)

        # Commit
        success, stdout, stderr = self._run_git(["commit", "-m", message])

        if not success:
            if "nothing to commit" in stderr.lower() or "nothing to commit" in stdout.lower():
                cli.renderer.warning("Nothing to commit")
            else:
                cli.renderer.error(f"Commit failed: {stderr}")
            return CommandResult.fail(stderr or "Nothing to commit")

        cli.renderer.success(f"Committed: {message}")
        return CommandResult.ok(data={"message": message})

    async def _git_push(self, cli: "JottyCLI", force: bool = False) -> CommandResult:
        """Push to remote."""
        cli.renderer.info("Pushing to remote...")

        args = ["push"]
        if force:
            args.append("--force")

        # Get current branch
        _, branch, _ = self._run_git(["rev-parse", "--abbrev-ref", "HEAD"])
        branch = branch.strip()

        # Check if upstream is set
        _, upstream, _ = self._run_git(["rev-parse", "--abbrev-ref", "@{u}"])
        if not upstream.strip():
            args.extend(["-u", "origin", branch])

        success, stdout, stderr = self._run_git(args)

        if not success:
            if "no upstream branch" in stderr.lower():
                cli.renderer.info(f"Setting upstream for {branch}...")
                success, stdout, stderr = self._run_git(["push", "-u", "origin", branch])

            if not success:
                cli.renderer.error(f"Push failed: {stderr}")
                return CommandResult.fail(stderr)

        cli.renderer.success(f"Pushed to origin/{branch}")
        if stdout.strip():
            cli.renderer.print(f"[dim]{stdout.strip()}[/dim]")

        return CommandResult.ok(data={"branch": branch})

    async def _git_pull(self, cli: "JottyCLI", rebase: bool = False) -> CommandResult:
        """Pull from remote."""
        cli.renderer.info("Pulling from remote...")

        args = ["pull"]
        if rebase:
            args.append("--rebase")

        success, stdout, stderr = self._run_git(args)

        if not success:
            if "conflict" in stderr.lower() or "conflict" in stdout.lower():
                cli.renderer.error("Merge conflicts detected!")
                cli.renderer.info("Resolve conflicts and run: /git commit")
            else:
                cli.renderer.error(f"Pull failed: {stderr}")
            return CommandResult.fail(stderr)

        if "Already up to date" in stdout or "Already up-to-date" in stdout:
            cli.renderer.success("Already up to date")
        else:
            cli.renderer.success("Pulled successfully")
            if stdout.strip():
                cli.renderer.print(f"[dim]{stdout.strip()[:500]}[/dim]")

        return CommandResult.ok(data={"output": stdout})

    async def _git_stash(self, cli: "JottyCLI", action: str = "push") -> CommandResult:
        """Manage git stash."""
        if action == "push" or action == "save":
            success, stdout, stderr = self._run_git(["stash", "push"])
            if success:
                cli.renderer.success("Changes stashed")
            else:
                cli.renderer.error(f"Stash failed: {stderr}")
                return CommandResult.fail(stderr)

        elif action == "pop":
            success, stdout, stderr = self._run_git(["stash", "pop"])
            if success:
                cli.renderer.success("Stash applied and removed")
            else:
                cli.renderer.error(f"Stash pop failed: {stderr}")
                return CommandResult.fail(stderr)

        elif action == "apply":
            success, stdout, stderr = self._run_git(["stash", "apply"])
            if success:
                cli.renderer.success("Stash applied (kept in stash)")
            else:
                cli.renderer.error(f"Stash apply failed: {stderr}")
                return CommandResult.fail(stderr)

        elif action == "list":
            success, stdout, stderr = self._run_git(["stash", "list"])
            if stdout.strip():
                cli.renderer.panel(stdout.strip(), title="Stash List", style="yellow")
            else:
                cli.renderer.info("No stashes")

        elif action == "drop":
            success, stdout, stderr = self._run_git(["stash", "drop"])
            if success:
                cli.renderer.success("Stash dropped")
            else:
                cli.renderer.error(f"Stash drop failed: {stderr}")
                return CommandResult.fail(stderr)

        elif action == "clear":
            success, stdout, stderr = self._run_git(["stash", "clear"])
            if success:
                cli.renderer.success("All stashes cleared")
            else:
                cli.renderer.error(f"Stash clear failed: {stderr}")
                return CommandResult.fail(stderr)

        else:
            cli.renderer.error(f"Unknown stash action: {action}")
            cli.renderer.info("Available: push, pop, apply, list, drop, clear")
            return CommandResult.fail("Unknown action")

        return CommandResult.ok()

    async def _git_checkout(self, cli: "JottyCLI", branch: str, flags: dict) -> CommandResult:
        """Checkout branch or create new."""
        if not branch:
            cli.renderer.error("Branch name required")
            cli.renderer.info("Usage: /git checkout <branch>")
            cli.renderer.info("       /git checkout -b <new-branch>")
            return CommandResult.fail("Branch required")

        args = ["checkout"]

        # Check for -b flag (create new branch)
        if flags.get("b") or flags.get("new"):
            args.append("-b")

        args.append(branch)

        success, stdout, stderr = self._run_git(args)

        if not success:
            if "did not match any" in stderr:
                cli.renderer.error(f"Branch not found: {branch}")
                cli.renderer.info("Create with: /git checkout -b {branch}")
            else:
                cli.renderer.error(f"Checkout failed: {stderr}")
            return CommandResult.fail(stderr)

        cli.renderer.success(f"Switched to: {branch}")
        return CommandResult.ok(data={"branch": branch})

    async def _git_fetch(self, cli: "JottyCLI") -> CommandResult:
        """Fetch from remote."""
        cli.renderer.info("Fetching from remote...")

        success, stdout, stderr = self._run_git(["fetch", "--all", "--prune"])

        if not success:
            cli.renderer.error(f"Fetch failed: {stderr}")
            return CommandResult.fail(stderr)

        cli.renderer.success("Fetched all remotes")
        if stdout.strip():
            cli.renderer.print(f"[dim]{stdout.strip()}[/dim]")

        return CommandResult.ok()

    def get_completions(self, partial: str) -> list:
        """Get subcommand completions."""
        subcommands = ["status", "log", "diff", "branch", "commit", "push", "pull", "stash", "checkout", "fetch"]
        return [s for s in subcommands if s.startswith(partial)]
