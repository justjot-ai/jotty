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
    usage = "/git [status|log|diff|branch|commit <message>]"
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

    def get_completions(self, partial: str) -> list:
        """Get subcommand completions."""
        subcommands = ["status", "log", "diff", "branch", "commit"]
        return [s for s in subcommands if s.startswith(partial)]
