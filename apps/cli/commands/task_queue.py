"""
Task Queue Command
==================

Manage coding tasks via Supervisor API.
"""

from typing import TYPE_CHECKING, Any, List

import aiohttp

from .base import BaseCommand, CommandResult, ParsedArgs

if TYPE_CHECKING:
    from ..app import JottyCLI


class TaskCommand(BaseCommand):
    """Task queue management for Supervisor Coder."""

    name = "task"
    aliases = ["t"]
    description = "Manage coding tasks via Supervisor API"
    usage = (
        "/task [create|list|get|start|pause|kill|delete|log|stats|templates|from-template] [args]"
    )
    category = "supervisor"

    SUPERVISOR_URL = "http://localhost:8080"

    SWIMLANES = ["suggested", "backlog", "pending", "in_progress", "completed", "failed"]

    async def execute(self, args: ParsedArgs, cli: "JottyCLI") -> CommandResult:
        """Execute task command."""
        subcommand = args.positional[0] if args.positional else "stats"

        if subcommand == "create" or subcommand == "new" or subcommand == "add":
            return await self._create_task(args, cli)
        elif subcommand == "list":
            return await self._list_tasks(args, cli)
        elif subcommand == "get":
            return await self._get_task(args, cli)
        elif subcommand == "start":
            return await self._start_task(args, cli)
        elif subcommand == "pause":
            return await self._pause_task(args, cli)
        elif subcommand == "kill":
            return await self._kill_task(args, cli)
        elif subcommand == "delete":
            return await self._delete_task(args, cli)
        elif subcommand == "log":
            return await self._get_log(args, cli)
        elif subcommand == "stats":
            return await self._show_stats(cli)
        elif subcommand == "templates":
            return await self._list_templates(cli)
        elif subcommand == "from-template" or subcommand == "template":
            return await self._create_from_template(args, cli)
        elif subcommand == "move":
            return await self._move_task(args, cli)
        else:
            return await self._show_stats(cli)

    async def _api_request(self, method: str, endpoint: str, data: dict = None) -> dict:
        """Make HTTP request to Supervisor API."""
        url = f"{self.SUPERVISOR_URL}{endpoint}"
        async with aiohttp.ClientSession() as session:
            if method == "GET":
                async with session.get(url) as resp:
                    return await resp.json()
            elif method == "POST":
                async with session.post(url, json=data) as resp:
                    return await resp.json()

    async def _generate_task_with_ai(self, brief_idea: str, cli: "JottyCLI") -> dict:
        """Use AI to generate a full task from a brief idea."""
        try:
            prompt = f"""Generate a coding task from this brief idea: "{brief_idea}"

Return a JSON object with:
- title: Clear, actionable task title (imperative form, e.g., "Fix login bug", "Add dark mode")
- description: Detailed description with:
  - What needs to be done
  - Acceptance criteria (as checklist)
  - Any relevant context
- priority: 1-5 (1=critical/security, 2=high/bugs, 3=medium/features, 4=low/refactor, 5=optional/docs)
- category: One of: bugfix, feature, refactor, api, ui, security, performance, testing, docs, devops

Respond ONLY with valid JSON, no markdown."""

            result = await self._api_request(
                "POST", "/api/chat", {"message": prompt, "history": []}
            )

            if result.get("success"):
                response_text = result.get("result", {}).get("response", "")
                # Try to parse JSON from response
                import json
                import re

                # Extract JSON from response (handle markdown code blocks)
                json_match = re.search(r"\{[\s\S]*\}", response_text)
                if json_match:
                    try:
                        return json.loads(json_match.group())
                    except json.JSONDecodeError:
                        pass

            # Fallback: return basic task
            return {
                "title": brief_idea.strip().capitalize(),
                "description": f"Task: {brief_idea}\n\n## Acceptance Criteria\n- [ ] Task completed",
                "priority": 3,
                "category": "feature",
            }

        except Exception as e:
            cli.renderer.warning(f"AI generation failed: {e}, using basic task")
            return {
                "title": brief_idea.strip().capitalize(),
                "description": brief_idea,
                "priority": 3,
                "category": "",
            }

    async def _create_task(self, args: ParsedArgs, cli: "JottyCLI") -> CommandResult:
        """Create a new task.

        Usage:
            /task create "title" ["description"] [--priority=N] [--agent=claude] [--lane=backlog] [--category=X]
            /task create "brief idea" --generate    # AI generates full task
        """
        # Check for --generate flag (AI-assisted creation)
        generate_ai = args.flags.get("generate", args.flags.get("g", False))

        if len(args.positional) < 2:
            # Show helpful usage with examples
            cli.renderer.panel(
                """Create a task with title and description:

  /task create "Fix login bug" "Users can't login with OAuth"

With options:
  /task create "Add dark mode" "Implement theme toggle" --priority=2 --lane=pending

AI-assisted (generates description from brief idea):
  /task create "fix the checkout flow" --generate

Options:
  --priority, -p    Priority 1-5 (1=critical, 5=low) [default: 3]
  --lane, -l        Swimlane: backlog, pending, suggested [default: backlog]
  --agent, -a       Agent: claude, cursor, opencode [default: claude]
  --category        Category tag for filtering
  --generate, -g    Use AI to expand brief idea into full task""",
                title="Task Create Usage",
                style="cyan",
            )
            return CommandResult.fail("Missing title")

        title = args.positional[1]
        description = args.positional[2] if len(args.positional) > 2 else ""
        priority = int(args.flags.get("priority", args.flags.get("p", 3)))
        agent_type = args.flags.get("agent", args.flags.get("a", "claude"))
        status = args.flags.get("lane", args.flags.get("status", args.flags.get("l", "backlog")))
        category = args.flags.get("category", args.flags.get("cat", ""))

        # AI-assisted generation
        if generate_ai:
            cli.renderer.info("Generating task with AI...")
            generated = await self._generate_task_with_ai(title, cli)
            if generated:
                title = generated.get("title", title)
                description = generated.get("description", description)
                priority = generated.get("priority", priority)
                category = generated.get("category", category)
                cli.renderer.success("AI generated task details")

        # Validate swimlane
        if status not in self.SWIMLANES:
            cli.renderer.warning(f"Unknown swimlane '{status}', using 'backlog'")
            status = "backlog"

        try:
            result = await self._api_request(
                "POST",
                "/api/tasks/create",
                {
                    "title": title,
                    "description": description,
                    "priority": priority,
                    "agent_type": agent_type,
                    "status": status,
                    "category": category,
                },
            )

            if result.get("success"):
                task_id = result.get("task_id")
                cli.renderer.success(f"Task created: {task_id}")
                cli.renderer.info(f"  Title: {title}")
                cli.renderer.info(f"  Lane: {status}, Priority: {priority}, Agent: {agent_type}")
                if category:
                    cli.renderer.info(f"  Category: {category}")
                return CommandResult.ok(data=result)
            else:
                cli.renderer.error(f"Failed to create task: {result.get('error')}")
                return CommandResult.fail(result.get("error", "Unknown error"))

        except Exception as e:
            cli.renderer.error(f"API error: {e}")
            return CommandResult.fail(str(e))

    async def _list_templates(self, cli: "JottyCLI") -> CommandResult:
        """List available task templates."""
        try:
            result = await self._api_request("GET", "/api/templates")

            if result.get("success"):
                templates = result.get("templates", [])
                if not templates:
                    cli.renderer.info("No templates available")
                    return CommandResult.ok(data=[])

                lines = []
                for tpl in templates:
                    name = tpl.get("name", "?")
                    title = tpl.get("title", "?")
                    category = tpl.get("category", "general")
                    priority = tpl.get("priority", 3)
                    lines.append(f"  {name:<15} | {category:<12} | P{priority} | {title}")

                header = f"  {'Name':<15} | {'Category':<12} | Pri | Title Pattern"
                cli.renderer.panel(
                    f"{header}\n  {'-' * 60}\n" + "\n".join(lines),
                    title="Task Templates",
                    style="cyan",
                )
                cli.renderer.info('\nUsage: /task from-template <name> "summary" ["description"]')
                return CommandResult.ok(data=templates)
            else:
                cli.renderer.error(f"Failed to list templates: {result.get('error')}")
                return CommandResult.fail(result.get("error", "Unknown error"))

        except Exception as e:
            cli.renderer.error(f"API error: {e}")
            return CommandResult.fail(str(e))

    async def _create_from_template(self, args: ParsedArgs, cli: "JottyCLI") -> CommandResult:
        """Create a task from a template.

        Usage:
            /task from-template <template> "summary" ["description"] [--priority=N] [--agent=claude] [--lane=backlog]
        """
        if len(args.positional) < 3:
            cli.renderer.error(
                'Usage: /task from-template <template> "summary" ["description"] [--lane=backlog]'
            )
            cli.renderer.info("\nAvailable templates:")
            await self._list_templates(cli)
            return CommandResult.fail("Missing template or summary")

        template_name = args.positional[1]
        summary = args.positional[2]
        description = args.positional[3] if len(args.positional) > 3 else ""
        priority = args.flags.get("priority", args.flags.get("p"))
        agent_type = args.flags.get("agent", args.flags.get("a"))
        status = args.flags.get("lane", args.flags.get("status", args.flags.get("l", "backlog")))
        category = args.flags.get("category", args.flags.get("cat"))

        # Validate swimlane
        if status not in self.SWIMLANES:
            cli.renderer.warning(f"Unknown swimlane '{status}', using 'backlog'")
            status = "backlog"

        try:
            request_data = {
                "template": template_name,
                "summary": summary,
                "description": description,
                "status": status,
            }

            # Only include overrides if specified
            if priority:
                request_data["priority"] = int(priority)
            if agent_type:
                request_data["agent_type"] = agent_type
            if category:
                request_data["category"] = category

            result = await self._api_request("POST", "/api/tasks/from-template", request_data)

            if result.get("success"):
                task_id = result.get("task_id")
                title = result.get("title")
                cli.renderer.success(f"Task created from template '{template_name}'")
                cli.renderer.info(f"  ID: {task_id}")
                cli.renderer.info(f"  Title: {title}")
                cli.renderer.info(f"  Lane: {status}")
                return CommandResult.ok(data=result)
            else:
                cli.renderer.error(f"Failed to create task: {result.get('error')}")
                return CommandResult.fail(result.get("error", "Unknown error"))

        except Exception as e:
            cli.renderer.error(f"API error: {e}")
            return CommandResult.fail(str(e))

    async def _move_task(self, args: ParsedArgs, cli: "JottyCLI") -> CommandResult:
        """Move a task to a different swimlane.

        Usage:
            /task move <task_id> <lane>
        """
        if len(args.positional) < 3:
            cli.renderer.error("Usage: /task move <task_id> <lane>")
            cli.renderer.info(f"Available lanes: {', '.join(self.SWIMLANES)}")
            return CommandResult.fail("Missing task_id or lane")

        task_id = args.positional[1]
        new_lane = args.positional[2].lower()

        if new_lane not in self.SWIMLANES:
            cli.renderer.error(f"Unknown swimlane: {new_lane}")
            cli.renderer.info(f"Available lanes: {', '.join(self.SWIMLANES)}")
            return CommandResult.fail("Invalid swimlane")

        try:
            result = await self._api_request(
                "POST", f"/api/tasks/{task_id}/status", {"status": new_lane}
            )

            if result.get("success"):
                cli.renderer.success(f"Task {task_id} moved to '{new_lane}'")
                return CommandResult.ok(data=result)
            else:
                cli.renderer.error(f"Failed to move task: {result.get('error')}")
                return CommandResult.fail(result.get("error", "Unknown error"))

        except Exception as e:
            cli.renderer.error(f"API error: {e}")
            return CommandResult.fail(str(e))

    async def _list_tasks(self, args: ParsedArgs, cli: "JottyCLI") -> CommandResult:
        """List tasks with optional status filter.

        Usage:
            /task list [--status=pending|in_progress|completed|failed|backlog]
            /task list [--lane=pending]  (alias for --status)
        """
        status = args.flags.get("status", args.flags.get("lane", args.flags.get("l")))

        try:
            endpoint = f"/api/tasks?status={status}" if status else "/api/tasks"
            result = await self._api_request("GET", endpoint)

            if result.get("success"):
                tasks = result.get("tasks", [])
                if not tasks:
                    if status:
                        cli.renderer.info(f"No tasks in '{status}' lane")
                    else:
                        cli.renderer.info("No tasks found")
                    return CommandResult.ok(data=[])

                # Build table data
                table_data = []
                for task in tasks:
                    table_data.append(
                        {
                            "ID": task.get("task_id", task.get("id", "?"))[:20],
                            "Title": (task.get("title", "Untitled"))[:40],
                            "Status": task.get("status", "?"),
                            "Priority": task.get("priority", 3),
                            "Agent": task.get("agent_type", "claude"),
                        }
                    )

                # Render table
                self._render_task_table(table_data, cli)
                cli.renderer.info(f"Total: {len(tasks)} tasks")
                return CommandResult.ok(data=tasks)
            else:
                cli.renderer.error(f"Failed to list tasks: {result.get('error')}")
                return CommandResult.fail(result.get("error", "Unknown error"))

        except Exception as e:
            cli.renderer.error(f"API error: {e}")
            return CommandResult.fail(str(e))

    def _render_task_table(self, data: List[dict], cli: "JottyCLI") -> Any:
        """Render task table."""
        if not data:
            return

        # Header
        header = f"{'ID':<22} {'Title':<42} {'Status':<12} {'Pri':<4} {'Agent':<8}"
        cli.renderer.print(f"[bold]{header}[/bold]")
        cli.renderer.print("-" * 90)

        # Rows
        for row in data:
            status = row["Status"]
            status_color = {
                "completed": "green",
                "in_progress": "yellow",
                "pending": "cyan",
                "failed": "red",
                "backlog": "dim",
                "suggested": "dim",
            }.get(status, "white")

            line = f"{row['ID']:<22} {row['Title']:<42} [{status_color}]{status:<12}[/{status_color}] {row['Priority']:<4} {row['Agent']:<8}"
            cli.renderer.print(line)

    async def _get_task(self, args: ParsedArgs, cli: "JottyCLI") -> CommandResult:
        """Get task details."""
        if len(args.positional) < 2:
            cli.renderer.error("Usage: /task get <task_id>")
            return CommandResult.fail("Missing task_id")

        task_id = args.positional[1]

        try:
            result = await self._api_request("GET", "/api/state")
            task_details = result.get("task_details", {})

            if task_id in task_details:
                task = task_details[task_id]
                info = {
                    "Task ID": task.get("task_id", task_id),
                    "Title": task.get("title", "Untitled"),
                    "Description": task.get("description", "")[:200]
                    + ("..." if len(task.get("description", "")) > 200 else ""),
                    "Status": task.get("status", "?"),
                    "Priority": task.get("priority", 3),
                    "Category": task.get("category", "-"),
                    "Agent": task.get("agent_type", "claude"),
                    "Created": task.get("created_at", "?"),
                    "PID": task.get("pid") or "-",
                }
                cli.renderer.tree(info, title=f"Task: {task_id}")
                return CommandResult.ok(data=task)
            else:
                cli.renderer.error(f"Task not found: {task_id}")
                return CommandResult.fail("Task not found")

        except Exception as e:
            cli.renderer.error(f"API error: {e}")
            return CommandResult.fail(str(e))

    async def _start_task(self, args: ParsedArgs, cli: "JottyCLI") -> CommandResult:
        """Start a task (move to pending for spawning)."""
        if len(args.positional) < 2:
            cli.renderer.error("Usage: /task start <task_id>")
            return CommandResult.fail("Missing task_id")

        task_id = args.positional[1]

        try:
            result = await self._api_request(
                "POST", f"/api/tasks/{task_id}/status", {"status": "pending"}
            )

            if result.get("success"):
                cli.renderer.success(f"Task {task_id} moved to pending (will be spawned)")
                return CommandResult.ok(data=result)
            else:
                cli.renderer.error(f"Failed to start task: {result.get('error')}")
                return CommandResult.fail(result.get("error", "Unknown error"))

        except Exception as e:
            cli.renderer.error(f"API error: {e}")
            return CommandResult.fail(str(e))

    async def _pause_task(self, args: ParsedArgs, cli: "JottyCLI") -> CommandResult:
        """Pause a running task."""
        if len(args.positional) < 2:
            cli.renderer.error("Usage: /task pause <task_id>")
            return CommandResult.fail("Missing task_id")

        task_id = args.positional[1]

        try:
            result = await self._api_request("POST", "/api/tasks/pause", {"task_id": task_id})

            if result.get("success"):
                cli.renderer.success(f"Task {task_id} paused")
                return CommandResult.ok(data=result)
            else:
                cli.renderer.error(f"Failed to pause task: {result.get('error')}")
                return CommandResult.fail(result.get("error", "Unknown error"))

        except Exception as e:
            cli.renderer.error(f"API error: {e}")
            return CommandResult.fail(str(e))

    async def _kill_task(self, args: ParsedArgs, cli: "JottyCLI") -> CommandResult:
        """Kill a running task."""
        if len(args.positional) < 2:
            cli.renderer.error("Usage: /task kill <task_id>")
            return CommandResult.fail("Missing task_id")

        task_id = args.positional[1]

        try:
            result = await self._api_request("POST", "/api/tasks/kill", {"task_id": task_id})

            if result.get("success"):
                cli.renderer.success(f"Task {task_id} killed")
                return CommandResult.ok(data=result)
            else:
                cli.renderer.error(f"Failed to kill task: {result.get('error')}")
                return CommandResult.fail(result.get("error", "Unknown error"))

        except Exception as e:
            cli.renderer.error(f"API error: {e}")
            return CommandResult.fail(str(e))

    async def _delete_task(self, args: ParsedArgs, cli: "JottyCLI") -> CommandResult:
        """Delete a task."""
        if len(args.positional) < 2:
            cli.renderer.error("Usage: /task delete <task_id>")
            return CommandResult.fail("Missing task_id")

        task_id = args.positional[1]

        try:
            result = await self._api_request("POST", "/api/tasks/delete", {"task_id": task_id})

            if result.get("success"):
                cli.renderer.success(f"Task {task_id} deleted")
                return CommandResult.ok(data=result)
            else:
                cli.renderer.error(f"Failed to delete task: {result.get('error')}")
                return CommandResult.fail(result.get("error", "Unknown error"))

        except Exception as e:
            cli.renderer.error(f"API error: {e}")
            return CommandResult.fail(str(e))

    async def _get_log(self, args: ParsedArgs, cli: "JottyCLI") -> CommandResult:
        """Get task output log."""
        if len(args.positional) < 2:
            cli.renderer.error("Usage: /task log <task_id> [--limit=N]")
            return CommandResult.fail("Missing task_id")

        task_id = args.positional[1]
        limit = int(args.flags.get("limit", args.flags.get("n", 100)))

        try:
            result = await self._api_request("GET", f"/api/output/{task_id}?limit={limit}")

            if result.get("success"):
                output = result.get("output", "")
                if output:
                    cli.renderer.panel(output, title=f"Log: {task_id}", style="cyan")
                else:
                    cli.renderer.info("No output yet")
                return CommandResult.ok(data={"output": output})
            else:
                cli.renderer.error(f"Failed to get log: {result.get('error')}")
                return CommandResult.fail(result.get("error", "Unknown error"))

        except Exception as e:
            cli.renderer.error(f"API error: {e}")
            return CommandResult.fail(str(e))

    async def _show_stats(self, cli: "JottyCLI") -> CommandResult:
        """Show queue statistics."""
        try:
            result = await self._api_request("GET", "/api/state")

            # Count tasks by status
            stats = {
                "Suggested": len(result.get("suggested_tasks", [])),
                "Backlog": len(result.get("backlog_tasks", [])),
                "Pending": len(result.get("pending_tasks", [])),
                "In Progress": len(result.get("in_progress_tasks", [])),
                "Completed": len(result.get("completed_task_files", [])),
                "Failed": len(result.get("failed_task_files", [])),
            }

            total = sum(stats.values())

            # Display
            lines = []
            for status, count in stats.items():
                bar = "█" * min(count, 20)
                lines.append(f"{status:<12}: {count:>3} {bar}")

            lines.append(f"\n{'Total':<12}: {total:>3}")

            cli.renderer.panel("\n".join(lines), title="Task Queue Statistics", style="magenta")

            # Show quick actions
            cli.renderer.info("\nQuick actions:")
            cli.renderer.info('  /task create "title"          - Create new task')
            cli.renderer.info("  /task templates               - List templates")
            cli.renderer.info("  /task from-template bug-fix   - Create from template")
            cli.renderer.info("  /task list --lane=pending     - List by swimlane")

            return CommandResult.ok(data=stats)

        except aiohttp.ClientConnectorError:
            cli.renderer.error("Cannot connect to Supervisor at localhost:8080")
            cli.renderer.info("Make sure the Supervisor container is running")
            return CommandResult.fail("Connection failed")
        except Exception as e:
            cli.renderer.error(f"API error: {e}")
            return CommandResult.fail(str(e))

    def get_completions(self, partial: str) -> list:
        """Get subcommand completions."""
        subcommands = [
            "create",
            "new",
            "add",
            "list",
            "get",
            "start",
            "pause",
            "kill",
            "delete",
            "log",
            "stats",
            "templates",
            "from-template",
            "move",
        ]
        return [s for s in subcommands if s.startswith(partial)]
