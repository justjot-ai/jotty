#!/usr/bin/env python3
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from enum import Enum
from dataclasses import dataclass, asdict
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich.prompt import Prompt, Confirm
from rich.columns import Columns
import sys

console = Console()

class Priority(Enum):
    LOW = ("low", "blue")
    MEDIUM = ("medium", "yellow")
    HIGH = ("high", "red")
    CRITICAL = ("critical", "bright_red")

class Status(Enum):
    TODO = ("todo", "cyan")
    IN_PROGRESS = ("in_progress", "yellow")
    IN_REVIEW = ("in_review", "magenta")
    BLOCKED = ("blocked", "red")
    DONE = ("done", "green")

class RecurringType(Enum):
    NONE = "none"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"

@dataclass
class Task:
    id: str
    title: str
    description: str
    priority: str
    status: str
    project: str
    tags: List[str]
    due_date: Optional[str]
    created_at: str
    updated_at: str
    subtasks: List[Dict]
    recurring: str
    assigned_to: Optional[str] = None
    
    def to_dict(self):
        return asdict(self)

class TaskManager:
    def __init__(self, data_file="tasks_data.json"):
        self.data_file = data_file
        self.tasks: List[Task] = []
        self.projects: List[str] = []
        self.load_data()
    
    def load_data(self):
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    self.tasks = [Task(**task) for task in data.get('tasks', [])]
                    self.projects = data.get('projects', [])
            except Exception as e:
                console.print(f"[red]Error loading data: {e}[/red]")
                self.tasks = []
                self.projects = []
        else:
            self.tasks = []
            self.projects = []
    
    def save_data(self):
        try:
            data = {
                'tasks': [task.to_dict() for task in self.tasks],
                'projects': self.projects
            }
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            console.print(f"[red]Error saving data: {e}[/red]")
    
    def generate_id(self) -> str:
        if not self.tasks:
            return "T001"
        max_id = max([int(task.id[1:]) for task in self.tasks])
        return f"T{str(max_id + 1).zfill(3)}"
    
    def create_task(self, title: str, description: str, priority: str, 
                   project: str, tags: List[str], due_date: Optional[str],
                   recurring: str = "none") -> Task:
        task = Task(
            id=self.generate_id(),
            title=title,
            description=description,
            priority=priority,
            status="todo",
            project=project,
            tags=tags,
            due_date=due_date,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            subtasks=[],
            recurring=recurring
        )
        self.tasks.append(task)
        if project and project not in self.projects:
            self.projects.append(project)
        self.save_data()
        return task
    
    def update_task_status(self, task_id: str, status: str):
        task = self.get_task(task_id)
        if task:
            task.status = status
            task.updated_at = datetime.now().isoformat()
            self.save_data()
    
    def get_task(self, task_id: str) -> Optional[Task]:
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None
    
    def delete_task(self, task_id: str):
        self.tasks = [t for t in self.tasks if t.id != task_id]
        self.save_data()
    
    def add_subtask(self, task_id: str, subtask_title: str):
        task = self.get_task(task_id)
        if task:
            subtask = {
                'id': f"{task_id}-ST{len(task.subtasks) + 1}",
                'title': subtask_title,
                'completed': False
            }
            task.subtasks.append(subtask)
            task.updated_at = datetime.now().isoformat()
            self.save_data()
    
    def toggle_subtask(self, task_id: str, subtask_index: int):
        task = self.get_task(task_id)
        if task and 0 <= subtask_index < len(task.subtasks):
            task.subtasks[subtask_index]['completed'] = not task.subtasks[subtask_index]['completed']
            task.updated_at = datetime.now().isoformat()
            self.save_data()
    
    def get_tasks_by_filter(self, **filters) -> List[Task]:
        filtered = self.tasks
        
        if 'status' in filters:
            filtered = [t for t in filtered if t.status == filters['status']]
        if 'priority' in filters:
            filtered = [t for t in filtered if t.priority == filters['priority']]
        if 'project' in filters:
            filtered = [t for t in filtered if t.project == filters['project']]
        if 'tags' in filters:
            filtered = [t for t in filtered if any(tag in t.tags for tag in filters['tags'])]
        if 'due_today' in filters and filters['due_today']:
            today = datetime.now().date().isoformat()
            filtered = [t for t in filtered if t.due_date == today]
        
        return filtered
    
    def search_tasks(self, query: str) -> List[Task]:
        query = query.lower()
        return [t for t in self.tasks if 
                query in t.title.lower() or 
                query in t.description.lower() or
                query in t.id.lower()]

class TaskUI:
    def __init__(self):
        self.manager = TaskManager()
        self.current_view = "kanban"
        self.selected_index = 0
        self.current_filter = {}
    
    def display_kanban_board(self):
        console.clear()
        console.print(Panel("[bold cyan]Kanban Board View[/bold cyan]", expand=False))
        
        columns_data = {}
        for status in Status:
            columns_data[status.value[0]] = []
        
        tasks = self.manager.get_tasks_by_filter(**self.current_filter)
        for task in tasks:
            if task.status in columns_data:
                columns_data[task.status].append(task)
        
        tables = []
        for status in Status:
            table = Table(title=f"[{status.value[1]}]{status.value[0].upper().replace('_', ' ')}[/{status.value[1]}]",
                         show_header=False, border_style=status.value[1])
            
            for task in columns_data[status.value[0]]:
                priority_color = next((p.value[1] for p in Priority if p.value[0] == task.priority), "white")
                card = f"[{priority_color}]●[/{priority_color}] {task.id}\n{task.title[:30]}"
                if task.due_date:
                    card += f"\n📅 {task.due_date}"
                if task.tags:
                    card += f"\n🏷️  {', '.join(task.tags[:3])}"
                table.add_row(card)
            
            if not columns_data[status.value[0]]:
                table.add_row("[dim]No tasks[/dim]")
            
            tables.append(table)
        
        console.print(Columns(tables, equal=True, expand=True))
    
    def display_list_view(self):
        console.clear()
        console.print(Panel("[bold cyan]List View[/bold cyan]", expand=False))
        
        tasks = self.manager.get_tasks_by_filter(**self.current_filter)
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("ID", style="cyan", width=6)
        table.add_column("Title", style="white", width=30)
        table.add_column("Status", width=15)
        table.add_column("Priority", width=10)
        table.add_column("Project", style="yellow", width=15)
        table.add_column("Due Date", width=12)
        table.add_column("Tags", width=20)
        
        for i, task in enumerate(tasks):
            status_color = next((s.value[1] for s in Status if s.value[0] == task.status), "white")
            priority_color = next((p.value[1] for p in Priority if p.value[0] == task.priority), "white")
            
            row_style = "on dark_blue" if i == self.selected_index else ""
            
            table.add_row(
                task.id,
                task.title[:28] + "..." if len(task.title) > 30 else task.title,
                f"[{status_color}]{task.status}[/{status_color}]",
                f"[{priority_color}]{task.priority}[/{priority_color}]",
                task.project or "-",
                task.due_date or "-",
                ", ".join(task.tags[:2]) if task.tags else "-",
                style=row_style
            )
        
        if not tasks:
            console.print("[dim]No tasks found[/dim]")
        else:
            console.print(table)
    
    def display_daily_planner(self):
        console.clear()
        console.print(Panel("[bold cyan]Daily Planner - Today's Focus[/bold cyan]", expand=False))
        
        today = datetime.now().date().isoformat()
        tasks = self.manager.get_tasks_by_filter(due_today=True)
        
        console.print(f"\n[bold]📅 {datetime.now().strftime('%A, %B %d, %Y')}[/bold]\n")
        
        if not tasks:
            console.print("[dim]No tasks due today. You're all clear! ✨[/dim]\n")
        else:
            for priority in [Priority.CRITICAL, Priority.HIGH, Priority.MEDIUM, Priority.LOW]:
                priority_tasks = [t for t in tasks if t.priority == priority.value[0]]
                if priority_tasks:
                    console.print(f"\n[{priority.value[1]}]{'━' * 60}[/{priority.value[1]}]")
                    console.print(f"[bold {priority.value[1]}]{priority.value[0].upper()} PRIORITY[/bold {priority.value[1]}]")
                    console.print(f"[{priority.value[1]}]{'━' * 60}[/{priority.value[1]}]\n")
                    
                    for task in priority_tasks:
                        status_color = next((s.value[1] for s in Status if s.value[0] == task.status), "white")
                        console.print(f"  [{status_color}]■[/{status_color}] {task.id} - {task.title}")
                        console.print(f"    Status: [{status_color}]{task.status}[/{status_color}]")
                        if task.project:
                            console.print(f"    Project: [yellow]{task.project}[/yellow]")
                        if task.subtasks:
                            completed = sum(1 for st in task.subtasks if st['completed'])
                            console.print(f"    Subtasks: {completed}/{len(task.subtasks)} completed")
                        console.print()
        
        overdue_tasks = [t for t in self.manager.tasks 
                        if t.due_date and t.due_date < today and t.status != "done"]
        if overdue_tasks:
            console.print(f"\n[bold red]⚠️  OVERDUE TASKS ({len(overdue_tasks)})[/bold red]")
            for task in overdue_tasks[:5]:
                console.print(f"  [red]■[/red] {task.id} - {task.title} (Due: {task.due_date})")
    
    def display_task_detail(self, task: Task):
        console.clear()
        
        priority_color = next((p.value[1] for p in Priority if p.value[0] == task.priority), "white")
        status_color = next((s.value[1] for s in Status if s.value[0] == task.status), "white")
        
        detail = f"""[bold cyan]{task.id}[/bold cyan] - [bold]{task.title}[/bold]

[dim]Description:[/dim]
{task.description or 'No description'}

[dim]Status:[/dim] [{status_color}]{task.status}[/{status_color}]
[dim]Priority:[/dim] [{priority_color}]{task.priority}[/{priority_color}]
[dim]Project:[/dim] {task.project or 'None'}
[dim]Tags:[/dim] {', '.join(task.tags) if task.tags else 'None'}
[dim]Due Date:[/dim] {task.due_date or 'Not set'}
[dim]Recurring:[/dim] {task.recurring}
[dim]Created:[/dim] {task.created_at[:10]}
[dim]Updated:[/dim] {task.updated_at[:10]}
"""
        
        if task.subtasks:
            detail += "\n[bold]Subtasks:[/bold]\n"
            for i, subtask in enumerate(task.subtasks):
                check = "✓" if subtask['completed'] else "○"
                style = "green" if subtask['completed'] else "white"
                detail += f"  [{style}]{check}[/{style}] {i+1}. {subtask['title']}\n"
        
        console.print(Panel(detail, title="Task Details", border_style="cyan"))
    
    def create_task_interactive(self):
        console.clear()
        console.print(Panel("[bold green]Create New Task[/bold green]", expand=False))
        
        title = Prompt.ask("\n[cyan]Task Title[/cyan]")
        description = Prompt.ask("[cyan]Description[/cyan]", default="")
        
        console.print("\n[yellow]Priority:[/yellow] 1=Low, 2=Medium, 3=High, 4=Critical")
        priority_map = {"1": "low", "2": "medium", "3": "high", "4": "critical"}
        priority_choice = Prompt.ask("Select priority", choices=["1", "2", "3", "4"], default="2")
        priority = priority_map[priority_choice]
        
        project = Prompt.ask("[cyan]Project[/cyan]", default="")
        tags_input = Prompt.ask("[cyan]Tags (comma-separated)[/cyan]", default="")
        tags = [t.strip() for t in tags_input.split(",") if t.strip()]
        
        due_date_input = Prompt.ask("[cyan]Due Date (YYYY-MM-DD)[/cyan]", default="")
        due_date = due_date_input if due_date_input else None
        
        console.print("\n[yellow]Recurring:[/yellow] 0=None, 1=Daily, 2=Weekly, 3=Monthly")
        recurring_map = {"0": "none", "1": "daily", "2": "weekly", "3": "monthly"}
        recurring_choice = Prompt.ask("Select recurring", choices=["0", "1", "2", "3"], default="0")
        recurring = recurring_map[recurring_choice]
        
        task = self.manager.create_task(title, description, priority, project, tags, due_date, recurring)
        console.print(f"\n[green]✓ Task {task.id} created successfully![/green]")
        
        if Confirm.ask("\nAdd subtasks?", default=False):
            while True:
                subtask_title = Prompt.ask("[cyan]Subtask title[/cyan] (or press Enter to finish)", default="")
                if not subtask_title:
                    break
                self.manager.add_subtask(task.id, subtask_title)
                console.print("[green]✓ Subtask added[/green]")
        
        Prompt.ask("\nPress Enter to continue")
    
    def search_interactive(self):
        console.clear()
        console.print(Panel("[bold cyan]Search Tasks[/bold cyan]", expand=False))
        
        query = Prompt.ask("\n[cyan]Search query[/cyan]")
        results = self.manager.search_tasks(query)
        
        if not results:
            console.print(f"\n[yellow]No tasks found matching '{query}'[/yellow]")
        else:
            console.print(f"\n[green]Found {len(results)} task(s):[/green]\n")
            for task in results:
                priority_color = next((p.value[1] for p in Priority if p.value[0] == task.priority), "white")
                console.print(f"  [{priority_color}]●[/{priority_color}] {task.id} - {task.title}")
                console.print(f"    Status: {task.status} | Priority: {task.priority}")
                if task.project:
                    console.print(f"    Project: {task.project}")
                console.print()
        
        Prompt.ask("\nPress Enter to continue")
    
    def filter_interactive(self):
        console.clear()
        console.print(Panel("[bold cyan]Filter Tasks[/bold cyan]", expand=False))
        
        console.print("\n[yellow]Filter Options:[/yellow]")
        console.print("1. By Status")
        console.print("2. By Priority")
        console.print("3. By Project")
        console.print("4. Clear Filters")
        console.print("5. Back")
        
        choice = Prompt.ask("\nSelect option", choices=["1", "2", "3", "4", "5"])
        
        if choice == "1":
            console.print("\n[yellow]Status:[/yellow]")
            for i, status in enumerate(Status, 1):
                console.print(f"{i}. {status.value[0]}")
            status_choice = Prompt.ask("Select status", choices=[str(i) for i in range(1, len(Status)+1)])
            self.current_filter['status'] = list(Status)[int(status_choice)-1].value[0]
            console.print(f"[green]✓ Filtered by status: {self.current_filter['status']}[/green]")
        
        elif choice == "2":
            console.print("\n[yellow]Priority:[/yellow]")
            for i, priority in enumerate(Priority, 1):
                console.print(f"{i}. {priority.value[0]}")
            priority_choice = Prompt.ask("Select priority", choices=[str(i) for i in range(1, len(Priority)+1)])
            self.current_filter['priority'] = list(Priority)[int(priority_choice)-1].value[0]
            console.print(f"[green]✓ Filtered by priority: {self.current_filter['priority']}[/green]")
        
        elif choice == "3":
            if not self.manager.projects:
                console.print("\n[yellow]No projects found[/yellow]")
            else:
                console.print("\n[yellow]Projects:[/yellow]")
                for i, project in enumerate(self.manager.projects, 1):
                    console.print(f"{i}. {project}")
                project_choice = Prompt.ask("Select project", 
                                          choices=[str(i) for i in range(1, len(self.manager.projects)+1)])
                self.current_filter['project'] = self.manager.projects[int(project_choice)-1]
                console.print(f"[green]✓ Filtered by project: {self.current_filter['project']}[/green]")
        
        elif choice == "4":
            self.current_filter = {}
            console.print("[green]✓ All filters cleared[/green]")
        
        if choice != "5":
            Prompt.ask("\nPress Enter to continue")
    
    def update_task_interactive(self):
        console.clear()
        console.print(Panel("[bold cyan]Update Task[/bold cyan]", expand=False))
        
        task_id = Prompt.ask("\n[cyan]Enter Task ID[/cyan]").upper()
        task = self.manager.get_task(task_id)
        
        if not task:
            console.print(f"[red]Task {task_id} not found[/red]")
            Prompt.ask("\nPress Enter to continue")
            return
        
        self.display_task_detail(task)
        
        console.print("\n[yellow]Update Options:[/yellow]")
        console.print("1. Update Status")
        console.print("2. Update Priority")
        console.print("3. Add Subtask")
        console.print("4. Toggle Subtask")
        console.print("5. Delete Task")
        console.print("6. Back")
        
        choice = Prompt.ask("\nSelect option", choices=["1", "2", "3", "4", "5", "6"])
        
        if choice == "1":
            console.print("\n[yellow]Status:[/yellow]")
            for i, status in enumerate(Status, 1):
                console.print(f"{i}. {status.value[0]}")
            status_choice = Prompt.ask("Select status", choices=[str(i) for i in range(1, len(Status)+1)])
            new_status = list(Status)[int(status_choice)-1].value[0]
            self.manager.update_task_status(task_id, new_status)
            console.print(f"[green]✓ Status updated to {new_status}[/green]")
        
        elif choice == "2":
            console.print("\n[yellow]Priority:[/yellow]")
            for i, priority in enumerate(Priority, 1):
                console.print(f"{i}. {priority.value[0]}")
            priority_choice = Prompt.ask("Select priority", choices=[str(i) for i in range(1, len(Priority)+1)])
            task.priority = list(Priority)[int(priority_choice)-1].value[0]
            task.updated_at = datetime.now().isoformat()
            self.manager.save_data()
            console.print(f"[green]✓ Priority updated to {task.priority}[/green]")
        
        elif choice == "3":
            subtask_title = Prompt.ask("[cyan]Subtask title[/cyan]")
            self.manager.add_subtask(task_id, subtask_title)
            console.print("[green]✓ Subtask added[/green]")
        
        elif choice == "4":
            if not task.subtasks:
                console.print("[yellow]No subtasks available[/yellow]")
            else:
                for i, subtask in enumerate(task.subtasks, 1):
                    console.print(f"{i}. [{subtask['completed'] and 'green' or 'white'}]{subtask['title']}[/]")
                subtask_index = int(Prompt.ask("Select subtask number", 
                                              choices=[str(i) for i in range(1, len(task.subtasks)+1)])) - 1
                self.manager.toggle_subtask(task_id, subtask_index)
                console.print("[green]✓ Subtask toggled[/green]")
        
        elif choice == "5":
            if Confirm.ask(f"\n[red]Delete task {task_id}?[/red]", default=False):
                self.manager.delete_task(task_id)
                console.print("[green]✓ Task deleted[/green]")
        
        if choice != "6":
            Prompt.ask("\nPress Enter to continue")
    
    def show_help(self):
        console.clear()
        help_text = """[bold cyan]Unified Task Manager - Help[/bold cyan]

[bold yellow]Views:[/bold yellow]
  • Kanban Board - Visual workflow columns
  • List View - Detailed table of all tasks
  • Daily Planner - Focus on today's tasks

[bold yellow]Navigation:[/bold yellow]
  • j/k - Move down/up (vim-style)
  • h/l - Switch views left/right
  • Enter - Select/View details

[bold yellow]Features:[/bold yellow]
  • Create tasks with priorities, projects, and tags
  • Add subtasks for complex work breakdown
  • Set recurring tasks (daily, weekly, monthly)
  • Search and filter tasks
  • Track overdue items
  • Color-coded priorities and statuses

[bold yellow]Status Workflow:[/bold yellow]
  TODO → IN_PROGRESS → IN_REVIEW → DONE
         ↓
      BLOCKED

[bold yellow]Priority Levels:[/bold yellow]
  • LOW - Nice to have
  • MEDIUM - Standard priority
  • HIGH - Important work
  • CRITICAL - Urgent, top priority

[bold yellow]Data Storage:[/bold yellow]
  All tasks saved to: tasks_data.json
"""
        console.print(Panel(help_text, border_style="cyan"))
        Prompt.ask("\nPress Enter to continue")
    
    def show_statistics(self):
        console.clear()
        console.print(Panel("[bold cyan]Task Statistics[/bold cyan]", expand=False))
        
        total = len(self.manager.tasks)
        by_status = {}
        by_priority = {}
        by_project = {}
        
        for task in self.manager.tasks:
            by_status[task.status] = by_status.get(task.status, 0) + 1
            by_priority[task.priority] = by_priority.get(task.priority, 0) + 1
            if task.project:
                by_project[task.project] = by_project.get(task.project, 0) + 1
        
        console.print(f"\n[bold]Total Tasks:[/bold] {total}\n")
        
        console.print("[bold yellow]By Status:[/bold yellow]")
        for status in Status:
            count = by_status.get(status.value[0], 0)
            bar = "█" * (count * 2)
            console.print(f"  [{status.value[1]}]{status.value[0]:15}[/{status.value[1]}] {count:3} {bar}")
        
        console.print("\n[bold yellow]By Priority:[/bold yellow]")
        for priority in Priority:
            count = by_priority.get(priority.value[0], 0)
            bar = "█" * (count * 2)
            console.print(f"  [{priority.value[1]}]{priority.value[0]:15}[/{priority.value[1]}] {count:3} {bar}")
        
        if by_project:
            console.print("\n[bold yellow]By Project:[/bold yellow]")
            for project, count in sorted(by_project.items(), key=lambda x: x[1], reverse=True)[:10]:
                bar = "█" * (count * 2)
                console.print(f"  {project:15} {count:3} {bar}")
        
        overdue = sum(1 for t in self.manager.tasks 
                     if t.due_date and t.due_date < datetime.now().date().isoformat() and t.status != "done")
        if overdue > 0:
            console.print(f"\n[bold red]⚠️  {overdue} Overdue Tasks[/bold red]")
        
        Prompt.ask("\nPress Enter to continue")
    
    def main_menu(self):
        while True:
            console.clear()
            
            title = Text("UNIFIED TASK MANAGER", style="bold cyan", justify="center")
            console.print(Panel(title, border_style="cyan"))
            
            filter_info = ""
            if self.current_filter:
                filters = ", ".join([f"{k}={v}" for k, v in self.current_filter.items()])
                filter_info = f"\n[yellow]Active Filters: {filters}[/yellow]"
            
            console.print(f"\n[dim]Tasks: {len(self.manager.tasks)} | Projects: {len(self.manager.projects)}[/dim]{filter_info}\n")
            
            console.print("[bold yellow]Views:[/bold yellow]")
            console.print("  1. 📊 Kanban Board")
            console.print("  2. 📋 List View")
            console.print("  3. 📅 Daily Planner")
            console.print("\n[bold yellow]Actions:[/bold yellow]")
            console.print("  4. ➕ Create Task")
            console.print("  5. ✏️  Update Task")
            console.print("  6. 🔍 Search Tasks")
            console.print("  7. 🎯 Filter Tasks")
            console.print("  8. 📈 Statistics")
            console.print("  9. ❓ Help")
            console.print("  0. 🚪 Exit")
            
            choice = Prompt.ask("\n[cyan]Select option[/cyan]", 
                              choices=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
            
            if choice == "1":
                self.display_kanban_board()
                Prompt.ask("\nPress Enter to continue")
            elif choice == "2":
                self.display_list_view()
                Prompt.ask("\nPress Enter to continue")
            elif choice == "3":
                self.display_daily_planner()
                Prompt.ask("\nPress Enter to continue")
            elif choice == "4":
                self.create_task_interactive()
            elif choice == "5":
                self.update_task_interactive()
            elif choice == "6":
                self.search_interactive()
            elif choice == "7":
                self.filter_interactive()
            elif choice == "8":
                self.show_statistics()
            elif choice == "9":
                self.show_help()
            elif choice == "0":
                if Confirm.ask("\n[yellow]Exit Task Manager?[/yellow]", default=True):
                    console.print("\n[cyan]Goodbye! Stay productive! ✨[/cyan]\n")
                    break

def main():
    try:
        ui = TaskUI()
        ui.main_menu()
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Interrupted. Goodbye![/yellow]\n")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]\n")
        sys.exit(1)

if __name__ == "__main__":
    main()