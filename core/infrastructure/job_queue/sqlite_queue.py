"""
SQLite Task Queue Implementation
Preserves all supervisor functionality
"""

import sqlite3
import json
import subprocess
import os
from datetime import datetime
from contextlib import contextmanager
from typing import Optional, List, Dict, Any
from pathlib import Path

from .task_queue import TaskQueue
from .task import Task


class SQLiteTaskQueue(TaskQueue):
    """
    SQLite-backed task queue
    Compatible with supervisor's StateManager database schema
    """
    
    def __init__(self, db_path: str, init_schema: bool = True) -> None:
        """
        Initialize SQLite task queue
        
        Args:
            db_path: Path to SQLite database file
            init_schema: If True, initialize database schema if it doesn't exist
        """
        self.db_path = db_path
        self.db_dir = Path(db_path).parent
        self.db_dir.mkdir(parents=True, exist_ok=True)
        
        if init_schema:
            self._init_schema()
    
    @contextmanager
    def _get_connection(self) -> Any:
        """Get database connection with proper settings (matches supervisor)"""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for better concurrency (matches supervisor)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=30000")
        try:
            yield conn
        finally:
            conn.close()
    
    def _init_schema(self) -> Any:
        """Initialize database schema if it doesn't exist"""
        schema_file = Path(__file__).parent.parent.parent.parent / "supervisor" / "task_schema.sql"
        
        # If supervisor schema exists, use it
        if schema_file.exists():
            with open(schema_file, 'r') as f:
                schema_sql = f.read()
        else:
            # Otherwise, create minimal schema
            schema_sql = """
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT,
                category TEXT NOT NULL DEFAULT '',
                priority INTEGER NOT NULL DEFAULT 3,
                tags TEXT,
                status TEXT NOT NULL CHECK(status IN ('suggested', 'backlog', 'pending', 'in_progress', 'completed', 'failed', 'blocked', 'cancelled')) DEFAULT 'backlog',
                progress_percent INTEGER DEFAULT 0,
                pid INTEGER,
                worktree_path TEXT,
                git_branch TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                last_heartbeat TIMESTAMP,
                estimated_hours REAL,
                actual_hours REAL,
                error_message TEXT,
                retry_count INTEGER DEFAULT 0,
                max_retries INTEGER DEFAULT 3,
                depends_on TEXT,
                blocks TEXT,
                created_by TEXT DEFAULT 'system',
                assigned_to TEXT,
                agent_type TEXT DEFAULT 'claude',
                log_file TEXT,
                context_files TEXT,
                task_content TEXT,
                filename TEXT,
                reference_apps TEXT,
                estimated_effort TEXT
            );
            
            CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
            CREATE INDEX IF NOT EXISTS idx_tasks_priority ON tasks(priority);
            CREATE INDEX IF NOT EXISTS idx_tasks_category ON tasks(category);
            CREATE INDEX IF NOT EXISTS idx_tasks_pid ON tasks(pid);
            CREATE INDEX IF NOT EXISTS idx_tasks_created_at ON tasks(created_at);
            """
        
        with self._get_connection() as conn:
            conn.executescript(schema_sql)
            conn.commit()
    
    async def enqueue(self, task: Task) -> str:
        """Add task to queue"""
        with self._get_connection() as conn:
            # Convert task to dict for insertion
            task_dict = task.to_dict()
            
            # Build INSERT query
            columns = []
            values = []
            placeholders = []
            
            for key, value in task_dict.items():
                if key == 'metadata':  # Skip metadata, store as JSON in a field if needed
                    continue
                if value is not None:
                    # Convert enums to their values for SQLite
                    if hasattr(value, 'value'):
                        value = value.value
                    elif hasattr(value, 'name'):
                        value = value.name
                    columns.append(key)
                    values.append(value)
                    placeholders.append('?')
            
            query = f"""
                INSERT OR REPLACE INTO tasks ({', '.join(columns)})
                VALUES ({', '.join(placeholders)})
            """
            
            conn.execute(query, values)
            conn.commit()
            
            return task.task_id
    
    async def dequeue(self, filters: Optional[Dict[str, Any]] = None) -> Optional[Task]:
        """Get next pending task (compatible with supervisor)"""
        filters = filters or {}
        agent_type = filters.get('agent_type')
        
        with self._get_connection() as conn:
            if agent_type:
                row = conn.execute("""
                    SELECT * FROM tasks
                    WHERE status = 'pending' AND COALESCE(agent_type, 'claude') = ?
                    ORDER BY priority ASC, task_id ASC
                    LIMIT 1
                """, (agent_type,)).fetchone()
            else:
                row = conn.execute("""
                    SELECT * FROM tasks
                    WHERE status = 'pending'
                    ORDER BY priority ASC, task_id ASC
                    LIMIT 1
                """).fetchone()
            
            if row:
                return Task.from_dict(dict(row))
            return None
    
    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID"""
        with self._get_connection() as conn:
            row = conn.execute("""
                SELECT * FROM tasks WHERE task_id = ?
            """, (task_id,)).fetchone()
            
            if row:
                return Task.from_dict(dict(row))
            return None
    
    async def update_status(self, task_id: str, status: str, pid: Optional[int] = None, error: Optional[str] = None, log_file: Optional[str] = None, agent_type: Optional[str] = None, **kwargs: Any) -> bool:
        """Update task status (compatible with supervisor)"""
        with self._get_connection() as conn:
            if status == 'in_progress':
                if agent_type:
                    conn.execute("""
                        UPDATE tasks
                        SET status = ?, pid = ?, log_file = ?, agent_type = ?,
                            started_at = CURRENT_TIMESTAMP, last_heartbeat = CURRENT_TIMESTAMP
                        WHERE task_id = ?
                    """, (status, pid, log_file, agent_type, task_id))
                else:
                    conn.execute("""
                        UPDATE tasks
                        SET status = ?, pid = ?, log_file = ?,
                            started_at = CURRENT_TIMESTAMP, last_heartbeat = CURRENT_TIMESTAMP
                        WHERE task_id = ?
                    """, (status, pid, log_file, task_id))
            elif status == 'completed':
                conn.execute("""
                    UPDATE tasks
                    SET status = ?, completed_at = CURRENT_TIMESTAMP, pid = NULL
                    WHERE task_id = ?
                """, (status, task_id))
            elif status == 'failed':
                conn.execute("""
                    UPDATE tasks
                    SET status = ?, error_message = ?, completed_at = CURRENT_TIMESTAMP, pid = NULL
                    WHERE task_id = ?
                """, (status, error, task_id))
            else:
                # For other statuses (suggested, backlog, pending), update status and clear pid/log_file if moving away from in_progress
                if pid is None:
                    # Clear pid, log_file, and all completion fields when explicitly set to None (moving away from in_progress/completed/failed)
                    conn.execute("""
                        UPDATE tasks
                        SET status = ?, pid = NULL, log_file = NULL, started_at = NULL, last_heartbeat = NULL, completed_at = NULL, error_message = NULL
                        WHERE task_id = ?
                    """, (status, task_id))
                else:
                    conn.execute("""
                        UPDATE tasks SET status = ? WHERE task_id = ?
                    """, (status, task_id))
            
            conn.commit()
            rows_updated = conn.total_changes
            if rows_updated == 0:
                # Task might not exist - log for debugging
                import logging
                logging.warning(f"No rows updated for task {task_id} - task may not exist")
            return rows_updated > 0
    
    async def heartbeat(self, task_id: str) -> bool:
        """Update task heartbeat"""
        with self._get_connection() as conn:
            conn.execute("""
                UPDATE tasks SET last_heartbeat = CURRENT_TIMESTAMP WHERE task_id = ?
            """, (task_id,))
            conn.commit()
            return conn.total_changes > 0
    
    async def get_running_count(self) -> int:
        """Get count of running tasks"""
        with self._get_connection() as conn:
            count = conn.execute("""
                SELECT COUNT(*) FROM tasks WHERE pid IS NOT NULL
            """).fetchone()[0]
            return count
    
    async def get_running_count_by_agent(self, agent_type: str) -> int:
        """Get count of running tasks for specific agent type"""
        with self._get_connection() as conn:
            count = conn.execute("""
                SELECT COUNT(*) FROM tasks
                WHERE pid IS NOT NULL AND COALESCE(agent_type, 'claude') = ?
            """, (agent_type,)).fetchone()[0]
            return count
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics (compatible with supervisor)"""
        with self._get_connection() as conn:
            stats = {
                'pending': conn.execute("SELECT COUNT(*) FROM tasks WHERE status = 'pending'").fetchone()[0],
                'in_progress': conn.execute("SELECT COUNT(*) FROM tasks WHERE status = 'in_progress'").fetchone()[0],
                'completed': conn.execute("SELECT COUNT(*) FROM tasks WHERE status = 'completed'").fetchone()[0],
                'failed': conn.execute("SELECT COUNT(*) FROM tasks WHERE status = 'failed'").fetchone()[0],
                'active_pids': conn.execute("SELECT COUNT(*) FROM tasks WHERE pid IS NOT NULL").fetchone()[0],
            }
            
            # Get active PIDs
            pids = conn.execute("SELECT pid FROM tasks WHERE pid IS NOT NULL").fetchall()
            stats['pids'] = [row[0] for row in pids]
            
            # Get per-agent stats
            agent_stats = conn.execute("""
                SELECT agent_type, COUNT(*) as count
                FROM tasks
                WHERE pid IS NOT NULL
                GROUP BY agent_type
            """).fetchall()
            stats['by_agent'] = {row[0] or 'claude': row[1] for row in agent_stats}
            
            return stats
    
    async def get_tasks_by_status(self, status: str) -> List[Task]:
        """Get all tasks with given status"""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM tasks WHERE status = ?
                ORDER BY priority ASC, created_at ASC
            """, (status,)).fetchall()
            
            return [Task.from_dict(dict(row)) for row in rows]
    
    async def get_running_tasks(self) -> List[Task]:
        """Get all running tasks"""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM tasks WHERE status = 'in_progress'
                ORDER BY started_at ASC
            """).fetchall()

            return [Task.from_dict(dict(row)) for row in rows]

    async def get_by_filename(self, filename: str) -> Optional[Task]:
        """Get task by filename (legacy support)"""
        with self._get_connection() as conn:
            row = conn.execute("""
                SELECT * FROM tasks WHERE filename = ?
            """, (filename,)).fetchone()

            if row:
                return Task.from_dict(dict(row))
            return None

    async def update_task_priority(self, task_id: str, priority: int) -> bool:
        """Update task priority"""
        if priority < 1 or priority > 5:
            return False
        
        with self._get_connection() as conn:
            conn.execute("""
                UPDATE tasks SET priority = ? WHERE task_id = ?
            """, (priority, task_id))
            conn.commit()
            return conn.total_changes > 0
    
    async def update_task_metadata(self, task_id: str, title: Optional[str] = None, description: Optional[str] = None, priority: Optional[int] = None, category: Optional[str] = None, context_files: Optional[str] = None, agent_type: Optional[str] = None, **kwargs: Any) -> bool:
        """Update task metadata"""
        with self._get_connection() as conn:
            updates = []
            params = []
            
            if title is not None:
                updates.append("title = ?")
                params.append(title)
            
            if description is not None:
                updates.append("description = ?")
                params.append(description)
            
            if priority is not None:
                if priority < 1 or priority > 5:
                    return False
                updates.append("priority = ?")
                params.append(priority)
            
            if category is not None:
                updates.append("category = ?")
                params.append(category)
            
            if context_files is not None:
                updates.append("context_files = ?")
                params.append(context_files)
            
            if agent_type is not None:
                if agent_type not in ['claude', 'cursor', 'opencode']:
                    return False
                updates.append("agent_type = ?")
                params.append(agent_type)
            
            if not updates:
                return True
            
            params.append(task_id)
            query = f"UPDATE tasks SET {', '.join(updates)} WHERE task_id = ?"
            
            conn.execute(query, params)
            conn.commit()
            return conn.total_changes > 0
    
    async def delete_task(self, task_id: str) -> bool:
        """Delete a task"""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM tasks WHERE task_id = ?", (task_id,))
            conn.commit()
            return conn.total_changes > 0
    
    async def create_task(self, title: str, description: str = '', priority: int = 3, category: str = '', context_files: Optional[str] = None, status: str = 'backlog', agent_type: Optional[str] = None, **kwargs: Any) -> Optional[str]:
        """Create a new task (compatible with supervisor)"""
        if agent_type is None:
            agent_type = 'claude'
        elif agent_type not in ['claude', 'cursor', 'opencode']:
            agent_type = 'claude'
        
        # Generate task_id: TASK-YYYYMMDD-XXXXX (matches supervisor)
        now = datetime.now()
        date_part = now.strftime("%Y%m%d")
        
        with self._get_connection() as conn:
            # Find next available number for today
            rows = conn.execute("""
                SELECT task_id FROM tasks
                WHERE task_id LIKE ?
                ORDER BY task_id DESC
                LIMIT 1
            """, (f"TASK-{date_part}-%",)).fetchall()
            
            if rows:
                last_id = rows[0]['task_id']
                last_num = int(last_id.split('-')[-1])
                next_num = last_num + 1
            else:
                next_num = 1
            
            task_id = f"TASK-{date_part}-{next_num:05d}"
            
            # Insert new task
            conn.execute("""
                INSERT INTO tasks (
                    task_id, title, description, category, priority, status, context_files,
                    created_by, agent_type
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                task_id, title, description, category, priority, status, context_files,
                'AI' if kwargs.get('suggested_by') else 'user', agent_type
            ))
            conn.commit()
            
            return task_id
    
    async def reset_task_to_backlog(self, task_id: str) -> bool:
        """Reset failed task back to backlog"""
        with self._get_connection() as conn:
            conn.execute("""
                UPDATE tasks
                SET status = 'backlog',
                    error_message = NULL,
                    pid = NULL,
                    started_at = NULL,
                    completed_at = NULL,
                    last_heartbeat = NULL,
                    retry_count = retry_count + 1
                WHERE task_id = ?
            """, (task_id,))
            conn.commit()
            return conn.total_changes > 0
    
    async def validate_pids(self) -> int:
        """Clean up stale PIDs (matches supervisor)"""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT task_id, pid FROM tasks WHERE pid IS NOT NULL").fetchall()
            cleaned = 0
            
            for row in rows:
                task_id, pid = row['task_id'], row['pid']
                try:
                    # Check if process exists
                    subprocess.run(['kill', '-0', str(pid)], check=True, capture_output=True)
                except (subprocess.CalledProcessError, FileNotFoundError):
                    # Process doesn't exist, clean up
                    conn.execute("""
                        UPDATE tasks SET pid = NULL, status = 'failed',
                        error_message = 'Process died unexpectedly',
                        completed_at = CURRENT_TIMESTAMP
                        WHERE task_id = ?
                    """, (task_id,))
                    cleaned += 1
            
            if cleaned > 0:
                conn.commit()
            
            return cleaned
    
    async def export_to_json(self) -> Dict[str, Any]:
        """Export state to JSON (compatible with supervisor format)"""
        await self.validate_pids()  # Clean up stale PIDs first
        
        with self._get_connection() as conn:
            tasks = conn.execute("""
                SELECT * FROM tasks
                ORDER BY priority ASC, created_at ASC
            """).fetchall()
            
            task_details = {}
            for t in tasks:
                task_dict = dict(t)
                task_id = task_dict['task_id']
                task_details[task_id] = {
                    'task_id': task_id,
                    'title': task_dict.get('title', ''),
                    'description': task_dict.get('description', ''),
                    'status': task_dict.get('status', 'backlog'),
                    'pid': task_dict.get('pid'),
                    'priority': task_dict.get('priority', 3),
                    'category': task_dict.get('category', ''),
                    'started_at': task_dict.get('started_at'),
                    'completed_at': task_dict.get('completed_at'),
                    'last_heartbeat': task_dict.get('last_heartbeat'),
                    'error_message': task_dict.get('error_message'),
                    'context_files': task_dict.get('context_files'),
                    'retry_count': task_dict.get('retry_count', 0),
                    'agent_type': task_dict.get('agent_type') or 'claude',
                }
            
            state = {
                'version': '2.0-sqlite-jotty',
                'total_tasks': len(tasks),
                'completed_tasks': sum(1 for t in tasks if t['status'] == 'completed'),
                'failed_tasks': sum(1 for t in tasks if t['status'] == 'failed'),
                'suggested_tasks': [t['task_id'] for t in tasks if t['status'] == 'suggested'],
                'backlog_tasks': [t['task_id'] for t in tasks if t['status'] == 'backlog'],
                'pending_tasks': [t['task_id'] for t in tasks if t['status'] == 'pending'],
                'in_progress_tasks': [t['task_id'] for t in tasks if t['status'] == 'in_progress'],
                'completed_task_files': [t['task_id'] for t in tasks if t['status'] == 'completed'],
                'failed_task_files': [t['task_id'] for t in tasks if t['status'] == 'failed'],
                'task_pids': {t['task_id']: t['pid'] for t in tasks if t['pid']},
                'task_status': {t['task_id']: t['status'] for t in tasks},
                'task_details': task_details,
                'last_validated': datetime.now().isoformat()
            }
            
            return state
