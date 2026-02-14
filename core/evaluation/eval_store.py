"""
Eval Store — SQLite-backed evaluation results persistence (Cline evals pattern).

KISS: single file, no ORM, no migrations. SQLite is the database.
Records every eval run with model, task, success, time, cost, tokens.
Query for comparisons across models and tasks.

Usage:
    store = EvalStore()  # ~/.jotty/evals.db
    run_id = store.start_run(model="claude-sonnet-4-20250514", benchmark="coding")
    store.record_result(run_id, task_id="fib", success=True, time=2.3, cost=0.01)
    store.record_result(run_id, task_id="sort", success=False, time=5.1, cost=0.02)
    store.finish_run(run_id)

    # Compare models
    report = store.compare_models(benchmark="coding")
    # {'claude-sonnet-4-20250514': {'pass_rate': 0.5, ...}, ...}
"""

import sqlite3
import time
import uuid
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

_DEFAULT_DB = Path.home() / ".jotty" / "evals.db"


class EvalStore:
    """SQLite-backed eval results store."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path) if db_path else _DEFAULT_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                benchmark TEXT NOT NULL,
                started_at REAL NOT NULL,
                finished_at REAL,
                status TEXT DEFAULT 'running',
                metadata TEXT DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL REFERENCES runs(id),
                task_id TEXT NOT NULL,
                success INTEGER NOT NULL,
                answer TEXT,
                error TEXT,
                execution_time REAL DEFAULT 0,
                cost REAL DEFAULT 0,
                tokens_used INTEGER DEFAULT 0,
                recorded_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_results_run ON results(run_id);
            CREATE INDEX IF NOT EXISTS idx_runs_model ON runs(model);
        """)
        self._conn.commit()

    def start_run(
        self, model: str, benchmark: str, metadata: Optional[Dict] = None
    ) -> str:
        """Start an eval run. Returns run_id."""
        run_id = str(uuid.uuid4())[:8]
        import json
        self._conn.execute(
            "INSERT INTO runs (id, model, benchmark, started_at, metadata) VALUES (?,?,?,?,?)",
            (run_id, model, benchmark, time.time(), json.dumps(metadata or {})),
        )
        self._conn.commit()
        logger.info(f"Eval run started: {run_id} ({model} on {benchmark})")
        return run_id

    def record_result(
        self,
        run_id: str,
        task_id: str,
        success: bool,
        answer: str = "",
        error: str = "",
        execution_time: float = 0,
        cost: float = 0,
        tokens_used: int = 0,
    ):
        """Record a single task result."""
        self._conn.execute(
            "INSERT INTO results (run_id,task_id,success,answer,error,execution_time,cost,tokens_used,recorded_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (run_id, task_id, int(success), answer, error, execution_time, cost, tokens_used, time.time()),
        )
        self._conn.commit()

    def finish_run(self, run_id: str) -> None:
        """Mark a run as finished."""
        self._conn.execute(
            "UPDATE runs SET finished_at=?, status='finished' WHERE id=?",
            (time.time(), run_id),
        )
        self._conn.commit()

    def get_run_summary(self, run_id: str) -> Dict[str, Any]:
        """Get summary for a single run."""
        run = self._conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if not run:
            return {}
        results = self._conn.execute(
            "SELECT * FROM results WHERE run_id=?", (run_id,)
        ).fetchall()
        total = len(results)
        passed = sum(1 for r in results if r['success'])
        total_time = sum(r['execution_time'] for r in results)
        total_cost = sum(r['cost'] for r in results)
        total_tokens = sum(r['tokens_used'] for r in results)
        return {
            'run_id': run_id,
            'model': run['model'],
            'benchmark': run['benchmark'],
            'status': run['status'],
            'total': total,
            'passed': passed,
            'pass_rate': passed / total if total else 0,
            'avg_time': total_time / total if total else 0,
            'total_cost': total_cost,
            'total_tokens': total_tokens,
        }

    def compare_models(self, benchmark: Optional[str] = None) -> Dict[str, Any]:
        """Compare all models across runs. Returns {model: {pass_rate, ...}}."""
        where = "WHERE r.benchmark=?" if benchmark else ""
        params = (benchmark,) if benchmark else ()
        rows = self._conn.execute(f"""
            SELECT r.model,
                   COUNT(*) as total,
                   SUM(res.success) as passed,
                   AVG(res.execution_time) as avg_time,
                   SUM(res.cost) as total_cost,
                   SUM(res.tokens_used) as total_tokens
            FROM results res
            JOIN runs r ON res.run_id = r.id
            {where}
            GROUP BY r.model
            ORDER BY passed DESC
        """, params).fetchall()
        return {
            row['model']: {
                'total': row['total'],
                'passed': row['passed'],
                'pass_rate': row['passed'] / row['total'] if row['total'] else 0,
                'avg_time': row['avg_time'],
                'total_cost': row['total_cost'],
                'total_tokens': row['total_tokens'],
            }
            for row in rows
        }

    def list_runs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """List recent runs."""
        rows = self._conn.execute(
            "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self._conn.close()
