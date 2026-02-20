"""
LearningStore - Persistent SQLite-backed storage for all learning data.
=======================================================================

Single source of truth for learning state across agents, swarms, pipelines,
and orchestrator. Replaces scattered JSON files with centralized, queryable
storage.

Tables:
    episodes        — Execution episodes with outcomes
    patterns        — Extracted behavioral patterns (cross-domain transfer)
    reflections     — Mid-execution reflection entries
    value_estimates — TD/Q value estimates per (state, action) pair

Author: Jotty Team
Date: February 2026
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# DATA MODELS
# =============================================================================


@dataclass
class EpisodeRecord:
    """Record of a single execution episode."""

    episode_id: str
    unit_type: str  # "agent", "swarm", "pipeline", "orchestrator"
    unit_name: str
    domain: str
    task_type: str
    context: Dict[str, Any]
    action: Dict[str, Any]
    outcome: Dict[str, Any]
    success: bool
    quality: float  # 0.0 - 1.0
    execution_time: float
    cost: float
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    parent_episode_id: Optional[str] = None  # For nested episodes (pipeline stages)
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class PatternRecord:
    """An extracted behavioral pattern usable for transfer learning."""

    pattern_id: str
    source_domain: str
    pattern_type: str  # "success_strategy", "failure_avoidance", "tool_preference"
    description: str
    conditions: Dict[str, Any]  # When this pattern applies
    recommendation: str  # What to do
    confidence: float  # How reliable this pattern is (0.0 - 1.0)
    evidence_count: int  # How many episodes support this
    applicable_domains: List[str]  # Domains this transfers to
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class ReflectionRecord:
    """Mid-execution reflection entry."""

    reflection_id: str
    episode_id: str
    step: int
    unit_name: str
    observation: str  # What happened
    analysis: str  # Why it happened
    adjustment: str  # What to change
    applied: bool  # Whether the adjustment was applied
    improvement: Optional[float] = None  # Quality delta after applying
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class ValueEstimate:
    """Value estimate for a (state, action) pair."""

    state_key: str
    action_key: str
    domain: str
    value: float
    td_error: float
    update_count: int
    last_updated: float = 0.0
    unit_name: str = ""
    lessons: List[str] = field(default_factory=list)
    avg_reward: float = 0.0

    def __post_init__(self) -> None:
        if self.last_updated == 0.0:
            self.last_updated = time.time()


# =============================================================================
# LEARNING STORE
# =============================================================================


class LearningStore:
    """
    SQLite-backed persistent store for all learning data.

    Thread-safe, supports concurrent reads, serialized writes.
    All execution units (agents, swarms, pipelines, orchestrator) share
    this single store.

    Usage:
        store = LearningStore()  # Uses default path
        store.save_episode(episode_record)
        episodes = store.query_episodes(domain="coding", limit=10)
        patterns = store.get_patterns(domain="coding")
    """

    _instances: Dict[str, "LearningStore"] = {}
    _lock = threading.Lock()

    def __init__(self, db_path: Optional[str] = None) -> None:
        if db_path is None:
            db_dir = Path.home() / "jotty" / "learning"
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(db_dir / "learning.db")

        self._db_path = db_path
        self._local = threading.local()
        self._init_schema()
        logger.info(f"LearningStore initialized: {db_path}")

    @classmethod
    def get_instance(cls, db_path: Optional[str] = None) -> "LearningStore":
        """Get or create a singleton instance for the given path."""
        key = db_path or "default"
        with cls._lock:
            if key not in cls._instances:
                cls._instances[key] = cls(db_path)
            return cls._instances[key]

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self._db_path, timeout=10.0)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
        return self._local.conn

    def _init_schema(self) -> None:
        """Create tables if they don't exist, then migrate and index."""
        conn = self._get_conn()

        # Step 1: Create tables (no indexes on columns that may not exist yet)
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id TEXT PRIMARY KEY,
                unit_type TEXT NOT NULL,
                domain TEXT NOT NULL DEFAULT '',
                task_type TEXT NOT NULL DEFAULT '',
                context TEXT NOT NULL DEFAULT '{}',
                action TEXT NOT NULL DEFAULT '{}',
                outcome TEXT NOT NULL DEFAULT '{}',
                success INTEGER NOT NULL DEFAULT 0,
                quality REAL NOT NULL DEFAULT 0.0,
                execution_time REAL NOT NULL DEFAULT 0.0,
                cost REAL NOT NULL DEFAULT 0.0,
                timestamp REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_episodes_domain ON episodes(domain);
            CREATE INDEX IF NOT EXISTS idx_episodes_task ON episodes(task_type);
            CREATE INDEX IF NOT EXISTS idx_episodes_time ON episodes(timestamp);

            CREATE TABLE IF NOT EXISTS patterns (
                pattern_id TEXT PRIMARY KEY,
                source_domain TEXT NOT NULL,
                pattern_type TEXT NOT NULL,
                description TEXT NOT NULL,
                conditions TEXT NOT NULL DEFAULT '{}',
                recommendation TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.0,
                evidence_count INTEGER NOT NULL DEFAULT 0,
                applicable_domains TEXT NOT NULL DEFAULT '[]',
                timestamp REAL NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_patterns_domain ON patterns(source_domain);
            CREATE INDEX IF NOT EXISTS idx_patterns_type ON patterns(pattern_type);

            CREATE TABLE IF NOT EXISTS reflections (
                reflection_id TEXT PRIMARY KEY,
                episode_id TEXT NOT NULL,
                step INTEGER NOT NULL,
                observation TEXT NOT NULL,
                analysis TEXT NOT NULL,
                adjustment TEXT NOT NULL,
                applied INTEGER NOT NULL DEFAULT 0,
                improvement REAL,
                timestamp REAL NOT NULL,
                FOREIGN KEY (episode_id) REFERENCES episodes(episode_id)
            );

            CREATE INDEX IF NOT EXISTS idx_reflections_episode ON reflections(episode_id);

            CREATE TABLE IF NOT EXISTS value_estimates (
                state_key TEXT NOT NULL,
                action_key TEXT NOT NULL,
                domain TEXT NOT NULL DEFAULT '',
                value REAL NOT NULL DEFAULT 0.0,
                td_error REAL NOT NULL DEFAULT 0.0,
                update_count INTEGER NOT NULL DEFAULT 0,
                last_updated REAL NOT NULL,
                PRIMARY KEY (state_key, action_key, domain)
            );

            CREATE INDEX IF NOT EXISTS idx_values_domain ON value_estimates(domain);
        """
        )
        conn.commit()

        # Step 2: Migrate — add columns that may be missing from old DBs
        self._migrate_schema(conn)

        # Step 3: Create indexes on migrated columns (now guaranteed to exist)
        for idx_sql in [
            "CREATE INDEX IF NOT EXISTS idx_episodes_unit ON episodes(unit_type, unit_name)",
            "CREATE INDEX IF NOT EXISTS idx_episodes_parent ON episodes(parent_episode_id)",
            "CREATE INDEX IF NOT EXISTS idx_values_unit ON value_estimates(unit_name)",
        ]:
            try:
                conn.execute(idx_sql)
            except sqlite3.OperationalError:
                pass
        conn.commit()

    def _migrate_schema(self, conn: sqlite3.Connection) -> None:
        """Add new columns to existing tables that predate schema changes."""
        _table_migrations = {
            "episodes": [
                ("unit_name", "TEXT NOT NULL DEFAULT ''"),
                ("error_type", "TEXT"),
                ("error_message", "TEXT"),
                ("parent_episode_id", "TEXT"),
                ("metadata", "TEXT NOT NULL DEFAULT '{}'"),
            ],
            "reflections": [
                ("unit_name", "TEXT NOT NULL DEFAULT ''"),
            ],
            "value_estimates": [
                ("unit_name", "TEXT NOT NULL DEFAULT ''"),
                ("lessons", "TEXT NOT NULL DEFAULT '[]'"),
                ("avg_reward", "REAL NOT NULL DEFAULT 0.0"),
            ],
        }
        for table, migrations in _table_migrations.items():
            existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
            for col_name, col_type in migrations:
                if col_name not in existing:
                    try:
                        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}")
                        logger.info(f"Migrated {table}: added {col_name}")
                    except sqlite3.OperationalError:
                        pass
        conn.commit()

    # =========================================================================
    # EPISODES
    # =========================================================================

    def save_episode(self, episode: EpisodeRecord) -> None:
        """Save an execution episode."""
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO episodes
               (episode_id, unit_type, unit_name, domain, task_type,
                context, action, outcome, success, quality,
                execution_time, cost, error_type, error_message,
                parent_episode_id, timestamp, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                episode.episode_id,
                episode.unit_type,
                episode.unit_name,
                episode.domain,
                episode.task_type,
                json.dumps(episode.context, default=str),
                json.dumps(episode.action, default=str),
                json.dumps(episode.outcome, default=str),
                1 if episode.success else 0,
                episode.quality,
                episode.execution_time,
                episode.cost,
                episode.error_type,
                episode.error_message,
                episode.parent_episode_id,
                episode.timestamp,
                json.dumps(episode.metadata, default=str),
            ),
        )
        conn.commit()

    def update_episode_quality(
        self, episode_id: str, quality: float, outcome_patch: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update quality score (and optionally patch outcome) for an existing episode."""
        conn = self._get_conn()
        if outcome_patch:
            row = conn.execute(
                "SELECT outcome FROM episodes WHERE episode_id = ?", (episode_id,)
            ).fetchone()
            if row:
                existing = json.loads(row["outcome"]) if row["outcome"] else {}
                existing.update(outcome_patch)
                conn.execute(
                    "UPDATE episodes SET quality = ?, outcome = ? WHERE episode_id = ?",
                    (quality, json.dumps(existing, default=str), episode_id),
                )
            else:
                conn.execute(
                    "UPDATE episodes SET quality = ? WHERE episode_id = ?",
                    (quality, episode_id),
                )
        else:
            conn.execute(
                "UPDATE episodes SET quality = ? WHERE episode_id = ?",
                (quality, episode_id),
            )
        conn.commit()

    def get_episode_count(self, domain: Optional[str] = None) -> int:
        """Get total number of recorded episodes, optionally filtered by domain."""
        conn = self._get_conn()
        if domain:
            row = conn.execute(
                "SELECT COUNT(*) FROM episodes WHERE domain = ?", (domain,)
            ).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()
        return row[0] if row else 0

    def query_episodes(
        self,
        domain: Optional[str] = None,
        unit_type: Optional[str] = None,
        unit_name: Optional[str] = None,
        task_type: Optional[str] = None,
        success_only: bool = False,
        limit: int = 50,
        since: Optional[float] = None,
    ) -> List[EpisodeRecord]:
        """Query episodes with filters."""
        conn = self._get_conn()
        conditions = []
        params: list = []

        if domain:
            conditions.append("domain = ?")
            params.append(domain)
        if unit_type:
            conditions.append("unit_type = ?")
            params.append(unit_type)
        if unit_name:
            conditions.append("unit_name = ?")
            params.append(unit_name)
        if task_type:
            conditions.append("task_type = ?")
            params.append(task_type)
        if success_only:
            conditions.append("success = 1")
        if since:
            conditions.append("timestamp > ?")
            params.append(since)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"SELECT * FROM episodes {where} ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        return [self._row_to_episode(row) for row in rows]

    def get_success_rate(
        self, domain: str = "", task_type: str = "", window: int = 50
    ) -> Tuple[float, int]:
        """Get success rate for domain/task_type over recent episodes."""
        conn = self._get_conn()
        conditions = []
        params: list = []

        if domain:
            conditions.append("domain = ?")
            params.append(domain)
        if task_type:
            conditions.append("task_type = ?")
            params.append(task_type)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"""
            SELECT success, COUNT(*) as cnt FROM (
                SELECT success FROM episodes {where}
                ORDER BY timestamp DESC LIMIT ?
            ) GROUP BY success
        """
        params.append(window)

        rows = conn.execute(query, params).fetchall()
        total = sum(row["cnt"] for row in rows)
        successes = sum(row["cnt"] for row in rows if row["success"] == 1)

        return (successes / total if total > 0 else 0.0, total)

    def _row_to_episode(self, row: sqlite3.Row) -> EpisodeRecord:
        """Convert a database row to an EpisodeRecord."""
        return EpisodeRecord(
            episode_id=row["episode_id"],
            unit_type=row["unit_type"],
            unit_name=row["unit_name"],
            domain=row["domain"],
            task_type=row["task_type"],
            context=json.loads(row["context"]),
            action=json.loads(row["action"]),
            outcome=json.loads(row["outcome"]),
            success=bool(row["success"]),
            quality=row["quality"],
            execution_time=row["execution_time"],
            cost=row["cost"],
            error_type=row["error_type"],
            error_message=row["error_message"],
            parent_episode_id=row["parent_episode_id"],
            timestamp=row["timestamp"],
            metadata=json.loads(row["metadata"]),
        )

    # =========================================================================
    # PATTERNS
    # =========================================================================

    def save_pattern(self, pattern: PatternRecord) -> None:
        """Save or update a behavioral pattern."""
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO patterns
               (pattern_id, source_domain, pattern_type, description,
                conditions, recommendation, confidence, evidence_count,
                applicable_domains, timestamp, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pattern.pattern_id,
                pattern.source_domain,
                pattern.pattern_type,
                pattern.description,
                json.dumps(pattern.conditions, default=str),
                pattern.recommendation,
                pattern.confidence,
                pattern.evidence_count,
                json.dumps(pattern.applicable_domains),
                pattern.timestamp,
                json.dumps(pattern.metadata, default=str),
            ),
        )
        conn.commit()

    def get_patterns(
        self,
        domain: Optional[str] = None,
        pattern_type: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = 20,
    ) -> List[PatternRecord]:
        """Get behavioral patterns, optionally filtered."""
        conn = self._get_conn()
        conditions = ["confidence >= ?"]
        params: list = [min_confidence]

        if domain:
            conditions.append("(source_domain = ? OR applicable_domains LIKE ?)")
            params.extend([domain, f'%"{domain}"%'])
        if pattern_type:
            conditions.append("pattern_type = ?")
            params.append(pattern_type)

        where = f"WHERE {' AND '.join(conditions)}"
        query = f"""SELECT * FROM patterns {where}
                    ORDER BY confidence DESC, evidence_count DESC LIMIT ?"""
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        return [self._row_to_pattern(row) for row in rows]

    def delete_patterns(self, pattern_ids: List[str]) -> int:
        """Delete patterns by their IDs. Returns count of deleted rows."""
        if not pattern_ids:
            return 0
        conn = self._get_conn()
        placeholders = ",".join("?" * len(pattern_ids))
        cursor = conn.execute(
            f"DELETE FROM patterns WHERE pattern_id IN ({placeholders})",
            pattern_ids,
        )
        conn.commit()
        deleted = cursor.rowcount
        if deleted:
            logger.info(f"Deleted {deleted} patterns")
        return deleted

    def _row_to_pattern(self, row: sqlite3.Row) -> PatternRecord:
        return PatternRecord(
            pattern_id=row["pattern_id"],
            source_domain=row["source_domain"],
            pattern_type=row["pattern_type"],
            description=row["description"],
            conditions=json.loads(row["conditions"]),
            recommendation=row["recommendation"],
            confidence=row["confidence"],
            evidence_count=row["evidence_count"],
            applicable_domains=json.loads(row["applicable_domains"]),
            timestamp=row["timestamp"],
            metadata=json.loads(row["metadata"]),
        )

    # =========================================================================
    # REFLECTIONS
    # =========================================================================

    def save_reflection(self, reflection: ReflectionRecord) -> None:
        """Save a mid-execution reflection."""
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO reflections
               (reflection_id, episode_id, step, unit_name,
                observation, analysis, adjustment, applied,
                improvement, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                reflection.reflection_id,
                reflection.episode_id,
                reflection.step,
                reflection.unit_name,
                reflection.observation,
                reflection.analysis,
                reflection.adjustment,
                1 if reflection.applied else 0,
                reflection.improvement,
                reflection.timestamp,
            ),
        )
        conn.commit()

    def get_reflections(
        self, episode_id: Optional[str] = None, unit_name: Optional[str] = None, limit: int = 20
    ) -> List[ReflectionRecord]:
        """Get reflections for an episode or unit."""
        conn = self._get_conn()
        conditions = []
        params: list = []

        if episode_id:
            conditions.append("episode_id = ?")
            params.append(episode_id)
        if unit_name:
            conditions.append("unit_name = ?")
            params.append(unit_name)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"SELECT * FROM reflections {where} ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        return [
            ReflectionRecord(
                reflection_id=row["reflection_id"],
                episode_id=row["episode_id"],
                step=row["step"],
                unit_name=row["unit_name"],
                observation=row["observation"],
                analysis=row["analysis"],
                adjustment=row["adjustment"],
                applied=bool(row["applied"]),
                improvement=row["improvement"],
                timestamp=row["timestamp"],
            )
            for row in rows
        ]

    # =========================================================================
    # VALUE ESTIMATES
    # =========================================================================

    def save_value(self, estimate: ValueEstimate) -> None:
        """Save or update a value estimate."""
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO value_estimates
               (state_key, action_key, domain, value, td_error,
                update_count, last_updated, unit_name, lessons, avg_reward)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                estimate.state_key,
                estimate.action_key,
                estimate.domain,
                estimate.value,
                estimate.td_error,
                estimate.update_count,
                estimate.last_updated,
                estimate.unit_name,
                json.dumps(estimate.lessons),
                estimate.avg_reward,
            ),
        )
        conn.commit()

    def get_value(
        self, state_key: str, action_key: str, domain: str = ""
    ) -> Optional[ValueEstimate]:
        """Get value estimate for a (state, action, domain) tuple."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM value_estimates WHERE state_key=? AND action_key=? AND domain=?",
            (state_key, action_key, domain),
        ).fetchone()
        if not row:
            return None
        return self._row_to_value(row)

    def _row_to_value(self, row: sqlite3.Row) -> ValueEstimate:
        """Convert a database row to a ValueEstimate."""
        lessons_raw = row["lessons"] if "lessons" in row.keys() else "[]"
        try:
            lessons = json.loads(lessons_raw) if lessons_raw else []
        except (json.JSONDecodeError, TypeError):
            lessons = []
        return ValueEstimate(
            state_key=row["state_key"],
            action_key=row["action_key"],
            domain=row["domain"],
            value=row["value"],
            td_error=row["td_error"],
            update_count=row["update_count"],
            last_updated=row["last_updated"],
            unit_name=row["unit_name"] if "unit_name" in row.keys() else "",
            lessons=lessons,
            avg_reward=row["avg_reward"] if "avg_reward" in row.keys() else 0.0,
        )

    def get_values_for_domain(
        self, domain: str, unit_name: Optional[str] = None, limit: int = 50
    ) -> List[ValueEstimate]:
        """Get all value estimates for a domain, optionally filtered by unit_name."""
        conn = self._get_conn()
        if unit_name:
            rows = conn.execute(
                """SELECT * FROM value_estimates
                   WHERE domain = ? AND unit_name = ?
                   ORDER BY value DESC LIMIT ?""",
                (domain, unit_name, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT * FROM value_estimates
                   WHERE domain = ?
                   ORDER BY value DESC LIMIT ?""",
                (domain, limit),
            ).fetchall()
        return [self._row_to_value(row) for row in rows]

    def get_lessons(
        self,
        domain: str,
        state_key: Optional[str] = None,
        unit_name: Optional[str] = None,
        limit: int = 20,
    ) -> List[str]:
        """Get accumulated lessons for a domain, optionally filtered."""
        conn = self._get_conn()
        conditions = ["domain = ?"]
        params: list = [domain]

        if state_key:
            conditions.append("state_key = ?")
            params.append(state_key)
        if unit_name:
            conditions.append("unit_name = ?")
            params.append(unit_name)

        where = f"WHERE {' AND '.join(conditions)}"
        rows = conn.execute(
            f"""SELECT lessons FROM value_estimates {where}
                ORDER BY update_count DESC LIMIT ?""",
            params + [limit],
        ).fetchall()

        all_lessons: List[str] = []
        for row in rows:
            try:
                parsed = json.loads(row["lessons"]) if row["lessons"] else []
                all_lessons.extend(parsed)
            except (json.JSONDecodeError, TypeError):
                pass
        # Deduplicate while preserving order
        seen: set = set()
        unique: List[str] = []
        for lesson in all_lessons:
            if lesson not in seen:
                seen.add(lesson)
                unique.append(lesson)
        return unique

    # =========================================================================
    # ANALYTICS
    # =========================================================================

    def get_improvement_report(self, domain: str = "", recent_window: int = 20) -> Dict[str, Any]:
        """Compute improvement report comparing recent vs historical performance."""
        conn = self._get_conn()

        condition = "WHERE domain = ?" if domain else ""
        params: list = [domain] if domain else []

        rows = conn.execute(
            f"SELECT success, quality, timestamp FROM episodes {condition} ORDER BY timestamp DESC",
            params,
        ).fetchall()

        if not rows:
            return {"total": 0, "improving": False}

        recent = rows[:recent_window]
        historical = rows[recent_window:]

        recent_rate = sum(1 for r in recent if r["success"]) / len(recent) if recent else 0
        hist_rate = (
            sum(1 for r in historical if r["success"]) / len(historical) if historical else 0
        )

        recent_quality = sum(r["quality"] for r in recent) / len(recent) if recent else 0
        hist_quality = sum(r["quality"] for r in historical) / len(historical) if historical else 0

        return {
            "total": len(rows),
            "recent_success_rate": round(recent_rate, 3),
            "historical_success_rate": round(hist_rate, 3),
            "trend": round(recent_rate - hist_rate, 3),
            "recent_quality": round(recent_quality, 3),
            "historical_quality": round(hist_quality, 3),
            "quality_trend": round(recent_quality - hist_quality, 3),
            "improving": recent_rate > hist_rate and len(historical) >= 2,
        }

    def get_failure_analysis(
        self, domain: str = "", task_type: str = "", limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get recent failures AND low-quality episodes for learning.

        Returns episodes where success=0 OR quality < 0.6, ordered by
        most recent first.
        """
        conn = self._get_conn()
        conditions = ["(success = 0 OR quality < 0.6)"]
        params: list = []

        if domain:
            conditions.append("domain = ?")
            params.append(domain)
        if task_type:
            conditions.append("task_type = ?")
            params.append(task_type)

        where = f"WHERE {' AND '.join(conditions)}"
        rows = conn.execute(
            f"""SELECT episode_id, unit_name, task_type, error_type, error_message,
                       context, action, outcome, quality, timestamp
                FROM episodes {where}
                ORDER BY timestamp DESC LIMIT ?""",
            params + [limit],
        ).fetchall()

        results = []
        for row in rows:
            quality = row["quality"] if row["quality"] is not None else 1.0
            error_type = row["error_type"] or ("low_quality" if quality < 0.6 else "")
            outcome = json.loads(row["outcome"])

            # Build specific failure description from structural analysis
            error_message = row["error_message"] or ""
            if not error_message and quality < 0.6:
                missing = []
                if not outcome.get("has_code"):
                    missing.append("no code blocks")
                elif outcome.get("code_block_count", 0) < 3:
                    missing.append(
                        f"only {outcome.get('code_block_count', 0)} code blocks (need 3+)"
                    )
                if not outcome.get("has_math"):
                    missing.append("no math/formulas")
                if not outcome.get("has_headings"):
                    missing.append("no section headings")
                if outcome.get("word_count", 0) < 500:
                    missing.append(f"too short ({outcome.get('word_count', 0)} words)")
                if outcome.get("goal_coverage", 1.0) < 0.5:
                    missing.append(f"low goal coverage ({outcome.get('goal_coverage', 0):.0%})")
                if missing:
                    error_message = f"Quality {quality:.2f} — missing: {', '.join(missing)}"
                else:
                    error_message = f"Quality was {quality:.2f} — below threshold"

            results.append(
                {
                    "episode_id": row["episode_id"],
                    "unit_name": row["unit_name"],
                    "task_type": row["task_type"],
                    "error_type": error_type,
                    "error_message": error_message,
                    "context": json.loads(row["context"]),
                    "action": json.loads(row["action"]),
                    "outcome": outcome,
                    "timestamp": row["timestamp"],
                }
            )
        return results

    def close(self) -> None:
        """Close the database connection."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
