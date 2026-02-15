"""
Session Manager - Complete Output & Persistence Management

A-Team Design Principles:
- Single source of truth for all output paths
- Auto-creates session folders
- Auto-loads previous state if present
- Persists memories, Q-tables, brain state, task lists
- Generates beautified logs automatically
- No hardcoded paths anywhere

Author: A-Team
Date: Dec 29, 2025
"""

import json
import logging
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Manages all output and persistence for a ReVal session.

    Features:
    - Per-session timestamped folders
    - Auto-load from outputs/latest/
    - Persistent memories, Q-tables, brain state
    - Session task list tracking
    - Beautified log generation
    - Auto-cleanup old runs

    Directory Structure:
        outputs/
        ├── latest/ → symlink to most recent run
        ├── run_20251229_220000/
        │   ├── jotty_state/
        │   │   ├── memories/
        │   │   │   ├── shared_memory.json
        │   │   │   └── agent_memories/
        │   │   │       ├── SQLGenerator.json
        │   │   │       └── BusinessTermResolver.json
        │   │   ├── q_tables/
        │   │   │   └── q_predictor.json
        │   │   ├── brain_state/
        │   │   │   └── consolidated.json
        │   │   └── markovian_todos/
        │   │       └── session_todo.md
        │   ├── logs/
        │   │   ├── raw/
        │   │   │   └── test_run_TIMESTAMP.log
        │   │   └── beautified/
        │   │       └── BEAUTIFUL_TIMESTAMP.log
        │   ├── results/
        │   │   └── query_results.json
        │   └── config_snapshot.json
    """

    def __init__(self, config: "SwarmConfig") -> None:
        """
        Initialize session manager.

        Args:
            config: SwarmConfig with persistence settings
        """
        self.config = config
        self.base_dir = Path(config.output_base_dir).expanduser().resolve()

        # Create or load session
        if config.create_run_folder:
            self.session_dir = self._create_session_folder()
        else:
            self.session_dir = self.base_dir

        # Setup directory structure
        self._setup_directories()

        # Save config snapshot
        self._save_config_snapshot()

        logger.info(f" Session initialized: {self.session_dir}")

    def _create_session_folder(self) -> Path:
        """Create timestamped session folder and update 'latest' symlink."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = self.base_dir / f"run_{timestamp}"
        session_dir.mkdir(parents=True, exist_ok=True)

        # Update 'latest' symlink
        latest_link = self.base_dir / "latest"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(session_dir.name)

        logger.info(f" Created session folder: {session_dir.name}")
        return session_dir

    def _setup_directories(self) -> Any:
        """Create all necessary subdirectories."""
        dirs = [
            self.session_dir / "jotty_state" / "memories" / "agent_memories",
            self.session_dir / "jotty_state" / "q_tables",
            self.session_dir / "jotty_state" / "brain_state",
            self.session_dir / "jotty_state" / "markovian_todos",
            self.session_dir / "logs" / "raw",
            self.session_dir / "logs" / "beautified",
            self.session_dir / "logs" / "debug",
            self.session_dir / "results",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def _save_config_snapshot(self) -> Any:
        """Save config used for this session."""
        config_file = self.session_dir / "config_snapshot.json"
        with open(config_file, "w") as f:
            json.dump(self.config.to_flat_dict(), f, indent=2, default=str)

    # =========================================================================
    # AUTO-LOADING
    # =========================================================================

    def load_previous_state(self) -> Optional[Dict[str, Any]]:
        """
        Auto-load state from outputs/latest/ if it exists.

        Returns:
            Dict with loaded state or None if no previous state
        """
        if not self.config.auto_load_on_start:
            logger.info("⏭ Auto-load disabled")
            return None

        latest_dir = self.base_dir / "latest"
        if not latest_dir.exists():
            logger.info(" No previous state found")
            return None

        try:
            state = {}

            # Load memories
            if self.config.persist_memories:
                state["memories"] = self._load_memories(latest_dir)

            # Load Q-tables
            if self.config.persist_q_tables:
                state["q_tables"] = self._load_q_tables(latest_dir)

            # Load brain state
            if self.config.persist_brain_state:
                state["brain_state"] = self._load_brain_state(latest_dir)

            # Load task lists
            if self.config.persist_todos:
                state["todos"] = self._load_todos(latest_dir)

            logger.info(f" Loaded previous state from {latest_dir}")
            return state

        except Exception as e:
            logger.warning(f" Failed to load previous state: {e}")
            return None

    def _load_memories(self, source_dir: Path) -> Dict[str, Any]:
        """Load all memory files."""
        memories = {}
        mem_dir = source_dir / "jotty_state" / "memories"

        # Load shared memory
        shared_file = mem_dir / "shared_memory.json"
        if shared_file.exists():
            with open(shared_file) as f:
                memories["shared"] = json.load(f)

        # Load agent memories
        agent_dir = mem_dir / "agent_memories"
        if agent_dir.exists():
            memories["agents"] = {}
            for mem_file in agent_dir.glob("*.json"):
                agent_name = mem_file.stem
                with open(mem_file) as f:
                    memories["agents"][agent_name] = json.load(f)

        logger.info(f" Loaded {len(memories.get('agents', {}))} agent memories")
        return memories

    def _load_q_tables(self, source_dir: Path) -> Dict[str, Any]:
        """Load Q-table files."""
        q_tables = {}
        q_dir = source_dir / "jotty_state" / "q_tables"

        if q_dir.exists():
            for q_file in q_dir.glob("*.json"):
                with open(q_file) as f:
                    q_tables[q_file.stem] = json.load(f)

        logger.info(f" Loaded {len(q_tables)} Q-tables")
        return q_tables

    def _load_brain_state(self, source_dir: Path) -> Optional[Dict[str, Any]]:
        """Load brain consolidation state."""
        brain_file = source_dir / "jotty_state" / "brain_state" / "consolidated.json"
        if brain_file.exists():
            with open(brain_file) as f:
                state = json.load(f)
            logger.info(f" Loaded brain state")
            return state
        return None

    def _load_todos(self, source_dir: Path) -> Optional[str]:
        """Load session task list markdown."""
        todo_file = source_dir / "jotty_state" / "markovian_todos" / "session_todo.md"
        if todo_file.exists():
            with open(todo_file) as f:
                content = f.read()
            logger.info(f" Loaded session task list")
            return content
        return None

    # =========================================================================
    # SAVING
    # =========================================================================

    def save_state(self, state: Dict[str, Any]) -> None:
        """
        Save complete ReVal state.

        Args:
            state: Dict containing memories, q_tables, brain_state, etc.
        """
        try:
            if self.config.persist_memories and "memories" in state:
                self._save_memories(state["memories"])

            if self.config.persist_q_tables and "q_tables" in state:
                self._save_q_tables(state["q_tables"])

            if self.config.persist_brain_state and "brain_state" in state:
                self._save_brain_state(state["brain_state"])

            if self.config.persist_todos and "todos" in state:
                self._save_todos(state["todos"])

            logger.info(f" Saved state to {self.session_dir}")

        except Exception as e:
            logger.error(f" Failed to save state: {e}", exc_info=True)

    def _save_memories(self, memories: Dict[str, Any]) -> Any:
        """Save all memory files."""
        mem_dir = self.session_dir / "jotty_state" / "memories"

        # Save shared memory
        if "shared" in memories:
            shared_file = mem_dir / "shared_memory.json"
            with open(shared_file, "w") as f:
                json.dump(memories["shared"], f, indent=2, default=str)

        # Save agent memories
        if "agents" in memories:
            agent_dir = mem_dir / "agent_memories"
            for agent_name, mem_data in memories["agents"].items():
                mem_file = agent_dir / f"{agent_name}.json"
                with open(mem_file, "w") as f:
                    json.dump(mem_data, f, indent=2, default=str)

    def _save_q_tables(self, q_tables: Dict[str, Any]) -> Any:
        """Save Q-table files."""
        q_dir = self.session_dir / "jotty_state" / "q_tables"
        for name, data in q_tables.items():
            q_file = q_dir / f"{name}.json"
            with open(q_file, "w") as f:
                json.dump(data, f, indent=2, default=str)

    def _save_brain_state(self, brain_state: Dict[str, Any]) -> Any:
        """Save brain consolidation state."""
        brain_file = self.session_dir / "jotty_state" / "brain_state" / "consolidated.json"
        with open(brain_file, "w") as f:
            json.dump(brain_state, f, indent=2, default=str)

    def _save_todos(self, todos: str) -> Any:
        """Save session task list markdown."""
        todo_file = self.session_dir / "jotty_state" / "markovian_todos" / "session_todo.md"
        with open(todo_file, "w") as f:
            f.write(todos)

    # =========================================================================
    # RESULTS & LOGS
    # =========================================================================

    def save_results(self, results: Dict[str, Any]) -> None:
        """Save query execution results."""
        result_file = self.session_dir / "results" / "query_results.json"
        with open(result_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f" Saved results to {result_file}")

    def get_log_path(self, log_type: str = "raw") -> Path:
        """Get path for log file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.session_dir / "logs" / log_type / f"{log_type}_{timestamp}.log"

    def get_beautified_log_dir(self) -> Path:
        """Get directory for beautified logs."""
        return self.session_dir / "logs" / "beautified"

    def get_debug_log_dir(self) -> Path:
        """Get directory for debug logs."""
        return self.session_dir / "logs" / "debug"

    # =========================================================================
    # CLEANUP
    # =========================================================================

    def cleanup_old_runs(self) -> None:
        """Remove old run folders beyond max_runs_to_keep."""
        if not self.config.max_runs_to_keep:
            return

        # Get all run folders
        run_folders = sorted(
            [d for d in self.base_dir.glob("run_*") if d.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        # Keep only the newest N
        to_remove = run_folders[self.config.max_runs_to_keep :]
        for folder in to_remove:
            shutil.rmtree(folder)
            logger.info(f" Removed old run: {folder.name}")

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def get_session_info(self) -> Dict[str, Any]:
        """Get information about current session."""
        return {
            "session_dir": str(self.session_dir),
            "session_name": self.session_dir.name,
            "created": datetime.fromtimestamp(self.session_dir.stat().st_mtime).isoformat(),
            "has_memories": (self.session_dir / "jotty_state" / "memories").exists(),
            "has_q_tables": (self.session_dir / "jotty_state" / "q_tables").exists(),
            "has_brain_state": (
                self.session_dir / "jotty_state" / "brain_state" / "consolidated.json"
            ).exists(),
            "has_todos": (
                self.session_dir / "jotty_state" / "markovian_todos" / "session_todo.md"
            ).exists(),
        }
