"""Persistence configuration — state storage, auto-save/load, retention."""

from dataclasses import dataclass


@dataclass
class PersistenceConfig:
    """Storage and persistence settings."""
    output_base_dir: str = "./outputs"
    create_run_folder: bool = True
    auto_save_interval: int = 3
    auto_load_on_start: bool = True
    save_interval: int = 1
    persist_memories: bool = True
    persist_q_tables: bool = True
    persist_brain_state: bool = True
    persist_todos: bool = True
    persist_agent_outputs: bool = True
    storage_format: str = "json"
    compress_large_files: bool = True
    max_runs_to_keep: int = 10
    enable_backups: bool = True
    backup_interval: int = 100
    max_backups: int = 10
    base_path: str = "~/.jotty"
    auto_load: bool = True
    auto_save: bool = True
