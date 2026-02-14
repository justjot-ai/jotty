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

    def __post_init__(self):
        # Positive integer fields
        _pos_int_fields = {
            'auto_save_interval': self.auto_save_interval,
            'save_interval': self.save_interval,
            'max_runs_to_keep': self.max_runs_to_keep,
            'backup_interval': self.backup_interval,
            'max_backups': self.max_backups,
        }
        for name, val in _pos_int_fields.items():
            if val < 1:
                raise ValueError(f"{name} must be >= 1, got {val}")

        # Storage format validation
        valid_formats = {"json", "sqlite", "msgpack"}
        if self.storage_format not in valid_formats:
            raise ValueError(
                f"storage_format must be one of {valid_formats}, got '{self.storage_format}'"
            )
