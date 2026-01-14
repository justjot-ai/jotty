"""
Memory Persistence Layer

Provides persistence for HierarchicalMemory to disk.
Memory system stores data in-memory by default, this adds file-based persistence.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from .cortex import HierarchicalMemory
from ..foundation.data_structures import MemoryLevel, MemoryEntry, JottyConfig

logger = logging.getLogger(__name__)


class MemoryPersistence:
    """Handles persistence of HierarchicalMemory to disk."""
    
    def __init__(self, memory: HierarchicalMemory, persistence_dir: Path):
        """
        Initialize memory persistence.
        
        Args:
            memory: HierarchicalMemory instance to persist
            persistence_dir: Directory to store memory files
        """
        self.memory = memory
        self.persistence_dir = Path(persistence_dir)
        self.persistence_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths for each memory level
        self.level_files = {
            MemoryLevel.EPISODIC: self.persistence_dir / "episodic_memories.json",
            MemoryLevel.SEMANTIC: self.persistence_dir / "semantic_memories.json",
            MemoryLevel.PROCEDURAL: self.persistence_dir / "procedural_memories.json",
            MemoryLevel.META: self.persistence_dir / "meta_memories.json",
            MemoryLevel.CAUSAL: self.persistence_dir / "causal_memories.json"
        }
    
    def save(self) -> bool:
        """
        Save memory to disk.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            for level, file_path in self.level_files.items():
                memories = self.memory.memories[level]
                
                # Convert MemoryEntry objects to dicts
                memories_data = []
                for entry in memories.values():
                    entry_dict = {
                        "key": entry.key,
                        "content": entry.content,
                        "level": entry.level.value,
                        "context": entry.context,
                        "created_at": entry.created_at.isoformat() if entry.created_at else None,
                        "last_accessed": entry.last_accessed.isoformat() if entry.last_accessed else None,
                        "access_count": entry.access_count,
                        "ucb_visits": entry.ucb_visits,
                        "token_count": entry.token_count,
                        "default_value": entry.default_value,
                        "goal_values": {
                            goal: {
                                "value": gv.value,
                                "access_count": gv.access_count
                            }
                            for goal, gv in entry.goal_values.items()
                        },
                        "causal_links": entry.causal_links,
                        "content_hash": entry.content_hash,
                        "similar_entries": entry.similar_entries,
                        "source_episode": entry.source_episode,
                        "source_agent": entry.source_agent,
                        "is_protected": entry.is_protected,
                        "protection_reason": entry.protection_reason
                    }
                    memories_data.append(entry_dict)
                
                # Save to file
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(memories_data, f, indent=2, ensure_ascii=False, default=str)
                
                logger.info(f"Saved {len(memories_data)} {level.value} memories to {file_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")
            return False
    
    def load(self) -> bool:
        """
        Load memory from disk.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            from ..foundation.data_structures import GoalValue
            
            for level, file_path in self.level_files.items():
                if not file_path.exists():
                    continue
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    memories_data = json.load(f)
                
                # Convert dicts back to MemoryEntry objects
                for entry_dict in memories_data:
                    # Reconstruct goal_values
                    goal_values = {}
                    for goal, gv_data in entry_dict.get('goal_values', {}).items():
                        goal_value = GoalValue(
                            value=gv_data.get('value', 0.5),
                            access_count=gv_data.get('access_count', 0)
                        )
                        goal_values[goal] = goal_value
                    
                    # Create MemoryEntry
                    entry = MemoryEntry(
                        key=entry_dict['key'],
                        content=entry_dict['content'],
                        level=MemoryLevel(entry_dict['level']),
                        context=entry_dict['context'],
                        token_count=entry_dict.get('token_count', 0),
                        default_value=entry_dict.get('default_value', 0.5),
                        goal_values=goal_values,
                        causal_links=entry_dict.get('causal_links', []),
                        content_hash=entry_dict.get('content_hash', ''),
                        similar_entries=entry_dict.get('similar_entries', []),
                        source_episode=entry_dict.get('source_episode', 0),
                        source_agent=entry_dict.get('source_agent', ''),
                        is_protected=entry_dict.get('is_protected', False),
                        protection_reason=entry_dict.get('protection_reason', '')
                    )
                    
                    # Set timestamps
                    if entry_dict.get('created_at'):
                        entry.created_at = datetime.fromisoformat(entry_dict['created_at'])
                    if entry_dict.get('last_accessed'):
                        entry.last_accessed = datetime.fromisoformat(entry_dict['last_accessed'])
                    
                    entry.access_count = entry_dict.get('access_count', 0)
                    entry.ucb_visits = entry_dict.get('ucb_visits', 0)
                    
                    # Store in memory
                    self.memory.memories[level][entry.key] = entry
                
                logger.info(f"Loaded {len(memories_data)} {level.value} memories from {file_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load memory: {e}")
            return False


def enable_memory_persistence(
    memory: HierarchicalMemory,
    persistence_dir: Optional[Path] = None
) -> MemoryPersistence:
    """
    Enable persistence for a HierarchicalMemory instance.
    
    Args:
        memory: HierarchicalMemory instance
        persistence_dir: Directory for persistence files (default: ./memory_data/{agent_name})
    
    Returns:
        MemoryPersistence instance
    """
    if persistence_dir is None:
        persistence_dir = Path(f"./memory_data/{memory.agent_name}")
    
    persistence = MemoryPersistence(memory, persistence_dir)
    
    # Try to load existing memory
    persistence.load()
    
    return persistence
