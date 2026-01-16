"""
MongoDB Backend for Jotty HierarchicalMemory
============================================

Extends Jotty's memory persistence layer with MongoDB backend.
Follows DRY principle by extending existing MemoryPersistence interface.

**Usage**:
```python
from Jotty.core.memory import HierarchicalMemory
from Jotty.core.memory.mongodb_backend import MongoDBMemoryBackend

# Create memory with MongoDB backend
memory = HierarchicalMemory(agent_name="supervisor")
backend = MongoDBMemoryBackend(memory, mongo_uri="mongodb://...")

# Use Jotty's standard API
memory.store(content="...", level=MemoryLevel.EPISODIC, ...)
backend.save()  # Persist to MongoDB
```
"""

import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pymongo import MongoClient, ASCENDING, DESCENDING, TEXT

from .cortex import HierarchicalMemory
from ..foundation.data_structures import MemoryLevel, MemoryEntry, GoalValue

logger = logging.getLogger(__name__)


class MongoDBMemoryBackend:
    """
    MongoDB persistence backend for Jotty's HierarchicalMemory.
    
    Extends MemoryPersistence pattern but uses MongoDB instead of files.
    Follows DRY: Uses Jotty's existing MemoryEntry and MemoryLevel structures.
    """
    
    def __init__(
        self,
        memory: HierarchicalMemory,
        mongo_uri: Optional[str] = None,
        agent_name: Optional[str] = None
    ):
        """
        Initialize MongoDB backend for HierarchicalMemory.
        
        Args:
            memory: HierarchicalMemory instance to persist
            mongo_uri: MongoDB connection URI (default: from env MONGODB_URI)
            agent_name: Agent name (default: from memory.agent_name)
        """
        self.memory = memory
        self.agent_name = agent_name or memory.agent_name
        
        # Connect to MongoDB
        mongo_uri = mongo_uri or os.getenv(
            "MONGODB_URI",
            "mongodb://justjot-mongo:27017/justjot"
        )
        
        self.client = MongoClient(mongo_uri)
        
        # Extract database name from URI
        if "/" in mongo_uri.split("@")[-1]:
            db_name = mongo_uri.split("/")[-1].split("?")[0]
        else:
            db_name = "justjot"
        
        self.db = self.client[db_name]
        self.memories_collection = self.db.memories
        
        # Create indexes
        self._ensure_indexes()
        
        logger.info(f"✅ MongoDB backend initialized for agent '{self.agent_name}'")
    
    def _ensure_indexes(self):
        """Create necessary indexes for efficient queries."""
        try:
            # Compound index for agent + level + time
            self.memories_collection.create_index([
                ("agent_name", ASCENDING),
                ("level", ASCENDING),
                ("timestamp", DESCENDING)
            ], name="agent_level_time")
            
            # Index for goal-based retrieval
            self.memories_collection.create_index([
                ("agent_name", ASCENDING),
                ("goal", ASCENDING)
            ], name="agent_goal")
            
            # Text index for content search
            self.memories_collection.create_index([
                ("content", TEXT)
            ], name="content_text")
            
            # TTL index for automatic cleanup of old episodic memories (30 days)
            self.memories_collection.create_index(
                "timestamp",
                expireAfterSeconds=30*24*60*60,
                partialFilterExpression={"level": "EPISODIC"},
                name="episodic_ttl"
            )
            
        except Exception as e:
            logger.warning(f"Index creation warning: {e}")
    
    def save(self) -> bool:
        """
        Save HierarchicalMemory to MongoDB.
        
        Converts Jotty's MemoryEntry objects to MongoDB documents.
        Follows DRY: Uses existing MemoryEntry structure.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            saved_count = 0
            
            for level in MemoryLevel:
                memories = self.memory.memories[level]
                
                for entry in memories.values():
                    # Convert MemoryEntry to MongoDB document
                    doc = self._entry_to_document(entry, level)
                    
                    # Upsert (update if exists, insert if not)
                    self.memories_collection.update_one(
                        {"agent_name": self.agent_name, "key": entry.key},
                        {"$set": doc},
                        upsert=True
                    )
                    saved_count += 1
                
                logger.debug(f"Saved {len(memories)} {level.value} memories to MongoDB")
            
            logger.info(f"✅ Saved {saved_count} total memories to MongoDB")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save memory to MongoDB: {e}")
            return False
    
    def load(self) -> bool:
        """
        Load HierarchicalMemory from MongoDB.
        
        Converts MongoDB documents back to Jotty's MemoryEntry objects.
        Follows DRY: Uses existing MemoryEntry structure.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            loaded_count = 0
            
            for level in MemoryLevel:
                # Query MongoDB for this agent and level
                cursor = self.memories_collection.find({
                    "agent_name": self.agent_name,
                    "level": level.value.upper()
                })
                
                for doc in cursor:
                    # Convert MongoDB document to MemoryEntry
                    entry = self._document_to_entry(doc, level)
                    
                    # Store in memory
                    self.memory.memories[level][entry.key] = entry
                    loaded_count += 1
                
                logger.debug(f"Loaded {loaded_count} {level.value} memories from MongoDB")
            
            logger.info(f"✅ Loaded {loaded_count} total memories from MongoDB")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load memory from MongoDB: {e}")
            return False
    
    def _entry_to_document(self, entry: MemoryEntry, level: MemoryLevel) -> Dict[str, Any]:
        """Convert Jotty MemoryEntry to MongoDB document."""
        # Convert goal_values
        goal_values_dict = {}
        for goal, gv in entry.goal_values.items():
            goal_values_dict[goal] = {
                "value": gv.value,
                "access_count": gv.access_count
            }
        
        # Build document matching Jotty's structure
        doc = {
            "agent_name": self.agent_name,
            "key": entry.key,
            "level": level.value.upper(),
            "content": entry.content,
            "context": entry.context,
            "goal": entry.context.get("goal", f"plan_{entry.context.get('category', 'general')}"),
            "value": entry.default_value,
            "timestamp": entry.created_at or datetime.utcnow(),
            "access_count": entry.access_count,
            "last_accessed": entry.last_accessed or datetime.utcnow(),
            "created_at": entry.created_at or datetime.utcnow(),
            "goal_values": goal_values_dict,
            "token_count": entry.token_count,
            "causal_links": entry.causal_links,
            "content_hash": entry.content_hash,
            "similar_entries": entry.similar_entries,
            "source_episode": entry.source_episode,
            "source_agent": entry.source_agent,
            "is_protected": entry.is_protected,
            "protection_reason": entry.protection_reason
        }
        
        return doc
    
    def _document_to_entry(self, doc: Dict[str, Any], level: MemoryLevel) -> MemoryEntry:
        """Convert MongoDB document to Jotty MemoryEntry."""
        # Reconstruct goal_values
        goal_values = {}
        for goal, gv_data in doc.get("goal_values", {}).items():
            goal_value = GoalValue(
                value=gv_data.get("value", 0.5),
                access_count=gv_data.get("access_count", 0)
            )
            goal_values[goal] = goal_value
        
        # Create MemoryEntry using Jotty's structure
        entry = MemoryEntry(
            key=doc["key"],
            content=doc["content"],
            level=level,
            context=doc.get("context", {}),
            token_count=doc.get("token_count", 0),
            default_value=doc.get("value", 0.5),
            goal_values=goal_values,
            causal_links=doc.get("causal_links", []),
            content_hash=doc.get("content_hash", ""),
            similar_entries=doc.get("similar_entries", []),
            source_episode=doc.get("source_episode", 0),
            source_agent=doc.get("source_agent", ""),
            is_protected=doc.get("is_protected", False),
            protection_reason=doc.get("protection_reason", "")
        )
        
        # Set timestamps
        if doc.get("created_at"):
            entry.created_at = doc["created_at"] if isinstance(doc["created_at"], datetime) else datetime.fromisoformat(str(doc["created_at"]))
        if doc.get("last_accessed"):
            entry.last_accessed = doc["last_accessed"] if isinstance(doc["last_accessed"], datetime) else datetime.fromisoformat(str(doc["last_accessed"]))
        
        entry.access_count = doc.get("access_count", 0)
        entry.ucb_visits = doc.get("ucb_visits", 0)
        
        return entry
    
    def retrieve_by_goal(
        self,
        goal: str,
        levels: Optional[List[MemoryLevel]] = None,
        max_memories: int = 10
    ) -> List[MemoryEntry]:
        """
        Retrieve memories by goal (for planning context).
        
        Uses Jotty's MemoryEntry structure - no duplication.
        
        Args:
            goal: Goal to retrieve memories for (e.g., "plan_UI")
            levels: Memory levels to retrieve (default: all)
            max_memories: Maximum memories per level
        
        Returns:
            List of MemoryEntry objects
        """
        if levels is None:
            levels = list(MemoryLevel)
        
        entries = []
        
        for level in levels:
            cursor = self.memories_collection.find({
                "agent_name": self.agent_name,
                "level": level.value.upper(),
                "goal": goal
            }).sort("timestamp", DESCENDING).limit(max_memories)
            
            for doc in cursor:
                entry = self._document_to_entry(doc, level)
                entries.append(entry)
        
        return entries
    
    def store_task_outcome(
        self,
        task_id: str,
        task_spec: str,
        outcome: Dict[str, Any]
    ) -> bool:
        """
        Store task outcome using Jotty's memory API.
        
        Uses Jotty's HierarchicalMemory.store() - no duplication.
        
        Args:
            task_id: Task ID
            task_spec: Task specification
            outcome: Outcome dict with status, duration, errors, category
        
        Returns:
            True if stored successfully
        """
        try:
            status = outcome.get('status', 'unknown')
            category = outcome.get('category', 'general')
            duration = outcome.get('duration', 0)
            errors = outcome.get('errors', [])
            
            # Build content using Jotty's format
            content = f"Task {task_id}: {task_spec}\nOutcome: {status}\nDuration: {duration}s"
            if errors:
                content += f"\nErrors: {str(errors[:3])}"
            
            # Use Jotty's memory API - EPISODIC level
            self.memory.store(
                content=content,
                level=MemoryLevel.EPISODIC,
                context={
                    'task_id': task_id,
                    'category': category,
                    'outcome': status,
                    'duration': duration,
                    'retry_count': outcome.get('retry_count', 0),
                    'error_messages': errors,
                    'task_spec': task_spec,
                    'goal': f"plan_{category}"
                },
                goal=f"plan_{category}",
                initial_value=0.7 if status == "completed" else 0.3
            )
            
            # Persist to MongoDB
            self.save()
            
            logger.info(f"✅ Stored task outcome: {task_id} ({status})")
            
            # Extract patterns if successful (using Jotty's consolidation)
            if status == "completed":
                self._extract_success_pattern(task_spec, category, task_id)
            
            # Analyze failures
            elif status == "failed" and errors:
                self._analyze_failure_pattern(task_spec, category, errors, task_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store task outcome: {e}")
            return False
    
    def retrieve_planning_context(
        self,
        task_description: str,
        category: str = "general",
        max_memories: int = 10
    ) -> Dict[str, Any]:
        """
        Retrieve planning context using Jotty's memory API.
        
        Uses Jotty's retrieval - no duplication.
        
        Args:
            task_description: Task description
            category: Task category
            max_memories: Max memories per level
        
        Returns:
            Dict with patterns, procedures, similar_tasks
        """
        goal = f"plan_{category if category and category.strip() else 'general'}"
        
        # Use Jotty's retrieval API via backend (loads from MongoDB first)
        # First ensure memory is synced from MongoDB
        self.load()
        
        # Then use Jotty's retrieve method
        # Jotty's retrieve needs budget_tokens, estimate from max_memories
        budget_tokens = max_memories * 200  # ~200 tokens per memory
        
        memories = self.memory.retrieve(
            query=task_description,
            goal=goal,
            budget_tokens=budget_tokens,
            levels=[MemoryLevel.EPISODIC, MemoryLevel.SEMANTIC, MemoryLevel.PROCEDURAL]
        )
        
        # Format for supervisor compatibility
        context = {
            'patterns': [],
            'procedures': [],
            'similar_tasks': []
        }
        
        for entry in memories:
            if entry.level == MemoryLevel.SEMANTIC:
                context['patterns'].append({
                    'content': entry.content,
                    'value': entry.default_value,
                    'timestamp': entry.created_at.isoformat() if entry.created_at else None
                })
            elif entry.level == MemoryLevel.PROCEDURAL:
                context['procedures'].append({
                    'content': entry.content,
                    'value': entry.default_value,
                    'timestamp': entry.created_at.isoformat() if entry.created_at else None
                })
            elif entry.level == MemoryLevel.EPISODIC:
                ctx = entry.context
                context['similar_tasks'].append({
                    'description': ctx.get('task_spec', entry.content[:100]),
                    'outcome': ctx.get('outcome', 'unknown'),
                    'duration': ctx.get('duration', 0),
                    'timestamp': entry.created_at.isoformat() if entry.created_at else None
                })
        
        return context
    
    def _extract_success_pattern(self, task_spec: str, category: str, task_id: str):
        """Extract success pattern using Jotty's memory API."""
        try:
            # Use Jotty's memory to store pattern
            pattern = f"Successfully executed: {task_spec[:80]}... in category '{category}'"
            
            self.memory.store(
                content=pattern,
                level=MemoryLevel.SEMANTIC,
                context={
                    'extracted_from_task': True,
                    'category': category,
                    'success_rate': 1.0,
                    'task_id': task_id
                },
                goal=f"plan_{category}",
                initial_value=0.8
            )
            
            self.save()
            logger.info(f"   ✅ Extracted success pattern for category '{category}'")
            
        except Exception as e:
            logger.warning(f"   ⚠️  Pattern extraction failed: {e}")
    
    def _analyze_failure_pattern(self, task_spec: str, category: str, errors: List[str], task_id: str):
        """Analyze failure pattern using Jotty's memory API."""
        try:
            if not errors:
                return
            
            # Categorize errors
            error_type = self._categorize_error(errors[0])
            
            # Store failure pattern using Jotty's memory
            pattern = f"{category} tasks fail due to: {error_type}"
            
            self.memory.store(
                content=pattern,
                level=MemoryLevel.SEMANTIC,
                context={
                    'failure_pattern': True,
                    'category': category,
                    'error_type': error_type,
                    'task_id': task_id,
                    'prevention_tip': self._get_prevention_tip(error_type)
                },
                goal=f"plan_{category}",
                initial_value=0.6
            )
            
            self.save()
            logger.info(f"   ⚠️  Extracted failure pattern: {error_type} in {category}")
            
        except Exception as e:
            logger.warning(f"   ⚠️  Failure pattern analysis failed: {e}")
    
    def _categorize_error(self, error: str) -> str:
        """Categorize error message."""
        error_lower = error.lower()
        
        if 'typescript' in error_lower or 'type' in error_lower:
            return "TypeScript errors"
        elif 'build' in error_lower or 'compile' in error_lower:
            return "Build/Compilation errors"
        elif 'test' in error_lower:
            return "Test failures"
        elif 'import' in error_lower or 'module' in error_lower:
            return "Import/Module errors"
        else:
            return "Other errors"
    
    def _get_prevention_tip(self, error_type: str) -> str:
        """Get prevention tip for error type."""
        tips = {
            "TypeScript errors": "Enable TypeScript strict mode and fix type errors before committing",
            "Build/Compilation errors": "Run build locally before pushing changes",
            "Test failures": "Run tests locally and fix failing tests before committing"
        }
        return tips.get(error_type, "Review error logs and fix underlying issue")


def enable_mongodb_memory(
    memory: HierarchicalMemory,
    mongo_uri: Optional[str] = None
) -> MongoDBMemoryBackend:
    """
    Enable MongoDB persistence for HierarchicalMemory.
    
    Similar to enable_memory_persistence() but uses MongoDB.
    Follows DRY: Same pattern as file-based persistence.
    
    Args:
        memory: HierarchicalMemory instance
        mongo_uri: MongoDB URI (default: from env)
    
    Returns:
        MongoDBMemoryBackend instance
    """
    backend = MongoDBMemoryBackend(memory, mongo_uri=mongo_uri)
    
    # Try to load existing memory
    backend.load()
    
    return backend
