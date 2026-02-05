"""
JustJot Memory Sync Adapter
============================

Bidirectional sync between Jotty's Cortex (HierarchicalMemory) and JustJot's MemoryGraph.

Architecture:
    ┌─────────────────────┐          ┌─────────────────────┐
    │  Jotty Cortex       │          │  JustJot MemoryGraph│
    │  (5-level Python)   │◄────────►│  (4-type TypeScript)│
    └─────────────────────┘   sync   └─────────────────────┘

Schema Mapping:
    | Jotty Cortex          | JustJot MemoryGraph    |
    |-----------------------|------------------------|
    | MemoryEntry.content   | Memory.content         |
    | MemoryEntry.context   | Memory.context         |
    | MemoryEntry.level     | Memory.cognitive_type  |
    | MemoryEntry.goal_values | Memory.importance    |
    | MemoryEntry.timestamp | Memory.created_at      |
    | agent_name            | entity_id + type='agent'|

Level Mapping:
    Jotty 5-level → JustJot 4-type
    - EPISODIC   → episodic
    - SEMANTIC   → semantic
    - PROCEDURAL → procedural
    - META       → semantic (merged)
    - CAUSAL     → semantic (merged)

    JustJot 4-type → Jotty 5-level
    - episodic   → EPISODIC
    - semantic   → SEMANTIC
    - procedural → PROCEDURAL
    - working    → EPISODIC (temporary)
"""

import logging
import json
import hashlib
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Configurable URLs via environment variables
JUSTJOT_API_URL = os.environ.get('JUSTJOT_API_URL', 'http://localhost:3000')

# Import Jotty types
try:
    from ..foundation.data_structures import MemoryEntry, MemoryLevel, GoalValue
except ImportError:
    MemoryEntry = None
    MemoryLevel = None
    GoalValue = None


@dataclass
class JustJotMemory:
    """
    Representation of JustJot Memory format (TypeScript-compatible).

    This mirrors the TypeScript Memory interface for interoperability.
    """
    memory_id: str
    entity_id: str
    entity_type: str = "agent"  # Default for Jotty agents

    content: str = ""
    content_hash: str = ""

    cognitive_type: str = "episodic"  # 'episodic', 'semantic', 'procedural', 'working'
    cognitive_confidence: float = 1.0
    cognitive_reasoning: str = ""

    # Importance
    importance_score: float = 0.5
    importance_factors: Dict[str, float] = field(default_factory=dict)

    # Context
    context: Dict[str, Any] = field(default_factory=dict)

    # Lifecycle
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'memory_id': self.memory_id,
            'entity_id': self.entity_id,
            'entity_type': self.entity_type,
            'content': self.content,
            'content_hash': self.content_hash,
            'cognitive_type': self.cognitive_type,
            'cognitive_confidence': self.cognitive_confidence,
            'cognitive_reasoning': self.cognitive_reasoning,
            'importance': {
                'score': self.importance_score,
                'factors': self.importance_factors,
                'decay_rate': 1.0,
                'last_updated': self.updated_at.isoformat()
            },
            'context': self.context,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'accessed_at': self.accessed_at.isoformat(),
            'access_count': self.access_count,
            'evolution': {
                'usefulness_score': self.importance_score,
                'retrieval_contexts': [],
                'strengthened_count': self.access_count,
                'weakened_count': 0
            },
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'JustJotMemory':
        """Create from dictionary."""
        importance = data.get('importance', {})

        return cls(
            memory_id=data['memory_id'],
            entity_id=data['entity_id'],
            entity_type=data.get('entity_type', 'agent'),
            content=data['content'],
            content_hash=data.get('content_hash', ''),
            cognitive_type=data.get('cognitive_type', 'episodic'),
            cognitive_confidence=data.get('cognitive_confidence', 1.0),
            cognitive_reasoning=data.get('cognitive_reasoning', ''),
            importance_score=importance.get('score', 0.5) if isinstance(importance, dict) else 0.5,
            importance_factors=importance.get('factors', {}) if isinstance(importance, dict) else {},
            context=data.get('context', {}),
            created_at=datetime.fromisoformat(data['created_at']) if isinstance(data.get('created_at'), str) else datetime.now(),
            updated_at=datetime.fromisoformat(data['updated_at']) if isinstance(data.get('updated_at'), str) else datetime.now(),
            accessed_at=datetime.fromisoformat(data['accessed_at']) if isinstance(data.get('accessed_at'), str) else datetime.now(),
            access_count=data.get('access_count', 0),
            metadata=data.get('metadata', {})
        )


class JustJotMemorySyncAdapter:
    """
    Bidirectional sync adapter between Jotty's Cortex and JustJot's MemoryGraph.

    Usage:
        from Jotty.core.memory.cortex import HierarchicalMemory
        from Jotty.core.memory.justjot_sync_adapter import JustJotMemorySyncAdapter

        # Set via environment: export JUSTJOT_API_URL=http://your-server:3000
        cortex = HierarchicalMemory(agent_name="my_agent", config=config)
        adapter = JustJotMemorySyncAdapter(cortex=cortex)
        # Or explicitly: adapter = JustJotMemorySyncAdapter(cortex=cortex, justjot_api_url="http://...")

        # Sync Jotty memory to JustJot
        justjot_mem = adapter.sync_to_justjot(memory_entry)

        # Sync JustJot memory to Jotty
        jotty_entry = adapter.sync_from_justjot(justjot_memory)

        # Full bidirectional sync
        await adapter.sync_all()
    """

    # Level mapping: Jotty 5-level → JustJot 4-type
    LEVEL_TO_COGNITIVE_TYPE = {
        'EPISODIC': 'episodic',
        'SEMANTIC': 'semantic',
        'PROCEDURAL': 'procedural',
        'META': 'semantic',      # Merge META into semantic
        'CAUSAL': 'semantic'     # Merge CAUSAL into semantic
    }

    # Reverse mapping: JustJot 4-type → Jotty 5-level
    COGNITIVE_TYPE_TO_LEVEL = {
        'episodic': 'EPISODIC',
        'semantic': 'SEMANTIC',
        'procedural': 'PROCEDURAL',
        'working': 'EPISODIC'    # Working memory is temporary episodic
    }

    def __init__(
        self,
        cortex=None,
        justjot_api_url: str = None,
        entity_type: str = "agent"
    ):
        """
        Initialize sync adapter.

        Args:
            cortex: Jotty HierarchicalMemory instance
            justjot_api_url: JustJot API base URL (defaults to JUSTJOT_API_URL env var)
            entity_type: Entity type for JustJot ('agent', 'user', 'process', etc.)
        """
        self.cortex = cortex
        self.api_url = (justjot_api_url or JUSTJOT_API_URL).rstrip('/')
        self.entity_type = entity_type

        # Track sync state
        self._last_sync: Optional[datetime] = None
        self._sync_count = 0
        self._sync_errors: List[str] = []

        logger.info(
            f"JustJotMemorySyncAdapter initialized: "
            f"api_url={self.api_url}, entity_type={self.entity_type}"
        )

    def level_to_cognitive_type(self, level) -> str:
        """
        Map Jotty 5-level to JustJot 4-type cognitive type.

        Mapping:
            EPISODIC   → episodic
            SEMANTIC   → semantic
            PROCEDURAL → procedural
            META       → semantic (merged)
            CAUSAL     → semantic (merged)
        """
        if hasattr(level, 'value'):
            level_str = level.value
        else:
            level_str = str(level).upper()

        return self.LEVEL_TO_COGNITIVE_TYPE.get(level_str, 'episodic')

    def cognitive_type_to_level(self, cognitive_type: str):
        """
        Map JustJot 4-type cognitive type to Jotty 5-level.

        Mapping:
            episodic   → EPISODIC
            semantic   → SEMANTIC
            procedural → PROCEDURAL
            working    → EPISODIC
        """
        level_str = self.COGNITIVE_TYPE_TO_LEVEL.get(cognitive_type.lower(), 'EPISODIC')

        if MemoryLevel:
            return MemoryLevel(level_str.lower())
        return level_str

    def goal_values_to_importance(self, goal_values: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """
        Transform Jotty goal_values to JustJot importance score and factors.

        Jotty goal_values: {goal_name: GoalValue(value=float, access_count=int)}
        JustJot importance: {score: float, factors: {recency, semantic, structural, engagement, contextual, usage}}

        Returns:
            (importance_score, importance_factors)
        """
        if not goal_values:
            return 0.5, {}

        # Compute average value across all goals
        total_value = 0.0
        total_access = 0

        for goal, gv in goal_values.items():
            if hasattr(gv, 'value'):
                total_value += gv.value
                total_access += getattr(gv, 'access_count', 1)
            elif isinstance(gv, dict):
                total_value += gv.get('value', 0.5)
                total_access += gv.get('access_count', 1)
            else:
                total_value += float(gv)
                total_access += 1

        avg_value = total_value / len(goal_values) if goal_values else 0.5

        # Map to JustJot importance factors
        factors = {
            'recency': 0.5,       # Will be computed by JustJot based on timestamps
            'semantic': avg_value,
            'structural': avg_value * 0.8,
            'engagement': min(1.0, total_access / 10),  # Normalize access count
            'contextual': 0.5,
            'usage': min(1.0, total_access / 20)
        }

        # Compute combined score (weighted average)
        weights = {
            'recency': 0.2,
            'semantic': 0.3,
            'structural': 0.1,
            'engagement': 0.15,
            'contextual': 0.1,
            'usage': 0.15
        }

        score = sum(factors[k] * weights[k] for k in factors)

        return score, factors

    def importance_to_goal_values(
        self,
        importance_score: float,
        importance_factors: Dict[str, float],
        goal: str = "default"
    ) -> Dict[str, Any]:
        """
        Transform JustJot importance to Jotty goal_values.

        Returns:
            {goal: GoalValue-like dict}
        """
        # Extract usage-based access count
        access_count = int(importance_factors.get('usage', 0.5) * 20)

        # Use semantic factor as the main value
        value = importance_factors.get('semantic', importance_score)

        if GoalValue:
            return {goal: GoalValue(value=value, access_count=access_count)}

        return {goal: {'value': value, 'access_count': access_count}}

    def sync_to_justjot(self, memory_entry) -> JustJotMemory:
        """
        Transform Jotty MemoryEntry → JustJot Memory format.

        Args:
            memory_entry: Jotty MemoryEntry object

        Returns:
            JustJotMemory object ready for API upload
        """
        # Generate memory_id from Jotty key
        memory_id = memory_entry.key if hasattr(memory_entry, 'key') else hashlib.md5(
            memory_entry.content.encode()
        ).hexdigest()

        # Get entity_id (agent name)
        entity_id = getattr(memory_entry, 'source_agent', None) or \
                   getattr(self.cortex, 'agent_name', 'unknown') if self.cortex else 'unknown'

        # Map level to cognitive type
        level = getattr(memory_entry, 'level', None)
        cognitive_type = self.level_to_cognitive_type(level) if level else 'episodic'

        # Transform goal values to importance
        goal_values = getattr(memory_entry, 'goal_values', {})
        importance_score, importance_factors = self.goal_values_to_importance(goal_values)

        # Extract context
        jotty_context = getattr(memory_entry, 'context', {}) or {}
        justjot_context = {
            'source_entity_id': entity_id,
            'tags': list(jotty_context.keys())[:10],  # Use context keys as tags
            'custom': jotty_context
        }

        # Build metadata
        metadata = {
            'jotty_key': memory_entry.key if hasattr(memory_entry, 'key') else None,
            'jotty_level': level.value if hasattr(level, 'value') else str(level),
            'jotty_default_value': getattr(memory_entry, 'default_value', 0.5),
            'jotty_is_protected': getattr(memory_entry, 'is_protected', False),
            'jotty_causal_links': getattr(memory_entry, 'causal_links', []),
            'synced_from': 'jotty'
        }

        # Create JustJotMemory
        return JustJotMemory(
            memory_id=memory_id,
            entity_id=entity_id,
            entity_type=self.entity_type,
            content=memory_entry.content,
            content_hash=hashlib.sha256(memory_entry.content.encode()).hexdigest(),
            cognitive_type=cognitive_type,
            cognitive_confidence=1.0,
            cognitive_reasoning=f"Synced from Jotty {level}",
            importance_score=importance_score,
            importance_factors=importance_factors,
            context=justjot_context,
            created_at=getattr(memory_entry, 'created_at', datetime.now()),
            updated_at=getattr(memory_entry, 'last_accessed', datetime.now()),
            accessed_at=getattr(memory_entry, 'last_accessed', datetime.now()),
            access_count=getattr(memory_entry, 'access_count', 0),
            metadata=metadata
        )

    def sync_from_justjot(self, justjot_memory: JustJotMemory):
        """
        Transform JustJot Memory → Jotty MemoryEntry format.

        Args:
            justjot_memory: JustJotMemory object

        Returns:
            MemoryEntry object ready for Cortex storage
        """
        if not MemoryEntry:
            # Return as dict if MemoryEntry not available
            return self._create_memory_dict(justjot_memory)

        # Map cognitive type to level
        level = self.cognitive_type_to_level(justjot_memory.cognitive_type)

        # Transform importance to goal values
        goal = justjot_memory.context.get('custom', {}).get('goal', 'default')
        goal_values = self.importance_to_goal_values(
            justjot_memory.importance_score,
            justjot_memory.importance_factors,
            goal
        )

        # Generate Jotty key
        domain = justjot_memory.context.get('custom', {}).get('domain', 'general')
        task_type = justjot_memory.context.get('custom', {}).get('task_type', 'general')
        content_hash = hashlib.md5(justjot_memory.content.encode()).hexdigest()[:16]
        key = f"{domain}:{task_type}:{content_hash}"

        # Build context
        context = justjot_memory.context.get('custom', {})
        context['justjot_entity_type'] = justjot_memory.entity_type
        context['justjot_memory_id'] = justjot_memory.memory_id
        context['synced_from'] = 'justjot'

        # Create MemoryEntry
        entry = MemoryEntry(
            key=key,
            content=justjot_memory.content,
            level=level,
            context=context,
            created_at=justjot_memory.created_at,
            last_accessed=justjot_memory.accessed_at,
            default_value=justjot_memory.importance_score,
            access_count=justjot_memory.access_count,
            ucb_visits=justjot_memory.access_count,
            token_count=len(justjot_memory.content) // 4 + 1,
            is_protected=justjot_memory.metadata.get('jotty_is_protected', False),
            causal_links=justjot_memory.metadata.get('jotty_causal_links', [])
        )

        # Set goal values
        entry.goal_values = goal_values

        return entry

    def _create_memory_dict(self, justjot_memory: JustJotMemory) -> Dict[str, Any]:
        """Create a memory dict when MemoryEntry is not available."""
        level_str = self.cognitive_type_to_level(justjot_memory.cognitive_type)

        domain = justjot_memory.context.get('custom', {}).get('domain', 'general')
        task_type = justjot_memory.context.get('custom', {}).get('task_type', 'general')
        content_hash = hashlib.md5(justjot_memory.content.encode()).hexdigest()[:16]

        return {
            'key': f"{domain}:{task_type}:{content_hash}",
            'content': justjot_memory.content,
            'level': level_str,
            'context': justjot_memory.context.get('custom', {}),
            'created_at': justjot_memory.created_at.isoformat(),
            'last_accessed': justjot_memory.accessed_at.isoformat(),
            'default_value': justjot_memory.importance_score,
            'access_count': justjot_memory.access_count,
            'goal_values': {
                'default': {
                    'value': justjot_memory.importance_score,
                    'access_count': justjot_memory.access_count
                }
            }
        }

    async def push_to_justjot(self, memory_entry) -> bool:
        """
        Push a single Jotty memory to JustJot API.

        Args:
            memory_entry: Jotty MemoryEntry

        Returns:
            True if push succeeded
        """
        try:
            import aiohttp

            justjot_memory = self.sync_to_justjot(memory_entry)

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}/api/memory/store",
                    json=justjot_memory.to_dict(),
                    headers={'Content-Type': 'application/json'}
                ) as response:
                    if response.status == 200:
                        self._sync_count += 1
                        return True
                    else:
                        error = await response.text()
                        self._sync_errors.append(f"Push failed: {error}")
                        return False

        except ImportError:
            logger.warning("aiohttp not available, using requests")
            return self._push_sync(memory_entry)
        except Exception as e:
            self._sync_errors.append(f"Push error: {str(e)}")
            logger.error(f"Push to JustJot failed: {e}", exc_info=True)
            return False

    def _push_sync(self, memory_entry) -> bool:
        """Synchronous push using requests."""
        try:
            import requests

            justjot_memory = self.sync_to_justjot(memory_entry)

            response = requests.post(
                f"{self.api_url}/api/memory/store",
                json=justjot_memory.to_dict(),
                headers={'Content-Type': 'application/json'},
                timeout=30
            )

            if response.status_code == 200:
                self._sync_count += 1
                return True
            else:
                self._sync_errors.append(f"Push failed: {response.text}")
                return False

        except Exception as e:
            self._sync_errors.append(f"Push error: {str(e)}")
            return False

    async def pull_from_justjot(self, entity_id: str = None, limit: int = 100) -> List:
        """
        Pull memories from JustJot API.

        Args:
            entity_id: Entity ID to pull (uses cortex agent_name if None)
            limit: Maximum memories to pull

        Returns:
            List of MemoryEntry objects
        """
        try:
            import aiohttp

            entity_id = entity_id or (self.cortex.agent_name if self.cortex else 'unknown')

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.api_url}/api/memory/query",
                    params={
                        'entity_id': entity_id,
                        'limit': limit
                    }
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        memories = data.get('memories', [])

                        return [
                            self.sync_from_justjot(JustJotMemory.from_dict(m))
                            for m in memories
                        ]
                    else:
                        error = await response.text()
                        self._sync_errors.append(f"Pull failed: {error}")
                        return []

        except ImportError:
            logger.warning("aiohttp not available, using requests")
            return self._pull_sync(entity_id, limit)
        except Exception as e:
            self._sync_errors.append(f"Pull error: {str(e)}")
            logger.error(f"Pull from JustJot failed: {e}", exc_info=True)
            return []

    def _pull_sync(self, entity_id: str, limit: int) -> List:
        """Synchronous pull using requests."""
        try:
            import requests

            response = requests.get(
                f"{self.api_url}/api/memory/query",
                params={
                    'entity_id': entity_id,
                    'limit': limit
                },
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                memories = data.get('memories', [])

                return [
                    self.sync_from_justjot(JustJotMemory.from_dict(m))
                    for m in memories
                ]
            else:
                self._sync_errors.append(f"Pull failed: {response.text}")
                return []

        except Exception as e:
            self._sync_errors.append(f"Pull error: {str(e)}")
            return []

    async def sync_all(self, direction: str = "both") -> Dict[str, Any]:
        """
        Perform full bidirectional sync.

        Args:
            direction: 'to_justjot', 'from_justjot', or 'both'

        Returns:
            Sync statistics
        """
        stats = {
            'pushed': 0,
            'pulled': 0,
            'errors': [],
            'timestamp': datetime.now().isoformat()
        }

        if direction in ('to_justjot', 'both') and self.cortex:
            # Push all Jotty memories to JustJot
            for level in self.cortex.memories:
                for key, entry in self.cortex.memories[level].items():
                    # Skip if already synced recently
                    if entry.metadata and entry.metadata.get('synced_from') == 'justjot':
                        continue

                    if await self.push_to_justjot(entry):
                        stats['pushed'] += 1

        if direction in ('from_justjot', 'both') and self.cortex:
            # Pull from JustJot and store in Cortex
            pulled = await self.pull_from_justjot()

            for entry in pulled:
                # Skip if already exists
                if entry.key in self.cortex.memories.get(entry.level, {}):
                    continue

                # Store in cortex
                self.cortex.memories[entry.level][entry.key] = entry
                stats['pulled'] += 1

        stats['errors'] = self._sync_errors[-10:]  # Last 10 errors
        self._last_sync = datetime.now()

        return stats

    def get_sync_status(self) -> Dict[str, Any]:
        """Get current sync status."""
        return {
            'last_sync': self._last_sync.isoformat() if self._last_sync else None,
            'sync_count': self._sync_count,
            'error_count': len(self._sync_errors),
            'recent_errors': self._sync_errors[-5:],
            'api_url': self.api_url,
            'entity_type': self.entity_type
        }
