"""
Jotty v6.0 - Enhanced Memory System
====================================

Hierarchical memory with all A-Team enhancements:
- Aristotle: Causal knowledge layer, goal-conditioned wisdom
- Shannon: Deduplication, compression, information-theoretic selection
- Dr. Agarwal: LLM-based consolidation, size-aware storage

Five levels:
1. EPISODIC - Raw experiences (fast decay)
2. SEMANTIC - Abstracted patterns (slow decay)  
3. PROCEDURAL - Action sequences (medium decay)
4. META - Learning wisdom (no decay)
5. CAUSAL - Why things work (no decay) [NEW]
"""

import json
import hashlib
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
logger = logging.getLogger(__name__)

def _get_dspy():
    import dspy
    return dspy

from ..foundation.data_structures import (
    MemoryEntry, MemoryLevel, GoalValue, SwarmConfig,
    GoalHierarchy, GoalNode, CausalLink, StoredEpisode
)
from .llm_rag import LLMRAGRetriever, DeduplicationEngine, CausalExtractor


_consolidation_loaded = False
_consolidation_cache = {}

def _get_consolidation():
    global _consolidation_loaded, _consolidation_cache
    if not _consolidation_loaded:
        from .consolidation import (
            PatternExtractionSignature, ProceduralExtractionSignature, MetaWisdomSignature,
            MemoryLevelClassificationSignature, ConsolidationValidationSignature,
            ConsolidationValidator, MemoryLevelClassifier, MemoryCluster,
        )
        _consolidation_cache.update({
            'PatternExtractionSignature': PatternExtractionSignature,
            'ProceduralExtractionSignature': ProceduralExtractionSignature,
            'MetaWisdomSignature': MetaWisdomSignature,
            'MemoryLevelClassificationSignature': MemoryLevelClassificationSignature,
            'ConsolidationValidationSignature': ConsolidationValidationSignature,
            'ConsolidationValidator': ConsolidationValidator,
            'MemoryLevelClassifier': MemoryLevelClassifier,
            'MemoryCluster': MemoryCluster,
        })
        _consolidation_loaded = True
    return _consolidation_cache

from ._retrieval_mixin import RetrievalMixin
from ._consolidation_mixin import ConsolidationMixin

class SwarmMemory(RetrievalMixin, ConsolidationMixin):
    """
    Enhanced hierarchical memory with all A-Team features.
    
    Levels:
    - EPISODIC: Raw experiences, high detail, fast decay
    - SEMANTIC: Abstracted patterns, extracted via LLM
    - PROCEDURAL: Action sequences, how to do things
    - META: Learning wisdom, never decays
    - CAUSAL: Why things work, enables reasoning about new situations
    
    Key features:
    - LLM-based retrieval (no embeddings)
    - Goal-conditioned values with transfer
    - Causal knowledge extraction
    - Automatic consolidation
    - Deduplication
    - Protection against forgetting
    """
    
    def __init__(self, agent_name: str, config: SwarmConfig):
        self.agent_name = agent_name
        self.config = config
        
        # Memory storage by level
        self.memories: Dict[MemoryLevel, Dict[str, MemoryEntry]] = {
            MemoryLevel.EPISODIC: {},
            MemoryLevel.SEMANTIC: {},
            MemoryLevel.PROCEDURAL: {},
            MemoryLevel.META: {},
            MemoryLevel.CAUSAL: {}
        }
        
        # Causal knowledge
        self.causal_links: Dict[str, CausalLink] = {}
        
        # Goal hierarchy
        self.goal_hierarchy = GoalHierarchy()
        
        # Capacities
        self.capacities = {
            MemoryLevel.EPISODIC: config.episodic_capacity,
            MemoryLevel.SEMANTIC: config.semantic_capacity,
            MemoryLevel.PROCEDURAL: config.procedural_capacity,
            MemoryLevel.META: config.meta_capacity,
            MemoryLevel.CAUSAL: config.causal_capacity
        }
        
        # LLM components
        self.retriever = LLMRAGRetriever(config)
        self.deduplicator = DeduplicationEngine(config)
        self.causal_extractor = CausalExtractor(config)
        
        # LLM consolidators (lazy — DSPy loaded on first use)
        dspy = _get_dspy()
        cons = _get_consolidation()
        self.pattern_extractor = dspy.ChainOfThought(cons['PatternExtractionSignature'])
        self.procedural_extractor = dspy.ChainOfThought(cons['ProceduralExtractionSignature'])
        self.meta_extractor = dspy.ChainOfThought(cons['MetaWisdomSignature'])
        
        # Statistics
        self.total_accesses = 0
        self.consolidation_count = 0
    
    # =========================================================================
    # STORAGE
    # =========================================================================
    
    def store(self,
              content: str,
              level: MemoryLevel,
              context: Dict[str, Any],
              goal: str,
              initial_value: float = 0.5,
              causal_links: List[str] = None,
              domain: Optional[str] = None,
              task_type: Optional[str] = None) -> MemoryEntry:
        """
        Store a new memory with hierarchical key generation and deduplication.
        
        Key format: {domain}:{task_type}:{content_hash}
        
        A-Team Note: For information-weighted storage, use store_with_outcome()
        which automatically adjusts storage detail based on event rarity.
        
        Args:
            domain: Domain identifier (e.g., 'sql', 'mermaid', 'plantuml'). 
                    If None, extracted from context['domain'] or defaults to 'general'
            task_type: Task type (e.g., 'date_filter', 'sequence_diagram').
                      If None, extracted from context['task_type'] or context['operation_type'] or defaults to 'general'
        """
        # Check size limit
        token_count = len(content) // 4 + 1
        if token_count > self.config.max_entry_tokens:
            # Chunk and store first chunk only (with note)
            content = content[:self.config.max_entry_tokens * 4]
            content += "\n[TRUNCATED - Original was longer]"
            token_count = self.config.max_entry_tokens
        
        # Extract domain/task_type from context if not provided (backward compatible)
        if domain is None:
            domain = context.get('domain', 'general')
        if task_type is None:
            task_type = context.get('task_type') or context.get('operation_type', 'general')
        
        # Generate hierarchical key: {domain}:{task_type}:{content_hash}
        content_hash = hashlib.md5(content.encode()).hexdigest()[:16]
        new_key = f"{domain}:{task_type}:{content_hash}"
        
        # Check for existing memory (new format - fast exact match)
        if level in self.memories and new_key in self.memories[level]:
            existing = self.memories[level][new_key]
            existing.access_count += 1
            existing.last_accessed = datetime.now()
            return existing
        
        # Check for old-format keys with same content (migration)
        # Two-pass approach: find first, then mutate (avoids RuntimeError from
        # modifying dict during iteration).
        if level in self.memories:
            migrate_old_key = None
            for old_key, old_entry in self.memories[level].items():
                # Old format: no colons (just hash)
                if ':' not in old_key:
                    old_hash = hashlib.md5(old_entry.content.encode()).hexdigest()[:16]
                    if old_hash == content_hash:
                        migrate_old_key = old_key
                        break  # Found match, stop iterating
            
            if migrate_old_key is not None:
                old_entry = self.memories[level].pop(migrate_old_key)
                old_entry.key = new_key
                # Update metadata with domain/task_type
                if not hasattr(old_entry, 'metadata') or old_entry.metadata is None:
                    old_entry.metadata = {}
                old_entry.metadata['domain'] = domain
                old_entry.metadata['task_type'] = task_type
                self.memories[level][new_key] = old_entry
                old_entry.access_count += 1
                old_entry.last_accessed = datetime.now()
                return old_entry
        
        # Create new entry
        entry = MemoryEntry(
            key=new_key,
            content=content,
            level=level,
            context=context,
            token_count=token_count,
            source_episode=context.get('episode', 0),
            source_agent=self.agent_name,
            causal_links=causal_links or []
        )
        
        # Store domain/task_type in metadata for easy filtering
        if not hasattr(entry, 'metadata') or entry.metadata is None:
            entry.metadata = {}
        entry.metadata['domain'] = domain
        entry.metadata['task_type'] = task_type
        
        # Set goal-conditioned value
        entry.goal_values[goal] = GoalValue(value=initial_value)
        entry.default_value = initial_value
        
        # Check for semantic duplicates (preserves existing deduplication behavior)
        if self.config.enable_deduplication:
            existing = list(self.memories[level].values())
            for existing_mem in existing:  # Check recent only for efficiency
                is_dup, sim, merged = self.deduplicator.check_duplicate(entry, existing_mem)
                if is_dup:
                    # Found semantic duplicate - merge into existing
                    # Update existing memory's key to new format if different
                    if existing_mem.key != new_key:
                        # Migrate to new key format
                        del self.memories[level][existing_mem.key]
                        existing_mem.key = new_key
                        if not hasattr(existing_mem, 'metadata') or existing_mem.metadata is None:
                            existing_mem.metadata = {}
                        existing_mem.metadata['domain'] = domain
                        existing_mem.metadata['task_type'] = task_type
                        self.memories[level][new_key] = existing_mem
                    
                    existing_mem.content = merged
                    existing_mem.similar_entries.append(new_key)
                    return existing_mem
        
        # Enforce capacity
        self._enforce_capacity(level)
        
        # Store
        self.memories[level][new_key] = entry
        
        # Update goal hierarchy
        if self.config.enable_goal_hierarchy:
            op_type = context.get('operation_type', task_type)
            entities = context.get('entities', [])
            self.goal_hierarchy.add_goal(goal, domain, op_type, entities)
        
        return entry
    
    def store_with_outcome(self,
                           content: str,
                           context: Dict[str, Any],
                           goal: str,
                           outcome: str = "neutral",
                           domain: Optional[str] = None,
                           task_type: Optional[str] = None) -> MemoryEntry:
        """
        A-Team Enhancement: Information-Weighted Storage.
        
        Shannon Insight: I(event) = -log P(event)
        
        - FAILURES are rare (hopefully) → HIGH information → store MORE details
        - SUCCESSES are common → LOW information → store LESS details
        - NEUTRAL events → NORMAL storage
        
        This ensures the memory focuses on learning from surprising events
        while avoiding redundant storage of common patterns.
        
        Args:
            content: The content to store
            context: Context dictionary
            goal: The goal being pursued
            outcome: One of "success", "failure", "neutral"
        
        Returns:
            The stored memory entry
        """
        if outcome == "failure":
            # Failures are rare = HIGH information content
            # Store MAXIMUM detail in CAUSAL memory (learn WHY it failed)
            enhanced_content = f"""❌ FAILURE ANALYSIS (High Information Event)
═══════════════════════════════════════════════════
{content}

FULL CONTEXT:
{json.dumps(context, default=str, indent=2)}

MEMORY NOTE: This is a rare failure event with high information content.
Store full trace to prevent similar failures.
═══════════════════════════════════════════════════"""
            
            return self.store(
                content=enhanced_content,
                level=MemoryLevel.CAUSAL,  # Causal for "why" learning
                context=context,
                goal=goal,
                initial_value=0.9,  # High value (rare = valuable)
                domain=domain,
                task_type=task_type
            )
        
        elif outcome == "success":
            # Successes should be common = LOW information content
            # Store SUMMARY in SEMANTIC memory (just the key insight)
            # Summarize to key insight
            lines = content.split('\n')
            summary = lines[0] if lines else content
            
            return self.store(
                content=f"✅ Success: {summary}",
                level=MemoryLevel.SEMANTIC,  # Semantic for patterns
                context=context,
                goal=goal,
                initial_value=0.4,  # Lower value (common = less valuable)
                domain=domain,
                task_type=task_type
            )
        
        else:
            # Neutral - normal storage
            return self.store(
                content=content,
                level=MemoryLevel.EPISODIC,  # Episodic for events
                context=context,
                goal=goal,
                initial_value=0.5,
                domain=domain,
                task_type=task_type
            )
    
    def _enforce_capacity(self, level: MemoryLevel):
        """Ensure level doesn't exceed capacity."""
        capacity = self.capacities[level]
        memories = self.memories[level]
        
        while len(memories) >= capacity:
            # Find lowest-value unprotected memory
            candidates = [
                (k, m) for k, m in memories.items()
                if not m.is_protected
            ]
            
            if not candidates:
                # All protected - force remove oldest
                oldest = min(memories.values(), key=lambda m: m.created_at)
                del memories[oldest.key]
            else:
                # Remove lowest value
                to_remove = min(candidates, key=lambda x: x[1].default_value)
                del memories[to_remove[0]]
    
