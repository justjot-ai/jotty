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
import dspy

logger = logging.getLogger(__name__)

from ..foundation.data_structures import (
    MemoryEntry, MemoryLevel, GoalValue, JottyConfig,
    GoalHierarchy, GoalNode, CausalLink, StoredEpisode
)
from .llm_rag import LLMRAGRetriever, DeduplicationEngine, CausalExtractor


# =============================================================================
# CONSOLIDATION SIGNATURES
# =============================================================================

class PatternExtractionSignature(dspy.Signature):
    """Extract patterns from episodic memories using chain-of-thought."""
    
    memories: str = dspy.InputField(desc="JSON list of related episodic memories")
    goal_context: str = dspy.InputField(desc="The goal context these memories relate to")
    domain: str = dspy.InputField(desc="Domain (e.g., sql, python, api, etc.)")
    
    reasoning: str = dspy.OutputField(desc="Analysis of what patterns emerge")
    pattern: str = dspy.OutputField(desc="The extracted pattern as a clear statement")
    confidence: float = dspy.OutputField(desc="Confidence in pattern 0.0-1.0")
    conditions: str = dspy.OutputField(desc="When this pattern applies")
    exceptions: str = dspy.OutputField(desc="When this pattern does NOT apply")


class ProceduralExtractionSignature(dspy.Signature):
    """Extract procedural knowledge (how to do things)."""
    
    success_traces: str = dspy.InputField(desc="Traces of successful episodes")
    failure_traces: str = dspy.InputField(desc="Traces of failed episodes")
    task_type: str = dspy.InputField(desc="Type of task")
    
    reasoning: str = dspy.OutputField(desc="Analysis of what steps lead to success")
    procedure: str = dspy.OutputField(desc="Step-by-step procedure")
    key_decisions: str = dspy.OutputField(desc="Critical decision points")


class MetaWisdomSignature(dspy.Signature):
    """Extract meta-level wisdom about learning itself."""
    
    learning_history: str = dspy.InputField(desc="Summary of learning progress")
    failure_analysis: str = dspy.InputField(desc="Common failure patterns")
    success_analysis: str = dspy.InputField(desc="Common success patterns")
    
    wisdom: str = dspy.OutputField(desc="Meta-level insight about when to apply what knowledge")
    applicability: str = dspy.OutputField(desc="When this wisdom applies")


# =============================================================================
# MEMORY LEVEL CLASSIFIER (A-Team Enhancement)
# =============================================================================

class MemoryLevelClassificationSignature(dspy.Signature):
    """
    A-Team Enhancement: LLM-based memory level classification.
    
    Instead of hardcoding which level to store to, use LLM to decide:
    - EPISODIC: Raw experiences, specific events, tool outputs
    - SEMANTIC: Patterns, abstractions, generalizations
    - PROCEDURAL: How-to knowledge, step sequences
    - META: Wisdom about learning, when to use what
    - CAUSAL: Why things work, cause-effect relationships
    """
    
    experience: str = dspy.InputField(desc="The experience/knowledge to classify")
    context: str = dspy.InputField(desc="Context: task type, agent, goal, outcome")
    
    reasoning: str = dspy.OutputField(desc="Why this memory level is appropriate")
    level: str = dspy.OutputField(desc="One of: EPISODIC, SEMANTIC, PROCEDURAL, META, CAUSAL")
    confidence: float = dspy.OutputField(desc="Confidence 0.0-1.0")
    should_store: bool = dspy.OutputField(desc="True if worth storing, False if redundant")


class MemoryLevelClassifier:
    """
    A-Team Enhancement: LLM-based automatic memory level classification.
    
    Replaces hardcoded level decisions with intelligent classification.
    
    Usage:
        classifier = MemoryLevelClassifier()
        level, confidence, should_store = classifier.classify(
            experience="Successfully mapped bank_code column using regex extraction",
            context={"task": "column_mapping", "agent": "diffuser", "outcome": "success"}
        )
        
        if should_store:
            memory.store(content=experience, level=level, ...)
    """
    
    def __init__(self, use_cot: bool = True):
        self.use_cot = use_cot
        if use_cot:
            self.classifier = dspy.ChainOfThought(MemoryLevelClassificationSignature)
        else:
            self.classifier = dspy.Predict(MemoryLevelClassificationSignature)
        
        # Level mapping
        self.level_map = {
            'EPISODIC': MemoryLevel.EPISODIC,
            'SEMANTIC': MemoryLevel.SEMANTIC,
            'PROCEDURAL': MemoryLevel.PROCEDURAL,
            'META': MemoryLevel.META,
            'CAUSAL': MemoryLevel.CAUSAL
        }
        
        # A-Team v8.0: NO keyword lists! Structure-based classification only
        # Removed level_hints entirely - uses _heuristic_classify with structural analysis
    
    def classify(self, experience: str, context: Dict[str, Any]) -> Tuple[MemoryLevel, float, bool]:
        """
        Classify experience into appropriate memory level.
        
        Returns:
            (MemoryLevel, confidence, should_store)
        """
        import json
        
        try:
            result = self.classifier(
                experience=experience,  # 🔥 NO LIMIT - FULL content
                context=json.dumps(context)
            )
            
            level_str = (result.level or 'EPISODIC').upper().strip()
            level = self.level_map.get(level_str, MemoryLevel.EPISODIC)
            confidence = float(result.confidence) if result.confidence else 0.5
            should_store = result.should_store if hasattr(result, 'should_store') else True

            return level, confidence, should_store

        except Exception as e:
            logger.debug(f"Classification failed: {e}, using heuristic")
            # Fallback to heuristic classification
            return self._heuristic_classify(experience), 0.5, True
    
    async def _classify_with_retry(self, experience: str, context: Dict[str, Any]) -> MemoryLevel:
        """
        A-Team v9.0: NO HEURISTIC FALLBACKS.
        
        If primary classification fails:
        1. Retry with context of failure
        2. If still fails, use FallbackClassificationAgent
        3. NEVER use hardcoded rules
        """
        from .modern_agents import UniversalRetryHandler, PatternDetector
        
        retry_handler = UniversalRetryHandler(max_retries=3)
        
        async def classify_attempt(**kwargs):
            exp = kwargs.get('experience', '')
            ctx = kwargs.get('context', {})
            
            result = self.classifier(
                experience=exp,
                context=json.dumps(ctx)
            )
            
            level_str = (result.level or 'EPISODIC').upper().strip()
            return self.level_map.get(level_str, MemoryLevel.EPISODIC)

        # Fallback agent (specialized for difficult cases)
        async def fallback_classifier(**kwargs):
            """
            Specialized agent for difficult classification cases.
            
            Gets full error context and tries harder.
            """
            task_info = kwargs.get('task', '')
            errors = kwargs.get('error_history', [])
            
            # Create a more detailed prompt with all context
            detailed_prompt = f"""
            DIFFICULT CLASSIFICATION TASK
            
            The primary classifier failed with these errors:
            {errors}
            
            Original task: {task_info}
            
            Please classify this experience into one of:
            - EPISODIC: Raw events, specific instances, tool outputs
            - SEMANTIC: Patterns, generalizations, abstractions
            - PROCEDURAL: How-to knowledge, step sequences
            - META: Wisdom about approach, when to use what
            - CAUSAL: Why things work, cause-effect relationships
            
            Think carefully and provide classification.
            """
            
            # Use a fresh classifier with detailed prompt
            class DetailedClassification(dspy.Signature):
                detailed_context = dspy.InputField()
                experience = dspy.InputField()
                level = dspy.OutputField()
                confidence = dspy.OutputField()
            
            specialist = dspy.ChainOfThought(DetailedClassification)
            result = specialist(
                detailed_context=detailed_prompt,
                experience=kwargs.get('original_input', {}).get('experience', '')
            )
            
            level_str = (result.level or 'EPISODIC').upper().strip()
            return self.level_map.get(level_str, MemoryLevel.EPISODIC)

        result = await retry_handler.execute_with_retry(
            agent_func=classify_attempt,
            task_description=f"Classify memory level for: {experience}...",
            initial_input={'experience': experience, 'context': context},
            specialist_agent=fallback_classifier
        )
        
        if result.is_certain:
            return result.value
        else:
            # Even specialist failed - return with uncertainty flag
            # (This is NOT a heuristic fallback, it's explicit uncertainty)
            logger.warning(f"Classification uncertain after all retries: {result.reasoning}")
            return MemoryLevel.EPISODIC  # Default with logged uncertainty


# =============================================================================
# MEMORY CLUSTER
# =============================================================================

@dataclass
class MemoryCluster:
    """A cluster of related memories for consolidation."""
    cluster_id: str
    goal_signature: str
    memories: List[MemoryEntry]
    
    # Cluster statistics
    avg_value: float = 0.0
    success_rate: float = 0.0
    common_keywords: List[str] = field(default_factory=list)
    
    # Extracted pattern (if consolidated)
    extracted_pattern: Optional[str] = None
    pattern_confidence: float = 0.0
    
    def compute_statistics(self):
        """
        Compute cluster statistics.
        
        A-Team Fix: Removed keyword extraction (loses semantic meaning).
        Uses content length and value distribution instead.
        """
        if not self.memories:
            return
        
        values = [m.default_value for m in self.memories]
        self.avg_value = sum(values) / len(values)
        
        # Success rate from memory values (value > 0.5 = successful use)
        successful = sum(1 for v in values if v > 0.5)
        self.success_rate = successful / len(values)
        
        # A-Team: Instead of keywords, store content signatures for hash-based similarity
        # This avoids keyword matching which loses semantic meaning
        # Keywords list now stores content length buckets for fast filtering
        length_buckets = []
        for m in self.memories:
            content_len = len(m.content)
            if content_len < 100:
                length_buckets.append("short")
            elif content_len < 500:
                length_buckets.append("medium")
            else:
                length_buckets.append("long")
        
        # Store most common length bucket (useful for clustering similar experiences)
        from collections import Counter
        bucket_counts = Counter(length_buckets)
        self.common_keywords = [f"content_{b}" for b, _ in bucket_counts.most_common(3)]


# =============================================================================
# MAIN HIERARCHICAL MEMORY
# =============================================================================

class HierarchicalMemory:
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
    
    def __init__(self, agent_name: str, config: JottyConfig):
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
        
        # LLM consolidators
        self.pattern_extractor = dspy.ChainOfThought(PatternExtractionSignature)
        self.procedural_extractor = dspy.ChainOfThought(ProceduralExtractionSignature)
        self.meta_extractor = dspy.ChainOfThought(MetaWisdomSignature)
        
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
        if level in self.memories:
            for old_key, old_entry in self.memories[level].items():
                # Old format: no colons (just hash)
                if ':' not in old_key:
                    old_hash = hashlib.md5(old_entry.content.encode()).hexdigest()[:16]
                    if old_hash == content_hash:
                        # Found old memory with same content - migrate to new key
                        del self.memories[level][old_key]
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
    
    # =========================================================================
    # RETRIEVAL
    # =========================================================================
    
    def retrieve(self,
                 query: str,
                 goal: str,
                 budget_tokens: int,
                 levels: List[MemoryLevel] = None,
                 context_hints: str = "") -> List[MemoryEntry]:
        """
        Retrieve relevant memories using LLM-based RAG.
        
        No embeddings - uses keyword pre-filter + LLM scoring.
        """
        if levels is None:
            levels = list(MemoryLevel)
        
        # Collect all candidates
        all_memories = []
        for level in levels:
            all_memories.extend(self.memories[level].values())
        
        if not all_memories:
            return []
        
        # Use LLM RAG retriever
        selected = self.retriever.retrieve(
            query=query,
            goal=goal,
            memories=all_memories,
            budget_tokens=budget_tokens,
            goal_hierarchy=self.goal_hierarchy if self.config.enable_goal_hierarchy else None,
            context_hints=context_hints
        )
        
        # Update access tracking
        self.total_accesses += 1
        for mem in selected:
            mem.access_count += 1
            mem.ucb_visits += 1
            mem.last_accessed = datetime.now()

        return selected

    async def retrieve_async(self,
                             query: str,
                             goal: str,
                             budget_tokens: int,
                             levels: List[MemoryLevel] = None,
                             context_hints: str = "") -> List[MemoryEntry]:
        """
        Async version of retrieve() for parallel memory retrieval.

        This allows multiple memory retrievals to run concurrently,
        dramatically reducing context building time.
        """
        import asyncio
        loop = asyncio.get_event_loop()

        # Run the synchronous retrieve in a thread pool to avoid blocking
        result = await loop.run_in_executor(
            None,
            lambda: self.retrieve(query, goal, budget_tokens, levels, context_hints)
        )
        return result

    def retrieve_by_domain(self,
                          domain: str,
                          goal: str,
                          budget_tokens: int,
                          levels: List[MemoryLevel] = None) -> List[MemoryEntry]:
        """
        Retrieve memories filtered by domain.
        
        Uses key prefix filtering for fast domain-based retrieval.
        
        Args:
            domain: Domain identifier (e.g., 'sql', 'mermaid', 'plantuml')
            goal: Goal for value-based ranking
            budget_tokens: Token budget for retrieval
            levels: Memory levels to search (default: all levels)
        
        Returns:
            List of MemoryEntry objects from specified domain, ranked by value
        """
        if levels is None:
            levels = list(MemoryLevel)
        
        # Collect memories from specified levels, filtered by domain prefix
        domain_memories = []
        for level in levels:
            if level in self.memories:
                # Filter by domain prefix (fast key-level filtering)
                for key, memory in self.memories[level].items():
                    if key.startswith(f'{domain}:'):
                        domain_memories.append(memory)
        
        if not domain_memories:
            return []
        
        # Use existing retriever for ranking by value and relevance
        selected = self.retriever.retrieve(
            query=f"Domain: {domain}",
            goal=goal,
            memories=domain_memories,
            budget_tokens=budget_tokens,
            goal_hierarchy=self.goal_hierarchy if self.config.enable_goal_hierarchy else None
        )
        
        # Update access tracking
        self.total_accesses += 1
        for mem in selected:
            mem.access_count += 1
            mem.ucb_visits += 1
            mem.last_accessed = datetime.now()
        
        return selected

    def retrieve_by_task_type(self,
                              task_type: str,
                              goal: str,
                              budget_tokens: int,
                              levels: List[MemoryLevel] = None) -> List[MemoryEntry]:
        """
        Retrieve memories filtered by task type.
        
        Uses key pattern matching for fast task-type-based retrieval.
        
        Args:
            task_type: Task type (e.g., 'date_filter', 'sequence_diagram')
            goal: Goal for value-based ranking
            budget_tokens: Token budget for retrieval
            levels: Memory levels to search (default: all levels)
        
        Returns:
            List of MemoryEntry objects for specified task type, ranked by value
        """
        if levels is None:
            levels = list(MemoryLevel)
        
        # Collect memories from specified levels, filtered by task type pattern
        task_memories = []
        for level in levels:
            if level in self.memories:
                # Filter by task type pattern (second part of key: domain:task_type:hash)
                for key, memory in self.memories[level].items():
                    if f':{task_type}:' in key:
                        task_memories.append(memory)
        
        if not task_memories:
            return []
        
        # Use existing retriever for ranking by value and relevance
        selected = self.retriever.retrieve(
            query=f"Task type: {task_type}",
            goal=goal,
            memories=task_memories,
            budget_tokens=budget_tokens,
            goal_hierarchy=self.goal_hierarchy if self.config.enable_goal_hierarchy else None
        )
        
        # Update access tracking
        self.total_accesses += 1
        for mem in selected:
            mem.access_count += 1
            mem.ucb_visits += 1
            mem.last_accessed = datetime.now()
        
        return selected

    def retrieve_and_synthesize(self,
                                 query: str,
                                 goal: str,
                                 levels: List[MemoryLevel] = None,
                                 context_hints: str = "") -> str:
        """
        🧠 Brain-Inspired Synthesis Retrieval (DEFAULT mode!)

        Retrieves memories and synthesizes them into coherent wisdom.
        This is MORE intelligent than discrete retrieval:
        - Finds emergent patterns
        - Resolves contradictions
        - Creates causal insights
        - Returns integrated schema

        Returns:
            Synthesized wisdom as text (not discrete memories!)
        """
        if levels is None:
            levels = list(MemoryLevel)

        # Collect all candidates
        all_memories = []
        for level in levels:
            all_memories.extend(self.memories[level].values())

        if not all_memories:
            return ""

        # Use retriever to synthesize
        synthesized = self.retriever.retrieve_and_synthesize(
            query=query,
            goal=goal,
            memories=all_memories,
            context_hints=context_hints
        )

        return synthesized

    async def retrieve_and_synthesize_async(self,
                                             query: str,
                                             goal: str,
                                             levels: List[MemoryLevel] = None,
                                             context_hints: str = "") -> str:
        """
        Async version of retrieve_and_synthesize for parallel retrieval.
        """
        import asyncio
        loop = asyncio.get_event_loop()

        # Run the synchronous synthesis in a thread pool
        result = await loop.run_in_executor(
            None,
            lambda: self.retrieve_and_synthesize(query, goal, levels, context_hints)
        )
        return result

    def retrieve_for_context(
        self,
        query: str,
        goal: str,
        context_type: str,
        budget_tokens: int,
        context_hints: str = ""
    ) -> List[MemoryEntry]:
        """
        A-Team Enhancement: Context-aware memory retrieval.
        
        Different context types prioritize different memory levels:
        - validation: PROCEDURAL > META > SEMANTIC (how-to first)
        - debugging: CAUSAL > EPISODIC > SEMANTIC (why first)
        - planning: META > SEMANTIC > PROCEDURAL (wisdom first)
        - exploration: EPISODIC > CAUSAL > SEMANTIC (examples first)
        - transformation: PROCEDURAL > SEMANTIC > EPISODIC (steps first)
        
        Usage:
            memories = memory.retrieve_for_context(
                query="How to map bank_code column?",
                goal="column_mapping",
                context_type="transformation",
                budget_tokens=5000
            )
        """
        from ..foundation.data_structures import ContextType
        
        # Context-specific level priorities
        context_level_priorities = {
            ContextType.VALIDATION.value: [
                MemoryLevel.PROCEDURAL, MemoryLevel.META, MemoryLevel.SEMANTIC,
                MemoryLevel.CAUSAL, MemoryLevel.EPISODIC
            ],
            ContextType.DEBUGGING.value: [
                MemoryLevel.CAUSAL, MemoryLevel.EPISODIC, MemoryLevel.SEMANTIC,
                MemoryLevel.PROCEDURAL, MemoryLevel.META
            ],
            ContextType.PLANNING.value: [
                MemoryLevel.META, MemoryLevel.SEMANTIC, MemoryLevel.PROCEDURAL,
                MemoryLevel.CAUSAL, MemoryLevel.EPISODIC
            ],
            ContextType.EXPLORATION.value: [
                MemoryLevel.EPISODIC, MemoryLevel.CAUSAL, MemoryLevel.SEMANTIC,
                MemoryLevel.PROCEDURAL, MemoryLevel.META
            ],
            ContextType.TRANSFORMATION.value: [
                MemoryLevel.PROCEDURAL, MemoryLevel.SEMANTIC, MemoryLevel.EPISODIC,
                MemoryLevel.CAUSAL, MemoryLevel.META
            ],
            ContextType.DEFAULT.value: list(MemoryLevel)
        }
        
        # Get prioritized levels
        levels = context_level_priorities.get(context_type, list(MemoryLevel))
        
        # Add context type hint
        enhanced_hints = f"CONTEXT TYPE: {context_type}\n{context_hints}"
        
        return self.retrieve(
            query=query,
            goal=goal,
            budget_tokens=budget_tokens,
            levels=levels,
            context_hints=enhanced_hints
        )
    
    def retrieve_causal(self, query: str, context: Dict[str, Any]) -> List[CausalLink]:
        """
        Retrieve relevant causal knowledge.
        
        Returns causal links that apply in the given context.
        """
        if not self.config.enable_causal_learning:
            return []
        
        relevant = []
        
        # Keyword matching on cause/effect
        query_lower = query.lower()
        
        for link in self.causal_links.values():
            # Check if cause or effect matches query
            if (query_lower in link.cause.lower() or 
                query_lower in link.effect.lower() or
                any(kw in link.cause.lower() or kw in link.effect.lower() 
                    for kw in query_lower.split() if len(kw) > 3)):
                
                # Check if conditions apply
                if link.applies_in_context(context):
                    relevant.append(link)
        
        # Sort by confidence
        relevant.sort(key=lambda x: x.confidence, reverse=True)
        
        return relevant  # 🔥 NO LIMIT - FULL content
    
    # =========================================================================
    # CONSOLIDATION
    # =========================================================================
    
    async def consolidate(self, episodes: List[StoredEpisode] = None):
        """
        Run consolidation to extract higher-level knowledge.
        
        Episodic → Semantic (patterns)
        Episodic → Procedural (how-to)
        All → Meta (wisdom)
        Episodes → Causal (why)
        """
        self.consolidation_count += 1
        
        # 1. Cluster episodic memories by goal
        clusters = self._cluster_episodic_memories()
        
        # 2. Extract semantic patterns
        for cluster in clusters:
            if len(cluster.memories) >= self.config.min_cluster_size:
                await self._extract_semantic_pattern(cluster)
        
        # 3. Extract procedural knowledge
        if episodes:
            await self._extract_procedural(episodes)
        
        # 4. Extract meta wisdom
        await self._extract_meta_wisdom()
        
        # 5. Extract causal knowledge
        if episodes and self.config.enable_causal_learning:
            await self._extract_causal(episodes)
        
        # 6. Prune old episodic memories
        self._prune_episodic()
    
    def _cluster_episodic_memories(self) -> List[MemoryCluster]:
        """Cluster episodic memories by goal signature."""
        episodic = self.memories[MemoryLevel.EPISODIC]
        
        # Group by first goal in goal_values
        goal_groups: Dict[str, List[MemoryEntry]] = defaultdict(list)
        
        for mem in episodic.values():
            if mem.goal_values:
                goal = next(iter(mem.goal_values.keys()))
                # Create goal signature (first 50 chars + domain)
                domain = mem.context.get('domain', 'general')
                signature = f"{domain}:{goal}"
                goal_groups[signature].append(mem)
        
        # Create clusters
        clusters = []
        for sig, mems in goal_groups.items():
            cluster = MemoryCluster(
                cluster_id=hashlib.md5(sig.encode()).hexdigest(),
                goal_signature=sig,
                memories=mems
            )
            cluster.compute_statistics()
            clusters.append(cluster)
        
        return clusters
    
    async def _extract_semantic_pattern(self, cluster: MemoryCluster):
        """Extract semantic pattern from episodic cluster."""
        # Format memories for LLM
        memory_data = []
        for mem in cluster.memories:  # 🔥 NO LIMIT - FULL content
            memory_data.append({
                "content": mem.content,
                "value": mem.default_value,
                "success": mem.default_value > 0.5
            })
        
        try:
            result = self.pattern_extractor(
                memories=json.dumps(memory_data, indent=2),
                goal_context=cluster.goal_signature,
                domain=cluster.goal_signature.split(":")[0]
            )
            
            confidence = float(result.confidence) if result.confidence else 0.5
            
            if confidence >= self.config.pattern_confidence_threshold:
                # Store as semantic memory
                pattern_content = f"""
PATTERN: {result.pattern}

CONDITIONS: {result.conditions}

EXCEPTIONS: {result.exceptions}

DERIVED FROM: {len(cluster.memories)} episodic memories
CONFIDENCE: {confidence:.2f}
""".strip()
                
                # Get a representative goal
                sample_mem = cluster.memories[0]
                goal = next(iter(sample_mem.goal_values.keys()), "general")
                
                self.store(
                    content=pattern_content,
                    level=MemoryLevel.SEMANTIC,
                    context={
                        'cluster_id': cluster.cluster_id,
                        'source_count': len(cluster.memories),
                        'domain': cluster.goal_signature.split(":")[0]
                    },
                    goal=goal,
                    initial_value=cluster.avg_value
                )
                
                cluster.extracted_pattern = result.pattern
                cluster.pattern_confidence = confidence
                
        except Exception as e:
            pass  # Consolidation failure is non-fatal
    
    async def _extract_procedural(self, episodes: List[StoredEpisode]):
        """Extract procedural knowledge from episode trajectories."""
        # Separate success and failure
        successes = [ep for ep in episodes if ep.success]
        failures = [ep for ep in episodes if not ep.success]
        
        if len(successes) < 3 or len(failures) < 2:
            return  # Not enough data
        
        # Group by task type
        task_groups: Dict[str, Tuple[List, List]] = defaultdict(lambda: ([], []))
        
        for ep in successes:
            domain = ep.kwargs.get('domain', 'general')
            task_groups[domain][0].append(ep)
        
        for ep in failures:
            domain = ep.kwargs.get('domain', 'general')
            task_groups[domain][1].append(ep)
        
        for task_type, (succ, fail) in task_groups.items():
            if len(succ) < 2:
                continue
            
            # Format traces
            success_traces = self._format_traces(succ)
            failure_traces = self._format_traces(fail)
            
            try:
                result = self.procedural_extractor(
                    success_traces=success_traces,
                    failure_traces=failure_traces,
                    task_type=task_type
                )
                
                procedure_content = f"""
PROCEDURE FOR: {task_type}

STEPS:
{result.procedure}

KEY DECISIONS:
{result.key_decisions}

ANALYSIS:
{result.reasoning}
""".strip()
                
                self.store(
                    content=procedure_content,
                    level=MemoryLevel.PROCEDURAL,
                    context={'task_type': task_type, 'source_episodes': len(succ) + len(fail)},
                    goal=f"{task_type}_procedure",
                    initial_value=0.7  # Procedures start with moderate value
                )
                
            except Exception:
                pass
    
    async def _extract_meta_wisdom(self):
        """Extract meta-level wisdom about learning."""
        # Analyze patterns across all levels
        all_high_value = []
        all_low_value = []
        
        for level in [MemoryLevel.SEMANTIC, MemoryLevel.PROCEDURAL]:
            for mem in self.memories[level].values():
                if mem.default_value > 0.8:
                    all_high_value.append(mem.content)
                elif mem.default_value < 0.3:
                    all_low_value.append(mem.content)
        
        if len(all_high_value) < 3:
            return
        
        try:
            result = self.meta_extractor(
                learning_history=f"Episodes: {self.consolidation_count * 100}, High-value patterns: {len(all_high_value)}, Low-value: {len(all_low_value)}",
                failure_analysis="\n".join(all_low_value),
                success_analysis="\n".join(all_high_value)
            )
            
            wisdom_content = f"""
META WISDOM:
{result.wisdom}

WHEN TO APPLY:
{result.applicability}
""".strip()
            
            self.store(
                content=wisdom_content,
                level=MemoryLevel.META,
                context={'consolidation': self.consolidation_count},
                goal="meta_wisdom",
                initial_value=0.9  # Meta starts high
            )
            
            # Protect meta memories
            for mem in self.memories[MemoryLevel.META].values():
                mem.is_protected = True
                mem.protection_reason = "META level"
                
        except Exception:
            pass
    
    async def _extract_causal(self, episodes: List[StoredEpisode]):
        """Extract causal knowledge from contrasting episodes."""
        successes = [{'id': ep.episode_id, 'query': ep.goal, 'result': str(ep.actor_output)}
                     for ep in episodes if ep.success]
        failures = [{'id': ep.episode_id, 'query': ep.goal, 'result': str(ep.actor_error)}
                    for ep in episodes if not ep.success]
        
        if len(successes) < 3 or len(failures) < 2:
            return
        
        links = self.causal_extractor.extract_from_episodes(
            success_episodes=successes,
            failure_episodes=failures,
            domain=episodes[0].kwargs.get('domain', 'general')  # Generic default (not 'sql'!)
        )
        
        for link_data in links:
            link_key = hashlib.md5(f"{link_data['cause']}{link_data['effect']}".encode()).hexdigest()
            
            if link_key in self.causal_links:
                # Update existing
                existing = self.causal_links[link_key]
                existing.update_confidence(True)
                existing.supporting_episodes.append(episodes[0].episode_id)
            else:
                # Create new
                causal_link = CausalLink(
                    cause=link_data['cause'],
                    effect=link_data['effect'],
                    confidence=link_data.get('confidence', 0.7),
                    conditions=link_data.get('conditions', []),
                    domain=episodes[0].kwargs.get('domain', 'general')
                )
                self.causal_links[link_key] = causal_link
                
                # Also store as CAUSAL memory
                causal_content = f"""
CAUSAL LINK:
CAUSE: {causal_link.cause}
EFFECT: {causal_link.effect}
CONFIDENCE: {causal_link.confidence:.2f}
CONDITIONS: {', '.join(causal_link.conditions) if causal_link.conditions else 'None'}
""".strip()
                
                self.store(
                    content=causal_content,
                    level=MemoryLevel.CAUSAL,
                    context={'causal_key': link_key},
                    goal="causal_knowledge",
                    initial_value=causal_link.confidence,
                    causal_links=[link_key]
                )
    
    def _format_traces(self, episodes: List[StoredEpisode]) -> str:
        """Format episode traces for LLM."""
        traces = []
        for ep in episodes:
            trace = f"Episode {ep.episode_id}:\n"
            trace += f"  Goal: {ep.goal}\n"
            trace += f"  Steps: {len(ep.trajectory)}\n"
            trace += f"  Result: {'SUCCESS' if ep.success else 'FAILURE'}\n"
            if ep.actor_error:
                trace += f"  Error: {ep.actor_error}\n"
            traces.append(trace)
        return "\n".join(traces)
    
    def _prune_episodic(self):
        """Prune old low-value episodic memories."""
        episodic = self.memories[MemoryLevel.EPISODIC]
        
        # Keep top 80% by value, all less than 1 day old
        now = datetime.now()
        one_day_ago = now - timedelta(days=1)
        
        to_remove = []
        for key, mem in episodic.items():
            if mem.created_at < one_day_ago and mem.default_value < 0.3:
                to_remove.append(key)
        
        # Remove up to 20%
        max_remove = len(episodic) // 5
        for key in to_remove[:max_remove]:
            del episodic[key]
    
    # =========================================================================
    # PROTECTION
    # =========================================================================
    
    def protect_high_value(self, threshold: float = None):
        """Mark high-value memories as protected."""
        threshold = threshold or self.config.protected_memory_threshold
        
        for level, memories in self.memories.items():
            for mem in memories.values():
                if mem.default_value >= threshold:
                    mem.is_protected = True
                    mem.protection_reason = f"Value >= {threshold}"
                elif level == MemoryLevel.META:
                    mem.is_protected = True
                    mem.protection_reason = "META level"
                elif level == MemoryLevel.CAUSAL:
                    mem.is_protected = True
                    mem.protection_reason = "CAUSAL knowledge"
    
    # =========================================================================
    # SERIALIZATION
    # =========================================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        data = {
            'agent_name': self.agent_name,
            'total_accesses': self.total_accesses,
            'consolidation_count': self.consolidation_count,
            'memories': {},
            'causal_links': {},
            'goal_hierarchy': {
                'nodes': {k: vars(v) for k, v in self.goal_hierarchy.nodes.items()},
                'root_id': self.goal_hierarchy.root_id
            }
        }
        
        for level in MemoryLevel:
            data['memories'][level.value] = {}
            for key, mem in self.memories[level].items():
                mem_dict = {
                    'key': mem.key,
                    'content': mem.content,
                    'level': mem.level.value,
                    'context': mem.context,
                    'created_at': mem.created_at.isoformat(),
                    'last_accessed': mem.last_accessed.isoformat(),
                    'goal_values': {
                        g: {'value': gv.value, 'access_count': gv.access_count}
                        for g, gv in mem.goal_values.items()
                    },
                    'default_value': mem.default_value,
                    'access_count': mem.access_count,
                    'ucb_visits': mem.ucb_visits,
                    'token_count': mem.token_count,
                    'is_protected': mem.is_protected,
                    'protection_reason': mem.protection_reason,
                    'causal_links': mem.causal_links,
                    'metadata': getattr(mem, 'metadata', {})  # Include metadata for domain/task_type
                }
                data['memories'][level.value][key] = mem_dict
        
        for key, link in self.causal_links.items():
            data['causal_links'][key] = {
                'cause': link.cause,
                'effect': link.effect,
                'confidence': link.confidence,
                'conditions': link.conditions,
                'exceptions': link.exceptions,
                'domain': link.domain
            }
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], config: JottyConfig) -> 'HierarchicalMemory':
        """
        Deserialize from dictionary with automatic key migration.
        
        Migrates old-format keys (hash only) to new hierarchical format (domain:task_type:hash).
        """
        memory = cls(data['agent_name'], config)
        memory.total_accesses = data.get('total_accesses', 0)
        memory.consolidation_count = data.get('consolidation_count', 0)
        
        # Load memories with automatic migration
        for level_str, memories in data.get('memories', {}).items():
            level = MemoryLevel(level_str)
            for key, mem_data in memories.items():
                # Check if old format (no colons = old hash-only key)
                if ':' not in key:
                    # Old format - migrate to new format
                    content = mem_data['content']
                    domain = mem_data.get('context', {}).get('domain', 'general')
                    task_type = mem_data.get('context', {}).get('task_type') or \
                               mem_data.get('context', {}).get('operation_type', 'general')
                    
                    # Generate new hierarchical key
                    content_hash = hashlib.md5(content.encode()).hexdigest()[:16]
                    new_key = f"{domain}:{task_type}:{content_hash}"
                    
                    # Update key in memory data
                    mem_data['key'] = new_key
                    key = new_key
                    
                    # Update metadata
                    if 'metadata' not in mem_data:
                        mem_data['metadata'] = {}
                    mem_data['metadata']['domain'] = domain
                    mem_data['metadata']['task_type'] = task_type
                
                entry = MemoryEntry(
                    key=mem_data['key'],
                    content=mem_data['content'],
                    level=level,
                    context=mem_data['context'],
                    created_at=datetime.fromisoformat(mem_data['created_at']),
                    last_accessed=datetime.fromisoformat(mem_data['last_accessed']),
                    default_value=mem_data['default_value'],
                    access_count=mem_data['access_count'],
                    ucb_visits=mem_data['ucb_visits'],
                    token_count=mem_data['token_count'],
                    is_protected=mem_data.get('is_protected', False),
                    protection_reason=mem_data.get('protection_reason', ''),
                    causal_links=mem_data.get('causal_links', [])
                )
                
                # Restore metadata if present
                if 'metadata' in mem_data:
                    entry.metadata = mem_data['metadata']
                
                # Restore goal values
                for goal, gv_data in mem_data.get('goal_values', {}).items():
                    entry.goal_values[goal] = GoalValue(
                        value=gv_data['value'],
                        access_count=gv_data['access_count']
                    )
                
                memory.memories[level][key] = entry
        
        # Load causal links
        for key, link_data in data.get('causal_links', {}).items():
            memory.causal_links[key] = CausalLink(
                cause=link_data['cause'],
                effect=link_data['effect'],
                confidence=link_data['confidence'],
                conditions=link_data.get('conditions', []),
                exceptions=link_data.get('exceptions', []),
                domain=link_data.get('domain', 'general')
            )
        
        # Load goal hierarchy
        gh_data = data.get('goal_hierarchy', {})
        memory.goal_hierarchy.root_id = gh_data.get('root_id', 'root')
        for node_id, node_data in gh_data.get('nodes', {}).items():
            memory.goal_hierarchy.nodes[node_id] = GoalNode(
                goal_id=node_data['goal_id'],
                goal_text=node_data['goal_text'],
                parent_id=node_data.get('parent_id'),
                children_ids=node_data.get('children_ids', []),
                domain=node_data.get('domain', 'general'),
                operation_type=node_data.get('operation_type', 'query'),
                entities=node_data.get('entities', []),
                episode_count=node_data.get('episode_count', 0),
                success_rate=node_data.get('success_rate', 0.5)
            )
        
        return memory
    
    # =========================================================================
    # STATISTICS
    # =========================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics."""
        stats = {
            'total_memories': sum(len(m) for m in self.memories.values()),
            'by_level': {level.value: len(mems) for level, mems in self.memories.items()},
            'total_accesses': self.total_accesses,
            'consolidation_count': self.consolidation_count,
            'causal_links': len(self.causal_links),
            'unique_goals': len(self.goal_hierarchy.nodes),
            'protected_memories': sum(
                1 for mems in self.memories.values() 
                for m in mems.values() if m.is_protected
            )
        }
        return stats
    
    def get_consolidated_knowledge(self, goal: str = None, max_items: int = 10) -> str:
        """
        Get consolidated knowledge to inject into prompts.
        
        THIS IS HOW MEMORY CONSOLIDATION MANIFESTS IN LLM AGENTS!
        
        Returns natural language lessons from semantic/procedural/meta/causal levels.
        """
        consolidated = []
        
        # Semantic patterns (abstracted learnings)
        semantic_mems = self.memories.get(MemoryLevel.SEMANTIC, {})
        if semantic_mems:
            # Sort by value if goal provided
            if goal:
                sorted_semantic = sorted(
                    semantic_mems.values(),
                    key=lambda m: m.get_value(goal),
                    reverse=True
                )
            else:
                # Sort by access count
                sorted_semantic = sorted(
                    semantic_mems.values(),
                    key=lambda m: m.access_count,
                    reverse=True
                )
            
            for mem in sorted_semantic[:max_items//2]:
                consolidated.append(('PATTERN', mem.content, mem.get_value(goal) if goal else 0.5))
        
        # Procedural knowledge (how-to)
        procedural_mems = self.memories.get(MemoryLevel.PROCEDURAL, {})
        if procedural_mems:
            sorted_procedural = sorted(
                procedural_mems.values(),
                key=lambda m: m.access_count,
                reverse=True
            )
            for mem in sorted_procedural[:max_items//4]:
                consolidated.append(('PROCEDURE', mem.content, 0.5))
        
        # Meta wisdom
        meta_mems = self.memories.get(MemoryLevel.META, {})
        if meta_mems:
            for mem in list(meta_mems.values())[:max_items//4]:
                consolidated.append(('WISDOM', mem.content, 0.5))
        
        # Causal knowledge (WHY things work)
        if self.causal_links:
            sorted_causal = sorted(
                self.causal_links.values(),
                key=lambda link: link.confidence,
                reverse=True
            )
            for link in sorted_causal[:3]:
                causal_str = f"CAUSE: {link.cause} → EFFECT: {link.effect}"
                if link.conditions:
                    causal_str += f" (when: {', '.join(link.conditions)})"
                consolidated.append(('CAUSAL', causal_str, link.confidence))
        
        if not consolidated:
            return ""
        
        # Format as natural language
        context = "# Consolidated Knowledge (Long-Term Memory):\n"
        
        # Group by type
        patterns = [c for c in consolidated if c[0] == 'PATTERN']
        procedures = [c for c in consolidated if c[0] == 'PROCEDURE']
        wisdom = [c for c in consolidated if c[0] == 'WISDOM']
        causal = [c for c in consolidated if c[0] == 'CAUSAL']
        
        if patterns:
            context += "\n## Learned Patterns:\n"
            for _, content, value in patterns:
                if goal and value > 0:
                    context += f"- {content[:200]}... (value: {value:.2f})\n"
                else:
                    context += f"- {content[:200]}...\n"
        
        if procedures:
            context += "\n## Procedural Knowledge:\n"
            for _, content, _ in procedures:
                context += f"- {content[:200]}...\n"
        
        if wisdom:
            context += "\n## Meta Wisdom:\n"
            for _, content, _ in wisdom:
                context += f"- {content[:200]}...\n"
        
        if causal:
            context += "\n## Causal Understanding (WHY things work):\n"
            for _, content, conf in causal:
                context += f"- {content} (confidence: {conf:.2f})\n"
        
        return context