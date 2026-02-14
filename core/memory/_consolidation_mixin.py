"""SwarmMemory mixin — consolidation, serialization, and statistics."""
from __future__ import annotations

import json
import hashlib
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)

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


class ConsolidationMixin:
    """Mixin providing consolidation, serialization, and statistics."""

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
    
    def _cluster_episodic_memories(self) -> List:
        """Cluster episodic memories by goal signature."""
        cons = _get_consolidation()
        _MemoryCluster = cons['MemoryCluster']
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
            cluster = _MemoryCluster(
                cluster_id=hashlib.md5(sig.encode()).hexdigest(),
                goal_signature=sig,
                memories=mems
            )
            cluster.compute_statistics()
            clusters.append(cluster)
        
        return clusters
    
    async def _extract_semantic_pattern(self, cluster):
        """Extract semantic pattern from episodic cluster."""
        # Format memories for LLM
        memory_data = []
        for mem in cluster.memories: # NO LIMIT - FULL content
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
    
    def protect_high_value(self, threshold: float = None) -> None:
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
    def from_dict(cls, data: Dict[str, Any], config: SwarmConfig) -> 'SwarmMemory':
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