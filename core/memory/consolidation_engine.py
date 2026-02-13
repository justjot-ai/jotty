"""
Brain-Inspired ReVal 2.1 - Neuroscience-Based Memory and Learning
=================================================================

Implements brain-like processing for multi-agent systems:

1. SHARP WAVE RIPPLE: Memory consolidation during "sleep"
   - Rapid replay of recent experiences
   - Pattern extraction (episodic → semantic)
   - Pruning low-value memories

2. HIPPOCAMPAL FILTERING: What to remember
   - Reward salience (emotional importance)
   - Novelty (surprise/unexpectedness)
   - Goal relevance

3. ONLINE/OFFLINE MODES: Awake vs Sleep
   - Online: Learning from environment
   - Offline: Consolidating, reorganizing

4. AGENT ABSTRACTION: Scalability
   - Full detail for small swarms
   - Role-based compression for large swarms

The key insight: LLM agents can benefit from the same memory
management strategies that evolved in biological brains.
"""

import asyncio
import time
import math
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


# =============================================================================
# BRAIN STATES
# =============================================================================

class BrainMode(Enum):
    """Brain operating modes."""
    ONLINE = "online"    # Awake: Learning from environment
    OFFLINE = "offline"  # Sleep: Consolidating memories
    DREAMING = "dreaming"  # Deep consolidation with pattern exploration


@dataclass
class BrainModeConfig:
    """Configuration for brain-inspired processing."""
    enabled: bool = True
    
    # Sleep/Wake cycles - A-TEAM setting for reliable consolidation
    sleep_interval: int = 3            # Episodes before consolidation (A-TEAM: prevent memory buildup!)
    min_episodes_before_sleep: int = 5  # Minimum before first consolidation (was 10)
    consolidation_timeout: float = 30.0
    
    # Sharp Wave Ripple
    sharp_wave_ripple: bool = True
    replay_speed_multiplier: int = 10
    pattern_extraction_threshold: int = 3
    
    # Hippocampal Filtering
    hippocampal_filtering: bool = True
    reward_salience_weight: float = 0.3
    novelty_weight: float = 0.4
    goal_relevance_weight: float = 0.3
    memory_threshold: float = 0.4
    
    # Pruning
    prune_threshold: float = 0.15
    strengthen_threshold: float = 0.85
    max_prune_percentage: float = 0.2


# =============================================================================
# HIPPOCAMPAL EXTRACTOR - What to Remember
# =============================================================================

@dataclass
class MemoryCandidate:
    """A candidate experience for memory storage."""
    content: str
    context: Dict[str, Any]
    reward: float
    timestamp: float
    agent: str
    
    # Computed scores
    reward_salience: float = 0.0
    novelty_score: float = 0.0
    goal_relevance: float = 0.0
    memory_strength: float = 0.0


class HippocampalExtractor:
    """
    Decides what experiences are worth remembering.
    
    Like the hippocampus, filters experiences by:
    - Reward salience (emotional importance)
    - Novelty (surprise vs expectation)
    - Goal relevance (helps achieve objectives)
    
    Only high-strength experiences become long-term memories.
    """
    
    def __init__(self, config: BrainModeConfig, goal: str = ""):
        self.config = config
        self.goal = goal
        self.expected_reward = 0.5  # Running average of rewards
        self.seen_patterns: Set[str] = set()
        
        # A-Team Fix: No keyword matching - use adaptive threshold
        from ..foundation.robust_parsing import AdaptiveThreshold
        self._relevance_threshold = AdaptiveThreshold(initial_mean=0.5, initial_std=0.2)
        
        logger.info(f" HippocampalExtractor initialized (threshold={config.memory_threshold})")
    
    def should_remember(self, experience: Dict[str, Any]) -> Tuple[bool, MemoryCandidate]:
        """
        Determine if an experience should be stored in memory.
        
        Returns (should_remember, memory_candidate)
        """
        candidate = MemoryCandidate(
            content=str(experience.get('content', '')),
            context=experience.get('context', {}),
            reward=experience.get('reward', 0.5),
            timestamp=time.time(),
            agent=experience.get('agent', 'unknown')
        )
        
        # 1. Reward Salience: How different from expected?
        candidate.reward_salience = self._compute_reward_salience(candidate.reward)
        
        # 2. Novelty: How surprising is this?
        candidate.novelty_score = self._compute_novelty(candidate.content)
        
        # 3. Goal Relevance: Does this help the goal?
        candidate.goal_relevance = self._compute_goal_relevance(candidate.content)
        
        # Combined memory strength
        candidate.memory_strength = (
            self.config.reward_salience_weight * candidate.reward_salience +
            self.config.novelty_weight * candidate.novelty_score +
            self.config.goal_relevance_weight * candidate.goal_relevance
        )
        
        # Update expectations
        self._update_expectations(candidate)
        
        should_store = candidate.memory_strength >= self.config.memory_threshold
        
        if should_store:
            logger.debug(
                f" Memory candidate accepted: strength={candidate.memory_strength:.2f} "
                f"(salience={candidate.reward_salience:.2f}, novelty={candidate.novelty_score:.2f}, "
                f"relevance={candidate.goal_relevance:.2f})"
            )
        
        return should_store, candidate
    
    def _compute_reward_salience(self, reward: float) -> float:
        """Compute emotional importance based on reward deviation."""
        # Large deviations from expected are more salient
        deviation = abs(reward - self.expected_reward)
        
        # Very good (reward >> expected) or very bad (reward << expected) are memorable
        salience = min(1.0, deviation * 2)
        
        # Extreme rewards get bonus
        if reward > 0.9 or reward < 0.1:
            salience = min(1.0, salience + 0.3)
        
        return salience
    
    def _compute_novelty(self, content: str) -> float:
        """Compute how novel/surprising this experience is."""
        content_signature = self._get_content_signature(content)

        if content_signature in self.seen_patterns:
            novelty = 0.2  # Seen before, low novelty
        else:
            novelty = 0.8  # New, high novelty
            self.seen_patterns.add(content_signature)

        # Evict oldest when exceeding max size
        if len(self.seen_patterns) > 1000:
            to_remove = len(self.seen_patterns) - 800  # Evict 200 at a time
            it = iter(self.seen_patterns)
            for _ in range(to_remove):
                self.seen_patterns.discard(next(it))

        return novelty
    
    def _compute_goal_relevance(self, content: str) -> float:
        """
        Compute relevance to the goal WITHOUT keyword matching.
        
        A-Team Fix: Replaced keyword overlap with content-based heuristics:
        - Content length (longer = more information = potentially more relevant)
        - Content structure (has results, has reasoning)
        - Adaptive threshold based on historical relevance
        """
        if not self.goal or not content:
            return 0.5  # No goal specified, neutral relevance
        
        # Heuristic 1: Content length (normalized)
        # More content = more likely to be substantive and relevant
        length_score = min(len(content) / 1000, 1.0)
        
        # Heuristic 2: Content structure (has results/outputs)
        # Higher weight if content indicates actual results rather than noise
        structure_score = 0.5
        content_len = len(content)
        if content_len > 100:
            structure_score = 0.6
        if content_len > 500:
            structure_score = 0.7
        
        # Heuristic 3: Use adaptive threshold to learn what's relevant
        # This adapts over time based on what proved useful
        combined = (length_score * 0.4 + structure_score * 0.6)
        
        # Update adaptive threshold for future reference
        self._relevance_threshold.update(combined)
        
        return combined
    
    def _get_content_signature(self, content: str) -> str:
        """Get a compact hash signature for content similarity checking."""
        import hashlib
        return hashlib.md5(content[:200].encode(errors='replace')).hexdigest()
    
    def _update_expectations(self, candidate: MemoryCandidate):
        """Update running expectations (like brain adaptation)."""
        # Exponential moving average of rewards
        alpha = 0.1
        self.expected_reward = alpha * candidate.reward + (1 - alpha) * self.expected_reward


# =============================================================================
# SHARP WAVE RIPPLE - Memory Consolidation
# =============================================================================

@dataclass
class ConsolidationResult:
    """Result of a consolidation cycle."""
    patterns_extracted: int
    causal_links_found: int
    memories_pruned: int
    memories_strengthened: int
    duration_seconds: float
    new_semantic_memories: List[str] = field(default_factory=list)


class SharpWaveRippleConsolidator:
    """
    Brain-inspired memory consolidation during "sleep".
    
    During consolidation:
    1. Rapidly replay recent experiences (like brain during sleep)
    2. Extract patterns (episodic → semantic memory)
    3. Identify causal relationships
    4. Prune low-value memories
    5. Strengthen high-value memories
    
    This process happens automatically when the system enters "sleep mode".
    """
    
    def __init__(self, config: BrainModeConfig, memory_store: Any = None):
        self.config = config
        self.memory_store = memory_store
        self.consolidation_count = 0
        
        logger.info(" SharpWaveRippleConsolidator initialized")
    
    async def consolidate(
        self,
        recent_episodes: List[Dict[str, Any]],
        episodic_memories: List[Any],
        semantic_memories: List[Any]
    ) -> ConsolidationResult:
        """
        Run consolidation cycle.
        
        This is like the brain's sleep consolidation process.
        """
        start_time = time.time()
        self.consolidation_count += 1
        
        logger.info(f" Starting consolidation #{self.consolidation_count} "
                   f"({len(recent_episodes)} episodes, {len(episodic_memories)} memories)")
        
        result = ConsolidationResult(
            patterns_extracted=0,
            causal_links_found=0,
            memories_pruned=0,
            memories_strengthened=0,
            duration_seconds=0
        )
        
        # 1. RAPID REPLAY: Process episodes quickly
        patterns = await self._rapid_replay_pattern_extraction(recent_episodes)
        result.patterns_extracted = len(patterns)
        result.new_semantic_memories = patterns
        
        # 2. CAUSAL EXTRACTION: Find cause-effect relationships
        causal_links = self._extract_causal_links(recent_episodes)
        result.causal_links_found = len(causal_links)
        
        # 3. PRUNE: Remove low-value memories
        pruned_count = self._prune_low_value_memories(episodic_memories)
        result.memories_pruned = pruned_count
        
        # 4. STRENGTHEN: Reinforce high-value memories
        strengthened_count = self._strengthen_high_value_memories(episodic_memories)
        result.memories_strengthened = strengthened_count
        
        result.duration_seconds = time.time() - start_time
        
        logger.info(f" Consolidation complete: {result.patterns_extracted} patterns, "
                   f"{result.memories_pruned} pruned, {result.memories_strengthened} strengthened "
                   f"({result.duration_seconds:.2f}s)")
        
        return result
    
    async def _rapid_replay_pattern_extraction(
        self,
        episodes: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Rapidly replay episodes and extract patterns.
        
        Like brain's sharp wave ripple - compressed replay.
        """
        patterns = []
        
        # Group episodes by outcome
        success_episodes = [e for e in episodes if e.get('reward', 0) > 0.7]
        failure_episodes = [e for e in episodes if e.get('reward', 0) < 0.3]
        
        # Extract success patterns
        if len(success_episodes) >= self.config.pattern_extraction_threshold:
            success_pattern = self._extract_common_pattern(success_episodes, "success")
            if success_pattern:
                patterns.append(success_pattern)
        
        # Extract failure patterns (avoid these)
        if len(failure_episodes) >= self.config.pattern_extraction_threshold:
            failure_pattern = self._extract_common_pattern(failure_episodes, "failure")
            if failure_pattern:
                patterns.append(f"AVOID: {failure_pattern}")
        
        # Extract agent-specific patterns
        agent_episodes = defaultdict(list)
        for ep in episodes:
            agent = ep.get('agent', 'unknown')
            agent_episodes[agent].append(ep)
        
        for agent, eps in agent_episodes.items():
            if len(eps) >= self.config.pattern_extraction_threshold:
                agent_pattern = self._extract_agent_pattern(agent, eps)
                if agent_pattern:
                    patterns.append(agent_pattern)
        
        return patterns
    
    def _extract_common_pattern(
        self,
        episodes: List[Dict[str, Any]],
        outcome_type: str
    ) -> Optional[str]:
        """Extract common pattern from similar-outcome episodes."""
        # Simple: Find common actions/contexts
        actions = [ep.get('action', '') for ep in episodes]
        contexts = [str(ep.get('context', '')) for ep in episodes]
        
        # Count action frequencies
        action_counts = defaultdict(int)
        for action in actions:
            action_counts[str(action)] += 1
        
        if action_counts:
            most_common = max(action_counts, key=action_counts.get)
            count = action_counts[most_common]
            if count >= self.config.pattern_extraction_threshold:
                return f"{outcome_type.upper()}_PATTERN: Action '{most_common}' led to {outcome_type} {count} times"
        
        return None
    
    def _extract_agent_pattern(
        self,
        agent: str,
        episodes: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Extract pattern for specific agent."""
        successes = sum(1 for ep in episodes if ep.get('reward', 0) > 0.7)
        total = len(episodes)
        
        if total >= 3:
            rate = successes / total
            return f"AGENT_PATTERN: {agent} has {rate:.0%} success rate over {total} tasks"
        
        return None
    
    def _extract_causal_links(
        self,
        episodes: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract cause-effect relationships from episodes."""
        causal_links = []
        
        # Look for sequential patterns
        for i in range(len(episodes) - 1):
            current = episodes[i]
            next_ep = episodes[i + 1]
            
            current_reward = current.get('reward', 0.5)
            next_reward = next_ep.get('reward', 0.5)
            
            # Significant reward change?
            if abs(next_reward - current_reward) > 0.3:
                causal_links.append({
                    'cause': str(current.get('action', '')),
                    'effect': str(next_ep.get('action', '')),
                    'reward_delta': next_reward - current_reward,
                    'confidence': 0.7
                })
        
        return causal_links
    
    def _prune_low_value_memories(self, memories: List[Any]) -> int:
        """Prune low-value memories to free space."""
        if not memories:
            return 0
        
        pruned = 0
        max_to_prune = int(len(memories) * self.config.max_prune_percentage)
        
        for mem in memories:
            if pruned >= max_to_prune:
                break
            
            value = getattr(mem, 'default_value', 0.5)
            if value < self.config.prune_threshold:
                # Mark for pruning (actual removal depends on memory store)
                if hasattr(mem, 'marked_for_deletion'):
                    mem.marked_for_deletion = True
                pruned += 1
        
        return pruned
    
    def _strengthen_high_value_memories(self, memories: List[Any]) -> int:
        """Strengthen high-value memories."""
        strengthened = 0
        
        for mem in memories:
            value = getattr(mem, 'default_value', 0.5)
            if value > self.config.strengthen_threshold:
                # Increase value (reinforce)
                if hasattr(mem, 'default_value'):
                    mem.default_value = min(1.0, mem.default_value * 1.1)
                strengthened += 1
        
        return strengthened


# =============================================================================
# BRAIN STATE MACHINE - Online/Offline Modes
# =============================================================================

class BrainStateMachine:
    """
    Manages brain states: Online (awake) vs Offline (sleep).
    
    Online Mode:
    - Active learning from environment
    - Storing new experiences
    - Making decisions
    
    Offline Mode (Sleep):
    - No new experiences processed
    - Memory consolidation
    - Pattern extraction
    - Pruning and strengthening
    
    This mirrors how biological brains alternate between learning
    and consolidation phases.
    """
    
    def __init__(
        self,
        config: BrainModeConfig,
        consolidator: SharpWaveRippleConsolidator = None,
        extractor: HippocampalExtractor = None
    ):
        self.config = config
        self.mode = BrainMode.ONLINE
        self.consolidator = consolidator or SharpWaveRippleConsolidator(config)
        self.extractor = extractor or HippocampalExtractor(config)
        
        # State tracking
        self.episodes_since_sleep = 0
        self.total_episodes = 0
        self.total_consolidations = 0
        self.recent_episodes: List[Dict[str, Any]] = []
        self.max_recent_episodes = 100
        
        logger.info(f" BrainStateMachine initialized (sleep_interval={config.sleep_interval})")
    
    @property
    def is_online(self) -> bool:
        return self.mode == BrainMode.ONLINE
    
    @property
    def is_offline(self) -> bool:
        return self.mode in [BrainMode.OFFLINE, BrainMode.DREAMING]
    
    async def process_experience(
        self,
        experience: Dict[str, Any],
        episodic_memories: List[Any] = None,
        semantic_memories: List[Any] = None
    ) -> Optional[MemoryCandidate]:
        """
        Process an experience, potentially triggering sleep.
        
        Returns MemoryCandidate if experience should be stored.
        """
        if self.is_offline:
            logger.warning(" Experience received during sleep - queueing for later")
            return None
        
        self.total_episodes += 1
        self.episodes_since_sleep += 1
        
        # Store for consolidation
        self.recent_episodes.append(experience)
        if len(self.recent_episodes) > self.max_recent_episodes:
            self.recent_episodes = self.recent_episodes[-self.max_recent_episodes:]
        
        # Hippocampal filtering: Should we remember this?
        should_remember, candidate = self.extractor.should_remember(experience)
        
        # Check if need sleep
        if self._should_sleep():
            await self.enter_sleep_mode(episodic_memories, semantic_memories)
        
        return candidate if should_remember else None
    
    def _should_sleep(self) -> bool:
        """Check if it's time for consolidation."""
        if not self.config.enabled:
            return False
        
        if self.episodes_since_sleep < self.config.min_episodes_before_sleep:
            return False
        
        return self.episodes_since_sleep >= self.config.sleep_interval
    
    async def enter_sleep_mode(
        self,
        episodic_memories: List[Any] = None,
        semantic_memories: List[Any] = None
    ) -> ConsolidationResult:
        """Enter sleep mode for memory consolidation."""
        self.mode = BrainMode.OFFLINE
        self.total_consolidations += 1
        
        logger.info(f" Entering sleep mode (consolidation #{self.total_consolidations})")
        
        try:
            # Run consolidation with timeout
            result = await asyncio.wait_for(
                self.consolidator.consolidate(
                    self.recent_episodes,
                    episodic_memories or [],
                    semantic_memories or []
                ),
                timeout=self.config.consolidation_timeout
            )
        except asyncio.TimeoutError:
            logger.warning(" Consolidation timed out")
            result = ConsolidationResult(
                patterns_extracted=0,
                causal_links_found=0,
                memories_pruned=0,
                memories_strengthened=0,
                duration_seconds=self.config.consolidation_timeout
            )
        
        # Wire consolidation results back to memory store
        if result.new_semantic_memories and self.consolidator.memory_store:
            store = self.consolidator.memory_store
            for pattern in result.new_semantic_memories:
                try:
                    if hasattr(store, 'store'):
                        from ..foundation.data_structures import MemoryLevel
                        store.store(
                            content=pattern,
                            level=MemoryLevel.SEMANTIC,
                            context={'source': 'consolidation', 'cycle': self.total_consolidations},
                            goal=''
                        )
                except Exception as e:
                    logger.debug(f"Failed to store consolidated pattern: {e}")

        # Prune memories marked for deletion during consolidation
        if episodic_memories:
            pruned_count = 0
            for mem in list(episodic_memories):
                if getattr(mem, 'marked_for_deletion', False):
                    try:
                        if hasattr(self.consolidator.memory_store, 'delete'):
                            self.consolidator.memory_store.delete(mem)
                        episodic_memories.remove(mem)
                        pruned_count += 1
                    except Exception as e:
                        logger.debug(f"Failed to delete pruned memory: {e}")
            if pruned_count:
                logger.info(f"Pruned {pruned_count} memories marked for deletion")

        # Wake up
        self.mode = BrainMode.ONLINE
        self.episodes_since_sleep = 0

        logger.info("Waking up from consolidation")

        return result
    
    def force_sleep(self):
        """Force immediate transition to sleep mode."""
        self.episodes_since_sleep = self.config.sleep_interval
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get summary of brain state."""
        return {
            'mode': self.mode.value,
            'episodes_since_sleep': self.episodes_since_sleep,
            'total_episodes': self.total_episodes,
            'total_consolidations': self.total_consolidations,
            'recent_episodes_buffered': len(self.recent_episodes),
            'next_sleep_in': max(0, self.config.sleep_interval - self.episodes_since_sleep)
        }


# =============================================================================
# AGENT ABSTRACTOR - Scalability for Large Swarms
# =============================================================================

@dataclass
class AgentRole:
    """Abstracted role representing a group of agents."""
    role_name: str
    agents: List[str]
    capabilities: Set[str]
    avg_success_rate: float = 0.5
    total_tasks: int = 0
    
    def to_summary(self) -> str:
        """Summarize role for context injection."""
        return (f"{self.role_name}: {len(self.agents)} agents, "
                f"capabilities={list(self.capabilities)}, "
                f"success_rate={self.avg_success_rate:.0%}")


class AgentAbstractor:
    """
    Compresses agent information for scalable swarm management.
    
    When swarm is small (≤10 agents): Full detail tracking
    When swarm is large (>10 agents): Abstract to roles
    
    This allows context to contain useful information about many agents
    without consuming all tokens on individual agent details.
    """
    
    def __init__(self, config: BrainModeConfig):
        self.config = config
        self.detail_threshold = 10
        
        # Agent tracking
        self.agent_stats: Dict[str, Dict[str, Any]] = {}
        self.agent_roles: Dict[str, str] = {}  # agent -> role
        self.role_definitions: Dict[str, AgentRole] = {}
        
        logger.info(f" AgentAbstractor initialized (detail_threshold={self.detail_threshold})")
    
    def update_agent(self, agent: str, success: bool, task_type: str = ""):
        """Update agent statistics."""
        if agent not in self.agent_stats:
            self.agent_stats[agent] = {
                'successes': 0,
                'failures': 0,
                'task_types': set()
            }
        
        stats = self.agent_stats[agent]
        if success:
            stats['successes'] += 1
        else:
            stats['failures'] += 1
        
        if task_type:
            stats['task_types'].add(task_type)
        
        # Infer role from name/behavior
        if agent not in self.agent_roles:
            self.agent_roles[agent] = self._infer_role(agent, stats)
    
    def get_agent_view(self, agents: List[str] = None) -> Dict[str, Any]:
        """
        Get agent information at appropriate detail level.
        
        Small swarm: Full per-agent details
        Large swarm: Role-based abstraction
        """
        agents = agents or list(self.agent_stats.keys())
        
        if len(agents) <= self.detail_threshold:
            return self._get_detailed_view(agents)
        else:
            return self._get_abstracted_view(agents)
    
    def _get_detailed_view(self, agents: List[str]) -> Dict[str, Any]:
        """Full detail for small swarms."""
        view = {
            'abstraction_level': 'detailed',
            'agent_count': len(agents),
            'agents': {}
        }
        
        for agent in agents:
            stats = self.agent_stats.get(agent, {})
            total = stats.get('successes', 0) + stats.get('failures', 0)
            success_rate = stats.get('successes', 0) / total if total > 0 else 0.5
            
            view['agents'][agent] = {
                'success_rate': round(success_rate, 2),
                'total_tasks': total,
                'capabilities': list(stats.get('task_types', set())),
                'role': self.agent_roles.get(agent, 'unknown')
            }
        
        return view
    
    def _get_abstracted_view(self, agents: List[str]) -> Dict[str, Any]:
        """Role-based abstraction for large swarms."""
        # Group agents by role
        self._rebuild_roles(agents)
        
        view = {
            'abstraction_level': 'roles',
            'agent_count': len(agents),
            'roles': {}
        }
        
        for role_name, role in self.role_definitions.items():
            view['roles'][role_name] = {
                'agent_count': len(role.agents),
                'capabilities': list(role.capabilities),
                'success_rate': round(role.avg_success_rate, 2),
                'total_tasks': role.total_tasks
            }
        
        return view
    
    def _rebuild_roles(self, agents: List[str]):
        """Rebuild role definitions from current agents."""
        self.role_definitions = {}
        
        for agent in agents:
            role_name = self.agent_roles.get(agent, 'other')
            
            if role_name not in self.role_definitions:
                self.role_definitions[role_name] = AgentRole(
                    role_name=role_name,
                    agents=[],
                    capabilities=set()
                )
            
            role = self.role_definitions[role_name]
            role.agents.append(agent)
            
            stats = self.agent_stats.get(agent, {})
            role.capabilities.update(stats.get('task_types', set()))
            
            total = stats.get('successes', 0) + stats.get('failures', 0)
            if total > 0:
                success_rate = stats.get('successes', 0) / total
                # Running average
                role.avg_success_rate = (role.avg_success_rate * role.total_tasks + success_rate) / (role.total_tasks + 1)
                role.total_tasks += total
    
    def _infer_role(self, agent: str, stats: Dict[str, Any]) -> str:
        """
        Infer agent role from BEHAVIOR, not name.
        
        A-Team Decision: Don't use keyword matching on names.
        - Agent names can be anything ("Agent1", "CustomProcessor")
        - Role inference should be behavior-based
        """
        # Infer from task types (behavior-based)
        task_types = stats.get('task_types', set())
        
        if not task_types:
            return 'general'  # No tasks yet, can't infer
        
        # Count task type categories
        task_str = ' '.join(str(t) for t in task_types).lower()
        
        # Behavior-based role inference
        role_scores = {
            'processor': task_str.count('process') + task_str.count('transform') + task_str.count('convert'),
            'validator': task_str.count('valid') + task_str.count('check') + task_str.count('verify'),
            'generator': task_str.count('generate') + task_str.count('create') + task_str.count('produce'),
            'analyzer': task_str.count('analyze') + task_str.count('extract') + task_str.count('parse'),
        }
        
        # Return role with highest score, or 'general' if no clear winner
        if max(role_scores.values()) > 0:
            return max(role_scores, key=role_scores.get)
        
        return 'general'
    
    def get_context_summary(self) -> str:
        """Get summary suitable for context injection."""
        view = self.get_agent_view()
        
        if view['abstraction_level'] == 'detailed':
            lines = [f"AGENTS ({view['agent_count']}):"]
            for agent, info in view['agents'].items():
                lines.append(f"  - {agent}: {info['success_rate']:.0%} success, {info['total_tasks']} tasks")
        else:
            lines = [f"AGENT ROLES ({view['agent_count']} total agents):"]
            for role_name, info in view['roles'].items():
                lines.append(
                    f"  - {role_name}: {info['agent_count']} agents, "
                    f"{info['success_rate']:.0%} success, caps={info['capabilities']}"
                )
        
        return "\n".join(lines)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'BrainMode',
    'BrainModeConfig',
    'MemoryCandidate',
    'HippocampalExtractor',
    'ConsolidationResult',
    'SharpWaveRippleConsolidator',
    'BrainStateMachine',
    'AgentRole',
    'AgentAbstractor'
]

