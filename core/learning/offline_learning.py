"""
Jotty v6.0 - Enhanced Offline Learning
======================================

All A-Team offline learning enhancements:
- Dr. Manning: Experience replay with prioritization
- Dr. Chen: Counterfactual learning
- Shannon: Efficient batch processing

Features:
- Episode buffer with prioritized replay
- Counterfactual analysis ("what if agent decided differently?")
- Pattern extraction across episodes
- Causal discovery from contrasting outcomes
"""

import json
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import dspy

from ..foundation.data_structures import (
    SwarmConfig, StoredEpisode, MemoryLevel,
    ValidationResult, AgentContribution, CausalLink
)
from ..foundation.configs.learning import LearningConfig as FocusedLearningConfig
from ..memory.cortex import SwarmMemory
from .learning import TDLambdaLearner, AdaptiveLearningRate


def _ensure_swarm_config(config):
    """Accept LearningConfig or SwarmConfig, return SwarmConfig."""
    if isinstance(config, FocusedLearningConfig):
        return SwarmConfig.from_configs(learning=config)
    return config


# =============================================================================
# COUNTERFACTUAL SIGNATURES
# =============================================================================

class CounterfactualSignature(dspy.Signature):
    """Analyze what would have happened with different decisions."""
    
    episode_summary: str = dspy.InputField(desc="Summary of what happened in the episode")
    agent_decision: str = dspy.InputField(desc="The agent's actual decision and reasoning")
    alternative_decision: str = dspy.InputField(desc="The opposite decision")
    context: str = dspy.InputField(desc="Relevant context and past experiences")
    
    reasoning: str = dspy.OutputField(desc="Analysis of what would have happened")
    estimated_outcome: str = dspy.OutputField(desc="'success', 'failure', or 'uncertain'")
    confidence: float = dspy.OutputField(desc="Confidence in counterfactual estimate")
    lesson: str = dspy.OutputField(desc="Key lesson from this analysis")


class PatternDiscoverySignature(dspy.Signature):
    """Discover patterns from episode batches."""
    
    success_patterns: str = dspy.InputField(desc="Common elements in successful episodes")
    failure_patterns: str = dspy.InputField(desc="Common elements in failed episodes")
    domain: str = dspy.InputField(desc="Domain context")
    
    reasoning: str = dspy.OutputField(desc="Analysis of discriminative patterns")
    success_predictors: str = dspy.OutputField(desc="JSON list of success predictors")
    failure_predictors: str = dspy.OutputField(desc="JSON list of failure predictors")
    recommendations: str = dspy.OutputField(desc="Recommendations for future episodes")


# =============================================================================
# EPISODE BUFFER WITH PRIORITIZATION
# =============================================================================

class PrioritizedEpisodeBuffer:
    """
    Episode buffer with prioritized replay.
    
    Priority based on:
    - TD error magnitude (surprising episodes)
    - Recency (recent episodes more relevant)
    - Outcome (failures often more informative)
    """
    
    def __init__(self, capacity: int, alpha: float = 0.6):
        """
        Parameters:
            capacity: Maximum episodes to store
            alpha: Priority exponent (0 = uniform, 1 = fully prioritized)
        """
        self.capacity = capacity
        self.alpha = alpha
        
        self.episodes: List[StoredEpisode] = []
        self.priorities: List[float] = []
    
    def add(self, episode: StoredEpisode, td_error: float = None):
        """Add episode with priority."""
        # Compute priority
        if td_error is not None:
            priority = abs(td_error) + 0.01  # Small constant to ensure non-zero
        else:
            # Default priority based on success/failure
            priority = 0.5 if episode.success else 1.0
        
        # Add recency bonus
        priority *= 1.0  # Current episode gets full priority
        
        self.episodes.append(episode)
        self.priorities.append(priority)
        
        # Enforce capacity
        while len(self.episodes) > self.capacity:
            # Remove lowest priority (excluding very recent)
            if len(self.episodes) > 10:
                candidates = list(range(len(self.episodes) - 10))
                if candidates:
                    min_idx = min(candidates, key=lambda i: self.priorities[i])
                    self.episodes.pop(min_idx)
                    self.priorities.pop(min_idx)
                else:
                    self.episodes.pop(0)
                    self.priorities.pop(0)
            else:
                self.episodes.pop(0)
                self.priorities.pop(0)
    
    def sample(self, batch_size: int) -> List[Tuple[StoredEpisode, float]]:
        """
        Sample episodes with prioritization.
        
        Returns list of (episode, sampling_weight) tuples.
        """
        if len(self.episodes) == 0:
            return []
        
        # Compute sampling probabilities
        total = sum(p ** self.alpha for p in self.priorities)
        probs = [(p ** self.alpha) / total for p in self.priorities]
        
        # Sample
        n = min(batch_size, len(self.episodes))
        indices = random.choices(range(len(self.episodes)), weights=probs, k=n)
        
        # Compute importance sampling weights
        max_weight = (len(self.episodes) * min(probs)) ** (-1)  # Normalize
        
        samples = []
        for idx in indices:
            weight = (len(self.episodes) * probs[idx]) ** (-1)
            weight = weight / max_weight  # Normalize to [0, 1]
            samples.append((self.episodes[idx], weight))
        
        return samples
    
    def get_recent(self, n: int) -> List[StoredEpisode]:
        """Get most recent n episodes."""
        return self.episodes[-n:]
    
    def get_by_outcome(self, success: bool, n: int = None) -> List[StoredEpisode]:
        """Get episodes by outcome."""
        filtered = [ep for ep in self.episodes if ep.success == success]
        if n:
            return filtered[-n:]
        return filtered
    
    def update_priority(self, episode_id: int, new_priority: float):
        """Update priority for an episode."""
        for i, ep in enumerate(self.episodes):
            if ep.episode_id == episode_id:
                self.priorities[i] = new_priority
                break
    
    def __len__(self):
        return len(self.episodes)


# =============================================================================
# COUNTERFACTUAL LEARNER
# =============================================================================

class CounterfactualLearner:
    """
    Learns from counterfactual analysis of agent decisions.
    
    Asks "what if the agent had decided differently?"
    to improve credit assignment and learn from mistakes.
    """
    
    def __init__(self, config):
        self.config = _ensure_swarm_config(config)
        self.analyzer = dspy.ChainOfThought(CounterfactualSignature)
        
        # Cache to avoid redundant analysis
        self.analyzed_episodes: set = set()
    
    async def analyze_episode(self,
                               episode: StoredEpisode,
                               agent_memories: Dict[str, SwarmMemory]) -> Dict[str, Any]:
        """
        Analyze counterfactuals for an episode.
        
        Returns analysis with estimated alternative outcomes.
        """
        if episode.episode_id in self.analyzed_episodes:
            return {"already_analyzed": True}
        
        self.analyzed_episodes.add(episode.episode_id)
        
        results = {
            "episode_id": episode.episode_id,
            "actual_outcome": "success" if episode.success else "failure",
            "agent_analyses": {}
        }
        
        # Analyze each agent's decision
        all_results = episode.architect_results + episode.auditor_results
        
        for result in all_results:
            # Get agent's decision
            if result.should_proceed is not None:
                actual_decision = "proceed" if result.should_proceed else "block"
                alternative = "block" if result.should_proceed else "proceed"
            else:
                actual_decision = "valid" if result.is_valid else "invalid"
                alternative = "invalid" if result.is_valid else "valid"
            
            # Build context from agent's memory
            context = ""
            if result.agent_name in agent_memories:
                mem = agent_memories[result.agent_name]
                relevant = mem.retrieve(
                    query=episode.goal,
                    goal=episode.goal,
                    budget_tokens=3000
                )
                context = "\n".join(m.content for m in relevant)
            
            try:
                analysis = self.analyzer(
                    episode_summary=f"Goal: {episode.goal}\nOutcome: {'success' if episode.success else 'failure'}",
                    agent_decision=f"Decision: {actual_decision}\nReasoning: {result.reasoning}",
                    alternative_decision=alternative,
                    context=context or "No relevant context available"
                )
                
                results["agent_analyses"][result.agent_name] = {
                    "actual_decision": actual_decision,
                    "alternative": alternative,
                    "estimated_outcome": analysis.estimated_outcome,
                    "confidence": float(analysis.confidence) if analysis.confidence else 0.5,
                    "lesson": analysis.lesson
                }
                
            except Exception as e:
                results["agent_analyses"][result.agent_name] = {
                    "error": str(e)
                }
        
        return results
    
    def compute_counterfactual_credit(self,
                                       counterfactual_analysis: Dict,
                                       original_contributions: Dict[str, AgentContribution]) -> Dict[str, float]:
        """
        Adjust credit based on counterfactual analysis.
        """
        adjustments = {}
        
        actual_outcome = counterfactual_analysis.get("actual_outcome", "success")
        
        for agent_name, analysis in counterfactual_analysis.get("agent_analyses", {}).items():
            if "error" in analysis:
                adjustments[agent_name] = 0.0
                continue
            
            estimated = analysis.get("estimated_outcome", "uncertain")
            confidence = analysis.get("confidence", 0.5)
            
            # If alternative would have been better, negative adjustment
            # If alternative would have been worse, positive adjustment
            if actual_outcome == "success":
                if estimated == "failure":
                    # Good decision - alternative would have failed
                    adjustments[agent_name] = 0.2 * confidence
                elif estimated == "success":
                    # Decision didn't matter much
                    adjustments[agent_name] = 0.0
                else:
                    adjustments[agent_name] = 0.05
            else:  # actual failure
                if estimated == "success":
                    # Bad decision - alternative would have succeeded
                    adjustments[agent_name] = -0.2 * confidence
                elif estimated == "failure":
                    # Even alternative would have failed
                    adjustments[agent_name] = 0.05  # Small positive - not entirely agent's fault
                else:
                    adjustments[agent_name] = -0.05
        
        return adjustments


# =============================================================================
# PATTERN DISCOVERY
# =============================================================================

class PatternDiscovery:
    """
    Discovers patterns from episode batches.
    
    Identifies:
    - Success predictors
    - Failure predictors
    - Discriminative features
    """
    
    def __init__(self, config):
        self.config = _ensure_swarm_config(config)
        self.discoverer = dspy.ChainOfThought(PatternDiscoverySignature)
    
    async def discover_patterns(self,
                                 successes: List[StoredEpisode],
                                 failures: List[StoredEpisode],
                                 domain: str = "general") -> Dict[str, Any]:
        """
        Discover patterns distinguishing success from failure.
        """
        if len(successes) < 2 or len(failures) < 2:
            return {"insufficient_data": True}
        
        # Extract features from episodes
        success_features = self._extract_features(successes)
        failure_features = self._extract_features(failures)
        
        try:
            result = self.discoverer(
                success_patterns=json.dumps(success_features, indent=2),
                failure_patterns=json.dumps(failure_features, indent=2),
                domain=domain
            )
            
            # Parse predictors
            try:
                success_predictors = json.loads(result.success_predictors)
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                logger.debug(f"Success predictors parsing failed: {e}")
                success_predictors = [result.success_predictors]

            try:
                failure_predictors = json.loads(result.failure_predictors)
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                logger.debug(f"Failure predictors parsing failed: {e}")
                failure_predictors = [result.failure_predictors]
            
            return {
                "success_predictors": success_predictors,
                "failure_predictors": failure_predictors,
                "recommendations": result.recommendations,
                "analysis": result.reasoning
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _extract_features(self, episodes: List[StoredEpisode]) -> Dict[str, Any]:
        """Extract common features from episodes."""
        features = {
            "count": len(episodes),
            "common_keywords": [],
            "avg_trajectory_length": 0,
            "common_tools": [],
            "confidence_range": [0, 0]
        }
        
        all_keywords = []
        trajectory_lengths = []
        all_tools = []
        confidences = []
        
        for ep in episodes:
            # Keywords from goal
            words = ep.goal.lower().split()
            all_keywords.extend([w for w in words if len(w) > 3])
            
            # Trajectory length
            trajectory_lengths.append(len(ep.trajectory))
            
            # Tools used
            for step in ep.trajectory:
                if 'tool' in step:
                    all_tools.append(step['tool'])
            
            # Confidences
            for r in ep.architect_results + ep.auditor_results:
                confidences.append(r.confidence)
        
        # Compute statistics
        from collections import Counter
        
        keyword_counts = Counter(all_keywords)
        features["common_keywords"] = [kw for kw, _ in keyword_counts.most_common(10)]
        
        if trajectory_lengths:
            features["avg_trajectory_length"] = sum(trajectory_lengths) / len(trajectory_lengths)
        
        tool_counts = Counter(all_tools)
        features["common_tools"] = [t for t, _ in tool_counts.most_common(5)]
        
        if confidences:
            features["confidence_range"] = [min(confidences), max(confidences)]
        
        return features


# =============================================================================
# MAIN OFFLINE LEARNER
# =============================================================================

class OfflineLearner:
    """
    Complete offline learning system.
    
    Combines:
    - Prioritized experience replay
    - Counterfactual analysis
    - Pattern discovery
    - Batch TD updates
    - Causal knowledge extraction
    """
    
    def __init__(self, config):
        self.config = _ensure_swarm_config(config)
        
        # Episode buffer
        self.buffer = PrioritizedEpisodeBuffer(
            capacity=config.episode_buffer_size
        )
        
        # Components
        self.counterfactual = CounterfactualLearner(config)
        self.pattern_discovery = PatternDiscovery(config)
        
        # TD learner reference (set from Jotty)
        self.td_learner: Optional[TDLambdaLearner] = None
        
        # Statistics
        self.update_count = 0
        self.patterns_discovered = []
        self.counterfactual_lessons = []
    
    def store_episode(self, episode: StoredEpisode, td_error: float = None):
        """Store episode in buffer."""
        self.buffer.add(episode, td_error)
    
    async def batch_update(self,
                           agent_memories: Dict[str, SwarmMemory],
                           current_episode: int) -> Dict[str, Any]:
        """
        Run batch offline learning update.
        
        Called every offline_update_interval episodes.
        """
        self.update_count += 1
        
        results = {
            "update_number": self.update_count,
            "episode_trigger": current_episode,
            "buffer_size": len(self.buffer),
            "replay_updates": 0,
            "patterns_found": 0,
            "counterfactuals_analyzed": 0
        }
        
        # 1. Prioritized experience replay
        replay_results = await self._experience_replay(agent_memories)
        results["replay_updates"] = replay_results.get("updates", 0)
        
        # 2. Pattern discovery
        pattern_results = await self._discover_patterns()
        results["patterns_found"] = len(pattern_results.get("success_predictors", []))
        self.patterns_discovered.extend(pattern_results.get("success_predictors", []))
        
        # 3. Counterfactual analysis (sample of recent episodes)
        cf_results = await self._counterfactual_analysis(agent_memories)
        results["counterfactuals_analyzed"] = cf_results.get("analyzed", 0)
        
        # 4. Update pattern storage in memories
        await self._store_discovered_patterns(agent_memories, pattern_results)
        
        return results
    
    async def _experience_replay(self,
                                  agent_memories: Dict[str, SwarmMemory]) -> Dict[str, Any]:
        """Run experience replay with TD updates."""
        if self.td_learner is None:
            return {"updates": 0, "error": "No TD learner configured"}
        
        # Sample episodes
        samples = self.buffer.sample(self.config.replay_batch_size)
        
        updates = 0
        
        for episode, weight in samples:
            # Replay TD updates for each agent
            for agent_name, memory in agent_memories.items():
                # Get memories that were accessed in this episode
                accessed_keys = episode.memories_accessed.get(agent_name, [])
                
                if not accessed_keys:
                    continue
                
                # Simulate TD update
                self.td_learner.start_episode(episode.goal)
                
                # Record accesses
                for key in accessed_keys:
                    all_mems = {}
                    for level in MemoryLevel:
                        all_mems.update(memory.memories[level])
                    
                    if key in all_mems:
                        self.td_learner.record_access(all_mems[key])
                
                # End episode with weighted reward
                weighted_reward = episode.final_reward * weight
                self.td_learner.end_episode(weighted_reward, all_mems)
                
                updates += 1
        
        return {"updates": updates}
    
    async def _discover_patterns(self) -> Dict[str, Any]:
        """Run pattern discovery on buffered episodes."""
        successes = self.buffer.get_by_outcome(success=True, n=50)
        failures = self.buffer.get_by_outcome(success=False, n=50)
        
        # Determine domain from most common
        domains = defaultdict(int)
        for ep in successes + failures:
            d = ep.kwargs.get('domain', 'general')
            domains[d] += 1
        
        domain = max(domains.keys(), key=lambda d: domains[d]) if domains else 'general'
        
        return await self.pattern_discovery.discover_patterns(successes, failures, domain)
    
    async def _counterfactual_analysis(self,
                                        agent_memories: Dict[str, SwarmMemory]) -> Dict[str, Any]:
        """Run counterfactual analysis on sample of episodes."""
        # Sample recent failures (most valuable for learning)
        failures = self.buffer.get_by_outcome(success=False, n=10)
        
        analyzed = 0
        lessons = []
        
        for episode in failures[-self.config.counterfactual_samples:]:
            analysis = await self.counterfactual.analyze_episode(episode, agent_memories)
            
            if "already_analyzed" not in analysis:
                analyzed += 1
                
                # Extract lessons
                for agent_name, agent_analysis in analysis.get("agent_analyses", {}).items():
                    if "lesson" in agent_analysis:
                        lessons.append({
                            "episode": episode.episode_id,
                            "agent": agent_name,
                            "lesson": agent_analysis["lesson"]
                        })
        
        self.counterfactual_lessons.extend(lessons)
        
        return {"analyzed": analyzed, "lessons": len(lessons)}
    
    async def _store_discovered_patterns(self,
                                          agent_memories: Dict[str, SwarmMemory],
                                          pattern_results: Dict[str, Any]):
        """Store discovered patterns in agent memories."""
        if "error" in pattern_results or "insufficient_data" in pattern_results:
            return
        
        # Store as SEMANTIC memories
        success_predictors = pattern_results.get("success_predictors", [])
        failure_predictors = pattern_results.get("failure_predictors", [])
        recommendations = pattern_results.get("recommendations", "")
        
        pattern_content = f"""
DISCOVERED PATTERNS (Offline Learning Update #{self.update_count}):

SUCCESS PREDICTORS:
{json.dumps(success_predictors, indent=2)}

FAILURE PREDICTORS:
{json.dumps(failure_predictors, indent=2)}

RECOMMENDATIONS:
{recommendations}
""".strip()
        
        # Store in first available memory (shared pattern)
        for memory in agent_memories.values():
            memory.store(
                content=pattern_content,
                level=MemoryLevel.SEMANTIC,
                context={
                    "source": "offline_learning",
                    "update": self.update_count
                },
                goal="pattern_discovery",
                initial_value=0.7
            )
            break  # Store once
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get offline learning statistics."""
        return {
            "update_count": self.update_count,
            "buffer_size": len(self.buffer),
            "patterns_discovered": len(self.patterns_discovered),
            "counterfactual_lessons": len(self.counterfactual_lessons),
            "success_episodes": len(self.buffer.get_by_outcome(True)),
            "failure_episodes": len(self.buffer.get_by_outcome(False))
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for persistence."""
        return {
            "update_count": self.update_count,
            "patterns_discovered": self.patterns_discovered,  # Keep last 100
            "counterfactual_lessons": self.counterfactual_lessons,
            "episodes": [
                {
                    "episode_id": ep.episode_id,
                    "goal": ep.goal,
                    "success": ep.success,
                    "final_reward": ep.final_reward,
                    "timestamp": ep.timestamp.isoformat()
                }
                for ep in self.buffer.episodes  # Keep summaries of last 500
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], config: SwarmConfig) -> 'OfflineLearner':
        """Deserialize from persistence."""
        learner = cls(config)
        learner.update_count = data.get("update_count", 0)
        learner.patterns_discovered = data.get("patterns_discovered", [])
        learner.counterfactual_lessons = data.get("counterfactual_lessons", [])
        # Note: Full episodes not restored, only summaries available
        return learner
