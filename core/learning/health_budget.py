"""
Learning health monitoring and dynamic budget management.

Detects pathological learning behaviors and manages context budgets.
"""

import math
import hashlib
import logging
from typing import Dict, List, Any, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)

from ..foundation.data_structures import (
    SwarmConfig, MemoryEntry, MemoryLevel, GoalValue,
    ValidationResult, AgentContribution, StoredEpisode,
    LearningMetrics, AlertType, GoalHierarchy, CausalLink
)
if TYPE_CHECKING:
    from ..memory.cortex import SwarmMemory




# =============================================================================
# LEARNING HEALTH MONITOR (Enhanced)
# =============================================================================

class LearningHealthMonitor:
    """
    Monitors learning health with all pathological behavior detection.
    
    Alerts:
    - Reward hacking
    - Distribution shift
    - Conservative collapse
    - Catastrophic forgetting
    - Learning stall
    - Goal drift
    """
    
    def __init__(self, config: SwarmConfig):
        self.config = config
        self.metrics = LearningMetrics()
        
        # Tracking
        self.approval_rates: Dict[str, List[float]] = defaultdict(list)
        self.goal_distributions: List[Dict[str, int]] = []
        self.value_snapshots: List[Dict[str, float]] = []
    
    def record_episode(self,
                        success: bool,
                        goal: str,
                        architect_decisions: List[bool],
                        auditor_decisions: List[bool],
                        value_updates: List[Tuple[str, float, float]]) -> List[str]:
        """
        Record episode and check for issues.
        
        Returns list of alert messages.
        """
        alerts = []
        
        # Update metrics
        self.metrics.episode_count += 1
        self.metrics.success_count += 1 if success else 0
        self.metrics.recent_successes.append(success)
        self.metrics.goals_seen.add(goal)
        
        # Track value changes
        for key, old_v, new_v in value_updates:
            self.metrics.value_changes.append(new_v - old_v)
        
        # Check for reward hacking
        if self._detect_reward_hacking():
            alerts.append(f"ALERT[{AlertType.REWARD_HACKING.value}]: Success rate suspiciously high (>{self.config.suspicion_threshold:.0%})")
        
        # Check for conservative collapse
        approval_rate = sum(architect_decisions) / len(architect_decisions) if architect_decisions else 0.5
        if self._detect_conservative_collapse(approval_rate):
            alerts.append(f"ALERT[{AlertType.CONSERVATIVE_COLLAPSE.value}]: Rejection rate too high, may be over-conservative")
        
        # Check for learning stall
        if self._detect_learning_stall():
            alerts.append(f"ALERT[{AlertType.LEARNING_STALL.value}]: Learning appears stalled, values not changing")
        
        # Check for goal drift
        drift = self._detect_goal_drift(goal)
        if drift:
            alerts.append(f"ALERT[{AlertType.GOAL_DRIFT.value}]: Goal distribution shifting: {drift}")
        
        return alerts
    
    def _detect_reward_hacking(self) -> bool:
        """Detect suspiciously high success rates."""
        if len(self.metrics.recent_successes) < 50:
            return False
        
        recent = self.metrics.recent_successes
        rate = sum(recent) / len(recent)
        
        return rate > self.config.suspicion_threshold
    
    def _detect_conservative_collapse(self, current_approval: float) -> bool:
        """Detect if agents are rejecting too much."""
        # Need history
        if self.metrics.episode_count < 20:
            return False
        
        # Very low approval rate
        return current_approval < self.config.min_rejection_rate
    
    def _detect_learning_stall(self) -> bool:
        """Detect if learning has stalled."""
        if len(self.metrics.value_changes) < 100:
            return False
        
        recent = self.metrics.value_changes
        avg_change = sum(abs(v) for v in recent) / len(recent)
        
        return avg_change < self.config.stall_threshold
    
    def _detect_goal_drift(self, current_goal: str) -> Optional[str]:
        """Detect if goal distribution is shifting unusually."""
        # Track goal frequency
        goal_id = current_goal  # Truncate for tracking
        
        if not hasattr(self, 'goal_counts'):
            self.goal_counts: Dict[str, int] = defaultdict(int)
            self.recent_goals: List[str] = []
        
        self.goal_counts[goal_id] += 1
        self.recent_goals.append(goal_id)
        
        if len(self.recent_goals) > 100:
            self.recent_goals = self.recent_goals
        
        # Check if single goal is dominating
        if len(self.recent_goals) >= 50:
            recent_unique = len(set(self.recent_goals))
            if recent_unique == 1:
                return f"Single goal dominating: {goal_id}..."
        
        return None
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get summary of learning health."""
        return {
            "episode_count": self.metrics.episode_count,
            "success_rate": self.metrics.get_success_rate(),
            "learning_velocity": self.metrics.get_learning_velocity(),
            "is_stalled": self.metrics.is_learning_stalled(),
            "unique_goals": len(self.metrics.goals_seen),
            "causal_links": self.metrics.causal_links_discovered
        }


# =============================================================================
# CONTEXT BUDGET MANAGER (Enhanced)
# =============================================================================

class DynamicBudgetManager:
    """
    Dynamic context budget allocation.
    
    Instead of fixed allocation, adapts based on:
    - Query complexity
    - Trajectory length
    - Tool output sizes
    - Available relevant memories
    """
    
    def __init__(self, config: SwarmConfig):
        self.config = config
        self.total_budget = config.max_context_tokens
        
        # Base allocations
        self.base_allocations = {
            'system_prompt': config.system_prompt_budget,
            'current_input': config.current_input_budget,
            'trajectory': config.trajectory_budget,
            'tool_output': config.tool_output_budget,
        }
    
    def compute_allocation(self,
                           system_prompt_tokens: int,
                           input_tokens: int,
                           trajectory_tokens: int,
                           tool_output_tokens: int) -> Dict[str, int]:
        """
        Compute dynamic budget allocation.
        
        Returns dict with actual tokens to allocate for each category.
        """
        if not self.config.enable_dynamic_budget:
            # Static allocation
            return {
                'system_prompt': self.base_allocations['system_prompt'],
                'current_input': self.base_allocations['current_input'],
                'trajectory': self.base_allocations['trajectory'],
                'tool_output': self.base_allocations['tool_output'],
                'memory': self.config.memory_budget
            }
        
        # Use actual sizes
        actual_usage = {
            'system_prompt': system_prompt_tokens,
            'current_input': input_tokens,
            'trajectory': trajectory_tokens,
            'tool_output': tool_output_tokens
        }
        
        # Compute remaining for memory
        used = sum(actual_usage.values())
        memory_budget = self.total_budget - used
        
        # Enforce bounds
        memory_budget = max(
            self.config.min_memory_budget,
            min(self.config.max_memory_budget, memory_budget)
        )
        
        # If memory budget would cause overflow, reduce trajectory
        total_with_memory = used + memory_budget
        if total_with_memory > self.total_budget:
            overage = total_with_memory - self.total_budget
            actual_usage['trajectory'] = max(5000, actual_usage['trajectory'] - overage)
        
        actual_usage['memory'] = memory_budget
        return actual_usage
    
    def select_within_budget(self,
                              items: List[MemoryEntry],
                              budget: int,
                              goal: str,
                              max_items: int = 50) -> List[MemoryEntry]:
        """
        Select items within budget - NO TRUNCATION.
        
        Items are included fully or not at all.
        """
        # Sort by value
        sorted_items = sorted(
            items,
            key=lambda m: m.get_value(goal),
            reverse=True
        )
        
        selected = []
        tokens_used = 0
        
        for item in sorted_items:
            if len(selected) >= max_items:
                break
            
            # Check size limit
            if item.token_count > self.config.max_entry_tokens:
                continue  # Skip oversized
            
            if tokens_used + item.token_count <= budget:
                selected.append(item)
                tokens_used += item.token_count
        
        return selected
    
    def get_learned_context(self, memories: Dict[str, MemoryEntry], goal: str = None) -> str:
        """
        Get learned context to inject into prompts.
        
        THIS IS HOW TD(λ) LEARNING MANIFESTS IN LLM AGENTS!
        
        Returns natural language lessons from value updates.
        """
        if not memories:
            return ""
        
        goal = goal or self.current_goal
        if not goal:
            return ""
        
        # Collect memories with significant learned values
        high_value_memories = []
        low_value_memories = []
        improved_memories = []
        
        for key, memory in memories.items():
            if goal not in memory.goal_values:
                continue
            
            goal_val = memory.goal_values[goal]
            value = goal_val.value
            
            # Check if value was updated significantly
            if key in self.values_at_access:
                old_value = self.values_at_access[key]
                improvement = value - old_value
                
                if abs(improvement) > 0.1:  # Significant update
                    improved_memories.append((memory, value, improvement))
            
            if value > 0.7:
                high_value_memories.append((memory, value))
            elif value < 0.3:
                low_value_memories.append((memory, value))
        
        if not (high_value_memories or low_value_memories or improved_memories):
            return ""
        
        context = "# TD(λ) Learned Values:\n"
        
        # High-value lessons
        if high_value_memories:
            context += "\n## High-Value Patterns (Learned from Success):\n"
            for memory, value in sorted(high_value_memories, key=lambda x: x[1], reverse=True)[:5]:
                context += f"- {memory.content[:150]}... (V={value:.3f})\n"
        
        # Low-value lessons (what to avoid)
        if low_value_memories:
            context += "\n## Low-Value Patterns (Learned from Failure):\n"
            for memory, value in sorted(low_value_memories, key=lambda x: x[1])[:5]:
                context += f"- AVOID: {memory.content[:150]}... (V={value:.3f})\n"
        
        # Recently improved
        if improved_memories:
            context += "\n## Recently Updated Understanding:\n"
            for memory, value, improvement in sorted(improved_memories, key=lambda x: abs(x[2]), reverse=True)[:3]:
                direction = "↑" if improvement > 0 else "↓"
                context += f"- {direction} {memory.content[:150]}... (V={value:.3f}, Δ={improvement:+.3f})\n"
        
        # Add eligibility trace info (which memories are most relevant now)
        if self.traces:
            context += "\n## Currently Relevant (High Eligibility):\n"
            sorted_traces = sorted(self.traces.items(), key=lambda x: x[1], reverse=True)[:3]
            for key, trace in sorted_traces:
                if key in memories and trace > 0.5:
                    memory = memories[key]
                    context += f"- {memory.content[:150]}... (trace={trace:.2f})\n"
        