"""
SwarmManager Learning Delegation Mixin
=======================================

Extracted from swarm_manager.py — delegates learning operations
to SwarmLearningPipeline and MASLearning.
"""

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class LearningDelegationMixin:
    """Mixin for learning system delegation."""

    def _auto_load_learnings(self):
        """Delegate to SwarmLearningPipeline."""
        self.learning.auto_load()
        # Sync credit_weights reference after load (may have been replaced)
        self.credit_weights = self.learning.credit_weights
        # Log MAS stats
        if hasattr(self, 'mas_learning') and self.mas_learning:
            try:
                stats = self.mas_learning.get_statistics()
                logger.info(f"MAS Learning ready: {stats['fix_database']['total_fixes']} fixes, "
                           f"{stats['sessions']['total_sessions']} sessions")
            except Exception:
                pass

    def _auto_save_learnings(self):
        """Delegate to SwarmLearningPipeline."""
        self.learning.auto_save(
            mas_learning=getattr(self, 'mas_learning', None),
            swarm_terminal=getattr(self, 'swarm_terminal', None),
            provider_registry=getattr(self, 'provider_registry', None),
        )
        # Save HierarchicalMemory persistence
        if hasattr(self, 'memory_persistence') and self.memory_persistence:
            try:
                self.memory_persistence.save()
            except Exception as e:
                logger.debug(f"Could not auto-save memory: {e}")

    def load_relevant_learnings(self, task_description: str, agent_types: List[str] = None) -> Dict[str, Any]:
        """Load learnings relevant to the current task."""
        if not hasattr(self, 'mas_learning') or not self.mas_learning:
            return {}
        return self.mas_learning.load_relevant_learnings(
            task_description=task_description,
            agent_types=agent_types or [a.name for a in self.agents],
        )

    def record_agent_result(self, agent_name: str, task_type: str, success: bool,
                            time_taken: float, output_quality: float = 0.0):
        """Record an agent's task result for learning."""
        if hasattr(self, 'mas_learning') and self.mas_learning:
            self.mas_learning.record_agent_task(
                agent_type=agent_name, task_type=task_type,
                success=success, time_taken=time_taken, output_quality=output_quality,
            )

    def record_session_result(self, task_description: str,
                              agent_performances: Dict[str, Dict[str, Any]],
                              total_time: float, success: bool,
                              fixes_applied: List[Dict[str, Any]] = None,
                              stigmergy_signals: int = 0):
        """Record session results for future learning."""
        if hasattr(self, 'mas_learning') and self.mas_learning:
            self.mas_learning.record_session(
                task_description=task_description, agent_performances=agent_performances,
                fixes_applied=fixes_applied or [], stigmergy_signals=stigmergy_signals,
                total_time=total_time, success=success,
            )

    def get_transferable_context(self, query: str, agent: str = None) -> str:
        """Get transferable learnings as context for an agent."""
        return self.learning.get_transferable_context(query, agent)

    def get_swarm_wisdom(self, query: str) -> str:
        """Get collective swarm wisdom for a task."""
        return self.learning.get_swarm_wisdom(query)

    def get_agent_specializations(self) -> Dict[str, str]:
        """Get current specializations of all agents."""
        return self.learning.get_agent_specializations()

    def get_best_agent_for_task(self, query: str) -> Optional[str]:
        """Recommend the best agent for a task."""
        return self.learning.get_best_agent_for_task(query)
