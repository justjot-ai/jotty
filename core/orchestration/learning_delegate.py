"""
LearningDelegate - Learning Operations (Composed)
===================================================

Delegates learning operations to SwarmLearningPipeline and MASLearning.
Standalone composed class, replaces LearningDelegationMixin.

Dependencies are passed explicitly via constructor.
"""

import logging
from typing import Dict, Any, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .learning_pipeline import SwarmLearningPipeline
    from .mas_learning import MASLearning
    from Jotty.core.foundation.agent_config import AgentConfig

logger = logging.getLogger(__name__)


class LearningDelegate:
    """
    Delegates learning operations to the learning subsystems.

    Composed into Orchestrator instead of mixed in.
    Takes explicit references to the learning pipeline and MAS learning.
    """

    def __init__(
        self,
        get_learning,
        get_mas_learning,
        get_agents,
    ):
        """
        Args:
            get_learning: Callable returning SwarmLearningPipeline
            get_mas_learning: Callable returning MASLearning
            get_agents: Callable returning List[AgentConfig]
        """
        self._get_learning = get_learning
        self._get_mas = get_mas_learning
        self._get_agents = get_agents

    @property
    def learning(self) -> 'SwarmLearningPipeline':
        return self._get_learning()

    @property
    def mas_learning(self) -> Optional['MASLearning']:
        try:
            return self._get_mas()
        except Exception:
            return None

    @property
    def agents(self) -> List['AgentConfig']:
        return self._get_agents()

    def auto_load_learnings(self):
        """Load all persisted learnings."""
        self.learning.auto_load()
        # Log MAS stats
        mas = self.mas_learning
        if mas:
            try:
                stats = mas.get_statistics()
                logger.info(
                    f"MAS Learning ready: {stats['fix_database']['total_fixes']} fixes, "
                    f"{stats['sessions']['total_sessions']} sessions"
                )
            except Exception:
                pass

    def auto_save_learnings(
        self,
        mas_learning=None,
        swarm_terminal=None,
        provider_registry=None,
        memory_persistence=None,
    ):
        """Save all learnings to disk."""
        self.learning.auto_save(
            mas_learning=mas_learning or self.mas_learning,
            swarm_terminal=swarm_terminal,
            provider_registry=provider_registry,
        )
        if memory_persistence:
            try:
                memory_persistence.save()
            except Exception as e:
                logger.debug(f"Could not auto-save memory: {e}")

    def load_relevant_learnings(
        self,
        task_description: str,
        agent_types: List[str] = None,
    ) -> Dict[str, Any]:
        """Load learnings relevant to the current task."""
        mas = self.mas_learning
        if not mas:
            return {}
        return mas.load_relevant_learnings(
            task_description=task_description,
            agent_types=agent_types or [a.name for a in self.agents],
        )

    def record_agent_result(
        self,
        agent_name: str,
        task_type: str,
        success: bool,
        time_taken: float,
        output_quality: float = 0.0,
    ):
        """Record an agent's task result for learning."""
        mas = self.mas_learning
        if mas:
            mas.record_agent_task(
                agent_type=agent_name,
                task_type=task_type,
                success=success,
                time_taken=time_taken,
                output_quality=output_quality,
            )

    def record_session_result(
        self,
        task_description: str,
        agent_performances: Dict[str, Dict[str, Any]],
        total_time: float,
        success: bool,
        fixes_applied: List[Dict[str, Any]] = None,
        stigmergy_signals: int = 0,
    ):
        """Record session results for future learning."""
        mas = self.mas_learning
        if mas:
            mas.record_session(
                task_description=task_description,
                agent_performances=agent_performances,
                fixes_applied=fixes_applied or [],
                stigmergy_signals=stigmergy_signals,
                total_time=total_time,
                success=success,
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
