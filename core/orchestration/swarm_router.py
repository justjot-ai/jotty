"""
SwarmRouter: Centralized task routing for SwarmManager.
========================================================

Extracted routing concerns from SwarmManager to reduce its scope.
SwarmManager delegates routing decisions to this class.

Responsibilities:
- Agent selection for tasks (zero-config, complexity-based)
- Task decomposition and sub-task assignment
- Model tier routing (cheap/balanced/quality)
- Integration with SwarmIntelligence for RL-informed routing

This is one of the focused sub-managers that reduces SwarmManager's
2700-line god-object into composable delegates.
"""

import logging
from typing import Dict, List, Any, Optional, Callable

logger = logging.getLogger(__name__)


class SwarmRouter:
    """
    Centralized routing logic extracted from SwarmManager.

    Combines:
    - SwarmIntelligence smart_route (coordination protocols + RL)
    - Model tier routing (cheap/balanced/quality LLM selection)
    - Zero-config agent creation from natural language
    - Task complexity estimation for mode selection (single vs multi)
    """

    def __init__(
        self,
        get_swarm_intelligence: Callable = None,
        get_agents: Callable = None,
        get_model_tier_router: Callable = None,
        get_learning: Callable = None,
    ):
        """
        Args:
            get_swarm_intelligence: Callable returning SwarmIntelligence instance
            get_agents: Callable returning list of AgentConfig
            get_model_tier_router: Callable returning ModelTierRouter
            get_learning: Callable returning LearningPipeline (for order_agents_for_goal)
        """
        self._get_si = get_swarm_intelligence or (lambda: None)
        self._get_agents = get_agents or (lambda: [])
        self._get_mtr = get_model_tier_router or (lambda: None)
        self._get_learning = get_learning or (lambda: None)

    def select_agent(
        self,
        task: str,
        task_type: str = "general",
        prefer_coalition: bool = False
    ) -> Dict[str, Any]:
        """
        Select the best agent for a task using all available routing strategies.

        Tries in order:
        1. SwarmIntelligence smart_route (RL + coordination)
        2. Model tier routing (complexity-based LLM selection)
        3. First available agent (fallback)

        Returns:
            Dict with 'agent', 'method', 'confidence', 'model_tier'
        """
        result = {
            'agent': None,
            'method': 'fallback',
            'confidence': 0.5,
            'model_tier': None,
            'rl_advantage': 0.0,
        }

        agents = self._get_agents()
        if not agents:
            return result

        # Strategy 1: SwarmIntelligence smart_route
        si = self._get_si()
        if si and si.agent_profiles:
            try:
                route = si.smart_route(
                    task_id=f"route_{hash(task) % 10000}",
                    task_type=task_type,
                    task_description=task[:200],
                    prefer_coalition=prefer_coalition,
                )
                if route.get('assigned_agent'):
                    result['agent'] = route['assigned_agent']
                    result['method'] = route.get('method', 'smart_route')
                    result['confidence'] = route.get('confidence', 0.7)
                    result['rl_advantage'] = route.get('rl_advantage', 0.0)
            except Exception as e:
                logger.debug(f"Smart route failed, using fallback: {e}")

        # Strategy 2: Model tier routing
        mtr = self._get_mtr()
        if mtr:
            try:
                tier = mtr.select_tier(task)
                result['model_tier'] = tier
            except Exception as e:
                logger.debug(f"Model tier routing failed: {e}")

        # Strategy 3: Fallback to first agent
        if not result['agent'] and agents:
            result['agent'] = agents[0].name if hasattr(agents[0], 'name') else str(agents[0])
            result['method'] = 'fallback'
            result['confidence'] = 0.5

        return result

    def order_agents_for_goal(self, goal: str) -> List[Any]:
        """
        Order agents for a multi-agent run using learning (trust + stigmergy + TRAS).
        Single entry point: delegates to LearningPipeline.order_agents_for_goal.
        Returns a new list; caller should assign (e.g. self.agents = router.order_agents_for_goal(goal)).
        """
        lp = self._get_learning()
        agents = self._get_agents()
        if not lp:
            return list(agents) if agents else []
        return lp.order_agents_for_goal(goal, agents)

    def estimate_complexity(self, task: str) -> Dict[str, Any]:
        """
        Estimate task complexity for mode selection.

        Returns:
            Dict with 'complexity' (0-1), 'recommended_mode' ('single'/'multi'),
            'recommended_agents' (int)
        """
        # Simple heuristic based on task characteristics
        task_lower = task.lower()

        complexity = 0.3  # base

        # Indicators of complex tasks
        complex_keywords = [
            'research', 'analyze', 'compare', 'multi-step',
            'comprehensive', 'investigate', 'design', 'architect'
        ]
        for kw in complex_keywords:
            if kw in task_lower:
                complexity += 0.1

        # Length as proxy
        if len(task) > 200:
            complexity += 0.1
        if len(task) > 500:
            complexity += 0.1

        complexity = min(1.0, complexity)

        return {
            'complexity': complexity,
            'recommended_mode': 'multi' if complexity >= 0.6 else 'single',
            'recommended_agents': min(5, max(1, int(complexity * 5))),
        }

    def route_by_executor_type(
        self,
        available_skills: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group available skills by their executor_type.

        Returns:
            Mapping of executor_type string to list of matching skill dicts.
            Skills without an executor_type are placed under ``'general'``.
        """
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for skill in available_skills:
            etype = skill.get('executor_type', 'general') or 'general'
            groups.setdefault(etype, []).append(skill)
        return groups

    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics from SwarmIntelligence."""
        si = self._get_si()
        if not si:
            return {'status': 'no_swarm_intelligence'}

        return {
            'agent_count': len(si.agent_profiles),
            'has_td_learner': si._td_learner is not None,
            'rl_loop_closed': si._td_learner is not None,
            'tree_built': si._tree_built,
            'active_coalitions': len(si.coalitions),
            'collective_memory_size': len(si.collective_memory),
        }
