"""
SwarmRouter: Centralized task routing for Orchestrator.
========================================================

Extracted routing concerns from Orchestrator to reduce its scope.
Orchestrator delegates routing decisions to this class.

Responsibilities:
- Agent selection for tasks (zero-config, complexity-based)
- Task decomposition and sub-task assignment
- Model tier routing (cheap/balanced/quality)
- Integration with SwarmIntelligence for RL-informed routing

This is one of the focused sub-managers that reduces Orchestrator's
2700-line god-object into composable delegates.
"""

import logging
from typing import Dict, List, Any, Optional, Callable

logger = logging.getLogger(__name__)


class SwarmRouter:
    """
    Centralized routing logic extracted from Orchestrator.

    Combines:
    - SwarmIntelligence smart_route (coordination protocols + RL)
    - Model tier routing (cheap/balanced/quality LLM selection)
    - Zero-config agent creation from natural language
    - Task complexity estimation for mode selection (single vs multi)
    """

    def __init__(self, get_swarm_intelligence: Callable = None, get_agents: Callable = None, get_model_tier_router: Callable = None, get_learning: Callable = None) -> None:
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
        # =====================================================================
        # MULTI-STRATEGY AGENT ROUTING EXPLAINED
        # =====================================================================
        # PROBLEM: Given a task, which agent should handle it?
        #
        # CHALLENGES:
        # 1. No agent has a hardcoded "specialty" (zero-config design)
        # 2. Agent performance varies by task type (some are better at X than Y)
        # 3. Some tasks benefit from multiple agents (coalition)
        # 4. Different tasks need different LLM tiers (simple vs complex)
        # 5. We want to learn from experience (RL) but need fallbacks
        #
        # SOLUTION: Multi-strategy routing with graceful degradation
        # - Strategy 1: Use RL (if available) to pick best agent based on past performance
        # - Strategy 2: Use task complexity to select LLM tier (cheap/balanced/quality)
        # - Strategy 3: Fallback to first available agent if nothing else works
        #
        # WHY THIS DESIGN:
        # - Strategy 1 (RL) requires training data → won't work on first run
        # - Strategy 2 (tier routing) is stateless → always available
        # - Strategy 3 (fallback) guarantees we always return SOMETHING
        #
        # EXAMPLE ROUTING DECISION:
        # Task: "Research quantum computing trends and write a report"
        # - Strategy 1: RL sees "research" tasks succeeded 80% with researcher agent
        #               → Returns researcher, confidence=0.8, rl_advantage=0.15
        # - Strategy 2: Task is complex (200+ chars, keyword "research")
        #               → Returns model_tier="quality" (use GPT-4 not GPT-3.5)
        # - Strategy 3: (Not needed, Strategy 1 succeeded)
        # - Final result: {'agent': 'researcher', 'method': 'smart_route',
        #                  'confidence': 0.8, 'model_tier': 'quality', 'rl_advantage': 0.15}
        # =====================================================================

        # Initialize result dict with safe defaults
        # These will be overwritten if routing strategies succeed
        result = {
            'agent': None,              # Agent name (str) or None
            'method': 'fallback',       # How agent was selected
            'confidence': 0.5,          # How confident we are (0-1)
            'model_tier': None,         # LLM tier: 'cheap'/'balanced'/'quality'
            'rl_advantage': 0.0,        # RL advantage over random (0-1)
        }

        agents = self._get_agents()
        if not agents:
            return result

        # =====================================================================
        # STRATEGY 1: SWARM INTELLIGENCE SMART ROUTE (RL-based)
        # =====================================================================
        # Uses reinforcement learning to select agents based on:
        # - Past performance on similar tasks (TD-Lambda value estimates)
        # - Agent coordination history (which pairs work well together)
        # - Task type matching (agent specialization learned from data)
        #
        # HOW IT WORKS:
        # 1. SwarmIntelligence maintains agent_profiles with success rates per task_type
        # 2. For each agent, it computes an RL advantage score:
        #    advantage = agent_value - average_value
        # 3. Picks agent with highest advantage (or coalition if prefer_coalition=True)
        # 4. Returns confidence based on how much better the best agent is vs others
        #
        # EXAMPLE:
        # - Agent A has 80% success on "research" tasks (value=0.8)
        # - Agent B has 50% success on "research" tasks (value=0.5)
        # - Average = 0.65
        # - Advantage A = 0.8 - 0.65 = 0.15 (PICK THIS)
        # - Advantage B = 0.5 - 0.65 = -0.15
        # - Confidence = 0.8 (Agent A's raw success rate)
        #
        # WHY CONFIDENCE MATTERS:
        # High confidence (>0.8) → Trust this agent, give it autonomy
        # Low confidence (<0.6) → Uncertain, maybe use coalition or human review
        # =====================================================================
        si = self._get_si()
        if si and si.agent_profiles:
            try:
                route = si.smart_route(
                    task_id=f"route_{hash(task) % 10000}",
                    task_type=task_type,
                    task_description=task[:200],  # Truncate for efficiency
                    prefer_coalition=prefer_coalition,
                )
                if route.get('assigned_agent'):
                    result['agent'] = route['assigned_agent']
                    result['method'] = route.get('method', 'smart_route')
                    result['confidence'] = route.get('confidence', 0.7)
                    result['rl_advantage'] = route.get('rl_advantage', 0.0)
            except Exception as e:
                logger.debug(f"Smart route failed, using fallback: {e}")

        # =====================================================================
        # STRATEGY 2: MODEL TIER ROUTING (complexity-based LLM selection)
        # =====================================================================
        # Independently selects which LLM tier to use based on task complexity.
        # This is ORTHOGONAL to agent selection (Strategy 1 picks WHO,
        # Strategy 2 picks WHICH MODEL they should use).
        #
        # THREE TIERS:
        # - 'cheap': Fast, low-cost models (GPT-3.5, Claude Haiku)
        #   → Use for: Simple tasks, fact lookup, formatting
        # - 'balanced': Mid-tier models (GPT-4o-mini, Claude Sonnet)
        #   → Use for: Most tasks, general reasoning
        # - 'quality': Premium models (GPT-4, Claude Opus)
        #   → Use for: Complex reasoning, long context, critical tasks
        #
        # HOW TIER SELECTION WORKS:
        # ModelTierRouter analyzes task characteristics:
        # - Length (short tasks → cheap, long tasks → quality)
        # - Keywords (e.g., "analyze", "research" → quality)
        # - Structure (multi-step → quality, single-step → cheap)
        #
        # EXAMPLE:
        # Task: "What is 2+2?" → tier='cheap' (simple arithmetic)
        # Task: "Research and compare 10 cloud providers" → tier='quality' (complex)
        #
        # WHY TIER ROUTING MATTERS:
        # - Cost optimization: Don't use GPT-4 for "Hello world"
        # - Performance: Complex tasks need better models
        # - Latency: Cheap models are faster for simple tasks
        # =====================================================================
        mtr = self._get_mtr()
        if mtr:
            try:
                tier = mtr.select_tier(task)
                result['model_tier'] = tier
            except Exception as e:
                logger.debug(f"Model tier routing failed: {e}")

        # =====================================================================
        # STRATEGY 3: FALLBACK TO FIRST AVAILABLE AGENT
        # =====================================================================
        # If all else fails, pick the first agent in the list.
        # This guarantees we ALWAYS return a valid agent (graceful degradation).
        #
        # WHEN THIS HAPPENS:
        # - First run (no RL training data yet)
        # - SwarmIntelligence disabled
        # - All agents have equal (or unknown) performance
        # - smart_route() raised an exception
        #
        # CONFIDENCE:
        # - Set to 0.5 (neutral) because we have no data to inform the decision
        # - Caller can check confidence < 0.6 and decide to ask for human review
        #
        # EXAMPLE:
        # Agents = [researcher, coder, writer]
        # No RL data available
        # → Returns 'researcher' with confidence=0.5, method='fallback'
        # =====================================================================
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
