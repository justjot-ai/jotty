"""
ZeroConfigAgentFactory - Extracted from Orchestrator
=====================================================

LLM-driven agent creation: analyzes task to decide single vs multi-agent.
"""

import logging
import re
import json
from typing import List, Optional, Any

from Jotty.core.infrastructure.foundation.agent_config import AgentConfig
from Jotty.core.infrastructure.utils.async_utils import StatusReporter

logger = logging.getLogger(__name__)


class ZeroConfigAgentFactory:
    """
    Zero-config: LLM decides if task needs multiple agents.
    Uses LLM to analyze task for parallel sub-goals.
    MAS-ZERO: Also assigns per-subtask strategies.
    """

    # Strategy hints: keywords -> recommended strategy (MAS-ZERO sub-MAS assignment)
    STRATEGY_HINTS = {
        'compare': 'ensemble',
        'vs': 'ensemble',
        'analyze': 'ensemble',
        'debate': 'ensemble',
        'create': 'direct',
        'generate': 'direct',
        'build': 'direct',
        'research': 'self_refine',
        'investigate': 'self_refine',
        'summarize': 'direct',
    }

    def create_agents(self, task: str, status_callback: Any = None) -> List[AgentConfig]:
        """
        Analyze task and create appropriate agent configs.

        Returns single AutoAgent for sequential workflows,
        multiple for truly independent parallel sub-goals.
        MAS-ZERO: assigns per-subtask strategy via metadata.
        """
        from Jotty.core.modes.agent.base.auto_agent import AutoAgent
        import dspy

        _status = StatusReporter(status_callback, logger, emoji="")

        try:
            if hasattr(dspy.settings, 'lm') and dspy.settings.lm:
                _status("LLM analyzing", "checking for parallel sub-goals")
                decision = self._llm_decide_agents(task)
                if decision and len(decision) > 1:
                    agents = []
                    for i, sub_goal in enumerate(decision):
                        agent = AutoAgent()
                        agent_name = self._derive_agent_name(sub_goal, i)
                        # MAS-ZERO: assign per-subtask strategy
                        strategy = self._infer_subtask_strategy(sub_goal)
                        agent_config = AgentConfig(
                            name=agent_name,
                            agent=agent,
                            capabilities=[sub_goal],
                            is_executor=True,
                            metadata={'strategy': strategy},
                        )
                        agents.append(agent_config)
                        _status(f"  {agent_name}", f"{sub_goal[:50]} (strategy={strategy})")
                    _status("Multi-agent mode", f"{len(agents)} parallel agents")
                    return agents
                else:
                    _status("Single-agent mode", "sequential workflow detected")
        except Exception as e:
            logger.debug(f"LLM agent decision failed, using single agent: {e}")
            _status("Single-agent mode", "fallback (LLM decision failed)")

        return [AgentConfig(name="auto", agent=AutoAgent())]

    def _infer_subtask_strategy(self, sub_goal: str) -> str:
        """
        Infer the best strategy for a sub-task (MAS-ZERO sub-MAS assignment).

        KISS: Keyword-based heuristic. No LLM call needed for this.
        """
        goal_lower = sub_goal.lower()
        for keyword, strategy in self.STRATEGY_HINTS.items():
            if keyword in goal_lower:
                return strategy
        return 'direct'

    def _llm_decide_agents(self, task: str) -> List[str]:
        """Use LLM to decide if task has parallel sub-goals."""
        import dspy

        class AgentDecisionSignature(dspy.Signature):
            """Analyze if task has INDEPENDENT sub-goals that can run in PARALLEL.

            Default to single agent for most tasks.
            Use parallel agents when there are ENUMERABLE, INDEPENDENT items to research/analyze.

            PARALLEL (use when items are enumerable and independent):
            - "Research 5 programming languages" -> PARALLEL (each language is independent)
            - "Compare Python, Rust, Go, and Julia for ML" -> PARALLEL (independent per-language research)
            - "Analyze top 10 stocks" -> PARALLEL (each stock is independent)
            - "Compare A vs B AND compare C vs D" -> PARALLEL if different domains

            SEQUENTIAL (use when steps build on each other):
            - "Research X and generate PDF" -> SEQUENTIAL (PDF needs research output)
            - "Create checklist for X" -> SEQUENTIAL (research + create + format)
            - "Analyze company X for multiple aspects" -> SEQUENTIAL (one entity, multiple views)

            AVOID creating multiple agents for:
            - Tasks where one step's output feeds the next
            - Tasks with a single entity examined from multiple angles
            - Tasks that will produce overlapping research
            """
            task: str = dspy.InputField(desc="The task to analyze")
            is_parallel: bool = dspy.OutputField(desc="True if task has enumerable independent sub-goals (e.g., research N different items). Default False for ambiguous cases.")
            sub_goals: str = dspy.OutputField(desc="If parallel, JSON list of 2-4 DISTINCT sub-goals (no duplicates). If sequential, empty list []")

        try:
            predictor = dspy.Predict(AgentDecisionSignature)
            result = predictor(task=task)

            logger.info(f" LLM decision: is_parallel={result.is_parallel}, sub_goals={result.sub_goals[:100]}")

            if result.is_parallel:
                sub_goals = self._parse_sub_goals(result.sub_goals)
                if isinstance(sub_goals, list) and len(sub_goals) > 1:
                    sub_goals = self._deduplicate_sub_goals(sub_goals)
                    sub_goals = sub_goals[:4]
                    if len(sub_goals) > 1:
                        logger.info(f" LLM detected {len(sub_goals)} parallel sub-goals: {sub_goals}")
                        return sub_goals
                    else:
                        logger.info(" After deduplication: single agent optimal")
            else:
                logger.info(" LLM detected sequential workflow - single agent optimal")
        except Exception as e:
            logger.debug(f"Agent decision parsing failed: {e}")

        return []

    def _parse_sub_goals(self, raw: str) -> List[str]:
        """Parse sub-goals from LLM output — handles JSON, numbered lists, etc."""
        import re
        raw = str(raw).strip()

        # Method 1: JSON parse
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(g).strip() for g in parsed if str(g).strip()]
        except (json.JSONDecodeError, ValueError):
            pass

        # Method 2: Extract JSON array from text
        json_match = re.search(r'\[.*?\]', raw, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
                if isinstance(parsed, list):
                    return [str(g).strip() for g in parsed if str(g).strip()]
            except (json.JSONDecodeError, ValueError):
                pass

        # Method 3: Numbered list (1. ..., 2. ...)
        numbered = re.findall(r'^\s*\d+[\.\)]\s*(.+)$', raw, re.MULTILINE)
        if len(numbered) >= 2:
            return [g.strip() for g in numbered if g.strip()]

        # Method 4: Bullet list (- ..., * ...)
        bullets = re.findall(r'^\s*[-*]\s+(.+)$', raw, re.MULTILINE)
        if len(bullets) >= 2:
            return [g.strip() for g in bullets if g.strip()]

        return []

    def _deduplicate_sub_goals(self, sub_goals: List[str]) -> List[str]:
        """Remove duplicate or highly similar sub-goals using word overlap."""
        if len(sub_goals) <= 1:
            return sub_goals

        def get_key_words(text: str) -> set:
            stop_words = {'the', 'a', 'an', 'and', 'or', 'for', 'to', 'of', 'in', 'on',
                          'with', 'is', 'are', 'be', 'that', 'this',
                          # Generic task words — exclude from similarity to avoid collapsing
                          # "Research Python..." and "Research Rust..." as duplicates
                          'research', 'analyze', 'compare', 'evaluate', 'investigate',
                          'strengths', 'weaknesses', 'features', 'capabilities',
                          'development', 'programming', 'language', 'languages'}
            return set(w.lower() for w in text.split() if len(w) > 2 and w.lower() not in stop_words)

        def similarity(a: str, b: str) -> float:
            words_a, words_b = get_key_words(a), get_key_words(b)
            if not words_a or not words_b:
                return 0.0
            return len(words_a & words_b) / len(words_a | words_b)

        unique_goals = []
        for goal in sub_goals:
            if not any(similarity(goal, existing) > 0.5 for existing in unique_goals):
                unique_goals.append(goal)

        if len(unique_goals) < len(sub_goals):
            logger.info(f"Deduplicated: {len(sub_goals)} -> {len(unique_goals)} sub-goals")

        return unique_goals

    def _derive_agent_name(self, sub_goal: str, index: int) -> str:
        """Derive a logical, descriptive agent name from sub-goal."""
        import dspy

        goal_lower = sub_goal.lower()

        # Try LLM-based name extraction
        try:
            if hasattr(dspy.settings, 'lm') and dspy.settings.lm:
                class AgentNameSignature(dspy.Signature):
                    """Extract a short, descriptive agent name from task description.

                    The name should:
                    - Be 2-3 words max, joined by underscore
                    - Capture the MAIN TOPIC or ENTITY being worked on
                    - Be specific (not generic like "researcher" or "analyst")

                    Examples:
                    - "Research BaFin KGAB framework" -> "bafin_kgab"
                    - "Analyze Tesla stock technicals" -> "tesla_technicals"
                    - "Compare EU vs US regulations" -> "eu_us_regs"
                    - "Generate summary of AI news" -> "ai_news"
                    """
                    task: str = dspy.InputField(desc="Task description")
                    name: str = dspy.OutputField(desc="Short agent name (2-3 words, underscore separated, no generic words)")

                predictor = dspy.Predict(AgentNameSignature)
                result = predictor(task=sub_goal)
                name = result.name.strip().lower()
                name = re.sub(r'["\']', '', name)
                name = re.sub(r'\s+', '_', name)
                name = name[:20]
                if name and len(name) >= 3:
                    return name
        except Exception as e:
            logger.debug(f"LLM agent naming failed, using heuristics: {e}")

        # Heuristic fallback
        entities = re.findall(r'\b([A-Z]{2,6})\b', sub_goal)
        proper_nouns = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', sub_goal)

        domain_patterns = [
            (r'fundamental', 'fundamentals'),
            (r'technical', 'technicals'),
            (r'sentiment', 'sentiment'),
            (r'regulatory|regulation|compliance', 'regulatory'),
            (r'risk\s*management', 'risk_mgmt'),
            (r'market\s*analysis', 'market'),
            (r'competitor', 'competitors'),
            (r'valuation', 'valuation'),
            (r'earnings', 'earnings'),
            (r'news|headline', 'news'),
        ]

        for pattern, name in domain_patterns:
            if re.search(pattern, goal_lower):
                if entities:
                    return f"{entities[0].lower()}_{name}"
                return name

        cleaned = re.sub(
            r'^(research|analyze|generate|create|get|find|compare|evaluate|assess|review|summarize|identify)\s+',
            '', goal_lower,
        )
        cleaned = re.sub(r'\s+(analysis|report|data|information|research|requirements|framework)$', '', cleaned)
        cleaned = re.sub(r'\b(the|a|an|and|or|for|with|from|to|of|on|in)\b', '', cleaned)

        words = [w.strip() for w in cleaned.split() if len(w.strip()) > 2]
        if words:
            return '_'.join(words[:2])[:18]

        if entities:
            return entities[0].lower()
        if proper_nouns:
            return proper_nouns[0].lower().replace(' ', '_')[:15]

        return f'task_{index + 1}'
