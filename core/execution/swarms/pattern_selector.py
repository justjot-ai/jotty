"""
Pattern Selector - AUTO Pattern Selection for Unified Swarm
===========================================================

Intelligently selects the best coordination pattern based on:
1. Task analysis (keywords, structure, complexity)
2. Historical learning (TD-Lambda Q-values)
3. Swarm Intelligence (transfer learning)
4. Memory retrieval (similar past tasks)
5. Fallback heuristics (keyword-based rules)

Used by SwarmLearning when COORDINATION = CoordinationPattern.AUTO.

Author: Jotty Team
Date: February 2026
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from Jotty.core.infrastructure.foundation.types.execution_types import CoordinationPattern

logger = logging.getLogger(__name__)


class PatternSelector:
    """Selects optimal coordination pattern using multi-source learning."""

    def __init__(
        self,
        memory=None,
        td_learner=None,
        swarm_intelligence=None,
        swarm_name: str = "base_swarm",
    ) -> None:
        """
        Initialize pattern selector with learning components.

        Args:
            memory: MemorySystem for retrieving similar past tasks
            td_learner: TDLambdaLearner for Q-value lookup
            swarm_intelligence: SwarmIntelligence for transfer learning
            swarm_name: Name of swarm (for learning tracking)
        """
        self.memory = memory
        self.td_learner = td_learner
        self.swarm_intelligence = swarm_intelligence
        self.swarm_name = swarm_name

    async def select_pattern(
        self, task: Any, context: Dict[str, Any] | None = None, agent_count: int = 1
    ) -> CoordinationPattern:
        """
        Select optimal coordination pattern.

        Uses 5-layer decision making:
        1. Memory: Retrieve similar past tasks and their successful patterns
        2. TD-Lambda: Check learned Q-values for state-action pairs
        3. Swarm Intelligence: Use transfer learning from similar task types
        4. Task Analysis: Analyze task structure and keywords
        5. Fallback: Rule-based heuristics

        Args:
            task: The task to execute (str or dict)
            context: Additional context about the task
            agent_count: Number of agents in the team

        Returns:
            Recommended CoordinationPattern
        """
        task_str = str(task) if task else ""
        context = context or {}

        # Layer 1: Memory retrieval (best pattern from similar past tasks)
        pattern_from_memory = await self._check_memory(task_str)
        if pattern_from_memory:
            logger.info(f"Pattern from memory: {pattern_from_memory.value}")
            return pattern_from_memory

        # Layer 2: TD-Lambda learned values (state = task_type, action = pattern)
        pattern_from_td = self._check_td_lambda(task_str)
        if pattern_from_td:
            logger.info(f"Pattern from TD-Lambda: {pattern_from_td.value}")
            return pattern_from_td

        # Layer 3: Swarm Intelligence transfer learning
        pattern_from_si = self._check_swarm_intelligence(task_str)
        if pattern_from_si:
            logger.info(f"Pattern from Swarm Intelligence: {pattern_from_si.value}")
            return pattern_from_si

        # Layer 4: Task analysis (keywords, structure)
        pattern_from_analysis = self._analyze_task(task_str, agent_count)
        if pattern_from_analysis:
            logger.info(f"Pattern from task analysis: {pattern_from_analysis.value}")
            return pattern_from_analysis

        # Layer 5: Fallback heuristic
        fallback = self._fallback_heuristic(agent_count)
        logger.info(f"Pattern from fallback: {fallback.value}")
        return fallback

    async def _check_memory(self, task_str: str) -> Optional[CoordinationPattern]:
        """Check memory for similar tasks and their successful patterns."""
        if not self.memory:
            return None

        try:
            from Jotty.core.infrastructure.foundation.data_structures import MemoryLevel

            # Retrieve similar procedural memories (learned patterns)
            results = self.memory.retrieve(
                query=task_str,
                level=MemoryLevel.PROCEDURAL,
                goal="pattern_selection",
                top_k=3,
            )

            if not results:
                return None

            # Find most successful pattern from similar tasks
            pattern_scores = {}
            for result in results:
                try:
                    import json

                    data = json.loads(result.content)
                    pattern = data.get("pattern")
                    success = data.get("success", 0.5)
                    relevance = result.relevance

                    if pattern:
                        # Weight by both success rate and relevance
                        score = success * relevance
                        pattern_scores[pattern] = pattern_scores.get(pattern, 0) + score
                except Exception:
                    continue

            if pattern_scores:
                best_pattern = max(pattern_scores.items(), key=lambda x: x[1])[0]
                try:
                    return CoordinationPattern(best_pattern)
                except ValueError:
                    return None

        except Exception as e:
            logger.debug(f"Memory pattern check failed: {e}")

        return None

    def _check_td_lambda(self, task_str: str) -> Optional[CoordinationPattern]:
        """Check TD-Lambda for learned Q-values."""
        if not self.td_learner:
            return None

        try:
            # State = task characteristics
            state_hash = self._hash_task(task_str)
            state = {"swarm": self.swarm_name, "task_hash": state_hash}

            # Get Q-values for all pattern actions
            pattern_q_values = {}
            for pattern in CoordinationPattern:
                if pattern in (
                    CoordinationPattern.NONE,
                    CoordinationPattern.PIPELINE,
                    CoordinationPattern.AUTO,
                    CoordinationPattern.CUSTOM,
                ):
                    continue  # Skip meta-patterns

                action = {"pattern": pattern.value}
                q_value = self.td_learner.get_q_value(state, action)
                if q_value > 0.1:  # Only consider if learned (not default 0)
                    pattern_q_values[pattern] = q_value

            if pattern_q_values:
                best_pattern = max(pattern_q_values.items(), key=lambda x: x[1])[0]
                logger.debug(
                    f"TD-Lambda Q-values: {[(p.value, round(q, 2)) for p, q in pattern_q_values.items()]}"
                )
                return best_pattern

        except Exception as e:
            logger.debug(f"TD-Lambda pattern check failed: {e}")

        return None

    def _check_swarm_intelligence(self, task_str: str) -> Optional[CoordinationPattern]:
        """Check Swarm Intelligence for transfer learning recommendations."""
        if not self.swarm_intelligence:
            return None

        try:
            # Get task type from keywords
            task_type = self._extract_task_type(task_str)

            # Check if similar task types have successful patterns
            if hasattr(self.swarm_intelligence, "pattern_learner"):
                pattern = self.swarm_intelligence.pattern_learner.get_best_pattern(task_type)
                if pattern:
                    try:
                        return CoordinationPattern(pattern)
                    except ValueError:
                        return None

        except Exception as e:
            logger.debug(f"Swarm Intelligence pattern check failed: {e}")

        return None

    def _analyze_task(self, task_str: str, agent_count: int) -> Optional[CoordinationPattern]:
        """
        Analyze task structure and keywords to suggest pattern.

        Rules:
        - Multiple independent items → PARALLEL
        - Sequence words (then, after, next) → SEQUENTIAL
        - Comparison/review words → CONSENSUS
        - Discussion/debate words → DEBATE
        - Improvement/iteration words → ITERATIVE
        - Complex multi-step → HIERARCHICAL
        - Incremental building → BLACKBOARD
        """
        task_lower = task_str.lower()

        # PARALLEL: Multiple independent items
        if any(
            word in task_lower
            for word in ["each", "all", "every", "multiple", "several", "compare"]
        ):
            if agent_count >= 2:
                # Check for list-like structure
                if re.search(r"\d+\.", task_str) or re.search(r"[-•*]\s+", task_str):
                    return CoordinationPattern.PARALLEL
                # Check for "compare X, Y, Z"
                if "," in task_str and agent_count >= 3:
                    return CoordinationPattern.PARALLEL

        # SEQUENTIAL: Sequence words
        if any(
            word in task_lower for word in ["then", "after", "next", "first", "step", "pipeline"]
        ):
            return CoordinationPattern.SEQUENTIAL

        # CONSENSUS: Review/voting words
        if any(word in task_lower for word in ["review", "evaluate", "vote", "agree", "consensus"]):
            if agent_count >= 3:
                return CoordinationPattern.CONSENSUS

        # DEBATE: Discussion words
        if any(
            word in task_lower
            for word in ["debate", "discuss", "argue", "pros and cons", "perspectives"]
        ):
            if agent_count >= 2:
                return CoordinationPattern.DEBATE

        # ITERATIVE: Improvement/loop words
        if any(
            word in task_lower
            for word in ["improve", "refine", "iterate", "until", "optimize", "enhance"]
        ):
            return CoordinationPattern.ITERATIVE

        # HIERARCHICAL: Complex/delegation words
        if any(
            word in task_lower
            for word in ["manage", "delegate", "coordinate", "oversee", "complex project"]
        ):
            if agent_count >= 3:
                return CoordinationPattern.HIERARCHICAL

        # BLACKBOARD: Incremental building
        if any(
            word in task_lower
            for word in [
                "collaborate",
                "build",
                "contribute",
                "incrementally",
                "shared",
                "workspace",
            ]
        ):
            if agent_count >= 2:
                return CoordinationPattern.BLACKBOARD

        return None

    def _fallback_heuristic(self, agent_count: int) -> CoordinationPattern:
        """Fallback heuristic based on agent count."""
        if agent_count == 1:
            return CoordinationPattern.SEQUENTIAL
        elif agent_count == 2:
            return CoordinationPattern.SEQUENTIAL  # Pipeline works well for pairs
        elif agent_count >= 3:
            return CoordinationPattern.PARALLEL  # Default to parallel for teams
        else:
            return CoordinationPattern.SEQUENTIAL

    def _hash_task(self, task_str: str) -> str:
        """Create stable hash of task characteristics."""
        import hashlib

        # Extract key features
        task_lower = task_str.lower()
        keywords = self._extract_keywords(task_lower)
        task_type = self._extract_task_type(task_lower)

        # Hash combination of type + keywords
        features = f"{task_type}:{','.join(sorted(keywords))}"
        return hashlib.md5(features.encode()).hexdigest()[:8]

    def _extract_task_type(self, task_str: str) -> str:
        """Extract high-level task type from string."""
        task_lower = task_str.lower()

        # Primary task types
        if any(w in task_lower for w in ["research", "find", "search", "investigate"]):
            return "research"
        elif any(w in task_lower for w in ["analyze", "analysis", "examine"]):
            return "analysis"
        elif any(w in task_lower for w in ["code", "implement", "develop", "program"]):
            return "coding"
        elif any(w in task_lower for w in ["review", "evaluate", "assess"]):
            return "review"
        elif any(w in task_lower for w in ["write", "create", "generate", "compose"]):
            return "generation"
        elif any(w in task_lower for w in ["test", "verify", "validate"]):
            return "testing"
        else:
            return "general"

    def _extract_keywords(self, task_str: str) -> List[str]:
        """Extract key action words from task."""
        action_words = [
            "research",
            "analyze",
            "code",
            "review",
            "test",
            "write",
            "compare",
            "evaluate",
            "improve",
            "optimize",
            "debug",
            "fix",
            "create",
            "design",
            "implement",
        ]

        words = set()
        for word in action_words:
            if word in task_str.lower():
                words.add(word)

        return list(words)


__all__ = ["PatternSelector"]
