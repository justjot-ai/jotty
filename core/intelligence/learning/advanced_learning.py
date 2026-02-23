"""
Advanced Learning Components
============================

Closes 5 gaps + integrates 3 research techniques into Jotty's learning system:

1. LLMJudge         — LLM-based quality scoring via Haiku (augments heuristic)
2. Reflexion        — Natural language failure reflection (Shinn et al. 2023)
3. FewShotCurator   — Auto-curate best episodes as DSPy few-shot examples
4. MCTSPlanner      — Language Agent Tree Search for multi-step planning (LATS)
5. VoyagerSkillLib  — Auto-extract reusable skill patterns from successes

Each component integrates with existing infrastructure:
- LearningStore (SQLite) for persistence
- LearningService for episode recording
- TDLambdaLearner for value updates
- SwarmTaskBoard for task planning
- DSPy for structured LLM interaction
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# 1. LLM-AS-JUDGE — Replaces pure-heuristic quality scoring
# =============================================================================


@dataclass
class JudgeVerdict:
    """Result from LLM quality judge."""

    quality: float  # 0.0 - 1.0
    reasoning: str
    aspects: Dict[str, float] = field(default_factory=dict)
    source: str = "llm"  # "llm" or "heuristic" (fallback)


class LLMJudge:
    """LLM-based quality judge using a cheap model (Haiku).

    Falls back to heuristic scoring when LLM is unavailable or fails.
    Uses DSPy ChainOfThought for structured output.

    Integration: Called from LearningService.record() to upgrade
    quality scores, and from _post_execute_learning for episode grading.
    """

    _instance: Optional["LLMJudge"] = None

    def __init__(self) -> None:
        self._lm: Any = None
        self._judge_module: Any = None
        self._init_attempts = 0

    @classmethod
    def get_instance(cls) -> "LLMJudge":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _ensure_init(self) -> bool:
        """Lazy-init the DSPy judge module. Returns True if ready."""
        if self._judge_module is not None:
            return True
        if self._init_attempts > 2:
            return False
        self._init_attempts += 1
        try:
            import dspy

            from Jotty.core.infrastructure.foundation.unified_lm_provider import (
                get_fast_lm,
            )

            self._lm = get_fast_lm()

            class QualityJudgeSignature(dspy.Signature):
                """Judge the quality of an AI-generated response for a given goal.
                Score from 0.0 (terrible) to 1.0 (excellent).
                Consider: accuracy, completeness, structure, depth, actionability."""

                goal: str = dspy.InputField(desc="The original task/goal")
                response_excerpt: str = dspy.InputField(desc="First 2000 chars of the response")
                quality_score: float = dspy.OutputField(desc="Quality score 0.0-1.0")
                reasoning: str = dspy.OutputField(desc="Brief reasoning for score")

            self._judge_module = dspy.ChainOfThought(QualityJudgeSignature)
            return True
        except Exception as e:
            logger.debug(f"LLMJudge init failed: {e}")
            return False

    def judge(self, goal: str, response: str, heuristic_score: float = 0.0) -> JudgeVerdict:
        """Score response quality using LLM, with heuristic fallback.

        The final score blends LLM and heuristic scores (70/30) for
        robustness — the heuristic catches structural issues the LLM
        might miss, while the LLM catches semantic quality.

        Args:
            goal: The original task/goal
            response: The full response text
            heuristic_score: Pre-computed heuristic score (from analyze_response)

        Returns:
            JudgeVerdict with blended quality score
        """
        if not response or not self._ensure_init():
            return JudgeVerdict(
                quality=heuristic_score, reasoning="LLM judge unavailable", source="heuristic"
            )

        try:
            import dspy

            excerpt = response[:2000]
            with dspy.context(lm=self._lm):
                result = self._judge_module(goal=goal, response_excerpt=excerpt)

            llm_score = float(result.quality_score)
            llm_score = max(0.0, min(1.0, llm_score))

            # Blend: 70% LLM + 30% heuristic for robustness
            blended = 0.7 * llm_score + 0.3 * heuristic_score
            return JudgeVerdict(
                quality=round(blended, 3),
                reasoning=str(result.reasoning)[:300],
                aspects={"llm_raw": llm_score, "heuristic": heuristic_score},
                source="llm",
            )
        except Exception as e:
            logger.debug(f"LLMJudge scoring failed, using heuristic: {e}")
            return JudgeVerdict(
                quality=heuristic_score,
                reasoning=f"Fallback to heuristic: {e}",
                source="heuristic",
            )


# =============================================================================
# 2. REFLEXION — Natural language failure reflection (Shinn et al.)
# =============================================================================


class Reflexion:
    """Reflect on execution failures, store reflections, and retrieve on retry.

    Implements the Reflexion paper's key insight: when an agent fails, it
    generates a natural language "reflection" on what went wrong. On the
    next attempt at a similar task, the reflection is injected into the
    prompt, preventing repeated mistakes.

    Integration:
    - _post_execute_learning calls reflect_on_failure() for failed episodes
    - _pre_execute_learning calls get_relevant_reflections() to inject context
    - ReflectionRecord (already in LearningStore) persists reflections
    """

    _instance: Optional["Reflexion"] = None

    def __init__(self) -> None:
        self._lm: Any = None
        self._reflect_module: Any = None
        self._init_attempts = 0

    @classmethod
    def get_instance(cls) -> "Reflexion":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _ensure_init(self) -> bool:
        if self._reflect_module is not None:
            return True
        if self._init_attempts > 2:
            return False
        self._init_attempts += 1
        try:
            import dspy

            from Jotty.core.infrastructure.foundation.unified_lm_provider import (
                get_fast_lm,
            )

            self._lm = get_fast_lm()

            class ReflectionSignature(dspy.Signature):
                """Reflect on a failed AI execution to learn what went wrong.
                Produce a concise, actionable reflection that will help on retry."""

                goal: str = dspy.InputField(desc="What the agent was trying to do")
                output_excerpt: str = dspy.InputField(
                    desc="What the agent actually produced (truncated)"
                )
                error_info: str = dspy.InputField(desc="Error type and message, if any")
                observation: str = dspy.OutputField(desc="What happened (factual, 1-2 sentences)")
                analysis: str = dspy.OutputField(desc="Why it failed (root cause, 1-2 sentences)")
                adjustment: str = dspy.OutputField(
                    desc="What to do differently next time (actionable, 1-2 sentences)"
                )

            self._reflect_module = dspy.ChainOfThought(ReflectionSignature)
            return True
        except Exception as e:
            logger.debug(f"Reflexion init failed: {e}")
            return False

    def reflect_on_failure(
        self,
        episode_id: str,
        unit_name: str,
        goal: str,
        output: str,
        error_type: str = "",
        error_message: str = "",
        step: int = 0,
    ) -> Optional[Dict[str, str]]:
        """Generate a reflection on a failed episode and persist it.

        Args:
            episode_id: The failed episode's ID
            unit_name: Name of the agent/swarm that failed
            goal: What was being attempted
            output: What was actually produced
            error_type: Classification of the error
            error_message: Error details

        Returns:
            Dict with observation/analysis/adjustment, or None on failure
        """
        if not self._ensure_init():
            return None

        try:
            import dspy

            from .learning_store import LearningStore, ReflectionRecord

            error_info = (
                f"{error_type}: {error_message}"
                if error_type
                else "No explicit error; output was low quality"
            )

            with dspy.context(lm=self._lm):
                result = self._reflect_module(
                    goal=goal,
                    output_excerpt=output[:1500],
                    error_info=error_info[:500],
                )

            reflection = ReflectionRecord(
                reflection_id=f"refl_{uuid.uuid4().hex[:12]}",
                episode_id=episode_id,
                step=step,
                unit_name=unit_name,
                observation=str(result.observation)[:500],
                analysis=str(result.analysis)[:500],
                adjustment=str(result.adjustment)[:500],
                applied=False,
            )

            store = LearningStore.get_instance()
            store.save_reflection(reflection)
            logger.info(
                f"Reflexion: stored reflection for {unit_name} "
                f"(episode {episode_id[:12]}): {reflection.adjustment[:80]}"
            )
            return {
                "observation": reflection.observation,
                "analysis": reflection.analysis,
                "adjustment": reflection.adjustment,
            }
        except Exception as e:
            logger.debug(f"Reflexion generation failed: {e}")
            return None

    def get_relevant_reflections(self, unit_name: str, limit: int = 3) -> List[str]:
        """Retrieve recent reflections for a unit, formatted for prompt injection.

        Returns actionable adjustment strings ready to inject into the agent's
        system prompt as "lessons learned from past failures."
        """
        try:
            from .learning_store import LearningStore

            store = LearningStore.get_instance()
            reflections = store.get_reflections(unit_name=unit_name, limit=limit)
            if not reflections:
                return []
            return [
                f"[Past failure] {r.observation} → Fix: {r.adjustment}"
                for r in reflections
                if r.adjustment
            ]
        except Exception as e:
            logger.debug(f"Reflexion retrieval failed: {e}")
            return []


# =============================================================================
# 3. FEW-SHOT CURATOR — Auto-curate best episodes as DSPy examples
# =============================================================================


class FewShotCurator:
    """Auto-curate successful episodes from LearningStore as DSPy examples.

    Bridges the gap between Jotty's episode recording and DSPy's few-shot
    optimization. Queries the best episodes (high quality + success) and
    converts them into dspy.Example objects for BootstrapFewShot or MIPRO.

    Integration:
    - Agents can call get_examples() to seed their DSPy modules
    - BootstrapFewShot can use curated examples as the training set
    - MIPRO can use them for prompt optimization
    """

    _instance: Optional["FewShotCurator"] = None

    @classmethod
    def get_instance(cls) -> "FewShotCurator":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_examples(
        self,
        domain: str = "",
        task_type: str = "",
        n: int = 5,
        min_quality: float = 0.7,
    ) -> List[Any]:
        """Query best episodes and convert to dspy.Example objects.

        Each Example has fields: goal, approach, output — matching the
        typical agent signature pattern.

        Args:
            domain: Filter by domain (e.g., "coding", "research")
            task_type: Filter by task type
            n: Number of examples to return
            min_quality: Minimum quality threshold

        Returns:
            List of dspy.Example objects ready for BootstrapFewShot
        """
        try:
            import dspy

            from .learning_store import LearningStore

            store = LearningStore.get_instance()
            episodes = store.query_episodes(
                domain=domain or None,
                task_type=task_type or None,
                success_only=True,
                limit=n * 3,  # Over-fetch then filter by quality
            )

            # Filter by quality and sort descending
            episodes = [ep for ep in episodes if ep.quality >= min_quality]
            episodes.sort(key=lambda e: e.quality, reverse=True)
            episodes = episodes[:n]

            examples = []
            for ep in episodes:
                goal = ep.context.get("task", ep.context.get("goal", ep.task_type))
                approach = ep.action.get("paradigm", ep.action.get("model", ""))
                output_text = ep.outcome.get("content", ep.outcome.get("summary", ""))

                if not goal or not output_text:
                    continue

                ex = dspy.Example(
                    goal=str(goal)[:500],
                    approach=str(approach)[:200],
                    output=str(output_text)[:2000],
                ).with_inputs("goal")
                examples.append(ex)

            logger.debug(
                f"FewShotCurator: curated {len(examples)} examples "
                f"(domain={domain}, task_type={task_type})"
            )
            return examples
        except Exception as e:
            logger.debug(f"FewShotCurator failed: {e}")
            return []

    def get_distilled_examples(
        self,
        domain: str = "",
        agent_name: str = "",
        n: int = 10,
    ) -> List[Any]:
        """Get DSPy examples from distilled lessons (more concise than episodes).

        Uses the DistilledLesson table for compact, LLM-extracted facts.
        """
        try:
            import dspy

            from .learning_store import LearningStore

            store = LearningStore.get_instance()
            lessons = store.get_distilled_lessons(
                domain=domain or None,
                agent_name=agent_name or None,
                min_confidence=0.5,
                limit=n,
            )

            return [
                dspy.Example(
                    context=lesson.applicability,
                    lesson=lesson.lesson,
                    confidence=str(lesson.confidence),
                ).with_inputs("context")
                for lesson in lessons
                if lesson.lesson
            ]
        except Exception as e:
            logger.debug(f"FewShotCurator distilled failed: {e}")
            return []

    def optimize_module(
        self,
        module: Any,
        domain: str = "",
        task_type: str = "",
        n_examples: int = 10,
        max_bootstrapped: int = 4,
    ) -> Any:
        """Run DSPy BootstrapFewShot on a module using curated episodes.

        This is the key integration: episodes recorded by LearningService
        are used to optimize DSPy agent signatures automatically.

        Args:
            module: DSPy module to optimize
            domain: Episode domain filter
            task_type: Episode task type filter
            n_examples: Number of training examples
            max_bootstrapped: Max bootstrapped demos

        Returns:
            Optimized DSPy module (or original if optimization fails)
        """
        try:
            import dspy

            from Jotty.core.infrastructure.foundation.unified_lm_provider import (
                get_fast_lm,
            )

            examples = self.get_examples(domain, task_type, n=n_examples, min_quality=0.75)
            if len(examples) < 3:
                logger.debug("FewShotCurator: not enough examples for optimization")
                return module

            optimizer = dspy.BootstrapFewShot(
                max_bootstrapped_demos=max_bootstrapped,
                max_labeled_demos=min(len(examples), 8),
            )

            fast_lm = get_fast_lm()
            with dspy.context(lm=fast_lm):
                optimized = optimizer.compile(module, trainset=examples)

            logger.info(f"FewShotCurator: optimized module with {len(examples)} examples")
            return optimized
        except Exception as e:
            logger.debug(f"FewShotCurator optimization failed: {e}")
            return module


# =============================================================================
# 4. MCTS PLANNER — Language Agent Tree Search (LATS)
# =============================================================================


@dataclass
class MCTSNode:
    """Node in the MCTS search tree."""

    state: Dict[str, Any]
    action: str = ""  # Action that led to this node
    parent: Optional["MCTSNode"] = None
    children: List["MCTSNode"] = field(default_factory=list)
    value: float = 0.0
    visits: int = 0
    depth: int = 0

    @property
    def ucb(self) -> float:
        """UCB1 score for tree policy selection."""
        if self.visits == 0:
            return float("inf")
        if self.parent is None or self.parent.visits == 0:
            return self.value
        exploitation = self.value / self.visits
        exploration = 1.41 * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration


class MCTSPlanner:
    """Language Agent Tree Search (LATS) for multi-step planning.

    Uses LLM to:
    1. EXPAND: Generate possible next actions from current state
    2. EVALUATE: Score the quality of a plan path
    3. BACKPROPAGATE: Update node values up the tree
    4. SELECT: UCB1 to choose which branch to explore

    Integration:
    - SwarmTaskBoard.get_next_task() can use plan() for multi-step lookahead
    - Agents can use plan() for complex decomposition before acting
    - Results feed back into TDLambdaLearner for value learning

    Designed for moderate budgets (5-20 LLM calls), not exhaustive search.
    """

    def __init__(
        self,
        max_iterations: int = 10,
        max_depth: int = 5,
        n_expand: int = 3,
    ) -> None:
        self.max_iterations = max_iterations
        self.max_depth = max_depth
        self.n_expand = n_expand  # Actions to generate per expansion
        self._lm: Any = None
        self._expand_module: Any = None
        self._eval_module: Any = None

    def _ensure_init(self) -> bool:
        if self._expand_module is not None:
            return True
        try:
            import dspy

            from Jotty.core.infrastructure.foundation.unified_lm_provider import (
                get_fast_lm,
            )

            self._lm = get_fast_lm()

            class ExpandSignature(dspy.Signature):
                """Given a goal and current plan path, generate the next
                possible actions. Return exactly N actions, one per line."""

                goal: str = dspy.InputField()
                plan_so_far: str = dspy.InputField(desc="Actions taken so far (one per line)")
                n_actions: int = dspy.InputField(desc="Number of candidate actions to generate")
                candidate_actions: str = dspy.OutputField(
                    desc="Candidate next actions, one per line"
                )

            class EvalSignature(dspy.Signature):
                """Evaluate how promising a plan path is for achieving the goal.
                Score from 0.0 (will fail) to 1.0 (very likely to succeed)."""

                goal: str = dspy.InputField()
                plan_path: str = dspy.InputField(desc="Full sequence of planned actions")
                success_probability: float = dspy.OutputField(desc="Probability of success 0.0-1.0")

            self._expand_module = dspy.Predict(ExpandSignature)
            self._eval_module = dspy.Predict(EvalSignature)
            return True
        except Exception as e:
            logger.debug(f"MCTSPlanner init failed: {e}")
            return False

    def plan(self, goal: str, context: str = "") -> List[str]:
        """Run MCTS to find the best action sequence for a goal.

        Args:
            goal: What to achieve
            context: Additional context (past reflections, learned context)

        Returns:
            Ordered list of action strings (best plan found)
        """
        if not self._ensure_init():
            return [goal]  # Fallback: treat goal as single action

        root = MCTSNode(state={"goal": goal, "context": context})

        for _ in range(self.max_iterations):
            # SELECT — walk tree using UCB1
            node = self._select(root)

            if node.depth >= self.max_depth:
                # At max depth, just evaluate
                value = self._evaluate(node, goal)
                self._backpropagate(node, value)
                continue

            # EXPAND — generate child actions
            children = self._expand(node, goal)
            if not children:
                break

            # EVALUATE — score the best new child
            best_child = children[0]
            value = self._evaluate(best_child, goal)

            # BACKPROPAGATE — update values up the tree
            self._backpropagate(best_child, value)

        return self._extract_best_plan(root)

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Walk tree using UCB1 until reaching a leaf."""
        while node.children:
            node = max(node.children, key=lambda n: n.ucb)
        return node

    def _expand(self, node: MCTSNode, goal: str) -> List[MCTSNode]:
        """Generate child nodes by asking LLM for possible next actions."""
        try:
            import dspy

            plan_so_far = self._path_to_string(node)
            with dspy.context(lm=self._lm):
                result = self._expand_module(
                    goal=goal,
                    plan_so_far=plan_so_far or "(no steps yet)",
                    n_actions=self.n_expand,
                )

            actions = [a.strip() for a in str(result.candidate_actions).split("\n") if a.strip()]
            actions = actions[: self.n_expand]

            children = []
            for action in actions:
                child = MCTSNode(
                    state={**node.state, f"step_{node.depth + 1}": action},
                    action=action,
                    parent=node,
                    depth=node.depth + 1,
                )
                node.children.append(child)
                children.append(child)

            return children
        except Exception as e:
            logger.debug(f"MCTS expand failed: {e}")
            return []

    def _evaluate(self, node: MCTSNode, goal: str) -> float:
        """Evaluate a node's plan path using LLM."""
        try:
            import dspy

            plan_path = self._path_to_string(node)
            with dspy.context(lm=self._lm):
                result = self._eval_module(goal=goal, plan_path=plan_path)
            score = float(result.success_probability)
            return max(0.0, min(1.0, score))
        except Exception:
            return 0.5  # Neutral default

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """Update values from leaf to root."""
        current: Optional[MCTSNode] = node
        while current is not None:
            current.visits += 1
            current.value += value
            current = current.parent

    def _path_to_string(self, node: MCTSNode) -> str:
        """Extract action sequence from root to this node."""
        actions = []
        current: Optional[MCTSNode] = node
        while current is not None:
            if current.action:
                actions.append(current.action)
            current = current.parent
        actions.reverse()
        return "\n".join(f"{i + 1}. {a}" for i, a in enumerate(actions))

    def _extract_best_plan(self, root: MCTSNode) -> List[str]:
        """Extract the highest-value path from root."""
        plan = []
        node = root
        while node.children:
            node = max(node.children, key=lambda n: n.value / max(n.visits, 1))
            plan.append(node.action)
        return plan if plan else [root.state.get("goal", "")]


# =============================================================================
# 5. VOYAGER-STYLE SKILL LIBRARY — Auto-extract reusable patterns
# =============================================================================


class VoyagerSkillLib:
    """Auto-extract and grow a library of reusable skill patterns.

    Inspired by Voyager (NVIDIA): when a task succeeds with high quality,
    extract the approach as a reusable "skill pattern" in PatternRecord.
    Future tasks can look up relevant patterns and reuse proven strategies.

    Integration:
    - Triggered by _post_execute_learning when quality > threshold
    - Patterns stored in LearningStore's patterns table
    - _pre_execute_learning retrieves patterns via get_applicable_patterns()
    """

    _instance: Optional["VoyagerSkillLib"] = None

    @classmethod
    def get_instance(cls) -> "VoyagerSkillLib":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def extract_skill_pattern(
        self,
        episode_id: str,
        domain: str,
        task_type: str,
        goal: str,
        approach: str,
        quality: float,
    ) -> Optional[str]:
        """Extract a reusable skill pattern from a successful high-quality episode.

        Only extracts if quality >= 0.8 to ensure patterns are reliable.

        Returns:
            Pattern ID if extracted, None otherwise
        """
        if quality < 0.8:
            return None

        try:
            from .learning_store import LearningStore, PatternRecord

            store = LearningStore.get_instance()

            # Check for existing similar pattern to avoid duplicates
            existing = store.get_patterns(domain=domain, pattern_type="success_strategy", limit=20)
            for p in existing:
                if p.description == goal[:200]:
                    # Boost confidence of existing pattern
                    p.confidence = min(1.0, p.confidence + 0.05)
                    p.evidence_count += 1
                    store.save_pattern(p)
                    return p.pattern_id

            pattern = PatternRecord(
                pattern_id=f"skill_{uuid.uuid4().hex[:12]}",
                source_domain=domain,
                pattern_type="success_strategy",
                description=goal[:200],
                conditions={"task_type": task_type, "min_quality": 0.7},
                recommendation=approach[:500],
                confidence=min(1.0, quality),
                evidence_count=1,
                applicable_domains=[domain],
                metadata={"episode_id": episode_id, "extracted_quality": quality},
            )
            store.save_pattern(pattern)
            logger.info(f"VoyagerSkillLib: extracted pattern '{goal[:60]}' (q={quality:.2f})")
            return pattern.pattern_id
        except Exception as e:
            logger.debug(f"VoyagerSkillLib extraction failed: {e}")
            return None

    def get_applicable_patterns(
        self,
        domain: str,
        task_type: str = "",
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Get proven skill patterns applicable to current task.

        Returns patterns sorted by confidence, ready for prompt injection.
        """
        try:
            from .learning_store import LearningStore

            store = LearningStore.get_instance()
            patterns = store.get_patterns(
                domain=domain,
                pattern_type="success_strategy",
                min_confidence=0.5,
                limit=limit,
            )
            return [
                {
                    "strategy": p.recommendation,
                    "confidence": p.confidence,
                    "evidence": p.evidence_count,
                    "description": p.description,
                }
                for p in patterns
            ]
        except Exception as e:
            logger.debug(f"VoyagerSkillLib retrieval failed: {e}")
            return []


# =============================================================================
# UNIFIED CONTEXT BUILDER — Assembles all learning signals for prompt injection
# =============================================================================


def build_advanced_learning_context(
    unit_name: str,
    domain: str = "",
    task_type: str = "",
    goal: str = "",
    max_lines: int = 15,
) -> str:
    """Build a comprehensive learning context string for prompt injection.

    Combines signals from all advanced learning components:
    1. Past reflections (Reflexion)
    2. Proven skill patterns (VoyagerSkillLib)
    3. Distilled lessons (FewShotCurator)

    This augments the existing get_learned_context() from td_lambda.py.

    Args:
        unit_name: Current agent/swarm name
        domain: Current domain
        task_type: Current task type
        goal: Current goal
        max_lines: Maximum output lines

    Returns:
        Formatted context string for prompt injection
    """
    lines: List[str] = []

    # 1. Reflexion: past failure lessons
    try:
        reflexion = Reflexion.get_instance()
        reflections = reflexion.get_relevant_reflections(unit_name, limit=3)
        for r in reflections[:3]:
            lines.append(r)
    except Exception:
        pass

    # 2. VoyagerSkillLib: proven strategies
    try:
        skill_lib = VoyagerSkillLib.get_instance()
        patterns = skill_lib.get_applicable_patterns(domain, task_type, limit=3)
        for p in patterns[:3]:
            conf = p["confidence"]
            lines.append(f"[Proven strategy, {conf:.0%} confidence] {p['strategy'][:150]}")
    except Exception:
        pass

    # 3. Distilled lessons
    try:
        curator = FewShotCurator.get_instance()
        examples = curator.get_distilled_examples(domain=domain, n=3)
        for ex in examples[:3]:
            lines.append(f"[Learned lesson] {ex.lesson[:150]}")
    except Exception:
        pass

    if not lines:
        return ""

    return "ADVANCED LEARNING CONTEXT:\n" + "\n".join(f"- {line}" for line in lines[:max_lines])
