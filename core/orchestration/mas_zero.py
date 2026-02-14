"""
MAS-ZERO Inspired Components
=============================

Inference-time MAS optimization inspired by MAS-ZERO (Ke et al., NeurIPS 2025).
Implements 8 key improvements for Jotty v2:

1. CandidateVerifier     — LLM-based answer selection from multiple candidates
2. MetaFeedbackEvaluator — Solvability + completeness checks at MAS level
3. ExperienceLibrary     — Per-problem experience accumulation
4. BuildingBlocksRunner  — Run multiple strategies in parallel (MAS-Init)
5. TaskDifficulty        — TOO_HARD escalation signal
6. SubTaskStrategy       — Per-subtask MAS strategy assignment

DRY: Uses existing DSPy infrastructure, reuses TaskPlanner patterns.
KISS: Minimal classes, no unnecessary abstractions.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from Jotty.core.foundation.exceptions import AgentExecutionError

logger = logging.getLogger(__name__)

# Lazy DSPy import — thread-safe via double-checked locking
_dspy = None
_dspy_lock = threading.Lock()


def _get_dspy():
    global _dspy
    if _dspy is None:
        with _dspy_lock:
            if _dspy is None:
                import dspy
                _dspy = dspy
    return _dspy


# =============================================================================
# ENUMS & DATA STRUCTURES
# =============================================================================

class TaskDifficulty(Enum):
    """Agent's self-assessed difficulty of a sub-task."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    TOO_HARD = "too_hard"  # Triggers re-decomposition


class SubTaskStrategy(Enum):
    """Strategy to use for a sub-task (MAS-ZERO building blocks)."""
    DIRECT = "direct"              # Single agent, direct execution
    SELF_REFINE = "self_refine"    # Execute then self-critique
    ENSEMBLE = "ensemble"          # Multi-perspective then synthesize
    DECOMPOSE = "decompose"        # Further break down sub-task


@dataclass
class EpisodeExperience:
    """Single experience entry within a problem's execution."""
    iteration: int
    agent_name: str
    sub_goal: str
    strategy: str
    output_summary: str
    success: bool
    feedback: str = ""
    solvability: float = 0.0     # 0-1: how solvable was this sub-task
    completeness: float = 0.0    # 0-1: how complete relative to goal
    timestamp: float = field(default_factory=time.time)


@dataclass
class CandidateAnswer:
    """A candidate answer from any strategy/iteration."""
    source: str           # e.g., "direct", "ensemble", "iteration_2"
    agent_name: str
    output: Any
    success: bool
    confidence: float = 0.5
    execution_time: float = 0.0


# =============================================================================
# EXPERIENCE LIBRARY (Per-Problem Memory)
# =============================================================================

class ExperienceLibrary:
    """
    Accumulates MAS designs + intermediate outputs + feedback within
    a single problem's execution. Enables iterative refinement.

    KISS: Just a list with helper methods. No database, no persistence.
    """

    def __init__(self):
        self.experiences: List[EpisodeExperience] = []
        self.candidates: List[CandidateAnswer] = []
        self._iteration = 0

    def record(self, agent_name: str, sub_goal: str, strategy -> None: str,
               output_summary: str, success: bool, feedback: str = "",
               solvability: float = 0.0, completeness: float = 0.0):
        """Record an experience from this problem's execution."""
        self.experiences.append(EpisodeExperience(
            iteration=self._iteration,
            agent_name=agent_name,
            sub_goal=sub_goal,
            strategy=strategy,
            output_summary=output_summary[:500],
            success=success,
            feedback=feedback,
            solvability=solvability,
            completeness=completeness,
        ))

    def add_candidate(self, source: str, agent_name: str, output -> None: Any,
                      success: bool, confidence: float = 0.5,
                      execution_time: float = 0.0):
        """Add a candidate answer to the pool."""
        self.candidates.append(CandidateAnswer(
            source=source,
            agent_name=agent_name,
            output=output,
            success=success,
            confidence=confidence,
            execution_time=execution_time,
        ))

    def next_iteration(self) -> None:
        """Advance to next refinement iteration."""
        self._iteration += 1

    def get_context_summary(self, max_chars: int = 2000) -> str:
        """Get experience summary for LLM context (for meta-design)."""
        if not self.experiences:
            return ""

        lines = ["## Previous Iterations:"]
        for exp in self.experiences[-6:]:  # Last 6 experiences
            status = "OK" if exp.success else "FAILED"
            lines.append(
                f"- Iter {exp.iteration} [{exp.agent_name}] {status}: "
                f"{exp.sub_goal[:80]} (solvability={exp.solvability:.0%})"
            )
            if exp.feedback:
                lines.append(f"  Feedback: {exp.feedback[:120]}")

        summary = "\n".join(lines)
        return summary[:max_chars]

    @property
    def successful_candidates(self) -> List[CandidateAnswer]:
        return [c for c in self.candidates if c.success]

    def clear(self) -> None:
        """Reset for new problem."""
        self.experiences.clear()
        self.candidates.clear()
        self._iteration = 0


# =============================================================================
# CANDIDATE VERIFIER (MAS-Verify)
# =============================================================================

class CandidateVerifier:
    """
    Selects the best answer from multiple candidates using LLM verification.

    MAS-ZERO approach: rank by frequency, filter invalid, LLM picks best.
    KISS: Single LLM call for selection. Falls back to highest confidence.
    DRY: Reuses existing DSPy LM.
    """

    def verify(self, goal: str, candidates: List[CandidateAnswer]) -> CandidateAnswer:
        """
        Select the best candidate answer.

        Args:
            goal: Original task goal
            candidates: List of candidate answers

        Returns:
            Best CandidateAnswer
        """
        successful = [c for c in candidates if c.success]

        if not successful:
            # Return least-bad candidate
            return max(candidates, key=lambda c: c.confidence) if candidates else None

        if len(successful) == 1:
            return successful[0]

        # Try LLM-based verification
        try:
            return self._llm_verify(goal, successful)
        except Exception as e:
            logger.debug(f"LLM verification failed, using confidence fallback: {e}")
            return max(successful, key=lambda c: c.confidence)

    def _llm_verify(self, goal: str, candidates: List[CandidateAnswer]) -> CandidateAnswer:
        """Use LLM to select the best candidate."""
        dspy = _get_dspy()

        if not hasattr(dspy.settings, 'lm') or dspy.settings.lm is None:
            return max(candidates, key=lambda c: c.confidence)

        # Format candidates for LLM
        candidate_descriptions = []
        for i, c in enumerate(candidates):
            output_str = str(c.output)[:600] if c.output else "(empty)"
            candidate_descriptions.append(
                f"[{i}] Source: {c.source}, Agent: {c.agent_name}\n"
                f"    Output: {output_str}"
            )

        prompt = (
            f"Goal: {goal}\n\n"
            f"Select the BEST answer (most complete, accurate, and relevant).\n\n"
            f"Candidates:\n{''.join(candidate_descriptions)}\n\n"
            f"Reply with ONLY the number [0-{len(candidates)-1}] of the best candidate."
        )

        lm = dspy.settings.lm
        response = lm(prompt=prompt)
        text = response[0] if isinstance(response, list) else str(response)

        # Parse selection
        import re
        match = re.search(r'\[?(\d+)\]?', text.strip())
        if match:
            idx = int(match.group(1))
            if 0 <= idx < len(candidates):
                logger.info(f"MAS-Verify: selected candidate [{idx}] ({candidates[idx].source})")
                return candidates[idx]

        return max(candidates, key=lambda c: c.confidence)


# =============================================================================
# META-FEEDBACK EVALUATOR (Solvability + Completeness)
# =============================================================================

class MetaFeedbackEvaluator:
    """
    Evaluates MAS design quality at the system level.

    Two criteria from MAS-ZERO:
    - Solvability: Can each sub-task be independently solved?
    - Completeness: Do all sub-tasks cover the full problem?

    KISS: Single LLM call returns both metrics + feedback.
    DRY: Reuses DSPy LM.
    """

    def evaluate(
        self,
        goal: str,
        sub_tasks: List[Dict[str, Any]],
        results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Evaluate MAS design after execution.

        Args:
            goal: Original task goal
            sub_tasks: List of {agent_name, sub_goal, strategy}
            results: Dict of agent_name -> result

        Returns:
            Dict with solvability, completeness, feedback, should_refine
        """
        # Quick check: if all succeeded, likely good enough
        all_success = all(
            r.success if hasattr(r, 'success') else r.get('success', False)
            for r in results.values()
        )

        if all_success and len(results) >= len(sub_tasks):
            return {
                'solvability': 1.0,
                'completeness': 1.0,
                'feedback': '',
                'should_refine': False,
                'too_hard_tasks': [],
            }

        # LLM evaluation for mixed results
        try:
            return self._llm_evaluate(goal, sub_tasks, results)
        except Exception as e:
            logger.debug(f"Meta-feedback LLM evaluation failed: {e}")
            # Heuristic fallback
            success_count = sum(
                1 for r in results.values()
                if (hasattr(r, 'success') and r.success)
                or (isinstance(r, dict) and r.get('success'))
            )
            solvability = success_count / max(len(results), 1)
            return {
                'solvability': solvability,
                'completeness': solvability,  # rough proxy
                'feedback': 'Some sub-tasks failed' if solvability < 1.0 else '',
                'should_refine': solvability < 0.75,  # Refine if >25% of tasks failed
                'too_hard_tasks': [],
            }

    def _llm_evaluate(
        self,
        goal: str,
        sub_tasks: List[Dict[str, Any]],
        results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """LLM-based meta-feedback evaluation."""
        dspy = _get_dspy()

        if not hasattr(dspy.settings, 'lm') or dspy.settings.lm is None:
            raise AgentExecutionError("No LLM configured")

        # Format sub-tasks and results
        task_lines = []
        for st in sub_tasks:
            agent = st.get('agent_name', 'unknown')
            sub_goal = st.get('sub_goal', '')[:100]
            result = results.get(agent)
            if result is None:
                status = "NOT_EXECUTED"
            elif hasattr(result, 'success'):
                status = "OK" if result.success else "FAILED"
            elif isinstance(result, dict):
                status = "OK" if result.get('success') else "FAILED"
            else:
                status = "UNKNOWN"
            task_lines.append(f"- [{agent}] {sub_goal} -> {status}")

        prompt = (
            f"Goal: {goal}\n\n"
            f"Sub-tasks and results:\n{''.join(task_lines)}\n\n"
            f"Evaluate:\n"
            f"1. SOLVABILITY (0.0-1.0): Can each sub-task be independently solved?\n"
            f"2. COMPLETENESS (0.0-1.0): Do sub-tasks collectively cover the full goal?\n"
            f"3. TOO_HARD: List any sub-task agent names that are too hard and need re-decomposition\n"
            f"4. FEEDBACK: Brief actionable feedback for improvement\n\n"
            f"Reply as: solvability=X completeness=Y too_hard=[...] feedback=..."
        )

        lm = dspy.settings.lm
        response = lm(prompt=prompt)
        text = response[0] if isinstance(response, list) else str(response)

        # Parse response
        import re
        solv_match = re.search(r'solvability\s*=\s*([\d.]+)', text, re.IGNORECASE)
        comp_match = re.search(r'completeness\s*=\s*([\d.]+)', text, re.IGNORECASE)
        hard_match = re.search(r'too_hard\s*=\s*\[(.*?)\]', text, re.IGNORECASE)
        fb_match = re.search(r'feedback\s*=\s*(.+)', text, re.IGNORECASE)

        solvability = float(solv_match.group(1)) if solv_match else 0.5
        completeness = float(comp_match.group(1)) if comp_match else 0.5
        too_hard = [t.strip().strip("'\"") for t in hard_match.group(1).split(',') if t.strip()] if hard_match else []
        feedback = fb_match.group(1).strip() if fb_match else text[:200]

        # Clamp values
        solvability = max(0.0, min(1.0, solvability))
        completeness = max(0.0, min(1.0, completeness))

        return {
            'solvability': solvability,
            'completeness': completeness,
            'feedback': feedback,
            'should_refine': solvability < 0.6 or completeness < 0.6,
            'too_hard_tasks': too_hard,
        }


# =============================================================================
# NOTE: BuildingBlocksRunner was removed (KISS principle).
# Jotty's AutoAgent already has its own ensemble pipeline internally,
# so running building blocks in single-agent mode would just duplicate
# LLM calls without adding value. In multi-agent mode, MAS-Verify
# already provides answer selection across agents.
# =============================================================================


# =============================================================================
# DYNAMIC REDUCTION CHECKER
# =============================================================================

def should_reduce_to_single(
    goal: str,
    agents_count: int,
    experience: Optional[ExperienceLibrary] = None,
) -> bool:
    """
    Check if multi-agent should be reduced to single-agent.

    MAS-ZERO insight: simpler systems can outperform complex MAS.

    KISS: Simple heuristics, no LLM call needed.

    Args:
        goal: Task goal
        agents_count: Number of agents currently configured
        experience: Optional experience from previous iterations

    Returns:
        True if should reduce to single agent
    """
    if agents_count <= 1:
        return False

    # Check experience: if previous multi-agent attempts mostly failed, reduce
    if experience and experience.experiences:
        recent = experience.experiences[-4:]
        fail_rate = sum(1 for e in recent if not e.success) / len(recent)
        if fail_rate >= 0.75:
            logger.info("Dynamic reduction: >75% failure rate in recent experience, reducing to single agent")
            return True

    # Check for trivially short goals (likely don't need multi-agent)
    word_count = len(goal.split())
    if word_count <= 3 and agents_count > 2:
        logger.info(f"Dynamic reduction: trivial goal ({word_count} words) with {agents_count} agents")
        return True

    return False


# =============================================================================
# SINGLETON INSTANCES (reusable, no state)
# =============================================================================

_verifier = None
_evaluator = None


def get_candidate_verifier() -> CandidateVerifier:
    """Get shared CandidateVerifier instance."""
    global _verifier
    if _verifier is None:
        _verifier = CandidateVerifier()
    return _verifier


def get_meta_feedback_evaluator() -> MetaFeedbackEvaluator:
    """Get shared MetaFeedbackEvaluator instance."""
    global _evaluator
    if _evaluator is None:
        _evaluator = MetaFeedbackEvaluator()
    return _evaluator
