"""
TrajectoryParser - Generic Trajectory Parsing & Tagging for ReVal

Parses DSPy ReAct trajectories and tags attempts as:
- 'answer': Successful attempt with valid result
- 'error': Failed attempt or invalid result
- 'exploratory': Uncertain outcome, learning phase

 GENERIC: Works for ANY domain (SQL, code, marketing, finance, etc.)
 NO HARDCODING: Zero domain-specific assumptions
 LLM-READY: Supports semantic tagging via LLM

A-Team Design: Turing (architecture) + Sutton (RL tagging) + Chomsky (semantic understanding)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TaggedAttempt:
    """
    Single attempt from ReAct exploration with standardized tag.

    Tags (standardized across ALL domains):
    - 'answer': Successful attempt with valid result
    - 'error': Failed attempt or invalid result
    - 'exploratory': Uncertain outcome, learning phase

     GENERIC: No domain-specific fields
     STANDARD: Same structure for SQL, code, marketing, etc.
    """

    output: Any  # The actual output (query, code, config, etc.)
    tag: str  # 'answer', 'error', or 'exploratory'
    execution_status: str  # 'success', 'failed', 'uncertain'
    execution_result: str  # Full observation from tool
    reasoning: str  # Thought/reasoning for this attempt
    attempt_number: int = 0
    tool_name: str = ""  # Which tool was called
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())

    def is_answer(self) -> bool:
        """Check if this is a successful answer."""
        return self.tag == "answer"

    def is_error(self) -> bool:
        """Check if this is a failed attempt."""
        return self.tag == "error"

    def is_exploratory(self) -> bool:
        """Check if this is exploratory (uncertain)."""
        return self.tag == "exploratory"


class TrajectoryParser:
    """
    Generic trajectory parser for ANY DSPy ReAct agent.

    Extracts tool calls from trajectory and tags them as:
    - 'answer': Successful execution with valid result
    - 'error': Failed execution or invalid result
    - 'exploratory': Uncertain outcome, learning attempt

     GENERIC: Works for SQL, code, marketing, finance, etc.
     NO HARDCODING: No domain-specific assumptions
     LLM-BASED: Semantic understanding of success/failure

    A-Team Consensus: This is the ONLY place where trajectory parsing happens.
    Agents should NOT parse their own trajectories.
    """

    def __init__(self, lm: Optional[Any] = None) -> None:
        """
        Initialize parser.

        Args:
            lm: Optional LLM for semantic tagging (future enhancement)
        """
        self.lm = lm
        logger.info(" TrajectoryParser initialized (generic, domain-agnostic)")

    def parse_trajectory(
        self,
        result: Any,
        tool_name_filter: Optional[str] = None,
        expected_outcome: Optional[str] = None,
    ) -> List[TaggedAttempt]:
        """
        Parse DSPy ReAct trajectory and tag attempts.

        Args:
            result: DSPy Prediction with _store containing trajectory
            tool_name_filter: Only parse calls to this tool (e.g. 'execute_query')
            expected_outcome: What we expected (for semantic tagging)

        Returns:
            List of TaggedAttempt with generic tags
        """
        attempts = []

        # Check if result has trajectory data
        if not hasattr(result, "_store"):
            logger.debug(" No trajectory (_store) in result")
            return attempts

        store = result._store

        # CRITICAL FIX: DSPy ReAct stores trajectory as a DICT in _store['trajectory']!
        # NOT as top-level keys in _store!
        # See: https://dspy.ai/api/modules/ReAct/
        if "trajectory" in store and isinstance(store["trajectory"], dict):
            trajectory = store["trajectory"]
            logger.info(f" Parsing trajectory dict with {len(trajectory)} keys")
        else:
            # Fallback: check if trajectory keys are at top level
            trajectory = store
            logger.info(f" Parsing trajectory (top-level) with {len(store)} keys")

        i = 0

        # Parse all tool calls in trajectory
        while True:
            # DSPy ReAct trajectory format:
            # thought_0, tool_name_0, tool_args_0, observation_0
            # thought_1, tool_name_1, tool_args_1, observation_1
            tool_name_key = f"tool_name_{i}"

            if tool_name_key not in trajectory:
                break  # No more tool calls

            tool_name = trajectory.get(tool_name_key, "")
            tool_args = trajectory.get(f"tool_args_{i}", {})
            observation = trajectory.get(f"observation_{i}", "")
            thought = trajectory.get(f"thought_{i}", "")

            # Filter by tool if specified
            if tool_name_filter and tool_name != tool_name_filter:
                i += 1
                continue

            # Extract output from tool args (GENERIC)
            output = self._extract_output(tool_args)

            # Tag the attempt (GENERIC - no domain logic)
            tag = self._tag_attempt(observation, expected_outcome)

            # Create tagged attempt
            attempt = TaggedAttempt(
                output=output,
                tag=tag,
                execution_status=self._get_status(tag),
                execution_result=str(observation),
                reasoning=str(thought) if thought else f"Trajectory step {i}",
                attempt_number=len(attempts) + 1,
                tool_name=tool_name,
            )

            attempts.append(attempt)
            logger.info(f" Attempt {attempt.attempt_number}: tag='{tag}', tool='{tool_name}'")

            i += 1

        logger.info(f" Parsed {len(attempts)} attempts from trajectory")
        return attempts

    def _tag_attempt(self, observation: str, expected_outcome: Optional[str] = None) -> str:
        """
        Tag attempt as 'answer', 'error', or 'exploratory'.

        GENERIC logic - no domain assumptions!

        Rules:
        1. Explicit errors → 'error'
        2. Success indicators → 'answer'
        3. Empty/null (uncertain) → 'exploratory'
        4. Default → 'exploratory'
        """
        obs_str = str(observation).lower()

        # 1. Check for explicit errors (GENERIC keywords)
        error_indicators = [
            "error",
            "exception",
            "failed",
            "failure",
            "invalid",
            "traceback",
            "runtimeerror",
            "syntaxerror",
            "valueerror",
            "typeerror",
            "keyerror",
            "attributeerror",
            "indexerror",
        ]
        if any(indicator in obs_str for indicator in error_indicators):
            return "error"

        # 2. Check for success indicators (GENERIC keywords)
        success_indicators = [
            "success",
            "complete",
            "done",
            "finished",
            "result",
            "output",
            "returned",
            "=",
        ]
        if any(indicator in obs_str for indicator in success_indicators):
            # Double-check it's not an empty result disguised as success
            null_indicators = ["none", "null", "empty", "no result", "[]", "{}"]
            if not any(null in obs_str for null in null_indicators):
                return "answer"

        # 3. Default: exploratory (uncertain outcome)
        return "exploratory"

    def _extract_output(self, tool_args: Any) -> Any:
        """
        Extract output from tool arguments (GENERIC).

        Tries common patterns across different domains.
        """
        if isinstance(tool_args, dict):
            # Try common field names across domains
            return (
                tool_args.get("query")
                or tool_args.get("code")  # SQL / database queries
                or tool_args.get("prompt")  # Code generation
                or tool_args.get("input")  # Marketing / content
                or tool_args.get("data")  # Generic input
                or tool_args.get("text")  # Data processing
                or tool_args.get("command")  # Text generation
                or str(tool_args)  # Shell commands  # Fallback: full dict
            )
        return str(tool_args)

    def _get_status(self, tag: str) -> str:
        """Map tag to execution status (standardized)."""
        return {"answer": "success", "error": "failed", "exploratory": "uncertain"}.get(
            tag, "unknown"
        )


def create_parser(lm: Optional[Any] = None) -> TrajectoryParser:
    """Factory function to create parser."""
    return TrajectoryParser(lm=lm)
