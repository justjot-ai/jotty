"""
Context Utilities for Jotty Agents
==================================

Provides context compression, error detection, and retry utilities
learned from BaseSwarmAgent patterns.

Features:
- Automatic context compression when LLM calls fail due to length
- Smart error type detection (context length, timeout, parse errors)
- Trajectory preservation during retries
"""

import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Categorized error types for different handling strategies."""
    CONTEXT_LENGTH = "context_length"
    TIMEOUT = "timeout"
    PARSE_ERROR = "parse_error"
    RATE_LIMIT = "rate_limit"
    NETWORK = "network"
    TOOL_ERROR = "tool_error"
    UNKNOWN = "unknown"


@dataclass
class CompressionResult:
    """Result of context compression."""
    original_length: int
    compressed_length: int
    compression_ratio: float
    content: str
    preserved_trajectory: str = ""


class ContextCompressor:
    """
    Intelligent context compression for LLM calls.

    Strategies:
    1. Preserve recent trajectory (current work)
    2. Compress old conversation history
    3. Keep critical keywords/entities
    """

    def __init__(self, max_compression_retries: int = 3):
        self.max_compression_retries = max_compression_retries
        self._compression_ratio = 0.7  # Start at 70%

    def compress(
        self,
        content: str,
        target_ratio: float = 0.5,
        preserve_keywords: Optional[List[str]] = None,
        trajectory: str = ""
    ) -> CompressionResult:
        """
        Compress content while preserving important information.

        Args:
            content: The content to compress
            target_ratio: Target compression ratio (0.5 = keep 50%)
            preserve_keywords: Keywords to ensure are kept
            trajectory: Recent work to preserve completely

        Returns:
            CompressionResult with compressed content
        """
        if not content or not content.strip():
            return CompressionResult(
                original_length=0,
                compressed_length=0,
                compression_ratio=1.0,
                content="",
                preserved_trajectory=trajectory
            )

        original_length = len(content)
        target_length = int(original_length * target_ratio)

        # Strategy 1: Split into sections and keep most recent
        sections = self._split_into_sections(content)

        # Strategy 2: Score sections by relevance
        if preserve_keywords:
            sections = self._score_and_sort_sections(sections, preserve_keywords)

        # Strategy 3: Build compressed content from most important sections
        compressed = self._build_compressed_content(sections, target_length)

        # Add compression marker
        if len(compressed) < original_length:
            compressed = "...[earlier context compressed]...\n\n" + compressed

        return CompressionResult(
            original_length=original_length,
            compressed_length=len(compressed),
            compression_ratio=len(compressed) / original_length if original_length > 0 else 1.0,
            content=compressed,
            preserved_trajectory=trajectory
        )

    def _split_into_sections(self, content: str) -> List[Dict[str, Any]]:
        """Split content into logical sections."""
        sections = []

        # Split by double newlines (paragraphs)
        paragraphs = content.split('\n\n')

        for i, para in enumerate(paragraphs):
            if para.strip():
                sections.append({
                    'text': para.strip(),
                    'position': i,
                    'length': len(para),
                    'score': 0.0,
                    'is_recent': i >= len(paragraphs) - 3  # Last 3 paragraphs are recent
                })

        return sections

    def _score_and_sort_sections(
        self,
        sections: List[Dict[str, Any]],
        keywords: List[str]
    ) -> List[Dict[str, Any]]:
        """Score sections by keyword relevance and recency."""
        for section in sections:
            text_lower = section['text'].lower()

            # Keyword matching score
            keyword_score = sum(1 for kw in keywords if kw.lower() in text_lower)

            # Recency bonus
            recency_score = 2.0 if section['is_recent'] else 0.0

            # Combine scores
            section['score'] = keyword_score + recency_score

        # Sort by score (highest first), then by position (most recent first)
        sections.sort(key=lambda s: (-s['score'], -s['position']))

        return sections

    def _build_compressed_content(
        self,
        sections: List[Dict[str, Any]],
        target_length: int
    ) -> str:
        """Build compressed content from scored sections."""
        selected = []
        current_length = 0

        # Always include recent sections first
        recent_sections = [s for s in sections if s['is_recent']]
        other_sections = [s for s in sections if not s['is_recent']]

        # Add recent sections
        for section in recent_sections:
            if current_length + section['length'] <= target_length:
                selected.append(section)
                current_length += section['length']

        # Add other high-scoring sections
        for section in other_sections:
            if current_length + section['length'] <= target_length:
                selected.append(section)
                current_length += section['length']

        # Sort by original position for coherent output
        selected.sort(key=lambda s: s['position'])

        return '\n\n'.join(s['text'] for s in selected)


class ErrorDetector:
    """
    Detect and categorize errors for appropriate handling.

    Different error types require different strategies:
    - Context length: Compress and retry
    - Timeout: Exponential backoff retry
    - Parse error: Simplify prompt and retry
    - Rate limit: Wait and retry
    """

    # Error patterns by category
    CONTEXT_LENGTH_PATTERNS = [
        "input is too long",
        "context length exceeded",
        "maximum context length",
        "token limit exceeded",
        "too many tokens",
        "context window exceeded",
        "prompt is too long",
        "request too large",
    ]

    TIMEOUT_PATTERNS = [
        "timeout",
        "timed out",
        "request timeout",
        "connection timeout",
        "read timeout",
    ]

    PARSE_PATTERNS = [
        "failed to parse",
        "json decode error",
        "invalid json",
        "parse error",
        "adapter parse error",
        "validation error",
    ]

    RATE_LIMIT_PATTERNS = [
        "rate limit",
        "too many requests",
        "quota exceeded",
        "throttled",
        "429",
    ]

    NETWORK_PATTERNS = [
        "network error",
        "connection error",
        "connection refused",
        "dns",
        "socket",
        "ssl",
        "certificate",
    ]

    @classmethod
    def detect(cls, error: Exception) -> ErrorType:
        """
        Detect the type of error.

        Args:
            error: The exception to categorize

        Returns:
            ErrorType enum value
        """
        error_str = str(error).lower()
        error_type_name = type(error).__name__.lower()

        # Check each category
        if cls._matches_patterns(error_str, cls.CONTEXT_LENGTH_PATTERNS):
            return ErrorType.CONTEXT_LENGTH

        if cls._matches_patterns(error_str, cls.TIMEOUT_PATTERNS) or "timeout" in error_type_name:
            return ErrorType.TIMEOUT

        if cls._matches_patterns(error_str, cls.PARSE_PATTERNS) or "parse" in error_type_name:
            return ErrorType.PARSE_ERROR

        if cls._matches_patterns(error_str, cls.RATE_LIMIT_PATTERNS):
            return ErrorType.RATE_LIMIT

        if cls._matches_patterns(error_str, cls.NETWORK_PATTERNS):
            return ErrorType.NETWORK

        return ErrorType.UNKNOWN

    @classmethod
    def _matches_patterns(cls, text: str, patterns: List[str]) -> bool:
        """Check if text matches any pattern."""
        return any(pattern in text for pattern in patterns)

    @classmethod
    def get_retry_strategy(cls, error_type: ErrorType) -> Dict[str, Any]:
        """
        Get recommended retry strategy for error type.

        Returns:
            Dict with retry configuration
        """
        strategies = {
            ErrorType.CONTEXT_LENGTH: {
                'should_retry': True,
                'action': 'compress',
                'max_retries': 3,
                'delay_seconds': 0,
            },
            ErrorType.TIMEOUT: {
                'should_retry': True,
                'action': 'backoff',
                'max_retries': 3,
                'delay_seconds': 2,  # Will be multiplied by attempt
            },
            ErrorType.PARSE_ERROR: {
                'should_retry': True,
                'action': 'simplify',
                'max_retries': 2,
                'delay_seconds': 0,
            },
            ErrorType.RATE_LIMIT: {
                'should_retry': True,
                'action': 'wait',
                'max_retries': 3,
                'delay_seconds': 30,
            },
            ErrorType.NETWORK: {
                'should_retry': True,
                'action': 'backoff',
                'max_retries': 3,
                'delay_seconds': 1,
            },
            ErrorType.UNKNOWN: {
                'should_retry': False,
                'action': 'fail',
                'max_retries': 0,
                'delay_seconds': 0,
            },
        }
        return strategies.get(error_type, strategies[ErrorType.UNKNOWN])


@dataclass
class ExecutionTrajectory:
    """
    Captures the trajectory of an execution for preservation during retries.

    When an LLM call fails mid-execution (e.g., context too long),
    we want to preserve the work done so far rather than starting over.
    """
    steps_completed: List[Dict[str, Any]]
    outputs_collected: Dict[str, Any]
    current_step_index: int
    partial_result: Optional[str] = None
    reasoning_so_far: str = ""

    def to_context(self) -> str:
        """Convert trajectory to context string for LLM."""
        if not self.steps_completed:
            return ""

        parts = ["[Progress so far]"]

        for i, step in enumerate(self.steps_completed):
            step_desc = step.get('description', f'Step {i+1}')
            step_output = step.get('output', '')
            if step_output:
                output_preview = str(step_output)[:200]
                parts.append(f"Step {i+1}: {step_desc}\nResult: {output_preview}")
            else:
                parts.append(f"Step {i+1}: {step_desc} (completed)")

        if self.reasoning_so_far:
            parts.append(f"\nReasoning: {self.reasoning_so_far}")

        return "\n".join(parts)

    def add_step(self, step: Dict[str, Any], output: Any = None):
        """Add a completed step to trajectory."""
        step_record = {
            'description': step.get('description', ''),
            'skill': step.get('skill_name', ''),
            'tool': step.get('tool_name', ''),
            'output': output,
        }
        self.steps_completed.append(step_record)
        self.current_step_index += 1

        if output:
            key = step.get('output_key', f'step_{self.current_step_index}')
            self.outputs_collected[key] = output


# ---------------------------------------------------------------------------
# Enrichment-context stripping — single source of truth for markers that
# separate the original user task from injected learning/ensemble context.
#
# WRITE SIDE (inject context):
#   - Orchestrator: enriches goal with ensemble synthesis
#   - AgentRunner: appends Q-learning / transfer-learning / swarm-intelligence
#   - AutoAgent: appends ensemble synthesis
#
# READ SIDE (strip context):
#   - _plan_utils_mixin._clean_task_for_query()
#   - swarm_lean.clean_task_for_execution()
#   - auto_agent._clean_for_display()
#
# All three consumers now call strip_enrichment_context() below.
# ---------------------------------------------------------------------------

ENRICHMENT_MARKERS: tuple[str, ...] = (
    # Ensemble / multi-perspective
    '\n[Multi-Perspective Analysis',
    '\n[Multi-Perspective',
    # Learning context (Q-learning, transfer learning)
    '\nLearned Insights:',
    '\n# Transferable Learnings',
    '\n# Q-Learning Lessons',
    # Agent runner injections
    '\n[Learning Context',
    '\n[Judge feedback',
    '\n[Skill guidance',
    # Swarm intelligence hints
    '\n[Learned]',
    '\n[Analysis]:',
    '\n[Consensus]:',
    '\n[Tensions]:',
    '\n[Blind Spots]:',
    # Meta-learning sections
    '\n## Task Type Pattern',
    '\n## Role Advice',
    '\n## Meta-Learning Advice',
    # Common separators before context
    '\nBased on previous learnings:',
    '\nRecommended approach:',
    '\nPrevious success patterns:',
    '\nRelevant past experience:',
    '\n\n---\n',
    # Paradigm executor injections (relay, refinement, debate)
    "\n[Previous agent '",
    '\n\nHere is the current draft. Improve it:\n',
    '\nOther agents produced these solutions.',
)


def strip_enrichment_context(task: str) -> str:
    """
    Strip injected enrichment context from a task string.

    Returns only the original user request, discarding any learning context,
    ensemble synthesis, transfer learnings, etc. that were appended.

    This is the **single source of truth** for context stripping — replaces
    three duplicate implementations across the codebase.

    Args:
        task: Task string that may contain enrichment context.

    Returns:
        Clean task string (original request only).
    """
    if not task:
        return task

    cleaned = task
    for marker in ENRICHMENT_MARKERS:
        if marker in cleaned:
            cleaned = cleaned.split(marker)[0]

    # Also handle double-newline variants of the markers
    for marker in ENRICHMENT_MARKERS:
        double_nl = '\n' + marker.lstrip('\n')
        if double_nl in cleaned:
            cleaned = cleaned.split(double_nl)[0]

    return cleaned.strip()


def create_compressor() -> ContextCompressor:
    """Factory function to create a context compressor."""
    return ContextCompressor()


def detect_error_type(error: Exception) -> Tuple[ErrorType, Dict[str, Any]]:
    """
    Convenience function to detect error type and get retry strategy.

    Returns:
        Tuple of (ErrorType, retry_strategy_dict)
    """
    error_type = ErrorDetector.detect(error)
    strategy = ErrorDetector.get_retry_strategy(error_type)
    return error_type, strategy
