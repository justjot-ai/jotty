"""TaskPlanner Inference Mixin - Task type and capability inference."""

import json
import logging
from enum import IntEnum
from typing import Dict, Any, List, Optional, Tuple

from ..types.execution_types import TaskType

logger = logging.getLogger(__name__)


class ContextPriority(IntEnum):
    """Priority levels for context sections during compression.

    Higher priority sections are preserved; lower ones are compressed first.
    """
    CRITICAL = 4   # Current instruction, tool schemas — never compress
    HIGH = 3       # Recent tool results, last 2-3 trajectory steps
    MEDIUM = 2     # Older conversation history, memory context
    LOW = 1        # System boilerplate, examples, verbose descriptions


class InferenceMixin:
    # Per-session cache: avoids redundant LLM calls for same task string.
    # Key = first 200 chars of task (stable identifier), Value = (TaskType, reasoning, confidence)
    _task_type_cache: dict = {}

    def infer_task_type(self, task: str) -> Tuple:
        """
        Infer task type using LLM semantic understanding.

        Cached: repeated calls for the same task return instantly (saves ~3-5s per call).

        Args:
            task: Task description

        Returns:
            (TaskType, reasoning, confidence)
        """
        # TaskType imported at module level from _execution_types

        # Check cache first — task type for same text doesn't change
        cache_key = task[:200]
        if cache_key in InferenceMixin._task_type_cache:
            cached = InferenceMixin._task_type_cache[cache_key]
            logger.info(f" Task type inferred: {cached[0].value} (confidence: {cached[2]:.2f}) [cached]")
            return cached

        try:
            import dspy
            import asyncio
            import re

            # Prepare task for inference (preserves full context)
            task_for_inference = self._abstract_task_for_planning(task)

            # Use fast LM (Haiku) for classification - much faster than Sonnet
            classification_lm = self._fast_lm or dspy.settings.lm

            # Call with fast LM for quick classification
            if classification_lm:
                with dspy.context(lm=classification_lm):
                    result = self.task_type_inferrer(task_description=task_for_inference)
                logger.debug(f"Task type inference using fast model: {self._fast_model}")
            else:
                # Fallback to default LM
                result = self.task_type_inferrer(task_description=task_for_inference)

            # Parse task_type field
            task_type_str = str(result.task_type).lower().strip().split()[0] if result.task_type else 'unknown'

            task_type_map = {
                'research': TaskType.RESEARCH,
                'comparison': TaskType.COMPARISON,
                'creation': TaskType.CREATION,
                'communication': TaskType.COMMUNICATION,
                'analysis': TaskType.ANALYSIS,
                'automation': TaskType.AUTOMATION,
            }
            task_type = task_type_map.get(task_type_str, TaskType.UNKNOWN)

            # Override if LLM returned unknown but keywords clearly indicate creation
            if task_type == TaskType.UNKNOWN:
                task_lower = task.lower()
                if any(w in task_lower for w in ['create', 'build', 'make', 'write', 'generate', 'implement']):
                    task_type = TaskType.CREATION
                    logger.info(f"Overriding 'unknown' to 'creation' based on keywords")

            # Parse confidence
            try:
                confidence_match = re.search(r'(\d+\.?\d*)', str(result.confidence))
                confidence = float(confidence_match.group(1)) if confidence_match else 0.7
                confidence = max(0.0, min(1.0, confidence))
            except (ValueError, TypeError, AttributeError):
                confidence = 0.7

            reasoning = str(result.reasoning).strip() if result.reasoning else f"Inferred as {task_type_str}"

            logger.info(f" Task type inferred: {task_type.value} (confidence: {confidence:.2f})")
            result_tuple = (task_type, reasoning, confidence)
            InferenceMixin._task_type_cache[cache_key] = result_tuple
            return result_tuple

        except Exception as e:
            error_str = str(e)
            logger.warning(f"Task type inference failed: {e}, using keyword fallback")

            # Detect if LLM asked for clarification instead of classifying
            clarification_markers = [
                'could you', 'please provide', 'what', 'which', 'can you',
                'i need', 'clarify', 'more information', 'specify'
            ]
            if any(marker in error_str.lower() for marker in clarification_markers):
                logger.info("LLM asked for clarification - defaulting to analysis")
                return TaskType.ANALYSIS, "LLM requested clarification - default to analysis", 0.5

            # Enhanced keyword fallback with more coverage
            task_lower = task.lower()

            # Creation keywords
            if any(w in task_lower for w in ['create', 'generate', 'make', 'build', 'write', 'implement', 'develop', 'code']):
                return TaskType.CREATION, "Keyword fallback: creation task", 0.6
            # Comparison keywords
            elif any(w in task_lower for w in ['compare', 'vs', 'versus', 'comparison', 'difference', 'better']):
                return TaskType.COMPARISON, "Keyword fallback: comparison task", 0.6
            # Research keywords
            elif any(w in task_lower for w in ['research', 'find', 'search', 'discover', 'look up', 'what is']):
                return TaskType.RESEARCH, "Keyword fallback: research task", 0.6
            # Analysis keywords (including calculation)
            elif any(w in task_lower for w in ['analyze', 'analysis', 'evaluate', 'calculate', 'compute', 'answer', 'solve', 'sum', 'result']):
                return TaskType.ANALYSIS, "Keyword fallback: analysis task", 0.6
            # Automation keywords
            elif any(w in task_lower for w in ['automate', 'schedule', 'pipeline', 'workflow', 'cron']):
                return TaskType.AUTOMATION, "Keyword fallback: automation task", 0.6
            # Communication keywords
            elif any(w in task_lower for w in ['send', 'email', 'notify', 'message', 'communicate']):
                return TaskType.COMMUNICATION, "Keyword fallback: communication task", 0.6

            # Default to ANALYSIS for any ambiguous task (not UNKNOWN)
            # This ensures we always try to do something useful
            fallback = (TaskType.ANALYSIS, "Ambiguous task - defaulting to analysis", 0.4)
            InferenceMixin._task_type_cache[cache_key] = fallback
            return fallback

    def infer_capabilities(self, task: str) -> tuple[List[str], str]:
        """
        Infer required capabilities from task description.

        Uses fast LLM to determine what capabilities (data-fetch, communicate,
        visualize, etc.) are needed to complete the task.

        Args:
            task: Task description

        Returns:
            (capabilities, reasoning) - List of capability strings and explanation
        """
        # Default capabilities based on common patterns
        default_capabilities = ["analyze"]

        try:
            import dspy

            # Clean task for inference
            task_for_inference = self._abstract_task_for_planning(task)

            # Use fast LM for quick classification
            result = self._call_with_retry(
                module=self.capability_inferrer,
                kwargs={'task_description': task_for_inference},
                max_retries=2,
                lm=self._fast_lm
            )

            # Parse capabilities from result
            capabilities_str = str(result.capabilities).strip()

            # Try to parse as JSON
            try:
                if capabilities_str.startswith('['):
                    capabilities = json.loads(capabilities_str)
                else:
                    # Extract from text like "data-fetch, communicate"
                    capabilities = [c.strip().lower() for c in capabilities_str.replace('"', '').replace('[', '').replace(']', '').split(',')]
            except json.JSONDecodeError:
                # Fallback: extract known capabilities from text
                known_caps = ['data-fetch', 'research', 'analyze', 'visualize', 'document', 'communicate', 'file-ops', 'code', 'media']
                capabilities = [c for c in known_caps if c in capabilities_str.lower()]

            # Validate and clean
            capabilities = [c.strip().lower() for c in capabilities if c.strip()][:4]

            if not capabilities:
                capabilities = default_capabilities

            reasoning = str(result.reasoning).strip() if result.reasoning else "Inferred from task"

            logger.info(f" Capabilities inferred: {capabilities}")
            return capabilities, reasoning

        except Exception as e:
            logger.warning(f"Capability inference failed: {e}, using keyword fallback")

            # Keyword-based fallback
            task_lower = task.lower()
            capabilities = []

            if any(w in task_lower for w in ['weather', 'stock', 'price', 'data', 'fetch', 'get']):
                capabilities.append('data-fetch')
            if any(w in task_lower for w in ['research', 'search', 'find', 'look up']):
                capabilities.append('research')
            if any(w in task_lower for w in ['chart', 'graph', 'slide', 'visual', 'diagram']):
                capabilities.append('visualize')
            if any(w in task_lower for w in ['pdf', 'report', 'document']):
                capabilities.append('document')
            if any(w in task_lower for w in ['telegram', 'slack', 'email', 'send', 'notify']):
                capabilities.append('communicate')
            if any(w in task_lower for w in ['file', 'save', 'write', 'read']):
                capabilities.append('file-ops')
            if any(w in task_lower for w in ['analyze', 'calculate', 'compute']):
                capabilities.append('analyze')

            if not capabilities:
                capabilities = default_capabilities

            return capabilities, "Keyword-based fallback"

    # Skills that depend on external services that may be unreliable
    # These are deprioritized in skill selection to prefer more reliable alternatives
    DEPRIORITIZED_SKILLS = {
        'search-to-justjot-idea',      # JustJot API issues
        'notion-research-documentation',  # Requires Notion API setup
        'reddit-trending-to-justjot',  # JustJot API issues
        'notebooklm-pdf',              # Requires browser sign-in
        'oauth-automation',            # Requires browser interaction
    }

    # =========================================================================
    # PROACTIVE CONTEXT GUARD
    # =========================================================================

    # Rough chars-per-token for estimation (conservative for Claude/GPT)
    _CHARS_PER_TOKEN = 3.5
    # Default context budget (tokens). Override via model-specific values.
    _DEFAULT_CONTEXT_BUDGET = 100_000

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Estimate token count from text length. ~3.5 chars/token for English."""
        return int(len(text) / InferenceMixin._CHARS_PER_TOKEN)

    def _proactive_context_guard(
        self,
        sections: List[tuple],
        budget_tokens: int = 0,
        reserve_output_tokens: int = 4000,
    ) -> str:
        """Assemble context from prioritized sections, compressing to fit budget.

        Builds context by including sections in priority order (CRITICAL first).
        If total exceeds budget, LOW sections are truncated, then MEDIUM, etc.
        CRITICAL sections are never compressed.

        Args:
            sections: List of (ContextPriority, label, text) tuples
            budget_tokens: Max input tokens. 0 = use _DEFAULT_CONTEXT_BUDGET
            reserve_output_tokens: Tokens reserved for model output

        Returns:
            Assembled context string fitting within budget
        """
        budget = budget_tokens or self._DEFAULT_CONTEXT_BUDGET
        available = budget - reserve_output_tokens

        # Sort by priority descending (CRITICAL first)
        sorted_sections = sorted(sections, key=lambda s: s[0], reverse=True)

        # First pass: compute total
        total_tokens = sum(self._estimate_tokens(s[2]) for s in sorted_sections)

        if total_tokens <= available:
            # Everything fits — no compression needed
            return "\n\n".join(s[2] for s in sorted_sections if s[2])

        # Second pass: compress from lowest priority up
        result_sections = []
        used_tokens = 0

        for priority, label, text in sorted_sections:
            section_tokens = self._estimate_tokens(text)

            if priority >= ContextPriority.CRITICAL:
                # Never compress critical sections
                result_sections.append(text)
                used_tokens += section_tokens
            elif used_tokens + section_tokens <= available:
                # Fits within remaining budget
                result_sections.append(text)
                used_tokens += section_tokens
            else:
                # Must truncate
                remaining_budget = max(0, available - used_tokens)
                if remaining_budget > 100:  # Only include if meaningful space left
                    max_chars = int(remaining_budget * self._CHARS_PER_TOKEN)
                    truncated = text[:max_chars] + f"\n[...{label} truncated...]"
                    result_sections.append(truncated)
                    used_tokens += remaining_budget
                else:
                    result_sections.append(f"[{label} omitted — context budget]")

        logger.debug(
            f"Context guard: {total_tokens} tokens → ~{used_tokens} tokens "
            f"(budget: {available})"
        )
        return "\n\n".join(s for s in result_sections if s)

    # =========================================================================
    # CONTEXT COMPRESSION RETRY
    # =========================================================================

    _CONTEXT_OVERFLOW_PATTERNS = (
        'context_length_exceeded', 'max_tokens', 'context window',
        'maximum context', 'token limit', 'too many tokens',
        'input too long', 'prompt is too long',
    )

    async def _call_with_compression_retry(self, call_fn: Any, conversation: str, instruction: str, max_retries: int = 3) -> Any:
        """Wrap an LLM call with automatic context compression on overflow.

        On context_length_exceeded errors:
        1. Compress conversation by ratio (0.7 → 0.49 → 0.34)
        2. Preserve recent trajectory (last 3 tool calls)
        3. Retry the call with compressed context

        Args:
            call_fn: Callable(conversation, instruction) → result
            conversation: Full conversation/context string
            instruction: Task instruction (never compressed)
            max_retries: Maximum compression retries

        Returns:
            Result from call_fn on success

        Raises:
            Last exception if all retries exhausted
        """
        import asyncio as _aio

        compression_ratio = 0.7
        current_conversation = conversation
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                if _aio.iscoroutinefunction(call_fn):
                    return await call_fn(current_conversation, instruction)
                else:
                    loop = _aio.get_running_loop()
                    return await loop.run_in_executor(
                        None, call_fn, current_conversation, instruction)

            except Exception as e:
                error_str = str(e).lower()
                last_error = e

                # Only retry on context overflow errors
                is_overflow = any(
                    pat in error_str for pat in self._CONTEXT_OVERFLOW_PATTERNS)
                if not is_overflow or attempt >= max_retries:
                    raise

                # Compress: keep the most recent portion
                target_len = int(len(current_conversation) * compression_ratio)
                current_conversation = self._compress_context(
                    current_conversation, target_len)
                compression_ratio *= 0.7  # Progressive: 0.7 → 0.49 → 0.34

                logger.warning(
                    f"Context overflow (attempt {attempt + 1}/{max_retries}): "
                    f"compressed to {len(current_conversation)} chars "
                    f"(ratio={compression_ratio / 0.7:.2f})")

        raise last_error  # Should not reach here, but safety net

    @staticmethod
    def _compress_context(text: str, target_length: int) -> str:
        """Compress context to target length, preserving recent content.

        Strategy: keep the first 20% (system prompt / task description) and
        the last portion up to budget. Inserts a marker at the splice point.
        """
        if len(text) <= target_length:
            return text

        # Reserve 20% for header, rest for recent content
        header_budget = int(target_length * 0.2)
        tail_budget = target_length - header_budget - 50  # 50 for marker

        header = text[:header_budget]
        tail = text[-tail_budget:] if tail_budget > 0 else ""
        marker = "\n\n[... context compressed ...]\n\n"

        return header + marker + tail

