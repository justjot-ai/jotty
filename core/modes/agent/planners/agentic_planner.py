"""
Agentic Planner - Fully LLM-based planning (no hardcoded logic)

Replaces all rule-based planning with agentic LLM decisions.
No keyword matching, no hardcoded flows, fully adaptive.

Supports both:
- Raw string tasks (simple planning)
- TaskGraph tasks (structured planning with metadata)
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from Jotty.core.infrastructure.foundation.exceptions import AgentExecutionError

# DSPy loaded lazily to avoid ~6s import at module level
DSPY_AVAILABLE = True  # Assumed; checked on first use
_dspy_module = None


def _get_dspy() -> Any:
    """Lazy-load DSPy on first use."""
    global _dspy_module, DSPY_AVAILABLE
    if _dspy_module is None:
        try:
            import dspy as _dspy

            _dspy_module = _dspy
        except ImportError:
            DSPY_AVAILABLE = False
            _dspy_module = False
    return _dspy_module if _dspy_module else None


# ExecutionStepSchema imported from _execution_types at top of file

# Import context utilities for error handling and compression
try:
    from ..utils.context_utils import (
        ContextCompressor,
        ErrorDetector,
        ErrorType,
        ExecutionTrajectory,
        detect_error_type,
    )

    CONTEXT_UTILS_AVAILABLE = True
except ImportError:
    CONTEXT_UTILS_AVAILABLE = False
    ContextCompressor = None
    ErrorDetector = None
    ErrorType = None

# Shared types — no circular dependency (lives in _execution_types.py)

logger = logging.getLogger(__name__)

# Import TaskGraph if available (for enhanced planning)
try:
    from ..autonomous.intent_parser import TaskGraph

    TASK_GRAPH_AVAILABLE = True
except ImportError:
    TASK_GRAPH_AVAILABLE = False
    TaskGraph = None


# =============================================================================
# DSPy Signatures (extracted to planner_signatures.py)
# =============================================================================

# Signatures imported lazily to avoid DSPy import at module level
_signatures_loaded = False
TaskTypeInferenceSignature = None
CapabilityInferenceSignature = None
ExecutionPlanningSignature = None
SkillSelectionSignature = None
ReflectivePlanningSignature = None


def _load_signatures() -> None:
    """Load DSPy signatures on first use."""
    global _signatures_loaded, TaskTypeInferenceSignature, CapabilityInferenceSignature
    global ExecutionPlanningSignature, SkillSelectionSignature, ReflectivePlanningSignature
    if _signatures_loaded:
        return
    _signatures_loaded = True
    from .planner_signatures import CapabilityInferenceSignature as _CI
    from .planner_signatures import ExecutionPlanningSignature as _EP
    from .planner_signatures import ReflectivePlanningSignature as _RP
    from .planner_signatures import SkillSelectionSignature as _SS
    from .planner_signatures import TaskTypeInferenceSignature as _TT

    TaskTypeInferenceSignature = _TT
    CapabilityInferenceSignature = _CI
    ExecutionPlanningSignature = _EP
    SkillSelectionSignature = _SS
    ReflectivePlanningSignature = _RP


# =============================================================================
# Agentic Planner
# =============================================================================


from ..mixins.inference import InferenceMixin
from ..mixins.plan_utils import PlanUtilsMixin
from ..mixins.skill_selection import SkillSelectionMixin


class TaskPlanner(InferenceMixin, SkillSelectionMixin, PlanUtilsMixin):
    """
    Fully agentic planner - no hardcoded logic.

    All planning decisions made by LLM:
    - Task type inference (semantic, not keyword matching)
    - Skill selection (capability-based matching)
    - Execution planning (adaptive, context-aware)
    - Dependency resolution (intelligent)
    """

    # Global semaphore to limit concurrent LLM calls (prevents rate limiting)
    _llm_semaphore = None
    _max_concurrent_llm_calls = 1  # Serialize LLM calls by default
    _semaphore_lock = None  # Initialized lazily as class-level threading.Lock

    @classmethod
    def set_max_concurrent_llm_calls(cls, max_calls: int) -> None:
        """Set maximum concurrent LLM calls across all planner instances."""
        cls._max_concurrent_llm_calls = max(1, max_calls)
        cls._llm_semaphore = None  # Reset to recreate with new limit

    @classmethod
    def _get_semaphore(cls) -> Any:
        """Get or create the global LLM semaphore (thread-safe)."""
        if cls._llm_semaphore is not None:
            return cls._llm_semaphore
        import threading

        if cls._semaphore_lock is None:
            cls._semaphore_lock = threading.Lock()
        with cls._semaphore_lock:
            # Double-checked locking: re-check after acquiring lock
            if cls._llm_semaphore is None:
                cls._llm_semaphore = threading.Semaphore(cls._max_concurrent_llm_calls)
        return cls._llm_semaphore

    def __init__(self, fast_model: str = "haiku") -> None:
        """Initialize agentic planner.

        Args:
            fast_model: Model for fast classification tasks (default: haiku).
                        Use 'haiku' for speed, 'sonnet' for accuracy.
        """
        dspy = _get_dspy()
        if not dspy:
            raise AgentExecutionError("DSPy required for TaskPlanner")
        _load_signatures()

        # Use ChainOfThought for execution planning (research: Plan-and-Solve benefits from explicit reasoning)
        self.execution_planner = dspy.ChainOfThought(ExecutionPlanningSignature)
        self._use_typed_predictor = False

        # Reflective planner for replanning after failures (Reflexion-style)
        self.reflective_planner = dspy.ChainOfThought(ReflectivePlanningSignature)

        self.task_type_inferrer = dspy.ChainOfThought(TaskTypeInferenceSignature)
        self.skill_selector = dspy.ChainOfThought(SkillSelectionSignature)
        self.capability_inferrer = dspy.Predict(CapabilityInferenceSignature)

        # Store signatures for JSON schema extraction
        self._signatures = {
            "task_type": TaskTypeInferenceSignature,
            "execution": ExecutionPlanningSignature,
            "skill_selection": SkillSelectionSignature,
            "capability": CapabilityInferenceSignature,
            "reflective": ReflectivePlanningSignature,
        }

        # Context compression for handling context length errors
        self._compressor = ContextCompressor() if CONTEXT_UTILS_AVAILABLE else None
        self._max_compression_retries = 3

        # Fast LM for classification tasks (task type inference, skill selection)
        # Uses Haiku by default for speed - these are simple classification tasks
        self._fast_lm = None
        self._fast_model = fast_model
        self._init_fast_lm()

        logger.info(f" TaskPlanner initialized (fast_model={fast_model} for classification)")

    def _init_fast_lm(self) -> None:
        """Initialize fast LM for routing/classification tasks.

        Priority:
        1. Gemini 2.0 Flash via OpenRouter (fastest: ~3.6s avg, cheapest)
        2. DirectAnthropicLM Haiku (fast: ~5.6s avg)
        3. DSPy global LM fallback (Sonnet — slower but works)
        """
        import os

        from Jotty.core.infrastructure.foundation.config_defaults import LLM_PLANNING_MAX_TOKENS

        # Use global LM singleton (shared across all components)
        try:
            from Jotty.core.infrastructure.foundation.llm_singleton import get_global_lm

            # Try to use Haiku for fast planning/routing
            self._fast_lm = get_global_lm(provider="anthropic", model="claude-haiku-4-5-20251001")
            self._fast_model = "haiku"
            logger.info(f"Fast LM: Using global LM (routing/classification)")
        except Exception as e:
            logger.warning(f"Could not get global LM for planning: {e}")
            # Fallback: try global LM without specifying model
            try:
                from Jotty.core.infrastructure.foundation.llm_singleton import get_global_lm

                self._fast_lm = get_global_lm()
                self._fast_model = "default"
                logger.info(f"Fast LM: Using global LM default model")
            except Exception as e2:
                logger.warning(f"Global LM not available: {e2}")
                self._fast_lm = None

    def _call_with_retry(
        self,
        module: Any,
        kwargs: Dict[str, Any],
        compressible_fields: Optional[List[str]] = None,
        max_retries: int = 5,
        lm: Optional[Any] = None,
    ) -> Any:
        """
        Call a DSPy module with automatic retry and context compression.

        Learned from BaseSwarmAgent pattern:
        - Detect error types (context length, timeout, parse, rate limit)
        - Compress context on context length errors
        - Exponential backoff on timeouts
        - Wait on rate limits (uses global semaphore to serialize calls)
        - Preserve trajectory/progress

        Args:
            module: DSPy module to call
            kwargs: Arguments to pass to module
            compressible_fields: Fields that can be compressed (e.g., 'available_skills')
            max_retries: Maximum retry attempts (default 5 for rate limit resilience)
            lm: Optional LM to use (for fast classification tasks)

        Returns:
            Module result or raises exception
        """
        import time

        compression_ratio = 0.7
        last_error = None
        semaphore = self._get_semaphore()

        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    logger.info(f"   Retry {attempt}/{max_retries}")

                # Use semaphore to serialize LLM calls (prevents rate limiting)
                with semaphore:
                    # Use specified LM or default
                    if lm:
                        with _get_dspy().context(lm=lm):
                            return module(**kwargs)
                    else:
                        return module(**kwargs)

            except Exception as e:
                last_error = e

                # Detect error type
                if CONTEXT_UTILS_AVAILABLE:
                    error_type, strategy = detect_error_type(e)
                    logger.debug(f"   Error type detected: {error_type.value}")
                else:
                    # Fallback detection
                    error_str = str(e).lower()
                    error_type_str = type(e).__name__.lower()

                    if any(p in error_str for p in ["context", "token", "too long"]):
                        error_type = "context_length"
                        strategy = {"should_retry": True, "action": "compress"}
                    elif (
                        any(
                            p in error_str
                            for p in [
                                "rate limit",
                                "rate_limit",
                                "ratelimit",
                                "too many requests",
                                "429",
                            ]
                        )
                        or "ratelimit" in error_type_str
                    ):
                        # Rate limit error - wait longer before retry
                        error_type = "rate_limit"
                        # Extract wait time from error message if available (e.g., "Try again in 60 seconds")
                        import re

                        wait_match = re.search(r"(\d+)\s*seconds?", error_str)
                        wait_time = int(wait_match.group(1)) if wait_match else 60
                        strategy = {
                            "should_retry": True,
                            "action": "wait",
                            "delay_seconds": wait_time,
                        }
                        logger.warning(f"Rate limit hit, will wait {wait_time}s before retry")
                    elif "timeout" in error_str:
                        error_type = "timeout"
                        strategy = {"should_retry": True, "action": "backoff", "delay_seconds": 2}
                    else:
                        error_type = "unknown"
                        strategy = {"should_retry": False}

                if not strategy.get("should_retry") or attempt >= max_retries:
                    raise

                action = strategy.get("action", "fail")

                if action == "compress" and compressible_fields and self._compressor:
                    # Compress specified fields
                    for field_name in compressible_fields:
                        if field_name in kwargs and kwargs[field_name]:
                            original = kwargs[field_name]
                            if isinstance(original, str) and len(original) > 1000:
                                result = self._compressor.compress(
                                    original, target_ratio=compression_ratio
                                )
                                kwargs[field_name] = result.content
                                logger.info(
                                    f"   Compressed {field_name}: {result.original_length} → {result.compressed_length} chars"
                                )

                    compression_ratio *= 0.7  # More aggressive next time

                elif action == "backoff":
                    delay = strategy.get("delay_seconds", 1) * (2**attempt)
                    delay = min(delay, 30)  # Cap at 30s
                    logger.info(f"   Backing off for {delay}s...")
                    time.sleep(delay)

                elif action == "wait":
                    delay = strategy.get("delay_seconds", 30)
                    logger.info(f"   Rate limited, waiting {delay}s...")
                    time.sleep(delay)

        # Should not reach here, but just in case
        if last_error:
            raise last_error
        raise AgentExecutionError("Unexpected state in retry logic")

    async def _acall_with_retry(
        self,
        module: Any,
        kwargs: Dict[str, Any],
        compressible_fields: Optional[List[str]] = None,
        max_retries: int = 5,
        lm: Optional[Any] = None,
    ) -> Any:
        """
        Async version of _call_with_retry using DSPy's native .acall().

        Advantages over the sync version:
        - Does not block the event loop (no thread pool needed)
        - Uses asyncio.sleep() for backoff (event loop stays responsive)
        - Native async all the way to the LLM provider

        Falls back to sync _call_with_retry if module has no .acall().
        """
        import asyncio

        # Fallback if module doesn't support async
        if not hasattr(module, "acall"):
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,
                lambda: self._call_with_retry(module, kwargs, compressible_fields, max_retries, lm),
            )

        compression_ratio = 0.7
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    logger.info(f"   Retry {attempt}/{max_retries}")

                # Use specified LM context or default
                if lm:
                    with _get_dspy().context(lm=lm):
                        return await module.acall(**kwargs)
                else:
                    return await module.acall(**kwargs)

            except Exception as e:
                last_error = e

                # Detect error type (same logic as sync version)
                if CONTEXT_UTILS_AVAILABLE:
                    error_type, strategy = detect_error_type(e)
                    logger.debug(f"   Error type detected: {error_type.value}")
                else:
                    error_str = str(e).lower()
                    error_type_str = type(e).__name__.lower()

                    if any(p in error_str for p in ["context", "token", "too long"]):
                        error_type = "context_length"
                        strategy = {"should_retry": True, "action": "compress"}
                    elif (
                        any(
                            p in error_str
                            for p in [
                                "rate limit",
                                "rate_limit",
                                "ratelimit",
                                "too many requests",
                                "429",
                            ]
                        )
                        or "ratelimit" in error_type_str
                    ):
                        import re

                        wait_match = re.search(r"(\d+)\s*seconds?", error_str)
                        wait_time = int(wait_match.group(1)) if wait_match else 60
                        strategy = {
                            "should_retry": True,
                            "action": "wait",
                            "delay_seconds": wait_time,
                        }
                        logger.warning(f"Rate limit hit, will wait {wait_time}s before retry")
                    elif "timeout" in error_str:
                        error_type = "timeout"
                        strategy = {"should_retry": True, "action": "backoff", "delay_seconds": 2}
                    else:
                        error_type = "unknown"
                        strategy = {"should_retry": False}

                if not strategy.get("should_retry") or attempt >= max_retries:
                    raise

                action = strategy.get("action", "fail")

                if action == "compress" and compressible_fields and self._compressor:
                    for field_name in compressible_fields:
                        if field_name in kwargs and kwargs[field_name]:
                            original = kwargs[field_name]
                            if isinstance(original, str) and len(original) > 1000:
                                result = self._compressor.compress(
                                    original, target_ratio=compression_ratio
                                )
                                kwargs[field_name] = result.content
                                logger.info(
                                    f"   Compressed {field_name}: {result.original_length} → {result.compressed_length} chars"
                                )
                    compression_ratio *= 0.7

                elif action == "backoff":
                    delay = strategy.get("delay_seconds", 1) * (2**attempt)
                    delay = min(delay, 30)
                    logger.info(f"   Backing off for {delay}s...")
                    await asyncio.sleep(delay)

                elif action == "wait":
                    delay = strategy.get("delay_seconds", 30)
                    logger.info(f"   Rate limited, waiting {delay}s...")
                    await asyncio.sleep(delay)

        if last_error:
            raise last_error
        raise AgentExecutionError("Unexpected state in async retry logic")

    def plan_execution(
        self,
        task: str,
        task_type: Any,
        skills: List[Dict[str, Any]],
        previous_outputs: Optional[Dict[str, Any]] = None,
        max_steps: int = 10,
    ) -> Any:
        """
        Plan execution steps using LLM reasoning.

        Args:
            task: Task description
            task_type: Inferred task type
            skills: Available skills (if empty, uses default file-operations)
            previous_outputs: Outputs from previous steps
            max_steps: Maximum steps

        Returns:
            (execution_steps, reasoning)
        """
        try:
            # If no skills provided, add default file-operations skill for creation tasks
            if not skills:
                task_type_value = task_type.value if hasattr(task_type, "value") else str(task_type)
                if task_type_value in ["creation", "unknown"]:
                    logger.info("No skills provided, adding default file-operations skill")
                    skills = [
                        {
                            "name": "file-operations",
                            "description": "Create, read, write files",
                            "tools": [
                                {
                                    "name": "write_file_tool",
                                    "params": {"path": "string", "content": "string"},
                                },
                                {"name": "read_file_tool", "params": {"path": "string"}},
                            ],
                        }
                    ]
                else:
                    logger.warning(f"No skills available for task type '{task_type_value}'")
                    return [], f"No skills available for task type '{task_type_value}'"

            # Format skills with tool schemas for the LLM
            formatted_skills = self._format_skills_for_planner(skills)
            skills_json = json.dumps(formatted_skills, indent=2)
            outputs_json = json.dumps(previous_outputs or {}, indent=2)

            abstracted_task = self._abstract_task_for_planning(task)
            logger.debug(f"Task: '{abstracted_task[:80]}'")

            logger.info(f"Calling LLM for execution plan...")
            logger.debug(f"   Task: {abstracted_task[:100]}")
            logger.debug(f"   Skills count: {len(skills)}")

            task_type_str = task_type.value if hasattr(task_type, "value") else str(task_type)
            planner_kwargs = {
                "task_description": abstracted_task,
                "task_type": task_type_str,
                "available_skills": skills_json,
                "previous_outputs": outputs_json,
                "max_steps": max_steps,
                "config": {"response_format": {"type": "json_object"}},
            }

            result = self._call_with_retry(
                module=self.execution_planner,
                kwargs=planner_kwargs,
                compressible_fields=["available_skills", "previous_outputs"],
                max_retries=self._max_compression_retries,
                lm=self._fast_lm,
            )

            logger.info(f"LLM response received")
            raw_plan = getattr(result, "execution_plan", None)
            logger.debug(f"   Raw execution_plan type: {type(raw_plan)}")
            logger.debug(
                f"   Raw execution_plan (first 500 chars): {str(raw_plan)[:500] if raw_plan else 'NONE'}"
            )

            # All normalization + parsing in one place
            steps = self._parse_plan_to_steps(raw_plan, skills, task, task_type, max_steps)

            # Determine reasoning
            if not steps:
                # Try fallback plan
                logger.warning("Execution plan resulted in 0 steps, using fallback plan")
                fallback_plan_data = self._create_fallback_plan(task, task_type, skills)
                steps = self._parse_plan_to_steps(
                    fallback_plan_data, skills, task, task_type, max_steps
                )
                reasoning = f"Fallback plan created: {len(steps)} steps"
            else:
                reasoning = result.reasoning or f"Planned {len(steps)} steps"

            # Post-plan quality check: decompose composite skills for complex tasks
            decomposed = self._maybe_decompose_plan(steps, skills, task, task_type)
            if decomposed is not None:
                logger.info(f" Plan decomposed: {len(steps)} steps → {len(decomposed)} steps")
                steps = decomposed
                reasoning = f"Decomposed for quality: {reasoning}"

            # Post-plan enrichment: auto-populate I/O contracts from tool schemas
            # The fast LLM often omits inputs_needed/outputs_produced and uses
            # bare ${step_0} refs instead of field-level ${step_0.holdings}
            steps = self._enrich_io_contracts(steps)

            used_skills = {step.skill_name for step in steps}
            if len(steps) > 0:
                logger.info(f" Plan uses {len(used_skills)} skills: {used_skills}")

            logger.info(f" Planned {len(steps)} execution steps")
            logger.debug(f"   Reasoning: {reasoning}")
            if hasattr(result, "estimated_complexity"):
                logger.debug(f"   Complexity: {result.estimated_complexity}")

            return steps, reasoning

        except Exception as e:
            logger.error(f"Execution planning failed: {e}", exc_info=True)
            logger.warning("Attempting fallback plan due to execution planning failure")
            try:
                fallback_plan_data = self._create_fallback_plan(task, task_type, skills)
                logger.info(
                    f" Fallback plan generated {len(fallback_plan_data)} steps: {fallback_plan_data}"
                )

                if not fallback_plan_data:
                    logger.error("Fallback plan returned empty list!")
                    return [], f"Planning failed: {e}"

                steps = self._parse_plan_to_steps(
                    fallback_plan_data, skills, task, task_type, max_steps
                )

                if steps:
                    logger.info(f" Fallback plan created: {len(steps)} steps")
                    return steps, f"Fallback plan (planning failed: {str(e)[:100]})"
                else:
                    logger.error(
                        f" Fallback plan generated steps but 0 were converted to ExecutionStep objects"
                    )
            except Exception as fallback_e:
                logger.error(f"Fallback plan also failed: {fallback_e}", exc_info=True)

            return [], f"Planning failed: {e}"

    async def aplan_execution(
        self,
        task: str,
        task_type: Any,
        skills: List[Dict[str, Any]],
        previous_outputs: Optional[Dict[str, Any]] = None,
        max_steps: int = 10,
    ) -> Any:
        """
        Async version of plan_execution using DSPy .acall().

        Non-blocking: uses _acall_with_retry (asyncio.sleep for backoff,
        module.acall for LLM calls). No thread pool needed.

        Shares ALL parsing/fallback logic with the sync version.
        """
        try:
            # Pre-processing (CPU-only, identical to sync version)
            if not skills:
                task_type_value = task_type.value if hasattr(task_type, "value") else str(task_type)
                if task_type_value in ["creation", "unknown"]:
                    skills = [
                        {
                            "name": "file-operations",
                            "description": "Create, read, write files",
                            "tools": [
                                {
                                    "name": "write_file_tool",
                                    "params": {"path": "string", "content": "string"},
                                },
                                {"name": "read_file_tool", "params": {"path": "string"}},
                            ],
                        }
                    ]
                else:
                    return [], f"No skills available for task type '{task_type_value}'"

            # Format skills (reuse sync method — pure CPU)
            formatted_skills = self._format_skills_for_planner(skills)
            skills_json = json.dumps(formatted_skills, indent=2)
            outputs_json = json.dumps(previous_outputs or {}, indent=2)

            abstracted_task = self._abstract_task_for_planning(task)
            task_type_str = task_type.value if hasattr(task_type, "value") else str(task_type)

            planner_kwargs = {
                "task_description": abstracted_task,
                "task_type": task_type_str,
                "available_skills": skills_json,
                "previous_outputs": outputs_json,
                "max_steps": max_steps,
                "config": {"response_format": {"type": "json_object"}},
            }

            logger.info(f" Calling LLM for execution plan (async)...")

            # ── ASYNC LLM CALL ─────────────────────────────────
            result = await self._acall_with_retry(
                module=self.execution_planner,
                kwargs=planner_kwargs,
                compressible_fields=["available_skills", "previous_outputs"],
                max_retries=self._max_compression_retries,
                lm=self._fast_lm,
            )

            logger.info(f" LLM response received (async)")

            # Post-processing: parse plan (reuse sync method — pure CPU)
            raw_plan = getattr(result, "execution_plan", None)
            steps = self._parse_plan_to_steps(raw_plan, skills, task, task_type, max_steps)

            if not steps:
                logger.warning("Async plan resulted in 0 steps, using fallback plan")
                fallback_plan_data = self._create_fallback_plan(task, task_type, skills)
                steps = self._parse_plan_to_steps(
                    fallback_plan_data, skills, task, task_type, max_steps
                )
                reasoning = f"Fallback plan created: {len(steps)} steps"
            else:
                reasoning = result.reasoning or f"Planned {len(steps)} steps"

            # Post-plan quality check
            decomposed = self._maybe_decompose_plan(steps, skills, task, task_type)
            if decomposed is not None:
                logger.info(f"Plan decomposed: {len(steps)} steps -> {len(decomposed)} steps")
                steps = decomposed
                reasoning = f"Decomposed for quality: {reasoning}"

            # Post-plan enrichment: auto-populate I/O contracts from tool schemas
            steps = self._enrich_io_contracts(steps)

            logger.info(f"Planned {len(steps)} execution steps (async)")
            return steps, reasoning

        except Exception as e:
            logger.error(f"Async execution planning failed: {e}", exc_info=True)
            try:
                fallback_plan_data = self._create_fallback_plan(task, task_type, skills)
                steps = self._parse_plan_to_steps(
                    fallback_plan_data, skills, task, task_type, max_steps
                )
                if steps:
                    return steps, f"Fallback plan (async planning failed: {str(e)[:100]})"
            except Exception as fallback_e:
                logger.error(f"Async fallback plan also failed: {fallback_e}")
            return [], f"Planning failed: {e}"

    async def areplan_with_reflection(
        self,
        task: str,
        task_type: Any,
        skills: List[Dict[str, Any]],
        failed_steps: List[Dict[str, Any]],
        completed_outputs: Optional[Dict[str, Any]] = None,
        excluded_skills: Optional[List[str]] = None,
        max_steps: int = 5,
    ) -> Any:
        """
        Async version of replan_with_reflection using DSPy .acall().

        Non-blocking: uses _acall_with_retry for the LLM call.
        """
        excluded_set = set(excluded_skills or [])
        filtered_skills = [s for s in skills if s.get("name") not in excluded_set]
        abstracted_task = self._abstract_task_for_planning(task)
        task_type_str = task_type.value if hasattr(task_type, "value") else str(task_type)

        formatted_skills = []
        for s in filtered_skills:
            formatted_skills.append(
                {
                    "name": s.get("name", ""),
                    "description": s.get("description", ""),
                    "tools": s.get("tools", []),
                }
            )
        skills_json = json.dumps(formatted_skills, indent=2)
        failed_json = json.dumps(failed_steps, default=str)
        outputs_json = json.dumps(completed_outputs or {}, default=str)
        excluded_json = json.dumps(list(excluded_set))

        try:
            # ── ASYNC LLM CALL ─────────────────────────────────
            result = await self._acall_with_retry(
                module=self.reflective_planner,
                kwargs={
                    "task_description": abstracted_task,
                    "task_type": task_type_str,
                    "available_skills": skills_json,
                    "failed_steps": failed_json,
                    "completed_outputs": outputs_json,
                    "excluded_skills": excluded_json,
                    "max_steps": max_steps,
                },
                compressible_fields=["available_skills", "completed_outputs"],
                max_retries=self._max_compression_retries,
                lm=self._fast_lm,
            )

            raw_plan = getattr(result, "corrected_plan", None)
            reflection = str(getattr(result, "reflection", ""))
            reasoning = str(getattr(result, "reasoning", ""))
            steps = self._parse_plan_to_steps(raw_plan, filtered_skills, task, task_type, max_steps)
            steps = self._enrich_io_contracts(steps)

            if steps:
                logger.info(f"Reflective replan produced {len(steps)} new steps (async)")
                return steps, reflection, reasoning

        except Exception as e:
            logger.warning(f"Async reflective replanning failed: {e}, falling back")

        # Fallback to async plan_execution
        try:
            steps, reasoning = await self.aplan_execution(
                task=task,
                task_type=task_type,
                skills=filtered_skills,
                previous_outputs=completed_outputs,
                max_steps=max_steps,
            )
            return steps, "Fallback: regular replanning (reflection failed)", reasoning
        except Exception as e:
            logger.error(f"Async fallback replanning failed: {e}")
            return [], f"All replanning failed: {e}", ""

    def _format_skills_for_planner(self, skills: List[Dict[str, Any]]) -> list:
        """
        Format skills with tool schemas for the planner LLM.

        Extracted from plan_execution() so both sync and async versions
        share the same formatting logic.  Includes ``executor_type``
        (api / gui / hybrid / general) so the planner can prefer fast
        API shortcuts over slow GUI interactions.
        """
        formatted_skills = []
        for s in skills:
            skill_name = s.get("name", "")
            skill_dict = {"name": skill_name, "description": s.get("description", ""), "tools": []}

            # Propagate executor_type from skill metadata for hybrid routing
            if s.get("executor_type"):
                skill_dict["executor_type"] = s["executor_type"]

            tools_raw = s.get("tools", [])
            if isinstance(tools_raw, dict):
                tool_names = list(tools_raw.keys())
            elif isinstance(tools_raw, list):
                tool_names = [t.get("name") if isinstance(t, dict) else t for t in tools_raw]
            else:
                tool_names = []

            try:
                from ..registry.skills_registry import get_skills_registry

                registry = get_skills_registry()
                if registry:
                    skill_obj = registry.get_skill(skill_name)
                    if skill_obj and hasattr(skill_obj, "tools") and skill_obj.tools:
                        if not tool_names:
                            tool_names = list(skill_obj.tools.keys())
                        for tool_name in tool_names:
                            tool_func = skill_obj.tools.get(tool_name)
                            if tool_func:
                                tool_schema = self._extract_tool_schema(tool_func, tool_name)
                                skill_dict["tools"].append(tool_schema)
                            else:
                                skill_dict["tools"].append({"name": tool_name})
                        # Extract executor_type from loaded skill if not in metadata
                        if "executor_type" not in skill_dict:
                            etype = getattr(skill_obj, "executor_type", "")
                            if etype:
                                skill_dict["executor_type"] = etype
                    else:
                        skill_dict["tools"] = [{"name": name} for name in tool_names]
                else:
                    skill_dict["tools"] = [{"name": name} for name in tool_names]
            except Exception as e:
                logger.warning(f"Could not enrich tool schemas for {skill_name}: {e}")
                skill_dict["tools"] = [{"name": name} for name in tool_names]

            formatted_skills.append(skill_dict)

        logger.info(f"Formatted {len(formatted_skills)} skills with tool schemas for LLM")
        return formatted_skills

    # _normalize_raw_plan, _parse_plan_to_steps → moved to PlanUtilsMixin

    def replan_with_reflection(
        self,
        task: str,
        task_type: Any,
        skills: List[Dict[str, Any]],
        failed_steps: List[Dict[str, Any]],
        completed_outputs: Optional[Dict[str, Any]] = None,
        excluded_skills: Optional[List[str]] = None,
        max_steps: int = 10,
    ) -> tuple:
        """
        Replan after failure using Reflexion-style analysis.

        Filters excluded skills, formats failure context, calls reflective_planner,
        and parses the result using _parse_plan_to_steps.

        Args:
            task: Original task description
            task_type: Task type (enum or string)
            skills: Available skills
            failed_steps: List of dicts with {skill_name, tool_name, error, params}
            completed_outputs: Outputs from successful steps
            excluded_skills: Skills to blacklist
            max_steps: Maximum remaining steps

        Returns:
            (steps, reflection, reasoning) - steps is list of ExecutionStep,
            reflection is failure analysis, reasoning is plan explanation
        """
        excluded_set = set(excluded_skills or [])

        # Filter excluded skills from available set
        filtered_skills = [s for s in skills if s.get("name") not in excluded_set]

        # Format inputs for reflective planner
        abstracted_task = self._abstract_task_for_planning(task)
        task_type_str = task_type.value if hasattr(task_type, "value") else str(task_type)

        formatted_skills = []
        for s in filtered_skills:
            formatted_skills.append(
                {
                    "name": s.get("name", ""),
                    "description": s.get("description", ""),
                    "tools": s.get("tools", []),
                }
            )
        skills_json = json.dumps(formatted_skills, indent=2)
        failed_json = json.dumps(failed_steps, default=str)
        outputs_json = json.dumps(completed_outputs or {}, default=str)
        excluded_json = json.dumps(list(excluded_set))

        try:
            result = self._call_with_retry(
                module=self.reflective_planner,
                kwargs={
                    "task_description": abstracted_task,
                    "task_type": task_type_str,
                    "available_skills": skills_json,
                    "failed_steps": failed_json,
                    "completed_outputs": outputs_json,
                    "excluded_skills": excluded_json,
                    "max_steps": max_steps,
                },
                compressible_fields=["available_skills", "completed_outputs"],
                max_retries=self._max_compression_retries,
                lm=self._fast_lm,  # Use fast LM for replanning (routing task)
            )

            raw_plan = getattr(result, "corrected_plan", None)
            reflection = str(getattr(result, "reflection", ""))
            reasoning = str(getattr(result, "reasoning", ""))

            logger.info(f" Reflective replanning: reflection='{reflection[:100]}...'")

            steps = self._parse_plan_to_steps(raw_plan, filtered_skills, task, task_type, max_steps)
            steps = self._enrich_io_contracts(steps)

            if steps:
                logger.info(f" Reflective replan produced {len(steps)} new steps")
                return steps, reflection, reasoning

        except Exception as e:
            logger.warning(f"Reflective replanning failed: {e}, falling back to regular replanning")

        # Fallback to regular plan_execution if reflection fails
        try:
            steps, reasoning = self.plan_execution(
                task=task,
                task_type=task_type,
                skills=filtered_skills,
                previous_outputs=completed_outputs,
                max_steps=max_steps,
            )
            return steps, "Fallback: regular replanning (reflection failed)", reasoning
        except Exception as e:
            logger.error(f"Fallback replanning also failed: {e}")
            return [], f"All replanning failed: {e}", ""

    # _maybe_decompose_plan, _extract_comparison_entities → moved to PlanUtilsMixin

    # =========================================================================
    # POLICY EXPLORER — Alternative plan generation when stuck
    # =========================================================================

    def explore_alternative_plans(
        self,
        task: str,
        task_type: Any,
        skills: List[Dict[str, Any]],
        failed_approaches: List[str],
        max_alternatives: int = 3,
    ) -> List[tuple]:
        """Generate alternative approaches when the current plan is stuck.

        Uses LLM to brainstorm fundamentally different strategies, not just
        minor variations. Each alternative avoids previously failed approaches.

        Args:
            task: Original task description
            task_type: Task type
            skills: Available skills
            failed_approaches: Descriptions of what was already tried and failed
            max_alternatives: Max number of alternatives to generate

        Returns:
            List of (steps, reasoning) tuples — each is a complete alternative plan
        """
        dspy = _get_dspy()
        if not dspy:
            return []

        failed_summary = "\n".join(f"- FAILED: {a}" for a in failed_approaches[-5:])
        task_type_str = task_type.value if hasattr(task_type, "value") else str(task_type)
        formatted_skills = (
            self._format_skills_for_planner(skills)
            if hasattr(self, "_format_skills_for_planner")
            else skills
        )
        skills_json = json.dumps(formatted_skills, indent=2)

        prompt = (
            f"Task: {task}\n"
            f"Task type: {task_type_str}\n"
            f"Previous failed approaches:\n{failed_summary}\n\n"
            f"Generate a COMPLETELY DIFFERENT approach using these skills:\n{skills_json}\n"
            f"The new approach must avoid the failed strategies entirely."
        )

        alternatives = []
        for i in range(max_alternatives):
            try:
                result = self._call_with_retry(
                    module=self.execution_planner,
                    kwargs={
                        "task_description": prompt,
                        "task_type": task_type_str,
                        "available_skills": skills_json,
                        "previous_outputs": json.dumps({"failed": failed_approaches}),
                        "max_steps": 5,
                    },
                    max_retries=1,
                    lm=self._fast_lm,
                )
                raw_plan = getattr(result, "execution_plan", None)
                steps = self._parse_plan_to_steps(raw_plan, skills, task, task_type, 5)
                if steps:
                    reasoning = getattr(result, "reasoning", f"Alternative plan {i+1}")
                    alternatives.append((steps, str(reasoning)))
            except Exception as e:
                logger.debug(f"Alternative plan {i+1} generation failed: {e}")
                break

        logger.info(f"Policy explorer: generated {len(alternatives)} alternative plans")
        return alternatives

    # =========================================================================
    # DATA-FLOW DEPENDENCY INFERENCE
    # =========================================================================

    @staticmethod
    def infer_data_dependencies(steps: list) -> list:
        """Infer data-flow dependencies between execution steps.

        Analyzes step parameters and output keys to determine which steps
        produce data that other steps consume. This enables DAG-based
        parallel execution of independent steps.

        Args:
            steps: List of ExecutionStep objects

        Returns:
            Same steps with updated depends_on fields
        """
        # Build a map of output_key → step_index
        output_map: Dict[str, int] = {}
        for i, step in enumerate(steps):
            out_key = getattr(step, "output_key", None) or f"step_{i}"
            output_map[out_key] = i

        # For each step, check if its params reference earlier step outputs
        for i, step in enumerate(steps):
            params = getattr(step, "params", {}) or {}
            inferred_deps = set(getattr(step, "depends_on", None) or [])

            for param_value in params.values():
                if not isinstance(param_value, str):
                    continue
                # Check for template references like {{step_0.result}} or {research_output}
                for out_key, step_idx in output_map.items():
                    if step_idx >= i:
                        continue  # Can't depend on later or self
                    if out_key in param_value:
                        inferred_deps.add(step_idx)

            # If no explicit or inferred deps and this isn't step 0,
            # assume sequential dependency (safe default)
            if not inferred_deps and i > 0:
                # Check if this step's skill produces data the next needs
                # If it can't be determined, leave depends_on empty (= parallelizable)
                pass

            if hasattr(step, "depends_on"):
                step.depends_on = sorted(inferred_deps)

        return steps


@dataclass
class TaskPlan:
    """Execution plan with enhanced metadata."""

    task_graph: Optional[Any] = None  # TaskGraph if available
    steps: List[Any] = field(default_factory=list)  # List[ExecutionStep] - imported lazily
    estimated_time: Optional[str] = None
    required_tools: List[str] = field(default_factory=list)
    required_credentials: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Convenience Functions
# =============================================================================


def create_agentic_planner() -> TaskPlanner:
    """Create a new agentic planner instance."""
    return TaskPlanner()
