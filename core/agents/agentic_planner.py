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
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from Jotty.core.foundation.exceptions import AgentExecutionError

# DSPy loaded lazily to avoid ~6s import at module level
DSPY_AVAILABLE = True  # Assumed; checked on first use
_dspy_module = None

def _get_dspy():
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
from ._execution_types import ExecutionStep, ExecutionStepSchema, TaskType, ExecutionResult

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

def _load_signatures():
    """Load DSPy signatures on first use."""
    global _signatures_loaded, TaskTypeInferenceSignature, CapabilityInferenceSignature
    global ExecutionPlanningSignature, SkillSelectionSignature, ReflectivePlanningSignature
    if _signatures_loaded:
        return
    _signatures_loaded = True
    from .planner_signatures import (
        TaskTypeInferenceSignature as _TT,
        CapabilityInferenceSignature as _CI,
        ExecutionPlanningSignature as _EP,
        SkillSelectionSignature as _SS,
        ReflectivePlanningSignature as _RP,
    )
    TaskTypeInferenceSignature = _TT
    CapabilityInferenceSignature = _CI
    ExecutionPlanningSignature = _EP
    SkillSelectionSignature = _SS
    ReflectivePlanningSignature = _RP


# =============================================================================
# Agentic Planner
# =============================================================================


from ._inference_mixin import InferenceMixin
from ._skill_selection_mixin import SkillSelectionMixin
from ._plan_utils_mixin import PlanUtilsMixin

class AgenticPlanner(InferenceMixin, SkillSelectionMixin, PlanUtilsMixin):
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
    def set_max_concurrent_llm_calls(cls, max_calls: int):
        """Set maximum concurrent LLM calls across all planner instances."""
        cls._max_concurrent_llm_calls = max(1, max_calls)
        cls._llm_semaphore = None  # Reset to recreate with new limit

    @classmethod
    def _get_semaphore(cls):
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

    def __init__(self, fast_model: str = "haiku"):
        """Initialize agentic planner.

        Args:
            fast_model: Model for fast classification tasks (default: haiku).
                        Use 'haiku' for speed, 'sonnet' for accuracy.
        """
        dspy = _get_dspy()
        if not dspy:
            raise AgentExecutionError("DSPy required for AgenticPlanner")
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
            'task_type': TaskTypeInferenceSignature,
            'execution': ExecutionPlanningSignature,
            'skill_selection': SkillSelectionSignature,
            'capability': CapabilityInferenceSignature,
            'reflective': ReflectivePlanningSignature,
        }

        # Context compression for handling context length errors
        self._compressor = ContextCompressor() if CONTEXT_UTILS_AVAILABLE else None
        self._max_compression_retries = 3

        # Fast LM for classification tasks (task type inference, skill selection)
        # Uses Haiku by default for speed - these are simple classification tasks
        self._fast_lm = None
        self._fast_model = fast_model
        self._init_fast_lm()

        logger.info(f"🧠 AgenticPlanner initialized (fast_model={fast_model} for classification)")

    def _init_fast_lm(self):
        """Initialize fast LM for routing/classification tasks.

        Priority:
        1. Gemini 2.0 Flash via OpenRouter (fastest: ~3.6s avg, cheapest)
        2. DirectAnthropicLM Haiku (fast: ~5.6s avg)
        3. DSPy global LM fallback (Sonnet — slower but works)
        """
        import os

        from ..foundation.config_defaults import LLM_PLANNING_MAX_TOKENS

        # 1. Try Gemini Flash via OpenRouter (35% faster than Haiku, 8x cheaper)
        or_key = os.environ.get('OPENROUTER_API_KEY')
        if or_key:
            try:
                import dspy
                self._fast_lm = dspy.LM(
                    'openrouter/google/gemini-2.0-flash-001',
                    api_key=or_key,
                    max_tokens=LLM_PLANNING_MAX_TOKENS,
                )
                self._fast_model = 'gemini-2.0-flash'
                logger.info(f"Fast LM: Gemini 2.0 Flash via OpenRouter (routing/classification)")
                return
            except Exception as e:
                logger.debug(f"Gemini Flash not available: {e}")

        # 2. Try DirectAnthropicLM Haiku (fast, reliable)
        if os.environ.get('ANTHROPIC_API_KEY'):
            try:
                from ..foundation.direct_anthropic_lm import DirectAnthropicLM
                self._fast_lm = DirectAnthropicLM(
                    model='haiku',
                    max_tokens=LLM_PLANNING_MAX_TOKENS,
                )
                self._fast_model = 'haiku'
                logger.info(f"Fast LM: Anthropic Haiku (routing/classification)")
                return
            except Exception as e:
                logger.debug(f"DirectAnthropicLM Haiku not available: {e}")

        # 3. Fallback: use whatever DSPy has configured (Sonnet)
        self._fast_lm = None
        logger.info(f"Fast LM: using default DSPy LM (no dedicated routing model)")

    def _call_with_retry(
        self,
        module,
        kwargs: Dict[str, Any],
        compressible_fields: Optional[List[str]] = None,
        max_retries: int = 5,
        lm: Optional[Any] = None
    ):
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

                    if any(p in error_str for p in ['context', 'token', 'too long']):
                        error_type = 'context_length'
                        strategy = {'should_retry': True, 'action': 'compress'}
                    elif any(p in error_str for p in ['rate limit', 'rate_limit', 'ratelimit', 'too many requests', '429']) or 'ratelimit' in error_type_str:
                        # Rate limit error - wait longer before retry
                        error_type = 'rate_limit'
                        # Extract wait time from error message if available (e.g., "Try again in 60 seconds")
                        import re
                        wait_match = re.search(r'(\d+)\s*seconds?', error_str)
                        wait_time = int(wait_match.group(1)) if wait_match else 60
                        strategy = {'should_retry': True, 'action': 'wait', 'delay_seconds': wait_time}
                        logger.warning(f"Rate limit hit, will wait {wait_time}s before retry")
                    elif 'timeout' in error_str:
                        error_type = 'timeout'
                        strategy = {'should_retry': True, 'action': 'backoff', 'delay_seconds': 2}
                    else:
                        error_type = 'unknown'
                        strategy = {'should_retry': False}

                if not strategy.get('should_retry') or attempt >= max_retries:
                    raise

                action = strategy.get('action', 'fail')

                if action == 'compress' and compressible_fields and self._compressor:
                    # Compress specified fields
                    for field in compressible_fields:
                        if field in kwargs and kwargs[field]:
                            original = kwargs[field]
                            if isinstance(original, str) and len(original) > 1000:
                                result = self._compressor.compress(
                                    original,
                                    target_ratio=compression_ratio
                                )
                                kwargs[field] = result.content
                                logger.info(f"   Compressed {field}: {result.original_length} → {result.compressed_length} chars")

                    compression_ratio *= 0.7  # More aggressive next time

                elif action == 'backoff':
                    delay = strategy.get('delay_seconds', 1) * (2 ** attempt)
                    delay = min(delay, 30)  # Cap at 30s
                    logger.info(f"   Backing off for {delay}s...")
                    time.sleep(delay)

                elif action == 'wait':
                    delay = strategy.get('delay_seconds', 30)
                    logger.info(f"   Rate limited, waiting {delay}s...")
                    time.sleep(delay)

        # Should not reach here, but just in case
        if last_error:
            raise last_error
        raise AgentExecutionError("Unexpected state in retry logic")

    async def _acall_with_retry(
        self,
        module,
        kwargs: Dict[str, Any],
        compressible_fields: Optional[List[str]] = None,
        max_retries: int = 5,
        lm: Optional[Any] = None
    ):
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
        if not hasattr(module, 'acall'):
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

                    if any(p in error_str for p in ['context', 'token', 'too long']):
                        error_type = 'context_length'
                        strategy = {'should_retry': True, 'action': 'compress'}
                    elif any(p in error_str for p in ['rate limit', 'rate_limit', 'ratelimit', 'too many requests', '429']) or 'ratelimit' in error_type_str:
                        import re
                        wait_match = re.search(r'(\d+)\s*seconds?', error_str)
                        wait_time = int(wait_match.group(1)) if wait_match else 60
                        strategy = {'should_retry': True, 'action': 'wait', 'delay_seconds': wait_time}
                        logger.warning(f"Rate limit hit, will wait {wait_time}s before retry")
                    elif 'timeout' in error_str:
                        error_type = 'timeout'
                        strategy = {'should_retry': True, 'action': 'backoff', 'delay_seconds': 2}
                    else:
                        error_type = 'unknown'
                        strategy = {'should_retry': False}

                if not strategy.get('should_retry') or attempt >= max_retries:
                    raise

                action = strategy.get('action', 'fail')

                if action == 'compress' and compressible_fields and self._compressor:
                    for field in compressible_fields:
                        if field in kwargs and kwargs[field]:
                            original = kwargs[field]
                            if isinstance(original, str) and len(original) > 1000:
                                result = self._compressor.compress(
                                    original,
                                    target_ratio=compression_ratio
                                )
                                kwargs[field] = result.content
                                logger.info(f"   Compressed {field}: {result.original_length} → {result.compressed_length} chars")
                    compression_ratio *= 0.7

                elif action == 'backoff':
                    delay = strategy.get('delay_seconds', 1) * (2 ** attempt)
                    delay = min(delay, 30)
                    logger.info(f"   Backing off for {delay}s...")
                    await asyncio.sleep(delay)

                elif action == 'wait':
                    delay = strategy.get('delay_seconds', 30)
                    logger.info(f"   Rate limited, waiting {delay}s...")
                    await asyncio.sleep(delay)

        if last_error:
            raise last_error
        raise AgentExecutionError("Unexpected state in async retry logic")

    def plan_execution(
        self,
        task: str,
        task_type,
        skills: List[Dict[str, Any]],
        previous_outputs: Optional[Dict[str, Any]] = None,
        max_steps: int = 10
    ):
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
                task_type_value = task_type.value if hasattr(task_type, 'value') else str(task_type)
                if task_type_value in ['creation', 'unknown']:
                    logger.info("No skills provided, adding default file-operations skill")
                    skills = [{
                        'name': 'file-operations',
                        'description': 'Create, read, write files',
                        'tools': [
                            {'name': 'write_file_tool', 'params': {'path': 'string', 'content': 'string'}},
                            {'name': 'read_file_tool', 'params': {'path': 'string'}}
                        ]
                    }]
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

            task_type_str = task_type.value if hasattr(task_type, 'value') else str(task_type)
            planner_kwargs = {
                'task_description': abstracted_task,
                'task_type': task_type_str,
                'available_skills': skills_json,
                'previous_outputs': outputs_json,
                'max_steps': max_steps,
                'config': {"response_format": {"type": "json_object"}}
            }

            result = self._call_with_retry(
                module=self.execution_planner,
                kwargs=planner_kwargs,
                compressible_fields=['available_skills', 'previous_outputs'],
                max_retries=self._max_compression_retries,
                lm=self._fast_lm,
            )

            logger.info(f"LLM response received")
            raw_plan = getattr(result, 'execution_plan', None)
            logger.debug(f"   Raw execution_plan type: {type(raw_plan)}")
            logger.debug(f"   Raw execution_plan (first 500 chars): {str(raw_plan)[:500] if raw_plan else 'NONE'}")

            # All normalization + parsing in one place
            steps = self._parse_plan_to_steps(raw_plan, skills, task, task_type, max_steps)

            # Determine reasoning
            if not steps:
                # Try fallback plan
                logger.warning("Execution plan resulted in 0 steps, using fallback plan")
                fallback_plan_data = self._create_fallback_plan(task, task_type, skills)
                steps = self._parse_plan_to_steps(fallback_plan_data, skills, task, task_type, max_steps)
                reasoning = f"Fallback plan created: {len(steps)} steps"
            else:
                reasoning = result.reasoning or f"Planned {len(steps)} steps"

            # Post-plan quality check: decompose composite skills for complex tasks
            decomposed = self._maybe_decompose_plan(steps, skills, task, task_type)
            if decomposed is not None:
                logger.info(f"🔀 Plan decomposed: {len(steps)} steps → {len(decomposed)} steps")
                steps = decomposed
                reasoning = f"Decomposed for quality: {reasoning}"

            used_skills = {step.skill_name for step in steps}
            if len(steps) > 0:
                logger.info(f"📋 Plan uses {len(used_skills)} skills: {used_skills}")

            logger.info(f"📝 Planned {len(steps)} execution steps")
            logger.debug(f"   Reasoning: {reasoning}")
            if hasattr(result, 'estimated_complexity'):
                logger.debug(f"   Complexity: {result.estimated_complexity}")

            return steps, reasoning

        except Exception as e:
            logger.error(f"Execution planning failed: {e}", exc_info=True)
            logger.warning("Attempting fallback plan due to execution planning failure")
            try:
                fallback_plan_data = self._create_fallback_plan(task, task_type, skills)
                logger.info(f"🔧 Fallback plan generated {len(fallback_plan_data)} steps: {fallback_plan_data}")

                if not fallback_plan_data:
                    logger.error("Fallback plan returned empty list!")
                    return [], f"Planning failed: {e}"

                steps = self._parse_plan_to_steps(fallback_plan_data, skills, task, task_type, max_steps)

                if steps:
                    logger.info(f"✅ Fallback plan created: {len(steps)} steps")
                    return steps, f"Fallback plan (planning failed: {str(e)[:100]})"
                else:
                    logger.error(f"❌ Fallback plan generated steps but 0 were converted to ExecutionStep objects")
            except Exception as fallback_e:
                logger.error(f"Fallback plan also failed: {fallback_e}", exc_info=True)

            return [], f"Planning failed: {e}"

    async def aplan_execution(
        self,
        task: str,
        task_type,
        skills: List[Dict[str, Any]],
        previous_outputs: Optional[Dict[str, Any]] = None,
        max_steps: int = 10
    ):
        """
        Async version of plan_execution using DSPy .acall().

        Non-blocking: uses _acall_with_retry (asyncio.sleep for backoff,
        module.acall for LLM calls). No thread pool needed.

        Shares ALL parsing/fallback logic with the sync version.
        """
        try:
            # Pre-processing (CPU-only, identical to sync version)
            if not skills:
                task_type_value = task_type.value if hasattr(task_type, 'value') else str(task_type)
                if task_type_value in ['creation', 'unknown']:
                    skills = [{
                        'name': 'file-operations',
                        'description': 'Create, read, write files',
                        'tools': [
                            {'name': 'write_file_tool', 'params': {'path': 'string', 'content': 'string'}},
                            {'name': 'read_file_tool', 'params': {'path': 'string'}}
                        ]
                    }]
                else:
                    return [], f"No skills available for task type '{task_type_value}'"

            # Format skills (reuse sync method — pure CPU)
            formatted_skills = self._format_skills_for_planner(skills)
            skills_json = json.dumps(formatted_skills, indent=2)
            outputs_json = json.dumps(previous_outputs or {}, indent=2)

            abstracted_task = self._abstract_task_for_planning(task)
            task_type_str = task_type.value if hasattr(task_type, 'value') else str(task_type)

            planner_kwargs = {
                'task_description': abstracted_task,
                'task_type': task_type_str,
                'available_skills': skills_json,
                'previous_outputs': outputs_json,
                'max_steps': max_steps,
                'config': {"response_format": {"type": "json_object"}}
            }

            logger.info(f"📤 Calling LLM for execution plan (async)...")

            # ── ASYNC LLM CALL ─────────────────────────────────
            result = await self._acall_with_retry(
                module=self.execution_planner,
                kwargs=planner_kwargs,
                compressible_fields=['available_skills', 'previous_outputs'],
                max_retries=self._max_compression_retries,
                lm=self._fast_lm,
            )

            logger.info(f"📥 LLM response received (async)")

            # Post-processing: parse plan (reuse sync method — pure CPU)
            raw_plan = getattr(result, 'execution_plan', None)
            steps = self._parse_plan_to_steps(raw_plan, skills, task, task_type, max_steps)

            if not steps:
                logger.warning("Async plan resulted in 0 steps, using fallback plan")
                fallback_plan_data = self._create_fallback_plan(task, task_type, skills)
                steps = self._parse_plan_to_steps(fallback_plan_data, skills, task, task_type, max_steps)
                reasoning = f"Fallback plan created: {len(steps)} steps"
            else:
                reasoning = result.reasoning or f"Planned {len(steps)} steps"

            # Post-plan quality check
            decomposed = self._maybe_decompose_plan(steps, skills, task, task_type)
            if decomposed is not None:
                logger.info(f"Plan decomposed: {len(steps)} steps -> {len(decomposed)} steps")
                steps = decomposed
                reasoning = f"Decomposed for quality: {reasoning}"

            logger.info(f"Planned {len(steps)} execution steps (async)")
            return steps, reasoning

        except Exception as e:
            logger.error(f"Async execution planning failed: {e}", exc_info=True)
            try:
                fallback_plan_data = self._create_fallback_plan(task, task_type, skills)
                steps = self._parse_plan_to_steps(fallback_plan_data, skills, task, task_type, max_steps)
                if steps:
                    return steps, f"Fallback plan (async planning failed: {str(e)[:100]})"
            except Exception as fallback_e:
                logger.error(f"Async fallback plan also failed: {fallback_e}")
            return [], f"Planning failed: {e}"

    async def areplan_with_reflection(
        self,
        task: str,
        task_type,
        skills: List[Dict[str, Any]],
        failed_steps: List[Dict[str, Any]],
        completed_outputs: Optional[Dict[str, Any]] = None,
        excluded_skills: Optional[List[str]] = None,
        max_steps: int = 5,
    ):
        """
        Async version of replan_with_reflection using DSPy .acall().

        Non-blocking: uses _acall_with_retry for the LLM call.
        """
        excluded_set = set(excluded_skills or [])
        filtered_skills = [s for s in skills if s.get('name') not in excluded_set]
        abstracted_task = self._abstract_task_for_planning(task)
        task_type_str = task_type.value if hasattr(task_type, 'value') else str(task_type)

        formatted_skills = []
        for s in filtered_skills:
            formatted_skills.append({
                'name': s.get('name', ''),
                'description': s.get('description', ''),
                'tools': s.get('tools', []),
            })
        skills_json = json.dumps(formatted_skills, indent=2)
        failed_json = json.dumps(failed_steps, default=str)
        outputs_json = json.dumps(completed_outputs or {}, default=str)
        excluded_json = json.dumps(list(excluded_set))

        try:
            # ── ASYNC LLM CALL ─────────────────────────────────
            result = await self._acall_with_retry(
                module=self.reflective_planner,
                kwargs={
                    'task_description': abstracted_task,
                    'task_type': task_type_str,
                    'available_skills': skills_json,
                    'failed_steps': failed_json,
                    'completed_outputs': outputs_json,
                    'excluded_skills': excluded_json,
                    'max_steps': max_steps,
                },
                compressible_fields=['available_skills', 'completed_outputs'],
                max_retries=self._max_compression_retries,
                lm=self._fast_lm,
            )

            raw_plan = getattr(result, 'corrected_plan', None)
            reflection = str(getattr(result, 'reflection', ''))
            reasoning = str(getattr(result, 'reasoning', ''))
            steps = self._parse_plan_to_steps(raw_plan, filtered_skills, task, task_type, max_steps)

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
        share the same formatting logic.
        """
        formatted_skills = []
        for s in skills:
            skill_name = s.get('name', '')
            skill_dict = {
                'name': skill_name,
                'description': s.get('description', ''),
                'tools': []
            }

            tools_raw = s.get('tools', [])
            if isinstance(tools_raw, dict):
                tool_names = list(tools_raw.keys())
            elif isinstance(tools_raw, list):
                tool_names = [t.get('name') if isinstance(t, dict) else t for t in tools_raw]
            else:
                tool_names = []

            try:
                from ..registry.skills_registry import get_skills_registry
                registry = get_skills_registry()
                if registry:
                    skill_obj = registry.get_skill(skill_name)
                    if skill_obj and hasattr(skill_obj, 'tools') and skill_obj.tools:
                        if not tool_names:
                            tool_names = list(skill_obj.tools.keys())
                        for tool_name in tool_names:
                            tool_func = skill_obj.tools.get(tool_name)
                            if tool_func:
                                tool_schema = self._extract_tool_schema(tool_func, tool_name)
                                skill_dict['tools'].append(tool_schema)
                            else:
                                skill_dict['tools'].append({'name': tool_name})
                    else:
                        skill_dict['tools'] = [{'name': name} for name in tool_names]
                else:
                    skill_dict['tools'] = [{'name': name} for name in tool_names]
            except Exception as e:
                logger.warning(f"Could not enrich tool schemas for {skill_name}: {e}")
                skill_dict['tools'] = [{'name': name} for name in tool_names]

            formatted_skills.append(skill_dict)

        logger.info(f"Formatted {len(formatted_skills)} skills with tool schemas for LLM")
        return formatted_skills

    def _normalize_raw_plan(
        self,
        raw_plan,
        skills: Optional[List[Dict[str, Any]]] = None,
        task: str = '',
        task_type=None,
    ) -> list:
        """
        Single, robust normalizer: convert any LLM plan output to a list of dicts.

        Handles all known LLM output formats in a single pipeline:
        1. Already a list (JSONAdapter working correctly)
        2. Direct JSON string starting with '['
        3. JSON inside markdown code block (```json ... ```)
        4. JSON array embedded in prose text
        5. Skill-name extraction from unstructured text
        6. Direct LLM retry with explicit JSON-only prompt (last resort)

        Args:
            raw_plan: Raw plan from DSPy (list, Pydantic model, string, or None)
            skills: Available skills (needed for Method 6 LLM retry)
            task: Original task description (needed for Method 6)
            task_type: Task type (needed for Method 6)

        Returns:
            List of dicts (may be empty if all parsing fails)
        """
        import re

        # Already a list — nothing to normalize
        if isinstance(raw_plan, list):
            logger.info(f"   Plan data is list: {len(raw_plan)} steps")
            return raw_plan

        if not raw_plan:
            return []

        plan_str = str(raw_plan).strip()

        # Method 1: Direct JSON parse (LLM returned clean JSON string)
        if plan_str.startswith('['):
            try:
                plan_data = json.loads(plan_str)
                logger.info(f"   Direct JSON parse successful: {len(plan_data)} steps")
                return plan_data if isinstance(plan_data, list) else [plan_data]
            except json.JSONDecodeError:
                pass

        # Method 2: Extract from markdown code block
        if '```' in plan_str:
            json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', plan_str, re.DOTALL)
            if json_match:
                try:
                    plan_data = json.loads(json_match.group(1).strip())
                    logger.info(f"   Extracted from code block: {len(plan_data)} steps")
                    return plan_data if isinstance(plan_data, list) else [plan_data]
                except json.JSONDecodeError:
                    pass

        # Method 3: Find JSON array anywhere in text
        array_match = re.search(r'\[\s*\{.*?\}\s*\]', plan_str, re.DOTALL)
        if array_match:
            try:
                plan_data = json.loads(array_match.group(0))
                logger.info(f"   Extracted JSON array from text: {len(plan_data)} steps")
                return plan_data if isinstance(plan_data, list) else [plan_data]
            except json.JSONDecodeError:
                pass

        # Method 4: Extract skill names from unstructured text
        plan_data = self._extract_plan_from_text(plan_str)
        if plan_data:
            logger.info(f"   Extracted via skill-name helper: {len(plan_data)} steps")
            return plan_data

        # Method 5: Direct LLM retry with explicit JSON-only prompt (last resort)
        if skills:
            try:
                dspy = _get_dspy()
                lm = dspy.settings.lm if dspy else None
                if lm:
                    skill_info = []
                    for s in skills:
                        tools = s.get('tools', [])
                        if isinstance(tools, list) and tools:
                            tool_name = tools[0].get('name', tools[0]) if isinstance(tools[0], dict) else tools[0]
                            skill_info.append(f"{s.get('name')}/{tool_name}")
                        else:
                            skill_info.append(s.get('name', ''))

                    task_type_str = task_type.value if hasattr(task_type, 'value') else str(task_type or 'general')
                    direct_prompt = (
                        f"Return ONLY a JSON array with 2-3 steps. Select ONLY the most relevant skills.\n\n"
                        f"Task: {task}\nTask type: {task_type_str}\n"
                        f"Available skills: {skill_info}\n\n"
                        f'Select 2-3 most relevant skills. Return JSON array:\n'
                        f'[{{"skill_name": "skill-name", "tool_name": "tool-name", "params": {{}}, '
                        f'"description": "what it does", "depends_on": [], "output_key": "step_0", "optional": false}}]\n\n'
                        f'JSON:'
                    )

                    response = lm(prompt=direct_prompt)
                    response_text = (response[0] if isinstance(response, list) else str(response)).strip()
                    logger.debug(f"   Direct LLM response (first 200): {response_text[:200]}")

                    if response_text.startswith('['):
                        plan_data = json.loads(response_text)
                        logger.info(f"   Direct LLM retry successful: {len(plan_data)} steps")
                        return plan_data if isinstance(plan_data, list) else [plan_data]
                    elif '[' in response_text:
                        start = response_text.find('[')
                        end = response_text.rfind(']') + 1
                        if end > start:
                            plan_data = json.loads(response_text[start:end])
                            logger.info(f"   Extracted from direct LLM: {len(plan_data)} steps")
                            return plan_data if isinstance(plan_data, list) else [plan_data]
            except Exception as e:
                logger.warning(f"   Direct LLM retry failed: {e}")

        logger.error(f"Could not parse plan. Raw (first 300 chars): {plan_str[:300]}")
        return []

    def _parse_plan_to_steps(
        self,
        raw_plan,
        skills: List[Dict[str, Any]],
        task: str,
        task_type=None,
        max_steps: int = 10,
    ) -> list:
        """
        Parse raw plan data into ExecutionStep objects.

        Phase 1 (normalization) is delegated to _normalize_raw_plan().
        Phase 2 converts the normalized list of dicts into ExecutionStep objects
        with fuzzy skill matching, tool inference, and param building.

        Args:
            raw_plan: Raw plan from DSPy (list of dicts/Pydantic models, or string, or None)
            skills: Available skills list
            task: Original task description
            task_type: Task type (for fallback plan, optional)
            max_steps: Maximum steps to parse

        Returns:
            List of ExecutionStep objects
        """
        # --- Phase 1: Normalize to list of dicts ---
        plan_data = self._normalize_raw_plan(raw_plan, skills=skills, task=task, task_type=task_type)

        if not plan_data:
            return []

        # --- Phase 2: Convert plan_data to ExecutionStep objects ---
        # ExecutionStep imported at module level from _execution_types
        steps = []

        # Build tool-to-skill mapping
        tool_to_skill = {}
        for s in skills:
            skill_name_map = s.get('name', '')
            for t in s.get('tools', []):
                t_name = t.get('name') if isinstance(t, dict) else t
                if t_name:
                    tool_to_skill[t_name] = skill_name_map

        available_skill_names = {s.get('name', '') for s in skills if s.get('name')}
        logger.info(f"📋 Available skills for validation: {sorted(available_skill_names)}")

        def get_val(obj, key, default=''):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        def find_matching_skill(name: str) -> str:
            """Find matching skill using exact, contains, or word-overlap match."""
            if not name:
                return ''
            name_lower = name.lower().strip()
            if name_lower in {s.lower() for s in available_skill_names}:
                for s in available_skill_names:
                    if s.lower() == name_lower:
                        return s
            for s in available_skill_names:
                if name_lower in s.lower() or s.lower() in name_lower:
                    logger.debug(f"Fuzzy matched '{name}' -> '{s}'")
                    return s
            name_words = set(name_lower.replace('-', '_').replace(' ', '_').split('_'))
            for s in available_skill_names:
                skill_words = set(s.lower().replace('-', '_').split('_'))
                if name_words & skill_words:
                    logger.debug(f"Word overlap matched '{name}' -> '{s}'")
                    return s
            return ''

        for i, step_data in enumerate(plan_data[:max_steps]):
            try:
                logger.debug(f"Processing step {i+1}: {step_data}")

                skill_name = get_val(step_data, 'skill_name') or get_val(step_data, 'skill', '')
                tool_name = get_val(step_data, 'tool_name') or get_val(step_data, 'tool', '') or get_val(step_data, 'action', '')

                # Infer skill from tool if skill is empty
                if not skill_name and tool_name:
                    skill_name = tool_to_skill.get(tool_name, '')

                # Infer skill from available skills if only one selected
                if not skill_name and len(available_skill_names) == 1:
                    skill_name = list(available_skill_names)[0]
                    logger.info(f"Auto-inferred skill_name='{skill_name}' (only one skill available)")
                elif not skill_name and len(available_skill_names) <= 3:
                    desc = str(get_val(step_data, 'description', task)).lower()
                    for candidate in available_skill_names:
                        if candidate.replace('-', ' ') in desc or any(w in desc for w in candidate.split('-')):
                            skill_name = candidate
                            logger.info(f"Inferred skill_name='{skill_name}' from description match")
                            break

                # Try fuzzy matching if exact skill not found
                if skill_name and skill_name not in available_skill_names:
                    matched = find_matching_skill(skill_name)
                    if matched:
                        logger.info(f"Skill name normalized: '{skill_name}' -> '{matched}'")
                        skill_name = matched

                # FALLBACK: Infer from description or use default
                description = get_val(step_data, 'description', f'Step {i+1}')
                if not skill_name or skill_name not in available_skill_names:
                    desc_lower = str(description).lower()

                    inferred_skill = None
                    if any(w in desc_lower for w in ['search', 'find', 'lookup', 'research', 'web', 'news', 'fetch data']):
                        inferred_skill = 'web-search'
                    elif any(w in desc_lower for w in ['create', 'write', 'generate', 'save', 'file', 'report']):
                        inferred_skill = 'file-operations'
                    elif any(w in desc_lower for w in ['chart', 'graph', 'plot', 'visualiz']):
                        inferred_skill = 'chart-creator'
                    elif any(w in desc_lower for w in ['mindmap', 'diagram', 'map']):
                        inferred_skill = 'mindmap-generator'
                    elif any(w in desc_lower for w in ['analyz', 'compar', 'evaluat']):
                        inferred_skill = 'web-search'

                    if inferred_skill and inferred_skill in available_skill_names:
                        logger.info(f"Step {i+1}: Inferred skill '{inferred_skill}' from description")
                        skill_name = inferred_skill
                    elif 'web-search' in available_skill_names:
                        skill_name = 'web-search'
                    elif 'file-operations' in available_skill_names:
                        skill_name = 'file-operations'
                    elif available_skill_names:
                        skill_name = list(available_skill_names)[0]
                    else:
                        logger.warning(f"Skipping step {i+1}: '{str(description)[:50]}' - no skills available")
                        continue

                # Infer tool_name from skill if empty
                if not tool_name:
                    for s in skills:
                        if s.get('name') == skill_name:
                            skill_tools = s.get('tools', [])
                            if skill_tools:
                                tool_names_list = [t.get('name') if isinstance(t, dict) else t for t in skill_tools]
                                desc_lower = description.lower()
                                task_lower = task.lower()

                                if skill_name == 'file-operations':
                                    if any(w in desc_lower for w in ['directory', 'folder', 'mkdir']):
                                        if 'create_directory_tool' in tool_names_list:
                                            tool_name = 'create_directory_tool'
                                    elif any(w in task_lower or w in desc_lower for w in ['create', 'write', 'generate', 'make']):
                                        if any(ext in desc_lower for ext in ['.py', '.js', '.ts', '.json', '.md', '.txt', '.html', '.css', 'file']):
                                            if 'write_file_tool' in tool_names_list:
                                                tool_name = 'write_file_tool'
                                        elif 'write_file_tool' in tool_names_list:
                                            tool_name = 'write_file_tool'
                                    elif any(w in task_lower or w in desc_lower for w in ['read', 'load', 'get']):
                                        if 'read_file_tool' in tool_names_list:
                                            tool_name = 'read_file_tool'

                                if not tool_name:
                                    first_tool = skill_tools[0]
                                    tool_name = first_tool.get('name') if isinstance(first_tool, dict) else first_tool

                                logger.debug(f"Inferred tool_name='{tool_name}' from skill '{skill_name}'")
                            break

                # Handle params
                step_params = (
                    get_val(step_data, 'params') or
                    get_val(step_data, 'parameters') or
                    get_val(step_data, 'tool_parameters') or
                    get_val(step_data, 'inputs') or
                    get_val(step_data, 'tool_input') or
                    {}
                )
                if isinstance(step_params, str):
                    step_params = {}

                # Build fallback params, then merge: LLM params override fallback,
                # but fallback fills in any required params the LLM missed.
                prev_output = f'step_{i-1}' if i > 0 else None
                param_source = task if tool_name in ['write_file_tool', 'read_file_tool'] else task
                fallback_params = self._build_skill_params(skill_name, param_source, prev_output, tool_name)

                if not step_params:
                    step_params = fallback_params
                    logger.debug(f"Built params for step {i+1}: {list(step_params.keys())}")
                else:
                    # Merge: fill missing required params from fallback
                    merged = dict(fallback_params)
                    merged.update(step_params)  # LLM params take priority
                    step_params = merged

                # Extract verification and fallback_skill (research-backed fields)
                verification = get_val(step_data, 'verification', '')
                fallback_skill = get_val(step_data, 'fallback_skill', '')

                step = ExecutionStep(
                    skill_name=skill_name,
                    tool_name=tool_name,
                    params=step_params,
                    description=description,
                    depends_on=get_val(step_data, 'depends_on', []),
                    output_key=get_val(step_data, 'output_key', f'step_{i}'),
                    optional=get_val(step_data, 'optional', False),
                    verification=verification,
                    fallback_skill=fallback_skill,
                )
                steps.append(step)
            except Exception as e:
                logger.warning(f"Failed to create step {i+1}: {e}")
                continue

        return steps

    def replan_with_reflection(
        self,
        task: str,
        task_type,
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
        filtered_skills = [s for s in skills if s.get('name') not in excluded_set]

        # Format inputs for reflective planner
        abstracted_task = self._abstract_task_for_planning(task)
        task_type_str = task_type.value if hasattr(task_type, 'value') else str(task_type)

        formatted_skills = []
        for s in filtered_skills:
            formatted_skills.append({
                'name': s.get('name', ''),
                'description': s.get('description', ''),
                'tools': s.get('tools', []),
            })
        skills_json = json.dumps(formatted_skills, indent=2)
        failed_json = json.dumps(failed_steps, default=str)
        outputs_json = json.dumps(completed_outputs or {}, default=str)
        excluded_json = json.dumps(list(excluded_set))

        try:
            result = self._call_with_retry(
                module=self.reflective_planner,
                kwargs={
                    'task_description': abstracted_task,
                    'task_type': task_type_str,
                    'available_skills': skills_json,
                    'failed_steps': failed_json,
                    'completed_outputs': outputs_json,
                    'excluded_skills': excluded_json,
                    'max_steps': max_steps,
                },
                compressible_fields=['available_skills', 'completed_outputs'],
                max_retries=self._max_compression_retries,
                lm=self._fast_lm,  # Use fast LM for replanning (routing task)
            )

            raw_plan = getattr(result, 'corrected_plan', None)
            reflection = str(getattr(result, 'reflection', ''))
            reasoning = str(getattr(result, 'reasoning', ''))

            logger.info(f"🔄 Reflective replanning: reflection='{reflection[:100]}...'")

            steps = self._parse_plan_to_steps(raw_plan, filtered_skills, task, task_type, max_steps)

            if steps:
                logger.info(f"🔄 Reflective replan produced {len(steps)} new steps")
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

    # =========================================================================
    # POST-PLAN QUALITY CHECK: Decompose composite skills for complex tasks
    # =========================================================================

    def _maybe_decompose_plan(
        self,
        steps: list,
        skills: List[Dict[str, Any]],
        task: str,
        task_type,
    ) -> Optional[list]:
        """
        Check if plan quality can be improved by decomposing composite skills.

        For comparison/research tasks with 1-step composite plans, decompose
        into granular steps for better quality (separate searches per entity,
        dedicated synthesis, dedicated formatting).

        Returns:
            Decomposed steps list, or None if no decomposition needed.
        """
        if not steps or len(steps) > 2:
            # Multi-step plans are already decomposed
            return None

        # Clean task: strip injected learning context before entity extraction
        clean_task = self._clean_task_for_query(task) if hasattr(self, '_clean_task_for_query') else task
        task_lower = clean_task.lower()

        # Detect comparison tasks
        comparison_markers = ['vs', 'versus', 'compare', 'comparison', 'difference between', 'vs.']
        is_comparison = any(m in task_lower for m in comparison_markers)

        # Detect deep research tasks (multi-entity or requiring depth)
        research_markers = ['research on', 'deep dive', 'comprehensive', 'detailed analysis']
        is_deep_research = any(m in task_lower for m in research_markers)

        if not is_comparison and not is_deep_research:
            return None

        # Check if current plan uses a composite skill
        composite_skills = {'search-summarize-pdf-telegram', 'search-summarize-pdf-telegram-v2',
                           'content-research-writer', 'content-pipeline'}
        uses_composite = any(s.skill_name in composite_skills for s in steps)

        if not uses_composite and len(steps) >= 2:
            return None  # Already granular enough

        # Extract entities from comparison task (e.g., "Paytm vs PhonePe")
        entities = self._extract_comparison_entities(clean_task)
        if not entities:
            entities = [task]  # Fallback: treat whole task as one entity

        # Build available skill names — check FULL registry, not just selected skills
        available = {s.get('name', ''): s for s in skills}
        try:
            from ..registry.skills_registry import get_skills_registry
            registry = get_skills_registry()
            if registry:
                for sname, sdef in registry.loaded_skills.items():
                    if sname not in available:
                        available[sname] = {'name': sname}
        except Exception:
            pass

        # Build decomposed plan
        # ExecutionStep imported at module level from _execution_types
        decomposed = []

        # Detect delivery channels
        delivery_skills = []
        task_wants_pdf = any(w in task_lower for w in ['pdf', 'report', 'document'])
        task_wants_telegram = 'telegram' in task_lower
        task_wants_slack = 'slack' in task_lower

        # Extract the topic/context from the task (e.g., "payment gateway" from "Paytm vs PhonePe payment gateway")
        # Strategy: look for domain terms that appear near the entities
        import re as _re
        stop_words = {
            'vs', 'vs.', 'versus', 'compare', 'comparison', 'research', 'create',
            'generate', 'send', 'via', 'telegram', 'slack', 'pdf', 'report',
            'and', 'the', 'a', 'an', 'on', 'for', 'with', 'difference', 'between',
            'it', 'its', 'to', 'of', 'in', 'is', 'be', 'document', 'make', 'build',
        }
        # All words from entity names (lowercased)
        entity_words = set()
        for e in entities:
            for w in e.lower().split():
                entity_words.add(w)

        topic_words = []
        for w in clean_task.lower().split():
            w_clean = w.strip('.,;!?')
            if len(w_clean) <= 2:
                continue
            if w_clean in stop_words:
                continue
            if w_clean in entity_words:
                continue
            topic_words.append(w_clean)
        topic_context = ' '.join(topic_words[:3])  # e.g., "payment gateway"

        # If one entity is longer (contains domain context like "payment gateway"),
        # extract the shared domain words to enrich shorter entities
        domain_context = ''
        if len(entities) >= 2:
            # Find the longest entity — it likely has the domain context
            longest = max(entities, key=len)
            shortest = min(entities, key=len)
            if len(longest.split()) > len(shortest.split()):
                # Extract domain words from longest that aren't in the shortest
                shortest_words = {w.lower() for w in shortest.split()}
                domain_words = [w for w in longest.split() if w.lower() not in shortest_words
                               and w.lower() not in {e.split()[0].lower() for e in entities}]
                domain_context = ' '.join(domain_words)

        # If no domain context from entities, use topic_context from task
        if not domain_context:
            domain_context = topic_context

        # Step(s): Research each entity separately
        for i, entity in enumerate(entities[:4]):  # Max 4 entities
            entity_clean = entity.strip()
            # Enrich short entities with domain context
            if domain_context and domain_context.lower() not in entity_clean.lower():
                search_query = f'{entity_clean} {domain_context}'.strip()
            else:
                search_query = entity_clean
            search_params = {
                'query': search_query,
                'max_results': 5,
            }
            decomposed.append(ExecutionStep(
                skill_name='web-search',
                tool_name='search_web_tool',
                params=search_params,
                description=f'Research: {search_query}',
                output_key=f'research_{i}',
                depends_on=[],
            ))

        # Step: Synthesize comparison using LLM
        # Build content reference from previous steps
        research_refs = ' '.join(
            f'${{research_{i}.results}}' for i in range(len(entities[:4]))
        )
        entity_names = ' vs '.join(e.strip() for e in entities[:4])

        # Prefer claude-cli-llm for synthesis (more control), fallback to summarize
        if 'claude-cli-llm' in available:
            synth_skill = 'claude-cli-llm'
            synth_tool = 'generate_text_tool'
        elif 'summarize' in available:
            synth_skill = 'summarize'
            synth_tool = 'summarize_text_tool'
        else:
            synth_skill = 'claude-cli-llm'
            synth_tool = 'generate_text_tool'

        decomposed.append(ExecutionStep(
            skill_name=synth_skill,
            tool_name=synth_tool,
            params={
                'prompt': (
                    f'Create a detailed, structured comparison of {entity_names}. '
                    f'Format as a professional markdown report with these sections:\n'
                    f'# {entity_names} Comparison Report\n'
                    f'## Executive Summary\n'
                    f'## Feature Comparison\n'
                    f'| Feature | {" | ".join(e.strip() for e in entities[:4])} |\n'
                    f'## Pricing & Plans\n'
                    f'## Pros and Cons\n'
                    f'## Recommendation\n\n'
                    f'Use the following research data:\n{research_refs}'
                ),
            },
            description=f'Synthesize structured comparison: {entity_names}',
            output_key='synthesis',
            depends_on=list(range(len(entities[:4]))),
        ))

        # Step: Generate PDF if requested
        if task_wants_pdf or task_wants_telegram:
            pdf_skill = 'simple-pdf-generator' if 'simple-pdf-generator' in available else 'document-converter'
            pdf_tool = 'generate_pdf_tool' if pdf_skill == 'simple-pdf-generator' else 'convert_to_pdf_tool'
            # Content comes from synthesis — handle both generate_text_tool (text) and summarize_text_tool (summary)
            content_ref = '${synthesis.text}' if synth_skill == 'claude-cli-llm' else '${synthesis.summary}'
            decomposed.append(ExecutionStep(
                skill_name=pdf_skill,
                tool_name=pdf_tool,
                params={
                    'content': content_ref,
                    'title': f'{entity_names} Comparison Report',
                    'topic': entity_names,
                },
                description=f'Generate PDF report: {entity_names}',
                output_key='pdf_output',
                depends_on=[len(entities[:4])],  # Depends on synthesis step
            ))

        # Step: Send via Telegram if requested
        if task_wants_telegram and 'telegram-sender' in available:
            decomposed.append(ExecutionStep(
                skill_name='telegram-sender',
                tool_name='send_telegram_file_tool',
                params={
                    'file_path': '${pdf_output.pdf_path}',
                    'caption': f'📊 {entity_names} Comparison Report',
                },
                description=f'Send comparison report via Telegram',
                output_key='telegram_send',
                depends_on=[len(decomposed) - 1],
                optional=True,
            ))

        # Step: Send via Slack if requested
        if task_wants_slack and 'slack' in available:
            decomposed.append(ExecutionStep(
                skill_name='slack',
                tool_name='send_slack_message_tool',
                params={
                    'file_path': '${pdf_output.pdf_path}',
                    'message': f'📊 {entity_names} Comparison Report',
                },
                description=f'Send comparison report via Slack',
                output_key='slack_send',
                depends_on=[len(decomposed) - 1],
                optional=True,
            ))

        logger.info(f"🔀 Decomposed {len(steps)}-step composite plan → {len(decomposed)} granular steps")
        for i, step in enumerate(decomposed):
            logger.info(f"   Step {i+1}: {step.skill_name}/{step.tool_name} → {step.output_key}")

        return decomposed

    def _extract_comparison_entities(self, task: str) -> List[str]:
        """
        Extract entities being compared from task description.

        Examples:
            "Compare Paytm vs PhonePe" -> ["Paytm", "PhonePe"]
            "Research Paytm vs PhonePe vs Razorpay" -> ["Paytm", "PhonePe", "Razorpay"]
            "difference between React and Vue" -> ["React", "Vue"]
        """
        import re

        # Normalize separators
        task_clean = task

        # Pattern 1: "X vs Y vs Z" or "X vs. Y"
        vs_match = re.split(r'\b(?:vs\.?|versus)\b', task_clean, flags=re.IGNORECASE)
        if len(vs_match) >= 2:
            entities = []
            for part in vs_match:
                # Clean each part: remove common prefixes/suffixes
                part = re.sub(r'^\s*(compare|research|analyze|research on|create|generate|send|make|build)\s+', '', part, flags=re.IGNORECASE)
                part = re.sub(r'\s*(comparison|compare|report|pdf|document|analysis|review|overview)\b.*$', '', part, flags=re.IGNORECASE)
                part = re.sub(r'\s*(via telegram|via slack|and send|and create|,.*$)\s*$', '', part, flags=re.IGNORECASE)
                part = part.strip().rstrip(',. ')
                if part and len(part) > 1:
                    entities.append(part)
            if len(entities) >= 2:
                return entities

        # Pattern 2: "difference between X and Y"
        between_match = re.search(
            r'(?:difference|comparison)\s+between\s+(.+?)\s+and\s+(.+?)(?:\s*[,.]|\s+(?:and|create|generate|send|via))',
            task_clean, flags=re.IGNORECASE
        )
        if between_match:
            return [between_match.group(1).strip(), between_match.group(2).strip()]

        # Pattern 3: "compare X and Y"
        compare_match = re.search(
            r'compare\s+(.+?)\s+and\s+(.+?)(?:\s*[,.]|\s+(?:create|generate|send|via)|$)',
            task_clean, flags=re.IGNORECASE
        )
        if compare_match:
            return [compare_match.group(1).strip(), compare_match.group(2).strip()]

        return []


@dataclass
class ExecutionPlan:
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

def create_agentic_planner() -> AgenticPlanner:
    """Create a new agentic planner instance."""
    return AgenticPlanner()
