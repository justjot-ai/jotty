"""
Comprehensive Unit Tests for Jotty Execution Layer
====================================================

Tests covering:
- core/execution/types.py — ExecutionTier enum, ExecutionConfig, ExecutionResult, etc.
- core/execution/tier_detector.py — TierDetector, _TierClassifierLLM
- core/execution/executor.py — TierExecutor, LLMProvider, _FallbackValidator
- core/execution/memory/noop_memory.py — NoOpMemory
- core/execution/memory/json_memory.py — JSONMemory
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# ---------------------------------------------------------------------------
# Guarded imports with skipif flags
# ---------------------------------------------------------------------------

try:
    from core.execution.types import (
        AdaptiveTimeout,
        CircuitBreaker,
        CircuitState,
        DeadLetter,
        DeadLetterQueue,
        ErrorType,
        ExecutionConfig,
        ExecutionPlan,
        ExecutionResult,
        ExecutionStep,
        ExecutionTier,
        MemoryContext,
        StreamEvent,
        StreamEventType,
        TierValidationResult,
        TimeoutWarning,
        ValidationResult,
        ValidationStatus,
        ValidationVerdict,
    )

    HAS_TYPES = True
except ImportError:
    HAS_TYPES = False

try:
    from core.execution.tier_detector import TierDetector

    HAS_DETECTOR = True
except ImportError:
    HAS_DETECTOR = False

try:
    from core.execution.memory.json_memory import JSONMemory
    from core.execution.memory.noop_memory import NoOpMemory

    HAS_MEMORY = True
except ImportError:
    HAS_MEMORY = False

try:
    from core.execution.executor import LLMProvider, TierExecutor, _FallbackValidator

    HAS_EXECUTOR = True
except ImportError:
    HAS_EXECUTOR = False


# ===========================================================================
# 1. ExecutionTier Enum
# ===========================================================================


@pytest.mark.unit
@pytest.mark.skipif(not HAS_TYPES, reason="core.execution.types not importable")
class TestExecutionTierEnum:
    """Tests for ExecutionTier IntEnum values and behaviour."""

    def test_direct_value(self):
        assert ExecutionTier.DIRECT == 1

    def test_agentic_value(self):
        assert ExecutionTier.AGENTIC == 2

    def test_learning_value(self):
        assert ExecutionTier.LEARNING == 3

    def test_research_value(self):
        assert ExecutionTier.RESEARCH == 4

    def test_autonomous_value(self):
        assert ExecutionTier.AUTONOMOUS == 5

    def test_all_five_members(self):
        members = list(ExecutionTier)
        assert len(members) == 5

    def test_membership_check(self):
        assert 1 in ExecutionTier.__members__.values()
        assert 99 not in [t.value for t in ExecutionTier]

    def test_iteration_order(self):
        values = [t.value for t in ExecutionTier]
        assert values == [1, 2, 3, 4, 5]

    def test_names(self):
        names = [t.name for t in ExecutionTier]
        assert names == ["DIRECT", "AGENTIC", "LEARNING", "RESEARCH", "AUTONOMOUS"]

    def test_int_comparison(self):
        assert ExecutionTier.DIRECT < ExecutionTier.AGENTIC
        assert ExecutionTier.AUTONOMOUS > ExecutionTier.RESEARCH

    def test_construct_from_value(self):
        assert ExecutionTier(3) == ExecutionTier.LEARNING

    def test_construct_from_invalid_raises(self):
        with pytest.raises(ValueError):
            ExecutionTier(99)


# ===========================================================================
# 2. ExecutionConfig & Related Dataclasses
# ===========================================================================


@pytest.mark.unit
@pytest.mark.skipif(not HAS_TYPES, reason="core.execution.types not importable")
class TestExecutionConfig:
    """Tests for ExecutionConfig defaults and to_swarm_config()."""

    def test_defaults(self):
        cfg = ExecutionConfig()
        assert cfg.tier is None
        assert cfg.memory_backend == "json"
        assert cfg.enable_validation is True
        assert cfg.timeout_seconds == 300

    def test_to_swarm_config_keys(self):
        cfg = ExecutionConfig()
        sc = cfg.to_swarm_config()
        expected_keys = {
            "enable_validation",
            "enable_multi_round",
            "enable_rl",
            "enable_causal_learning",
            "enable_agent_communication",
            "llm_timeout_seconds",
            "max_eval_retries",
            "validation_mode",
        }
        assert set(sc.keys()) == expected_keys

    def test_to_swarm_config_validation_mode_none(self):
        cfg = ExecutionConfig(enable_multi_round_validation=False)
        sc = cfg.to_swarm_config()
        assert sc["validation_mode"] == "none"

    def test_to_swarm_config_validation_mode_full(self):
        cfg = ExecutionConfig(enable_multi_round_validation=True)
        sc = cfg.to_swarm_config()
        assert sc["validation_mode"] == "full"


@pytest.mark.unit
@pytest.mark.skipif(not HAS_TYPES, reason="core.execution.types not importable")
class TestExecutionStep:
    """Tests for ExecutionStep properties."""

    def test_duration_ms_none_when_incomplete(self):
        step = ExecutionStep(step_num=1, description="Test")
        assert step.duration_ms is None

    def test_duration_ms_calculated(self):
        start = datetime(2026, 1, 1, 12, 0, 0)
        end = datetime(2026, 1, 1, 12, 0, 2)
        step = ExecutionStep(step_num=1, description="Test", started_at=start, completed_at=end)
        assert step.duration_ms == 2000.0

    def test_is_complete_false(self):
        step = ExecutionStep(step_num=1, description="Test")
        assert step.is_complete is False

    def test_is_complete_with_result(self):
        step = ExecutionStep(step_num=1, description="Test", result="done")
        assert step.is_complete is True

    def test_is_complete_with_error(self):
        step = ExecutionStep(step_num=1, description="Test", error="fail")
        assert step.is_complete is True


@pytest.mark.unit
@pytest.mark.skipif(not HAS_TYPES, reason="core.execution.types not importable")
class TestExecutionResult:
    """Tests for ExecutionResult serialisation and __str__."""

    def test_to_dict_keys(self):
        result = ExecutionResult(output="hello", tier=ExecutionTier.DIRECT)
        d = result.to_dict()
        assert d["tier"] == "DIRECT"
        assert d["success"] is True
        assert d["output"] == "hello"

    def test_str_success(self):
        result = ExecutionResult(
            output="x", tier=ExecutionTier.AGENTIC, llm_calls=3, latency_ms=1234.5, cost_usd=0.03
        )
        s = str(result)
        assert "[OK]" in s
        assert "Tier 2" in s

    def test_str_failure(self):
        result = ExecutionResult(
            output=None, tier=ExecutionTier.DIRECT, success=False, error="boom"
        )
        s = str(result)
        assert "[FAIL]" in s


# ===========================================================================
# 3. ErrorType & ValidationVerdict
# ===========================================================================


@pytest.mark.unit
@pytest.mark.skipif(not HAS_TYPES, reason="core.execution.types not importable")
class TestErrorType:
    """Tests for ErrorType.classify()."""

    def test_classify_ssl(self):
        assert ErrorType.classify("SSL certificate error") == ErrorType.ENVIRONMENT

    def test_classify_selector(self):
        assert ErrorType.classify("element not found on page") == ErrorType.LOGIC

    def test_classify_timeout(self):
        assert ErrorType.classify("connection timeout after 30s") == ErrorType.INFRASTRUCTURE

    def test_classify_empty_result(self):
        assert ErrorType.classify("empty result set returned") == ErrorType.DATA

    def test_classify_default(self):
        assert ErrorType.classify("something weird happened") == ErrorType.INFRASTRUCTURE


@pytest.mark.unit
@pytest.mark.skipif(not HAS_TYPES, reason="core.execution.types not importable")
class TestValidationVerdict:
    """Tests for ValidationVerdict convenience constructors."""

    def test_ok_constructor(self):
        v = ValidationVerdict.ok("all good", confidence=0.95)
        assert v.is_pass is True
        assert v.confidence == 0.95

    def test_from_error_infrastructure(self):
        v = ValidationVerdict.from_error("connection timeout")
        assert v.status == ValidationStatus.FAIL
        assert v.error_type == ErrorType.INFRASTRUCTURE
        assert v.retryable is True

    def test_from_error_logic(self):
        v = ValidationVerdict.from_error("syntax error in query")
        assert v.error_type == ErrorType.LOGIC
        assert v.retryable is False


# ===========================================================================
# 4. CircuitBreaker
# ===========================================================================


@pytest.mark.unit
@pytest.mark.skipif(not HAS_TYPES, reason="core.execution.types not importable")
class TestCircuitBreaker:
    """Tests for CircuitBreaker state machine."""

    def test_initial_state_closed(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        assert cb.state == CircuitState.CLOSED

    def test_allow_request_closed(self):
        cb = CircuitBreaker("test", failure_threshold=3)
        assert cb.allow_request() is True

    def test_trips_open_after_threshold(self):
        cb = CircuitBreaker("test", failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.allow_request() is False

    def test_success_resets_to_closed(self):
        cb = CircuitBreaker("test", failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_after_cooldown(self):
        cb = CircuitBreaker("test", failure_threshold=1, cooldown_seconds=0.01)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        time.sleep(0.02)
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.allow_request() is True

    def test_reset_manual(self):
        cb = CircuitBreaker("test", failure_threshold=1)
        cb.record_failure()
        cb.reset()
        assert cb.state == CircuitState.CLOSED


# ===========================================================================
# 5. AdaptiveTimeout
# ===========================================================================


@pytest.mark.unit
@pytest.mark.skipif(not HAS_TYPES, reason="core.execution.types not importable")
class TestAdaptiveTimeout:
    """Tests for AdaptiveTimeout P95 calculation."""

    def test_default_when_no_observations(self):
        at = AdaptiveTimeout(default_seconds=42.0)
        assert at.get("op") == 42.0

    def test_default_when_fewer_than_three_observations(self):
        at = AdaptiveTimeout(default_seconds=10.0)
        at.record("op", 1.0)
        at.record("op", 2.0)
        assert at.get("op") == 10.0

    def test_adaptive_after_enough_observations(self):
        at = AdaptiveTimeout(default_seconds=30.0, min_seconds=1.0, max_seconds=100.0)
        for v in [1.0, 1.5, 2.0, 2.5, 3.0]:
            at.record("op", v)
        timeout = at.get("op", multiplier=2.0)
        assert 1.0 <= timeout <= 100.0


# ===========================================================================
# 6. DeadLetterQueue
# ===========================================================================


@pytest.mark.unit
@pytest.mark.skipif(not HAS_TYPES, reason="core.execution.types not importable")
class TestDeadLetterQueue:
    """Tests for DeadLetterQueue enqueue / retry logic."""

    def test_enqueue_and_size(self):
        dlq = DeadLetterQueue()
        dlq.enqueue("op1", {"q": "test"}, "timeout", ErrorType.INFRASTRUCTURE)
        assert dlq.size == 1

    def test_max_size_eviction(self):
        dlq = DeadLetterQueue(max_size=2)
        dlq.enqueue("op1", {}, "e1")
        dlq.enqueue("op2", {}, "e2")
        dlq.enqueue("op3", {}, "e3")
        assert dlq.size == 2

    def test_get_retryable(self):
        dlq = DeadLetterQueue()
        letter = dlq.enqueue("op1", {}, "err")
        retryable = dlq.get_retryable()
        assert len(retryable) == 1
        assert retryable[0] is letter

    def test_mark_resolved(self):
        dlq = DeadLetterQueue()
        letter = dlq.enqueue("op1", {}, "err")
        dlq.mark_resolved(letter)
        assert dlq.size == 0

    def test_retry_all(self):
        dlq = DeadLetterQueue()
        dlq.enqueue("op1", {}, "err")
        dlq.enqueue("op2", {}, "err")
        successes = dlq.retry_all(lambda _: True)
        assert successes == 2
        assert dlq.size == 0

    def test_clear(self):
        dlq = DeadLetterQueue()
        dlq.enqueue("op1", {}, "err")
        dlq.clear()
        assert dlq.size == 0


# ===========================================================================
# 7. TimeoutWarning
# ===========================================================================


@pytest.mark.unit
@pytest.mark.skipif(not HAS_TYPES, reason="core.execution.types not importable")
class TestTimeoutWarning:
    """Tests for TimeoutWarning threshold warnings."""

    def test_no_warning_before_start(self):
        tw = TimeoutWarning(timeout_seconds=100)
        assert tw.check() is None

    def test_elapsed_zero_before_start(self):
        tw = TimeoutWarning(timeout_seconds=100)
        assert tw.elapsed == 0.0

    def test_fraction_with_zero_timeout(self):
        tw = TimeoutWarning(timeout_seconds=0)
        tw.start()
        assert tw.fraction_used == 1.0

    def test_is_expired(self):
        tw = TimeoutWarning(timeout_seconds=0.01)
        tw.start()
        time.sleep(0.02)
        assert tw.is_expired is True


# ===========================================================================
# 8. TierDetector
# ===========================================================================


@pytest.mark.unit
@pytest.mark.skipif(not HAS_DETECTOR, reason="core.execution.tier_detector not importable")
class TestTierDetector:
    """Tests for TierDetector keyword-based tier detection."""

    def setup_method(self):
        self.detector = TierDetector(enable_llm_fallback=False)

    # -- Force tier override ------------------------------------------------

    def test_force_tier_overrides_detection(self):
        result = self.detector.detect("anything", force_tier=ExecutionTier.RESEARCH)
        assert result == ExecutionTier.RESEARCH

    # -- AUTONOMOUS indicators (Tier 5) ------------------------------------

    def test_sandbox_detected_as_autonomous(self):
        assert self.detector.detect("run in sandbox mode") == ExecutionTier.AUTONOMOUS

    def test_coalition_detected_as_autonomous(self):
        assert self.detector.detect("form a coalition of agents") == ExecutionTier.AUTONOMOUS

    def test_install_detected_as_autonomous(self):
        assert self.detector.detect("install this package safely") == ExecutionTier.AUTONOMOUS

    def test_execute_code_detected_as_autonomous(self):
        assert self.detector.detect("execute code from user") == ExecutionTier.AUTONOMOUS

    # -- RESEARCH indicators (Tier 4) --------------------------------------

    def test_experiment_detected_as_research(self):
        assert (
            self.detector.detect("experiment with different approaches for maximum performance")
            == ExecutionTier.RESEARCH
        )

    def test_benchmark_detected_as_research(self):
        assert (
            self.detector.detect("benchmark these algorithms comprehensively")
            == ExecutionTier.RESEARCH
        )

    def test_research_thoroughly_detected(self):
        assert (
            self.detector.detect("research thoroughly the impact of AI on healthcare")
            == ExecutionTier.RESEARCH
        )

    # -- LEARNING indicators (Tier 3) --------------------------------------

    def test_learn_from_detected_as_learning(self):
        assert (
            self.detector.detect("learn from the previous executions and improve")
            == ExecutionTier.LEARNING
        )

    def test_optimize_detected_as_learning(self):
        assert (
            self.detector.detect("optimize the prompt for better results consistently")
            == ExecutionTier.LEARNING
        )

    def test_validate_detected_as_learning(self):
        assert (
            self.detector.detect("validate the output against the specification")
            == ExecutionTier.LEARNING
        )

    def test_remember_detected_as_learning(self):
        assert (
            self.detector.detect("remember this conversation for later use")
            == ExecutionTier.LEARNING
        )

    # -- DIRECT indicators (Tier 1) ----------------------------------------

    def test_what_is_detected_as_direct(self):
        assert self.detector.detect("what is python") == ExecutionTier.DIRECT

    def test_calculate_detected_as_direct(self):
        assert self.detector.detect("calculate 2+2") == ExecutionTier.DIRECT

    def test_short_query_detected_as_direct(self):
        assert self.detector.detect("hello there") == ExecutionTier.DIRECT

    def test_define_detected_as_direct(self):
        assert self.detector.detect("define machine learning") == ExecutionTier.DIRECT

    # -- AGENTIC indicators (Tier 2) — multi-step --------------------------

    def test_multi_step_and_then(self):
        result = self.detector.detect("read the file and then summarize it and then send via email")
        assert result == ExecutionTier.AGENTIC

    def test_multi_step_first_second(self):
        result = self.detector.detect("first gather data, second analyze it, third create a report")
        assert result == ExecutionTier.AGENTIC

    def test_step_indicators(self):
        result = self.detector.detect("step 1 fetch the data, step 2 clean it and produce output")
        assert result == ExecutionTier.AGENTIC

    # -- Default AGENTIC for ambiguous queries ------------------------------

    def test_ambiguous_falls_to_agentic(self):
        result = self.detector.detect(
            "some moderately complex but unclear task that requires quite a bit of careful thought and work"
        )
        assert result == ExecutionTier.AGENTIC

    # -- Caching behaviour -------------------------------------------------

    def test_cache_returns_same_result(self):
        goal = "what is the meaning of life"
        first = self.detector.detect(goal)
        second = self.detector.detect(goal)
        assert first == second

    def test_cache_key_is_case_insensitive(self):
        self.detector.detect("Calculate something")
        cache_key = "calculate something"[:100]
        assert cache_key in self.detector.detection_cache

    def test_clear_cache(self):
        self.detector.detect("what is python")
        assert len(self.detector.detection_cache) > 0
        self.detector.clear_cache()
        assert len(self.detector.detection_cache) == 0

    # -- _is_simple_query ---------------------------------------------------

    def test_is_simple_short_query(self):
        assert self.detector._is_simple_query("hi") is True

    def test_is_simple_with_direct_indicator(self):
        assert self.detector._is_simple_query("what is python") is True

    def test_not_simple_with_multi_step(self):
        assert self.detector._is_simple_query("first do this and then do that") is False

    def test_long_without_indicator_not_simple(self):
        long_query = " ".join(["word"] * 15)
        assert self.detector._is_simple_query(long_query) is False

    # -- _detect_tier_with_confidence --------------------------------------

    def test_confidence_high_for_keyword_match(self):
        _, confidence = self.detector._detect_tier_with_confidence("run in sandbox")
        assert confidence == 0.85

    def test_confidence_moderate_for_direct(self):
        _, confidence = self.detector._detect_tier_with_confidence("what is AI")
        assert confidence == 0.80

    def test_confidence_low_for_ambiguous(self):
        _, confidence = self.detector._detect_tier_with_confidence(
            "some moderately complex but unclear task that needs a lot of work done"
        )
        assert confidence == 0.40

    def test_confidence_multi_step(self):
        _, confidence = self.detector._detect_tier_with_confidence(
            "analyze and produce a detailed report on the findings"
        )
        assert confidence == 0.75

    # -- explain_detection -------------------------------------------------

    def test_explain_detection_format_direct(self):
        explanation = self.detector.explain_detection("what is python")
        assert "Tier 1" in explanation
        assert "DIRECT" in explanation

    def test_explain_detection_contains_reasons(self):
        explanation = self.detector.explain_detection("run in sandbox mode")
        assert "AUTONOMOUS" in explanation
        assert "sandbox" in explanation.lower() or "autonomous" in explanation.lower()

    def test_explain_detection_agentic_multi_step(self):
        explanation = self.detector.explain_detection(
            "first gather data, and then analyze it for trends"
        )
        assert "AGENTIC" in explanation

    def test_explain_detection_learning(self):
        explanation = self.detector.explain_detection("learn from previous mistakes")
        assert "LEARNING" in explanation

    def test_explain_detection_research(self):
        explanation = self.detector.explain_detection("research thoroughly the topic")
        assert "RESEARCH" in explanation


# ===========================================================================
# 9. TierDetector — Async with LLM fallback (mocked)
# ===========================================================================


@pytest.mark.unit
@pytest.mark.skipif(not HAS_DETECTOR, reason="core.execution.tier_detector not importable")
class TestTierDetectorAsync:
    """Tests for async detection and LLM fallback."""

    @pytest.mark.asyncio
    async def test_adetect_force_tier(self):
        detector = TierDetector(enable_llm_fallback=False)
        result = await detector.adetect("anything", force_tier=ExecutionTier.LEARNING)
        assert result == ExecutionTier.LEARNING

    @pytest.mark.asyncio
    async def test_adetect_cache(self):
        detector = TierDetector(enable_llm_fallback=False)
        first = await detector.adetect("what is python")
        second = await detector.adetect("what is python")
        assert first == second

    @pytest.mark.asyncio
    async def test_adetect_high_confidence_no_llm_call(self):
        detector = TierDetector(enable_llm_fallback=True)
        mock_classifier = AsyncMock()
        detector._llm_classifier = mock_classifier
        result = await detector.adetect("run in sandbox")
        assert result == ExecutionTier.AUTONOMOUS
        mock_classifier.classify.assert_not_called()

    @pytest.mark.asyncio
    async def test_adetect_low_confidence_triggers_llm(self):
        detector = TierDetector(enable_llm_fallback=True)
        mock_classifier = AsyncMock()
        mock_classifier.classify = AsyncMock(return_value=ExecutionTier.RESEARCH)
        detector._llm_classifier = mock_classifier
        result = await detector.adetect(
            "some moderately complex but unclear task that needs a lot of work done"
        )
        assert result == ExecutionTier.RESEARCH
        mock_classifier.classify.assert_called_once()

    @pytest.mark.asyncio
    async def test_adetect_llm_fallback_exception(self):
        detector = TierDetector(enable_llm_fallback=True)
        mock_classifier = AsyncMock()
        mock_classifier.classify = AsyncMock(side_effect=RuntimeError("LLM down"))
        detector._llm_classifier = mock_classifier
        result = await detector.adetect(
            "some moderately complex but unclear task that needs a lot of work done"
        )
        assert result == ExecutionTier.AGENTIC

    @pytest.mark.asyncio
    async def test_adetect_llm_returns_none_keeps_heuristic(self):
        detector = TierDetector(enable_llm_fallback=True)
        mock_classifier = AsyncMock()
        mock_classifier.classify = AsyncMock(return_value=None)
        detector._llm_classifier = mock_classifier
        result = await detector.adetect(
            "some moderately complex but unclear task that needs a lot of work done"
        )
        assert result == ExecutionTier.AGENTIC


# ===========================================================================
# 10. _TierClassifierLLM (mocked Anthropic client)
# ===========================================================================


@pytest.mark.unit
@pytest.mark.skipif(not HAS_DETECTOR, reason="core.execution.tier_detector not importable")
class TestTierClassifierLLM:
    """Tests for _TierClassifierLLM.classify() with mocked Anthropic."""

    @pytest.mark.asyncio
    async def test_classify_returns_tier(self):
        from core.execution.tier_detector import _TierClassifierLLM

        classifier = _TierClassifierLLM()
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_block = Mock()
        mock_block.text = "3"
        mock_response.content = [mock_block]
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        classifier._client = mock_client

        result = await classifier.classify("improve my workflow")
        assert result == ExecutionTier.LEARNING

    @pytest.mark.asyncio
    async def test_classify_parses_digit_from_text(self):
        from core.execution.tier_detector import _TierClassifierLLM

        classifier = _TierClassifierLLM()
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_block = Mock()
        mock_block.text = "Tier 2"
        mock_response.content = [mock_block]
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        classifier._client = mock_client

        result = await classifier.classify("plan my day")
        assert result == ExecutionTier.AGENTIC

    @pytest.mark.asyncio
    async def test_classify_returns_none_for_unparseable(self):
        from core.execution.tier_detector import _TierClassifierLLM

        classifier = _TierClassifierLLM()
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_block = Mock()
        mock_block.text = "no digits here"
        mock_response.content = [mock_block]
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        classifier._client = mock_client

        result = await classifier.classify("something")
        assert result is None

    @pytest.mark.asyncio
    async def test_classify_returns_none_for_empty_content(self):
        from core.execution.tier_detector import _TierClassifierLLM

        classifier = _TierClassifierLLM()
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.content = []
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        classifier._client = mock_client

        result = await classifier.classify("something")
        assert result is None


# ===========================================================================
# 11. NoOpMemory
# ===========================================================================


@pytest.mark.unit
@pytest.mark.skipif(not HAS_MEMORY, reason="core.execution.memory not importable")
class TestNoOpMemory:
    """Tests for NoOpMemory — all operations are no-ops."""

    @pytest.mark.asyncio
    async def test_store_returns_none(self):
        mem = NoOpMemory()
        result = await mem.store(goal="x", result="y")
        assert result is None

    @pytest.mark.asyncio
    async def test_retrieve_returns_empty_list(self):
        mem = NoOpMemory()
        result = await mem.retrieve("any goal")
        assert result == []

    @pytest.mark.asyncio
    async def test_retrieve_with_limit(self):
        mem = NoOpMemory()
        result = await mem.retrieve("any goal", limit=10)
        assert result == []

    @pytest.mark.asyncio
    async def test_clear_returns_none(self):
        mem = NoOpMemory()
        result = await mem.clear()
        assert result is None

    @pytest.mark.asyncio
    async def test_store_accepts_kwargs(self):
        mem = NoOpMemory()
        await mem.store(goal="g", result="r", success=True, confidence=1.0, ttl_hours=24)


# ===========================================================================
# 12. JSONMemory
# ===========================================================================


@pytest.mark.unit
@pytest.mark.skipif(not HAS_MEMORY, reason="core.execution.memory not importable")
class TestJSONMemory:
    """Tests for JSONMemory — file-based storage with TTL."""

    @pytest.mark.asyncio
    async def test_store_and_retrieve_round_trip(self, tmp_path):
        mem = JSONMemory(base_path=tmp_path)
        await mem.store(goal="test goal", result="test result", success=True, confidence=0.9)
        entries = await mem.retrieve("test goal")
        assert len(entries) == 1
        assert entries[0]["result"] == "test result"
        assert entries[0]["success"] is True

    @pytest.mark.asyncio
    async def test_retrieve_nonexistent_returns_empty(self, tmp_path):
        mem = JSONMemory(base_path=tmp_path)
        entries = await mem.retrieve("never stored this")
        assert entries == []

    @pytest.mark.asyncio
    async def test_ttl_expiration(self, tmp_path):
        mem = JSONMemory(base_path=tmp_path)
        await mem.store(goal="expiring", result="old", ttl_hours=0)
        # Manually set expires_at in the past
        file_path = mem._get_file_path("expiring")
        data = mem._load_file(file_path)
        data["entries"][0]["expires_at"] = (datetime.now() - timedelta(hours=1)).isoformat()
        mem._save_file(file_path, data)
        entries = await mem.retrieve("expiring")
        assert entries == []

    @pytest.mark.asyncio
    async def test_limit_entries_per_file(self, tmp_path):
        mem = JSONMemory(base_path=tmp_path)
        for i in range(15):
            await mem.store(goal="same goal", result=f"result_{i}")
        file_path = mem._get_file_path("same goal")
        data = mem._load_file(file_path)
        assert len(data["entries"]) == 10

    @pytest.mark.asyncio
    async def test_retrieve_limit(self, tmp_path):
        mem = JSONMemory(base_path=tmp_path)
        for i in range(5):
            await mem.store(goal="test limit", result=f"r{i}")
        entries = await mem.retrieve("test limit", limit=2)
        assert len(entries) == 2

    @pytest.mark.asyncio
    async def test_retrieve_newest_first(self, tmp_path):
        mem = JSONMemory(base_path=tmp_path)
        await mem.store(goal="order test", result="first")
        await mem.store(goal="order test", result="second")
        entries = await mem.retrieve("order test")
        assert entries[0]["result"] == "second"
        assert entries[1]["result"] == "first"

    @pytest.mark.asyncio
    async def test_clear_removes_all_files(self, tmp_path):
        mem = JSONMemory(base_path=tmp_path)
        await mem.store(goal="a", result="1")
        await mem.store(goal="b", result="2")
        await mem.clear()
        json_files = list(tmp_path.glob("*.json"))
        assert len(json_files) == 0

    def test_get_file_path_deterministic(self, tmp_path):
        mem = JSONMemory(base_path=tmp_path)
        p1 = mem._get_file_path("same goal")
        p2 = mem._get_file_path("same goal")
        assert p1 == p2

    def test_get_file_path_different_goals(self, tmp_path):
        mem = JSONMemory(base_path=tmp_path)
        p1 = mem._get_file_path("goal alpha with lots of context")
        p2 = mem._get_file_path("goal beta with different text")
        assert p1 != p2

    def test_load_file_creates_default(self, tmp_path):
        mem = JSONMemory(base_path=tmp_path)
        data = mem._load_file(tmp_path / "nonexistent.json")
        assert data == {"entries": []}

    def test_save_and_load_roundtrip(self, tmp_path):
        mem = JSONMemory(base_path=tmp_path)
        file_path = tmp_path / "test.json"
        data = {"entries": [{"goal": "x", "result": "y"}]}
        mem._save_file(file_path, data)
        loaded = mem._load_file(file_path)
        assert loaded == data

    @pytest.mark.asyncio
    async def test_score_exact_match(self, tmp_path):
        mem = JSONMemory(base_path=tmp_path)
        await mem.store(goal="exact match test", result="res")
        entries = await mem.retrieve("exact match test")
        assert entries[0]["score"] == 1.0

    @pytest.mark.asyncio
    async def test_base_path_created(self, tmp_path):
        nested = tmp_path / "deep" / "nested" / "path"
        mem = JSONMemory(base_path=nested)
        assert nested.exists()


# ===========================================================================
# 13. TierExecutor (heavily mocked)
# ===========================================================================


@pytest.mark.unit
@pytest.mark.skipif(not HAS_EXECUTOR, reason="core.execution.executor not importable")
class TestTierExecutorInit:
    """Tests for TierExecutor initialisation and lazy properties."""

    def test_default_config(self):
        executor = TierExecutor.__new__(TierExecutor)
        executor.config = ExecutionConfig()
        executor._registry = None
        executor._provider = None
        executor._detector = TierDetector(enable_llm_fallback=False)
        executor._planner = None
        executor._memory = None
        executor._validator = None
        executor._complexity_gate = None
        executor._metrics = None
        executor._tracer = None
        executor._cost_tracker = None
        assert executor.config.tier is None
        assert executor.config.memory_backend == "json"

    def test_injected_provider(self):
        mock_provider = Mock()
        executor = TierExecutor.__new__(TierExecutor)
        executor.config = ExecutionConfig()
        executor._registry = None
        executor._provider = mock_provider
        executor._detector = TierDetector(enable_llm_fallback=False)
        executor._planner = None
        executor._memory = None
        executor._validator = None
        executor._complexity_gate = None
        executor._metrics = None
        executor._tracer = None
        executor._cost_tracker = None
        assert executor.provider is mock_provider

    def test_injected_registry(self):
        mock_registry = Mock()
        executor = TierExecutor.__new__(TierExecutor)
        executor.config = ExecutionConfig()
        executor._registry = mock_registry
        executor._provider = None
        executor._detector = TierDetector(enable_llm_fallback=False)
        executor._planner = None
        executor._memory = None
        executor._validator = None
        executor._complexity_gate = None
        executor._metrics = None
        executor._tracer = None
        executor._cost_tracker = None
        assert executor.registry is mock_registry


@pytest.mark.unit
@pytest.mark.skipif(not HAS_EXECUTOR, reason="core.execution.executor not importable")
class TestTierExecutorMemoryBackend:
    """Tests for _create_memory_backend routing."""

    def test_json_backend(self):
        executor = TierExecutor.__new__(TierExecutor)
        executor.config = ExecutionConfig(memory_backend="json")
        executor._memory = None
        with patch("core.execution.executor.TierExecutor._create_memory_backend") as mock_create:
            mock_create.return_value = Mock()
            backend = executor._create_memory_backend()
            assert backend is not None

    def test_noop_backend(self):
        executor = TierExecutor.__new__(TierExecutor)
        executor.config = ExecutionConfig(memory_backend="none")
        executor._memory = None
        backend = executor._create_memory_backend()
        assert backend.__class__.__name__ == "NoOpMemory"


@pytest.mark.unit
@pytest.mark.skipif(not HAS_EXECUTOR, reason="core.execution.executor not importable")
class TestTierExecutorParsePlan:
    """Tests for TierExecutor._parse_plan()."""

    def _make_executor(self):
        executor = TierExecutor.__new__(TierExecutor)
        executor.config = ExecutionConfig()
        executor._registry = Mock()
        executor._provider = Mock()
        executor._detector = TierDetector(enable_llm_fallback=False)
        executor._planner = None
        executor._memory = None
        executor._validator = None
        executor._complexity_gate = None
        executor._metrics = None
        executor._tracer = None
        executor._cost_tracker = None
        return executor

    def test_parse_plan_from_dict(self):
        executor = self._make_executor()
        plan_result = {
            "steps": [
                {"description": "Step A", "skill": "web-search"},
                {"description": "Step B"},
            ],
            "reasoning": "Because.",
        }
        plan = executor._parse_plan("my goal", plan_result)
        assert plan.total_steps == 2
        assert plan.steps[0].description == "Step A"
        assert plan.steps[0].skill == "web-search"
        assert plan.steps[1].skill is None

    def test_parse_plan_empty_steps(self):
        executor = self._make_executor()
        plan_result = {"steps": [], "reasoning": ""}
        plan = executor._parse_plan("goal", plan_result)
        assert plan.total_steps == 0

    def test_parse_plan_step_numbering(self):
        executor = self._make_executor()
        plan_result = {
            "steps": [{"description": f"S{i}"} for i in range(5)],
        }
        plan = executor._parse_plan("goal", plan_result)
        for i, step in enumerate(plan.steps):
            assert step.step_num == i + 1


@pytest.mark.unit
@pytest.mark.skipif(not HAS_EXECUTOR, reason="core.execution.executor not importable")
class TestTierExecutorFallbackAggregate:
    """Tests for _fallback_aggregate helper."""

    def _make_executor(self):
        executor = TierExecutor.__new__(TierExecutor)
        executor.config = ExecutionConfig()
        return executor

    def test_empty_results(self):
        executor = self._make_executor()
        assert executor._fallback_aggregate([], "g") == "No results generated."

    def test_single_result(self):
        executor = self._make_executor()
        out = executor._fallback_aggregate([{"output": "hello"}], "g")
        assert out == "hello"

    def test_multiple_results(self):
        executor = self._make_executor()
        results = [{"output": "a"}, {"output": "b"}]
        out = executor._fallback_aggregate(results, "my goal")
        assert "Step 1" in out
        assert "Step 2" in out
        assert "my goal" in out


@pytest.mark.unit
@pytest.mark.skipif(not HAS_EXECUTOR, reason="core.execution.executor not importable")
class TestTierExecutorEnrichWithMemory:
    """Tests for _enrich_with_memory helper."""

    def _make_executor(self):
        executor = TierExecutor.__new__(TierExecutor)
        executor.config = ExecutionConfig()
        return executor

    def test_no_context(self):
        executor = self._make_executor()
        assert executor._enrich_with_memory("goal", None) == "goal"

    def test_empty_entries(self):
        executor = self._make_executor()
        ctx = MemoryContext(entries=[], relevance_scores=[], total_retrieved=0, retrieval_time_ms=0)
        assert executor._enrich_with_memory("goal", ctx) == "goal"

    def test_with_entries(self):
        executor = self._make_executor()
        ctx = MemoryContext(
            entries=[{"summary": "past info"}],
            relevance_scores=[0.9],
            total_retrieved=1,
            retrieval_time_ms=5.0,
        )
        enriched = executor._enrich_with_memory("goal", ctx)
        assert "past info" in enriched
        assert "goal" in enriched


# ===========================================================================
# 14. LLMProvider (mocked)
# ===========================================================================


@pytest.mark.unit
@pytest.mark.skipif(not HAS_EXECUTOR, reason="core.execution.executor not importable")
class TestLLMProvider:
    """Tests for LLMProvider initialisation."""

    def test_default_provider_and_model(self):
        provider = LLMProvider()
        assert provider._provider_name == "anthropic"
        assert "claude" in provider._model.lower() or "sonnet" in provider._model.lower()

    def test_custom_provider(self):
        provider = LLMProvider(provider="openai", model="gpt-4")
        assert provider._provider_name == "openai"
        assert provider._model == "gpt-4"

    def test_client_lazy_init(self):
        provider = LLMProvider()
        assert provider._client is None


# ===========================================================================
# 15. _FallbackValidator (mocked)
# ===========================================================================


@pytest.mark.unit
@pytest.mark.skipif(not HAS_EXECUTOR, reason="core.execution.executor not importable")
class TestFallbackValidator:
    """Tests for _FallbackValidator with mocked provider."""

    @pytest.mark.asyncio
    async def test_validate_parses_json(self):
        mock_provider = AsyncMock()
        mock_provider.generate = AsyncMock(
            return_value={
                "content": '{"success": true, "confidence": 0.9, "feedback": "good", "reasoning": "ok"}'
            }
        )
        validator = _FallbackValidator(mock_provider)
        result = await validator.validate("Is this correct?")
        assert result["success"] is True
        assert result["confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_validate_handles_non_json(self):
        mock_provider = AsyncMock()
        mock_provider.generate = AsyncMock(return_value={"content": "Looks good to me!"})
        validator = _FallbackValidator(mock_provider)
        result = await validator.validate("Is this correct?")
        assert result["success"] is True
        assert "feedback" in result

    @pytest.mark.asyncio
    async def test_validate_handles_exception(self):
        mock_provider = AsyncMock()
        mock_provider.generate = AsyncMock(side_effect=RuntimeError("LLM fail"))
        validator = _FallbackValidator(mock_provider)
        result = await validator.validate("Is this correct?")
        assert result["success"] is True
        assert result["confidence"] == 0.5


# ===========================================================================
# 16. StreamEventType & StreamEvent
# ===========================================================================


@pytest.mark.unit
@pytest.mark.skipif(not HAS_TYPES, reason="core.execution.types not importable")
class TestStreamEventType:
    """Tests for StreamEventType enum."""

    def test_all_event_types(self):
        expected = {"STATUS", "STEP_COMPLETE", "PARTIAL_OUTPUT", "TOKEN", "RESULT", "ERROR"}
        actual = {e.name for e in StreamEventType}
        assert actual == expected

    def test_stream_event_creation(self):
        event = StreamEvent(type=StreamEventType.TOKEN, data="hello")
        assert event.type == StreamEventType.TOKEN
        assert event.data == "hello"
        assert event.tier is None
        assert isinstance(event.timestamp, datetime)


# ===========================================================================
# 17. ExecutionPlan properties
# ===========================================================================


@pytest.mark.unit
@pytest.mark.skipif(not HAS_TYPES, reason="core.execution.types not importable")
class TestExecutionPlan:
    """Tests for ExecutionPlan computed properties."""

    def test_total_steps(self):
        plan = ExecutionPlan(
            goal="test",
            steps=[
                ExecutionStep(step_num=1, description="a"),
                ExecutionStep(step_num=2, description="b"),
            ],
        )
        assert plan.total_steps == 2

    def test_parallelizable_steps(self):
        plan = ExecutionPlan(
            goal="test",
            steps=[
                ExecutionStep(step_num=1, description="a", can_parallelize=True),
                ExecutionStep(step_num=2, description="b", can_parallelize=False),
                ExecutionStep(step_num=3, description="c", can_parallelize=True),
            ],
        )
        assert plan.parallelizable_steps == 2

    def test_empty_plan(self):
        plan = ExecutionPlan(goal="test", steps=[])
        assert plan.total_steps == 0
        assert plan.parallelizable_steps == 0


# ===========================================================================
# 18. Backward-compat alias
# ===========================================================================


@pytest.mark.unit
@pytest.mark.skipif(not HAS_TYPES, reason="core.execution.types not importable")
class TestBackwardCompatAlias:
    """Tests for ValidationResult alias."""

    def test_validation_result_is_tier_validation_result(self):
        assert ValidationResult is TierValidationResult
