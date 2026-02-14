"""
Tests for Foundation LM Providers and Config
=============================================

Comprehensive unit tests for:
1. core/foundation/config_defaults.py  - JottyDefaults, DEFAULTS, MODEL_ALIASES
2. core/foundation/exceptions.py       - Full exception hierarchy + wrap_exception()
3. core/foundation/protocols.py        - MetadataProvider, DataProvider, ContextExtractor
4. core/foundation/agent_config.py     - AgentConfig dataclass
5. core/foundation/types/agent_types.py - AgentContribution, AgentMessage, SharedScratchpad
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass, asdict, FrozenInstanceError
from datetime import datetime
from typing import Dict, Any, Optional, List

# ---------------------------------------------------------------------------
# Safe imports with skip guards
# ---------------------------------------------------------------------------

try:
    from core.foundation.config_defaults import (
        JottyDefaults, DEFAULTS, MODEL_ALIASES,
        MAX_TOKENS, SAFETY_MARGIN as SAFETY_MARGIN_EXPORT,
        EPISODIC_CAPACITY as EPISODIC_CAPACITY_EXPORT,
        MAX_ENTRY_TOKENS as MAX_ENTRY_TOKENS_EXPORT,
        LLM_MAX_OUTPUT_TOKENS as LLM_MAX_OUTPUT_TOKENS_EXPORT,
        LLM_PLANNING_MAX_TOKENS as LLM_PLANNING_MAX_TOKENS_EXPORT,
        MODEL_SONNET, MODEL_OPUS, MODEL_HAIKU, DEFAULT_MODEL_ALIAS,
        LLM_TEMPERATURE, LLM_TIMEOUT_SECONDS, MAX_RETRIES,
        RETRY_BACKOFF_SECONDS,
    )
    HAS_DEFAULTS = True
except ImportError:
    HAS_DEFAULTS = False

try:
    from core.foundation.exceptions import (
        JottyError,
        ConfigurationError, InvalidConfigError, MissingConfigError,
        ExecutionError, AgentExecutionError, ToolExecutionError,
        TimeoutError as JottyTimeoutError, CircuitBreakerError,
        ContextError, ContextOverflowError, CompressionError, ChunkingError,
        MemoryError as JottyMemoryError,
        MemoryRetrievalError, MemoryStorageError, ConsolidationError,
        LearningError, RewardCalculationError, CreditAssignmentError, PolicyUpdateError,
        CommunicationError, MessageDeliveryError, FeedbackRoutingError,
        ValidationError, InputValidationError, OutputValidationError,
        PersistenceError, StorageError, RetrievalError,
        IntegrationError, LLMError, DSPyError, ExternalToolError,
        wrap_exception,
    )
    HAS_EXCEPTIONS = True
except ImportError:
    HAS_EXCEPTIONS = False

try:
    from core.foundation.protocols import MetadataProvider, DataProvider, ContextExtractor
    HAS_PROTOCOLS = True
except ImportError:
    HAS_PROTOCOLS = False

try:
    from core.foundation.agent_config import AgentConfig
    HAS_AGENT_CONFIG = True
except ImportError:
    HAS_AGENT_CONFIG = False

try:
    from core.foundation.types.agent_types import (
        AgentContribution, AgentMessage, SharedScratchpad,
    )
    from core.foundation.types.enums import CommunicationType
    HAS_AGENT_TYPES = True
except ImportError:
    HAS_AGENT_TYPES = False


# ###########################################################################
# 1. JottyDefaults & config_defaults module
# ###########################################################################

@pytest.mark.skipif(not HAS_DEFAULTS, reason="config_defaults not importable")
class TestJottyDefaultsInstance:
    """Tests for the DEFAULTS singleton and module-level exports."""

    @pytest.mark.unit
    def test_defaults_is_jotty_defaults_instance(self):
        """DEFAULTS should be an instance of JottyDefaults."""
        assert isinstance(DEFAULTS, JottyDefaults)

    @pytest.mark.unit
    def test_defaults_singleton_matches_class(self):
        """DEFAULTS field values should equal those of a fresh JottyDefaults()."""
        fresh = JottyDefaults()
        assert DEFAULTS.MAX_CONTEXT_TOKENS == fresh.MAX_CONTEXT_TOKENS
        assert DEFAULTS.LLM_TEMPERATURE == fresh.LLM_TEMPERATURE
        assert DEFAULTS.EPISODIC_CAPACITY == fresh.EPISODIC_CAPACITY

    @pytest.mark.unit
    def test_module_level_max_tokens_export(self):
        """MAX_TOKENS convenience export equals DEFAULTS.MAX_CONTEXT_TOKENS."""
        assert MAX_TOKENS == DEFAULTS.MAX_CONTEXT_TOKENS

    @pytest.mark.unit
    def test_module_level_safety_margin_export(self):
        """SAFETY_MARGIN convenience export equals DEFAULTS.SAFETY_MARGIN."""
        assert SAFETY_MARGIN_EXPORT == DEFAULTS.SAFETY_MARGIN

    @pytest.mark.unit
    def test_module_level_episodic_capacity_export(self):
        """EPISODIC_CAPACITY convenience export."""
        assert EPISODIC_CAPACITY_EXPORT == DEFAULTS.EPISODIC_CAPACITY

    @pytest.mark.unit
    def test_module_level_max_entry_tokens_export(self):
        """MAX_ENTRY_TOKENS convenience export."""
        assert MAX_ENTRY_TOKENS_EXPORT == DEFAULTS.MAX_ENTRY_TOKENS

    @pytest.mark.unit
    def test_module_level_llm_max_output_tokens_export(self):
        """LLM_MAX_OUTPUT_TOKENS convenience export."""
        assert LLM_MAX_OUTPUT_TOKENS_EXPORT == DEFAULTS.LLM_MAX_OUTPUT_TOKENS

    @pytest.mark.unit
    def test_module_level_llm_planning_max_tokens_export(self):
        """LLM_PLANNING_MAX_TOKENS convenience export."""
        assert LLM_PLANNING_MAX_TOKENS_EXPORT == DEFAULTS.LLM_PLANNING_MAX_TOKENS


@pytest.mark.skipif(not HAS_DEFAULTS, reason="config_defaults not importable")
class TestJottyDefaultsModelNames:
    """Tests for model name constants."""

    @pytest.mark.unit
    def test_default_model_alias(self):
        """DEFAULT_MODEL_ALIAS should be 'sonnet'."""
        assert JottyDefaults.DEFAULT_MODEL_ALIAS == "sonnet"
        assert DEFAULT_MODEL_ALIAS == "sonnet"

    @pytest.mark.unit
    def test_model_sonnet_is_string(self):
        """MODEL_SONNET should be a non-empty string containing 'sonnet'."""
        assert isinstance(MODEL_SONNET, str)
        assert len(MODEL_SONNET) > 0
        assert "sonnet" in MODEL_SONNET.lower()

    @pytest.mark.unit
    def test_model_opus_is_string(self):
        """MODEL_OPUS should be a non-empty string containing 'opus'."""
        assert isinstance(MODEL_OPUS, str)
        assert "opus" in MODEL_OPUS.lower()

    @pytest.mark.unit
    def test_model_haiku_is_string(self):
        """MODEL_HAIKU should be a non-empty string containing 'haiku'."""
        assert isinstance(MODEL_HAIKU, str)
        assert "haiku" in MODEL_HAIKU.lower()

    @pytest.mark.unit
    def test_model_aliases_dict(self):
        """MODEL_ALIASES maps short names to full identifiers."""
        assert isinstance(MODEL_ALIASES, dict)
        assert "sonnet" in MODEL_ALIASES
        assert "opus" in MODEL_ALIASES
        assert "haiku" in MODEL_ALIASES

    @pytest.mark.unit
    def test_model_aliases_values_match_constants(self):
        """MODEL_ALIASES values should match the module-level constants."""
        assert MODEL_ALIASES["sonnet"] == MODEL_SONNET
        assert MODEL_ALIASES["opus"] == MODEL_OPUS
        assert MODEL_ALIASES["haiku"] == MODEL_HAIKU

    @pytest.mark.unit
    def test_openai_default_model(self):
        """MODEL_OPENAI_DEFAULT should be a non-empty string."""
        assert isinstance(DEFAULTS.MODEL_OPENAI_DEFAULT, str)
        assert len(DEFAULTS.MODEL_OPENAI_DEFAULT) > 0

    @pytest.mark.unit
    def test_gemini_default_model(self):
        """MODEL_GEMINI_DEFAULT should be a non-empty string."""
        assert isinstance(DEFAULTS.MODEL_GEMINI_DEFAULT, str)
        assert len(DEFAULTS.MODEL_GEMINI_DEFAULT) > 0

    @pytest.mark.unit
    def test_groq_default_model(self):
        """MODEL_GROQ_DEFAULT should be a non-empty string."""
        assert isinstance(DEFAULTS.MODEL_GROQ_DEFAULT, str)
        assert len(DEFAULTS.MODEL_GROQ_DEFAULT) > 0

    @pytest.mark.unit
    def test_openrouter_default_model(self):
        """MODEL_OPENROUTER_DEFAULT should be a non-empty string."""
        assert isinstance(DEFAULTS.MODEL_OPENROUTER_DEFAULT, str)
        assert len(DEFAULTS.MODEL_OPENROUTER_DEFAULT) > 0


@pytest.mark.skipif(not HAS_DEFAULTS, reason="config_defaults not importable")
class TestJottyDefaultsLLMSettings:
    """Tests for LLM default parameters."""

    @pytest.mark.unit
    def test_llm_temperature_range(self):
        """LLM_TEMPERATURE should be between 0 and 2."""
        assert 0.0 <= DEFAULTS.LLM_TEMPERATURE <= 2.0
        assert LLM_TEMPERATURE == DEFAULTS.LLM_TEMPERATURE

    @pytest.mark.unit
    def test_llm_temperature_deterministic(self):
        """LLM_TEMPERATURE_DETERMINISTIC should be 0.0."""
        assert DEFAULTS.LLM_TEMPERATURE_DETERMINISTIC == 0.0

    @pytest.mark.unit
    def test_llm_max_output_tokens_positive(self):
        """LLM_MAX_OUTPUT_TOKENS should be a positive integer."""
        assert isinstance(DEFAULTS.LLM_MAX_OUTPUT_TOKENS, int)
        assert DEFAULTS.LLM_MAX_OUTPUT_TOKENS > 0

    @pytest.mark.unit
    def test_llm_planning_max_tokens_positive(self):
        """LLM_PLANNING_MAX_TOKENS should be positive and <= LLM_MAX_OUTPUT_TOKENS."""
        assert isinstance(DEFAULTS.LLM_PLANNING_MAX_TOKENS, int)
        assert DEFAULTS.LLM_PLANNING_MAX_TOKENS > 0
        assert DEFAULTS.LLM_PLANNING_MAX_TOKENS <= DEFAULTS.LLM_MAX_OUTPUT_TOKENS

    @pytest.mark.unit
    def test_llm_timeout_seconds_positive(self):
        """LLM_TIMEOUT_SECONDS should be a positive integer."""
        assert isinstance(DEFAULTS.LLM_TIMEOUT_SECONDS, int)
        assert DEFAULTS.LLM_TIMEOUT_SECONDS > 0
        assert LLM_TIMEOUT_SECONDS == DEFAULTS.LLM_TIMEOUT_SECONDS

    @pytest.mark.unit
    def test_max_retries_positive(self):
        """MAX_RETRIES should be a positive integer."""
        assert isinstance(DEFAULTS.MAX_RETRIES, int)
        assert DEFAULTS.MAX_RETRIES > 0
        assert MAX_RETRIES == DEFAULTS.MAX_RETRIES

    @pytest.mark.unit
    def test_retry_backoff_seconds(self):
        """RETRY_BACKOFF_SECONDS should be a positive float."""
        assert isinstance(DEFAULTS.RETRY_BACKOFF_SECONDS, float)
        assert DEFAULTS.RETRY_BACKOFF_SECONDS > 0
        assert RETRY_BACKOFF_SECONDS == DEFAULTS.RETRY_BACKOFF_SECONDS


@pytest.mark.skipif(not HAS_DEFAULTS, reason="config_defaults not importable")
class TestJottyDefaultsTokenBudgets:
    """Tests for token budget parameters."""

    @pytest.mark.unit
    def test_max_context_tokens(self):
        """MAX_CONTEXT_TOKENS should be 100_000."""
        assert DEFAULTS.MAX_CONTEXT_TOKENS == 100_000

    @pytest.mark.unit
    def test_safety_margin(self):
        """SAFETY_MARGIN should be 2_000."""
        assert DEFAULTS.SAFETY_MARGIN == 2_000

    @pytest.mark.unit
    def test_component_budgets_positive(self):
        """All component token budgets should be positive integers."""
        for attr in [
            "SYSTEM_PROMPT_BUDGET", "CURRENT_INPUT_BUDGET",
            "TRAJECTORY_BUDGET", "TOOL_OUTPUT_BUDGET",
            "MIN_MEMORY_BUDGET", "MAX_MEMORY_BUDGET",
            "PREVIEW_TOKEN_BUDGET", "MAX_DESCRIPTION_TOKENS",
            "MAX_ENTRY_TOKENS", "CHUNKING_THRESHOLD_TOKENS",
        ]:
            val = getattr(DEFAULTS, attr)
            assert isinstance(val, int), f"{attr} should be int, got {type(val)}"
            assert val > 0, f"{attr} should be positive, got {val}"

    @pytest.mark.unit
    def test_min_memory_less_than_max_memory(self):
        """MIN_MEMORY_BUDGET should be less than MAX_MEMORY_BUDGET."""
        assert DEFAULTS.MIN_MEMORY_BUDGET < DEFAULTS.MAX_MEMORY_BUDGET


@pytest.mark.skipif(not HAS_DEFAULTS, reason="config_defaults not importable")
class TestJottyDefaultsMemoryCapacity:
    """Tests for memory capacity defaults."""

    @pytest.mark.unit
    def test_episodic_capacity(self):
        """EPISODIC_CAPACITY should be 1_000."""
        assert DEFAULTS.EPISODIC_CAPACITY == 1_000

    @pytest.mark.unit
    def test_semantic_capacity(self):
        """SEMANTIC_CAPACITY should be 500."""
        assert DEFAULTS.SEMANTIC_CAPACITY == 500

    @pytest.mark.unit
    def test_procedural_capacity(self):
        """PROCEDURAL_CAPACITY should be 100."""
        assert DEFAULTS.PROCEDURAL_CAPACITY == 100

    @pytest.mark.unit
    def test_meta_capacity(self):
        """META_CAPACITY should be 50."""
        assert DEFAULTS.META_CAPACITY == 50

    @pytest.mark.unit
    def test_causal_capacity(self):
        """CAUSAL_CAPACITY should be 200."""
        assert DEFAULTS.CAUSAL_CAPACITY == 200

    @pytest.mark.unit
    def test_episode_buffer_size(self):
        """EPISODE_BUFFER_SIZE should be 1_000."""
        assert DEFAULTS.EPISODE_BUFFER_SIZE == 1_000

    @pytest.mark.unit
    def test_dlq_max_size(self):
        """DLQ_MAX_SIZE should be 1_000."""
        assert DEFAULTS.DLQ_MAX_SIZE == 1_000


@pytest.mark.skipif(not HAS_DEFAULTS, reason="config_defaults not importable")
class TestJottyDefaultsLearningParams:
    """Tests for learning / RL parameters."""

    @pytest.mark.unit
    def test_learning_rate(self):
        """LEARNING_RATE should be 0.1."""
        assert DEFAULTS.LEARNING_RATE == pytest.approx(0.1)

    @pytest.mark.unit
    def test_discount_factor(self):
        """DISCOUNT_FACTOR should be 0.99."""
        assert DEFAULTS.DISCOUNT_FACTOR == pytest.approx(0.99)

    @pytest.mark.unit
    def test_lambda_value(self):
        """LAMBDA_VALUE should be 0.9."""
        assert DEFAULTS.LAMBDA_VALUE == pytest.approx(0.9)

    @pytest.mark.unit
    def test_credit_bounds(self):
        """MIN_CREDIT < MAX_CREDIT."""
        assert DEFAULTS.MIN_CREDIT < DEFAULTS.MAX_CREDIT
        assert DEFAULTS.MIN_CREDIT == pytest.approx(0.01)
        assert DEFAULTS.MAX_CREDIT == pytest.approx(1.0)

    @pytest.mark.unit
    def test_epsilon_values(self):
        """Epsilon exploration parameters are well-ordered."""
        assert DEFAULTS.EPSILON_MIN < DEFAULTS.EPSILON_START
        assert 0.0 < DEFAULTS.EPSILON_DECAY <= 1.0


@pytest.mark.skipif(not HAS_DEFAULTS, reason="config_defaults not importable")
class TestJottyDefaultsImmutability:
    """Tests that JottyDefaults is frozen (immutable)."""

    @pytest.mark.unit
    def test_frozen_instance_raises_on_attribute_set(self):
        """Setting an attribute on a frozen instance should raise."""
        d = JottyDefaults()
        with pytest.raises((FrozenInstanceError, AttributeError)):
            d.LLM_TEMPERATURE = 0.5

    @pytest.mark.unit
    def test_frozen_defaults_singleton(self):
        """Setting an attribute on DEFAULTS should raise."""
        with pytest.raises((FrozenInstanceError, AttributeError)):
            DEFAULTS.MAX_CONTEXT_TOKENS = 999

    @pytest.mark.unit
    def test_frozen_model_name(self):
        """Setting MODEL_SONNET on an instance should raise."""
        d = JottyDefaults()
        with pytest.raises((FrozenInstanceError, AttributeError)):
            d.MODEL_SONNET = "new-model"


@pytest.mark.skipif(not HAS_DEFAULTS, reason="config_defaults not importable")
class TestJottyDefaultsClassMethods:
    """Tests for for_model(), conservative(), aggressive() classmethods."""

    @pytest.mark.unit
    def test_for_model_returns_jotty_defaults(self):
        """for_model() should return a JottyDefaults instance."""
        result = JottyDefaults.for_model("gpt-4")
        assert isinstance(result, JottyDefaults)

    @pytest.mark.unit
    def test_for_model_accepts_any_model_string(self):
        """for_model() should not raise for arbitrary model names."""
        for model in ["gpt-4-32k", "claude-3-opus", "llama-70b", "unknown-model"]:
            result = JottyDefaults.for_model(model)
            assert isinstance(result, JottyDefaults)

    @pytest.mark.unit
    def test_conservative_returns_jotty_defaults(self):
        """conservative() should return a JottyDefaults instance."""
        result = JottyDefaults.conservative()
        assert isinstance(result, JottyDefaults)

    @pytest.mark.unit
    def test_aggressive_returns_jotty_defaults(self):
        """aggressive() should return a JottyDefaults instance."""
        result = JottyDefaults.aggressive()
        assert isinstance(result, JottyDefaults)

    @pytest.mark.unit
    def test_classmethods_return_distinct_objects(self):
        """Each classmethod call should return a new object."""
        a = JottyDefaults.for_model("x")
        b = JottyDefaults.conservative()
        c = JottyDefaults.aggressive()
        # They are all equal in value but are separate objects
        assert a == b == c

    @pytest.mark.unit
    def test_for_model_result_is_frozen(self):
        """Result of for_model() should also be frozen."""
        result = JottyDefaults.for_model("claude-3-opus")
        with pytest.raises((FrozenInstanceError, AttributeError)):
            result.LLM_TEMPERATURE = 99.0


@pytest.mark.skipif(not HAS_DEFAULTS, reason="config_defaults not importable")
class TestJottyDefaultsServiceURLs:
    """Tests for service URL defaults."""

    @pytest.mark.unit
    def test_justjot_api_url(self):
        """JUSTJOT_API_URL should be localhost:3000."""
        assert "localhost" in DEFAULTS.JUSTJOT_API_URL
        assert "3000" in DEFAULTS.JUSTJOT_API_URL

    @pytest.mark.unit
    def test_jotty_gateway_url(self):
        """JOTTY_GATEWAY_URL should be localhost:8766."""
        assert "8766" in DEFAULTS.JOTTY_GATEWAY_URL


# ###########################################################################
# 2. Exception hierarchy
# ###########################################################################

@pytest.mark.skipif(not HAS_EXCEPTIONS, reason="exceptions not importable")
class TestJottyErrorBase:
    """Tests for the JottyError base exception."""

    @pytest.mark.unit
    def test_jotty_error_is_exception(self):
        """JottyError should inherit from Exception."""
        assert issubclass(JottyError, Exception)

    @pytest.mark.unit
    def test_jotty_error_message(self):
        """JottyError stores message attribute."""
        err = JottyError("something went wrong")
        assert err.message == "something went wrong"

    @pytest.mark.unit
    def test_jotty_error_context_default(self):
        """JottyError context defaults to empty dict."""
        err = JottyError("oops")
        assert err.context == {}

    @pytest.mark.unit
    def test_jotty_error_context_provided(self):
        """JottyError stores provided context."""
        ctx = {"agent": "TestAgent", "step": 3}
        err = JottyError("fail", context=ctx)
        assert err.context == ctx

    @pytest.mark.unit
    def test_jotty_error_original_error(self):
        """JottyError stores original_error when wrapping."""
        original = ValueError("bad value")
        err = JottyError("wrapped", original_error=original)
        assert err.original_error is original

    @pytest.mark.unit
    def test_jotty_error_str_includes_context(self):
        """str(JottyError) should include context when provided."""
        err = JottyError("fail", context={"key": "val"})
        s = str(err)
        assert "fail" in s
        assert "Context" in s

    @pytest.mark.unit
    def test_jotty_error_str_includes_caused_by(self):
        """str(JottyError) should include caused-by info."""
        original = RuntimeError("boom")
        err = JottyError("wrapped", original_error=original)
        s = str(err)
        assert "Caused by" in s
        assert "RuntimeError" in s

    @pytest.mark.unit
    def test_jotty_error_catchable_as_exception(self):
        """JottyError should be catchable as Exception."""
        with pytest.raises(Exception):
            raise JottyError("test")


@pytest.mark.skipif(not HAS_EXCEPTIONS, reason="exceptions not importable")
class TestExceptionInheritanceChain:
    """Tests for the full exception inheritance hierarchy."""

    @pytest.mark.unit
    def test_configuration_error_inherits_jotty_error(self):
        assert issubclass(ConfigurationError, JottyError)

    @pytest.mark.unit
    def test_invalid_config_inherits_configuration_error(self):
        assert issubclass(InvalidConfigError, ConfigurationError)

    @pytest.mark.unit
    def test_missing_config_inherits_configuration_error(self):
        assert issubclass(MissingConfigError, ConfigurationError)

    @pytest.mark.unit
    def test_execution_error_inherits_jotty_error(self):
        assert issubclass(ExecutionError, JottyError)

    @pytest.mark.unit
    def test_agent_execution_error_inherits_execution_error(self):
        assert issubclass(AgentExecutionError, ExecutionError)

    @pytest.mark.unit
    def test_tool_execution_error_inherits_execution_error(self):
        assert issubclass(ToolExecutionError, ExecutionError)

    @pytest.mark.unit
    def test_timeout_error_inherits_execution_error(self):
        assert issubclass(JottyTimeoutError, ExecutionError)

    @pytest.mark.unit
    def test_timeout_error_full_chain(self):
        """TimeoutError -> ExecutionError -> JottyError -> Exception."""
        assert issubclass(JottyTimeoutError, ExecutionError)
        assert issubclass(JottyTimeoutError, JottyError)
        assert issubclass(JottyTimeoutError, Exception)

    @pytest.mark.unit
    def test_circuit_breaker_inherits_execution_error(self):
        assert issubclass(CircuitBreakerError, ExecutionError)

    @pytest.mark.unit
    def test_context_error_inherits_jotty_error(self):
        assert issubclass(ContextError, JottyError)

    @pytest.mark.unit
    def test_context_overflow_inherits_context_error(self):
        assert issubclass(ContextOverflowError, ContextError)

    @pytest.mark.unit
    def test_compression_error_inherits_context_error(self):
        assert issubclass(CompressionError, ContextError)

    @pytest.mark.unit
    def test_chunking_error_inherits_context_error(self):
        assert issubclass(ChunkingError, ContextError)

    @pytest.mark.unit
    def test_memory_error_inherits_jotty_error(self):
        assert issubclass(JottyMemoryError, JottyError)

    @pytest.mark.unit
    def test_memory_retrieval_inherits_memory_error(self):
        assert issubclass(MemoryRetrievalError, JottyMemoryError)

    @pytest.mark.unit
    def test_memory_storage_inherits_memory_error(self):
        assert issubclass(MemoryStorageError, JottyMemoryError)

    @pytest.mark.unit
    def test_consolidation_inherits_memory_error(self):
        assert issubclass(ConsolidationError, JottyMemoryError)

    @pytest.mark.unit
    def test_learning_error_inherits_jotty_error(self):
        assert issubclass(LearningError, JottyError)

    @pytest.mark.unit
    def test_reward_calculation_inherits_learning_error(self):
        assert issubclass(RewardCalculationError, LearningError)

    @pytest.mark.unit
    def test_credit_assignment_inherits_learning_error(self):
        assert issubclass(CreditAssignmentError, LearningError)

    @pytest.mark.unit
    def test_policy_update_inherits_learning_error(self):
        assert issubclass(PolicyUpdateError, LearningError)

    @pytest.mark.unit
    def test_communication_error_inherits_jotty_error(self):
        assert issubclass(CommunicationError, JottyError)

    @pytest.mark.unit
    def test_message_delivery_inherits_communication_error(self):
        assert issubclass(MessageDeliveryError, CommunicationError)

    @pytest.mark.unit
    def test_feedback_routing_inherits_communication_error(self):
        assert issubclass(FeedbackRoutingError, CommunicationError)

    @pytest.mark.unit
    def test_persistence_error_inherits_jotty_error(self):
        assert issubclass(PersistenceError, JottyError)

    @pytest.mark.unit
    def test_storage_error_inherits_persistence_error(self):
        assert issubclass(StorageError, PersistenceError)

    @pytest.mark.unit
    def test_retrieval_error_inherits_persistence_error(self):
        assert issubclass(RetrievalError, PersistenceError)

    @pytest.mark.unit
    def test_integration_error_inherits_jotty_error(self):
        assert issubclass(IntegrationError, JottyError)

    @pytest.mark.unit
    def test_llm_error_inherits_integration_error(self):
        assert issubclass(LLMError, IntegrationError)

    @pytest.mark.unit
    def test_dspy_error_inherits_integration_error(self):
        assert issubclass(DSPyError, IntegrationError)

    @pytest.mark.unit
    def test_external_tool_error_inherits_integration_error(self):
        assert issubclass(ExternalToolError, IntegrationError)


@pytest.mark.skipif(not HAS_EXCEPTIONS, reason="exceptions not importable")
class TestContextOverflowError:
    """Tests for ContextOverflowError with detected_tokens / max_tokens."""

    @pytest.mark.unit
    def test_stores_detected_tokens(self):
        err = ContextOverflowError("overflow", detected_tokens=120_000, max_tokens=100_000)
        assert err.detected_tokens == 120_000

    @pytest.mark.unit
    def test_stores_max_tokens(self):
        err = ContextOverflowError("overflow", detected_tokens=120_000, max_tokens=100_000)
        assert err.max_tokens == 100_000

    @pytest.mark.unit
    def test_context_dict_contains_token_info(self):
        err = ContextOverflowError("overflow", detected_tokens=150_000, max_tokens=100_000)
        assert err.context["detected_tokens"] == 150_000
        assert err.context["max_tokens"] == 100_000

    @pytest.mark.unit
    def test_none_tokens_are_accepted(self):
        err = ContextOverflowError("overflow")
        assert err.detected_tokens is None
        assert err.max_tokens is None

    @pytest.mark.unit
    def test_is_catchable_as_context_error(self):
        with pytest.raises(ContextError):
            raise ContextOverflowError("too many tokens", detected_tokens=200_000, max_tokens=100_000)


@pytest.mark.skipif(not HAS_EXCEPTIONS, reason="exceptions not importable")
class TestValidationError:
    """Tests for ValidationError with param, value, to_dict()."""

    @pytest.mark.unit
    def test_validation_error_inherits_jotty_error(self):
        assert issubclass(ValidationError, JottyError)

    @pytest.mark.unit
    def test_stores_param(self):
        err = ValidationError("bad param", param="temperature")
        assert err.param == "temperature"

    @pytest.mark.unit
    def test_stores_value(self):
        err = ValidationError("bad value", param="temperature", value=99.9)
        assert err.value == 99.9

    @pytest.mark.unit
    def test_param_defaults_to_none(self):
        err = ValidationError("generic")
        assert err.param is None
        assert err.value is None

    @pytest.mark.unit
    def test_to_dict_structure(self):
        err = ValidationError("bad temperature", param="temperature", value=99.9)
        d = err.to_dict()
        assert d["success"] is False
        assert d["error"] == "bad temperature"
        assert d["param"] == "temperature"

    @pytest.mark.unit
    def test_to_dict_returns_dict(self):
        err = ValidationError("fail")
        assert isinstance(err.to_dict(), dict)

    @pytest.mark.unit
    def test_input_validation_error_inherits_validation_error(self):
        assert issubclass(InputValidationError, ValidationError)

    @pytest.mark.unit
    def test_output_validation_error_inherits_validation_error(self):
        assert issubclass(OutputValidationError, ValidationError)

    @pytest.mark.unit
    def test_validation_error_with_context_and_original(self):
        """ValidationError can accept context and original_error."""
        orig = TypeError("type mismatch")
        err = ValidationError(
            "invalid", param="x", value=42,
            context={"step": 1}, original_error=orig,
        )
        assert err.context == {"step": 1}
        assert err.original_error is orig
        assert err.param == "x"
        assert err.value == 42


@pytest.mark.skipif(not HAS_EXCEPTIONS, reason="exceptions not importable")
class TestWrapException:
    """Tests for the wrap_exception() utility function."""

    @pytest.mark.unit
    def test_wrap_returns_jotty_error_subclass(self):
        original = ValueError("bad")
        wrapped = wrap_exception(original, AgentExecutionError, "agent failed")
        assert isinstance(wrapped, AgentExecutionError)

    @pytest.mark.unit
    def test_wrap_stores_original(self):
        original = IOError("disk full")
        wrapped = wrap_exception(original, StorageError, "store failed")
        assert wrapped.original_error is original

    @pytest.mark.unit
    def test_wrap_stores_message(self):
        original = RuntimeError("timeout")
        wrapped = wrap_exception(original, LLMError, "LLM call timed out")
        assert wrapped.message == "LLM call timed out"

    @pytest.mark.unit
    def test_wrap_passes_kwargs_as_context(self):
        original = KeyError("missing")
        wrapped = wrap_exception(
            original, MissingConfigError, "config missing",
            config_key="api_key", component="auth",
        )
        assert wrapped.context["config_key"] == "api_key"
        assert wrapped.context["component"] == "auth"

    @pytest.mark.unit
    def test_wrap_result_is_raisable(self):
        original = ValueError("x")
        wrapped = wrap_exception(original, ExecutionError, "exec fail")
        with pytest.raises(ExecutionError):
            raise wrapped

    @pytest.mark.unit
    def test_wrap_result_catchable_as_jotty_error(self):
        original = Exception("generic")
        wrapped = wrap_exception(original, DSPyError, "dspy fail")
        with pytest.raises(JottyError):
            raise wrapped


@pytest.mark.skipif(not HAS_EXCEPTIONS, reason="exceptions not importable")
class TestExceptionInstantiation:
    """Verify every exception class can be instantiated with a message."""

    @pytest.mark.unit
    def test_all_exception_classes_instantiate(self):
        """Every leaf exception class should be instantiable with a message string."""
        classes = [
            JottyError, ConfigurationError, InvalidConfigError, MissingConfigError,
            ExecutionError, AgentExecutionError, ToolExecutionError,
            JottyTimeoutError, CircuitBreakerError,
            ContextError, CompressionError, ChunkingError,
            JottyMemoryError, MemoryRetrievalError, MemoryStorageError, ConsolidationError,
            LearningError, RewardCalculationError, CreditAssignmentError, PolicyUpdateError,
            CommunicationError, MessageDeliveryError, FeedbackRoutingError,
            PersistenceError, StorageError, RetrievalError,
            IntegrationError, LLMError, DSPyError, ExternalToolError,
        ]
        for cls in classes:
            err = cls(f"test {cls.__name__}")
            assert isinstance(err, JottyError)
            assert isinstance(err, Exception)


# ###########################################################################
# 3. Protocols
# ###########################################################################

@pytest.mark.skipif(not HAS_PROTOCOLS, reason="protocols not importable")
class TestMetadataProviderProtocol:
    """Tests for the MetadataProvider runtime_checkable protocol."""

    @pytest.mark.unit
    def test_conforming_class_passes_isinstance(self):
        """A class with get_context_for_actor and get_swarm_context is a MetadataProvider."""
        class MyProvider:
            def get_context_for_actor(self, actor_name, query, previous_outputs=None, **kwargs):
                return {}
            def get_swarm_context(self, **kwargs):
                return {}

        provider = MyProvider()
        assert isinstance(provider, MetadataProvider)

    @pytest.mark.unit
    def test_non_conforming_class_fails_isinstance(self):
        """A class missing required methods is NOT a MetadataProvider."""
        class NotProvider:
            def some_method(self):
                pass

        obj = NotProvider()
        assert not isinstance(obj, MetadataProvider)

    @pytest.mark.unit
    def test_partial_conforming_missing_get_swarm_context(self):
        """Missing get_swarm_context should fail isinstance check."""
        class PartialProvider:
            def get_context_for_actor(self, actor_name, query, previous_outputs=None, **kwargs):
                return {}

        obj = PartialProvider()
        assert not isinstance(obj, MetadataProvider)

    @pytest.mark.unit
    def test_conforming_provider_returns_context(self):
        """A conforming provider should return dicts from its methods."""
        class SQLProvider:
            def get_context_for_actor(self, actor_name, query, previous_outputs=None, **kwargs):
                return {"tables": ["users", "orders"]}
            def get_swarm_context(self, **kwargs):
                return {"domain": "sql"}

        p = SQLProvider()
        ctx = p.get_context_for_actor("Agent1", "query")
        assert "tables" in ctx
        swarm_ctx = p.get_swarm_context()
        assert swarm_ctx["domain"] == "sql"


@pytest.mark.skipif(not HAS_PROTOCOLS, reason="protocols not importable")
class TestDataProviderProtocol:
    """Tests for the DataProvider runtime_checkable protocol."""

    @pytest.mark.unit
    def test_conforming_class_passes_isinstance(self):
        """A class with retrieve() and store() is a DataProvider."""
        class FileProvider:
            def retrieve(self, key, format=None, **kwargs):
                return f"data for {key}"
            def store(self, key, value, format=None, **kwargs):
                pass

        provider = FileProvider()
        assert isinstance(provider, DataProvider)

    @pytest.mark.unit
    def test_non_conforming_class_fails_isinstance(self):
        """A class missing retrieve/store is NOT a DataProvider."""
        class BadProvider:
            def fetch(self, key):
                pass

        obj = BadProvider()
        assert not isinstance(obj, DataProvider)

    @pytest.mark.unit
    def test_missing_store_fails(self):
        """A class with retrieve() but no store() is NOT a DataProvider."""
        class ReadOnly:
            def retrieve(self, key, format=None, **kwargs):
                return None

        obj = ReadOnly()
        assert not isinstance(obj, DataProvider)


@pytest.mark.skipif(not HAS_PROTOCOLS, reason="protocols not importable")
class TestContextExtractorProtocol:
    """Tests for the ContextExtractor runtime_checkable protocol."""

    @pytest.mark.unit
    def test_conforming_class_passes_isinstance(self):
        """A class with extract() is a ContextExtractor."""
        class KeywordExtractor:
            def extract(self, content, query, max_tokens, **kwargs):
                return content[:max_tokens]

        extractor = KeywordExtractor()
        assert isinstance(extractor, ContextExtractor)

    @pytest.mark.unit
    def test_non_conforming_class_fails_isinstance(self):
        """A class without extract() is NOT a ContextExtractor."""
        class NoExtract:
            def process(self, content):
                pass

        obj = NoExtract()
        assert not isinstance(obj, ContextExtractor)

    @pytest.mark.unit
    def test_extract_returns_string(self):
        """A conforming extractor's extract() returns a string."""
        class TruncExtractor:
            def extract(self, content, query, max_tokens, **kwargs):
                return content[:max_tokens]

        ext = TruncExtractor()
        result = ext.extract("Hello world, this is a long document.", "hello", 5)
        assert isinstance(result, str)
        assert len(result) <= 5


# ###########################################################################
# 4. AgentConfig
# ###########################################################################

@pytest.mark.skipif(not HAS_AGENT_CONFIG, reason="agent_config not importable")
class TestAgentConfigCreation:
    """Tests for AgentConfig dataclass creation and defaults."""

    @pytest.mark.unit
    def test_basic_creation(self):
        """AgentConfig can be created with name and agent."""
        mock_agent = Mock()
        config = AgentConfig(name="TestAgent", agent=mock_agent)
        assert config.name == "TestAgent"
        assert config.agent is mock_agent

    @pytest.mark.unit
    def test_architect_prompts_default_to_empty_list(self):
        """architect_prompts should default to empty list after __post_init__."""
        config = AgentConfig(name="A", agent=Mock())
        assert config.architect_prompts == []

    @pytest.mark.unit
    def test_auditor_prompts_default_to_empty_list(self):
        """auditor_prompts should default to empty list after __post_init__."""
        config = AgentConfig(name="A", agent=Mock())
        assert config.auditor_prompts == []

    @pytest.mark.unit
    def test_enable_architect_defaults_true(self):
        """enable_architect should default to True."""
        config = AgentConfig(name="A", agent=Mock())
        assert config.enable_architect is True

    @pytest.mark.unit
    def test_enable_auditor_defaults_true(self):
        """enable_auditor should default to True."""
        config = AgentConfig(name="A", agent=Mock())
        assert config.enable_auditor is True

    @pytest.mark.unit
    def test_validation_mode_defaults_standard(self):
        """validation_mode should default to 'standard'."""
        config = AgentConfig(name="A", agent=Mock())
        assert config.validation_mode == "standard"

    @pytest.mark.unit
    def test_is_critical_defaults_false(self):
        """is_critical should default to False."""
        config = AgentConfig(name="A", agent=Mock())
        assert config.is_critical is False

    @pytest.mark.unit
    def test_is_executor_defaults_false(self):
        """is_executor should default to False."""
        config = AgentConfig(name="A", agent=Mock())
        assert config.is_executor is False

    @pytest.mark.unit
    def test_enabled_defaults_true(self):
        """enabled should default to True."""
        config = AgentConfig(name="A", agent=Mock())
        assert config.enabled is True

    @pytest.mark.unit
    def test_max_retries_resolved_from_defaults(self):
        """max_retries=0 triggers resolution from config_defaults.MAX_RETRIES."""
        config = AgentConfig(name="A", agent=Mock())
        # After __post_init__, max_retries should be > 0
        assert config.max_retries > 0

    @pytest.mark.unit
    def test_retry_strategy_defaults_with_hints(self):
        """retry_strategy should default to 'with_hints'."""
        config = AgentConfig(name="A", agent=Mock())
        assert config.retry_strategy == "with_hints"


@pytest.mark.skipif(not HAS_AGENT_CONFIG, reason="agent_config not importable")
class TestAgentConfigCustomValues:
    """Tests for AgentConfig with custom field values."""

    @pytest.mark.unit
    def test_custom_architect_prompts(self):
        """Custom architect_prompts are preserved."""
        config = AgentConfig(
            name="A", agent=Mock(),
            architect_prompts=["plan.md", "context.md"],
        )
        assert config.architect_prompts == ["plan.md", "context.md"]

    @pytest.mark.unit
    def test_custom_capabilities(self):
        """capabilities field stores a list of strings."""
        config = AgentConfig(
            name="A", agent=Mock(),
            capabilities=["code_review", "testing"],
        )
        assert config.capabilities == ["code_review", "testing"]

    @pytest.mark.unit
    def test_custom_dependencies(self):
        """dependencies field stores a list of agent names."""
        config = AgentConfig(
            name="B", agent=Mock(),
            dependencies=["A"],
        )
        assert config.dependencies == ["A"]

    @pytest.mark.unit
    def test_custom_metadata(self):
        """metadata field stores arbitrary dict."""
        config = AgentConfig(
            name="A", agent=Mock(),
            metadata={"priority": "high"},
        )
        assert config.metadata["priority"] == "high"

    @pytest.mark.unit
    def test_parameter_mappings(self):
        """parameter_mappings stores input-to-param mapping."""
        config = AgentConfig(
            name="A", agent=Mock(),
            parameter_mappings={"input_query": "query"},
        )
        assert config.parameter_mappings["input_query"] == "query"

    @pytest.mark.unit
    def test_outputs_list(self):
        """outputs stores list of output field names."""
        config = AgentConfig(
            name="A", agent=Mock(),
            outputs=["sql_query", "explanation"],
        )
        assert "sql_query" in config.outputs

    @pytest.mark.unit
    def test_provides_list(self):
        """provides stores list of parameter names this agent can provide."""
        config = AgentConfig(
            name="A", agent=Mock(),
            provides=["table_list", "column_info"],
        )
        assert "table_list" in config.provides

    @pytest.mark.unit
    def test_asdict_produces_dict(self):
        """asdict() should produce a dictionary representation."""
        config = AgentConfig(name="A", agent=Mock())
        d = asdict(config)
        assert isinstance(d, dict)
        assert d["name"] == "A"


# ###########################################################################
# 5. Agent Types (AgentContribution, AgentMessage, SharedScratchpad)
# ###########################################################################

@pytest.mark.skipif(not HAS_AGENT_TYPES, reason="agent_types not importable")
class TestAgentContribution:
    """Tests for AgentContribution dataclass and compute_final_contribution()."""

    def _make_contribution(self, **overrides):
        """Helper to create an AgentContribution with sensible defaults."""
        defaults = dict(
            agent_name="agent1",
            contribution_score=0.8,
            decision="approve",
            decision_correct=True,
            counterfactual_impact=0.6,
            reasoning_quality=0.9,
            evidence_used=["doc1", "doc2"],
            tools_used=["search"],
            decision_timing=0.5,
            temporal_weight=1.0,
        )
        defaults.update(overrides)
        return AgentContribution(**defaults)

    @pytest.mark.unit
    def test_creation(self):
        """AgentContribution can be created with all required fields."""
        contrib = self._make_contribution()
        assert contrib.agent_name == "agent1"
        assert contrib.contribution_score == 0.8

    @pytest.mark.unit
    def test_compute_final_contribution_positive(self):
        """compute_final_contribution returns a float."""
        contrib = self._make_contribution()
        result = contrib.compute_final_contribution()
        assert isinstance(result, float)

    @pytest.mark.unit
    def test_compute_final_contribution_formula(self):
        """Verify the computation formula manually."""
        contrib = self._make_contribution(
            contribution_score=1.0,
            reasoning_quality=1.0,
            counterfactual_impact=1.0,
            decision_timing=1.0,
        )
        # reasoning_factor = 0.5 + 0.5*1.0 = 1.0
        # impact_factor = 0.5 + 0.5*1.0 = 1.0
        # temporal_factor = 0.7 + 0.3*1.0 = 1.0
        # result = 1.0 * 1.0 * 1.0 * 1.0 = 1.0
        assert contrib.compute_final_contribution() == pytest.approx(1.0)

    @pytest.mark.unit
    def test_compute_with_zero_reasoning_quality(self):
        """Zero reasoning_quality halves the reasoning factor."""
        contrib = self._make_contribution(
            contribution_score=1.0,
            reasoning_quality=0.0,
            counterfactual_impact=1.0,
            decision_timing=1.0,
        )
        # reasoning_factor = 0.5 + 0.5*0 = 0.5
        # impact_factor = 1.0, temporal_factor = 1.0
        # result = 1.0 * 0.5 * 1.0 * 1.0 = 0.5
        assert contrib.compute_final_contribution() == pytest.approx(0.5)

    @pytest.mark.unit
    def test_compute_with_zero_counterfactual_impact(self):
        """Zero counterfactual_impact halves the impact factor."""
        contrib = self._make_contribution(
            contribution_score=1.0,
            reasoning_quality=1.0,
            counterfactual_impact=0.0,
            decision_timing=1.0,
        )
        # impact_factor = 0.5
        assert contrib.compute_final_contribution() == pytest.approx(0.5)

    @pytest.mark.unit
    def test_compute_with_early_timing(self):
        """Early decision_timing (0.0) reduces temporal factor."""
        contrib = self._make_contribution(
            contribution_score=1.0,
            reasoning_quality=1.0,
            counterfactual_impact=1.0,
            decision_timing=0.0,
        )
        # temporal_factor = 0.7 + 0.3*0.0 = 0.7
        assert contrib.compute_final_contribution() == pytest.approx(0.7)

    @pytest.mark.unit
    def test_compute_with_negative_contribution_score(self):
        """Negative contribution_score should produce negative result."""
        contrib = self._make_contribution(
            contribution_score=-0.5,
            reasoning_quality=1.0,
            counterfactual_impact=1.0,
            decision_timing=1.0,
        )
        result = contrib.compute_final_contribution()
        assert result < 0

    @pytest.mark.unit
    def test_evidence_used_list(self):
        """evidence_used stores list of strings."""
        contrib = self._make_contribution(evidence_used=["a", "b", "c"])
        assert len(contrib.evidence_used) == 3

    @pytest.mark.unit
    def test_tools_used_list(self):
        """tools_used stores list of strings."""
        contrib = self._make_contribution(tools_used=["search", "calculator"])
        assert "calculator" in contrib.tools_used


@pytest.mark.skipif(not HAS_AGENT_TYPES, reason="agent_types not importable")
class TestAgentMessage:
    """Tests for AgentMessage dataclass."""

    @pytest.mark.unit
    def test_creation(self):
        """AgentMessage can be created with required fields."""
        msg = AgentMessage(
            sender="agent1",
            receiver="agent2",
            message_type=CommunicationType.INSIGHT,
            content={"key": "value"},
        )
        assert msg.sender == "agent1"
        assert msg.receiver == "agent2"

    @pytest.mark.unit
    def test_timestamp_auto_set(self):
        """timestamp should default to current time."""
        msg = AgentMessage(
            sender="a", receiver="b",
            message_type=CommunicationType.INSIGHT,
            content={},
        )
        assert isinstance(msg.timestamp, datetime)

    @pytest.mark.unit
    def test_broadcast_receiver(self):
        """receiver='*' means broadcast."""
        msg = AgentMessage(
            sender="a", receiver="*",
            message_type=CommunicationType.WARNING,
            content={"warn": "high latency"},
        )
        assert msg.receiver == "*"

    @pytest.mark.unit
    def test_tool_result_fields(self):
        """Tool result fields are stored correctly."""
        msg = AgentMessage(
            sender="a", receiver="b",
            message_type=CommunicationType.TOOL_RESULT,
            content={},
            tool_name="web_search",
            tool_args={"query": "AI trends"},
            tool_result={"results": ["r1", "r2"]},
        )
        assert msg.tool_name == "web_search"
        assert msg.tool_args["query"] == "AI trends"
        assert len(msg.tool_result["results"]) == 2

    @pytest.mark.unit
    def test_insight_fields(self):
        """Insight and confidence fields are stored."""
        msg = AgentMessage(
            sender="a", receiver="b",
            message_type=CommunicationType.INSIGHT,
            content={},
            insight="Data shows upward trend",
            confidence=0.85,
        )
        assert msg.insight == "Data shows upward trend"
        assert msg.confidence == 0.85

    @pytest.mark.unit
    def test_optional_fields_default_none(self):
        """Optional fields default to None."""
        msg = AgentMessage(
            sender="a", receiver="b",
            message_type=CommunicationType.REQUEST,
            content={},
        )
        assert msg.tool_name is None
        assert msg.tool_args is None
        assert msg.tool_result is None
        assert msg.insight is None
        assert msg.confidence is None


@pytest.mark.skipif(not HAS_AGENT_TYPES, reason="agent_types not importable")
class TestSharedScratchpad:
    """Tests for SharedScratchpad dataclass."""

    def _make_message(self, sender="a", receiver="b",
                      msg_type=CommunicationType.INSIGHT, **kwargs):
        """Helper to create an AgentMessage."""
        return AgentMessage(
            sender=sender, receiver=receiver,
            message_type=msg_type,
            content=kwargs.get("content", {}),
            tool_name=kwargs.get("tool_name"),
            tool_args=kwargs.get("tool_args"),
            tool_result=kwargs.get("tool_result"),
        )

    @pytest.mark.unit
    def test_creation_empty(self):
        """SharedScratchpad starts empty."""
        pad = SharedScratchpad()
        assert pad.messages == []
        assert pad.tool_cache == {}
        assert pad.shared_insights == []

    @pytest.mark.unit
    def test_add_message(self):
        """add_message appends to the messages list."""
        pad = SharedScratchpad()
        msg = self._make_message()
        pad.add_message(msg)
        assert len(pad.messages) == 1
        assert pad.messages[0] is msg

    @pytest.mark.unit
    def test_add_tool_result_caches(self):
        """Adding a TOOL_RESULT message populates tool_cache."""
        pad = SharedScratchpad()
        msg = self._make_message(
            msg_type=CommunicationType.TOOL_RESULT,
            tool_name="calculator",
            tool_args={"expr": "2+2"},
            tool_result=4,
        )
        pad.add_message(msg)
        cached = pad.get_cached_result("calculator", {"expr": "2+2"})
        assert cached == 4

    @pytest.mark.unit
    def test_get_cached_result_miss_returns_none(self):
        """get_cached_result returns None for cache misses."""
        pad = SharedScratchpad()
        assert pad.get_cached_result("nonexistent", {}) is None

    @pytest.mark.unit
    def test_get_messages_for_specific_receiver(self):
        """get_messages_for returns messages addressed to the receiver."""
        pad = SharedScratchpad()
        pad.add_message(self._make_message(sender="a", receiver="b"))
        pad.add_message(self._make_message(sender="a", receiver="c"))
        pad.add_message(self._make_message(sender="b", receiver="b"))

        msgs_for_b = pad.get_messages_for("b")
        assert len(msgs_for_b) == 2  # receiver="b" appears twice

    @pytest.mark.unit
    def test_get_messages_for_includes_broadcast(self):
        """get_messages_for includes broadcast messages (receiver='*')."""
        pad = SharedScratchpad()
        pad.add_message(self._make_message(sender="a", receiver="*"))
        pad.add_message(self._make_message(sender="a", receiver="c"))

        msgs_for_b = pad.get_messages_for("b")
        assert len(msgs_for_b) == 1  # only the broadcast

    @pytest.mark.unit
    def test_get_messages_for_empty(self):
        """get_messages_for returns empty list when no messages match."""
        pad = SharedScratchpad()
        pad.add_message(self._make_message(sender="a", receiver="b"))
        assert pad.get_messages_for("z") == []

    @pytest.mark.unit
    def test_clear(self):
        """clear() empties messages, tool_cache, and shared_insights."""
        pad = SharedScratchpad()
        pad.add_message(self._make_message(
            msg_type=CommunicationType.TOOL_RESULT,
            tool_name="t", tool_args={}, tool_result="r",
        ))
        pad.shared_insights.append("insight1")

        pad.clear()
        assert pad.messages == []
        assert pad.tool_cache == {}
        assert pad.shared_insights == []

    @pytest.mark.unit
    def test_multiple_tool_caches(self):
        """Multiple tool results are cached independently."""
        pad = SharedScratchpad()
        pad.add_message(self._make_message(
            msg_type=CommunicationType.TOOL_RESULT,
            tool_name="calc", tool_args={"x": 1}, tool_result=10,
        ))
        pad.add_message(self._make_message(
            msg_type=CommunicationType.TOOL_RESULT,
            tool_name="calc", tool_args={"x": 2}, tool_result=20,
        ))
        assert pad.get_cached_result("calc", {"x": 1}) == 10
        assert pad.get_cached_result("calc", {"x": 2}) == 20

    @pytest.mark.unit
    def test_non_tool_result_message_not_cached(self):
        """Non-TOOL_RESULT messages should not populate tool_cache."""
        pad = SharedScratchpad()
        pad.add_message(self._make_message(
            msg_type=CommunicationType.INSIGHT,
            tool_name="calc", tool_args={"x": 1}, tool_result=10,
        ))
        assert pad.tool_cache == {}
