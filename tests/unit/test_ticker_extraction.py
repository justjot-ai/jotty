"""Tests for ticker extraction, error classification, and outcome storage."""

import json
import re
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _MockSwarmResult:
    """Minimal SwarmResult stand-in."""

    success: bool = True
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Ticker extraction standalone stub
#
# We can't import ResearchSwarm directly because the import chain triggers
# dspy metaclass issues in test environments. Instead, we replicate the
# _extract_ticker method with the class-level word sets for unit testing.
# ---------------------------------------------------------------------------


class _TickerExtractor:
    """Standalone stub replicating ResearchSwarm._extract_ticker for testing."""

    _SKIP_WORDS = {
        "RESEARCH",
        "STOCK",
        "COMPANY",
        "ANALYZE",
        "ANALYSIS",
        "LATEST",
        "NEWS",
        "PDF",
        "TELEGRAM",
        "SEND",
        "TO",
        "THE",
        "AND",
        "WITH",
        "FOR",
        "ABOUT",
        "GET",
        "SHOW",
        "FIND",
        "EXPLAIN",
        "CREATE",
        "COMPARE",
        "BETWEEN",
        "DIFFERENCES",
        "KEY",
        "WHAT",
        "HOW",
        "WHY",
        "WHEN",
        "WHERE",
        "WHO",
        "WHICH",
        "LIST",
        "DESCRIBE",
        "GENERATE",
        "MAKE",
        "WRITE",
        "BUILD",
        "TELL",
        "GIVE",
        "HELP",
        "USING",
        "FROM",
        "THAT",
        "THIS",
        "THESE",
        "THOSE",
        "THEIR",
        "THEM",
        "THAN",
        "THEN",
        "SHOULD",
        "WOULD",
        "COULD",
        "HAVE",
        "BEEN",
        "BEING",
        "DOES",
        "WILL",
        "CAN",
        "ARE",
        "NOT",
        "BUT",
        "ALL",
        "ALSO",
        "EACH",
        "EVERY",
        "MOST",
        "SOME",
        "SUCH",
        "LIKE",
        "OVER",
        "INTO",
        "MORE",
        "VERY",
    }

    _COMMON_ENGLISH_WORDS: set  # Loaded from the real class at module level

    def _extract_ticker(self, query: str) -> Optional[str]:
        """Extract stock ticker from query. Returns None if no valid ticker found."""
        explicit_patterns = [
            r"\b([A-Z]{1,15})\.(NS|BO|NSE|BSE)\b",
            r"\$([A-Z]{1,10})\b",
            r"\b(?:NSE|BSE|NYSE|NASDAQ)[:\s]([A-Z]{1,15})\b",
        ]
        for pat in explicit_patterns:
            m = re.search(pat, query, re.IGNORECASE)
            if m:
                return m.group(1).upper()

        words = query.split()
        for word in words:
            clean = re.sub(r'[,\-:.\'"!?]', "", word)
            if (
                clean.isupper()
                and 1 <= len(clean) <= 10
                and clean not in self._SKIP_WORDS
                and clean not in self._COMMON_ENGLISH_WORDS
            ):
                return clean

        return None


# Load the real _COMMON_ENGLISH_WORDS from the source module without importing
# the full class hierarchy. We read the source file and extract the set.
def _load_common_english_words() -> set:
    """Load _COMMON_ENGLISH_WORDS from ResearchSwarm source without importing it."""
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parents[2] / (
        "core/intelligence/orchestration/swarms/research_swarm/swarm.py"
    )
    text = src.read_text()
    # Find the _COMMON_ENGLISH_WORDS block — it starts after the assignment
    start = text.find("_COMMON_ENGLISH_WORDS = {")
    if start == -1:
        # Fallback: use the stub's own set
        return set()
    # Find matching closing brace
    brace_count = 0
    end = start
    for i in range(start, len(text)):
        if text[i] == "{":
            brace_count += 1
        elif text[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                end = i + 1
                break
    set_literal = text[text.index("{", start) : end]
    return ast.literal_eval(set_literal)


_TickerExtractor._COMMON_ENGLISH_WORDS = _load_common_english_words()


@pytest.fixture
def extractor():
    """Create a _TickerExtractor for testing."""
    return _TickerExtractor()


@pytest.fixture
def learning_mixin():
    """Create a SwarmLearningMixin instance for testing _classify_error."""
    from Jotty.core.intelligence.orchestration.swarms._base._learning_mixin import (
        SwarmLearningMixin,
    )

    instance = SwarmLearningMixin.__new__(SwarmLearningMixin)
    return instance


# ===========================================================================
# Ticker Extraction Tests
# ===========================================================================


class TestTickerExtraction:
    """Test improved ticker extraction with multi-signal approach."""

    @pytest.mark.unit
    def test_explicit_ticker_ns(self, extractor):
        """RELIANCE.NS should extract RELIANCE."""
        assert extractor._extract_ticker("Research RELIANCE.NS") == "RELIANCE"

    @pytest.mark.unit
    def test_explicit_ticker_bo(self, extractor):
        """INFY.BO should extract INFY."""
        assert extractor._extract_ticker("Analyze INFY.BO stock") == "INFY"

    @pytest.mark.unit
    def test_explicit_ticker_dollar(self, extractor):
        """$AAPL should extract AAPL."""
        assert extractor._extract_ticker("Analyze $AAPL") == "AAPL"

    @pytest.mark.unit
    def test_explicit_ticker_nse_prefix(self, extractor):
        """NSE:INFY should extract INFY."""
        assert extractor._extract_ticker("Research NSE:INFY") == "INFY"

    @pytest.mark.unit
    def test_explicit_ticker_nasdaq_prefix(self, extractor):
        """NASDAQ:GOOGL should extract GOOGL."""
        assert extractor._extract_ticker("Show NASDAQ:GOOGL analysis") == "GOOGL"

    @pytest.mark.unit
    def test_all_caps_ticker(self, extractor):
        """ALL-CAPS word that's not a common English word should be treated as ticker."""
        assert extractor._extract_ticker("Research INFY stock") == "INFY"

    @pytest.mark.unit
    def test_all_caps_ticker_tcs(self, extractor):
        """TCS should be extracted as a ticker."""
        assert extractor._extract_ticker("Analyze TCS") == "TCS"

    @pytest.mark.unit
    def test_no_ticker_in_question(self, extractor):
        """General question without a ticker should return None."""
        result = extractor._extract_ticker(
            "Explain the key differences between transformer and LSTM architectures"
        )
        assert result is None

    @pytest.mark.unit
    def test_no_ticker_general_query(self, extractor):
        """Question about ML should return None."""
        assert extractor._extract_ticker("What is machine learning?") is None

    @pytest.mark.unit
    def test_common_english_word_rejected(self, extractor):
        """BEST is a common English word, not a ticker."""
        assert extractor._extract_ticker("BEST stocks to buy") is None

    @pytest.mark.unit
    def test_explain_rejected(self, extractor):
        """EXPLAIN is a skip word, should not be returned as ticker."""
        assert extractor._extract_ticker("EXPLAIN how stock markets work") is None

    @pytest.mark.unit
    def test_mixed_case_not_ticker(self, extractor):
        """Mixed-case words are not tickers."""
        assert extractor._extract_ticker("Research about inflation trends") is None

    @pytest.mark.unit
    def test_empty_query(self, extractor):
        """Empty query should return None."""
        assert extractor._extract_ticker("") is None

    @pytest.mark.unit
    def test_ticker_with_comma(self, extractor):
        """Ticker followed by comma should be cleaned."""
        assert extractor._extract_ticker("Research HDFC, please") == "HDFC"

    @pytest.mark.unit
    def test_lowercase_ticker_pattern(self, extractor):
        """Explicit patterns should be case-insensitive."""
        assert extractor._extract_ticker("Research reliance.ns") == "RELIANCE"

    @pytest.mark.unit
    def test_all_lowercase_no_ticker(self, extractor):
        """All-lowercase query with no patterns should return None."""
        assert extractor._extract_ticker("tell me about recent trends in ai stocks") is None

    @pytest.mark.unit
    def test_key_differences(self, extractor):
        """The original bug: 'key differences between...' should NOT extract LSTM."""
        result = extractor._extract_ticker(
            "Explain the key differences between transformer and LSTM architectures"
        )
        assert result is None

    @pytest.mark.unit
    def test_compare_rejected(self, extractor):
        """COMPARE is a skip word."""
        assert extractor._extract_ticker("COMPARE two stocks") is None


# ===========================================================================
# Error Classification Tests
# ===========================================================================


class TestErrorClassification:
    """Test error categorization in learning."""

    @pytest.mark.unit
    def test_success_returns_none(self, learning_mixin):
        """Successful execution should have no error type."""
        result = _MockSwarmResult(success=True)
        assert learning_mixin._classify_error(True, result, 5.0) is None

    @pytest.mark.unit
    def test_invalid_input_classified(self, learning_mixin):
        """'No valid stock ticker' should be classified as invalid_input."""
        result = _MockSwarmResult(success=False, error="No valid stock ticker found in query")
        assert learning_mixin._classify_error(False, result, 5.0) == "invalid_input"

    @pytest.mark.unit
    def test_not_found_classified(self, learning_mixin):
        """'not found' errors should be classified as invalid_input."""
        result = _MockSwarmResult(success=False, error="Ticker XYZ not found")
        assert learning_mixin._classify_error(False, result, 5.0) == "invalid_input"

    @pytest.mark.unit
    def test_timeout_by_message(self, learning_mixin):
        """Timeout error message should classify as timeout."""
        result = _MockSwarmResult(success=False, error="Request timeout exceeded")
        assert learning_mixin._classify_error(False, result, 5.0) == "timeout"

    @pytest.mark.unit
    def test_timeout_by_duration(self, learning_mixin):
        """Execution > 120s should classify as timeout regardless of message."""
        result = _MockSwarmResult(success=False, error="Something failed")
        assert learning_mixin._classify_error(False, result, 130.0) == "timeout"

    @pytest.mark.unit
    def test_infrastructure_classified(self, learning_mixin):
        """API/connection errors should classify as infrastructure."""
        result = _MockSwarmResult(success=False, error="API rate limit exceeded")
        assert learning_mixin._classify_error(False, result, 5.0) == "infrastructure"

    @pytest.mark.unit
    def test_connection_error_classified(self, learning_mixin):
        """Connection error should classify as infrastructure."""
        result = _MockSwarmResult(success=False, error="Connection refused")
        assert learning_mixin._classify_error(False, result, 5.0) == "infrastructure"

    @pytest.mark.unit
    def test_auth_classified(self, learning_mixin):
        """Permission/auth errors should classify as authentication."""
        result = _MockSwarmResult(success=False, error="Permission denied: invalid token")
        assert learning_mixin._classify_error(False, result, 5.0) == "authentication"

    @pytest.mark.unit
    def test_generic_failure(self, learning_mixin):
        """Unknown errors should fall back to execution_failure."""
        result = _MockSwarmResult(success=False, error="Something unexpected happened")
        assert learning_mixin._classify_error(False, result, 5.0) == "execution_failure"

    @pytest.mark.unit
    def test_none_result(self, learning_mixin):
        """None result should classify as execution_failure."""
        assert learning_mixin._classify_error(False, None, 5.0) == "execution_failure"

    @pytest.mark.unit
    def test_result_without_error_attr(self, learning_mixin):
        """Result without error attribute should not crash."""
        result = object()  # No .error attribute
        assert learning_mixin._classify_error(False, result, 5.0) == "execution_failure"


# ===========================================================================
# Outcome Storage Tests
# ===========================================================================


class TestOutcomeStorage:
    """Test store_with_outcome integration in _record_trace."""

    @pytest.mark.unit
    def test_failure_stored_with_outcome(self):
        """Failures should be stored via store_with_outcome with outcome='failure'."""
        from Jotty.core.intelligence.orchestration.swarms._base._learning_mixin import (
            SwarmLearningMixin,
        )
        from Jotty.core.intelligence.orchestration.swarms._base.swarm_types import AgentRole

        instance = SwarmLearningMixin.__new__(SwarmLearningMixin)
        instance._traces = []

        # Mock memory
        mock_memory = MagicMock()
        instance._memory = mock_memory

        # Mock config
        mock_config = MagicMock()
        mock_config.name = "test_swarm"
        mock_config.domain = "test"
        mock_config.enable_learning = True
        instance.config = mock_config

        # Mock SwarmIntelligence
        instance._swarm_intelligence = None

        instance._record_trace(
            agent_name="TestAgent",
            agent_role=AgentRole.ACTOR,
            input_data={"query": "test"},
            output_data={"result": "failed"},
            execution_time=5.0,
            success=False,
            error="Something failed",
        )

        # Verify store_with_outcome was called (not plain store)
        mock_memory.store_with_outcome.assert_called_once()
        call_kwargs = mock_memory.store_with_outcome.call_args
        assert call_kwargs[1]["outcome"] == "failure"
        assert call_kwargs[1]["domain"] == "test"

    @pytest.mark.unit
    def test_success_stored_with_outcome(self):
        """Successes should be stored via store_with_outcome with outcome='success'."""
        from Jotty.core.intelligence.orchestration.swarms._base._learning_mixin import (
            SwarmLearningMixin,
        )
        from Jotty.core.intelligence.orchestration.swarms._base.swarm_types import AgentRole

        instance = SwarmLearningMixin.__new__(SwarmLearningMixin)
        instance._traces = []

        mock_memory = MagicMock()
        instance._memory = mock_memory

        mock_config = MagicMock()
        mock_config.name = "test_swarm"
        mock_config.domain = "test"
        mock_config.enable_learning = True
        instance.config = mock_config

        instance._swarm_intelligence = None

        instance._record_trace(
            agent_name="TestAgent",
            agent_role=AgentRole.ACTOR,
            input_data={"query": "test"},
            output_data={"result": "success"},
            execution_time=3.0,
            success=True,
        )

        mock_memory.store_with_outcome.assert_called_once()
        call_kwargs = mock_memory.store_with_outcome.call_args
        assert call_kwargs[1]["outcome"] == "success"


# ===========================================================================
# Richer Feedback Tests
# ===========================================================================


class TestRicherFeedback:
    """Test that richer feedback params are passed through the chain."""

    @pytest.mark.unit
    def test_send_executor_feedback_passes_error_message(self):
        """_send_executor_feedback should pass error_message and input_query."""
        from Jotty.core.intelligence.orchestration.swarms._base.swarm_learning import (
            SwarmLearning,
        )

        # Create concrete subclass to bypass ABC restriction
        class _ConcreteSwarm(SwarmLearning):
            async def execute(self, *args, **kwargs):
                pass

        instance = _ConcreteSwarm.__new__(_ConcreteSwarm)

        # Mock SwarmIntelligence
        mock_si = MagicMock()
        instance._swarm_intelligence = mock_si

        # Mock config
        mock_config = MagicMock()
        mock_config.name = "test_swarm"
        instance.config = mock_config

        instance._send_executor_feedback(
            task_type="research",
            success=False,
            tools_used=["web_search"],
            execution_time=5.0,
            error_type="invalid_input",
            error_message="No valid stock ticker found",
            input_query="Explain transformers",
        )

        # Verify receive_executor_feedback was called with new params
        mock_si.receive_executor_feedback.assert_called_once()
        call_kwargs = mock_si.receive_executor_feedback.call_args[1]
        assert call_kwargs["error_message"] == "No valid stock ticker found"
        assert call_kwargs["input_query"] == "Explain transformers"
        assert call_kwargs["error_type"] == "invalid_input"


# ===========================================================================
# Knowledge Mixin Tests
# ===========================================================================


class TestKnowledgeMixin:
    """Test causal retrieval and consolidated knowledge in knowledge mixin."""

    @pytest.mark.unit
    def test_analyze_prior_failures_includes_causal(self):
        """_analyze_prior_failures should include causal memory patterns."""
        from Jotty.core.intelligence.orchestration.swarms._base._knowledge_mixin import (
            SwarmKnowledgeMixin,
        )

        instance = SwarmKnowledgeMixin.__new__(SwarmKnowledgeMixin)

        # Mock config
        mock_config = MagicMock()
        mock_config.name = "test_swarm"
        mock_config.domain = "test"
        instance.config = mock_config

        # Mock evaluation history (empty)
        instance._evaluation_history = MagicMock()
        instance._evaluation_history.get_failures.return_value = []

        # Mock SwarmIntelligence (empty collective memory)
        mock_si = MagicMock()
        mock_si.collective_memory = []
        instance._swarm_intelligence = mock_si

        # Mock memory with causal patterns
        mock_causal = MagicMock()
        mock_causal.cause = "Invalid ticker input"
        mock_causal.effect = "Research swarm fails with UNKNOWN ticker"
        mock_causal.confidence = 0.85

        mock_memory = MagicMock()
        mock_memory.retrieve.return_value = []
        mock_memory.retrieve_causal.return_value = [mock_causal]
        instance._memory = mock_memory

        failures = instance._analyze_prior_failures()

        # Should have one causal failure pattern
        causal_failures = [f for f in failures if f.get("source") == "causal_memory"]
        assert len(causal_failures) == 1
        assert causal_failures[0]["cause"] == "Invalid ticker input"
        assert causal_failures[0]["effect"] == "Research swarm fails with UNKNOWN ticker"
        assert causal_failures[0]["confidence"] == 0.85

    @pytest.mark.unit
    def test_build_learned_context_includes_consolidated(self):
        """_build_learned_context_string should include consolidated memory wisdom."""
        from Jotty.core.intelligence.orchestration.swarms._base._knowledge_mixin import (
            SwarmKnowledgeMixin,
        )

        instance = SwarmKnowledgeMixin.__new__(SwarmKnowledgeMixin)

        # Mock memory with consolidated knowledge
        mock_memory = MagicMock()
        mock_memory.get_consolidated_knowledge.return_value = (
            "Pattern: Always validate ticker before research"
        )
        instance._memory = mock_memory

        # Mock config
        mock_config = MagicMock()
        mock_config.name = "test_swarm"
        instance.config = mock_config

        # Set up minimal learned context
        instance._learned_context = {
            "has_learning": True,
            "tool_performance": {"web_search": 0.9},
            "agent_scores": {},
            "weak_tools": [],
            "strong_tools": [{"tool": "web_search", "success_rate": 0.9}],
            "recommendations": [],
            "expert_knowledge": [],
            "prior_failures": [],
            "score_trends": {},
            "coordination": {},
        }
        instance._swarm_intelligence = None

        text = instance._build_learned_context_string()
        assert "Memory Wisdom" in text
        assert "validate ticker" in text
