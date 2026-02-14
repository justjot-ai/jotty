"""
Tests for the Utilities Subsystem Facade (Phase 2f).

Verifies budget tracker, circuit breaker, cache, and tokenizer accessors.
All tests run offline.
"""

import pytest


@pytest.mark.unit
class TestUtilsFacade:
    """Tests for utils facade accessor functions."""

    def test_get_budget_tracker_returns_tracker(self):
        from Jotty.core.utils.facade import get_budget_tracker
        from Jotty.core.utils.budget_tracker import BudgetTracker
        tracker = get_budget_tracker()
        assert isinstance(tracker, BudgetTracker)

    def test_get_budget_tracker_with_name(self):
        from Jotty.core.utils.facade import get_budget_tracker
        from Jotty.core.utils.budget_tracker import BudgetTracker
        tracker = get_budget_tracker(name="test-scope")
        assert isinstance(tracker, BudgetTracker)

    def test_get_circuit_breaker_returns_breaker(self):
        from Jotty.core.utils.facade import get_circuit_breaker
        from Jotty.core.utils.timeouts import CircuitBreaker
        breaker = get_circuit_breaker()
        assert isinstance(breaker, CircuitBreaker)

    def test_get_circuit_breaker_with_name(self):
        from Jotty.core.utils.facade import get_circuit_breaker
        from Jotty.core.utils.timeouts import CircuitBreaker
        breaker = get_circuit_breaker(name="test-breaker")
        assert isinstance(breaker, CircuitBreaker)

    def test_get_llm_cache_returns_cache(self):
        from Jotty.core.utils.facade import get_llm_cache
        from Jotty.core.utils.llm_cache import LLMCallCache
        cache = get_llm_cache()
        assert isinstance(cache, LLMCallCache)

    def test_get_tokenizer_returns_tokenizer(self):
        from Jotty.core.utils.facade import get_tokenizer
        from Jotty.core.utils.tokenizer import SmartTokenizer
        tokenizer = get_tokenizer()
        assert isinstance(tokenizer, SmartTokenizer)

    def test_list_components_returns_dict(self):
        from Jotty.core.utils.facade import list_components
        components = list_components()
        assert isinstance(components, dict)
        assert len(components) > 0

    def test_list_components_has_key_classes(self):
        from Jotty.core.utils.facade import list_components
        components = list_components()
        expected = [
            "BudgetTracker",
            "CircuitBreaker",
            "LLMCallCache",
            "SmartTokenizer",
        ]
        for name in expected:
            assert name in components, f"Missing component: {name}"

    def test_list_components_values_are_strings(self):
        from Jotty.core.utils.facade import list_components
        for name, desc in list_components().items():
            assert isinstance(desc, str)
            assert len(desc) > 0


@pytest.mark.unit
class TestUtilsFacadeFromInit:
    """Test facade functions are accessible from __init__."""

    def test_import_get_budget_tracker(self):
        from Jotty.core.utils import get_budget_tracker
        assert callable(get_budget_tracker)

    def test_import_get_circuit_breaker(self):
        from Jotty.core.utils import get_circuit_breaker
        assert callable(get_circuit_breaker)

    def test_import_get_llm_cache(self):
        from Jotty.core.utils import get_llm_cache
        assert callable(get_llm_cache)

    def test_import_get_tokenizer(self):
        from Jotty.core.utils import get_tokenizer
        assert callable(get_tokenizer)
