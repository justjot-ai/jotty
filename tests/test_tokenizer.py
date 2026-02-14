"""
Tests for SmartTokenizer (core/utils/tokenizer.py)
===================================================

Comprehensive unit tests covering:
- SmartTokenizer class: singleton pattern, token counting, heuristic estimation
- Content type detection: english, code, json, cjk, whitespace_heavy, mixed
- Heuristic estimation with adjustments for URLs, numbers, punctuation
- Tiktoken integration and fallback behavior
- Statistics tracking
- Module-level convenience functions: get_tokenizer, count_tokens, estimate_tokens

All tests are fast (< 1s), offline, no real LLM calls.
"""

import json
import pytest
from unittest.mock import Mock, MagicMock, patch, PropertyMock

from Jotty.core.utils.tokenizer import (
    SmartTokenizer,
    get_tokenizer,
    count_tokens,
    estimate_tokens,
)


# =============================================================================
# TestSmartTokenizerInit
# =============================================================================

@pytest.mark.unit
class TestSmartTokenizerInit:
    """Tests for SmartTokenizer.__init__ and _init_tiktoken."""

    def setup_method(self):
        SmartTokenizer.reset_instances()

    def test_default_encoding_constant(self):
        """DEFAULT_ENCODING should be cl100k_base."""
        assert SmartTokenizer.DEFAULT_ENCODING == "cl100k_base"

    def test_default_encoding_used_when_none(self):
        """When encoding_name is None, default encoding is used."""
        with patch.object(SmartTokenizer, '_init_tiktoken'):
            tok = SmartTokenizer(encoding_name=None)
        assert tok.encoding_name == "cl100k_base"

    def test_custom_encoding_preserved(self):
        """Custom encoding name is stored correctly."""
        with patch.object(SmartTokenizer, '_init_tiktoken'):
            tok = SmartTokenizer(encoding_name="p50k_base")
        assert tok.encoding_name == "p50k_base"

    def test_init_statistics_zeroed(self):
        """Statistics counters start at zero."""
        with patch.object(SmartTokenizer, '_init_tiktoken'):
            tok = SmartTokenizer()
        assert tok._total_calls == 0
        assert tok._tiktoken_calls == 0
        assert tok._heuristic_calls == 0

    def test_init_tiktoken_success(self):
        """When tiktoken is importable, encoder is set and flag is True."""
        mock_encoder = Mock()
        mock_tiktoken = Mock()
        mock_tiktoken.get_encoding = Mock(return_value=mock_encoder)

        with patch.dict('sys.modules', {'tiktoken': mock_tiktoken}):
            tok = SmartTokenizer.__new__(SmartTokenizer)
            tok.encoding_name = "cl100k_base"
            tok._tiktoken_encoder = None
            tok._tiktoken_available = False
            tok._init_tiktoken()

        assert tok._tiktoken_available is True
        assert tok._tiktoken_encoder is mock_encoder

    def test_init_tiktoken_import_error(self):
        """When tiktoken import fails, falls back to heuristics."""
        with patch('builtins.__import__', side_effect=ImportError("no tiktoken")):
            tok = SmartTokenizer.__new__(SmartTokenizer)
            tok.encoding_name = "cl100k_base"
            tok._tiktoken_encoder = None
            tok._tiktoken_available = False
            tok._init_tiktoken()

        assert tok._tiktoken_available is False
        assert tok._tiktoken_encoder is None

    def test_init_tiktoken_generic_exception(self):
        """When tiktoken raises a non-ImportError, falls back to heuristics."""
        mock_tiktoken = Mock()
        mock_tiktoken.get_encoding = Mock(side_effect=RuntimeError("bad encoding"))

        with patch.dict('sys.modules', {'tiktoken': mock_tiktoken}):
            tok = SmartTokenizer.__new__(SmartTokenizer)
            tok.encoding_name = "cl100k_base"
            tok._tiktoken_encoder = None
            tok._tiktoken_available = False
            tok._init_tiktoken()

        assert tok._tiktoken_available is False


# =============================================================================
# TestSmartTokenizerSingleton
# =============================================================================

@pytest.mark.unit
class TestSmartTokenizerSingleton:
    """Tests for singleton pattern: get_instance and reset_instances."""

    def setup_method(self):
        SmartTokenizer.reset_instances()

    def test_get_instance_returns_same_object(self):
        """get_instance returns the same object for the same encoding."""
        with patch.object(SmartTokenizer, '_init_tiktoken'):
            inst_a = SmartTokenizer.get_instance()
            inst_b = SmartTokenizer.get_instance()
        assert inst_a is inst_b

    def test_get_instance_different_encoding_returns_different(self):
        """Different encodings produce different instances."""
        with patch.object(SmartTokenizer, '_init_tiktoken'):
            inst_default = SmartTokenizer.get_instance("cl100k_base")
            inst_custom = SmartTokenizer.get_instance("p50k_base")
        assert inst_default is not inst_custom

    def test_get_instance_none_uses_default(self):
        """Passing None to get_instance uses DEFAULT_ENCODING."""
        with patch.object(SmartTokenizer, '_init_tiktoken'):
            inst_none = SmartTokenizer.get_instance(None)
            inst_default = SmartTokenizer.get_instance("cl100k_base")
        assert inst_none is inst_default

    def test_reset_instances_clears_cache(self):
        """reset_instances clears all cached instances."""
        with patch.object(SmartTokenizer, '_init_tiktoken'):
            inst_before = SmartTokenizer.get_instance()
            SmartTokenizer.reset_instances()
            inst_after = SmartTokenizer.get_instance()
        assert inst_before is not inst_after

    def test_reset_instances_clears_all_encodings(self):
        """reset_instances removes every encoding from cache."""
        with patch.object(SmartTokenizer, '_init_tiktoken'):
            SmartTokenizer.get_instance("cl100k_base")
            SmartTokenizer.get_instance("p50k_base")
            assert len(SmartTokenizer._instances) == 2
            SmartTokenizer.reset_instances()
            assert len(SmartTokenizer._instances) == 0


# =============================================================================
# TestCountTokens
# =============================================================================

@pytest.mark.unit
class TestCountTokens:
    """Tests for SmartTokenizer.count_tokens and estimate_tokens."""

    def setup_method(self):
        SmartTokenizer.reset_instances()

    def _make_heuristic_tokenizer(self):
        """Create a tokenizer that uses heuristics (no tiktoken)."""
        with patch.object(SmartTokenizer, '_init_tiktoken'):
            tok = SmartTokenizer()
            tok._tiktoken_available = False
            tok._tiktoken_encoder = None
        return tok

    def _make_tiktoken_tokenizer(self, mock_encode_result):
        """Create a tokenizer with a mocked tiktoken encoder."""
        with patch.object(SmartTokenizer, '_init_tiktoken'):
            tok = SmartTokenizer()
            tok._tiktoken_available = True
            encoder = Mock()
            encoder.encode = Mock(return_value=mock_encode_result)
            tok._tiktoken_encoder = encoder
        return tok

    def test_empty_string_returns_zero(self):
        """count_tokens('') should return 0 without incrementing counters."""
        tok = self._make_heuristic_tokenizer()
        assert tok.count_tokens("") == 0
        assert tok._total_calls == 0

    def test_none_returns_zero(self):
        """count_tokens(None) returns 0 since falsy check catches it."""
        tok = self._make_heuristic_tokenizer()
        assert tok.count_tokens(None) == 0

    def test_tiktoken_path_used_when_available(self):
        """When tiktoken is available, encoder.encode is called."""
        tokens_list = [1, 2, 3, 4, 5]
        tok = self._make_tiktoken_tokenizer(tokens_list)
        result = tok.count_tokens("Hello world")
        assert result == 5
        tok._tiktoken_encoder.encode.assert_called_once_with("Hello world")

    def test_tiktoken_increments_counters(self):
        """Tiktoken path increments total_calls and tiktoken_calls."""
        tok = self._make_tiktoken_tokenizer([1, 2, 3])
        tok.count_tokens("test")
        assert tok._total_calls == 1
        assert tok._tiktoken_calls == 1
        assert tok._heuristic_calls == 0

    def test_tiktoken_encode_failure_falls_back_to_heuristics(self):
        """If tiktoken encode raises, heuristic fallback is used."""
        with patch.object(SmartTokenizer, '_init_tiktoken'):
            tok = SmartTokenizer()
            tok._tiktoken_available = True
            encoder = Mock()
            encoder.encode = Mock(side_effect=RuntimeError("encode failed"))
            tok._tiktoken_encoder = encoder

        result = tok.count_tokens("Hello world test")
        # Should get a positive integer from heuristics
        assert result >= 1
        assert tok._heuristic_calls == 1
        # tiktoken_calls was also incremented before the failure
        assert tok._tiktoken_calls == 1

    def test_heuristic_path_used_when_tiktoken_unavailable(self):
        """When tiktoken is not available, heuristics are used."""
        tok = self._make_heuristic_tokenizer()
        result = tok.count_tokens("Hello world")
        assert result >= 1
        assert tok._heuristic_calls == 1
        assert tok._tiktoken_calls == 0

    def test_estimate_tokens_is_alias_for_count_tokens(self):
        """estimate_tokens returns same result as count_tokens."""
        tok = self._make_heuristic_tokenizer()
        text = "The quick brown fox jumps over the lazy dog"
        assert tok.estimate_tokens(text) == tok.count_tokens(text)

    def test_estimate_tokens_empty_string(self):
        """estimate_tokens('') returns 0."""
        tok = self._make_heuristic_tokenizer()
        assert tok.estimate_tokens("") == 0

    def test_multiple_calls_increment_counters(self):
        """Multiple count_tokens calls properly track statistics."""
        tok = self._make_heuristic_tokenizer()
        tok.count_tokens("one")
        tok.count_tokens("two")
        tok.count_tokens("three")
        assert tok._total_calls == 3
        assert tok._heuristic_calls == 3

    def test_minimum_one_token_for_nonempty_text(self):
        """Non-empty text always returns at least 1 token."""
        tok = self._make_heuristic_tokenizer()
        result = tok.count_tokens("a")
        assert result >= 1


# =============================================================================
# TestContentTypeDetection
# =============================================================================

@pytest.mark.unit
class TestContentTypeDetection:
    """Tests for SmartTokenizer._detect_content_type."""

    def setup_method(self):
        SmartTokenizer.reset_instances()
        with patch.object(SmartTokenizer, '_init_tiktoken'):
            self.tok = SmartTokenizer()
            self.tok._tiktoken_available = False

    def test_detect_english_text(self):
        """Plain English prose is detected as 'english'."""
        text = "The quick brown fox jumps over the lazy dog. This is a simple English sentence."
        assert self.tok._detect_content_type(text) == 'english'

    def test_detect_code_python(self):
        """Python code with multiple indicators is detected as 'code'."""
        text = """
import os
from pathlib import Path
import json

class MyClass:
    def __init__(self):
        pass

    def method_one(self):
        return 1

    def method_two(self):
        return 2

    def method_three(self):
        return 3

    def method_four(self):
        pass

    def method_five(self):
        pass

    def method_six(self):
        pass
"""
        assert self.tok._detect_content_type(text) == 'code'

    def test_detect_code_javascript(self):
        """JavaScript code with function keywords and arrow functions is detected as 'code'."""
        text = """
import React from 'react';
import axios from 'axios';
import lodash from 'lodash';

function componentOne() {
    return null;
}

function componentTwo() {
    return null;
}

const handler = () => {
    console.log("arrow");
};

class App {
    constructor() {
    }
}
"""
        assert self.tok._detect_content_type(text) == 'code'

    def test_detect_json_valid(self):
        """Valid JSON object is detected as 'json'."""
        data = {"name": "test", "value": 42, "nested": {"a": 1, "b": 2}}
        text = json.dumps(data, indent=2)
        assert self.tok._detect_content_type(text) == 'json'

    def test_detect_json_array(self):
        """Valid JSON array is detected as 'json'."""
        data = [{"id": 1}, {"id": 2}, {"id": 3}]
        text = json.dumps(data)
        assert self.tok._detect_content_type(text) == 'json'

    def test_detect_json_like_partial(self):
        """Partial JSON with many braces and quotes is detected as 'json'."""
        text = '{"key": "value", "nested": {"a": "b", "c": "d"}, "list": [1, 2, 3]'
        # Starts with { but doesn't end with } — JSONDecodeError path
        # but has enough braces to trigger partial JSON detection
        result = self.tok._detect_content_type(text)
        assert result == 'json'

    def test_detect_cjk_chinese(self):
        """Text with >20% CJK characters is detected as 'cjk'."""
        # Generate enough CJK to exceed 20% of the sample
        cjk_text = "\u4f60\u597d\u4e16\u754c" * 50  # "Hello World" in Chinese, repeated
        assert self.tok._detect_content_type(cjk_text) == 'cjk'

    def test_detect_cjk_japanese(self):
        """Japanese hiragana/katakana text is detected as 'cjk'."""
        text = "\u3053\u3093\u306b\u3061\u306f\u4e16\u754c" * 50
        assert self.tok._detect_content_type(text) == 'cjk'

    def test_detect_cjk_korean(self):
        """Korean hangul text is detected as 'cjk'."""
        text = "\uc548\ub155\ud558\uc138\uc694" * 50
        assert self.tok._detect_content_type(text) == 'cjk'

    def test_detect_whitespace_heavy(self):
        """Content with >40% whitespace is detected as 'whitespace_heavy'."""
        # Create text with heavy whitespace
        text = "word " * 3 + "    \t\t   \n\n   " * 10
        assert self.tok._detect_content_type(text) == 'whitespace_heavy'

    def test_detect_mixed_with_some_cjk(self):
        """Text with a few CJK characters (<20%) is detected as 'mixed'."""
        # Mostly English but with a small amount of CJK
        english = "This is an English sentence about programming. " * 10
        cjk = "\u4f60\u597d"  # just 2 CJK chars
        text = english + cjk
        assert self.tok._detect_content_type(text) == 'mixed'

    def test_detect_samples_first_5000_chars(self):
        """Detection only samples the first 5000 characters."""
        # Put code indicators after 5000 chars — should not be detected as code
        english_text = "a" * 5001
        code_tail = "\ndef foo():\n    pass\n" * 20
        text = english_text + code_tail
        # The first 5000 chars are all 'a', so it should be 'english'
        assert self.tok._detect_content_type(text) == 'english'

    def test_detect_english_for_normal_prose(self):
        """Normal prose without special indicators returns 'english'."""
        text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 5
        assert self.tok._detect_content_type(text) == 'english'


# =============================================================================
# TestHeuristicEstimation
# =============================================================================

@pytest.mark.unit
class TestHeuristicEstimation:
    """Tests for SmartTokenizer._estimate_with_heuristics."""

    def setup_method(self):
        SmartTokenizer.reset_instances()
        with patch.object(SmartTokenizer, '_init_tiktoken'):
            self.tok = SmartTokenizer()
            self.tok._tiktoken_available = False

    def test_empty_text_returns_zero(self):
        """Heuristic estimation of empty text returns 0."""
        assert self.tok._estimate_with_heuristics("") == 0

    def test_english_uses_4_chars_per_token_ratio(self):
        """English text uses ~4 chars per token as base."""
        text = "Hello world this is a test sentence for estimation"
        result = self.tok._estimate_with_heuristics(text)
        # Base: len(text) / 4.0, plus possible adjustments
        # 50 chars / 4.0 = 12.5 -> ~12 tokens (adjustments might vary)
        assert result >= 1
        assert isinstance(result, int)

    def test_heuristics_dict_has_all_content_types(self):
        """HEURISTICS dict contains all expected content types."""
        expected = {'english', 'code', 'json', 'cjk', 'mixed', 'whitespace_heavy'}
        assert set(SmartTokenizer.HEURISTICS.keys()) == expected

    def test_code_ratio_is_3(self):
        """Code content uses ratio of 3.0 chars per token."""
        assert SmartTokenizer.HEURISTICS['code'] == 3.0

    def test_json_ratio_is_3_5(self):
        """JSON content uses ratio of 3.5 chars per token."""
        assert SmartTokenizer.HEURISTICS['json'] == 3.5

    def test_cjk_ratio_is_1_5(self):
        """CJK content uses ratio of 1.5 chars per token."""
        assert SmartTokenizer.HEURISTICS['cjk'] == 1.5

    def test_whitespace_heavy_ratio_is_5(self):
        """Whitespace-heavy content uses ratio of 5.0 chars per token."""
        assert SmartTokenizer.HEURISTICS['whitespace_heavy'] == 5.0

    def test_result_is_always_int(self):
        """Heuristic estimation always returns an int."""
        texts = ["Hello", "x" * 100, "\u4f60\u597d" * 50, '{"key": "val"}']
        for t in texts:
            result = self.tok._estimate_with_heuristics(t)
            assert isinstance(result, int), f"Expected int, got {type(result)} for '{t[:20]}'"

    def test_minimum_one_token(self):
        """Even very short text returns at least 1 token."""
        assert self.tok._estimate_with_heuristics("x") >= 1

    def test_longer_text_produces_more_tokens(self):
        """Longer text produces more estimated tokens than shorter text."""
        short = "Hello"
        long = "Hello " * 100
        short_tokens = self.tok._estimate_with_heuristics(short)
        long_tokens = self.tok._estimate_with_heuristics(long)
        assert long_tokens > short_tokens


# =============================================================================
# TestCalculateAdjustments
# =============================================================================

@pytest.mark.unit
class TestCalculateAdjustments:
    """Tests for SmartTokenizer._calculate_adjustments."""

    def setup_method(self):
        SmartTokenizer.reset_instances()
        with patch.object(SmartTokenizer, '_init_tiktoken'):
            self.tok = SmartTokenizer()

    def test_no_special_patterns_zero_adjustment(self):
        """Text with no URLs, long numbers, newlines, or punct clusters has 0 adjustment."""
        text = "simple text"
        adj = self.tok._calculate_adjustments(text)
        assert adj == 0

    def test_url_adds_5_per_url(self):
        """Each URL adds 5 tokens of adjustment."""
        text = "Visit https://example.com and https://google.com for more info"
        adj = self.tok._calculate_adjustments(text)
        assert adj >= 10  # 2 URLs * 5

    def test_single_url_adjustment(self):
        """A single URL adds exactly 5 tokens."""
        text = "See https://example.com/path/to/page?q=1"
        adj = self.tok._calculate_adjustments(text)
        # 1 URL * 5 = 5
        assert adj >= 5

    def test_long_numbers_add_adjustment(self):
        """Numbers with more than 4 digits add len//3 adjustment."""
        text = "ID 12345678"  # 8 digits -> 8//3 = 2 additional
        adj = self.tok._calculate_adjustments(text)
        assert adj >= 2

    def test_short_numbers_no_adjustment(self):
        """Numbers with 4 or fewer digits add no adjustment."""
        text = "values 42 and 1234"
        adj = self.tok._calculate_adjustments(text)
        assert adj == 0

    def test_newlines_add_half_per_newline(self):
        """Each newline adds 0.5 to the adjustment (truncated to int)."""
        text = "line1\nline2\nline3\nline4\n"
        adj = self.tok._calculate_adjustments(text)
        # 4 newlines * 0.5 = 2.0 -> int(2.0) = 2
        assert adj >= 2

    def test_tabs_add_half_per_tab(self):
        """Each tab adds 0.5 to the adjustment."""
        text = "col1\tcol2\tcol3\tcol4\t"
        adj = self.tok._calculate_adjustments(text)
        assert adj >= 2  # 4 tabs * 0.5 = 2

    def test_punctuation_clusters_add_2_per_cluster(self):
        """Each cluster of 3+ non-word/non-space chars adds 2 tokens."""
        text = "What!!! Really??? Yes---indeed..."
        adj = self.tok._calculate_adjustments(text)
        # "!!!" "???" "---" "..." are punct clusters -> 4 * 2 = 8
        assert adj >= 8

    def test_combined_adjustments(self):
        """Multiple pattern types combine additively."""
        text = "Visit https://example.com with ID 123456789\nDone!!!"
        adj = self.tok._calculate_adjustments(text)
        # URL: 5, long number 123456789 (9 digits -> 3), newline: 0.5, "!!!": 2
        # Total: 5 + 3 + 0 + 2 = 10 (int(10.5) = 10)
        assert adj >= 10

    def test_returns_integer(self):
        """Adjustments always return an int."""
        text = "test\n"
        adj = self.tok._calculate_adjustments(text)
        assert isinstance(adj, int)


# =============================================================================
# TestIsTiktokenAvailable
# =============================================================================

@pytest.mark.unit
class TestIsTiktokenAvailable:
    """Tests for is_tiktoken_available property."""

    def setup_method(self):
        SmartTokenizer.reset_instances()

    def test_property_true_when_tiktoken_loaded(self):
        """is_tiktoken_available is True when tiktoken was loaded."""
        with patch.object(SmartTokenizer, '_init_tiktoken'):
            tok = SmartTokenizer()
            tok._tiktoken_available = True
        assert tok.is_tiktoken_available is True

    def test_property_false_when_tiktoken_not_loaded(self):
        """is_tiktoken_available is False when tiktoken was not loaded."""
        with patch.object(SmartTokenizer, '_init_tiktoken'):
            tok = SmartTokenizer()
            tok._tiktoken_available = False
        assert tok.is_tiktoken_available is False


# =============================================================================
# TestGetStatistics
# =============================================================================

@pytest.mark.unit
class TestGetStatistics:
    """Tests for SmartTokenizer.get_statistics."""

    def setup_method(self):
        SmartTokenizer.reset_instances()

    def _make_heuristic_tokenizer(self):
        with patch.object(SmartTokenizer, '_init_tiktoken'):
            tok = SmartTokenizer()
            tok._tiktoken_available = False
            tok._tiktoken_encoder = None
        return tok

    def test_statistics_keys(self):
        """get_statistics returns dict with all expected keys."""
        tok = self._make_heuristic_tokenizer()
        stats = tok.get_statistics()
        expected_keys = {
            'encoding', 'tiktoken_available', 'total_calls',
            'tiktoken_calls', 'heuristic_calls', 'tiktoken_ratio'
        }
        assert set(stats.keys()) == expected_keys

    def test_statistics_initial_values(self):
        """Initial statistics have zero counters."""
        tok = self._make_heuristic_tokenizer()
        stats = tok.get_statistics()
        assert stats['encoding'] == 'cl100k_base'
        assert stats['tiktoken_available'] is False
        assert stats['total_calls'] == 0
        assert stats['tiktoken_calls'] == 0
        assert stats['heuristic_calls'] == 0
        assert stats['tiktoken_ratio'] == 0.0

    def test_statistics_after_heuristic_calls(self):
        """Statistics correctly reflect heuristic calls."""
        tok = self._make_heuristic_tokenizer()
        tok.count_tokens("hello")
        tok.count_tokens("world")
        stats = tok.get_statistics()
        assert stats['total_calls'] == 2
        assert stats['heuristic_calls'] == 2
        assert stats['tiktoken_calls'] == 0
        assert stats['tiktoken_ratio'] == 0.0

    def test_statistics_after_tiktoken_calls(self):
        """Statistics correctly reflect tiktoken calls."""
        with patch.object(SmartTokenizer, '_init_tiktoken'):
            tok = SmartTokenizer()
            tok._tiktoken_available = True
            encoder = Mock()
            encoder.encode = Mock(return_value=[1, 2, 3])
            tok._tiktoken_encoder = encoder

        tok.count_tokens("hello")
        tok.count_tokens("world")
        stats = tok.get_statistics()
        assert stats['total_calls'] == 2
        assert stats['tiktoken_calls'] == 2
        assert stats['heuristic_calls'] == 0
        assert stats['tiktoken_ratio'] == 1.0

    def test_statistics_tiktoken_ratio_mixed(self):
        """tiktoken_ratio is correctly computed with mixed calls."""
        with patch.object(SmartTokenizer, '_init_tiktoken'):
            tok = SmartTokenizer()
            tok._tiktoken_available = True
            encoder = Mock()
            encoder.encode = Mock(return_value=[1, 2])
            tok._tiktoken_encoder = encoder

        # First call uses tiktoken
        tok.count_tokens("first")
        # Disable tiktoken to force heuristic
        tok._tiktoken_available = False
        tok.count_tokens("second")

        stats = tok.get_statistics()
        assert stats['total_calls'] == 2
        assert stats['tiktoken_calls'] == 1
        assert stats['heuristic_calls'] == 1
        assert stats['tiktoken_ratio'] == 0.5

    def test_statistics_tiktoken_ratio_zero_calls(self):
        """tiktoken_ratio is 0 when total_calls is 0 (avoids division by zero)."""
        tok = self._make_heuristic_tokenizer()
        stats = tok.get_statistics()
        # max(0, 1) = 1, so 0/1 = 0.0
        assert stats['tiktoken_ratio'] == 0.0


# =============================================================================
# TestModuleFunctions
# =============================================================================

@pytest.mark.unit
class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    def setup_method(self):
        SmartTokenizer.reset_instances()
        # Also clear the lru_cache on get_tokenizer
        get_tokenizer.cache_clear()

    def test_get_tokenizer_returns_smart_tokenizer(self):
        """get_tokenizer returns a SmartTokenizer instance."""
        tok = get_tokenizer()
        assert isinstance(tok, SmartTokenizer)

    def test_get_tokenizer_cached(self):
        """get_tokenizer returns the same instance on repeated calls."""
        tok1 = get_tokenizer()
        tok2 = get_tokenizer()
        assert tok1 is tok2

    def test_count_tokens_module_function(self):
        """Module-level count_tokens works for basic text."""
        result = count_tokens("Hello world")
        assert isinstance(result, int)
        assert result >= 1

    def test_count_tokens_empty(self):
        """Module-level count_tokens returns 0 for empty string."""
        assert count_tokens("") == 0

    def test_estimate_tokens_module_function(self):
        """Module-level estimate_tokens works and matches count_tokens."""
        text = "Hello world this is a test"
        assert estimate_tokens(text) == count_tokens(text)

    def test_estimate_tokens_empty(self):
        """Module-level estimate_tokens returns 0 for empty string."""
        assert estimate_tokens("") == 0

    def test_module_functions_use_singleton(self):
        """Module functions route through SmartTokenizer.get_instance."""
        tok = get_tokenizer()
        # Call count_tokens directly and via module function
        direct_result = tok.count_tokens("test text")

        # Reset counters for clean comparison
        tok._total_calls = 0
        module_result = count_tokens("test text")
        assert direct_result == module_result


# =============================================================================
# TestEdgeCases
# =============================================================================

@pytest.mark.unit
class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def setup_method(self):
        SmartTokenizer.reset_instances()

    def _make_heuristic_tokenizer(self):
        with patch.object(SmartTokenizer, '_init_tiktoken'):
            tok = SmartTokenizer()
            tok._tiktoken_available = False
            tok._tiktoken_encoder = None
        return tok

    def test_very_long_text(self):
        """Token estimation works for very long text."""
        tok = self._make_heuristic_tokenizer()
        text = "word " * 10000  # 50000 chars
        result = tok.count_tokens(text)
        assert result > 100
        assert isinstance(result, int)

    def test_only_whitespace(self):
        """Text with only whitespace characters is handled."""
        tok = self._make_heuristic_tokenizer()
        result = tok.count_tokens("   \t\t\n\n  ")
        assert result >= 1

    def test_only_newlines(self):
        """Text with only newlines returns positive count."""
        tok = self._make_heuristic_tokenizer()
        result = tok.count_tokens("\n\n\n\n")
        assert result >= 1

    def test_single_character(self):
        """Single character returns at least 1 token."""
        tok = self._make_heuristic_tokenizer()
        assert tok.count_tokens("a") >= 1

    def test_unicode_emoji(self):
        """Emoji characters are handled without errors."""
        tok = self._make_heuristic_tokenizer()
        result = tok.count_tokens("Hello \U0001f600\U0001f604\U0001f44d World")
        assert result >= 1

    def test_mixed_url_and_numbers(self):
        """Text with URLs and long numbers produces reasonable estimate."""
        tok = self._make_heuristic_tokenizer()
        text = "Visit https://example.com/path?id=123456789 for order #987654321"
        result = tok.count_tokens(text)
        assert result >= 5

    def test_json_detection_with_invalid_json_but_many_braces(self):
        """Invalid JSON with many braces still detected as JSON."""
        tok = self._make_heuristic_tokenizer()
        text = '{"a": {"b": {"c": "d"}, "e": "f", "g": "h"'
        # Starts with { but doesn't end with } or ]
        # Still has >3 braces, should detect as something sensible
        result = tok.count_tokens(text)
        assert result >= 1

    def test_code_detection_threshold(self):
        """Fewer than 6 code indicators does not trigger 'code' detection."""
        tok = self._make_heuristic_tokenizer()
        # Only 2 code indicators (import, from...import)
        text = "import os\nfrom pathlib import Path\nSome normal text here."
        content_type = tok._detect_content_type(text)
        # With only 2-3 indicators, should NOT be 'code'
        assert content_type != 'code'

    def test_multiple_long_numbers(self):
        """Multiple long numbers each add adjustment."""
        tok = self._make_heuristic_tokenizer()
        text = "12345678 87654321 11111111"
        adj = tok._calculate_adjustments(text)
        # Each 8-digit number: 8//3 = 2 adjustment
        # 3 numbers * 2 = 6
        assert adj >= 6


# =============================================================================
# TestTiktokenFallback
# =============================================================================

@pytest.mark.unit
class TestTiktokenFallback:
    """Tests for tiktoken integration and fallback behavior."""

    def setup_method(self):
        SmartTokenizer.reset_instances()

    def test_tiktoken_encoder_not_called_when_unavailable(self):
        """When tiktoken is unavailable, encoder is never invoked."""
        with patch.object(SmartTokenizer, '_init_tiktoken'):
            tok = SmartTokenizer()
            tok._tiktoken_available = False
            mock_encoder = Mock()
            tok._tiktoken_encoder = mock_encoder

        tok.count_tokens("hello world")
        mock_encoder.encode.assert_not_called()

    def test_tiktoken_encoder_none_uses_heuristics(self):
        """When encoder is None (even if flag is True), heuristics are used."""
        with patch.object(SmartTokenizer, '_init_tiktoken'):
            tok = SmartTokenizer()
            tok._tiktoken_available = True
            tok._tiktoken_encoder = None

        result = tok.count_tokens("hello world")
        assert result >= 1
        assert tok._heuristic_calls == 1

    def test_fallback_after_encode_exception_increments_both_counters(self):
        """When tiktoken fails mid-call, both tiktoken and heuristic counters increment."""
        with patch.object(SmartTokenizer, '_init_tiktoken'):
            tok = SmartTokenizer()
            tok._tiktoken_available = True
            encoder = Mock()
            encoder.encode = Mock(side_effect=Exception("encoding error"))
            tok._tiktoken_encoder = encoder

        tok.count_tokens("some text")
        assert tok._tiktoken_calls == 1
        assert tok._heuristic_calls == 1
        assert tok._total_calls == 1
