"""
Unit tests for ContextPriority, ContextChunk, SmartContextManager, and ContextChunker.

Tests cover:
- ContextPriority enum ordering and values
- ContextChunk dataclass defaults and auto token counting
- SmartContextManager initialization, registration, chunk management, priority detection
- ContextChunker import guard (DSPy-dependent)
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from Jotty.core.context.context_manager import ContextPriority, ContextChunk, SmartContextManager

try:
    from Jotty.core.context.chunker import ContextChunker
    HAS_CHUNKER = True
except ImportError:
    HAS_CHUNKER = False


@pytest.mark.unit
class TestContextPriority:
    """Tests for the ContextPriority enum."""

    def test_priority_ordering_by_value(self):
        """CRITICAL < HIGH < MEDIUM < LOW by numeric value."""
        assert ContextPriority.CRITICAL.value < ContextPriority.HIGH.value
        assert ContextPriority.HIGH.value < ContextPriority.MEDIUM.value
        assert ContextPriority.MEDIUM.value < ContextPriority.LOW.value

    def test_priority_exact_values(self):
        """CRITICAL=1, HIGH=2, MEDIUM=3, LOW=4."""
        assert ContextPriority.CRITICAL.value == 1
        assert ContextPriority.HIGH.value == 2
        assert ContextPriority.MEDIUM.value == 3
        assert ContextPriority.LOW.value == 4


@pytest.mark.unit
class TestContextChunk:
    """Tests for the ContextChunk dataclass."""

    @patch("Jotty.core.context.context_manager.SmartTokenizer")
    def test_defaults(self, mock_tokenizer_cls):
        """is_compressed defaults to False and original_tokens defaults to 0 before __post_init__."""
        mock_instance = Mock()
        mock_instance.count_tokens.return_value = 5
        mock_tokenizer_cls.get_instance.return_value = mock_instance

        chunk = ContextChunk(
            content="hello",
            priority=ContextPriority.LOW,
            category="misc",
        )
        assert chunk.is_compressed is False
        # original_tokens gets set to tokens in __post_init__ when not provided
        assert chunk.original_tokens == chunk.tokens

    @patch("Jotty.core.context.context_manager.SmartTokenizer")
    def test_auto_counts_tokens_in_post_init(self, mock_tokenizer_cls):
        """__post_init__ calls SmartTokenizer to auto-count tokens when tokens=0."""
        mock_instance = Mock()
        mock_instance.count_tokens.return_value = 42
        mock_tokenizer_cls.get_instance.return_value = mock_instance

        chunk = ContextChunk(
            content="some content for counting",
            priority=ContextPriority.MEDIUM,
            category="test",
        )
        assert chunk.tokens == 42
        assert chunk.original_tokens == 42
        mock_instance.count_tokens.assert_called_with("some content for counting")


@pytest.mark.unit
class TestSmartContextManager:
    """Tests for SmartContextManager initialization and methods."""

    @patch("Jotty.core.context.context_manager.dspy")
    def test_default_effective_limit(self, mock_dspy):
        """Default effective_limit = int(28000 * 0.85) = 23800."""
        manager = SmartContextManager()
        assert manager.effective_limit == int(28000 * 0.85)
        assert manager.effective_limit == 23800

    @patch("Jotty.core.context.context_manager.dspy")
    def test_custom_max_tokens(self, mock_dspy):
        """Custom max_tokens produces correct effective_limit."""
        manager = SmartContextManager(max_tokens=10000, safety_margin=0.90)
        assert manager.max_tokens == 10000
        assert manager.effective_limit == int(10000 * 0.90)
        assert manager.effective_limit == 9000

    @patch("Jotty.core.context.context_manager.dspy")
    def test_register_goal_stores_goal(self, mock_dspy):
        """register_goal stores the goal string internally."""
        manager = SmartContextManager()
        manager.register_goal("Maximize profit")
        assert manager._current_goal == "Maximize profit"

    @patch("Jotty.core.context.context_manager.dspy")
    def test_register_todo_stores_todo(self, mock_dspy):
        """register_todo stores the todo content internally."""
        manager = SmartContextManager()
        manager.register_todo("Step 1: Download data\nStep 2: Analyze")
        assert manager._current_todo == "Step 1: Download data\nStep 2: Analyze"

    @patch("Jotty.core.context.context_manager.SmartTokenizer")
    @patch("Jotty.core.context.context_manager.dspy")
    def test_add_chunk_appends_to_current_chunks(self, mock_dspy, mock_tokenizer_cls):
        """add_chunk creates a ContextChunk and appends it to current_chunks."""
        mock_instance = Mock()
        mock_instance.count_tokens.return_value = 10
        mock_tokenizer_cls.get_instance.return_value = mock_instance

        manager = SmartContextManager()
        assert len(manager.current_chunks) == 0

        manager.add_chunk("some trajectory data", category="trajectory")
        assert len(manager.current_chunks) == 1
        assert manager.current_chunks[0].content == "some trajectory data"
        assert manager.current_chunks[0].category == "trajectory"

    @patch("Jotty.core.context.context_manager.dspy")
    def test_auto_detect_priority_critical_for_task(self, mock_dspy):
        """_auto_detect_priority returns CRITICAL for 'task' category."""
        manager = SmartContextManager()
        priority = manager._auto_detect_priority("task", "do something")
        assert priority == ContextPriority.CRITICAL

    @patch("Jotty.core.context.context_manager.dspy")
    def test_auto_detect_priority_high_for_error(self, mock_dspy):
        """_auto_detect_priority returns HIGH for 'error' category."""
        manager = SmartContextManager()
        priority = manager._auto_detect_priority("error", "something failed")
        assert priority == ContextPriority.HIGH

    @patch("Jotty.core.context.context_manager.dspy")
    def test_auto_detect_priority_low_for_unknown_category(self, mock_dspy):
        """_auto_detect_priority returns LOW for unrecognized categories."""
        manager = SmartContextManager()
        priority = manager._auto_detect_priority("random_stuff", "nothing special")
        assert priority == ContextPriority.LOW


@pytest.mark.unit
@pytest.mark.skipif(not HAS_CHUNKER, reason="ContextChunker requires DSPy")
class TestContextChunker:
    """Tests for ContextChunker (requires DSPy)."""

    @patch("Jotty.core.context.chunker.dspy")
    def test_chunker_initializes_with_lm(self, mock_dspy):
        """ContextChunker stores the provided lm."""
        mock_lm = Mock()
        chunker = ContextChunker(lm=mock_lm)
        assert chunker.lm is mock_lm
