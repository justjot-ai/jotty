"""
Tests for Fact-Retrieval Executor
"""

import pytest
from Jotty.core.execution.fact_retrieval_executor import (
    FactRetrievalExecutor,
    AnswerFormat,
    QuestionAnalysis
)


class TestFactRetrievalExecutor:
    """Test fact-retrieval executor."""

    @pytest.mark.unit
    def test_tool_auto_detection(self):
        """Test automatic tool detection from questions."""

        executor = FactRetrievalExecutor()

        # Math question
        tools = executor._auto_detect_tools("Calculate 234 * 567", None)
        assert 'calculator' in tools

        # Search question
        tools = executor._auto_detect_tools("What is the capital of France?", None)
        assert 'web-search' in tools

        # Audio attachment
        tools = executor._auto_detect_tools("Transcribe this", ["audio.mp3"])
        assert 'whisper' in tools or 'openai-whisper-api' in tools

    @pytest.mark.unit
    def test_dependency_extraction(self):
        """Test extraction of step dependencies."""

        executor = FactRetrievalExecutor()

        # No dependencies
        deps = executor._extract_dependencies("What is the capital of France?")
        assert len(deps) == 0

        # Single dependency
        deps = executor._extract_dependencies(
            "What is the capital of {Step 1}?"
        )
        assert deps == [0]  # Step 1 → index 0

        # Multiple dependencies
        deps = executor._extract_dependencies(
            "Compare {Step 1} and {Step 2}"
        )
        assert set(deps) == {0, 1}

    @pytest.mark.unit
    def test_answer_format_validation(self):
        """Test answer format validation and fixing."""

        executor = FactRetrievalExecutor()

        # Number format
        result = executor._validate_and_fix_format("The answer is 42", AnswerFormat.NUMBER)
        assert result == "42"

        # Yes/No format
        result = executor._validate_and_fix_format("Yes, that's correct", AnswerFormat.YES_NO)
        assert result == "Yes"

        result = executor._validate_and_fix_format("No way", AnswerFormat.YES_NO)
        assert result == "No"

        # Location format (remove prefix)
        result = executor._validate_and_fix_format("in Paris", AnswerFormat.LOCATION)
        assert result == "Paris"


@pytest.mark.integration
class TestFactRetrievalExecutorIntegration:
    """Integration tests with real execution."""

    @pytest.mark.skipif(True, reason="Requires full setup")
    async def test_simple_question(self):
        """Test simple fact-retrieval question."""

        executor = FactRetrievalExecutor()

        result = await executor.execute("What is 2 + 2?")

        assert "4" in result

    @pytest.mark.skipif(True, reason="Requires full setup")
    async def test_multi_hop_question(self):
        """Test multi-hop reasoning question."""

        executor = FactRetrievalExecutor()

        result = await executor.execute(
            "What is the capital of the country where the Eiffel Tower is located?"
        )

        assert "Paris" in result
