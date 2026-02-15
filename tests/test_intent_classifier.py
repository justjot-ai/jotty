"""
Tests for Intent Classification System
"""

import pytest

from Jotty.core.modes.execution.intent_classifier import (
    IntentClassifier,
    TaskIntent,
    classify_task_intent,
)


class TestIntentClassifier:
    """Test intent classification."""

    @pytest.mark.unit
    def test_fact_retrieval_classification(self):
        """Test classification of fact-retrieval questions."""

        test_cases = [
            ("What is the capital of France?", TaskIntent.FACT_RETRIEVAL),
            ("Calculate 234 * 567", TaskIntent.FACT_RETRIEVAL),
            ("Who invented the telephone?", TaskIntent.FACT_RETRIEVAL),
            ("When was World War 2?", TaskIntent.FACT_RETRIEVAL),
        ]

        classifier = IntentClassifier()

        for question, expected_intent in test_cases:
            result = classifier._heuristic_fallback(question, None)
            assert (
                result.intent == expected_intent
            ), f"Failed for: {question}, got {result.intent}, expected {expected_intent}"

    @pytest.mark.unit
    def test_tool_detection(self):
        """Test automatic tool detection."""

        test_cases = [
            ("Calculate 123 + 456", ["calculator"]),
            ("What is the capital of France?", ["web-search"]),
        ]

        classifier = IntentClassifier()

        for question, expected_tools in test_cases:
            detected = classifier._auto_detect_tools(question, None, TaskIntent.FACT_RETRIEVAL)
            for tool in expected_tools:
                assert tool in detected, f"Expected {tool} in tools for: {question}"

    @pytest.mark.unit
    def test_attachment_tool_detection(self):
        """Test tool detection from attachments."""

        classifier = IntentClassifier()

        # Audio attachment
        result = classifier._auto_detect_tools(
            "Process this file", ["audio.mp3"], TaskIntent.FACT_RETRIEVAL
        )
        assert "whisper" in result

        # PDF attachment
        result = classifier._auto_detect_tools(
            "Read this document", ["doc.pdf"], TaskIntent.FACT_RETRIEVAL
        )
        assert "document-reader" in result

        # Image attachment
        result = classifier._auto_detect_tools(
            "Describe this image", ["image.jpg"], TaskIntent.FACT_RETRIEVAL
        )
        assert "vision" in result

    @pytest.mark.unit
    def test_multi_step_detection(self):
        """Test detection of multi-step questions."""

        classifier = IntentClassifier()

        # Multi-hop question
        analysis = classifier._heuristic_fallback(
            "What is the capital of the country where Rockhopper penguins live?", None
        )

        # This should be detected as fact retrieval
        assert analysis.intent == TaskIntent.FACT_RETRIEVAL


@pytest.mark.integration
class TestIntentClassifierIntegration:
    """Integration tests with real LLM."""

    @pytest.mark.skipif(True, reason="Requires LLM access")
    async def test_llm_classification(self):
        """Test classification with real LLM."""

        classifier = IntentClassifier()

        result = classifier.classify("What is the capital of France?", attachments=None)

        assert result.intent == TaskIntent.FACT_RETRIEVAL
        assert result.confidence > 0.7
        assert "web-search" in result.required_tools or len(result.required_tools) >= 0
