"""
Tests for Base Skill Architecture
==================================

Demonstrates how easy testing becomes with unified base classes.
"""

from typing import Any, Dict

import pytest

from .base_skill import BaseLLMSkill, BaseToolSkill
from .decorators import skill_tool, validate_params
from .helpers import create_llm_skill, create_tool_skill

# =============================================================================
# Mock LLM for Testing
# =============================================================================


class MockLLM:
    """Mock LLM that returns predictable responses."""

    def __init__(self, response: str = "mocked response"):
        self.response = response
        self.calls = []

    def __call__(self, prompt: str = None, messages: list = None, **kwargs):
        """Mock LLM call."""
        self.calls.append({"prompt": prompt, "messages": messages, **kwargs})

        # Return DSPy-style response
        class MockCompletion:
            def __init__(self, text):
                self.text = text

        class MockResponse:
            def __init__(self, text):
                self.completions = [MockCompletion(text)]

        return MockResponse(self.response)


# =============================================================================
# Test BaseToolSkill
# =============================================================================


class CalculatorSkill(BaseToolSkill):
    """Simple calculator for testing."""

    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        expression = params["expression"]
        result = eval(expression)  # Don't use eval in production!
        return self.success(result=result)


def test_base_tool_skill():
    """Test BaseToolSkill with calculator example."""
    calc = CalculatorSkill("calculator")

    # Test success case
    result = calc({"expression": "2 + 2"})
    assert result["success"] is True
    assert result["result"] == 4

    # Test error handling (missing param)
    result = calc({})
    assert result["success"] is False
    assert "error" in result
    assert "expression" in result["error"].lower()


# =============================================================================
# Test BaseLLMSkill
# =============================================================================


class SummarizerSkill(BaseLLMSkill):
    """Summarizer for testing."""

    def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        text = params["text"]
        summary = self.call_lm(f"Summarize: {text}")
        return self.success(summary=summary)


def test_base_llm_skill():
    """Test BaseLLMSkill with mocked LLM."""
    summarizer = SummarizerSkill("summarizer")

    # Mock the LLM
    mock_lm = MockLLM("This is a summary")
    summarizer._lm = mock_lm

    # Test LLM call
    result = summarizer({"text": "Long text here"})

    assert result["success"] is True
    assert result["summary"] == "This is a summary"

    # Verify LLM was called
    assert len(mock_lm.calls) == 1
    assert "Summarize: Long text" in mock_lm.calls[0]["prompt"]


def test_llm_skill_generate_text():
    """Test generate_text method."""
    skill = SummarizerSkill("test")
    skill._lm = MockLLM("Generated text")

    text = skill.generate_text(prompt="User request", system_prompt="You are helpful")

    assert text == "Generated text"


def test_llm_skill_generate_structured():
    """Test generate_structured method."""
    skill = SummarizerSkill("test")
    skill._lm = MockLLM('{"name": "John", "age": 30}')

    result = skill.generate_structured(
        prompt="Generate person data", schema={"name": "string", "age": "number"}
    )

    assert result["name"] == "John"
    assert result["age"] == 30


# =============================================================================
# Test Decorators
# =============================================================================


def test_validate_params():
    """Test @validate_params decorator."""

    class TestSkill(BaseToolSkill):
        @validate_params(required=["a", "b"], optional=["c"])
        def execute(self, params):
            return self.success(result=params["a"] + params["b"])

    skill = TestSkill("test")

    # Valid params
    result = skill({"a": 1, "b": 2})
    assert result["success"] is True
    assert result["result"] == 3

    # Missing required param
    result = skill({"a": 1})
    assert result["success"] is False
    assert "missing" in result["error"].lower()

    # Extra unknown param
    result = skill({"a": 1, "b": 2, "d": 3})
    assert result["success"] is False
    assert "unknown" in result["error"].lower()


def test_skill_tool_decorator():
    """Test @skill_tool decorator."""

    @skill_tool(name="adder", required_params=["a", "b"], use_llm=False)
    def add(skill, params):
        return {"sum": params["a"] + params["b"]}

    result = add({"a": 5, "b": 3})

    assert result["success"] is True
    assert result["sum"] == 8


def test_skill_tool_decorator_with_llm():
    """Test @skill_tool decorator with LLM."""

    @skill_tool(name="summarizer", required_params=["text"], use_llm=True)
    def summarize(skill, params):
        summary = skill.call_lm(f"Summarize: {params['text']}")
        return {"summary": summary}

    # Mock the LLM
    summarize.skill._lm = MockLLM("Summary here")

    result = summarize({"text": "Long text"})

    assert result["success"] is True
    assert result["summary"] == "Summary here"


# =============================================================================
# Test Helper Functions
# =============================================================================


def test_create_tool_skill():
    """Test create_tool_skill helper."""

    calc = create_tool_skill(
        name="calc", execute_fn=lambda params: {"result": params["a"] + params["b"]}
    )

    result = calc({"a": 10, "b": 5})

    assert result["success"] is True
    assert result["result"] == 15


def test_create_llm_skill():
    """Test create_llm_skill helper."""

    summarizer = create_llm_skill(
        name="summarizer",
        execute_fn=lambda skill, params: {"summary": skill.call_lm(f"Summarize: {params['text']}")},
    )

    # Mock LLM
    summarizer._lm = MockLLM("Summary")

    result = summarizer({"text": "Test"})

    assert result["success"] is True
    assert result["summary"] == "Summary"


# =============================================================================
# Test Error Handling
# =============================================================================


def test_error_handling():
    """Test automatic error handling."""

    class FailingSkill(BaseToolSkill):
        def execute(self, params):
            raise ValueError("Test error")

    skill = FailingSkill("failing")
    result = skill({"input": "test"})

    assert result["success"] is False
    assert "ValueError" in result["error"]
    assert "Test error" in result["error"]


def test_key_error_handling():
    """Test KeyError handling (missing params)."""

    class SkillWithMissingParam(BaseToolSkill):
        def execute(self, params):
            value = params["required_param"]  # Will raise KeyError
            return self.success(value=value)

    skill = SkillWithMissingParam("test")
    result = skill({"wrong_param": "value"})

    assert result["success"] is False
    assert "required_param" in result["error"]


# =============================================================================
# Test Status Callbacks
# =============================================================================


def test_status_callback():
    """Test status callback functionality."""

    status_messages = []

    def status_cb(message):
        status_messages.append(message)

    class StatusSkill(BaseToolSkill):
        def execute(self, params):
            self.update_status("Step 1")
            self.update_status("Step 2")
            return self.success(result="done")

    skill = StatusSkill("test")
    result = skill({"_status_callback": status_cb})

    assert result["success"] is True
    assert "Step 1" in status_messages
    assert "Step 2" in status_messages


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
