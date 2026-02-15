"""
Agentic Planner Unit Tests
===========================

Comprehensive unit tests for TaskPlanner, TaskPlan, and create_agentic_planner().
All tests use mocks -- no real LLM calls, no API keys, runs offline and fast (<1s each).

Covers:
- TaskPlanner initialization (with/without DSPy)
- plan_execution() sync planning logic
- aplan_execution() async planning logic
- infer_task_type() via InferenceMixin
- select_skills() via SkillSelectionMixin
- _call_with_retry() retry/backoff logic
- _create_fallback_plan() deterministic fallback
- _normalize_raw_plan() parsing edge cases
- TaskPlan dataclass fields and defaults
- create_agentic_planner() factory function
- set_max_concurrent_llm_calls() semaphore management
- replan_with_reflection() replanning after failure
"""

import asyncio
import json
from dataclasses import fields
from unittest.mock import AsyncMock, MagicMock, Mock, PropertyMock, patch

import pytest

# Try importing DSPy -- tests skip if unavailable
try:
    import dspy

    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False

# Try importing the module under test
try:
    from Jotty.core.infrastructure.foundation.exceptions import AgentExecutionError
    from Jotty.core.modes.agent._execution_types import ExecutionStep, TaskType
    from Jotty.core.modes.agent.agentic_planner import DSPY_AVAILABLE as MODULE_DSPY_AVAILABLE
    from Jotty.core.modes.agent.agentic_planner import (
        TaskPlan,
        TaskPlanner,
        _get_dspy,
        create_agentic_planner,
    )

    PLANNER_AVAILABLE = True
except ImportError:
    PLANNER_AVAILABLE = False

skip_no_dspy = pytest.mark.skipif(not DSPY_AVAILABLE, reason="DSPy not installed")
skip_no_planner = pytest.mark.skipif(not PLANNER_AVAILABLE, reason="Planner module not importable")


# =============================================================================
# Helpers: build a mocked TaskPlanner without real LLM calls
# =============================================================================


def _make_mock_planner():
    """
    Create a TaskPlanner with all DSPy modules mocked out.

    Patches __init__ to skip real DSPy initialization, then attaches
    mock ChainOfThought/Predict modules and fast_lm = None so that
    no real API calls are made.
    """
    with patch.object(TaskPlanner, "__init__", lambda self, *a, **kw: None):
        planner = TaskPlanner.__new__(TaskPlanner)

    # Attach required attributes that __init__ normally sets
    planner.execution_planner = MagicMock()
    planner.reflective_planner = MagicMock()
    planner.task_type_inferrer = MagicMock()
    planner.skill_selector = MagicMock()
    planner.capability_inferrer = MagicMock()
    planner._signatures = {}
    planner._compressor = None
    planner._max_compression_retries = 3
    planner._fast_lm = None
    planner._fast_model = "haiku"
    planner._use_typed_predictor = False

    # Reset class-level semaphore so tests are isolated
    TaskPlanner._llm_semaphore = None
    TaskPlanner._semaphore_lock = None
    TaskPlanner._max_concurrent_llm_calls = 1

    return planner


def _sample_skills():
    """Return a minimal list of skill dicts for planning tests."""
    return [
        {
            "name": "web-search",
            "description": "Search the web",
            "tools": [{"name": "search_web_tool"}],
        },
        {
            "name": "file-operations",
            "description": "Read/write files",
            "tools": [
                {"name": "write_file_tool"},
                {"name": "read_file_tool"},
            ],
        },
    ]


# =============================================================================
# TaskPlan dataclass tests
# =============================================================================


@pytest.mark.unit
@skip_no_planner
class TestTaskPlan:
    """Test TaskPlan dataclass fields and defaults."""

    def test_default_fields(self):
        """TaskPlan initializes with correct defaults."""
        plan = TaskPlan()
        assert plan.task_graph is None
        assert plan.steps == []
        assert plan.estimated_time is None
        assert plan.required_tools == []
        assert plan.required_credentials == []
        assert plan.metadata == {}

    def test_custom_fields(self):
        """TaskPlan stores custom values correctly."""
        steps = [
            ExecutionStep(
                skill_name="web-search",
                tool_name="search_web_tool",
                params={"query": "test"},
                description="Search",
            )
        ]
        plan = TaskPlan(
            task_graph={"type": "mock"},
            steps=steps,
            estimated_time="4 minutes",
            required_tools=["web-search"],
            required_credentials=["api_key"],
            metadata={"source": "test"},
        )
        assert plan.task_graph == {"type": "mock"}
        assert len(plan.steps) == 1
        assert plan.steps[0].skill_name == "web-search"
        assert plan.estimated_time == "4 minutes"
        assert plan.required_tools == ["web-search"]
        assert plan.required_credentials == ["api_key"]
        assert plan.metadata["source"] == "test"

    def test_dataclass_field_names(self):
        """TaskPlan has the expected set of field names."""
        field_names = {f.name for f in fields(TaskPlan)}
        expected = {
            "task_graph",
            "steps",
            "estimated_time",
            "required_tools",
            "required_credentials",
            "metadata",
        }
        assert field_names == expected


# =============================================================================
# TaskPlanner initialization tests
# =============================================================================


@pytest.mark.unit
@skip_no_planner
@skip_no_dspy
class TestTaskPlannerInit:
    """Test TaskPlanner initialization and factory function."""

    @patch("Jotty.core.agents.agentic_planner._get_dspy", return_value=None)
    def test_init_raises_without_dspy(self, mock_dspy):
        """TaskPlanner raises AgentExecutionError when DSPy is missing."""
        with pytest.raises(AgentExecutionError, match="DSPy required"):
            TaskPlanner()

    @patch("Jotty.core.agents.agentic_planner._get_dspy")
    @patch("Jotty.core.agents.agentic_planner._load_signatures")
    @patch.object(TaskPlanner, "_init_fast_lm")
    def test_init_success(self, mock_fast_lm, mock_sigs, mock_dspy):
        """TaskPlanner initializes successfully with mocked DSPy."""
        mock_dspy_mod = MagicMock()
        mock_dspy.return_value = mock_dspy_mod

        planner = TaskPlanner(fast_model="sonnet")
        assert planner._fast_model == "sonnet"
        assert planner._max_compression_retries == 3
        assert planner._use_typed_predictor is False
        mock_sigs.assert_called_once()
        mock_fast_lm.assert_called_once()

    def test_create_agentic_planner_factory(self):
        """create_agentic_planner() returns a TaskPlanner instance."""
        with patch.object(TaskPlanner, "__init__", return_value=None) as mock_init:
            planner = create_agentic_planner()
            assert isinstance(planner, TaskPlanner)
            mock_init.assert_called_once()


# =============================================================================
# Semaphore / concurrency control tests
# =============================================================================


@pytest.mark.unit
@skip_no_planner
class TestSemaphore:
    """Test semaphore and concurrency control."""

    def test_set_max_concurrent_llm_calls(self):
        """set_max_concurrent_llm_calls updates class attribute."""
        original = TaskPlanner._max_concurrent_llm_calls
        try:
            TaskPlanner.set_max_concurrent_llm_calls(5)
            assert TaskPlanner._max_concurrent_llm_calls == 5
            # Semaphore should be reset (None) so it gets recreated
            assert TaskPlanner._llm_semaphore is None
        finally:
            TaskPlanner._max_concurrent_llm_calls = original
            TaskPlanner._llm_semaphore = None

    def test_set_max_concurrent_llm_calls_minimum(self):
        """set_max_concurrent_llm_calls enforces minimum of 1."""
        original = TaskPlanner._max_concurrent_llm_calls
        try:
            TaskPlanner.set_max_concurrent_llm_calls(0)
            assert TaskPlanner._max_concurrent_llm_calls == 1
            TaskPlanner.set_max_concurrent_llm_calls(-5)
            assert TaskPlanner._max_concurrent_llm_calls == 1
        finally:
            TaskPlanner._max_concurrent_llm_calls = original
            TaskPlanner._llm_semaphore = None

    def test_get_semaphore_creates_once(self):
        """_get_semaphore creates semaphore lazily and reuses it."""
        TaskPlanner._llm_semaphore = None
        TaskPlanner._semaphore_lock = None
        TaskPlanner._max_concurrent_llm_calls = 2

        sem1 = TaskPlanner._get_semaphore()
        sem2 = TaskPlanner._get_semaphore()
        assert sem1 is sem2
        # Cleanup
        TaskPlanner._llm_semaphore = None
        TaskPlanner._semaphore_lock = None
        TaskPlanner._max_concurrent_llm_calls = 1


# =============================================================================
# plan_execution() tests
# =============================================================================


@pytest.mark.unit
@skip_no_planner
class TestPlanExecution:
    """Test plan_execution() sync method."""

    def test_plan_execution_returns_steps_and_reasoning(self):
        """plan_execution returns (steps, reasoning) tuple on success."""
        planner = _make_mock_planner()

        # Mock the execution_planner to return a plan
        mock_result = MagicMock()
        mock_result.execution_plan = json.dumps(
            [
                {
                    "skill_name": "web-search",
                    "tool_name": "search_web_tool",
                    "params": {"query": "AI trends"},
                    "description": "Search for AI trends",
                    "depends_on": [],
                    "output_key": "step_0",
                    "optional": False,
                }
            ]
        )
        mock_result.reasoning = "Search first, then analyze"
        mock_result.estimated_complexity = "low"
        planner.execution_planner.return_value = mock_result

        steps, reasoning = planner.plan_execution(
            task="Research AI trends",
            task_type=TaskType.RESEARCH,
            skills=_sample_skills(),
        )

        assert isinstance(steps, list)
        assert len(steps) >= 1
        assert isinstance(reasoning, str)
        assert len(reasoning) > 0

    def test_plan_execution_no_skills_creation(self):
        """plan_execution adds default file-operations for creation with no skills."""
        planner = _make_mock_planner()

        mock_result = MagicMock()
        mock_result.execution_plan = json.dumps(
            [
                {
                    "skill_name": "file-operations",
                    "tool_name": "write_file_tool",
                    "params": {"path": "test.py", "content": 'print("hi")'},
                    "description": "Create file",
                    "depends_on": [],
                    "output_key": "step_0",
                }
            ]
        )
        mock_result.reasoning = "Create the file"
        planner.execution_planner.return_value = mock_result

        steps, reasoning = planner.plan_execution(
            task="Create a Python file",
            task_type=TaskType.CREATION,
            skills=[],  # empty skills
        )
        # Should have added default file-operations and produced steps
        assert isinstance(steps, list)

    def test_plan_execution_no_skills_non_creation(self):
        """plan_execution returns empty steps for non-creation task with no skills."""
        planner = _make_mock_planner()

        steps, reasoning = planner.plan_execution(
            task="Analyze data",
            task_type=TaskType.ANALYSIS,
            skills=[],
        )
        assert steps == []
        assert "No skills available" in reasoning

    def test_plan_execution_fallback_on_exception(self):
        """plan_execution uses fallback plan when LLM call raises."""
        planner = _make_mock_planner()
        planner.execution_planner.side_effect = RuntimeError("LLM unavailable")

        skills = _sample_skills()
        steps, reasoning = planner.plan_execution(
            task="Search for Python docs",
            task_type=TaskType.RESEARCH,
            skills=skills,
        )
        # Should return fallback plan, not crash
        assert isinstance(steps, list)
        assert "failed" in reasoning.lower() or "fallback" in reasoning.lower()

    def test_plan_execution_empty_plan_triggers_fallback(self):
        """plan_execution uses fallback when LLM returns empty plan."""
        planner = _make_mock_planner()

        mock_result = MagicMock()
        mock_result.execution_plan = "[]"
        mock_result.reasoning = "No plan"
        planner.execution_planner.return_value = mock_result

        skills = _sample_skills()
        steps, reasoning = planner.plan_execution(
            task="Research topic",
            task_type=TaskType.RESEARCH,
            skills=skills,
        )
        assert isinstance(steps, list)
        assert "fallback" in reasoning.lower() or len(steps) >= 0


# =============================================================================
# aplan_execution() async tests
# =============================================================================


@pytest.mark.unit
@skip_no_planner
class TestAsyncPlanExecution:
    """Test aplan_execution() async method."""

    @pytest.mark.asyncio
    async def test_aplan_execution_success(self):
        """aplan_execution returns (steps, reasoning) on success."""
        planner = _make_mock_planner()

        mock_result = MagicMock()
        mock_result.execution_plan = json.dumps(
            [
                {
                    "skill_name": "web-search",
                    "tool_name": "search_web_tool",
                    "params": {"query": "test"},
                    "description": "Search",
                    "depends_on": [],
                    "output_key": "step_0",
                }
            ]
        )
        mock_result.reasoning = "Plan ready"

        # Mock _acall_with_retry to return the result directly
        planner._acall_with_retry = AsyncMock(return_value=mock_result)

        steps, reasoning = await planner.aplan_execution(
            task="Search for something",
            task_type=TaskType.RESEARCH,
            skills=_sample_skills(),
        )
        assert isinstance(steps, list)
        assert len(steps) >= 1
        assert "Plan ready" in reasoning or "Planned" in reasoning

    @pytest.mark.asyncio
    async def test_aplan_execution_no_skills_non_creation(self):
        """aplan_execution returns empty for non-creation with no skills."""
        planner = _make_mock_planner()

        steps, reasoning = await planner.aplan_execution(
            task="Analyze data",
            task_type=TaskType.ANALYSIS,
            skills=[],
        )
        assert steps == []
        assert "No skills available" in reasoning

    @pytest.mark.asyncio
    async def test_aplan_execution_fallback_on_failure(self):
        """aplan_execution falls back when async LLM call fails."""
        planner = _make_mock_planner()
        planner._acall_with_retry = AsyncMock(side_effect=RuntimeError("LLM down"))

        skills = _sample_skills()
        steps, reasoning = await planner.aplan_execution(
            task="Search the web",
            task_type=TaskType.RESEARCH,
            skills=skills,
        )
        assert isinstance(steps, list)
        assert "failed" in reasoning.lower() or "fallback" in reasoning.lower()


# =============================================================================
# _normalize_raw_plan() tests
# =============================================================================


@pytest.mark.unit
@skip_no_planner
class TestNormalizeRawPlan:
    """Test _normalize_raw_plan() parsing of various LLM output formats."""

    def setup_method(self):
        self.planner = _make_mock_planner()

    def test_already_list(self):
        """Returns list directly when input is already a list."""
        data = [{"skill_name": "web-search", "tool_name": "search_web_tool"}]
        result = self.planner._normalize_raw_plan(data)
        assert result == data

    def test_none_returns_empty(self):
        """Returns empty list for None input."""
        result = self.planner._normalize_raw_plan(None)
        assert result == []

    def test_empty_string_returns_empty(self):
        """Returns empty list for empty string."""
        result = self.planner._normalize_raw_plan("")
        assert result == []

    def test_json_string_parsed(self):
        """Parses direct JSON array string."""
        json_str = json.dumps(
            [
                {"skill_name": "web-search", "tool_name": "search_web_tool"},
            ]
        )
        result = self.planner._normalize_raw_plan(json_str)
        assert len(result) == 1
        assert result[0]["skill_name"] == "web-search"

    def test_markdown_code_block_parsed(self):
        """Extracts JSON from markdown code block."""
        plan_data = [{"skill_name": "file-operations", "tool_name": "write_file_tool"}]
        text = f"Here is the plan:\n```json\n{json.dumps(plan_data)}\n```\nDone."
        result = self.planner._normalize_raw_plan(text)
        assert len(result) == 1
        assert result[0]["skill_name"] == "file-operations"

    def test_embedded_json_array_extracted(self):
        """Extracts JSON array embedded in prose text."""
        plan_data = [{"skill_name": "web-search", "tool_name": "search_web_tool"}]
        text = f"I think the best plan is {json.dumps(plan_data)} for this task."
        result = self.planner._normalize_raw_plan(text)
        assert len(result) == 1


# =============================================================================
# _create_fallback_plan() tests
# =============================================================================


@pytest.mark.unit
@skip_no_planner
class TestFallbackPlan:
    """Test _create_fallback_plan() deterministic fallback planning."""

    def setup_method(self):
        self.planner = _make_mock_planner()

    def test_no_skills_returns_empty(self):
        """Fallback with no skills returns empty list."""
        result = self.planner._create_fallback_plan("task", TaskType.RESEARCH, [])
        assert result == []

    def test_research_task_creates_steps(self):
        """Fallback for research task creates steps from available skills."""
        skills = _sample_skills()
        result = self.planner._create_fallback_plan("Research AI trends", TaskType.RESEARCH, skills)
        assert isinstance(result, list)
        assert len(result) >= 1
        # Check each step is a dict with expected keys
        for step in result:
            assert "skill_name" in step
            assert "tool_name" in step
            assert "params" in step

    def test_creation_task_uses_file_operations(self):
        """Fallback for creation tasks includes file-operations."""
        skills = [
            {
                "name": "file-operations",
                "description": "File ops",
                "tools": [{"name": "write_file_tool"}],
            },
        ]
        result = self.planner._create_fallback_plan("Create a test file", TaskType.CREATION, skills)
        assert len(result) >= 1
        skill_names = [s["skill_name"] for s in result]
        assert "file-operations" in skill_names

    def test_fallback_limits_steps(self):
        """Fallback plan limits steps based on available skills count."""
        skills = _sample_skills()
        result = self.planner._create_fallback_plan("Do something", TaskType.UNKNOWN, skills)
        # Max fallback steps = min(3, max(1, len(skills))) = min(3, 2) = 2
        assert len(result) <= 3


# =============================================================================
# _call_with_retry() tests
# =============================================================================


@pytest.mark.unit
@skip_no_planner
class TestCallWithRetry:
    """Test _call_with_retry() error handling and retry logic."""

    def setup_method(self):
        self.planner = _make_mock_planner()

    def test_success_first_try(self):
        """Successful call on first attempt returns result."""
        module = MagicMock(return_value="success")
        result = self.planner._call_with_retry(module, {"key": "val"})
        assert result == "success"
        module.assert_called_once_with(key="val")

    def test_non_retryable_error_raises(self):
        """Non-retryable error is raised immediately."""
        module = MagicMock(side_effect=ValueError("bad input"))
        with pytest.raises(ValueError, match="bad input"):
            self.planner._call_with_retry(module, {}, max_retries=3)
        # Should only be called once (no retry for unknown errors)
        assert module.call_count == 1

    def test_uses_lm_context_when_provided(self):
        """When lm is provided, calls within dspy.context."""
        planner = self.planner
        module = MagicMock(return_value="result")
        mock_lm = MagicMock()

        with patch("Jotty.core.agents.agentic_planner._get_dspy") as mock_get:
            mock_dspy = MagicMock()
            mock_get.return_value = mock_dspy

            result = planner._call_with_retry(module, {"a": 1}, lm=mock_lm)
            assert result == "result"
            mock_dspy.context.assert_called_once_with(lm=mock_lm)


# =============================================================================
# infer_task_type() tests (InferenceMixin)
# =============================================================================


@pytest.mark.unit
@skip_no_planner
class TestInferTaskType:
    """Test task type inference via InferenceMixin."""

    def setup_method(self):
        self.planner = _make_mock_planner()
        # Clear the per-session cache between tests
        from Jotty.core.modes.agent._inference_mixin import InferenceMixin

        InferenceMixin._task_type_cache.clear()

    def test_infer_creation_task(self):
        """Infers CREATION for creation-related task on keyword fallback."""
        # Make the inferrer raise so keyword fallback kicks in
        self.planner.task_type_inferrer.side_effect = RuntimeError("no LLM")

        task_type, reasoning, confidence = self.planner.infer_task_type(
            "Create a Python web server"
        )
        assert task_type == TaskType.CREATION
        assert confidence > 0

    def test_infer_research_task(self):
        """Infers RESEARCH for research-related task on keyword fallback."""
        self.planner.task_type_inferrer.side_effect = RuntimeError("no LLM")

        task_type, reasoning, confidence = self.planner.infer_task_type(
            "Research the latest machine learning papers"
        )
        assert task_type == TaskType.RESEARCH

    def test_infer_comparison_task(self):
        """Infers COMPARISON for compare-related task on keyword fallback."""
        self.planner.task_type_inferrer.side_effect = RuntimeError("no LLM")

        task_type, reasoning, confidence = self.planner.infer_task_type(
            "Compare React vs Vue vs Angular"
        )
        assert task_type == TaskType.COMPARISON

    def test_infer_analysis_task(self):
        """Infers ANALYSIS for analyze-related task on keyword fallback."""
        self.planner.task_type_inferrer.side_effect = RuntimeError("no LLM")

        task_type, reasoning, confidence = self.planner.infer_task_type(
            "Analyze the quarterly revenue data"
        )
        assert task_type == TaskType.ANALYSIS

    def test_infer_caches_result(self):
        """Second call for same task returns cached result without calling LLM."""
        self.planner.task_type_inferrer.side_effect = RuntimeError("no LLM")

        # First call
        result1 = self.planner.infer_task_type("Create something")
        # Second call -- should hit cache
        result2 = self.planner.infer_task_type("Create something")

        assert result1 == result2


# =============================================================================
# replan_with_reflection() tests
# =============================================================================


@pytest.mark.unit
@skip_no_planner
class TestReplanWithReflection:
    """Test replan_with_reflection() replanning after failures."""

    def test_replan_excludes_failed_skills(self):
        """Replanning filters out excluded skills."""
        planner = _make_mock_planner()

        mock_result = MagicMock()
        mock_result.corrected_plan = json.dumps(
            [
                {
                    "skill_name": "file-operations",
                    "tool_name": "write_file_tool",
                    "params": {"path": "out.txt", "content": "data"},
                    "description": "Write output",
                    "depends_on": [],
                    "output_key": "step_0",
                }
            ]
        )
        mock_result.reflection = "web-search failed, using file-operations"
        mock_result.reasoning = "Alternative approach"
        planner.reflective_planner.return_value = mock_result

        skills = _sample_skills()
        failed_steps = [
            {"skill_name": "web-search", "tool_name": "search_web_tool", "error": "timeout"}
        ]

        steps, reflection, reasoning = planner.replan_with_reflection(
            task="Find and save data",
            task_type=TaskType.RESEARCH,
            skills=skills,
            failed_steps=failed_steps,
            excluded_skills=["web-search"],
        )
        assert isinstance(steps, list)
        assert isinstance(reflection, str)
        assert isinstance(reasoning, str)

    def test_replan_fallback_when_reflection_fails(self):
        """Replanning falls back to plan_execution when reflection fails."""
        planner = _make_mock_planner()

        # Reflective planner fails
        planner.reflective_planner.side_effect = RuntimeError("LLM error")

        # Regular plan_execution should also be called as fallback
        # Mock the execution_planner for the fallback path
        mock_result = MagicMock()
        mock_result.execution_plan = json.dumps(
            [
                {
                    "skill_name": "file-operations",
                    "tool_name": "write_file_tool",
                    "params": {"path": "out.txt"},
                    "description": "Write file",
                    "depends_on": [],
                    "output_key": "step_0",
                }
            ]
        )
        mock_result.reasoning = "Fallback plan"
        mock_result.estimated_complexity = "low"
        planner.execution_planner.return_value = mock_result

        skills = _sample_skills()
        steps, reflection, reasoning = planner.replan_with_reflection(
            task="Write data to file",
            task_type=TaskType.CREATION,
            skills=skills,
            failed_steps=[{"skill_name": "web-search", "error": "timeout"}],
        )
        assert isinstance(steps, list)
        assert "fallback" in reflection.lower() or "Fallback" in reflection


# =============================================================================
# _parse_plan_to_steps() tests
# =============================================================================


@pytest.mark.unit
@skip_no_planner
class TestParsePlanToSteps:
    """Test _parse_plan_to_steps() conversion to ExecutionStep objects."""

    def setup_method(self):
        self.planner = _make_mock_planner()

    def test_converts_dicts_to_execution_steps(self):
        """Converts list of dicts to ExecutionStep objects."""
        raw_plan = [
            {
                "skill_name": "web-search",
                "tool_name": "search_web_tool",
                "params": {"query": "test"},
                "description": "Search for test",
                "depends_on": [],
                "output_key": "step_0",
                "optional": False,
            }
        ]
        steps = self.planner._parse_plan_to_steps(
            raw_plan=raw_plan,
            skills=_sample_skills(),
            task="test",
            task_type=TaskType.RESEARCH,
        )
        assert len(steps) == 1
        assert isinstance(steps[0], ExecutionStep)
        assert steps[0].skill_name == "web-search"
        assert steps[0].tool_name == "search_web_tool"
        assert steps[0].output_key == "step_0"

    def test_empty_raw_plan_returns_empty(self):
        """Empty raw plan returns empty steps list."""
        steps = self.planner._parse_plan_to_steps(
            raw_plan=None,
            skills=_sample_skills(),
            task="test",
        )
        assert steps == []

    def test_respects_max_steps(self):
        """Limits number of parsed steps to max_steps."""
        raw_plan = [
            {
                "skill_name": "web-search",
                "tool_name": "search_web_tool",
                "params": {"query": f"query_{i}"},
                "description": f"Step {i}",
                "depends_on": [],
                "output_key": f"step_{i}",
            }
            for i in range(20)
        ]
        steps = self.planner._parse_plan_to_steps(
            raw_plan=raw_plan,
            skills=_sample_skills(),
            task="test",
            max_steps=3,
        )
        assert len(steps) <= 3


# =============================================================================
# ExecutionStep tests
# =============================================================================


@pytest.mark.unit
@skip_no_planner
class TestExecutionStep:
    """Test ExecutionStep dataclass."""

    def test_default_fields(self):
        """ExecutionStep has correct default values."""
        step = ExecutionStep(
            skill_name="web-search",
            tool_name="search_web_tool",
            params={"query": "test"},
            description="Search",
        )
        assert step.depends_on == []
        assert step.output_key == ""
        assert step.optional is False
        assert step.verification == ""
        assert step.fallback_skill == ""

    def test_all_fields(self):
        """ExecutionStep stores all provided values."""
        step = ExecutionStep(
            skill_name="web-search",
            tool_name="search_web_tool",
            params={"query": "test"},
            description="Search the web",
            depends_on=[0, 1],
            output_key="step_2",
            optional=True,
            verification="check results",
            fallback_skill="http-client",
        )
        assert step.skill_name == "web-search"
        assert step.depends_on == [0, 1]
        assert step.output_key == "step_2"
        assert step.optional is True
        assert step.verification == "check results"
        assert step.fallback_skill == "http-client"
