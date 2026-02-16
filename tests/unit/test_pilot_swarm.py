"""Tests for the Pilot Swarm.

Tests cover:
- Types, enums, config validation
- Each agent with mocked DSPy modules
- Subtask dependency handling
- Content assembly and context building
- Swarm execution with fully mocked agents
- Convenience functions (pilot, pilot_sync)
- Registration check
- Skill writing output validation
- Terminal safety checking
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import types
from Jotty.core.execution.swarms.pilot_swarm.types import (
    AVAILABLE_SWARMS,
    PilotConfig,
    PilotResult,
    Subtask,
    SubtaskStatus,
    SubtaskType,
)

# =============================================================================
# TYPE AND ENUM TESTS
# =============================================================================


class TestEnums:
    """Test all enums."""

    @pytest.mark.unit
    def test_subtask_type_values(self):
        assert SubtaskType.SEARCH.value == "search"
        assert SubtaskType.CODE.value == "code"
        assert SubtaskType.TERMINAL.value == "terminal"
        assert SubtaskType.CREATE_SKILL.value == "create_skill"
        assert SubtaskType.DELEGATE.value == "delegate"
        assert SubtaskType.ANALYZE.value == "analyze"
        assert SubtaskType.BROWSE.value == "browse"
        assert len(SubtaskType) == 7

    @pytest.mark.unit
    def test_subtask_status_values(self):
        assert SubtaskStatus.PENDING.value == "pending"
        assert SubtaskStatus.RUNNING.value == "running"
        assert SubtaskStatus.COMPLETED.value == "completed"
        assert SubtaskStatus.FAILED.value == "failed"
        assert SubtaskStatus.SKIPPED.value == "skipped"
        assert len(SubtaskStatus) == 5


class TestConfig:
    """Test PilotConfig."""

    @pytest.mark.unit
    def test_default_config(self):
        config = PilotConfig()
        assert config.name == "PilotSwarm"
        assert config.domain == "pilot"
        assert config.max_subtasks == 10
        assert config.max_retries == 2
        assert config.max_concurrent == 3
        assert config.allow_terminal is True
        assert config.allow_file_write is True
        assert config.allow_delegation is True
        assert config.send_telegram is False
        assert config.llm_model == "haiku"
        assert config.use_fast_predict is True

    @pytest.mark.unit
    def test_custom_config(self):
        config = PilotConfig(
            max_subtasks=5,
            allow_terminal=False,
            allow_file_write=False,
        )
        assert config.max_subtasks == 5
        assert config.allow_terminal is False
        assert config.allow_file_write is False

    @pytest.mark.unit
    def test_config_sets_llm_timeout(self):
        config = PilotConfig()
        assert config.llm_timeout > 0


class TestDataClasses:
    """Test dataclasses."""

    @pytest.mark.unit
    def test_subtask(self):
        st = Subtask(
            id="s1",
            type=SubtaskType.SEARCH,
            description="Search for Python frameworks",
            tool_hint="web-search",
            depends_on=[],
        )
        assert st.id == "s1"
        assert st.type == SubtaskType.SEARCH
        assert st.status == SubtaskStatus.PENDING
        assert st.result is None

    @pytest.mark.unit
    def test_subtask_with_dependencies(self):
        st = Subtask(
            id="s2",
            type=SubtaskType.CODE,
            description="Write code based on research",
            depends_on=["s1"],
        )
        assert st.depends_on == ["s1"]

    @pytest.mark.unit
    def test_pilot_result(self):
        result = PilotResult(
            success=True,
            swarm_name="PilotSwarm",
            domain="pilot",
            output={"goal": "test"},
            execution_time=5.0,
            goal="test goal",
            subtasks_completed=3,
            subtasks_total=4,
            artifacts=["/tmp/test.py"],
            skills_created=["my-skill"],
            delegated_to=["coding"],
        )
        assert result.success is True
        assert result.subtasks_completed == 3
        assert result.subtasks_total == 4
        assert len(result.artifacts) == 1
        assert len(result.skills_created) == 1
        assert len(result.delegated_to) == 1

    @pytest.mark.unit
    def test_available_swarms(self):
        assert len(AVAILABLE_SWARMS) > 0
        assert any("research" in s for s in AVAILABLE_SWARMS)
        assert any("coding" in s for s in AVAILABLE_SWARMS)


# =============================================================================
# AGENT TESTS (with mocked DSPy)
# =============================================================================


def _mock_dspy_result(**fields):
    """Create a mock DSPy prediction result."""
    mock = MagicMock()
    for k, v in fields.items():
        setattr(mock, k, v)
    return mock


class TestPilotPlannerAgent:
    """Test PilotPlannerAgent with mocked DSPy."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_plan_returns_subtasks(self):
        with patch("Jotty.core.execution.swarms.pilot_swarm.agents.BaseOlympiadAgent._get_lm"):
            from Jotty.core.execution.swarms.pilot_swarm.agents import PilotPlannerAgent

            agent = PilotPlannerAgent.__new__(PilotPlannerAgent)
            agent.model = "sonnet"
            agent.use_fast_predict = True
            agent.llm_timeout = 90
            agent._lm = MagicMock()
            agent.learned_context = ""
            agent._bus = None

            mock_result = _mock_dspy_result(
                subtasks_json=json.dumps(
                    [
                        {
                            "id": "s1",
                            "type": "search",
                            "description": "Research Python frameworks",
                            "tool_hint": "web-search",
                            "depends_on": [],
                        },
                        {
                            "id": "s2",
                            "type": "code",
                            "description": "Create comparison table",
                            "tool_hint": "",
                            "depends_on": ["s1"],
                        },
                    ]
                ),
                reasoning="First research, then synthesize into code.",
            )
            agent._planner = MagicMock(return_value=mock_result)

            result = await agent.plan(goal="Compare Python web frameworks")

            assert "subtasks" in result
            assert "reasoning" in result
            assert len(result["subtasks"]) == 2
            assert result["subtasks"][0]["type"] == "search"
            assert result["subtasks"][1]["depends_on"] == ["s1"]


class TestPilotSearchAgent:
    """Test PilotSearchAgent with mocked DSPy."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_search_returns_findings(self):
        with patch("Jotty.core.execution.swarms.pilot_swarm.agents.BaseOlympiadAgent._get_lm"):
            from Jotty.core.execution.swarms.pilot_swarm.agents import PilotSearchAgent

            agent = PilotSearchAgent.__new__(PilotSearchAgent)
            agent.model = "haiku"
            agent.use_fast_predict = True
            agent.llm_timeout = 90
            agent._lm = MagicMock()
            agent.learned_context = ""
            agent._bus = None

            mock_result = _mock_dspy_result(
                search_queries="Python FastAPI tutorial | Django vs FastAPI 2024",
                synthesis="FastAPI is a modern framework known for speed...",
                key_findings="FastAPI uses async by default | Django has larger ecosystem | Flask is minimal",
            )
            agent._searcher = MagicMock(return_value=mock_result)

            result = await agent.search(task="Find Python web frameworks")

            assert "queries" in result
            assert "synthesis" in result
            assert "key_findings" in result
            assert len(result["queries"]) == 2
            assert len(result["key_findings"]) == 3


class TestPilotCoderAgent:
    """Test PilotCoderAgent with mocked DSPy."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_code_returns_file_operations(self):
        with patch("Jotty.core.execution.swarms.pilot_swarm.agents.BaseOlympiadAgent._get_lm"):
            from Jotty.core.execution.swarms.pilot_swarm.agents import PilotCoderAgent

            agent = PilotCoderAgent.__new__(PilotCoderAgent)
            agent.model = "sonnet"
            agent.use_fast_predict = True
            agent.llm_timeout = 90
            agent._lm = MagicMock()
            agent.learned_context = ""
            agent._bus = None

            mock_result = _mock_dspy_result(
                file_operations_json=json.dumps(
                    [
                        {
                            "file_path": "/tmp/test_app.py",
                            "action": "create",
                            "content": "from fastapi import FastAPI\napp = FastAPI()",
                            "description": "Main FastAPI application",
                        },
                    ]
                ),
                explanation="Created a FastAPI app with a basic setup.",
            )
            agent._coder = MagicMock(return_value=mock_result)

            result = await agent.code(task="Create a FastAPI app")

            assert "file_operations" in result
            assert "explanation" in result
            assert len(result["file_operations"]) == 1
            assert result["file_operations"][0]["file_path"] == "/tmp/test_app.py"


class TestPilotTerminalAgent:
    """Test PilotTerminalAgent with mocked DSPy."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_returns_commands(self):
        with patch("Jotty.core.execution.swarms.pilot_swarm.agents.BaseOlympiadAgent._get_lm"):
            from Jotty.core.execution.swarms.pilot_swarm.agents import PilotTerminalAgent

            agent = PilotTerminalAgent.__new__(PilotTerminalAgent)
            agent.model = "haiku"
            agent.use_fast_predict = True
            agent.llm_timeout = 90
            agent._lm = MagicMock()
            agent.learned_context = ""
            agent._bus = None

            mock_result = _mock_dspy_result(
                commands_json=json.dumps(
                    [
                        {"command": "pip list", "purpose": "List installed packages", "safe": True},
                        {"command": "rm -rf /", "purpose": "Delete everything", "safe": False},
                    ]
                ),
                safety_assessment="pip list is safe. rm -rf / is extremely dangerous.",
            )
            agent._terminal = MagicMock(return_value=mock_result)

            result = await agent.execute(task="Check installed packages")

            assert "commands" in result
            assert "safety_assessment" in result
            assert len(result["commands"]) == 2
            assert result["commands"][0]["safe"] is True
            assert result["commands"][1]["safe"] is False


class TestPilotSkillWriterAgent:
    """Test PilotSkillWriterAgent with mocked DSPy."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_write_skill_returns_files(self):
        with patch("Jotty.core.execution.swarms.pilot_swarm.agents.BaseOlympiadAgent._get_lm"):
            from Jotty.core.execution.swarms.pilot_swarm.agents import PilotSkillWriterAgent

            agent = PilotSkillWriterAgent.__new__(PilotSkillWriterAgent)
            agent.model = "sonnet"
            agent.use_fast_predict = True
            agent.llm_timeout = 90
            agent._lm = MagicMock()
            agent.learned_context = ""
            agent._bus = None

            mock_result = _mock_dspy_result(
                skill_yaml="name: csv-to-json\ndescription: Convert CSV to JSON\nversion: 1.0.0\ntools:\n  - convert_csv_tool",
                tools_py='def convert_csv_tool(params):\n    """Convert CSV to JSON."""\n    return {"result": "converted"}',
                usage_example='result = convert_csv_tool({"file_path": "data.csv"})',
            )
            agent._writer = MagicMock(return_value=mock_result)

            result = await agent.write_skill(
                description="Convert CSV files to JSON format",
                skill_name="csv-to-json",
            )

            assert "skill_yaml" in result
            assert "tools_py" in result
            assert "usage_example" in result
            assert "csv-to-json" in result["skill_yaml"]
            assert "def convert_csv_tool" in result["tools_py"]


class TestPilotValidatorAgent:
    """Test PilotValidatorAgent with mocked DSPy."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_validate_success(self):
        with patch("Jotty.core.execution.swarms.pilot_swarm.agents.BaseOlympiadAgent._get_lm"):
            from Jotty.core.execution.swarms.pilot_swarm.agents import PilotValidatorAgent

            agent = PilotValidatorAgent.__new__(PilotValidatorAgent)
            agent.model = "haiku"
            agent.use_fast_predict = True
            agent.llm_timeout = 90
            agent._lm = MagicMock()
            agent.learned_context = ""
            agent._bus = None

            mock_result = _mock_dspy_result(
                success="true",
                assessment="All subtasks completed. Code was generated and tests pass.",
                remaining_gaps="",
            )
            agent._validator = MagicMock(return_value=mock_result)

            result = await agent.validate(
                goal="Create a FastAPI app",
                results_summary="[s1] code (completed): Created app.py",
            )

            assert result["success"] is True
            assert "assessment" in result
            assert result["remaining_gaps"] == []

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_validate_failure(self):
        with patch("Jotty.core.execution.swarms.pilot_swarm.agents.BaseOlympiadAgent._get_lm"):
            from Jotty.core.execution.swarms.pilot_swarm.agents import PilotValidatorAgent

            agent = PilotValidatorAgent.__new__(PilotValidatorAgent)
            agent.model = "haiku"
            agent.use_fast_predict = True
            agent.llm_timeout = 90
            agent._lm = MagicMock()
            agent.learned_context = ""
            agent._bus = None

            mock_result = _mock_dspy_result(
                success="false",
                assessment="Code was generated but tests were not written.",
                remaining_gaps="Add unit tests | Add integration tests",
            )
            agent._validator = MagicMock(return_value=mock_result)

            result = await agent.validate(
                goal="Create a FastAPI app with tests",
                results_summary="[s1] code (completed): Created app.py",
            )

            assert result["success"] is False
            assert len(result["remaining_gaps"]) == 2


# =============================================================================
# SWARM HELPER TESTS
# =============================================================================


class TestSwarmHelpers:
    """Test swarm helper methods."""

    @pytest.mark.unit
    def test_build_context_empty(self):
        from Jotty.core.execution.swarms.pilot_swarm.swarm import PilotSwarm

        swarm = PilotSwarm.__new__(PilotSwarm)
        ctx = swarm._build_context({})
        assert ctx == "No previous results."

    @pytest.mark.unit
    def test_build_context_with_results(self):
        from Jotty.core.execution.swarms.pilot_swarm.swarm import PilotSwarm

        swarm = PilotSwarm.__new__(PilotSwarm)
        ctx = swarm._build_context(
            {
                "s1": {"synthesis": "FastAPI is fast", "key_findings": ["async", "fast"]},
                "s2": {"explanation": "Created app.py with endpoints"},
            }
        )
        assert "[s1]" in ctx
        assert "[s2]" in ctx
        assert "FastAPI" in ctx

    @pytest.mark.unit
    def test_build_results_summary(self):
        from Jotty.core.execution.swarms.pilot_swarm.swarm import PilotSwarm

        swarm = PilotSwarm.__new__(PilotSwarm)

        subtasks = [
            Subtask(
                id="s1",
                type=SubtaskType.SEARCH,
                description="Research frameworks",
                status=SubtaskStatus.COMPLETED,
            ),
            Subtask(
                id="s2",
                type=SubtaskType.CODE,
                description="Write code",
                status=SubtaskStatus.FAILED,
            ),
        ]

        summary = swarm._build_results_summary(
            subtasks,
            {
                "s1": {"synthesis": "Found 5 frameworks"},
                "s2": {"error": "LLM timeout"},
            },
        )

        assert "[s1] search (completed)" in summary
        assert "[s2] code (failed)" in summary
        assert "ERROR: LLM timeout" in summary

    @pytest.mark.unit
    def test_build_results_summary_with_skill(self):
        from Jotty.core.execution.swarms.pilot_swarm.swarm import PilotSwarm

        swarm = PilotSwarm.__new__(PilotSwarm)

        subtasks = [
            Subtask(
                id="s1",
                type=SubtaskType.CREATE_SKILL,
                description="Create CSV skill",
                status=SubtaskStatus.COMPLETED,
            ),
        ]

        summary = swarm._build_results_summary(
            subtasks,
            {
                "s1": {"skill_name": "csv-converter"},
            },
        )

        assert "Created skill: csv-converter" in summary


# =============================================================================
# SUBTASK DEPENDENCY TESTS
# =============================================================================


class TestSubtaskDependencies:
    """Test subtask dependency handling."""

    @pytest.mark.unit
    def test_subtask_no_dependencies(self):
        st = Subtask(id="s1", type=SubtaskType.SEARCH, description="Search")
        assert st.depends_on == []

    @pytest.mark.unit
    def test_subtask_with_unmet_dependency(self):
        """Subtasks with unmet dependencies should be skippable."""
        st = Subtask(id="s2", type=SubtaskType.CODE, description="Code", depends_on=["s1"])
        completed_ids = set()
        unmet = [d for d in st.depends_on if d not in completed_ids]
        assert unmet == ["s1"]

    @pytest.mark.unit
    def test_subtask_with_met_dependency(self):
        """Subtasks with met dependencies should be executable."""
        st = Subtask(id="s2", type=SubtaskType.CODE, description="Code", depends_on=["s1"])
        completed_ids = {"s1"}
        unmet = [d for d in st.depends_on if d not in completed_ids]
        assert unmet == []


# =============================================================================
# SWARM REGISTRATION TESTS
# =============================================================================


class TestSwarmRegistration:
    """Test swarm registration."""

    @pytest.mark.unit
    def test_registered_in_swarm_registry(self):
        from Jotty.core.execution.swarms._base.registry import SwarmRegistry
        from Jotty.core.execution.swarms.pilot_swarm import PilotSwarm

        swarm_class = SwarmRegistry.get("pilot")
        assert swarm_class is PilotSwarm

    @pytest.mark.unit
    def test_lazy_import_from_core_swarms(self):
        from Jotty.core.execution.swarms import PilotSwarm

        assert PilotSwarm is not None

    @pytest.mark.unit
    def test_lazy_import_pilot(self):
        from Jotty.core.execution.swarms import pilot

        assert callable(pilot)

    @pytest.mark.unit
    def test_lazy_import_types(self):
        from Jotty.core.execution.swarms import SubtaskStatus, SubtaskType

        assert len(SubtaskType) == 7
        assert len(SubtaskStatus) == 5


# =============================================================================
# SWARM EXECUTION TESTS
# =============================================================================


class TestSwarmExecution:
    """Test swarm instantiation and config."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_swarm_instantiation(self):
        from Jotty.core.execution.swarms.pilot_swarm.swarm import PilotSwarm

        swarm = PilotSwarm()
        assert swarm.config.name == "PilotSwarm"
        assert swarm.config.domain == "pilot"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_swarm_with_custom_config(self):
        from Jotty.core.execution.swarms.pilot_swarm.swarm import PilotSwarm

        config = PilotConfig(max_subtasks=5, allow_terminal=False)
        swarm = PilotSwarm(config)
        assert swarm.config.max_subtasks == 5
        assert swarm.config.allow_terminal is False


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================


class TestConvenienceFunctions:
    """Test pilot and pilot_sync."""

    @pytest.mark.unit
    def test_pilot_is_callable(self):
        from Jotty.core.execution.swarms.pilot_swarm import pilot

        assert callable(pilot)

    @pytest.mark.unit
    def test_pilot_sync_is_callable(self):
        from Jotty.core.execution.swarms.pilot_swarm import pilot_sync

        assert callable(pilot_sync)


# =============================================================================
# SIGNATURES TESTS
# =============================================================================


class TestSignatures:
    """Test DSPy signatures can be instantiated."""

    @pytest.mark.unit
    def test_all_signatures_importable(self):
        import dspy
        from Jotty.core.execution.swarms.pilot_swarm.signatures import (
            CodePlanSignature,
            CoderSignature,
            PlannerSignature,
            SearchSignature,
            SingleFileWriteSignature,
            SkillWriterSignature,
            TerminalSignature,
            ValidatorSignature,
        )

        for sig in [
            PlannerSignature,
            SearchSignature,
            CoderSignature,
            CodePlanSignature,
            SingleFileWriteSignature,
            TerminalSignature,
            SkillWriterSignature,
            ValidatorSignature,
        ]:
            assert issubclass(sig, dspy.Signature)

    @pytest.mark.unit
    def test_signature_count(self):
        from Jotty.core.execution.swarms.pilot_swarm import signatures

        all_sigs = signatures.__all__
        assert len(all_sigs) == 8

    @pytest.mark.unit
    def test_code_plan_signature_fields(self):
        """CodePlanSignature has expected input/output fields."""
        from Jotty.core.execution.swarms.pilot_swarm.signatures import CodePlanSignature

        assert "task" in CodePlanSignature.input_fields
        assert "context" in CodePlanSignature.input_fields
        assert "files_json" in CodePlanSignature.output_fields
        assert "architecture_notes" in CodePlanSignature.output_fields

    @pytest.mark.unit
    def test_single_file_write_signature_fields(self):
        """SingleFileWriteSignature has expected input/output fields."""
        from Jotty.core.execution.swarms.pilot_swarm.signatures import SingleFileWriteSignature

        assert "file_path" in SingleFileWriteSignature.input_fields
        assert "file_description" in SingleFileWriteSignature.input_fields
        assert "project_context" in SingleFileWriteSignature.input_fields
        assert "file_content" in SingleFileWriteSignature.output_fields
        assert "explanation" in SingleFileWriteSignature.output_fields


# =============================================================================
# AGENTS IMPORT TESTS
# =============================================================================


class TestAgentsImport:
    """Test all agents are importable."""

    @pytest.mark.unit
    def test_all_agents_importable(self):
        from Jotty.core.execution.swarms.olympiad_learning_swarm.agents import BaseOlympiadAgent
        from Jotty.core.execution.swarms.pilot_swarm.agents import (
            PilotCoderAgent,
            PilotPlannerAgent,
            PilotSearchAgent,
            PilotSkillWriterAgent,
            PilotTerminalAgent,
            PilotValidatorAgent,
        )

        for agent_cls in [
            PilotPlannerAgent,
            PilotSearchAgent,
            PilotCoderAgent,
            PilotTerminalAgent,
            PilotSkillWriterAgent,
            PilotValidatorAgent,
        ]:
            assert issubclass(agent_cls, BaseOlympiadAgent)


# =============================================================================
# TERMINAL SAFETY TESTS
# =============================================================================


class TestTerminalSafety:
    """Test terminal command safety handling."""

    @pytest.mark.unit
    def test_unsafe_commands_skipped(self):
        """Verify that unsafe commands would be skipped in execution."""
        commands = [
            {"command": "ls -la", "purpose": "List files", "safe": True},
            {"command": "rm -rf /tmp/test", "purpose": "Delete files", "safe": False},
            {"command": "cat /etc/passwd", "purpose": "Read file", "safe": True},
        ]

        safe_commands = [c for c in commands if c.get("safe", False)]
        unsafe_commands = [c for c in commands if not c.get("safe", False)]

        assert len(safe_commands) == 2
        assert len(unsafe_commands) == 1
        assert unsafe_commands[0]["command"] == "rm -rf /tmp/test"

    @pytest.mark.unit
    def test_terminal_disabled_config(self):
        """Verify terminal can be disabled via config."""
        config = PilotConfig(allow_terminal=False)
        assert config.allow_terminal is False


# =============================================================================
# SKILL WRITER VALIDATION TESTS
# =============================================================================


class TestSkillWriterValidation:
    """Test skill writer output validation."""

    @pytest.mark.unit
    def test_skill_yaml_format(self):
        """Verify expected skill.yaml fields."""
        yaml_content = (
            "name: test-skill\ndescription: A test skill\nversion: 1.0.0\ntools:\n  - test_tool"
        )
        assert "name:" in yaml_content
        assert "description:" in yaml_content
        assert "version:" in yaml_content
        assert "tools:" in yaml_content

    @pytest.mark.unit
    def test_skill_name_sanitization(self):
        """Verify skill name is properly sanitized."""
        raw = "My Cool Skill!!"
        sanitized = "".join(c for c in raw.lower().replace(" ", "-") if c.isalnum() or c == "-")[
            :30
        ]
        assert sanitized == "my-cool-skill"
        assert len(sanitized) <= 30

    @pytest.mark.unit
    def test_skill_name_truncation(self):
        """Verify long skill names are truncated."""
        raw = "this-is-a-very-long-skill-name-that-exceeds-thirty-characters"
        sanitized = "".join(c for c in raw.lower().replace(" ", "-") if c.isalnum() or c == "-")[
            :30
        ]
        assert len(sanitized) <= 30


# =============================================================================
# BROWSE (VLM) HANDLER
# =============================================================================


class TestBrowseHandler:
    """Test the VLM browse subtask handler."""

    @pytest.mark.unit
    def test_browse_type_in_dispatch(self):
        """Verify BROWSE has its own handler, not a search fallback."""
        from Jotty.core.execution.swarms.pilot_swarm.swarm import PilotSwarm

        swarm = PilotSwarm()
        swarm._init_agents()
        dispatch = {
            SubtaskType.SEARCH: swarm._execute_search,
            SubtaskType.CODE: swarm._execute_code,
            SubtaskType.TERMINAL: swarm._execute_terminal,
            SubtaskType.CREATE_SKILL: swarm._execute_create_skill,
            SubtaskType.DELEGATE: swarm._execute_delegate,
            SubtaskType.ANALYZE: swarm._execute_analyze,
            SubtaskType.BROWSE: swarm._execute_browse,
        }
        # Browse should NOT be the same function as search
        assert dispatch[SubtaskType.BROWSE] != dispatch[SubtaskType.SEARCH]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_browse_fallback_when_no_vlm(self):
        """When visual-inspector is unavailable, browse falls back to search."""
        from Jotty.core.execution.swarms.pilot_swarm.swarm import PilotSwarm

        swarm = PilotSwarm()
        swarm._init_agents()

        subtask = Subtask(id="s1", type=SubtaskType.BROWSE, description="Analyze screenshot.png")

        # Mock the search fallback
        swarm._searcher = MagicMock()
        swarm._searcher.search = AsyncMock(
            return_value={
                "queries": ["screenshot analysis"],
                "synthesis": "fallback",
                "key_findings": [],
            }
        )

        # Patch importlib to fail (VLM not available)
        with patch("importlib.util.spec_from_file_location", side_effect=ImportError("no vlm")):
            result = await swarm._execute_browse(subtask, "", PilotConfig())

        # Should have fallen back to search
        assert result.get("synthesis") == "fallback"

    @pytest.mark.unit
    def test_browse_in_planner_signature(self):
        """Verify planner signature mentions browse type."""
        from Jotty.core.execution.swarms.pilot_swarm.signatures import PlannerSignature

        # Check the docstring includes browse
        assert "browse" in PlannerSignature.__doc__.lower()


# =============================================================================
# RETRY LOOP TESTS (Phase 4)
# =============================================================================


class TestRetryLoop:
    """Test Phase 4 retry loop."""

    @pytest.mark.unit
    def test_build_replan_context(self):
        """Replan context includes completed work and remaining gaps."""
        from Jotty.core.execution.swarms.pilot_swarm.swarm import PilotSwarm

        subtasks = [
            Subtask(
                id="s1",
                type=SubtaskType.SEARCH,
                description="Research frameworks",
                status=SubtaskStatus.COMPLETED,
            ),
            Subtask(
                id="s2",
                type=SubtaskType.CODE,
                description="Write app code",
                status=SubtaskStatus.COMPLETED,
            ),
        ]
        all_results = {
            "s1": {"synthesis": "Found FastAPI and Django"},
            "s2": {"explanation": "Created app.py"},
        }
        gaps = ["Add unit tests", "Add documentation"]

        ctx = PilotSwarm._build_replan_context(
            "Build web app with tests", subtasks, all_results, gaps
        )

        assert "ORIGINAL GOAL: Build web app with tests" in ctx
        assert "COMPLETED WORK:" in ctx
        assert "[s1] search (completed)" in ctx
        assert "[s2] code (completed)" in ctx
        assert "REMAINING GAPS" in ctx
        assert "Add unit tests" in ctx
        assert "Add documentation" in ctx
        assert "Do NOT repeat completed work" in ctx

    @pytest.mark.unit
    def test_build_replan_context_extracts_rich_keys(self):
        """Replan context extracts key_findings, read_content, visual_analysis."""
        from Jotty.core.execution.swarms.pilot_swarm.swarm import PilotSwarm

        subtasks = [
            Subtask(
                id="s1",
                type=SubtaskType.SEARCH,
                description="Research",
                status=SubtaskStatus.COMPLETED,
            ),
            Subtask(
                id="s2",
                type=SubtaskType.CODE,
                description="Read file",
                status=SubtaskStatus.COMPLETED,
            ),
            Subtask(
                id="s3",
                type=SubtaskType.BROWSE,
                description="Inspect image",
                status=SubtaskStatus.COMPLETED,
            ),
        ]
        all_results = {
            "s1": {"key_findings": ["Python is popular", "Rust is fast"]},
            "s2": {
                "file_operations": [{"file_path": "/tmp/x.py", "read_content": "class Foo: pass"}]
            },
            "s3": {"visual_analysis": "Screenshot shows a login form"},
        }

        ctx = PilotSwarm._build_replan_context(
            "Analyze codebase", subtasks, all_results, ["Fix bug"]
        )

        assert "Python is popular; Rust is fast" in ctx
        assert "class Foo: pass" in ctx
        assert "Screenshot shows a login form" in ctx

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_retry_loop_triggers_on_validation_failure(self):
        """When validator returns failure, planner is called again."""
        from Jotty.core.execution.swarms.pilot_swarm.swarm import PilotSwarm

        swarm = PilotSwarm.__new__(PilotSwarm)
        swarm.config = PilotConfig(max_retries=1)

        plan_result = {
            "subtasks": [{"id": "s1", "type": "analyze", "description": "Think", "depends_on": []}],
            "reasoning": "plan",
        }

        # Track how many times each phase runs
        phase_counts = {1: 0, 3: 0, 4: 0}

        # Validator: fail first time, succeed second
        validate_results = [
            {"success": False, "assessment": "Not done", "remaining_gaps": ["Missing tests"]},
            {"success": True, "assessment": "Done", "remaining_gaps": []},
        ]
        validate_idx = [0]

        # Mock planner/validator — these get called by _execute_phases but
        # their coroutine return values are consumed by executor.run_phase,
        # so we make run_phase return the right dict directly.
        swarm._planner = MagicMock()
        swarm._planner.plan = AsyncMock(return_value=plan_result)
        swarm._validator = MagicMock()
        swarm._validator.validate = AsyncMock(return_value={})

        # Mock searcher for analyze subtask
        swarm._searcher = MagicMock()
        swarm._searcher.search = AsyncMock(
            return_value={"synthesis": "analyzed", "key_findings": []}
        )

        # Mock executor — consumes the coroutine arg and returns our controlled dicts
        executor = MagicMock()

        async def mock_run_phase(phase_num, _name, _agent, _role, coro, **kwargs):
            # Consume the coroutine to avoid warnings
            if asyncio.iscoroutine(coro):
                coro.close()
            phase_counts[phase_num] = phase_counts.get(phase_num, 0) + 1
            if phase_num == 1:
                return plan_result
            if phase_num == 3:
                idx = validate_idx[0]
                validate_idx[0] += 1
                return validate_results[min(idx, len(validate_results) - 1)]
            if phase_num == 4:
                return plan_result
            return {}

        executor.run_phase = mock_run_phase
        executor.elapsed = MagicMock(return_value=1.0)

        result = await swarm._execute_phases(executor, "Build app with tests", "", swarm.config)

        assert result.retry_count == 1
        assert phase_counts[4] == 1  # re-plan happened once
        assert phase_counts[3] == 2  # validated twice

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_retry_loop_respects_max_retries(self):
        """Retry loop exits after max_retries even if validation keeps failing."""
        from Jotty.core.execution.swarms.pilot_swarm.swarm import PilotSwarm

        swarm = PilotSwarm.__new__(PilotSwarm)
        swarm.config = PilotConfig(max_retries=2)

        plan_result = {
            "subtasks": [{"id": "s1", "type": "analyze", "description": "Think", "depends_on": []}],
            "reasoning": "plan",
        }
        fail_validation = {
            "success": False,
            "assessment": "Still not done",
            "remaining_gaps": ["gap"],
        }

        swarm._planner = MagicMock()
        swarm._planner.plan = AsyncMock(return_value=plan_result)
        swarm._validator = MagicMock()
        swarm._validator.validate = AsyncMock(return_value=fail_validation)
        swarm._searcher = MagicMock()
        swarm._searcher.search = AsyncMock(
            return_value={"synthesis": "analyzed", "key_findings": []}
        )

        validate_count = [0]
        executor = MagicMock()

        async def mock_run_phase(phase_num, _name, _agent, _role, coro, **kwargs):
            if asyncio.iscoroutine(coro):
                coro.close()
            if phase_num == 1:
                return plan_result
            if phase_num == 3:
                validate_count[0] += 1
                return fail_validation
            if phase_num == 4:
                return plan_result
            return {}

        executor.run_phase = mock_run_phase
        executor.elapsed = MagicMock(return_value=2.0)

        result = await swarm._execute_phases(executor, "Impossible goal", "", swarm.config)

        assert result.retry_count == 2
        assert result.success is False
        assert validate_count[0] == 3  # initial + 2 retries

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_empty_replan_stops_retries(self):
        """If replanner returns no subtasks, stop retrying immediately."""
        from Jotty.core.execution.swarms.pilot_swarm.swarm import PilotSwarm

        swarm = PilotSwarm.__new__(PilotSwarm)
        swarm.config = PilotConfig(max_retries=3)

        plan_result = {
            "subtasks": [{"id": "s1", "type": "analyze", "description": "Think", "depends_on": []}],
            "reasoning": "plan",
        }
        empty_replan = {"subtasks": [], "reasoning": "nothing to do"}
        fail_validation = {"success": False, "assessment": "Not done", "remaining_gaps": ["gap"]}

        swarm._planner = MagicMock()
        swarm._planner.plan = AsyncMock(return_value=plan_result)
        swarm._validator = MagicMock()
        swarm._validator.validate = AsyncMock(return_value=fail_validation)
        swarm._searcher = MagicMock()
        swarm._searcher.search = AsyncMock(
            return_value={"synthesis": "analyzed", "key_findings": []}
        )

        executor = MagicMock()

        async def mock_run_phase(phase_num, _name, _agent, _role, coro, **kwargs):
            if asyncio.iscoroutine(coro):
                coro.close()
            if phase_num == 1:
                return plan_result
            if phase_num == 3:
                return fail_validation
            if phase_num == 4:
                return empty_replan  # replanner returns nothing
            return {}

        executor.run_phase = mock_run_phase
        executor.elapsed = MagicMock(return_value=1.0)

        result = await swarm._execute_phases(executor, "Some goal", "", swarm.config)

        # Should stop after first failed validation + empty replan, NOT retry 3 times
        assert result.retry_count == 0  # never incremented because break before increment
        assert result.success is False

    @pytest.mark.unit
    def test_pilot_result_has_retry_count(self):
        """PilotResult has retry_count field."""
        result = PilotResult(
            success=True,
            swarm_name="PilotSwarm",
            domain="pilot",
            output={},
            execution_time=1.0,
            goal="test",
            retry_count=3,
        )
        assert result.retry_count == 3

        # Default is 0
        result2 = PilotResult(
            success=True,
            swarm_name="PilotSwarm",
            domain="pilot",
            output={},
            execution_time=1.0,
            goal="test",
        )
        assert result2.retry_count == 0


# =============================================================================
# FILE READ/EDIT TESTS
# =============================================================================


class TestFileReadEdit:
    """Test file read and edit static methods."""

    @pytest.mark.unit
    def test_read_file_returns_content(self, tmp_path):
        """Read existing file returns its content."""
        from Jotty.core.execution.swarms.pilot_swarm.swarm import PilotSwarm

        f = tmp_path / "test.txt"
        f.write_text("hello world")
        result = PilotSwarm._read_file(str(f))
        assert result == "hello world"

    @pytest.mark.unit
    def test_read_file_not_found(self):
        """Read non-existent file returns error message."""
        from Jotty.core.execution.swarms.pilot_swarm.swarm import PilotSwarm

        result = PilotSwarm._read_file("/nonexistent/path/abc.txt")
        assert "[ERROR]" in result
        assert "not found" in result.lower()

    @pytest.mark.unit
    def test_read_file_truncates(self, tmp_path):
        """Read large file truncates at max_chars."""
        from Jotty.core.execution.swarms.pilot_swarm.swarm import PilotSwarm

        f = tmp_path / "big.txt"
        f.write_text("x" * 10000)
        result = PilotSwarm._read_file(str(f), max_chars=100)
        assert len(result) < 200  # truncated content + suffix
        assert "truncated" in result.lower()

    @pytest.mark.unit
    def test_edit_file_replaces_content(self, tmp_path):
        """Edit file replaces old_content with new_content."""
        from Jotty.core.execution.swarms.pilot_swarm.swarm import PilotSwarm

        f = tmp_path / "code.py"
        f.write_text("def foo():\n    return 1\n")
        success = PilotSwarm._edit_file(str(f), "return 1", "return 42")
        assert success is True
        assert "return 42" in f.read_text()
        assert "return 1" not in f.read_text()

    @pytest.mark.unit
    def test_edit_file_old_content_not_found(self, tmp_path):
        """Edit file returns False when old_content doesn't match."""
        from Jotty.core.execution.swarms.pilot_swarm.swarm import PilotSwarm

        f = tmp_path / "code.py"
        f.write_text("def foo():\n    return 1\n")
        success = PilotSwarm._edit_file(str(f), "NONEXISTENT STRING", "replacement")
        assert success is False
        assert f.read_text() == "def foo():\n    return 1\n"  # unchanged

    @pytest.mark.unit
    def test_edit_file_missing_file(self):
        """Edit file returns False for non-existent file."""
        from Jotty.core.execution.swarms.pilot_swarm.swarm import PilotSwarm

        success = PilotSwarm._edit_file("/nonexistent/file.py", "old", "new")
        assert success is False


# =============================================================================
# PARALLEL EXECUTION TESTS
# =============================================================================


class TestParallelExecution:
    """Test wave computation for parallel execution."""

    @pytest.mark.unit
    def test_compute_waves_no_deps(self):
        """Subtasks with no deps all go in wave 0."""
        from Jotty.core.execution.swarms.pilot_swarm.swarm import PilotSwarm

        subtasks = [
            Subtask(id="s1", type=SubtaskType.SEARCH, description="A"),
            Subtask(id="s2", type=SubtaskType.CODE, description="B"),
            Subtask(id="s3", type=SubtaskType.TERMINAL, description="C"),
        ]
        waves = PilotSwarm._compute_waves(subtasks)
        assert len(waves) == 1
        assert len(waves[0]) == 3

    @pytest.mark.unit
    def test_compute_waves_linear_deps(self):
        """Linear chain: s1 → s2 → s3, one per wave."""
        from Jotty.core.execution.swarms.pilot_swarm.swarm import PilotSwarm

        subtasks = [
            Subtask(id="s1", type=SubtaskType.SEARCH, description="A"),
            Subtask(id="s2", type=SubtaskType.CODE, description="B", depends_on=["s1"]),
            Subtask(id="s3", type=SubtaskType.TERMINAL, description="C", depends_on=["s2"]),
        ]
        waves = PilotSwarm._compute_waves(subtasks)
        assert len(waves) == 3
        assert waves[0][0].id == "s1"
        assert waves[1][0].id == "s2"
        assert waves[2][0].id == "s3"

    @pytest.mark.unit
    def test_compute_waves_diamond_deps(self):
        """Diamond: s1 → s2,s3 → s4."""
        from Jotty.core.execution.swarms.pilot_swarm.swarm import PilotSwarm

        subtasks = [
            Subtask(id="s1", type=SubtaskType.SEARCH, description="A"),
            Subtask(id="s2", type=SubtaskType.CODE, description="B", depends_on=["s1"]),
            Subtask(id="s3", type=SubtaskType.CODE, description="C", depends_on=["s1"]),
            Subtask(id="s4", type=SubtaskType.ANALYZE, description="D", depends_on=["s2", "s3"]),
        ]
        waves = PilotSwarm._compute_waves(subtasks)
        assert len(waves) == 3
        assert waves[0][0].id == "s1"
        # Wave 1 has s2 and s3 (in some order)
        wave1_ids = {st.id for st in waves[1]}
        assert wave1_ids == {"s2", "s3"}
        assert waves[2][0].id == "s4"

    @pytest.mark.unit
    def test_compute_waves_circular_deps_handled(self):
        """Circular deps get dumped into final wave."""
        from Jotty.core.execution.swarms.pilot_swarm.swarm import PilotSwarm

        subtasks = [
            Subtask(id="s1", type=SubtaskType.SEARCH, description="A", depends_on=["s2"]),
            Subtask(id="s2", type=SubtaskType.CODE, description="B", depends_on=["s1"]),
        ]
        waves = PilotSwarm._compute_waves(subtasks)
        # Should not crash — circular deps dumped into a wave
        assert len(waves) >= 1
        all_ids = {st.id for wave in waves for st in wave}
        assert all_ids == {"s1", "s2"}

    @pytest.mark.unit
    def test_compute_waves_empty(self):
        """Empty input returns empty waves."""
        from Jotty.core.execution.swarms.pilot_swarm.swarm import PilotSwarm

        waves = PilotSwarm._compute_waves([])
        assert waves == []

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_parallel_execution_is_concurrent(self):
        """Prove wave-based execution runs tasks concurrently, not sequentially.

        3 independent subtasks each sleep 0.2s. Sequential = 0.6s, parallel < 0.4s.
        """
        import time

        from Jotty.core.execution.swarms.pilot_swarm.swarm import PilotSwarm

        swarm = PilotSwarm.__new__(PilotSwarm)
        swarm.config = PilotConfig(max_concurrent=3)

        execution_log = []

        async def mock_execute_subtask(subtask, context, config):
            execution_log.append(("start", subtask.id, time.monotonic()))
            await asyncio.sleep(0.2)
            execution_log.append(("end", subtask.id, time.monotonic()))
            return {"synthesis": f"done {subtask.id}"}

        swarm._execute_subtask = mock_execute_subtask

        subtasks = [
            Subtask(id="s1", type=SubtaskType.SEARCH, description="A"),
            Subtask(id="s2", type=SubtaskType.SEARCH, description="B"),
            Subtask(id="s3", type=SubtaskType.SEARCH, description="C"),
        ]

        all_results = {}
        waves = PilotSwarm._compute_waves(subtasks)
        semaphore = asyncio.Semaphore(swarm.config.max_concurrent)

        start = time.monotonic()
        for wave in waves:

            async def _run(st):
                async with semaphore:
                    result = await mock_execute_subtask(st, "", swarm.config)
                    all_results[st.id] = result
                    st.status = SubtaskStatus.COMPLETED

            await asyncio.gather(*[_run(st) for st in wave])
        elapsed = time.monotonic() - start

        # All 3 completed
        assert len(all_results) == 3
        # Parallel: should take ~0.2s, not ~0.6s
        assert elapsed < 0.45, f"Took {elapsed:.2f}s — not parallel!"

        # Verify all 3 started before any finished
        starts = [t for ev, _, t in execution_log if ev == "start"]
        ends = [t for ev, _, t in execution_log if ev == "end"]
        assert max(starts) < min(ends), "Tasks didn't overlap — not concurrent"


# =============================================================================
# IMPROVED CONTEXT TESTS
# =============================================================================


class TestImprovedContext:
    """Test improved context accumulation."""

    @pytest.mark.unit
    def test_context_includes_read_content(self):
        """Context includes read_content from file operations."""
        from Jotty.core.execution.swarms.pilot_swarm.swarm import PilotSwarm

        swarm = PilotSwarm.__new__(PilotSwarm)

        results = {
            "s1": {
                "file_operations": [
                    {
                        "file_path": "/tmp/test.py",
                        "action": "read",
                        "read_content": "def hello(): return 42",
                    },
                ],
                "explanation": "Read file",
            },
        }
        ctx = swarm._build_context(results)
        assert "read_content" in ctx
        assert "def hello(): return 42" in ctx

    @pytest.mark.unit
    def test_context_uses_8_results(self):
        """Context keeps last 8 results (increased from 5)."""
        from Jotty.core.execution.swarms.pilot_swarm.swarm import PilotSwarm

        swarm = PilotSwarm.__new__(PilotSwarm)

        results = {}
        for i in range(10):
            results[f"s{i}"] = {"synthesis": f"Result {i}"}

        ctx = swarm._build_context(results)
        # Should have results s2..s9 (last 8)
        assert "[s2]" in ctx
        assert "[s9]" in ctx
        # s0 and s1 should be dropped
        assert "[s0]" not in ctx
        assert "[s1]" not in ctx


# =============================================================================
# SUBTASK RETRY TESTS
# =============================================================================


class TestSubtaskRetry:
    """Test subtask-level retry on transient errors."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_retry_on_timeout(self):
        """Retries on TimeoutError and succeeds on second attempt."""
        from Jotty.core.execution.swarms.pilot_swarm.swarm import PilotSwarm

        swarm = PilotSwarm.__new__(PilotSwarm)
        swarm.config = PilotConfig()

        call_count = [0]

        async def mock_execute_subtask(subtask, context, config):
            call_count[0] += 1
            if call_count[0] == 1:
                raise TimeoutError("LLM timed out")
            return {"synthesis": "success"}

        swarm._execute_subtask = mock_execute_subtask

        subtask = Subtask(id="s1", type=SubtaskType.SEARCH, description="Search")
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await swarm._execute_subtask_with_retry(
                subtask, "", swarm.config, max_retries=2
            )

        assert result == {"synthesis": "success"}
        assert call_count[0] == 2

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_no_retry_on_value_error(self):
        """Does NOT retry on ValueError — raises immediately."""
        from Jotty.core.execution.swarms.pilot_swarm.swarm import PilotSwarm

        swarm = PilotSwarm.__new__(PilotSwarm)
        swarm.config = PilotConfig()

        async def mock_execute_subtask(subtask, context, config):
            raise ValueError("Bad input")

        swarm._execute_subtask = mock_execute_subtask

        subtask = Subtask(id="s1", type=SubtaskType.SEARCH, description="Search")

        with pytest.raises(ValueError, match="Bad input"):
            await swarm._execute_subtask_with_retry(subtask, "", swarm.config, max_retries=2)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_retry_exhausted_raises(self):
        """After max retries, the transient error is raised."""
        from Jotty.core.execution.swarms.pilot_swarm.swarm import PilotSwarm

        swarm = PilotSwarm.__new__(PilotSwarm)
        swarm.config = PilotConfig()

        call_count = [0]

        async def mock_execute_subtask(subtask, context, config):
            call_count[0] += 1
            raise ConnectionError("Connection refused")

        swarm._execute_subtask = mock_execute_subtask

        subtask = Subtask(id="s1", type=SubtaskType.SEARCH, description="Search")

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(ConnectionError, match="Connection refused"):
                await swarm._execute_subtask_with_retry(subtask, "", swarm.config, max_retries=1)

        assert call_count[0] == 2  # initial + 1 retry

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_retry_on_rate_limit_string(self):
        """Retries when error message contains rate limit indicator."""
        from Jotty.core.execution.swarms.pilot_swarm.swarm import PilotSwarm

        swarm = PilotSwarm.__new__(PilotSwarm)
        swarm.config = PilotConfig()

        call_count = [0]

        async def mock_execute_subtask(subtask, context, config):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("API returned 429 rate limit exceeded")
            return {"synthesis": "success after rate limit"}

        swarm._execute_subtask = mock_execute_subtask

        subtask = Subtask(id="s1", type=SubtaskType.SEARCH, description="Search")
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await swarm._execute_subtask_with_retry(
                subtask, "", swarm.config, max_retries=2
            )

        assert result == {"synthesis": "success after rate limit"}
        assert call_count[0] == 2


# =============================================================================
# CODE HANDLER READ/EDIT INTEGRATION
# =============================================================================


class TestCodeHandlerReadEdit:
    """Test _execute_code_single_shot with read/edit actions."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_code_read_action(self, tmp_path):
        """Read action populates read_content in file operation."""
        from Jotty.core.execution.swarms.pilot_swarm.swarm import PilotSwarm

        swarm = PilotSwarm.__new__(PilotSwarm)
        config = PilotConfig()

        test_file = tmp_path / "existing.py"
        test_file.write_text("print('hello')")

        async def mock_code(**kwargs):
            return {
                "file_operations": [
                    {"file_path": str(test_file), "action": "read"},
                ],
                "explanation": "Read the file",
            }

        swarm._coder = MagicMock()
        swarm._coder.code = mock_code

        subtask = Subtask(id="s1", type=SubtaskType.CODE, description="Read a file")
        result = await swarm._execute_code_single_shot(subtask, "", config)

        assert result["file_operations"][0]["read_content"] == "print('hello')"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_code_edit_action(self, tmp_path):
        """Edit action surgically replaces content."""
        from Jotty.core.execution.swarms.pilot_swarm.swarm import PilotSwarm

        swarm = PilotSwarm.__new__(PilotSwarm)
        config = PilotConfig(allow_file_write=True)

        test_file = tmp_path / "code.py"
        test_file.write_text("def foo():\n    return 1\n")

        async def mock_code(**kwargs):
            return {
                "file_operations": [
                    {
                        "file_path": str(test_file),
                        "action": "edit",
                        "old_content": "return 1",
                        "content": "return 42",
                        "description": "Change return value",
                    },
                ],
                "explanation": "Edited code",
            }

        swarm._coder = MagicMock()
        swarm._coder.code = mock_code

        subtask = Subtask(id="s1", type=SubtaskType.CODE, description="Edit a file")
        result = await swarm._execute_code_single_shot(subtask, "", config)

        assert result["file_operations"][0]["edit_success"] is True
        assert "return 42" in test_file.read_text()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_code_edit_skipped_when_writes_disabled(self, tmp_path):
        """Edit action skipped when allow_file_write is False."""
        from Jotty.core.execution.swarms.pilot_swarm.swarm import PilotSwarm

        swarm = PilotSwarm.__new__(PilotSwarm)
        config = PilotConfig(allow_file_write=False)

        test_file = tmp_path / "code.py"
        test_file.write_text("original content")

        async def mock_code(**kwargs):
            return {
                "file_operations": [
                    {
                        "file_path": str(test_file),
                        "action": "edit",
                        "old_content": "original",
                        "content": "modified",
                    },
                ],
                "explanation": "Edit",
            }

        swarm._coder = MagicMock()
        swarm._coder.code = mock_code

        subtask = Subtask(id="s1", type=SubtaskType.CODE, description="Edit")
        result = await swarm._execute_code_single_shot(subtask, "", config)

        # File should be unchanged
        assert test_file.read_text() == "original content"


# =============================================================================
# TWO-PHASE CODE GENERATION TESTS
# =============================================================================


class TestTwoPhaseCodeGeneration:
    """Test the new two-phase code generation (plan → write loop)."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_plan_files_returns_metadata(self):
        """plan_files() returns file list with metadata (no code content)."""
        mock_plan_result = MagicMock()
        mock_plan_result.files_json = json.dumps(
            [
                {
                    "file_path": "app/main.py",
                    "description": "FastAPI entry point",
                    "language": "python",
                },
                {
                    "file_path": "app/models.py",
                    "description": "Pydantic models",
                    "language": "python",
                },
                {"file_path": "app/routes.py", "description": "API routes", "language": "python"},
            ]
        )
        mock_plan_result.architecture_notes = "Layered architecture with FastAPI"

        with patch(
            "Jotty.core.execution.swarms.pilot_swarm.agents.BaseOlympiadAgent.__init__",
            return_value=None,
        ):
            from Jotty.core.execution.swarms.pilot_swarm.agents import PilotCoderAgent

            agent = PilotCoderAgent.__new__(PilotCoderAgent)
            agent._lm = MagicMock()
            agent._file_writer_lm = None
            agent.model = "haiku"
            agent.use_fast_predict = True
            agent.llm_timeout = 90
            agent._planner = MagicMock(return_value=mock_plan_result)
            agent._broadcast = MagicMock()

            result = await agent.plan_files(
                task="Create a REST API with FastAPI",
                context="No existing code",
            )

        assert isinstance(result, dict)
        assert len(result["files"]) == 3
        assert result["files"][0]["file_path"] == "app/main.py"
        assert result["files"][0]["language"] == "python"
        # Verify no code content in plan
        for f in result["files"]:
            assert "content" not in f
        assert "FastAPI" in result["architecture_notes"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_write_single_file_returns_content(self):
        """write_single_file() returns file content and explanation."""
        mock_result = MagicMock()
        mock_result.file_content = (
            "from fastapi import FastAPI\n\napp = FastAPI()\n\n"
            '@app.get("/")\ndef root():\n    return {"message": "Hello"}\n'
        )
        mock_result.explanation = "Created FastAPI entry point with root endpoint"

        with patch(
            "Jotty.core.execution.swarms.pilot_swarm.agents.BaseOlympiadAgent.__init__",
            return_value=None,
        ):
            from Jotty.core.execution.swarms.pilot_swarm.agents import PilotCoderAgent

            agent = PilotCoderAgent.__new__(PilotCoderAgent)
            agent._lm = MagicMock()
            agent._file_writer_lm = MagicMock()
            agent.model = "haiku"
            agent.use_fast_predict = True
            agent.llm_timeout = 90
            agent._broadcast = MagicMock()
            agent._file_writer = MagicMock(return_value=mock_result)

            result = await agent.write_single_file(
                file_path="app/main.py",
                description="FastAPI entry point",
                project_context="Layered architecture",
            )

        assert result["file_path"] == "app/main.py"
        assert "FastAPI" in result["content"]
        assert "entry point" in result["explanation"].lower()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_write_single_file_strips_markdown_fences(self):
        """write_single_file() strips markdown code fences from LLM output."""
        mock_result = MagicMock()
        mock_result.file_content = '```python\nprint("hello")\n```'
        mock_result.explanation = "Simple script"

        with patch(
            "Jotty.core.execution.swarms.pilot_swarm.agents.BaseOlympiadAgent.__init__",
            return_value=None,
        ):
            from Jotty.core.execution.swarms.pilot_swarm.agents import PilotCoderAgent

            agent = PilotCoderAgent.__new__(PilotCoderAgent)
            agent._lm = MagicMock()
            agent._file_writer_lm = MagicMock()
            agent.model = "haiku"
            agent.use_fast_predict = True
            agent.llm_timeout = 90
            agent._broadcast = MagicMock()
            agent._file_writer = MagicMock(return_value=mock_result)

            result = await agent.write_single_file(file_path="hello.py", description="Hello script")

        assert "```" not in result["content"]
        assert 'print("hello")' in result["content"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_code_fast_path(self):
        """≤2 files in plan → delegates to _execute_code_single_shot."""
        from Jotty.core.execution.swarms.pilot_swarm.swarm import PilotSwarm

        swarm = PilotSwarm.__new__(PilotSwarm)
        config = PilotConfig(allow_file_write=False)
        subtask = Subtask(id="s1", type=SubtaskType.CODE, description="Create a script")

        mock_coder = MagicMock()
        mock_coder.plan_files = AsyncMock(
            return_value={
                "files": [{"file_path": "main.py", "description": "Entry", "language": "python"}],
                "architecture_notes": "Simple script",
            }
        )
        swarm._coder = mock_coder

        swarm._execute_code_single_shot = AsyncMock(
            return_value={
                "file_operations": [
                    {"file_path": "main.py", "action": "create", "content": "print()"}
                ],
                "explanation": "Created",
            }
        )

        result = await swarm._execute_code(subtask, "ctx", config)

        swarm._execute_code_single_shot.assert_called_once_with(subtask, "ctx", config)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_code_multi_file(self):
        """3+ files in plan → writes each file individually."""
        from Jotty.core.execution.swarms.pilot_swarm.swarm import PilotSwarm

        swarm = PilotSwarm.__new__(PilotSwarm)
        config = PilotConfig(allow_file_write=False)
        subtask = Subtask(id="s1", type=SubtaskType.CODE, description="Create API")

        file_plan = [
            {"file_path": "app/main.py", "description": "Entry", "language": "python"},
            {"file_path": "app/models.py", "description": "Models", "language": "python"},
            {"file_path": "app/routes.py", "description": "Routes", "language": "python"},
            {"file_path": "requirements.txt", "description": "Deps", "language": "text"},
        ]

        mock_coder = MagicMock()
        mock_coder.plan_files = AsyncMock(
            return_value={
                "files": file_plan,
                "architecture_notes": "Clean layered architecture",
            }
        )

        call_count = 0

        async def mock_write(file_path, description, project_context=""):
            nonlocal call_count
            call_count += 1
            return {
                "file_path": file_path,
                "content": f"# {file_path}\n",
                "explanation": f"Created {file_path}",
            }

        mock_coder.write_single_file = mock_write
        swarm._coder = mock_coder
        swarm._sandbox = None

        result = await swarm._execute_code(subtask, "ctx", config)

        assert len(result["file_operations"]) == 4
        assert call_count == 4
        assert result["explanation"] == "Clean layered architecture"
        # Relative paths get prefixed with project_dir (defaults to /tmp/pilot_projects/project/)
        paths = [op["file_path"] for op in result["file_operations"]]
        project_dir = result["project_dir"]
        assert all(p.startswith(project_dir) for p in paths)
        # Verify file names preserved
        basenames = [p.split("/")[-1] for p in paths]
        assert basenames == ["main.py", "models.py", "routes.py", "requirements.txt"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_code_multi_file_accumulates_context(self):
        """Each file gets context from previously written files."""
        from Jotty.core.execution.swarms.pilot_swarm.swarm import PilotSwarm

        swarm = PilotSwarm.__new__(PilotSwarm)
        config = PilotConfig(allow_file_write=False)
        subtask = Subtask(id="s1", type=SubtaskType.CODE, description="Create API")

        file_plan = [
            {"file_path": "a.py", "description": "First", "language": "python"},
            {"file_path": "b.py", "description": "Second", "language": "python"},
            {"file_path": "c.py", "description": "Third", "language": "python"},
        ]

        mock_coder = MagicMock()
        mock_coder.plan_files = AsyncMock(
            return_value={
                "files": file_plan,
                "architecture_notes": "Sequential deps",
            }
        )

        captured_contexts = []

        async def mock_write(file_path, description, project_context=""):
            captured_contexts.append(project_context)
            return {
                "file_path": file_path,
                "content": f"# {file_path}\n",
                "explanation": f"Created {file_path}",
            }

        mock_coder.write_single_file = mock_write
        swarm._coder = mock_coder
        swarm._sandbox = None

        await swarm._execute_code(subtask, "", config)

        # First file: no previous files
        assert "ALREADY WRITTEN FILES" not in captured_contexts[0]
        # Second file: should see first file
        assert "a.py" in captured_contexts[1]
        # Third file: should see first two
        assert "a.py" in captured_contexts[2]
        assert "b.py" in captured_contexts[2]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_code_multi_file_writes_to_disk(self, tmp_path):
        """Multi-file path writes files to disk when allow_file_write=True."""
        from Jotty.core.execution.swarms.pilot_swarm.swarm import PilotSwarm

        swarm = PilotSwarm.__new__(PilotSwarm)
        config = PilotConfig(allow_file_write=True)
        subtask = Subtask(id="s1", type=SubtaskType.CODE, description="Create project")

        file_plan = [
            {
                "file_path": str(tmp_path / "src" / "main.py"),
                "description": "Main",
                "language": "python",
            },
            {
                "file_path": str(tmp_path / "src" / "utils.py"),
                "description": "Utils",
                "language": "python",
            },
            {
                "file_path": str(tmp_path / "README.md"),
                "description": "Readme",
                "language": "markdown",
            },
        ]

        mock_coder = MagicMock()
        mock_coder.plan_files = AsyncMock(
            return_value={
                "files": file_plan,
                "architecture_notes": "Simple project",
            }
        )

        from pathlib import Path as _Path

        async def mock_write(file_path, description, project_context=""):
            return {
                "file_path": file_path,
                "content": f"# Generated: {_Path(file_path).name}\n",
                "explanation": f"Created {file_path}",
            }

        mock_coder.write_single_file = mock_write
        swarm._coder = mock_coder
        swarm._sandbox = None

        await swarm._execute_code(subtask, "", config)

        # Verify files were actually written
        assert (tmp_path / "src" / "main.py").exists()
        assert (tmp_path / "src" / "utils.py").exists()
        assert (tmp_path / "README.md").exists()
        assert "Generated: main.py" in (tmp_path / "src" / "main.py").read_text()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_sandbox_validation_on_python_files(self):
        """Python files are validated via SandboxManager syntax check."""
        from Jotty.core.execution.swarms.pilot_swarm.swarm import PilotSwarm

        swarm = PilotSwarm.__new__(PilotSwarm)
        config = PilotConfig(allow_file_write=False)
        subtask = Subtask(id="s1", type=SubtaskType.CODE, description="Create code")

        file_plan = [
            {"file_path": "a.py", "description": "First", "language": "python"},
            {"file_path": "b.py", "description": "Second", "language": "python"},
            {"file_path": "c.js", "description": "JS file", "language": "javascript"},
        ]

        mock_coder = MagicMock()
        mock_coder.plan_files = AsyncMock(
            return_value={
                "files": file_plan,
                "architecture_notes": "Mixed",
            }
        )

        async def mock_write(file_path, description, project_context=""):
            return {
                "file_path": file_path,
                "content": f"print('{file_path}')\n",
                "explanation": f"Created {file_path}",
            }

        mock_coder.write_single_file = mock_write
        swarm._coder = mock_coder

        mock_sandbox = MagicMock()
        mock_sandbox.execute_sandboxed = AsyncMock(return_value=MagicMock(success=True, error=""))
        swarm._sandbox = mock_sandbox

        await swarm._execute_code(subtask, "", config)

        # Only the 2 Python files should be validated, not the JS file
        assert mock_sandbox.execute_sandboxed.call_count == 2

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_sandbox_failure_does_not_block(self):
        """Syntax error in generated code logs warning but doesn't stop execution."""
        from Jotty.core.execution.swarms.pilot_swarm.swarm import PilotSwarm

        swarm = PilotSwarm.__new__(PilotSwarm)
        config = PilotConfig(allow_file_write=False)
        subtask = Subtask(id="s1", type=SubtaskType.CODE, description="Create code")

        file_plan = [
            {"file_path": "bad.py", "description": "Bad", "language": "python"},
            {"file_path": "good.py", "description": "Good", "language": "python"},
            {"file_path": "also.py", "description": "Also good", "language": "python"},
        ]

        mock_coder = MagicMock()
        mock_coder.plan_files = AsyncMock(
            return_value={
                "files": file_plan,
                "architecture_notes": "Test",
            }
        )

        async def mock_write(file_path, description, project_context=""):
            return {
                "file_path": file_path,
                "content": "def broken(\n" if "bad" in file_path else "print('ok')\n",
                "explanation": f"Created {file_path}",
            }

        mock_coder.write_single_file = mock_write
        swarm._coder = mock_coder

        call_idx = 0

        async def mock_sandbox_exec(**kwargs):
            nonlocal call_idx
            call_idx += 1
            if call_idx == 1:
                return MagicMock(success=False, error="SyntaxError: unexpected EOF")
            return MagicMock(success=True, error="")

        mock_sandbox = MagicMock()
        mock_sandbox.execute_sandboxed = mock_sandbox_exec
        swarm._sandbox = mock_sandbox

        result = await swarm._execute_code(subtask, "", config)

        # All 3 files should still be in the result
        assert len(result["file_operations"]) == 3


# =============================================================================
# HELPER METHOD TESTS (project context, sandbox)
# =============================================================================


class TestProjectContextBuilder:
    """Test _build_project_context() and _get_sandbox()."""

    @pytest.mark.unit
    def test_build_project_context_empty(self):
        from Jotty.core.execution.swarms.pilot_swarm.swarm import PilotSwarm

        result = PilotSwarm._build_project_context("", [])
        assert result == "No previous context."

    @pytest.mark.unit
    def test_build_project_context_with_notes(self):
        from Jotty.core.execution.swarms.pilot_swarm.swarm import PilotSwarm

        result = PilotSwarm._build_project_context("Clean architecture", [])
        assert "ARCHITECTURE:" in result
        assert "Clean architecture" in result

    @pytest.mark.unit
    def test_build_project_context_with_summaries(self):
        from Jotty.core.execution.swarms.pilot_swarm.swarm import PilotSwarm

        summaries = ["app/main.py: Entry point", "app/models.py: Data models"]
        result = PilotSwarm._build_project_context("Notes", summaries)
        assert "ALREADY WRITTEN FILES:" in result
        assert "app/main.py" in result
        assert "app/models.py" in result

    @pytest.mark.unit
    def test_build_project_context_accumulates(self):
        from Jotty.core.execution.swarms.pilot_swarm.swarm import PilotSwarm

        ctx1 = PilotSwarm._build_project_context("Notes", ["a.py: file A"])
        ctx2 = PilotSwarm._build_project_context("Notes", ["a.py: file A", "b.py: file B"])
        assert len(ctx2) > len(ctx1)
        assert "b.py" in ctx2

    @pytest.mark.unit
    def test_get_sandbox_returns_manager(self):
        from Jotty.core.execution.swarms.pilot_swarm.swarm import PilotSwarm

        swarm = PilotSwarm.__new__(PilotSwarm)

        with patch("Jotty.core.execution.swarms.pilot_swarm.swarm.get_sandbox_manager") as mock_get:
            mock_manager = MagicMock()
            mock_get.return_value = mock_manager
            result = swarm._get_sandbox()
            assert result is mock_manager

    @pytest.mark.unit
    def test_get_sandbox_caches_result(self):
        from Jotty.core.execution.swarms.pilot_swarm.swarm import PilotSwarm

        swarm = PilotSwarm.__new__(PilotSwarm)

        with patch("Jotty.core.execution.swarms.pilot_swarm.swarm.get_sandbox_manager") as mock_get:
            mock_manager = MagicMock()
            mock_get.return_value = mock_manager
            result1 = swarm._get_sandbox()
            result2 = swarm._get_sandbox()
            mock_get.assert_called_once()
            assert result1 is result2

    @pytest.mark.unit
    def test_get_sandbox_returns_none_on_error(self):
        from Jotty.core.execution.swarms.pilot_swarm.swarm import PilotSwarm

        swarm = PilotSwarm.__new__(PilotSwarm)

        with patch(
            "Jotty.core.execution.swarms.pilot_swarm.swarm.get_sandbox_manager",
            side_effect=RuntimeError("Docker unavailable"),
        ):
            result = swarm._get_sandbox()
            assert result is None


# =============================================================================
# TERMINAL SANDBOX INTEGRATION TESTS
# =============================================================================


class TestTerminalSandboxIntegration:
    """Test _execute_terminal() uses SandboxManager when available."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_terminal_uses_sandbox(self):
        """Terminal commands use SandboxManager for isolation."""
        from Jotty.core.execution.swarms.pilot_swarm.swarm import PilotSwarm

        swarm = PilotSwarm.__new__(PilotSwarm)
        config = PilotConfig(allow_terminal=True)
        subtask = Subtask(id="t1", type=SubtaskType.TERMINAL, description="Run tests")

        mock_terminal = MagicMock()
        mock_terminal.execute = AsyncMock(
            return_value={
                "commands": [{"command": "pytest tests/", "purpose": "Run tests", "safe": True}],
                "safety_assessment": "Safe",
            }
        )
        swarm._terminal = mock_terminal

        mock_sandbox = MagicMock()
        mock_sandbox.execute_sandboxed = AsyncMock(
            return_value=MagicMock(
                success=True,
                stdout="3 passed",
                stderr="",
                exit_code=0,
            )
        )
        swarm._sandbox = mock_sandbox

        result = await swarm._execute_terminal(subtask, "context", config)

        mock_sandbox.execute_sandboxed.assert_called_once()
        assert result["command_outputs"][0]["stdout"] == "3 passed"
        assert result["command_outputs"][0]["returncode"] == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_terminal_falls_back_without_sandbox(self):
        """Terminal uses raw subprocess when sandbox unavailable."""
        from Jotty.core.execution.swarms.pilot_swarm.swarm import PilotSwarm

        swarm = PilotSwarm.__new__(PilotSwarm)
        config = PilotConfig(allow_terminal=True)
        subtask = Subtask(id="t1", type=SubtaskType.TERMINAL, description="Echo")

        mock_terminal = MagicMock()
        mock_terminal.execute = AsyncMock(
            return_value={
                "commands": [{"command": "echo hello", "purpose": "Echo", "safe": True}],
                "safety_assessment": "Safe",
            }
        )
        swarm._terminal = mock_terminal
        swarm._sandbox = None

        with patch("subprocess.run") as mock_run:
            mock_proc = MagicMock()
            mock_proc.stdout = "hello\n"
            mock_proc.stderr = ""
            mock_proc.returncode = 0
            mock_run.return_value = mock_proc

            result = await swarm._execute_terminal(subtask, "", config)

        assert mock_run.called
        assert result["command_outputs"][0]["stdout"] == "hello\n"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_terminal_skips_unsafe_commands(self):
        """Unsafe commands are skipped regardless of sandbox."""
        from Jotty.core.execution.swarms.pilot_swarm.swarm import PilotSwarm

        swarm = PilotSwarm.__new__(PilotSwarm)
        config = PilotConfig(allow_terminal=True)
        subtask = Subtask(id="t1", type=SubtaskType.TERMINAL, description="Delete")

        mock_terminal = MagicMock()
        mock_terminal.execute = AsyncMock(
            return_value={
                "commands": [{"command": "rm -rf /", "purpose": "Destroy", "safe": False}],
                "safety_assessment": "Dangerous!",
            }
        )
        swarm._terminal = mock_terminal

        mock_sandbox = MagicMock()
        swarm._sandbox = mock_sandbox

        result = await swarm._execute_terminal(subtask, "", config)

        assert result["command_outputs"][0]["skipped"] is True
        mock_sandbox.execute_sandboxed.assert_not_called()


# =============================================================================
# PROJECT DIRECTORY + PROJECT NAME TESTS
# =============================================================================


class TestProjectNameAndDirectory:
    """Test project_name extraction from CodePlanSignature and directory wrapping."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_code_plan_returns_project_name(self):
        """plan_files() returns sanitized project_name."""
        mock_plan_result = MagicMock()
        mock_plan_result.files_json = json.dumps(
            [
                {"file_path": "app.py", "description": "Main", "language": "python"},
                {"file_path": "models.py", "description": "Models", "language": "python"},
                {"file_path": "tests/test_app.py", "description": "Tests", "language": "python"},
            ]
        )
        mock_plan_result.architecture_notes = "REST API"
        mock_plan_result.project_name = "Stock Trading Platform"

        with patch(
            "Jotty.core.execution.swarms.pilot_swarm.agents.BaseOlympiadAgent.__init__",
            return_value=None,
        ):
            from Jotty.core.execution.swarms.pilot_swarm.agents import PilotCoderAgent

            agent = PilotCoderAgent.__new__(PilotCoderAgent)
            agent._lm = MagicMock()
            agent._file_writer_lm = None
            agent.model = "haiku"
            agent.use_fast_predict = True
            agent.llm_timeout = 90
            agent._planner = MagicMock(return_value=mock_plan_result)
            agent._broadcast = MagicMock()

            result = await agent.plan_files(task="Create a trading platform")

        assert result["project_name"] == "stock-trading-platform"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_project_name_sanitization(self):
        """Special characters, spaces, and long names are sanitized."""
        import re

        test_cases = [
            ("My Cool Project!!", "my-cool-project"),
            ("UPPERCASE NAME", "uppercase-name"),
            ("has spaces and @#$ chars", "has-spaces-and--chars"),
            ("a" * 60, "a" * 40),  # truncated to 40
            ("", "project"),  # empty → fallback
        ]

        for raw, expected in test_cases:
            sanitized = re.sub(r"[^a-z0-9-]", "", raw.lower().replace(" ", "-"))[:40]
            if not sanitized:
                sanitized = "project"
            assert sanitized == expected, f"Failed for {raw!r}: got {sanitized!r}"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_code_creates_project_dir(self, tmp_path):
        """Files are written under project_dir, not CWD."""
        from Jotty.core.execution.swarms.pilot_swarm.swarm import PilotSwarm

        swarm = PilotSwarm.__new__(PilotSwarm)
        config = PilotConfig(allow_file_write=True, project_base_dir=str(tmp_path))
        subtask = Subtask(id="s1", type=SubtaskType.CODE, description="Create API")

        file_plan = [
            {"file_path": "main.py", "description": "Entry", "language": "python"},
            {"file_path": "models.py", "description": "Models", "language": "python"},
            {"file_path": "utils.py", "description": "Utils", "language": "python"},
        ]

        mock_coder = MagicMock()
        mock_coder.plan_files = AsyncMock(
            return_value={
                "files": file_plan,
                "architecture_notes": "Simple project",
                "project_name": "my-api",
            }
        )

        async def mock_write(file_path, description, project_context=""):
            return {
                "file_path": file_path,
                "content": f"# {file_path}\n",
                "explanation": f"Created {file_path}",
            }

        mock_coder.write_single_file = mock_write
        swarm._coder = mock_coder
        swarm._sandbox = None

        result = await swarm._execute_code(subtask, "", config)

        # Verify project_dir in result
        assert result["project_dir"] == str(tmp_path / "my-api")

        # Verify files are under project dir
        for op in result["file_operations"]:
            assert op["file_path"].startswith(str(tmp_path / "my-api"))

        # Verify files were written to disk
        assert (tmp_path / "my-api" / "main.py").exists()
        assert (tmp_path / "my-api" / "models.py").exists()
        assert (tmp_path / "my-api" / "utils.py").exists()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_code_prefixes_relative_paths(self):
        """Relative paths get project_dir prefix."""
        from Jotty.core.execution.swarms.pilot_swarm.swarm import PilotSwarm

        swarm = PilotSwarm.__new__(PilotSwarm)
        config = PilotConfig(allow_file_write=False, project_base_dir="/tmp/test_proj")
        subtask = Subtask(id="s1", type=SubtaskType.CODE, description="Create API")

        file_plan = [
            {"file_path": "src/app.py", "description": "App", "language": "python"},
            {"file_path": "src/config.py", "description": "Config", "language": "python"},
            {"file_path": "requirements.txt", "description": "Deps", "language": "text"},
        ]

        mock_coder = MagicMock()
        mock_coder.plan_files = AsyncMock(
            return_value={
                "files": file_plan,
                "architecture_notes": "Notes",
                "project_name": "test-app",
            }
        )

        async def mock_write(file_path, description, project_context=""):
            return {
                "file_path": file_path,
                "content": f"# content\n",
                "explanation": f"Created",
            }

        mock_coder.write_single_file = mock_write
        swarm._coder = mock_coder
        swarm._sandbox = None

        result = await swarm._execute_code(subtask, "", config)

        paths = [op["file_path"] for op in result["file_operations"]]
        assert paths == [
            "/tmp/test_proj/test-app/src/app.py",
            "/tmp/test_proj/test-app/src/config.py",
            "/tmp/test_proj/test-app/requirements.txt",
        ]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_code_preserves_absolute_paths(self):
        """Absolute paths are left unchanged."""
        from Jotty.core.execution.swarms.pilot_swarm.swarm import PilotSwarm

        swarm = PilotSwarm.__new__(PilotSwarm)
        config = PilotConfig(allow_file_write=False)
        subtask = Subtask(id="s1", type=SubtaskType.CODE, description="Create files")

        file_plan = [
            {"file_path": "/absolute/path/app.py", "description": "App", "language": "python"},
            {"file_path": "relative.py", "description": "Rel", "language": "python"},
            {"file_path": "/absolute/other.py", "description": "Other", "language": "python"},
        ]

        mock_coder = MagicMock()
        mock_coder.plan_files = AsyncMock(
            return_value={
                "files": file_plan,
                "architecture_notes": "Mixed paths",
                "project_name": "mixed",
            }
        )

        async def mock_write(file_path, description, project_context=""):
            return {"file_path": file_path, "content": "# x\n", "explanation": "Created"}

        mock_coder.write_single_file = mock_write
        swarm._coder = mock_coder
        swarm._sandbox = None

        result = await swarm._execute_code(subtask, "", config)

        paths = [op["file_path"] for op in result["file_operations"]]
        # Absolute paths kept as-is, relative prefixed
        assert paths[0] == "/absolute/path/app.py"
        assert paths[1].startswith("/tmp/pilot_projects/mixed/")
        assert paths[2] == "/absolute/other.py"

    @pytest.mark.unit
    def test_code_plan_signature_has_project_name_field(self):
        """CodePlanSignature has project_name output field."""
        from Jotty.core.execution.swarms.pilot_swarm.signatures import CodePlanSignature

        assert "project_name" in CodePlanSignature.output_fields


# =============================================================================
# AUTO-TEST TESTS
# =============================================================================


class TestAutoTest:
    """Test _auto_test_project() method."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_auto_test_skipped_no_requirements_no_tests(self, tmp_path):
        """Skipped when no requirements.txt or test files exist."""
        from Jotty.core.execution.swarms.pilot_swarm.swarm import PilotSwarm

        swarm = PilotSwarm.__new__(PilotSwarm)
        project_dir = tmp_path / "empty-project"
        project_dir.mkdir()

        result = await swarm._auto_test_project(project_dir, [])

        assert result["skipped"] is True
        assert "No requirements.txt" in result["reason"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_auto_test_creates_venv_and_runs(self, tmp_path):
        """Mock subprocess to verify venv creation, pip install, and pytest flow."""
        from Jotty.core.execution.swarms.pilot_swarm.swarm import PilotSwarm

        swarm = PilotSwarm.__new__(PilotSwarm)
        project_dir = tmp_path / "test-project"
        project_dir.mkdir()

        # Create requirements.txt and test file
        (project_dir / "requirements.txt").write_text("requests\n")
        tests_dir = project_dir / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_app.py").write_text("def test_one(): assert True\n")

        call_log = []

        def mock_subprocess_run(cmd, **kwargs):
            call_log.append(cmd)
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "1 passed\n"
            mock_result.stderr = ""
            return mock_result

        with patch("subprocess.run", side_effect=mock_subprocess_run):
            result = await swarm._auto_test_project(project_dir, [])

        assert result["success"] is True
        assert result["phase"] == "test"
        # Should have 3 calls: venv, pip install, pytest
        assert len(call_log) == 3
        assert "venv" in str(call_log[0])
        assert "pip" in str(call_log[1])
        assert "pytest" in str(call_log[2])

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_auto_test_pip_failure(self, tmp_path):
        """Returns failure when pip install fails."""
        from Jotty.core.execution.swarms.pilot_swarm.swarm import PilotSwarm

        swarm = PilotSwarm.__new__(PilotSwarm)
        project_dir = tmp_path / "bad-deps"
        project_dir.mkdir()
        (project_dir / "requirements.txt").write_text("nonexistent-package-xyz\n")

        call_count = [0]

        def mock_subprocess_run(cmd, **kwargs):
            call_count[0] += 1
            mock_result = MagicMock()
            if call_count[0] == 1:
                # venv creation succeeds
                mock_result.returncode = 0
                mock_result.stderr = ""
            else:
                # pip install fails
                mock_result.returncode = 1
                mock_result.stderr = "No matching distribution found"
            mock_result.stdout = ""
            return mock_result

        with patch("subprocess.run", side_effect=mock_subprocess_run):
            result = await swarm._auto_test_project(project_dir, [])

        assert result["success"] is False
        assert result["phase"] == "install"
        assert "No matching distribution" in result["error"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_auto_test_disabled_via_config(self):
        """auto_test=False skips testing entirely."""
        from Jotty.core.execution.swarms.pilot_swarm.swarm import PilotSwarm

        swarm = PilotSwarm.__new__(PilotSwarm)
        config = PilotConfig(
            allow_file_write=True,
            allow_terminal=True,
            auto_test=False,
        )
        subtask = Subtask(id="s1", type=SubtaskType.CODE, description="Create project")

        file_plan = [
            {"file_path": "a.py", "description": "A", "language": "python"},
            {"file_path": "b.py", "description": "B", "language": "python"},
            {"file_path": "c.py", "description": "C", "language": "python"},
        ]

        mock_coder = MagicMock()
        mock_coder.plan_files = AsyncMock(
            return_value={
                "files": file_plan,
                "architecture_notes": "Notes",
                "project_name": "test-proj",
            }
        )

        async def mock_write(file_path, description, project_context=""):
            return {"file_path": file_path, "content": "# x\n", "explanation": "Created"}

        mock_coder.write_single_file = mock_write
        swarm._coder = mock_coder
        swarm._sandbox = None

        result = await swarm._execute_code(subtask, "", config)

        # test_result should be empty (auto_test=False)
        assert result.get("test_result") == {}

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_auto_test_install_only(self, tmp_path):
        """Requirements installed but no test files → install_only phase."""
        from Jotty.core.execution.swarms.pilot_swarm.swarm import PilotSwarm

        swarm = PilotSwarm.__new__(PilotSwarm)
        project_dir = tmp_path / "no-tests"
        project_dir.mkdir()
        (project_dir / "requirements.txt").write_text("flask\n")
        # No test files

        def mock_subprocess_run(cmd, **kwargs):
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = ""
            mock_result.stderr = ""
            return mock_result

        with patch("subprocess.run", side_effect=mock_subprocess_run):
            result = await swarm._auto_test_project(project_dir, [])

        assert result["success"] is True
        assert result["phase"] == "install_only"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_auto_test_venv_failure(self, tmp_path):
        """Returns failure when venv creation fails."""
        from Jotty.core.execution.swarms.pilot_swarm.swarm import PilotSwarm

        swarm = PilotSwarm.__new__(PilotSwarm)
        project_dir = tmp_path / "venv-fail"
        project_dir.mkdir()
        (project_dir / "requirements.txt").write_text("flask\n")

        def mock_subprocess_run(cmd, **kwargs):
            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_result.stderr = "ensurepip not available"
            mock_result.stdout = ""
            return mock_result

        with patch("subprocess.run", side_effect=mock_subprocess_run):
            result = await swarm._auto_test_project(project_dir, [])

        assert result["success"] is False
        assert "Venv creation failed" in result["error"]


# =============================================================================
# CONFIG AUTO-TEST FIELD TESTS
# =============================================================================


class TestConfigAutoTestFields:
    """Test auto_test and project_base_dir config fields."""

    @pytest.mark.unit
    def test_config_has_auto_test_field(self):
        config = PilotConfig()
        assert config.auto_test is True

    @pytest.mark.unit
    def test_config_has_project_base_dir_field(self):
        config = PilotConfig()
        assert config.project_base_dir == ""

    @pytest.mark.unit
    def test_config_auto_test_can_be_disabled(self):
        config = PilotConfig(auto_test=False)
        assert config.auto_test is False

    @pytest.mark.unit
    def test_config_project_base_dir_custom(self):
        config = PilotConfig(project_base_dir="/custom/path")
        assert config.project_base_dir == "/custom/path"
