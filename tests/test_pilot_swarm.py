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
import logging
import pytest
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from dataclasses import dataclass

# Import types
from Jotty.core.swarms.pilot_swarm.types import (
    SubtaskType, SubtaskStatus, Subtask,
    PilotConfig, PilotResult, AVAILABLE_SWARMS,
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
        with patch('Jotty.core.swarms.pilot_swarm.agents.BaseOlympiadAgent._get_lm'):
            from Jotty.core.swarms.pilot_swarm.agents import PilotPlannerAgent
            agent = PilotPlannerAgent.__new__(PilotPlannerAgent)
            agent.model = "sonnet"
            agent.use_fast_predict = True
            agent.llm_timeout = 90
            agent._lm = MagicMock()
            agent.learned_context = ""
            agent._bus = None

            mock_result = _mock_dspy_result(
                subtasks_json=json.dumps([
                    {"id": "s1", "type": "search", "description": "Research Python frameworks",
                     "tool_hint": "web-search", "depends_on": []},
                    {"id": "s2", "type": "code", "description": "Create comparison table",
                     "tool_hint": "", "depends_on": ["s1"]},
                ]),
                reasoning="First research, then synthesize into code.",
            )
            agent._planner = MagicMock(return_value=mock_result)

            result = await agent.plan(goal="Compare Python web frameworks")

            assert 'subtasks' in result
            assert 'reasoning' in result
            assert len(result['subtasks']) == 2
            assert result['subtasks'][0]['type'] == 'search'
            assert result['subtasks'][1]['depends_on'] == ['s1']


class TestPilotSearchAgent:
    """Test PilotSearchAgent with mocked DSPy."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_search_returns_findings(self):
        with patch('Jotty.core.swarms.pilot_swarm.agents.BaseOlympiadAgent._get_lm'):
            from Jotty.core.swarms.pilot_swarm.agents import PilotSearchAgent
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

            assert 'queries' in result
            assert 'synthesis' in result
            assert 'key_findings' in result
            assert len(result['queries']) == 2
            assert len(result['key_findings']) == 3


class TestPilotCoderAgent:
    """Test PilotCoderAgent with mocked DSPy."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_code_returns_file_operations(self):
        with patch('Jotty.core.swarms.pilot_swarm.agents.BaseOlympiadAgent._get_lm'):
            from Jotty.core.swarms.pilot_swarm.agents import PilotCoderAgent
            agent = PilotCoderAgent.__new__(PilotCoderAgent)
            agent.model = "sonnet"
            agent.use_fast_predict = True
            agent.llm_timeout = 90
            agent._lm = MagicMock()
            agent.learned_context = ""
            agent._bus = None

            mock_result = _mock_dspy_result(
                file_operations_json=json.dumps([
                    {"file_path": "/tmp/test_app.py", "action": "create",
                     "content": "from fastapi import FastAPI\napp = FastAPI()",
                     "description": "Main FastAPI application"},
                ]),
                explanation="Created a FastAPI app with a basic setup.",
            )
            agent._coder = MagicMock(return_value=mock_result)

            result = await agent.code(task="Create a FastAPI app")

            assert 'file_operations' in result
            assert 'explanation' in result
            assert len(result['file_operations']) == 1
            assert result['file_operations'][0]['file_path'] == '/tmp/test_app.py'


class TestPilotTerminalAgent:
    """Test PilotTerminalAgent with mocked DSPy."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execute_returns_commands(self):
        with patch('Jotty.core.swarms.pilot_swarm.agents.BaseOlympiadAgent._get_lm'):
            from Jotty.core.swarms.pilot_swarm.agents import PilotTerminalAgent
            agent = PilotTerminalAgent.__new__(PilotTerminalAgent)
            agent.model = "haiku"
            agent.use_fast_predict = True
            agent.llm_timeout = 90
            agent._lm = MagicMock()
            agent.learned_context = ""
            agent._bus = None

            mock_result = _mock_dspy_result(
                commands_json=json.dumps([
                    {"command": "pip list", "purpose": "List installed packages", "safe": True},
                    {"command": "rm -rf /", "purpose": "Delete everything", "safe": False},
                ]),
                safety_assessment="pip list is safe. rm -rf / is extremely dangerous.",
            )
            agent._terminal = MagicMock(return_value=mock_result)

            result = await agent.execute(task="Check installed packages")

            assert 'commands' in result
            assert 'safety_assessment' in result
            assert len(result['commands']) == 2
            assert result['commands'][0]['safe'] is True
            assert result['commands'][1]['safe'] is False


class TestPilotSkillWriterAgent:
    """Test PilotSkillWriterAgent with mocked DSPy."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_write_skill_returns_files(self):
        with patch('Jotty.core.swarms.pilot_swarm.agents.BaseOlympiadAgent._get_lm'):
            from Jotty.core.swarms.pilot_swarm.agents import PilotSkillWriterAgent
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

            assert 'skill_yaml' in result
            assert 'tools_py' in result
            assert 'usage_example' in result
            assert 'csv-to-json' in result['skill_yaml']
            assert 'def convert_csv_tool' in result['tools_py']


class TestPilotValidatorAgent:
    """Test PilotValidatorAgent with mocked DSPy."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_validate_success(self):
        with patch('Jotty.core.swarms.pilot_swarm.agents.BaseOlympiadAgent._get_lm'):
            from Jotty.core.swarms.pilot_swarm.agents import PilotValidatorAgent
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

            assert result['success'] is True
            assert 'assessment' in result
            assert result['remaining_gaps'] == []

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_validate_failure(self):
        with patch('Jotty.core.swarms.pilot_swarm.agents.BaseOlympiadAgent._get_lm'):
            from Jotty.core.swarms.pilot_swarm.agents import PilotValidatorAgent
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

            assert result['success'] is False
            assert len(result['remaining_gaps']) == 2


# =============================================================================
# SWARM HELPER TESTS
# =============================================================================

class TestSwarmHelpers:
    """Test swarm helper methods."""

    @pytest.mark.unit
    def test_build_context_empty(self):
        from Jotty.core.swarms.pilot_swarm.swarm import PilotSwarm
        swarm = PilotSwarm.__new__(PilotSwarm)
        ctx = swarm._build_context({})
        assert ctx == "No previous results."

    @pytest.mark.unit
    def test_build_context_with_results(self):
        from Jotty.core.swarms.pilot_swarm.swarm import PilotSwarm
        swarm = PilotSwarm.__new__(PilotSwarm)
        ctx = swarm._build_context({
            's1': {'synthesis': 'FastAPI is fast', 'key_findings': ['async', 'fast']},
            's2': {'explanation': 'Created app.py with endpoints'},
        })
        assert '[s1]' in ctx
        assert '[s2]' in ctx
        assert 'FastAPI' in ctx

    @pytest.mark.unit
    def test_build_results_summary(self):
        from Jotty.core.swarms.pilot_swarm.swarm import PilotSwarm
        swarm = PilotSwarm.__new__(PilotSwarm)

        subtasks = [
            Subtask(id="s1", type=SubtaskType.SEARCH, description="Research frameworks",
                    status=SubtaskStatus.COMPLETED),
            Subtask(id="s2", type=SubtaskType.CODE, description="Write code",
                    status=SubtaskStatus.FAILED),
        ]

        summary = swarm._build_results_summary(subtasks, {
            's1': {'synthesis': 'Found 5 frameworks'},
            's2': {'error': 'LLM timeout'},
        })

        assert '[s1] search (completed)' in summary
        assert '[s2] code (failed)' in summary
        assert 'ERROR: LLM timeout' in summary

    @pytest.mark.unit
    def test_build_results_summary_with_skill(self):
        from Jotty.core.swarms.pilot_swarm.swarm import PilotSwarm
        swarm = PilotSwarm.__new__(PilotSwarm)

        subtasks = [
            Subtask(id="s1", type=SubtaskType.CREATE_SKILL, description="Create CSV skill",
                    status=SubtaskStatus.COMPLETED),
        ]

        summary = swarm._build_results_summary(subtasks, {
            's1': {'skill_name': 'csv-converter'},
        })

        assert 'Created skill: csv-converter' in summary


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
        st = Subtask(id="s2", type=SubtaskType.CODE, description="Code",
                     depends_on=["s1"])
        completed_ids = set()
        unmet = [d for d in st.depends_on if d not in completed_ids]
        assert unmet == ["s1"]

    @pytest.mark.unit
    def test_subtask_with_met_dependency(self):
        """Subtasks with met dependencies should be executable."""
        st = Subtask(id="s2", type=SubtaskType.CODE, description="Code",
                     depends_on=["s1"])
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
        from Jotty.core.swarms.pilot_swarm import PilotSwarm
        from Jotty.core.swarms.base_swarm import SwarmRegistry
        swarm_class = SwarmRegistry.get("pilot")
        assert swarm_class is PilotSwarm

    @pytest.mark.unit
    def test_lazy_import_from_core_swarms(self):
        from Jotty.core.swarms import PilotSwarm
        assert PilotSwarm is not None

    @pytest.mark.unit
    def test_lazy_import_pilot(self):
        from Jotty.core.swarms import pilot
        assert callable(pilot)

    @pytest.mark.unit
    def test_lazy_import_types(self):
        from Jotty.core.swarms import SubtaskType, SubtaskStatus
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
        from Jotty.core.swarms.pilot_swarm.swarm import PilotSwarm
        swarm = PilotSwarm()
        assert swarm.config.name == "PilotSwarm"
        assert swarm.config.domain == "pilot"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_swarm_with_custom_config(self):
        from Jotty.core.swarms.pilot_swarm.swarm import PilotSwarm
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
        from Jotty.core.swarms.pilot_swarm import pilot
        assert callable(pilot)

    @pytest.mark.unit
    def test_pilot_sync_is_callable(self):
        from Jotty.core.swarms.pilot_swarm import pilot_sync
        assert callable(pilot_sync)


# =============================================================================
# SIGNATURES TESTS
# =============================================================================

class TestSignatures:
    """Test DSPy signatures can be instantiated."""

    @pytest.mark.unit
    def test_all_signatures_importable(self):
        from Jotty.core.swarms.pilot_swarm.signatures import (
            PlannerSignature, SearchSignature, CoderSignature,
            TerminalSignature, SkillWriterSignature, ValidatorSignature,
        )
        import dspy
        for sig in [PlannerSignature, SearchSignature, CoderSignature,
                     TerminalSignature, SkillWriterSignature, ValidatorSignature]:
            assert issubclass(sig, dspy.Signature)

    @pytest.mark.unit
    def test_signature_count(self):
        from Jotty.core.swarms.pilot_swarm import signatures
        all_sigs = signatures.__all__
        assert len(all_sigs) == 6


# =============================================================================
# AGENTS IMPORT TESTS
# =============================================================================

class TestAgentsImport:
    """Test all agents are importable."""

    @pytest.mark.unit
    def test_all_agents_importable(self):
        from Jotty.core.swarms.pilot_swarm.agents import (
            PilotPlannerAgent, PilotSearchAgent, PilotCoderAgent,
            PilotTerminalAgent, PilotSkillWriterAgent, PilotValidatorAgent,
        )
        from Jotty.core.swarms.olympiad_learning_swarm.agents import BaseOlympiadAgent
        for agent_cls in [PilotPlannerAgent, PilotSearchAgent, PilotCoderAgent,
                          PilotTerminalAgent, PilotSkillWriterAgent, PilotValidatorAgent]:
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

        safe_commands = [c for c in commands if c.get('safe', False)]
        unsafe_commands = [c for c in commands if not c.get('safe', False)]

        assert len(safe_commands) == 2
        assert len(unsafe_commands) == 1
        assert unsafe_commands[0]['command'] == 'rm -rf /tmp/test'

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
        yaml_content = "name: test-skill\ndescription: A test skill\nversion: 1.0.0\ntools:\n  - test_tool"
        assert "name:" in yaml_content
        assert "description:" in yaml_content
        assert "version:" in yaml_content
        assert "tools:" in yaml_content

    @pytest.mark.unit
    def test_skill_name_sanitization(self):
        """Verify skill name is properly sanitized."""
        raw = "My Cool Skill!!"
        sanitized = ''.join(c for c in raw.lower().replace(' ', '-') if c.isalnum() or c == '-')[:30]
        assert sanitized == "my-cool-skill"
        assert len(sanitized) <= 30

    @pytest.mark.unit
    def test_skill_name_truncation(self):
        """Verify long skill names are truncated."""
        raw = "this-is-a-very-long-skill-name-that-exceeds-thirty-characters"
        sanitized = ''.join(c for c in raw.lower().replace(' ', '-') if c.isalnum() or c == '-')[:30]
        assert len(sanitized) <= 30
