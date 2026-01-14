"""
Unit tests for AgentConfig.

Tests agent configuration validation, parameter mappings, and tool setup.
"""
import pytest
from unittest.mock import Mock


class TestAgentConfigInitialization:
    """Test AgentConfig initialization."""

    @pytest.mark.unit
    def test_minimal_agent_config(self, mock_dspy_agent):
        """Test creating AgentConfig with minimal parameters."""
        from core.agent_config import AgentConfig

        config = AgentConfig(
            name="MinimalAgent",
            agent=mock_dspy_agent,
            architect_prompts=["architect.md"],
            auditor_prompts=["auditor.md"],
        )

        assert config.name == "MinimalAgent"
        assert config.agent == mock_dspy_agent
        assert config.architect_prompts == ["architect.md"]
        assert config.auditor_prompts == ["auditor.md"]

    @pytest.mark.unit
    def test_full_agent_config(self, mock_dspy_agent):
        """Test creating AgentConfig with all parameters."""
        from core.agent_config import AgentConfig

        config = AgentConfig(
            name="FullAgent",
            agent=mock_dspy_agent,
            architect_prompts=["architect.md"],
            auditor_prompts=["auditor.md"],
            parameter_mappings={"query": "context.query"},
            outputs=["sql_query", "result"],
            provides=["sql_query"],
            dependencies=["PreprocessorAgent"],
            capabilities=["sql_generation", "data_analysis"],
            is_executor=True,
            enable_architect=True,
            enable_auditor=True,
        )

        assert config.name == "FullAgent"
        assert config.parameter_mappings == {"query": "context.query"}
        assert config.outputs == ["sql_query", "result"]
        assert config.provides == ["sql_query"]
        assert config.dependencies == ["PreprocessorAgent"]
        assert config.capabilities == ["sql_generation", "data_analysis"]
        assert config.is_executor is True
        assert config.enable_architect is True
        assert config.enable_auditor is True


class TestParameterMappings:
    """Test parameter mapping validation."""

    @pytest.mark.unit
    def test_context_parameter_mapping(self, mock_dspy_agent):
        """Test parameter mapping from context."""
        from core.agent_config import AgentConfig

        config = AgentConfig(
            name="ContextAgent",
            agent=mock_dspy_agent,
            architect_prompts=["test.md"],
            auditor_prompts=["test.md"],
            parameter_mappings={
                "query": "context.query",
                "date": "context.current_date",
            }
        )

        assert "query" in config.parameter_mappings
        assert "date" in config.parameter_mappings
        assert config.parameter_mappings["query"] == "context.query"

    @pytest.mark.unit
    def test_agent_output_parameter_mapping(self, mock_dspy_agent):
        """Test parameter mapping from other agent outputs."""
        from core.agent_config import AgentConfig

        config = AgentConfig(
            name="DependentAgent",
            agent=mock_dspy_agent,
            architect_prompts=["test.md"],
            auditor_prompts=["test.md"],
            parameter_mappings={
                "preprocessed_data": "PreprocessorAgent.data",
                "metadata": "MetadataAgent.metadata",
            },
            dependencies=["PreprocessorAgent", "MetadataAgent"]
        )

        assert "preprocessed_data" in config.parameter_mappings
        assert "metadata" in config.parameter_mappings


class TestToolConfiguration:
    """Test tool configuration."""

    @pytest.mark.unit
    def test_architect_tools(self, mock_dspy_agent):
        """Test adding Architect tools."""
        from core.agent_config import AgentConfig

        mock_tool1 = Mock(name="tool1")
        mock_tool2 = Mock(name="tool2")

        config = AgentConfig(
            name="ToolAgent",
            agent=mock_dspy_agent,
            architect_prompts=["test.md"],
            auditor_prompts=["test.md"],
            architect_tools=[mock_tool1, mock_tool2],
        )

        assert len(config.architect_tools) == 2
        assert mock_tool1 in config.architect_tools
        assert mock_tool2 in config.architect_tools

    @pytest.mark.unit
    def test_auditor_tools(self, mock_dspy_agent):
        """Test adding Auditor tools."""
        from core.agent_config import AgentConfig

        mock_tool1 = Mock(name="validation_tool")
        mock_tool2 = Mock(name="quality_check_tool")

        config = AgentConfig(
            name="AuditorAgent",
            agent=mock_dspy_agent,
            architect_prompts=["test.md"],
            auditor_prompts=["test.md"],
            auditor_tools=[mock_tool1, mock_tool2],
        )

        assert len(config.auditor_tools) == 2
        assert mock_tool1 in config.auditor_tools
        assert mock_tool2 in config.auditor_tools


class TestValidationControl:
    """Test validation control flags."""

    @pytest.mark.unit
    def test_disable_validation(self, mock_dspy_agent):
        """Test disabling validation."""
        from core.agent_config import AgentConfig

        config = AgentConfig(
            name="NoValidationAgent",
            agent=mock_dspy_agent,
            architect_prompts=["test.md"],
            auditor_prompts=["test.md"],
            enable_architect=False,
            enable_auditor=False,
        )

        assert config.enable_architect is False
        assert config.enable_auditor is False

    @pytest.mark.unit
    def test_enable_architect_only(self, mock_dspy_agent):
        """Test enabling only Architect."""
        from core.agent_config import AgentConfig

        config = AgentConfig(
            name="ArchitectOnlyAgent",
            agent=mock_dspy_agent,
            architect_prompts=["test.md"],
            auditor_prompts=["test.md"],
            enable_architect=True,
            enable_auditor=False,
        )

        assert config.enable_architect is True
        assert config.enable_auditor is False

    @pytest.mark.unit
    def test_validation_modes(self, mock_dspy_agent):
        """Test different validation modes."""
        from core.agent_config import AgentConfig

        for mode in ["quick", "standard", "thorough"]:
            config = AgentConfig(
                name=f"{mode}Agent",
                agent=mock_dspy_agent,
                architect_prompts=["test.md"],
                auditor_prompts=["test.md"],
                validation_mode=mode,
            )

            assert config.validation_mode == mode


class TestDependencies:
    """Test dependency configuration."""

    @pytest.mark.unit
    def test_single_dependency(self, mock_dspy_agent):
        """Test agent with single dependency."""
        from core.agent_config import AgentConfig

        config = AgentConfig(
            name="DependentAgent",
            agent=mock_dspy_agent,
            architect_prompts=["test.md"],
            auditor_prompts=["test.md"],
            dependencies=["BaseAgent"],
        )

        assert len(config.dependencies) == 1
        assert "BaseAgent" in config.dependencies

    @pytest.mark.unit
    def test_multiple_dependencies(self, mock_dspy_agent):
        """Test agent with multiple dependencies."""
        from core.agent_config import AgentConfig

        config = AgentConfig(
            name="MultiDependentAgent",
            agent=mock_dspy_agent,
            architect_prompts=["test.md"],
            auditor_prompts=["test.md"],
            dependencies=["Agent1", "Agent2", "Agent3"],
        )

        assert len(config.dependencies) == 3
        assert "Agent1" in config.dependencies
        assert "Agent2" in config.dependencies
        assert "Agent3" in config.dependencies

    @pytest.mark.unit
    def test_no_dependencies(self, mock_dspy_agent):
        """Test independent agent."""
        from core.agent_config import AgentConfig

        config = AgentConfig(
            name="IndependentAgent",
            agent=mock_dspy_agent,
            architect_prompts=["test.md"],
            auditor_prompts=["test.md"],
        )

        assert config.dependencies is None or len(config.dependencies) == 0


class TestExecutorFlag:
    """Test executor flag for execution agents."""

    @pytest.mark.unit
    def test_executor_agent(self, mock_dspy_agent):
        """Test marking agent as executor."""
        from core.agent_config import AgentConfig

        config = AgentConfig(
            name="ExecutorAgent",
            agent=mock_dspy_agent,
            architect_prompts=["test.md"],
            auditor_prompts=["test.md"],
            is_executor=True,
        )

        assert config.is_executor is True

    @pytest.mark.unit
    def test_non_executor_agent(self, mock_dspy_agent):
        """Test non-executor agent."""
        from core.agent_config import AgentConfig

        config = AgentConfig(
            name="PlannerAgent",
            agent=mock_dspy_agent,
            architect_prompts=["test.md"],
            auditor_prompts=["test.md"],
            is_executor=False,
        )

        assert config.is_executor is False
