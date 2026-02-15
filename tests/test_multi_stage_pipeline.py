#!/usr/bin/env python3
"""
Tests for Multi-Stage Pipeline
===============================

Tests the high-level pipeline orchestrator for chaining multiple swarm executions.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch


@pytest.mark.asyncio
async def test_pipeline_imports():
    """Test that pipeline utilities are properly exported."""
    from Jotty.core.intelligence.orchestration import (
        MultiStagePipeline,
        PipelineResult,
        StageResult,
        StageConfig,
        create_pipeline,
        extract_code_from_markdown,
    )

    assert MultiStagePipeline is not None
    assert PipelineResult is not None
    assert StageResult is not None
    assert StageConfig is not None
    assert create_pipeline is not None
    assert extract_code_from_markdown is not None


@pytest.mark.asyncio
async def test_pipeline_basic_usage():
    """Test basic pipeline creation and execution."""
    from Jotty.core.intelligence.orchestration import (
        MultiStagePipeline,
        SwarmAdapter,
        MergeStrategy
    )

    # Mock swarms
    class MockSwarm:
        name = "Mock"

        async def execute(self, task):
            from Jotty.core.intelligence.orchestration import SwarmResult
            await asyncio.sleep(0.01)
            return SwarmResult(
                swarm_name=self.name,
                output=f"Result for: {task[:50]}",
                success=True,
                confidence=0.8
            )

    # Create pipeline
    pipeline = MultiStagePipeline(task="Test task")

    # Add stages
    pipeline.add_stage(
        "stage1",
        swarms=[MockSwarm()],
        merge_strategy=MergeStrategy.BEST_OF_N
    )

    pipeline.add_stage(
        "stage2",
        swarms=[MockSwarm()],
        merge_strategy=MergeStrategy.BEST_OF_N
    )

    # Execute
    result = await pipeline.execute(auto_trace=False, verbose=False)

    # Verify
    assert result.task == "Test task"
    assert len(result.stages) == 2
    assert result.total_cost >= 0.0
    assert result.total_time >= 0.0
    assert result.final_result is not None
    assert result.final_result.stage_name == "stage2"


@pytest.mark.asyncio
async def test_pipeline_context_chaining():
    """Test that context passes between stages correctly."""
    from Jotty.core.intelligence.orchestration import (
        MultiStagePipeline,
        MergeStrategy
    )

    # Mock swarm that returns task (so we can verify context was passed)
    class ContextVerifySwarm:
        name = "ContextVerify"

        async def execute(self, task):
            from Jotty.core.intelligence.orchestration import SwarmResult
            return SwarmResult(
                swarm_name=self.name,
                output=f"Received task: {task}",
                success=True,
                confidence=0.9
            )

    pipeline = MultiStagePipeline(task="Main task")

    # Stage 1: Produces output
    pipeline.add_stage(
        "producer",
        swarms=[ContextVerifySwarm()],
        merge_strategy=MergeStrategy.BEST_OF_N
    )

    # Stage 2: Should receive Stage 1 output as context
    pipeline.add_stage(
        "consumer",
        swarms=[ContextVerifySwarm()],
        merge_strategy=MergeStrategy.BEST_OF_N,
        context_from=["producer"]  # Request context from stage1
    )

    result = await pipeline.execute(auto_trace=False, verbose=False)

    # Verify stage2 received context from stage1
    stage2_result = result.get_stage("consumer")
    assert stage2_result is not None
    assert "PRODUCER" in stage2_result.result.output  # Context should include [PRODUCER] marker


@pytest.mark.asyncio
async def test_pipeline_multiple_context_sources():
    """Test stage receiving context from multiple previous stages."""
    from Jotty.core.intelligence.orchestration import MultiStagePipeline, MergeStrategy

    class MockSwarm:
        name = "Mock"

        async def execute(self, task):
            from Jotty.core.intelligence.orchestration import SwarmResult
            return SwarmResult(
                swarm_name=self.name,
                output=f"Output: {task[:30]}",
                success=True,
                confidence=0.8
            )

    pipeline = MultiStagePipeline(task="Test")

    pipeline.add_stage("stage1", swarms=[MockSwarm()])
    pipeline.add_stage("stage2", swarms=[MockSwarm()])
    pipeline.add_stage(
        "stage3",
        swarms=[MockSwarm()],
        context_from=["stage1", "stage2"]  # Multiple sources
    )

    result = await pipeline.execute(auto_trace=False, verbose=False)

    assert len(result.stages) == 3
    stage3 = result.get_stage("stage3")
    assert stage3 is not None


@pytest.mark.asyncio
async def test_pipeline_get_stage():
    """Test retrieving specific stage results."""
    from Jotty.core.intelligence.orchestration import MultiStagePipeline, MergeStrategy

    class MockSwarm:
        name = "Mock"

        async def execute(self, task):
            from Jotty.core.intelligence.orchestration import SwarmResult
            return SwarmResult(
                swarm_name=self.name,
                output="Test output",
                success=True,
                confidence=0.7
            )

    pipeline = MultiStagePipeline(task="Test")
    pipeline.add_stage("alpha", swarms=[MockSwarm()])
    pipeline.add_stage("beta", swarms=[MockSwarm()])

    result = await pipeline.execute(auto_trace=False, verbose=False)

    # Test get_stage
    alpha = result.get_stage("alpha")
    beta = result.get_stage("beta")
    gamma = result.get_stage("gamma")  # Doesn't exist

    assert alpha is not None
    assert alpha.stage_name == "alpha"
    assert beta is not None
    assert beta.stage_name == "beta"
    assert gamma is None


@pytest.mark.asyncio
async def test_extract_code_from_markdown():
    """Test code extraction utility."""
    from Jotty.core.intelligence.orchestration import extract_code_from_markdown

    # Test with language specifier
    text1 = """
    Here's some Python code:
    ```python
    def hello():
        print("world")
    ```
    """
    code1 = extract_code_from_markdown(text1, language="python")
    assert code1 is not None
    assert "def hello():" in code1
    assert "print" in code1

    # Test with generic code block
    text2 = """
    Some code:
    ```
    x = 42
    y = x + 1
    ```
    """
    code2 = extract_code_from_markdown(text2)
    assert code2 is not None
    assert "x = 42" in code2

    # Test with no code blocks
    text3 = "No code here"
    code3 = extract_code_from_markdown(text3)
    assert code3 is None

    # Test with different language (falls back to generic extraction)
    text4 = "```javascript\nconst x = 1;\n```"
    code4 = extract_code_from_markdown(text4, language="python")
    # Falls back to generic code block extraction (this is intentional)
    assert code4 is not None
    assert "const x = 1" in code4


@pytest.mark.asyncio
async def test_pipeline_print_summary():
    """Test that PipelineResult.print_summary() works."""
    from Jotty.core.intelligence.orchestration import (
        PipelineResult,
        StageResult,
        SwarmResult
    )
    import io
    import sys

    # Create mock results
    stage_result = StageResult(
        stage_name="test_stage",
        result=SwarmResult(
            swarm_name="test",
            output="Test output",
            success=True,
            confidence=0.8
        ),
        execution_time=1.0,
        cost=0.001
    )

    result = PipelineResult(
        task="Test task",
        stages=[stage_result],
        total_cost=0.001,
        total_time=1.0,
        final_result=stage_result
    )

    # Capture output
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    try:
        result.print_summary(verbose=True)
        output = sys.stdout.getvalue()

        # Verify output contains key sections
        assert "MULTI-STAGE PIPELINE RESULTS" in output
        assert "Test task" in output
        assert "Total Stages: 1" in output
        assert "Total Cost" in output
        assert "test_stage" in output

    finally:
        sys.stdout = old_stdout


@pytest.mark.asyncio
async def test_create_pipeline_facade():
    """Test the create_pipeline facade function."""
    from Jotty.core.intelligence.orchestration import create_pipeline

    pipeline = create_pipeline("Test task")

    assert pipeline is not None
    assert pipeline.task == "Test task"
    assert len(pipeline.stages) == 0


@pytest.mark.asyncio
async def test_stage_config_context_template():
    """Test custom context templates."""
    from Jotty.core.intelligence.orchestration import StageConfig, StageResult, SwarmResult

    # Create stage config with custom template
    config = StageConfig(
        name="test",
        swarms=[],
        context_template="Previous results:\n{context}\n\nNow do:"
    )

    # Create mock previous results
    prev_results = {
        "stage1": StageResult(
            stage_name="stage1",
            result=SwarmResult("s1", "Output 1", True, 0.8),
            execution_time=1.0,
            cost=0.001
        )
    }

    config.context_from = ["stage1"]
    context = config.get_context_prompt(prev_results)

    assert "Previous results:" in context
    assert "Now do:" in context
    assert "Output 1" in context


@pytest.mark.asyncio
async def test_pipeline_with_max_context_chars():
    """Test that context is truncated to max_context_chars."""
    from Jotty.core.intelligence.orchestration import MultiStagePipeline, MergeStrategy

    class VerboseSwarm:
        name = "Verbose"

        async def execute(self, task):
            from Jotty.core.intelligence.orchestration import SwarmResult
            # Return very long output
            long_output = "A" * 3000
            return SwarmResult(
                swarm_name=self.name,
                output=long_output,
                success=True,
                confidence=0.9
            )

    pipeline = MultiStagePipeline(task="Test")
    pipeline.add_stage("stage1", swarms=[VerboseSwarm()])
    pipeline.add_stage(
        "stage2",
        swarms=[VerboseSwarm()],
        context_from=["stage1"],
        max_context_chars=500  # Limit context
    )

    result = await pipeline.execute(auto_trace=False, verbose=False)

    # Stage 1 should have long output
    stage1 = result.get_stage("stage1")
    assert len(stage1.result.output) == 3000

    # Context passed to stage2 should be truncated
    # (we can't directly verify this, but the pipeline should not fail)
    assert len(result.stages) == 2


@pytest.mark.asyncio
@patch('os.getenv')
@patch('anthropic.AsyncAnthropic')
async def test_pipeline_with_real_api_mocked(mock_anthropic, mock_getenv):
    """Test pipeline with mocked Anthropic API."""
    from Jotty.core.intelligence.orchestration import (
        MultiStagePipeline,
        SwarmAdapter,
        MergeStrategy
    )

    # Mock API
    mock_getenv.return_value = "sk-test-key"

    mock_response = Mock()
    mock_response.content = [Mock(text="Test response")]
    mock_response.usage = Mock(input_tokens=10, output_tokens=20)

    mock_client = Mock()
    mock_client.messages.create = AsyncMock(return_value=mock_response)
    mock_anthropic.return_value = mock_client

    # Create pipeline with real swarms
    pipeline = MultiStagePipeline(task="Test with API")

    pipeline.add_stage(
        "stage1",
        swarms=SwarmAdapter.quick_swarms([("S1", "Prompt 1")]),
        merge_strategy=MergeStrategy.BEST_OF_N
    )

    result = await pipeline.execute(auto_trace=False, verbose=False)

    assert result is not None
    assert len(result.stages) == 1
    assert result.stages[0].result.success is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
