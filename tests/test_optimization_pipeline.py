"""
Tests for OptimizationPipeline

Tests the generic optimization pipeline with various domains.
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    dspy = None

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.orchestration.optimization_pipeline import (
    OptimizationPipeline,
    OptimizationConfig,
    IterationResult,
    create_optimization_pipeline
)
from core.foundation.agent_config import AgentConfig


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory."""
    output_dir = tmp_path / "test_outputs"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def simple_agents():
    """Create simple test agents."""
    class Agent1(dspy.Module if DSPY_AVAILABLE else object):
        def forward(self, task: str) -> str:
            return f"Agent1: {task}"
    
    class Agent2(dspy.Module if DSPY_AVAILABLE else object):
        def forward(self, input: str) -> str:
            return f"Agent2: {input}"
    
    return [
        AgentConfig(
            name="agent1",
            agent=Agent1(),
            outputs=["result"]
        ),
        AgentConfig(
            name="agent2",
            agent=Agent2(),
            parameter_mappings={"input": "result"}
        )
    ]


@pytest.fixture
def simple_evaluation():
    """Simple evaluation function."""
    async def evaluate(output, gold_standard, task, context):
        output_str = str(output).strip()
        gold_str = str(gold_standard).strip()
        score = 1.0 if output_str == gold_str else 0.0
        return {
            "score": score,
            "status": "CORRECT" if score == 1.0 else "INCORRECT"
        }
    return evaluate


@pytest.mark.asyncio
@pytest.mark.unit
async def test_basic_optimization(simple_agents, simple_evaluation, temp_output_dir):
    """Test basic optimization pipeline."""
    config = OptimizationConfig(
        max_iterations=3,
        required_pass_count=1,
        output_path=temp_output_dir
    )
    
    pipeline = OptimizationPipeline(agents=simple_agents, config=config)
    pipeline.config.evaluation_function = simple_evaluation
    
    result = await pipeline.optimize(
        task="Test task",
        context={},
        gold_standard="Agent2: Agent1: Test task"
    )
    
    assert result["status"] in ["completed", "stopped"]
    assert result["total_iterations"] > 0
    assert "iterations" in result
    assert "final_result" in result


@pytest.mark.asyncio
async def test_successful_optimization(simple_agents, simple_evaluation, temp_output_dir):
    """Test successful optimization (output matches gold standard)."""
    config = OptimizationConfig(
        max_iterations=3,
        required_pass_count=1,
        output_path=temp_output_dir
    )
    
    pipeline = OptimizationPipeline(agents=simple_agents, config=config)
    pipeline.config.evaluation_function = simple_evaluation
    
    result = await pipeline.optimize(
        task="Test task",
        context={},
        gold_standard="Agent2: Agent1: Test task"
    )
    
    # Should succeed on first iteration
    assert result["optimization_complete"] is True
    assert result["consecutive_passes"] >= 1
    assert result["final_result"]["evaluation_score"] == 1.0


@pytest.mark.asyncio
async def test_failed_optimization(simple_agents, temp_output_dir):
    """Test failed optimization (output doesn't match gold standard)."""
    async def evaluate(output, gold_standard, task, context):
        # Always fail
        return {"score": 0.0, "status": "INCORRECT"}
    
    config = OptimizationConfig(
        max_iterations=2,
        required_pass_count=1,
        output_path=temp_output_dir
    )
    
    pipeline = OptimizationPipeline(agents=simple_agents, config=config)
    pipeline.config.evaluation_function = evaluate
    
    result = await pipeline.optimize(
        task="Test task",
        context={},
        gold_standard="Different output"
    )
    
    # Should fail all iterations
    assert result["optimization_complete"] is False
    assert result["total_iterations"] == 2  # Max iterations reached
    assert result["consecutive_passes"] == 0


@pytest.mark.asyncio
async def test_teacher_model(simple_agents, temp_output_dir):
    """Test teacher model fallback."""
    class TeacherAgent(dspy.Module if DSPY_AVAILABLE else object):
        def forward(self, task: str, student_output: str, gold_standard: str, evaluation_feedback: str) -> str:
            return gold_standard  # Teacher provides correct answer
    
    agents = simple_agents + [
        AgentConfig(
            name="teacher",
            agent=TeacherAgent(),
            metadata={"is_teacher": True}
        )
    ]
    
    async def evaluate(output, gold_standard, task, context):
        score = 1.0 if str(output) == str(gold_standard) else 0.0
        return {"score": score, "status": "CORRECT" if score == 1.0 else "INCORRECT"}
    
    config = OptimizationConfig(
        max_iterations=3,
        required_pass_count=1,
        enable_teacher_model=True,
        output_path=temp_output_dir
    )
    
    pipeline = OptimizationPipeline(agents=agents, config=config)
    pipeline.config.evaluation_function = evaluate
    
    result = await pipeline.optimize(
        task="Test task",
        context={},
        gold_standard="Correct answer"
    )
    
    # Check if teacher was used
    teacher_used = any(it.get("has_teacher_output") for it in result["iterations"])
    assert teacher_used is True


@pytest.mark.asyncio
async def test_consecutive_passes(simple_agents, simple_evaluation, temp_output_dir):
    """Test required consecutive passes."""
    config = OptimizationConfig(
        max_iterations=5,
        required_pass_count=2,  # Need 2 consecutive passes
        output_path=temp_output_dir
    )
    
    pipeline = OptimizationPipeline(agents=simple_agents, config=config)
    pipeline.config.evaluation_function = simple_evaluation
    
    result = await pipeline.optimize(
        task="Test task",
        context={},
        gold_standard="Agent2: Agent1: Test task"
    )
    
    # Should succeed with 2 consecutive passes
    assert result["optimization_complete"] is True
    assert result["consecutive_passes"] >= 2


@pytest.mark.asyncio
async def test_thinking_log(simple_agents, simple_evaluation, temp_output_dir):
    """Test thinking log creation."""
    config = OptimizationConfig(
        max_iterations=2,
        required_pass_count=1,
        output_path=temp_output_dir,
        enable_thinking_log=True
    )
    
    pipeline = OptimizationPipeline(agents=simple_agents, config=config)
    pipeline.config.evaluation_function = simple_evaluation
    
    await pipeline.optimize(
        task="Test task",
        context={},
        gold_standard="Agent2: Agent1: Test task"
    )
    
    # Check if thinking log was created
    thinking_log = temp_output_dir / "thinking.log"
    assert thinking_log.exists()
    
    # Check if log has content
    content = thinking_log.read_text()
    assert len(content) > 0
    assert "Starting optimization" in content or "Iteration" in content


@pytest.mark.asyncio
async def test_parameter_mappings(simple_agents, temp_output_dir):
    """Test parameter mappings between agents."""
    async def evaluate(output, gold_standard, task, context):
        score = 1.0 if str(output) == str(gold_standard) else 0.0
        return {"score": score, "status": "CORRECT" if score == 1.0 else "INCORRECT"}
    
    config = OptimizationConfig(
        max_iterations=2,
        required_pass_count=1,
        output_path=temp_output_dir
    )
    
    pipeline = OptimizationPipeline(agents=simple_agents, config=config)
    pipeline.config.evaluation_function = evaluate
    
    result = await pipeline.optimize(
        task="Test task",
        context={},
        gold_standard="Agent2: Agent1: Test task"
    )
    
    # Should succeed because parameter mapping works
    assert result["optimization_complete"] is True


@pytest.mark.asyncio
async def test_max_iterations(simple_agents, temp_output_dir):
    """Test max iterations limit."""
    async def evaluate(output, gold_standard, task, context):
        # Always fail to force max iterations
        return {"score": 0.0, "status": "INCORRECT"}
    
    config = OptimizationConfig(
        max_iterations=3,
        required_pass_count=1,
        output_path=temp_output_dir
    )
    
    pipeline = OptimizationPipeline(agents=simple_agents, config=config)
    pipeline.config.evaluation_function = evaluate
    
    result = await pipeline.optimize(
        task="Test task",
        context={},
        gold_standard="Different output"
    )
    
    # Should stop at max iterations
    assert result["total_iterations"] == 3
    assert result["optimization_complete"] is False


@pytest.mark.asyncio
async def test_create_optimization_pipeline(simple_agents, temp_output_dir):
    """Test convenience function."""
    pipeline = create_optimization_pipeline(
        agents=simple_agents,
        max_iterations=3,
        required_pass_count=1,
        output_path=temp_output_dir
    )
    
    assert isinstance(pipeline, OptimizationPipeline)
    assert pipeline.config.max_iterations == 3
    assert pipeline.config.required_pass_count == 1
    assert pipeline.output_path == temp_output_dir


@pytest.mark.asyncio
async def test_error_handling(simple_agents, temp_output_dir):
    """Test error handling in agent execution."""
    class FailingAgent(dspy.Module if DSPY_AVAILABLE else object):
        def forward(self, task: str) -> str:
            raise ValueError("Agent failed!")
    
    agents = [
        AgentConfig(
            name="failing_agent",
            agent=FailingAgent()
        )
    ]
    
    config = OptimizationConfig(
        max_iterations=2,
        required_pass_count=1,
        output_path=temp_output_dir
    )
    
    pipeline = OptimizationPipeline(agents=agents, config=config)
    
    async def evaluate(output, gold_standard, task, context):
        return {"score": 0.0, "status": "ERROR"}
    
    pipeline.config.evaluation_function = evaluate
    
    result = await pipeline.optimize(
        task="Test task",
        context={},
        gold_standard="Expected"
    )
    
    # Should handle error gracefully
    assert result["status"] in ["completed", "stopped"]
    # Check if error was recorded
    errors = [it.get("error") for it in result["iterations"] if it.get("error")]
    assert len(errors) > 0


@pytest.mark.asyncio
async def test_gold_standard_provider(simple_agents, temp_output_dir):
    """Test dynamic gold standard provider."""
    async def gold_provider(task, context):
        return f"Gold: {task}"
    
    async def evaluate(output, gold_standard, task, context):
        score = 1.0 if str(output) == str(gold_standard) else 0.0
        return {"score": score, "status": "CORRECT" if score == 1.0 else "INCORRECT"}
    
    config = OptimizationConfig(
        max_iterations=2,
        required_pass_count=1,
        output_path=temp_output_dir,
        gold_standard_provider=gold_provider
    )
    
    pipeline = OptimizationPipeline(agents=simple_agents, config=config)
    pipeline.config.evaluation_function = evaluate
    
    result = await pipeline.optimize(
        task="Test task",
        context={}
        # No gold_standard provided - should use provider
    )
    
    assert result["status"] in ["completed", "stopped"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
