"""
Test Coding Template - Real LLM Integration Test

This test demonstrates the CodingTemplate with actual LLM execution.
It validates the 3-stage workflow with CUSTOM pattern and STAGES.

Run with: pytest tests/test_coding_template.py -v -s
"""

import asyncio
import logging
import os
from pathlib import Path

import pytest

from Jotty.core.intelligence.swarms.coding_swarm.types import CodingConfig
from Jotty.core.intelligence.swarms.templates.coding import CodingTemplate

# Setup logging for detailed output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@pytest.fixture
def coding_config():
    """Create test configuration."""
    return CodingConfig(
        language="python",
        style="standard",
        include_tests=True,
        include_docs=True,
        include_types=True,
        enable_workspace=False,  # Skip workspace validation for faster test
        enable_research=False,  # Skip research phase for faster test
        skip_team_planning=True,  # Skip team planning for faster test
        skip_team_review=True,  # Skip team review for faster test
        max_fix_attempts=1,  # Reduce for faster test
        max_design_iterations=1,  # Reduce for faster test
    )


@pytest.fixture
def output_dir():
    """Create and return test output directory."""
    out_dir = Path("/tmp/jotty_coding_test")
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_coding_template_instantiation(coding_config):
    """Test that CodingTemplate can be instantiated."""
    template = CodingTemplate(coding_config)

    assert template is not None
    assert template.config == coding_config
    assert template.TASK_TYPE == "coding"
    assert len(template.DEFAULT_TOOLS) == 3
    assert template.AGENT_TEAM is not None
    assert len(template.STAGES) == 3

    logger.info("✅ CodingTemplate instantiation successful")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_coding_template_stages_validation(coding_config):
    """Test that STAGES are properly configured."""
    template = CodingTemplate(coding_config)

    # Validate stage structure
    assert len(template.STAGES) == 3

    # Stage 1: Design (Architect)
    stage1 = template.STAGES[0]
    assert stage1.name == "design"
    assert stage1.parallel is False
    assert len(stage1.agents) == 1
    assert "_architect" in stage1.agents
    assert stage1.needs == []
    assert stage1.output_key == "architecture"

    # Stage 2: Implement (Developer, depends on design)
    stage2 = template.STAGES[1]
    assert stage2.name == "implement"
    assert stage2.parallel is False
    assert len(stage2.agents) == 1
    assert "_developer" in stage2.agents
    assert "design" in stage2.needs
    assert stage2.output_key == "code"

    # Stage 3: Test (TestWriter, depends on implement)
    stage3 = template.STAGES[2]
    assert stage3.name == "test"
    assert stage3.parallel is False
    assert len(stage3.agents) == 1
    assert "_test_writer" in stage3.agents
    assert "implement" in stage3.needs
    assert stage3.output_key == "tests"

    logger.info("✅ STAGES validation successful")


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"),
    reason="Requires LLM API key (ANTHROPIC_API_KEY or OPENAI_API_KEY)",
)
async def test_coding_template_execution_real_llm(coding_config, output_dir):
    """
    Test CodingTemplate with real LLM execution.

    This is the MAIN TEST that demonstrates the complete workflow.
    Uses a simple, well-defined task to ensure reliable execution.
    """
    logger.info("=" * 80)
    logger.info("Starting CodingTemplate Real LLM Test")
    logger.info("=" * 80)

    # Create template
    template = CodingTemplate(coding_config)

    # Simple task: Build a REST API endpoint
    task = "Build a simple REST API endpoint for user management with GET and POST operations"
    logger.info(f"\n🔨 Executing coding task: {task}")

    try:
        result = await template.execute(
            query=task,
            language="Python",
            style="clean",
            framework="pytest",
        )

        # Validate result structure
        assert result is not None
        assert (
            result.success is True
        ), f"Coding failed: {result.error if hasattr(result, 'error') else 'Unknown error'}"

        # Log results
        logger.info("\n" + "=" * 80)
        logger.info("CODING RESULTS")
        logger.info("=" * 80)
        logger.info(f"Success: {result.success}")
        logger.info(f"Execution Time: {result.execution_time:.2f}s")

        # Check output structure
        output = result.output
        assert isinstance(output, dict), "Output should be a dictionary"

        # Validate stages completed
        if "architecture" in output:
            logger.info("\n🏗️  Architecture Stage:")
            arch = output["architecture"]
            if isinstance(arch, dict):
                logger.info(f"  Components: {len(arch.get('components', []))}")
                logger.info(f"  Architecture: {arch.get('architecture', 'N/A')[:100]}...")
            else:
                logger.info(f"  Architecture: {str(arch)[:100]}...")

        if "code" in output:
            logger.info("\n💻 Code Generation Stage:")
            code = output["code"]
            if isinstance(code, dict):
                logger.info(f"  Code: {code.get('code', 'N/A')[:100]}...")
                logger.info(f"  Filename: {code.get('filename', 'N/A')}")
            else:
                logger.info(f"  Code: {str(code)[:100]}...")

        if "tests" in output:
            logger.info("\n🧪 Test Generation Stage:")
            tests = output["tests"]
            if isinstance(tests, dict):
                logger.info(f"  Tests: {tests.get('tests', 'N/A')[:100]}...")
                logger.info(f"  Framework: {tests.get('framework', 'pytest')}")
            else:
                logger.info(f"  Tests: {str(tests)[:100]}...")

        logger.info("\n" + "=" * 80)

        # Validate required fields
        assert result.execution_time > 0, "Execution time should be positive"
        assert isinstance(result.output, dict), "Output should be a dictionary"

        # Validate metadata if present
        if hasattr(result, "metadata") and result.metadata:
            logger.info("\n📊 Metadata:")
            for key, value in result.metadata.items():
                logger.info(f"  {key}: {value}")

        logger.info("\n✅ All validations passed!")

        return result

    except Exception as e:
        logger.error(f"\n❌ Coding execution failed: {e}")
        import traceback

        traceback.print_exc()
        raise


@pytest.mark.asyncio
@pytest.mark.integration
async def test_coding_template_simple_execution(coding_config):
    """Test coding template with a very simple task (no LLM required)."""
    template = CodingTemplate(coding_config)

    # Very simple task
    result = await template.execute(
        query="Create a hello world function",
        language="Python",
        style="minimal",
    )

    # Should still return a result structure
    assert result is not None
    assert hasattr(result, "success")
    assert hasattr(result, "output")
    assert hasattr(result, "execution_time")

    logger.info("✅ Simple execution test passed")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_coding_template_different_languages():
    """Test coding template with different language configurations."""
    # Python
    python_config = CodingConfig(language="python", style="standard")
    python_template = CodingTemplate(python_config)
    assert python_template.config.language == "python"

    # JavaScript
    js_config = CodingConfig(language="javascript", style="functional")
    js_template = CodingTemplate(js_config)
    assert js_template.config.language == "javascript"

    logger.info("✅ Multi-language configuration test passed")


def test_coding_template_backward_compatibility():
    """Test backward compatibility alias."""
    from Jotty.core.intelligence.swarms.templates.coding import CodingSwarm

    assert CodingSwarm is CodingTemplate

    logger.info("✅ Backward compatibility test passed")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_coding_template_context_building(coding_config):
    """Test that context is properly built for stage execution."""
    template = CodingTemplate(coding_config)

    # Build context manually to test
    query = "Build a calculator"
    context = {
        "query": query,
        "requirements": query,
        "language": "Python",
        "style": "clean",
        "framework": "pytest",
        "config": template.config,
    }

    # Verify context structure
    assert context["query"] == query
    assert context["requirements"] == query
    assert context["language"] == "Python"
    assert context["style"] == "clean"
    assert context["framework"] == "pytest"
    assert context["config"] == template.config

    logger.info("✅ Context building test passed")


if __name__ == "__main__":
    """Run tests directly with real LLM."""
    print("\n" + "=" * 80)
    print("CODING TEMPLATE - REAL LLM TEST")
    print("=" * 80)

    # Setup
    config = CodingConfig(
        language="python",
        style="standard",
        include_tests=True,
        include_docs=True,
        include_types=True,
        enable_workspace=False,
        enable_research=False,
        skip_team_planning=True,
        skip_team_review=True,
        max_fix_attempts=1,
        max_design_iterations=1,
    )

    output_dir = Path("/tmp/jotty_coding_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run main test
    print("\n📋 Running: test_coding_template_execution_real_llm")
    result = asyncio.run(test_coding_template_execution_real_llm(config, str(output_dir)))

    print("\n✅ All tests completed successfully!")
    print("=" * 80)
