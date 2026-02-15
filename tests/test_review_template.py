"""
Test Review Template - Integration Tests

Tests the ReviewTemplate for code review workflows.

Run with: pytest tests/test_review_template.py -v -s
"""

import logging

import pytest
from Jotty.core.intelligence.swarms.swarm_learning import SwarmBaseConfig
from Jotty.core.intelligence.swarms.templates.review import ReviewTemplate

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@pytest.fixture
def review_config():
    """Create test configuration."""
    return SwarmBaseConfig(name="Review", domain="review")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_review_template_instantiation(review_config):
    """Test that ReviewTemplate can be instantiated."""
    template = ReviewTemplate(review_config)

    assert template is not None
    assert template.config == review_config
    assert template.TASK_TYPE == "review"
    assert template.AGENT_TEAM is not None

    logger.info("✅ ReviewTemplate instantiation successful")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_review_template_placeholder_execution(review_config):
    """Test review template returns placeholder result."""
    template = ReviewTemplate(review_config)

    code_to_review = """
def add(a, b):
    return a + b
"""

    result = await template.execute(
        query=code_to_review,
        code=code_to_review,
        language="Python",
        focus_areas=["quality", "security"],
    )

    # Validate result structure
    assert result is not None
    assert result.success is True
    assert hasattr(result, "output")
    assert isinstance(result.output, dict)

    # Check placeholder output
    output = result.output
    assert "code_analyzed" in output
    assert output["code_analyzed"] is True
    assert output["language"] == "Python"
    assert "quality" in output["focus_areas"]
    assert "security" in output["focus_areas"]

    logger.info("✅ Placeholder execution test passed")


def test_review_template_backward_compatibility():
    """Test backward compatibility alias."""
    from Jotty.core.intelligence.swarms.templates.review import ReviewSwarm

    assert ReviewSwarm is ReviewTemplate

    logger.info("✅ Backward compatibility test passed")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_review_template_empty_code():
    """Test review template with empty code."""
    template = ReviewTemplate()

    result = await template.execute(
        query="",
        code="",
        language="Python",
    )

    assert result is not None
    assert result.success is True
    assert result.output["code_analyzed"] is False  # Empty code

    logger.info("✅ Empty code test passed")


if __name__ == "__main__":
    """Run tests directly."""
    import asyncio

    print("\n" + "=" * 80)
    print("REVIEW TEMPLATE - TESTS")
    print("=" * 80)

    config = SwarmBaseConfig(name="Review", domain="review")

    print("\n📋 Running: test_review_template_instantiation")
    asyncio.run(test_review_template_instantiation(config))

    print("\n📋 Running: test_review_template_placeholder_execution")
    asyncio.run(test_review_template_placeholder_execution(config))

    print("\n📋 Running: test_review_template_backward_compatibility")
    test_review_template_backward_compatibility()

    print("\n✅ All tests completed successfully!")
    print("=" * 80)
