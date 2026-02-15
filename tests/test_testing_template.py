"""
Test Testing Template - Integration Tests

Tests the TestingTemplate for test generation and QA workflows.

Run with: pytest tests/test_testing_template.py -v -s
"""

import logging

import pytest
from Jotty.core.intelligence.swarms._base.swarm_learning import SwarmBaseConfig
from Jotty.core.intelligence.swarms.templates.testing import TestingTemplate

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@pytest.fixture
def testing_config():
    """Create test configuration."""
    return SwarmBaseConfig(name="Testing", domain="testing")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_testing_template_instantiation(testing_config):
    """Test that TestingTemplate can be instantiated."""
    template = TestingTemplate(testing_config)

    assert template is not None
    assert template.config == testing_config
    assert template.TASK_TYPE == "testing"
    assert template.AGENT_TEAM is not None

    logger.info("✅ TestingTemplate instantiation successful")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_testing_template_placeholder_execution(testing_config):
    """Test testing template returns placeholder result."""
    template = TestingTemplate(testing_config)

    code_to_test = """
def calculate_total(items):
    return sum(item.price for item in items)
"""

    result = await template.execute(
        query=code_to_test,
        code=code_to_test,
        language="Python",
        test_framework="pytest",
    )

    # Validate result structure
    assert result is not None
    assert result.success is True
    assert hasattr(result, "output")
    assert isinstance(result.output, dict)

    # Check placeholder output
    output = result.output
    assert "code_tested" in output
    assert output["code_tested"] is True
    assert output["framework"] == "pytest"
    assert output["language"] == "Python"

    logger.info("✅ Placeholder execution test passed")


def test_testing_template_backward_compatibility():
    """Test backward compatibility alias."""
    from Jotty.core.intelligence.swarms.templates.testing import TestingSwarm

    assert TestingSwarm is TestingTemplate

    logger.info("✅ Backward compatibility test passed")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_testing_template_empty_code():
    """Test testing template with empty code."""
    template = TestingTemplate()

    result = await template.execute(
        query="",
        code="",
    )

    assert result is not None
    assert result.success is True
    assert result.output["code_tested"] is False  # No code

    logger.info("✅ Empty code test passed")


if __name__ == "__main__":
    """Run tests directly."""
    import asyncio

    print("\n" + "=" * 80)
    print("TESTING TEMPLATE - TESTS")
    print("=" * 80)

    config = SwarmBaseConfig(name="Testing", domain="testing")

    print("\n📋 Running: test_testing_template_instantiation")
    asyncio.run(test_testing_template_instantiation(config))

    print("\n📋 Running: test_testing_template_placeholder_execution")
    asyncio.run(test_testing_template_placeholder_execution(config))

    print("\n📋 Running: test_testing_template_backward_compatibility")
    test_testing_template_backward_compatibility()

    print("\n✅ All tests completed successfully!")
    print("=" * 80)
