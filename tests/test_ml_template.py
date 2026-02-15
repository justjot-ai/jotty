"""
Test ML Template - Integration Tests

Tests the MLTemplate for machine learning workflows.

Run with: pytest tests/test_ml_template.py -v -s
"""

import logging
import pytest

from Jotty.core.intelligence.swarms.templates.ml import MLTemplate
from Jotty.core.intelligence.swarms.swarm_learning import SwarmBaseConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@pytest.fixture
def ml_config():
    """Create test configuration."""
    return SwarmBaseConfig(name="ML", domain="ml")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_ml_template_instantiation(ml_config):
    """Test that MLTemplate can be instantiated."""
    template = MLTemplate(ml_config)

    assert template is not None
    assert template.config == ml_config
    assert template.TASK_TYPE == "ml"
    assert template.AGENT_TEAM is not None

    logger.info("✅ MLTemplate instantiation successful")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_ml_template_placeholder_execution(ml_config):
    """Test ML template returns placeholder result."""
    template = MLTemplate(ml_config)

    result = await template.execute(
        query="Train a classifier for fraud detection",
        data="transactions.csv",
        target="is_fraud",
        model_type="xgboost",
    )

    # Validate result structure
    assert result is not None
    assert result.success is True
    assert hasattr(result, "output")
    assert isinstance(result.output, dict)

    # Check placeholder output
    output = result.output
    assert output["model_type"] == "xgboost"
    assert output["target"] == "is_fraud"
    assert output["status"] == "trained"
    assert "placeholder" in output

    logger.info("✅ Placeholder execution test passed")


def test_ml_template_backward_compatibility():
    """Test backward compatibility alias."""
    from Jotty.core.intelligence.swarms.templates.ml import MLSwarm

    assert MLSwarm is MLTemplate

    logger.info("✅ Backward compatibility test passed")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_ml_template_auto_model_type():
    """Test ML template with auto model type."""
    template = MLTemplate()

    result = await template.execute(
        query="Build a model",
        data="data.csv",
        target="label",
    )

    assert result is not None
    assert result.success is True
    assert result.output["model_type"] == "auto"

    logger.info("✅ Auto model type test passed")


if __name__ == "__main__":
    """Run tests directly."""
    import asyncio

    print("\n" + "=" * 80)
    print("ML TEMPLATE - TESTS")
    print("=" * 80)

    config = SwarmBaseConfig(name="ML", domain="ml")

    print("\n📋 Running: test_ml_template_instantiation")
    asyncio.run(test_ml_template_instantiation(config))

    print("\n📋 Running: test_ml_template_placeholder_execution")
    asyncio.run(test_ml_template_placeholder_execution(config))

    print("\n📋 Running: test_ml_template_backward_compatibility")
    test_ml_template_backward_compatibility()

    print("\n✅ All tests completed successfully!")
    print("=" * 80)
