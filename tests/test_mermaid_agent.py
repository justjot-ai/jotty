"""
Test MermaidAgent - First migrated domain agent

Tests the new MermaidAgent in core/execution/agents/
"""

import logging

import pytest
from Jotty.core.intelligence.reasoning.agents import MermaidAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.asyncio
async def test_mermaid_agent_instantiation():
    """Test that MermaidAgent can be instantiated."""
    agent = MermaidAgent()

    assert agent is not None
    assert agent.config.name == "MermaidAgent"

    # Check capabilities
    assert hasattr(agent, "learning_domain")
    assert agent.learning_domain == "mermaid"
    assert hasattr(agent, "validation_domain")
    assert agent.validation_domain == "mermaid"

    logger.info("✅ MermaidAgent instantiation successful")


@pytest.mark.asyncio
async def test_mermaid_agent_with_learning_disabled():
    """Test MermaidAgent with learning disabled."""
    agent = MermaidAgent(enable_learning=False)

    assert agent is not None
    # Should not have learning capability
    assert not hasattr(agent, "gold_standards")

    # Should still have validation
    assert hasattr(agent, "validation_domain")

    logger.info("✅ MermaidAgent without learning successful")


@pytest.mark.asyncio
async def test_mermaid_agent_gold_standards():
    """Test that gold standards are loaded."""
    agent = MermaidAgent(enable_learning=True)

    # Check gold standards
    assert hasattr(agent, "gold_standards")
    assert len(agent.gold_standards) == 4  # 4 training cases

    # Check validation cases
    assert hasattr(agent, "validation_cases")
    assert len(agent.validation_cases) == 2  # 2 validation cases

    logger.info("✅ Gold standards loaded correctly")


@pytest.mark.asyncio
async def test_mermaid_agent_validation():
    """Test MermaidAgent validation capability."""
    agent = MermaidAgent()

    # Valid diagram
    valid_diagram = """graph TD
    A[Start]
    B[End]
    A --> B"""

    validation = await agent.validate(valid_diagram)

    assert validation["valid"] is True
    assert validation["score"] >= 0.7
    assert len(validation["errors"]) == 0

    logger.info("✅ Validation of valid diagram successful")


@pytest.mark.asyncio
async def test_mermaid_agent_validation_invalid():
    """Test validation of invalid diagram."""
    agent = MermaidAgent(strict_validation=False)

    # Invalid diagram (no proper start)
    invalid_diagram = "This is not a mermaid diagram"

    validation = await agent.validate(invalid_diagram)

    assert validation["valid"] is False
    assert validation["score"] < 0.7
    assert len(validation["errors"]) > 0

    logger.info("✅ Validation of invalid diagram successful")


@pytest.mark.asyncio
async def test_mermaid_agent_fallback():
    """Test fallback diagram generation."""
    agent = MermaidAgent()

    # Test fallback method
    fallback = agent._fallback_diagram("Test process", "flowchart")

    assert "graph TD" in fallback
    assert "Start" in fallback or "Test" in fallback

    logger.info("✅ Fallback diagram generation successful")


@pytest.mark.asyncio
async def test_mermaid_agent_detect_type():
    """Test diagram type detection."""
    agent = MermaidAgent()

    # Test different types
    assert agent._detect_diagram_type("graph TD\nA --> B") == "flowchart"
    assert agent._detect_diagram_type("sequenceDiagram\nA->>B") == "sequence"
    assert agent._detect_diagram_type("classDiagram\nclass A") == "class"
    assert agent._detect_diagram_type("stateDiagram-v2\n[*]-->A") == "state"

    logger.info("✅ Diagram type detection successful")


@pytest.mark.asyncio
async def test_mermaid_agent_learning_stats():
    """Test learning statistics."""
    agent = MermaidAgent(enable_learning=True)

    stats = agent.get_learning_stats()

    assert "training_examples" in stats
    assert stats["training_examples"] == 4
    assert "improvements_count" in stats

    logger.info("✅ Learning stats successful")


@pytest.mark.asyncio
async def test_mermaid_agent_validation_stats():
    """Test validation statistics."""
    agent = MermaidAgent()

    # Perform a validation
    await agent.validate("graph TD\nA --> B")

    stats = agent.get_validation_stats()

    assert "total_validations" in stats
    assert stats["total_validations"] >= 1
    assert "passed" in stats

    logger.info("✅ Validation stats successful")


def test_mermaid_agent_training_data():
    """Test training data structure."""
    training = MermaidAgent._get_default_training_cases()
    validation = MermaidAgent._get_default_validation_cases()

    # Check training cases
    assert len(training) == 4
    for case in training:
        assert "input" in case
        assert "output" in case
        assert "context" in case

    # Check validation cases
    assert len(validation) == 2
    for case in validation:
        assert "input" in case
        assert "output" in case

    logger.info("✅ Training data structure valid")


if __name__ == "__main__":
    """Run tests directly."""
    import asyncio

    print("\n" + "=" * 80)
    print("MERMAID AGENT - TESTS")
    print("=" * 80)

    print("\n📋 Running: test_mermaid_agent_instantiation")
    asyncio.run(test_mermaid_agent_instantiation())

    print("\n📋 Running: test_mermaid_agent_with_learning_disabled")
    asyncio.run(test_mermaid_agent_with_learning_disabled())

    print("\n📋 Running: test_mermaid_agent_gold_standards")
    asyncio.run(test_mermaid_agent_gold_standards())

    print("\n📋 Running: test_mermaid_agent_validation")
    asyncio.run(test_mermaid_agent_validation())

    print("\n📋 Running: test_mermaid_agent_validation_invalid")
    asyncio.run(test_mermaid_agent_validation_invalid())

    print("\n📋 Running: test_mermaid_agent_fallback")
    asyncio.run(test_mermaid_agent_fallback())

    print("\n📋 Running: test_mermaid_agent_detect_type")
    asyncio.run(test_mermaid_agent_detect_type())

    print("\n📋 Running: test_mermaid_agent_learning_stats")
    asyncio.run(test_mermaid_agent_learning_stats())

    print("\n📋 Running: test_mermaid_agent_validation_stats")
    asyncio.run(test_mermaid_agent_validation_stats())

    print("\n📋 Running: test_mermaid_agent_training_data")
    test_mermaid_agent_training_data()

    print("\n✅ All tests completed!")
    print("=" * 80)
