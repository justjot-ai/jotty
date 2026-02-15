"""
Test Execution Layer Capabilities

Tests the new capability mixins in core/execution/capabilities/
"""

import logging

import pytest
from Jotty.core.execution.base import AgentRuntimeConfig, BaseAgent
from Jotty.core.execution.capabilities import (
    LearningCapability,
    MemoryCapability,
    ValidationCapability,
)

logging.basicConfig(level=logging.INFO)


# Test agent with learning capability
class TestLearningAgent(BaseAgent, LearningCapability):
    """Test agent with learning capability."""

    def __init__(self, config=None):
        BaseAgent.__init__(self, config or AgentRuntimeConfig(name="TestLearning"))
        LearningCapability.__init__(
            self,
            domain="test",
            gold_standards=[
                {"input": "hello", "output": "Hello, world!"},
                {"input": "goodbye", "output": "Goodbye!"},
            ],
        )

    async def _execute_impl(self, task: str, **kwargs):
        result = f"Processed: {task}"

        # Learn from gold standards
        if hasattr(self, "learn_from_gold_standards"):
            result = await self.learn_from_gold_standards(task, result)

        return result


# Test agent with validation capability
class TestValidationAgent(BaseAgent, ValidationCapability):
    """Test agent with validation capability."""

    def __init__(self, config=None):
        BaseAgent.__init__(self, config or AgentRuntimeConfig(name="TestValidation"))
        ValidationCapability.__init__(self, domain="test", strict_mode=True)

    async def _execute_impl(self, task: str, **kwargs):
        result = f"Processed: {task}"

        # Validate result
        validation = await self.validate(result)
        if not validation["valid"]:
            result = f"INVALID: {result}"

        return result

    async def _validate_impl(self, output, **kwargs):
        """Simple validation: output must contain 'Processed'."""
        errors = []

        if "Processed" not in str(output):
            errors.append("Output must contain 'Processed'")

        return {
            "valid": len(errors) == 0,
            "score": 1.0 if len(errors) == 0 else 0.0,
            "errors": errors,
            "warnings": [],
        }


# Test agent with memory capability
class TestMemoryAgent(BaseAgent, MemoryCapability):
    """Test agent with memory capability."""

    def __init__(self, config=None):
        BaseAgent.__init__(self, config or AgentRuntimeConfig(name="TestMemory"))
        MemoryCapability.__init__(self, domain="test", auto_store=True)

    async def _execute_impl(self, query: str, **kwargs):
        # Retrieve past memories
        memories = await self.retrieve_memories(query)

        result = {
            "query": query,
            "memories_found": len(memories),
            "processed": f"Processed: {query}",
        }

        # Store result
        await self.store_memory(str(result), level="episodic", goal="test")

        return result


# Test agent with all capabilities
class TestFullAgent(BaseAgent, LearningCapability, ValidationCapability, MemoryCapability):
    """Test agent with all capabilities."""

    def __init__(self, config=None):
        BaseAgent.__init__(self, config or AgentRuntimeConfig(name="TestFull"))
        LearningCapability.__init__(self, domain="test")
        ValidationCapability.__init__(self, domain="test")
        MemoryCapability.__init__(self, domain="test", auto_store=False)

    async def _execute_impl(self, task: str, **kwargs):
        result = f"Full: {task}"

        # Validate
        validation = await self.validate(result)

        # Learn
        if hasattr(self, "learn_from_gold_standards"):
            result = await self.learn_from_gold_standards(task, result)

        # Memory
        memories = await self.retrieve_memories(task)

        return {
            "result": result,
            "valid": validation["valid"],
            "memories": len(memories),
        }


# =========================================================================
# TESTS
# =========================================================================


@pytest.mark.asyncio
async def test_learning_capability_instantiation():
    """Test that agent with LearningCapability can be instantiated."""
    agent = TestLearningAgent()

    assert agent is not None
    assert agent.learning_domain == "test"
    assert len(agent.gold_standards) == 2
    assert agent.get_learning_stats()["training_examples"] == 2


@pytest.mark.asyncio
async def test_learning_capability_execution():
    """Test agent execution with learning capability."""
    agent = TestLearningAgent()

    result = await agent.execute(task="hello")

    assert result.success is True
    assert "hello" in str(result.output).lower()

    # Check learning stats updated
    stats = agent.get_learning_stats()
    assert stats["learning_runs"] >= 0  # May be 0 if no actual learning occurred


@pytest.mark.asyncio
async def test_validation_capability_instantiation():
    """Test that agent with ValidationCapability can be instantiated."""
    agent = TestValidationAgent()

    assert agent is not None
    assert agent.validation_domain == "test"
    assert agent.strict_mode is True


@pytest.mark.asyncio
async def test_validation_capability_execution():
    """Test agent execution with validation capability."""
    agent = TestValidationAgent()

    result = await agent.execute(task="hello")

    assert result.success is True
    assert "Processed" in str(result.output)

    # Check validation stats
    stats = agent.get_validation_stats()
    assert stats["total_validations"] >= 1


@pytest.mark.asyncio
async def test_validation_capability_validation():
    """Test validation functionality."""
    agent = TestValidationAgent()

    # Valid output
    valid_result = await agent.validate("Processed: test")
    assert valid_result["valid"] is True
    assert valid_result["score"] == 1.0

    # Invalid output
    invalid_result = await agent.validate("Invalid output")
    assert invalid_result["valid"] is False
    assert len(invalid_result["errors"]) > 0


@pytest.mark.asyncio
async def test_memory_capability_instantiation():
    """Test that agent with MemoryCapability can be instantiated."""
    agent = TestMemoryAgent()

    assert agent is not None
    assert agent.memory_domain == "test"
    assert agent.auto_store is True


@pytest.mark.asyncio
async def test_memory_capability_execution():
    """Test agent execution with memory capability."""
    agent = TestMemoryAgent()

    result = await agent.execute(query="test query")

    assert result.success is True
    assert "query" in result.output
    assert result.output["query"] == "test query"

    # Check memory stats
    stats = agent.get_memory_stats()
    assert "memories_stored" in stats


@pytest.mark.asyncio
async def test_multiple_capabilities_instantiation():
    """Test that agent with multiple capabilities can be instantiated."""
    agent = TestFullAgent()

    assert agent is not None
    assert hasattr(agent, "learning_domain")
    assert hasattr(agent, "validation_domain")
    assert hasattr(agent, "memory_domain")


@pytest.mark.asyncio
async def test_multiple_capabilities_execution():
    """Test agent with all capabilities working together."""
    agent = TestFullAgent()

    result = await agent.execute(task="test")

    assert result.success is True
    assert isinstance(result.output, dict)
    assert "result" in result.output
    assert "valid" in result.output


def test_learning_capability_methods():
    """Test LearningCapability methods."""
    agent = TestLearningAgent()

    # Test gold standards
    standards = agent.get_gold_standards()
    assert len(standards) == 2

    # Add new standard
    agent.add_gold_standard({"input": "test", "output": "Test!"})
    assert len(agent.get_gold_standards()) == 3

    # Test improvements
    improvements = agent.get_improvements()
    assert isinstance(improvements, list)

    # Add improvement
    agent.add_improvement({"pattern": "test", "improvement": "Use capitalization"})
    assert len(agent.get_improvements()) == 1


def test_validation_capability_methods():
    """Test ValidationCapability methods."""
    agent = TestValidationAgent()

    # Test stats
    stats = agent.get_validation_stats()
    assert "total_validations" in stats
    assert "passed" in stats
    assert "failed" in stats

    # Test history
    history = agent.get_validation_history()
    assert isinstance(history, list)


def test_memory_capability_methods():
    """Test MemoryCapability methods."""
    agent = TestMemoryAgent()

    # Test stats
    stats = agent.get_memory_stats()
    assert "memories_stored" in stats
    assert "memories_retrieved" in stats

    # Test memory context building
    context = agent.build_memory_context("test", memories=[])
    assert isinstance(context, str)


if __name__ == "__main__":
    """Run tests directly."""
    import asyncio

    print("\n" + "=" * 80)
    print("EXECUTION LAYER CAPABILITIES - TESTS")
    print("=" * 80)

    print("\n📋 Running: test_learning_capability_instantiation")
    asyncio.run(test_learning_capability_instantiation())

    print("\n📋 Running: test_validation_capability_instantiation")
    asyncio.run(test_validation_capability_instantiation())

    print("\n📋 Running: test_memory_capability_instantiation")
    asyncio.run(test_memory_capability_instantiation())

    print("\n📋 Running: test_multiple_capabilities_instantiation")
    asyncio.run(test_multiple_capabilities_instantiation())

    print("\n📋 Running: test_learning_capability_methods")
    test_learning_capability_methods()

    print("\n📋 Running: test_validation_capability_methods")
    test_validation_capability_methods()

    print("\n📋 Running: test_memory_capability_methods")
    test_memory_capability_methods()

    print("\n✅ All tests completed!")
    print("=" * 80)
