"""
Sample agents and mock implementations for testing JOTTY.

Provides reusable mock agents and DSPy modules for test scenarios.
"""
from unittest.mock import Mock, AsyncMock, MagicMock
from typing import Any, Dict, List
from dataclasses import dataclass


# =============================================================================
# Simple Mock Agents
# =============================================================================

class SimpleMockAgent:
    """A simple mock agent that returns predefined outputs."""

    def __init__(self, output: str = "Test output"):
        self.output = output
        self.call_count = 0
        self.last_input = None

    def forward(self, **kwargs) -> Any:
        """Forward method that records calls."""
        self.call_count += 1
        self.last_input = kwargs
        return Mock(answer=self.output, output=self.output)

    def __call__(self, **kwargs) -> Any:
        """Call method that delegates to forward."""
        return self.forward(**kwargs)


class AsyncMockAgent:
    """An async mock agent for testing async workflows."""

    def __init__(self, output: str = "Async test output"):
        self.output = output
        self.call_count = 0
        self.last_input = None

    async def forward(self, **kwargs) -> Any:
        """Async forward method."""
        self.call_count += 1
        self.last_input = kwargs
        return Mock(answer=self.output, output=self.output)

    async def __call__(self, **kwargs) -> Any:
        """Async call method."""
        return await self.forward(**kwargs)


class FailingMockAgent:
    """A mock agent that always fails."""

    def __init__(self, error_message: str = "Agent execution failed"):
        self.error_message = error_message
        self.call_count = 0

    def forward(self, **kwargs) -> Any:
        """Forward method that raises an exception."""
        self.call_count += 1
        raise Exception(self.error_message)

    def __call__(self, **kwargs) -> Any:
        """Call method that raises an exception."""
        return self.forward(**kwargs)


class ConditionalMockAgent:
    """A mock agent that succeeds or fails based on input."""

    def __init__(self, success_condition: callable = None):
        self.success_condition = success_condition or (lambda x: True)
        self.call_count = 0

    def forward(self, **kwargs) -> Any:
        """Forward method with conditional success."""
        self.call_count += 1

        if self.success_condition(kwargs):
            return Mock(answer="Success", output="Success")
        else:
            raise Exception("Condition not met")

    def __call__(self, **kwargs) -> Any:
        """Call method with conditional success."""
        return self.forward(**kwargs)


# =============================================================================
# DSPy-like Mock Agents
# =============================================================================

@dataclass
class MockPrediction:
    """Mock DSPy Prediction object."""
    answer: str
    reasoning: str = ""
    confidence: float = 1.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MockChainOfThoughtAgent:
    """Mock DSPy ChainOfThought agent."""

    def __init__(self, answer: str = "Test answer", reasoning: str = "Test reasoning"):
        self.answer = answer
        self.reasoning = reasoning
        self.call_count = 0
        self.call_history = []

    def forward(self, **kwargs) -> MockPrediction:
        """Forward method that mimics ChainOfThought."""
        self.call_count += 1
        self.call_history.append(kwargs)

        return MockPrediction(
            answer=self.answer,
            reasoning=self.reasoning,
            metadata={"call_count": self.call_count}
        )

    def __call__(self, **kwargs) -> MockPrediction:
        """Call method."""
        return self.forward(**kwargs)


class MockReActAgent:
    """Mock DSPy ReAct agent with tool calling."""

    def __init__(self, final_answer: str = "ReAct answer", steps: int = 3):
        self.final_answer = final_answer
        self.steps = steps
        self.call_count = 0
        self.tool_calls = []

    def forward(self, **kwargs) -> MockPrediction:
        """Forward method that mimics ReAct."""
        self.call_count += 1

        # Simulate tool calls
        for i in range(self.steps):
            tool_call = {
                "tool": f"tool_{i}",
                "input": kwargs,
                "output": f"tool_{i}_output"
            }
            self.tool_calls.append(tool_call)

        return MockPrediction(
            answer=self.final_answer,
            reasoning=f"Executed {self.steps} steps",
            metadata={"tool_calls": len(self.tool_calls)}
        )

    def __call__(self, **kwargs) -> MockPrediction:
        """Call method."""
        return self.forward(**kwargs)


# =============================================================================
# Stateful Mock Agents
# =============================================================================

class StatefulMockAgent:
    """A mock agent that maintains state across calls."""

    def __init__(self):
        self.state = {}
        self.call_count = 0
        self.call_history = []

    def forward(self, **kwargs) -> MockPrediction:
        """Forward method that updates state."""
        self.call_count += 1
        self.call_history.append(kwargs)

        # Update state
        for key, value in kwargs.items():
            self.state[key] = value

        return MockPrediction(
            answer=f"State: {self.state}",
            metadata={"state": self.state.copy()}
        )

    def __call__(self, **kwargs) -> MockPrediction:
        """Call method."""
        return self.forward(**kwargs)

    def reset_state(self):
        """Reset agent state."""
        self.state = {}
        self.call_count = 0
        self.call_history = []


# =============================================================================
# Agent Factories
# =============================================================================

def create_mock_agent(agent_type: str = "simple", **kwargs) -> Any:
    """
    Factory function to create mock agents.

    Parameters:
    -----------
    agent_type : str
        Type of agent to create: 'simple', 'async', 'failing', 'conditional',
        'chainofthought', 'react', 'stateful'
    **kwargs : Any
        Additional arguments for agent initialization

    Returns:
    --------
    Mock agent instance
    """
    agent_classes = {
        "simple": SimpleMockAgent,
        "async": AsyncMockAgent,
        "failing": FailingMockAgent,
        "conditional": ConditionalMockAgent,
        "chainofthought": MockChainOfThoughtAgent,
        "react": MockReActAgent,
        "stateful": StatefulMockAgent,
    }

    agent_class = agent_classes.get(agent_type.lower())
    if not agent_class:
        raise ValueError(f"Unknown agent type: {agent_type}")

    return agent_class(**kwargs)


def create_agent_pipeline(num_agents: int = 3, agent_type: str = "simple") -> List[Any]:
    """
    Create a pipeline of mock agents.

    Parameters:
    -----------
    num_agents : int
        Number of agents in the pipeline
    agent_type : str
        Type of agents to create

    Returns:
    --------
    List of mock agents
    """
    return [
        create_mock_agent(agent_type, output=f"Agent {i} output")
        for i in range(num_agents)
    ]


# =============================================================================
# Mock Tool Functions
# =============================================================================

class MockTool:
    """A mock tool for testing tool-using agents."""

    def __init__(self, name: str, return_value: Any = "Tool result"):
        self.name = name
        self.return_value = return_value
        self.call_count = 0
        self.call_history = []

    def __call__(self, *args, **kwargs) -> Any:
        """Execute the tool."""
        self.call_count += 1
        self.call_history.append({"args": args, "kwargs": kwargs})
        return self.return_value


def create_mock_tools(num_tools: int = 3) -> List[MockTool]:
    """Create a list of mock tools."""
    return [
        MockTool(name=f"tool_{i}", return_value=f"tool_{i}_result")
        for i in range(num_tools)
    ]


# =============================================================================
# Utility Functions
# =============================================================================

def create_agent_config_for_testing(
    name: str = "TestAgent",
    agent_type: str = "simple",
    enable_validation: bool = False,
    dependencies: List[str] = None,
    **kwargs
):
    """
    Create an AgentConfig for testing.

    Parameters:
    -----------
    name : str
        Agent name
    agent_type : str
        Type of mock agent to create
    enable_validation : bool
        Whether to enable validation
    dependencies : List[str]
        Agent dependencies
    **kwargs : Any
        Additional AgentConfig parameters

    Returns:
    --------
    AgentConfig instance
    """
    try:
        from core.agent_config import AgentConfig
    except ImportError:
        raise ImportError("Cannot import AgentConfig - ensure JOTTY is available")

    agent = create_mock_agent(agent_type)

    return AgentConfig(
        name=name,
        agent=agent,
        architect_prompts=kwargs.get("architect_prompts", ["test_architect.md"]),
        auditor_prompts=kwargs.get("auditor_prompts", ["test_auditor.md"]),
        enable_architect=enable_validation,
        enable_auditor=enable_validation,
        dependencies=dependencies or [],
        **{k: v for k, v in kwargs.items() if k not in ["architect_prompts", "auditor_prompts"]}
    )
