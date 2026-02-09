"""
Pytest configuration and shared fixtures for JOTTY tests.

This module provides common fixtures and setup/teardown logic
for all test modules in the JOTTY framework.
"""
import os
import sys
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, MagicMock, AsyncMock
import pytest

# Exclude V1 archived tests from collection
collect_ignore_glob = ["_v1_archive/*"]

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import JOTTY components
try:
    from core.foundation.data_structures import SwarmConfig, JottyConfig, MemoryLevel, EpisodeResult
    from core.foundation.agent_config import AgentSpec, AgentConfig
    from core.orchestration.v2 import SwarmManager as Conductor  # V2 replacement
    from core.memory.cortex import HierarchicalMemory
    from core.orchestration.v2 import SwarmTaskBoard as MarkovianTODO  # V2 replacement
    JOTTY_AVAILABLE = True
except ImportError as e:
    print(f"JOTTY import failed: {e}")
    JOTTY_AVAILABLE = False

# Import DSPy if available
try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False


# =============================================================================
# Session-level Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for test outputs."""
    temp_path = tempfile.mkdtemp(prefix="jotty_test_")
    yield Path(temp_path)
    shutil.rmtree(temp_path, ignore_errors=True)


# =============================================================================
# Configuration Fixtures
# =============================================================================

@pytest.fixture
def default_jotty_config():
    """Provide default JOTTY configuration."""
    if not JOTTY_AVAILABLE:
        pytest.skip("JOTTY not available")

    return SwarmConfig(
        output_base_dir="./test_outputs",
        create_run_folder=False,
        enable_beautified_logs=False,
        log_level="WARNING",  # Reduce noise in tests
        max_actor_iters=10,
        max_episode_iterations=3,
        max_eval_iters=3,
        llm_timeout_seconds=30.0,
        actor_timeout=60.0,
        enable_validation=False,  # Disable for faster tests
        episodic_capacity=100,
        semantic_capacity=50,
        consolidation_interval=10,
    )


@pytest.fixture
def minimal_jotty_config():
    """Provide minimal JOTTY configuration for fast tests."""
    if not JOTTY_AVAILABLE:
        pytest.skip("JOTTY not available")

    return SwarmConfig(
        output_base_dir="./test_outputs",
        create_run_folder=False,
        enable_beautified_logs=False,
        log_level="ERROR",
        max_actor_iters=3,
        max_episode_iterations=1,
        max_eval_iters=1,
        llm_timeout_seconds=10.0,
        actor_timeout=20.0,
        enable_validation=False,
        episodic_capacity=10,
        semantic_capacity=5,
    )


# =============================================================================
# Mock Agent Fixtures
# =============================================================================

@pytest.fixture
def mock_dspy_agent():
    """Create a mock DSPy agent."""
    mock_agent = Mock()
    mock_agent.forward = Mock(return_value=Mock(answer="Test output"))
    mock_agent.__call__ = Mock(return_value=Mock(answer="Test output"))
    return mock_agent


@pytest.fixture
def mock_async_agent():
    """Create a mock async DSPy agent."""
    mock_agent = AsyncMock()
    mock_agent.forward = AsyncMock(return_value=Mock(answer="Test output"))
    mock_agent.__call__ = AsyncMock(return_value=Mock(answer="Test output"))
    return mock_agent


@pytest.fixture
def simple_agent_config(mock_dspy_agent):
    """Create a simple AgentSpec for testing."""
    if not JOTTY_AVAILABLE:
        pytest.skip("JOTTY not available")

    return AgentSpec(
        name="TestAgent",
        agent=mock_dspy_agent,
        architect_prompts=["test_architect.md"],
        auditor_prompts=["test_auditor.md"],
        enable_architect=False,
        enable_auditor=False,
    )


@pytest.fixture
def multi_agent_configs(mock_dspy_agent):
    """Create multiple AgentConfig instances for pipeline testing."""
    if not JOTTY_AVAILABLE:
        pytest.skip("JOTTY not available")

    return [
        AgentSpec(
            name="Agent1",
            agent=mock_dspy_agent,
            architect_prompts=["architect1.md"],
            auditor_prompts=["auditor1.md"],
            enable_architect=False,
            enable_auditor=False,
        ),
        AgentSpec(
            name="Agent2",
            agent=mock_dspy_agent,
            architect_prompts=["architect2.md"],
            auditor_prompts=["auditor2.md"],
            dependencies=["Agent1"],
            enable_architect=False,
            enable_auditor=False,
        ),
        AgentSpec(
            name="Agent3",
            agent=mock_dspy_agent,
            architect_prompts=["architect3.md"],
            auditor_prompts=["auditor3.md"],
            dependencies=["Agent2"],
            enable_architect=False,
            enable_auditor=False,
        ),
    ]


# =============================================================================
# Component Fixtures
# =============================================================================

@pytest.fixture
def mock_memory():
    """Create a mock HierarchicalMemory instance."""
    memory = Mock(spec=HierarchicalMemory)
    memory.store = Mock()
    memory.retrieve = Mock(return_value=[])
    memory.consolidate = Mock()
    return memory


@pytest.fixture
def mock_roadmap():
    """Create a mock MarkovianTODO instance."""
    roadmap = Mock(spec=MarkovianTODO)
    roadmap.add_item = Mock()
    roadmap.get_next_item = Mock(return_value=None)
    roadmap.mark_complete = Mock()
    return roadmap


@pytest.fixture
async def conductor_instance(simple_agent_config, minimal_jotty_config):
    """Create a Conductor instance for testing."""
    if not JOTTY_AVAILABLE:
        pytest.skip("JOTTY not available")

    conductor = Conductor(
        actors=[simple_agent_config],
        config=minimal_jotty_config,
    )

    yield conductor

    # Cleanup
    if hasattr(conductor, 'cleanup'):
        await conductor.cleanup()


# =============================================================================
# Mock LLM Fixtures
# =============================================================================

@pytest.fixture
def mock_llm_response():
    """Create a mock LLM response."""
    return Mock(
        content="This is a test response",
        usage=Mock(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15
        )
    )


@pytest.fixture
def mock_dspy_lm():
    """Create a mock DSPy language model."""
    if not DSPY_AVAILABLE:
        pytest.skip("DSPy not available")

    mock_lm = Mock()
    mock_lm.__call__ = Mock(return_value=["Test response"])
    return mock_lm


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def sample_context():
    """Provide sample context data for testing."""
    return {
        "query": "What is the total count of transactions?",
        "current_date": "2024-01-15",
        "user_id": "test_user_123",
    }


@pytest.fixture
def sample_episode_result():
    """Provide a sample EpisodeResult for testing."""
    if not JOTTY_AVAILABLE:
        pytest.skip("JOTTY not available")

    return EpisodeResult(
        episode_num=1,
        total_reward=1.0,
        steps=5,
        success=True,
        final_output="Test output",
        trajectory=[],
    )


# =============================================================================
# Utility Functions
# =============================================================================

@pytest.fixture
def assert_async():
    """Helper for asserting async operations."""
    async def _assert_async(coro, expected=None, error=None):
        """Execute coroutine and assert result or error."""
        if error:
            with pytest.raises(error):
                await coro
        else:
            result = await coro
            if expected is not None:
                assert result == expected
            return result
    return _assert_async


# =============================================================================
# Auto-use Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset any singleton instances between tests."""
    # Add logic to reset singletons if needed
    yield


@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Clean up temporary files after each test."""
    yield
    # Cleanup logic
    test_output_dir = Path("./test_outputs")
    if test_output_dir.exists():
        shutil.rmtree(test_output_dir, ignore_errors=True)


# =============================================================================
# Markers and Skip Conditions
# =============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "requires_dspy: mark test as requiring DSPy"
    )
    config.addinivalue_line(
        "markers", "requires_llm: mark test as requiring LLM API access"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to apply skip markers."""
    skip_dspy = pytest.mark.skip(reason="DSPy not available")
    skip_jotty = pytest.mark.skip(reason="JOTTY not available")

    for item in items:
        if "requires_dspy" in item.keywords and not DSPY_AVAILABLE:
            item.add_marker(skip_dspy)
        if not JOTTY_AVAILABLE:
            item.add_marker(skip_jotty)
