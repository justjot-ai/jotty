# JOTTY Testing Suite

Comprehensive test suite for the JOTTY multi-agent orchestration framework.

## Quick Start

### Install Testing Dependencies

```bash
pip install -r requirements-test.txt
```

### Run All Tests

```bash
pytest tests/
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest tests/unit/ -m unit

# Integration tests only
pytest tests/integration/ -m integration

# System tests only
pytest tests/system/ -m system

# Performance tests only
pytest tests/performance/ -m performance
```

### Run with Coverage

```bash
pytest --cov=core --cov-report=html tests/
```

View coverage report in `htmlcov/index.html`

### Run in Parallel

```bash
pytest -n auto tests/
```

## Test Structure

```
tests/
├── unit/                    # Unit tests for individual components
│   ├── test_conductor.py
│   ├── test_agent_config.py
│   ├── test_cortex.py
│   ├── test_axon.py
│   └── ...
├── integration/             # Integration tests for multiple components
│   ├── test_agent_orchestration.py
│   ├── test_validation_flow.py
│   └── ...
├── system/                  # End-to-end system tests
│   ├── test_workflows.py
│   ├── test_configuration_modes.py
│   └── ...
├── performance/             # Performance and scalability tests
│   ├── test_scalability.py
│   └── ...
├── compatibility/           # Compatibility tests
│   ├── test_dspy_integration.py
│   └── ...
├── fixtures/                # Shared test fixtures and utilities
│   ├── sample_agents.py
│   ├── sample_configs.py
│   └── mock_data.py
├── conftest.py             # Pytest configuration and fixtures
└── README.md               # This file
```

## Test Markers

Use markers to categorize and filter tests:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.system` - System tests
- `@pytest.mark.performance` - Performance tests
- `@pytest.mark.slow` - Slow tests (> 5 seconds)
- `@pytest.mark.asyncio` - Async tests
- `@pytest.mark.requires_llm` - Tests requiring LLM API access

### Example Usage

```python
@pytest.mark.unit
@pytest.mark.asyncio
async def test_my_feature():
    # Test implementation
    pass
```

### Running Tests by Marker

```bash
# Run only unit tests
pytest -m unit

# Run only fast tests (exclude slow)
pytest -m "not slow"

# Run unit and integration tests
pytest -m "unit or integration"
```

## Writing Tests

### Basic Test Structure

```python
import pytest

class TestMyComponent:
    """Test suite for MyComponent."""

    @pytest.mark.unit
    def test_basic_functionality(self):
        """Test basic functionality."""
        # Arrange
        component = MyComponent()

        # Act
        result = component.do_something()

        # Assert
        assert result == expected_value
```

### Using Fixtures

```python
@pytest.mark.unit
def test_with_fixture(simple_agent_config):
    """Test using a fixture."""
    assert simple_agent_config.name == "TestAgent"
```

### Async Tests

```python
@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_operation(conductor_instance):
    """Test async operation."""
    result = await conductor_instance.run(goal="Test")
    assert result is not None
```

### Mocking

```python
from unittest.mock import Mock, patch

@pytest.mark.unit
def test_with_mock():
    """Test with mocking."""
    with patch('module.function') as mock_func:
        mock_func.return_value = "mocked value"
        result = function_under_test()
        assert result == "expected"
```

## Available Fixtures

See `conftest.py` for all available fixtures:

### Configuration Fixtures
- `default_jotty_config` - Default JOTTY configuration
- `minimal_jotty_config` - Minimal config for fast tests

### Agent Fixtures
- `mock_dspy_agent` - Simple mock DSPy agent
- `mock_async_agent` - Async mock agent
- `simple_agent_config` - Basic AgentConfig
- `multi_agent_configs` - Multiple AgentConfig instances

### Component Fixtures
- `mock_memory` - Mock SwarmMemory
- `mock_roadmap` - Mock MarkovianTODO
- `conductor_instance` - Conductor instance

### Utility Fixtures
- `temp_dir` - Temporary directory for test outputs
- `sample_context` - Sample context data

## Test Data

Use fixtures from `tests/fixtures/` for test data:

```python
from tests.fixtures.sample_agents import create_mock_agent

def test_with_sample_agent():
    agent = create_mock_agent(agent_type="simple")
    result = agent.forward(query="test")
    assert result is not None
```

## Continuous Integration

Tests should be run in CI/CD pipeline:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pip install -r requirements-test.txt
    pytest --cov=core --cov-report=xml tests/

- name: Upload coverage
  uses: codecov/codecov-action@v3
```

## Debugging Tests

### Run Single Test

```bash
pytest tests/unit/test_conductor.py::TestConductorInitialization::test_conductor_init_with_single_agent
```

### Verbose Output

```bash
pytest -vv tests/
```

### Show Print Statements

```bash
pytest -s tests/
```

### Drop into Debugger on Failure

```bash
pytest --pdb tests/
```

### Show Locals on Failure

```bash
pytest -l tests/
```

## Performance Testing

Run performance benchmarks:

```bash
pytest tests/performance/ --benchmark-only
```

Generate benchmark report:

```bash
pytest tests/performance/ --benchmark-autosave
```

## Code Coverage Goals

- **Unit tests**: > 90% coverage
- **Integration tests**: > 80% coverage
- **Overall**: > 85% coverage

Check current coverage:

```bash
pytest --cov=core --cov-report=term-missing tests/
```

## Best Practices

1. **Test Naming**: Use descriptive names that explain what is being tested
2. **Test Organization**: Group related tests in classes
3. **One Assertion Per Test**: Focus each test on a single behavior
4. **Use Fixtures**: Reuse common setup code via fixtures
5. **Mock External Dependencies**: Don't rely on external services in tests
6. **Fast Tests**: Keep unit tests fast (< 1 second each)
7. **Isolated Tests**: Each test should be independent
8. **Clear Assertions**: Use descriptive assertion messages

## Troubleshooting

### Tests Not Found

Ensure test files follow naming convention:
- `test_*.py` or `*_test.py`
- Classes: `Test*`
- Functions: `test_*`

### Import Errors

Ensure JOTTY is in Python path:
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/Jotty"
```

### Async Test Warnings

Ensure `pytest-asyncio` is installed and tests are marked:
```python
@pytest.mark.asyncio
async def test_async():
    pass
```

## Contributing

When adding new features:
1. Write tests first (TDD approach)
2. Ensure all tests pass
3. Maintain coverage above thresholds
4. Document complex test scenarios

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest Async](https://pytest-asyncio.readthedocs.io/)
- [Python Testing Best Practices](https://realpython.com/pytest-python-testing/)
