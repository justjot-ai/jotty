# Contributing to Jotty

Thank you for your interest in contributing to Jotty! This guide will help you get started.

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/your-org/jotty.git
cd jotty

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-test.txt

# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run tests
pytest tests/ -v

# Run health check
python scripts/jotty_doctor.py
```

## 📋 Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

### 2. Make Changes

- Write clear, concise code
- Add type hints to all functions
- Include docstrings for public APIs
- Follow existing code style

### 3. Add Tests

```bash
# Create test file
touch tests/test_your_feature.py

# Write tests (see Test Patterns below)
# Run tests
pytest tests/test_your_feature.py -v
```

### 4. Run Quality Checks

```bash
# Format code
black Jotty/ --line-length=100
isort Jotty/ --profile=black

# Lint
flake8 Jotty/ --max-line-length=100

# Type check
mypy Jotty/core/swarms --ignore-missing-imports

# Security scan
bandit -r Jotty/

# Full health check
python scripts/jotty_doctor.py
```

### 5. Commit

```bash
git add .
git commit -m "feat: add new feature

- Detailed description
- Why this change is needed
- Any breaking changes"
```

**If pre-commit blocks your commit (black, flake8, import-linter, etc.):**

- Hooks with `allow_failure: true` (black, flake8, import-linter, jotty-doctor-imports) will **not** block the commit; they only report. Your commit will succeed.
- If **trailing-whitespace** or **fix end of files** modified files: run `git add -A` and commit again (they fixed the files on disk).
- If **black** or **isort** reformatted files: run `git add -A` and commit again.
- If something else blocks: run `black .` and `isort .`, then `git add -A` and commit again.
- To skip hooks once (emergency): `git commit --no-verify -m "your message"`. Prefer fixing the reported issues when you can.

### 6. Push and Create PR

```bash
git push origin feature/your-feature-name
# Then create Pull Request on GitHub
```

## 🎯 Code Standards

### Type Hints

**Required** for all functions:

```python
# ❌ Bad
def process_data(data):
    return data.upper()

# ✅ Good
def process_data(data: str) -> str:
    """Process input data."""
    return data.upper()
```

### Error Handling

Use specific exceptions with helpful messages:

```python
from Jotty.core.foundation.helpful_errors import InvalidConfigValueError

# ❌ Bad
if timeout <= 0:
    raise ValueError("Invalid timeout")

# ✅ Good
if timeout <= 0:
    raise InvalidConfigValueError(
        field="timeout",
        value=timeout,
        expected="> 0"
    )
```

### Docstrings

```python
def my_function(param1: str, param2: int) -> bool:
    """Short description of what this function does.

    Longer description if needed, explaining the purpose,
    algorithm, or important details.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When param2 is negative
    """
    pass
```

## 🧪 Test Patterns

### Unit Test Template

```python
import pytest
from Jotty.core.swarms import MySwarm

class TestMySwarm:
    @pytest.mark.unit
    def test_basic_functionality(self):
        """Test basic swarm functionality."""
        swarm = MySwarm()
        result = swarm.some_method()
        assert result.success is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_execution(self):
        """Test async swarm execution."""
        swarm = MySwarm()
        result = await swarm.execute("test task")
        assert result is not None
```

### Test Requirements

- **Coverage**: Aim for 80%+ coverage
- **Mocks**: Use mocks for LLM calls (no real API calls in tests)
- **Fast**: Tests should complete in < 1s each
- **Isolated**: Each test should be independent

## 🏗️ Adding a New Swarm

### 1. Create Directory Structure

```bash
mkdir -p Jotty/core/swarms/my_swarm
cd Jotty/core/swarms/my_swarm

# Create files
touch __init__.py types.py agents.py swarm.py README.md
```

### 2. Use Template

```python
# types.py
from dataclasses import dataclass
from ..swarm_types import SwarmConfig, SwarmResult

@dataclass
class MySwarmConfig(SwarmConfig):
    """Configuration for MySwarm."""
    custom_param: str = "default"

    def __post_init__(self) -> None:
        self.name = "MySwarm"
        self.domain = "my_domain"

@dataclass
class MySwarmResult(SwarmResult):
    """Result from MySwarm."""
    custom_output: str = ""
```

```python
# swarm.py
from ..base.domain_swarm import DomainSwarm
from .types import MySwarmConfig, MySwarmResult

class MySwarm(DomainSwarm):
    """MySwarm description."""

    def __init__(self, config: MySwarmConfig = None):
        super().__init__(config or MySwarmConfig())
        self._define_agents([
            # Your agents here
        ])

    async def execute(self, task: str) -> MySwarmResult:
        """Execute swarm task."""
        # Your logic here
        return MySwarmResult(
            success=True,
            swarm_name=self.config.name,
            domain=self.config.domain,
            output={"result": "success"},
            execution_time=0.0
        )
```

### 3. Add Documentation

Create `README.md` with:
- Purpose
- Quick start example
- Configuration options
- Use cases
- Examples

See `olympiad_learning_swarm/README.md` for template.

### 4. Add Tests

```python
# tests/test_my_swarm.py
import pytest
from Jotty.core.swarms.my_swarm import MySwarm

class TestMySwarm:
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_execution(self):
        swarm = MySwarm()
        result = await swarm.execute("test")
        assert result.success is True
```

### 5. Register in CLAUDE.md

Add to task→swarm mapping table.

## 📝 Commit Message Format

```
<type>: <short summary>

<longer description>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding/updating tests
- `chore`: Maintenance tasks

### Examples

```
feat: add jotty discover command

Created jotty_discover.py to help users find the right
swarm for their task. Includes keyword matching and
relevance scoring.

Closes #123
```

```
fix: resolve timeout in olympiad learning swarm

Increased NarrativeEditor timeout from 120s to 240s to
handle large content (37K+ chars).

Fixes #456
```

## 🐛 Reporting Bugs

### Before Submitting

1. Check existing issues
2. Run `python scripts/jotty_doctor.py`
3. Collect logs and error messages

### Bug Report Template

```markdown
**Description**
Clear description of the bug

**To Reproduce**
1. Step 1
2. Step 2
3. See error

**Expected Behavior**
What you expected to happen

**Environment**
- OS: [e.g., Ubuntu 22.04]
- Python version: [e.g., 3.11.2]
- Jotty version: [e.g., 1.0.0]

**Logs**
```
Paste relevant logs here
```
```

## 💡 Feature Requests

Use GitHub Issues with:
- Clear use case
- Expected behavior
- Why it's valuable
- Possible implementation approach

## 📚 Resources

- **Architecture**: `docs/JOTTY_ARCHITECTURE.md`
- **Happy Path**: `docs/HAPPY_PATH_GUIDE.md` - Complete flow from Telegram to response
- **Error Handling**: `docs/ERROR_HANDLING_GUIDE.md`
- **Quick Reference**: `Jotty/CLAUDE.md`
- **Progress Tracker**: `PROGRESS_TO_10.md`

## ❓ Questions

- GitHub Discussions
- Issues with `question` label

## 📄 License

By contributing, you agree that your contributions will be licensed
under the same license as the project.

---

**Thank you for making Jotty better!** 🎉
