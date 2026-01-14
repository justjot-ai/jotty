# Jotty AI Package Setup Guide

## Quick Start

### Installation

```bash
# Basic installation
pip install jotty-ai

# With optional dependencies
pip install jotty-ai[all]
# or specific extras
pip install jotty-ai[mongodb]
pip install jotty-ai[redis]
pip install jotty-ai[sql]

# Development installation
pip install -e ".[dev]"
```

### Usage

```python
from Jotty import Conductor, AgentConfig

# Create conductor with your agents
swarm = Conductor(
    actors=[
        AgentConfig(
            name="my_agent",
            agent=my_dspy_module,
            architect_prompts=["plan.md"],
            auditor_prompts=["validate.md"]
        )
    ],
)

# Run task
result = await swarm.run("Process this")
```

## Building the Package

### Prerequisites

```bash
pip install build twine
```

### Build

```bash
# Build source and wheel distributions
python -m build

# Outputs will be in dist/
# - jotty-ai-10.0.0.tar.gz (source distribution)
# - jotty_ai-10.0.0-py3-none-any.whl (wheel)
```

### Test Installation Locally

```bash
# Install from local build
pip install dist/jotty_ai-10.0.0-py3-none-any.whl

# Or install in development mode
pip install -e .
```

## Publishing to PyPI

### TestPyPI (for testing)

```bash
# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ jotty-ai
```

### Production PyPI

```bash
# Upload to PyPI (requires credentials)
python -m twine upload dist/*

# Or use environment variables
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-xxxxx  # Your PyPI API token
python -m twine upload dist/*
```

## Version Management

### Update Version

1. Update `__version__` in `Jotty/__init__.py`
2. Update `version` in `pyproject.toml`
3. Update `version` in `setup.py`
4. Commit changes
5. Tag release: `git tag v10.0.0`
6. Build and publish

### Semantic Versioning

- **MAJOR** (X.0.0): Breaking changes
- **MINOR** (0.X.0): New features, backward compatible
- **PATCH** (0.0.X): Bug fixes, backward compatible

## Directory Structure

```
Jotty/
├── pyproject.toml          # Modern package config
├── setup.py               # Fallback setup script
├── MANIFEST.in            # Data files to include
├── requirements.txt       # Runtime dependencies
├── requirements-dev.txt   # Dev dependencies
├── README.md              # Package documentation
├── LICENSE                # License file
├── CHANGELOG.md           # Version history
├── Jotty/                 # Main package
│   ├── __init__.py        # Package exports
│   └── core/              # Core modules
│       ├── orchestration/
│       ├── memory/
│       ├── learning/
│       └── ...
├── tests/                  # Test suite
├── examples/               # Example scripts
└── docs/                   # Documentation
```

## Data Files

Prompt files and other data are included via `MANIFEST.in`:

- `Jotty/core/swarm_prompts/*.md`
- `Jotty/core/validation_prompts/*.md`

These are accessible at runtime via:

```python
import importlib.resources

# Python 3.9+
with importlib.resources.path("Jotty.core.swarm_prompts", "architect_orchestration.md") as p:
    prompt_path = p

# Or read directly
prompt_text = importlib.resources.read_text("Jotty.core.swarm_prompts", "architect_orchestration.md")
```

## Integration with JustJot.ai

### Option 1: Use pip-installed package

```python
# In JustJot.ai requirements.txt
jotty-ai>=10.0.0

# Import as usual
from Jotty import Conductor, AgentConfig
```

### Option 2: Keep local development

```python
# Install in editable mode from local path
pip install -e /path/to/Jotty

# Or add to PYTHONPATH
export PYTHONPATH=/path/to/Jotty:$PYTHONPATH
```

### Option 3: Hybrid approach

```python
# Try pip-installed first, fallback to local
try:
    from Jotty import Conductor
except ImportError:
    import sys
    sys.path.insert(0, '/path/to/local/Jotty')
    from Jotty import Conductor
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Build and Publish

on:
  release:
    types: [created]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - run: pip install build twine
      - run: python -m build
      - run: python -m twine upload dist/*
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
```

## Troubleshooting

### Import Errors

If imports fail after installation:

1. Verify package installed: `pip show jotty-ai`
2. Check Python path: `python -c "import sys; print(sys.path)"`
3. Verify package structure: `python -c "import Jotty; print(Jotty.__file__)"`

### Data Files Not Found

If prompt files aren't accessible:

1. Verify `MANIFEST.in` includes the files
2. Check `package_data` in `setup.py` or `pyproject.toml`
3. Rebuild package: `python -m build --wheel`
4. Use `importlib.resources` for access

### Version Conflicts

If dependencies conflict:

1. Check installed versions: `pip list`
2. Use virtual environment: `python -m venv venv`
3. Pin specific versions in `requirements.txt`
4. Use extras for optional dependencies

## Next Steps

1. ✅ Package structure configured
2. ✅ Dependencies defined
3. ⏳ Test installation locally
4. ⏳ Run test suite
5. ⏳ Update documentation
6. ⏳ Publish to TestPyPI
7. ⏳ Publish to PyPI
