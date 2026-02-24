# Jotty

Jotty is a multi-agent AI framework with swarms, skills, memory/learning, and multiple interfaces (CLI, API, web, messaging). It provides a stable SDK, a layered core, and tooling for orchestrating autonomous workflows.

## What This Repo Contains

- `apps/` - Interface apps (API server, CLI, web, Telegram, WhatsApp)
- `sdk/` - Public SDK used by apps
- `core/` - Core framework (interface, capabilities, intelligence, infrastructure)
- `skills/` - Skill definitions loaded by the registry
- `docs/` - Architecture + subsystem docs
- `tests/` - Test suite and fixtures

## Architecture (High-Level)

Jotty follows a layered architecture to keep apps stable while the core evolves:

```
apps/  ->  sdk/  ->  core/interface/  ->  core/intelligence/
```

- Apps should import from the SDK.
- The SDK uses the internal API in `core/interface/`.
- Orchestration, learning, and memory live under `core/intelligence/`.

For details, see `docs/JOTTY_ARCHITECTURE.md`.

## Quick Start (Dev)

```bash
pip install -r requirements.txt
```

Common environment variables (see `CLAUDE.md` for the full list):

```bash
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"   # optional
```

Run a CLI session:

```bash
python -m Jotty.apps.cli
```

Run the web server (HTTP/WS gateway):

```bash
python web.py
```

## Usage Examples

### Chat + Workflow via SDK

```python
from Jotty import Jotty

j = Jotty()
response = await j.chat("Hello")
result = await j.run("Research AI trends and summarize")
```

### Swarm Orchestration

```python
from Jotty import Orchestrator

swarm = Orchestrator(agents="Researcher + Writer + Reviewer")
result = await swarm.run(goal="Write a 500-word article on quantum computing")
```

### Capability Discovery

```python
from Jotty import capabilities

caps = capabilities()
print(caps.keys())
```

## Documentation

- `docs/GETTING_STARTED.md`
- `docs/JOTTY_ARCHITECTURE.md`
- `docs/API_REFERENCE.md`
- `sdk/README.md`
- `tests/README.md`

## Counts (Auto-Derived)

Current counts (from repo tree):

- Skills: 198
- Swarms: 6

Refresh counts:

```bash
python scripts/count_capabilities.py
```

## Contributing & Tests

See `CONTRIBUTING.md` and `tests/README.md` for test instructions and markers.
