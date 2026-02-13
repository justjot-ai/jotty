# Jotty - AI Agent Framework

## Quick Reference for Claude

**Main Architecture Doc:** `docs/JOTTY_ARCHITECTURE.md` - READ THIS FIRST

## Testing Requirements

**MANDATORY**: Every code change to Jotty MUST include corresponding tests.

### Rules
1. Every new method/class gets a unit test
2. Every bug fix gets a regression test proving the fix
3. Tests use mocks — NEVER call real LLM providers
4. Tests run fast (< 1s each) and offline
5. Run `pytest tests/test_v3_execution.py -v` before considering work done

### V3 Test Patterns
- Use the `v3_executor` fixture from conftest.py (pre-wired with all mocks)
- Use `v3_observability_helpers` fixture for `assert_metrics_recorded()`, `assert_trace_exists()`, `assert_cost_tracked()` helpers
- Class-based: `class TestMyFeature:` with `@pytest.mark.unit` + `@pytest.mark.asyncio`
- Mock provider returns: `{'content': '...', 'usage': {'input_tokens': N, 'output_tokens': N}}`

### Running Tests
```bash
pytest tests/test_v3_execution.py -v        # V3 tests
pytest tests/ -m unit                       # All unit tests
pytest tests/ -m "not requires_llm"         # All offline tests
```

## The Five Layers (Top to Bottom)

```
1. INTERFACE    Telegram | Slack | Discord | WhatsApp | Web | CLI | SDK
       ↓
2. MODES        Chat (ChatAssistant) | API (MCP) | Workflow (AutoAgent)
       ↓
3. REGISTRY     UnifiedRegistry → Skills (Hands) + UI (Eyes) + Memory
       ↓
4. BRAIN        Swarms → Agents → SwarmIntelligence → TD-Lambda
       ↓
5. PERSISTENCE  ~/jotty/intelligence/*.json | ~/jotty/skills/
```

## Key Entry Points

| Entry Point | Command | Purpose |
|-------------|---------|---------|
| **CLI Interactive** | `python -m Jotty.cli` | REPL with slash commands |
| **CLI Single** | `python -m Jotty.cli -c "task"` | One-off execution |
| **Web Gateway** | `python Jotty/web.py` | HTTP/WS server (port 8766) |
| **Gateway Only** | `python -m Jotty.cli.gateway` | Webhooks for Telegram/Slack/etc |

## Imports

```python
from Jotty.core.registry import get_unified_registry
from Jotty.core.agents import AutoAgent, ChatAssistant
from Jotty.core.swarms import BaseSwarm, CodingSwarm
from Jotty.core.learning import TDLambdaLearner
from Jotty.core.api import JottyAPI
from Jotty.core.foundation.exceptions import JottyError
from Jotty.core.foundation.data_structures import SwarmConfig
from Jotty.core.memory.cortex import SwarmMemory
```

## Directory Structure

```
Jotty/
├── cli/                    # CLI application
│   ├── app.py              # JottyCLI main class
│   ├── gateway/            # UnifiedGateway + ChannelRouter
│   │   ├── server.py       # FastAPI + WebSocket server
│   │   ├── channels.py     # ChannelRouter + ChannelType enum
│   │   └── trust.py        # TrustManager (auth)
│   ├── commands/           # Slash commands (/run, /swarm, etc.)
│   ├── config/             # CLI configuration
│   │   └── schema.py       # CLIConfig dataclass
│   └── repl/               # REPL engine
├── core/
│   ├── agents/             # Agent implementations
│   │   ├── base/           # BaseAgent, DomainAgent, MetaAgent, AutonomousAgent
│   │   ├── auto_agent.py   # AutoAgent (workflow mode)
│   │   └── chat_assistant.py
│   ├── swarms/             # Swarm implementations
│   │   ├── base_swarm.py   # BaseSwarm with learning hooks
│   │   ├── domain_swarm.py # DomainSwarm (AgentTeam)
│   │   └── specialized/    # CodingSwarm, TestingSwarm, etc.
│   ├── registry/           # The unified registry system
│   │   ├── unified_registry.py  # MAIN ENTRY: get_unified_registry()
│   │   ├── skills_registry.py   # Skills (Hands) - 126 skills
│   │   ├── ui_registry.py       # UI (Eyes) - 16 components
│   │   └── api.py               # Registry HTTP API
│   ├── api/                # Programmatic API layer
│   │   ├── unified.py      # JottyAPI (main entry)
│   │   ├── chat_api.py     # ChatAPI
│   │   ├── workflow_api.py # WorkflowAPI
│   │   └── openapi.py      # OpenAPI 3.0 spec generator
│   ├── foundation/         # Cross-cutting concerns
│   │   ├── exceptions.py   # 30+ exception types
│   │   ├── data_structures.py  # SwarmConfig, etc.
│   │   └── agent_config.py # AgentConfig
│   ├── learning/           # TD-Lambda, memory systems
│   ├── memory/             # SwarmMemory (5 levels)
│   ├── orchestration/      # Orchestrator, SwarmIntelligence
│   └── integration/        # MCP, Claude CLI LM
├── skills/                 # Skill definitions (loaded lazily)
├── sdk/                    # Generated client libraries
├── web.py                  # Web server entry point
└── docs/                   # Documentation
    └── JOTTY_ARCHITECTURE.md  # MAIN ARCHITECTURE DOC
```

## How Learning Works

1. **Pre-execution**: `BaseSwarm._pre_execute_learning()` loads context from memory
2. **Execution**: Swarm runs agents with skills
3. **Post-execution**: `BaseSwarm._post_execute_learning()` stores to memory, updates TD-Lambda
4. **Persistence**: `SwarmIntelligence.save()` writes to `~/jotty/intelligence/{swarm}_{domain}.json`
5. **Next run**: Learning auto-loads from disk

## Adding New Components

### New Skill
```python
# skills/my-skill/skill.yaml
name: my-skill
description: "What this skill does"
tools:
  - my_tool

# skills/my-skill/tools.py
def my_tool(params: dict) -> dict:
    return {"result": "..."}
```

### New Swarm
```python
class MySwarm(DomainSwarm):
    def __init__(self):
        super().__init__(SwarmConfig(name="MySwarm", domain="my-domain"))
        self._define_agents([...])
```

### New Agent
```python
class MyAgent(DomainAgent):
    def __init__(self):
        super().__init__(signature=MySignature, config=DomainAgentConfig(name="MyAgent"))
```

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `ANTHROPIC_API_KEY` | Claude API access |
| `OPENAI_API_KEY` | OpenAI access |
| `GROQ_API_KEY` | Groq access (free tier) |
| `TELEGRAM_TOKEN` | Telegram bot token |
| `SLACK_SIGNING_SECRET` | Slack webhook verification |
| `DISCORD_PUBLIC_KEY` | Discord verification |
| `WHATSAPP_VERIFY_TOKEN` | WhatsApp webhook (default: "jotty") |

## Testing

```bash
# Run architecture verification
python3 docs/scripts/verify_architecture.py

# Test specific swarm
python3 -c "
import asyncio
from Jotty.core.swarms import CodingSwarm
swarm = CodingSwarm()
result = asyncio.run(swarm.execute('Test task'))
print(result)
"
```

## Common Patterns

### Get all skills
```python
registry = get_unified_registry()
skills = registry.list_skills()  # 126 skills
```

### Discover for task
```python
discovery = registry.discover_for_task("create a chart")
# Returns: {'skills': [...], 'ui': [...]}
```

### Convert to Claude tools
```python
claude_tools = registry.get_claude_tools(['web-search', 'calculator'])
```

### Run workflow
```python
agent = AutoAgent()
result = await agent.execute("Research X, create report, send via telegram")
```

## Important Files

| File | Purpose |
|------|---------|
| `core/registry/unified_registry.py` | Single entry point for all capabilities |
| `core/swarms/base_swarm.py` | Learning hooks (_pre/_post_execute_learning) |
| `core/orchestration/swarm_intelligence.py` | Learning state management |
| `cli/gateway/server.py` | UnifiedGateway (all webhooks) |
| `cli/gateway/channels.py` | ChannelRouter (message routing) |
| `cli/app.py` | JottyCLI (main CLI application) |

## What's NOT in This Codebase

- Frontend UI code is in separate `Jotty/ui/` Next.js app
- Telegram bot standalone is `telegram_bot/` (but gateway handles webhooks)
- JustJot.ai integration is in `common/justjot/`
