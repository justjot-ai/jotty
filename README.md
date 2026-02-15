# Jotty - AI Agent Framework

A brain-inspired, LLM-first multi-agent framework with hierarchical memory and reinforcement learning.

## Architecture Overview

```
USER INTERFACE     Telegram | Slack | Discord | WhatsApp | Web | CLI | SDK
       ↓
COMMAND LAYER      CommandRegistry → run | export | resume | justjot
       ↓
ORCHESTRATION V2   Orchestrator → AgentRunner → LeanExecutor
       ↓
AGENT LAYER        TaskPlanner → AutoAgent → ValidatorAgent
       ↓
MEMORY SYSTEM      5-Level MemoryCortex (Episodic → Semantic → Procedural → Meta → Causal)
       ↓
LEARNING SYSTEM    TD(λ) with Eligibility Traces + Adaptive Learning Rate
```

## Key Features

- **LLM-First Design**: All decisions via LLM, no hardcoded rules
- **Brain-Inspired Memory**: 5-level hierarchy with sleep consolidation
- **TD(λ) Learning**: Temporal difference with adaptive rates
- **273 Skills**: Modular, discoverable capabilities
- **Multi-Provider**: Auto-detection chain (Claude CLI → Cursor → Anthropic → OpenRouter → OpenAI → Groq)

## Quick Start

### Installation

```bash
# Clone and install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
```

### Usage

```bash
# Interactive CLI
python -m Jotty.cli

# Single command execution
python -m Jotty.cli -c "Research AI trends and create a report"

# Web Gateway (all channels)
python Jotty/web.py
```

## System Components

### Orchestration Layer (V2)

| Component | Purpose |
|-----------|---------|
| `Orchestrator` | Central orchestrator for multi-agent coordination |
| `AgentRunner` | Per-agent execution with validation |
| `LeanExecutor` | LLM-first lean execution path |
| `SwarmInstaller` | Template-based swarm creation |

### Agent Layer

| Agent | Role |
|-------|------|
| `TaskPlanner` | LLM-based task inference and workflow planning |
| `AutoAgent` | Autonomous task execution with skill discovery |
| `ValidatorAgent` | Pre/post validation (Architect + Auditor) |

### Memory System (Brain-Inspired)

```
EPISODIC     →  Raw experiences, fast decay (hours)
SEMANTIC     →  Patterns & abstractions, slow decay (days)
PROCEDURAL   →  How-to knowledge, medium decay
META         →  Learning wisdom, no decay
CAUSAL       →  Why relationships, no decay
```

Features:
- LLM-based memory classification
- RAG retrieval with deduplication
- Sleep consolidation (SharpWaveRipple)
- Causal relationship extraction

### Learning System (TD-Lambda)

```
TD Error: δ = r + γV(s') - V(s)
Eligibility Traces: e(s) = γλe(s) + 1
Value Update: V(s) += α·δ·e(s)
```

Components:
- `LLMTrajectoryPredictor` - Predict action outcomes
- `DivergenceMemory` - Learn from prediction errors
- `CooperativeCreditAssigner` - Multi-agent reward distribution
- `ShapedRewardManager` - Per-step and cooperative rewards

### Skills System

273 skills organized by category:

| Category | Examples |
|----------|----------|
| **INPUT** | web-search, file-read, api-fetch |
| **OUTPUT** | docx-tools, telegram-sender, pdf-generator |
| **TRANSFORM** | document-converter, data-analyzer |

Skills are lazily loaded from:
- `/skills/` - Repository skills
- `~/.claude/skills/` - User custom skills

## Provider Auto-Detection

```
Claude CLI → Cursor CLI → Anthropic API → OpenRouter → OpenAI → Groq
    ↓             ↓             ↓              ↓           ↓        ↓
  (Free)      (Free)        (Paid)        (Multi)     (Paid)   (Fast)
```

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `ANTHROPIC_API_KEY` | Claude API access |
| `OPENAI_API_KEY` | OpenAI access |
| `GROQ_API_KEY` | Groq access (free tier) |
| `OPENROUTER_API_KEY` | OpenRouter multi-model |
| `TELEGRAM_TOKEN` | Telegram bot token |
| `SLACK_SIGNING_SECRET` | Slack webhook verification |
| `DISCORD_PUBLIC_KEY` | Discord verification |

## Directory Structure

```
Jotty/
├── cli/                    # CLI application
│   ├── app.py              # JottyCLI main class
│   ├── gateway/            # UnifiedGateway + ChannelRouter
│   ├── commands/           # Slash commands
│   └── config/             # CLI configuration
├── core/
│   ├── orchestration/v2/   # Orchestrator, AgentRunner, LeanExecutor
│   ├── agents/             # TaskPlanner, AutoAgent, ValidatorAgent
│   ├── memory/             # MemoryCortex (5-level)
│   ├── learning/           # TD-Lambda, credit assignment
│   ├── registry/           # UnifiedRegistry, Skills, UI
│   └── foundation/         # Config, exceptions, data structures
├── skills/                 # 273 skill definitions
├── layers/                 # Layer-based imports
├── docs/                   # Architecture documentation
│   ├── JOTTY_ARCHITECTURE.md
│   └── JOTTY_V2_ARCHITECTURE.md
└── web.py                  # Web server entry point
```

## Execution Flow

```
1. User Input → JottyCLI
2. Command Routing → CommandRegistry
3. Planning → TaskPlanner (task type inference)
4. Skill Discovery → SkillsRegistry
5. Orchestration → Orchestrator → AgentRunner
6. Pre-Validation → ValidatorAgent (Architect)
7. Execution → AutoAgent + Skills
8. Post-Validation → ValidatorAgent (Auditor)
9. Learning → TD(λ) update + Memory storage
10. Output → UIRenderer → User
```

## API Usage

```python
from Jotty.layers.interface import JottyCLI
from Jotty.layers.modes import AutoAgent
from Jotty.layers.registry import get_unified_registry

# Get unified registry
registry = get_unified_registry()
skills = registry.list_skills()  # 273 skills

# Discover for task
discovery = registry.discover_for_task("create a chart")

# Run workflow
agent = AutoAgent()
result = await agent.execute("Research X, create report, send via telegram")
```

## Testing

```bash
# Run tests
pytest tests/

# Verify architecture
python docs/scripts/verify_architecture.py
```

## Design Patterns

| Pattern | Component | Purpose |
|---------|-----------|---------|
| Registry | Skills, Templates, Commands | Central lookup |
| Strategy | Validators, Classifiers | Interchangeable algorithms |
| Adapter | UnifiedLMProvider | Abstract LLM APIs |
| State | BrainStateMachine | AWAKE/SLEEP transitions |
| Lazy Init | Skills, Orchestrator | Fast startup |

## Documentation

- [Full Architecture](docs/JOTTY_ARCHITECTURE.md)
- [V2 Architecture Diagrams](docs/JOTTY_V2_ARCHITECTURE.md)
- [Component Relationships](docs/ARCHITECTURE_INTERLINKED.md)

## License

MIT
