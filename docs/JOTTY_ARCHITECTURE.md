# Jotty System Architecture - Complete Guide

## The Complete Stack

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          JOTTY SYSTEM ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                      🌐 INTERFACE LAYER                             │   │
│   │            External Entry Points (All Channels)                     │   │
│   │                                                                     │   │
│   │   📱Telegram  💬Slack  🎮Discord  📲WhatsApp  🌐Web  💻CLI  📦SDK   │   │
│   │        ↓         ↓         ↓          ↓        ↓      ↓      ↓      │   │
│   │   ┌─────────────────────────────────────────────────────────────┐   │   │
│   │   │  UnifiedGateway (FastAPI) ─► ChannelRouter ─► JottyCLI     │   │   │
│   │   └─────────────────────────────────────────────────────────────┘   │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                      ⚙️ EXECUTION MODES                             │   │
│   │                                                                     │   │
│   │   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐            │   │
│   │   │ 💬 CHAT     │    │ 🔌 API      │    │ 🔄 WORKFLOW │            │   │
│   │   │ ChatAssist  │    │ MCP Tools   │    │ AutoAgent   │            │   │
│   │   │ Interactive │    │ Programatic │    │ DAG Tasks   │            │   │
│   │   └─────────────┘    └─────────────┘    └─────────────┘            │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                     📋 UNIFIED REGISTRY                             │   │
│   │                  (Single Entry Point)                               │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│          ┌─────────────────────────┼─────────────────────────┐              │
│          │                         │                         │              │
│          ▼                         ▼                         ▼              │
│   ┌─────────────┐          ┌─────────────┐          ┌─────────────┐        │
│   │   🤚 HANDS  │          │   👁️ EYES   │          │   📝 MEMORY │        │
│   │   Skills    │          │     UI      │          │   Learning  │        │
│   │   Registry  │          │   Registry  │          │   System    │        │
│   │             │          │             │          │             │        │
│   │ 126 skills  │          │ 16 comps    │          │ 5 levels    │        │
│   │ What we DO  │          │ What we SEE │          │ What we     │        │
│   │             │          │             │          │ REMEMBER    │        │
│   └─────────────┘          └─────────────┘          └─────────────┘        │
│          │                         │                         │              │
│          └─────────────────────────┼─────────────────────────┘              │
│                                    ▼                                         │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                         🧠 BRAIN                                     │   │
│   │              Swarms + Agents (Coordination)                         │   │
│   │   SwarmIntelligence │ TD-Lambda │ MorphScorer │ CurriculumGen      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                       💾 PERSISTENCE                                │   │
│   │            ~/jotty/intelligence/  │  ~/jotty/skills/               │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## The Five Layers

| Layer | Components | Purpose |
|-------|------------|---------|
| **Interface** | Telegram, Slack, Discord, WhatsApp, Web, CLI, SDK | External entry points |
| **Modes** | Chat, API, Workflow | Execution patterns |
| **Registry** | Skills (Hands), UI (Eyes), Memory | Unified capabilities |
| **Brain** | Swarms, Agents, SwarmIntelligence | Coordination & learning |
| **Persistence** | JSON files, skill directories | Long-term storage |

---

## 1. 🧠 Brain - Swarms + Agents (Coordination)

The brain orchestrates all activity through hierarchical swarms and agents.

### Agent Hierarchy

```
BaseAgent (ABC)
│
├── DomainAgent (DSPy signatures for single tasks)
│
├── MetaAgent (self-improvement, evaluation)
│   ├── ValidationAgent (pre/post validation)
│   ├── ExpertAgent (gold standard evaluation)
│   ├── ReviewerAgent (pattern analysis)
│   ├── PlannerAgent (execution planning)
│   ├── ActorAgent (task execution)
│   ├── AuditorAgent (evaluation quality)
│   └── LearnerAgent (pattern extraction)
│
└── AutonomousAgent (open-ended problem solving)
    └── AutoAgent (legacy wrapper)
```

### Swarm Hierarchy

```
BaseSwarm (ABC)
│
└── DomainSwarm (declarative AgentTeam)
    ├── CodingSwarm (8 agents)
    ├── TestingSwarm (6 agents)
    ├── ReviewSwarm (5 agents)
    ├── DataAnalysisSwarm (7 agents)
    ├── FundamentalSwarm (8 agents)
    ├── DevOpsSwarm (6 agents)
    ├── IdeaWriterSwarm (8 agents)
    └── LearningSwarm (6 agents)
```

### Self-Improvement Loop

```
Expert → Reviewer → Planner → Actor → Auditor → Learner
   │                                              │
   └──────────────── feedback ────────────────────┘
```

### SwarmIntelligence Components

| Component | Purpose |
|-----------|---------|
| **MorphScorer** | RCS/RDS/TRAS credit assignment |
| **CurriculumGenerator** | DrZero-style self-curriculum |
| **ByzantineVerifier** | Multi-agent consensus |
| **StigmergyLayer** | Pheromone-based coordination |
| **ToolManager** | Tool success tracking |

---

## 2. 🤚 Hands - SkillsRegistry (What We DO)

Skills are the execution capabilities of the system.

### Architecture

```
SkillsRegistry
├── SkillDefinition
│   ├── name: str
│   ├── description: str
│   ├── tools: Dict[str, Callable]  (lazy loaded)
│   ├── tool_metadata: Dict[str, ToolMetadata]
│   ├── category: str
│   ├── mcp_enabled: bool
│   └── tags: List[str]
│
└── ToolMetadata
    ├── name: str
    ├── description: str
    ├── parameters: Dict (JSON Schema)
    ├── mcp_enabled: bool
    └── to_claude_tool() → Claude API format
```

### Current Skills (126)

| Category | Examples |
|----------|----------|
| **Web** | web-search, fetch-webpage, scraper |
| **Data** | calculator, data-analysis, csv-tools |
| **Media** | image-generator, audio-tools, video |
| **System** | file-operations, shell-exec, git |
| **AI** | llm-chat, embedding, summarize |

### Usage

```python
from Jotty.core.registry import get_unified_registry

registry = get_unified_registry()

# Get a skill
skill = registry.get_skill('web-search')
tools = skill.tools  # Lazy loaded

# Convert to Claude format
claude_tools = skill.to_claude_tools()

# Get MCP-enabled tools
mcp_tools = registry.get_mcp_tools()
```

---

## 3. 👁️ Eyes - UIRegistry (What We SEE)

UI components for rendering agent output.

### Architecture

```
UIRegistry
├── UIComponent
│   ├── component_type: str
│   ├── label: str
│   ├── category: str
│   ├── icon: str
│   ├── content_type: str (json, markdown, code)
│   ├── to_a2ui_func: Callable  (A2UI conversion)
│   ├── to_agui_func: Callable  (AGUI conversion)
│   └── has_adapters: bool
│
└── Categories
    ├── Content (text, code)
    ├── Data (data-table)
    ├── Diagrams (mermaid)
    ├── Visualization (chart, timeline)
    ├── Project (kanban, todos)
    ├── Media (image, audio, video)
    └── Layout (card)
```

### Current Components (16)

| Category | Components |
|----------|------------|
| **Content** | 📝 Text, 💻 Code |
| **Data** | 📋 Data Table |
| **Diagrams** | 📊 Mermaid |
| **Visualization** | 📈 Chart, 📅 Timeline |
| **Project** | 📌 Kanban, ✅ Todos |
| **Media** | 🖼️ Image, 🔊 Audio, 🎬 Video |

### Usage

```python
from Jotty.core.registry import get_unified_registry

registry = get_unified_registry()

# Get a component
chart = registry.ui.get('chart')

# Convert content to A2UI
a2ui_blocks = registry.ui.convert_to_a2ui('chart', data)

# Get by category
viz_components = registry.ui.get_by_category('Visualization')
```

---

## 4. 🧠 Memory - HierarchicalMemory (What We REMEMBER)

5-level memory system with learning integration.

### Memory Levels

```
┌─────────────────────────────────────────────────────────────────┐
│                    HIERARCHICAL MEMORY                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Level 5: META           (Learning wisdom, never decays)       │
│      ↑                                                          │
│   Level 4: CAUSAL         (Why things work, enables reasoning)  │
│      ↑                                                          │
│   Level 3: PROCEDURAL     (How to do things, action sequences)  │
│      ↑                                                          │
│   Level 2: SEMANTIC       (Abstracted patterns, LLM-extracted)  │
│      ↑                                                          │
│   Level 1: EPISODIC       (Raw experiences, fast decay)         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Memory Features

| Feature | Description |
|---------|-------------|
| **LLM Retrieval** | No embeddings, uses LLM for semantic matching |
| **Goal-Conditioned** | Values depend on current goal context |
| **Causal Links** | Why something works, enables transfer |
| **Auto-Consolidation** | Episodic → Semantic → Procedural |
| **Deduplication** | Prevents memory bloat |

### Integration with Learning

```
TD-Lambda Learner
      │
      ▼
┌─────────────────────┐
│  HierarchicalMemory │
├─────────────────────┤
│ store()             │ ← Store experience with value
│ retrieve()          │ ← Get relevant memories for goal
│ update_value()      │ ← TD(λ) value updates
│ consolidate()       │ ← Promote to higher levels
└─────────────────────┘
      │
      ▼
GoalHierarchy (Aristotle)
      │
      ▼
Knowledge Transfer
```

---

## 5. Execution Modes

### 💬 Chat Mode (ChatAssistant)

Interactive conversation with A2UI rendering.

```python
from Jotty.core.agents import ChatAssistant

assistant = ChatAssistant(state_manager=state)
response = await assistant.run(goal="What's my task backlog?")
# Returns A2UI widgets for rich rendering
```

**Features:**
- Task queries (backlog, completed, pending)
- System status
- General conversation
- A2UI widget output

### 🔌 API Mode (MCP Tools)

Programmatic tool execution via MCP protocol.

```python
from Jotty.core.registry import get_unified_registry

registry = get_unified_registry()

# Get tools in Claude format
tools = registry.get_claude_tools(['web-search', 'calculator'])

# Execute via MCP
from Jotty.core.integration import MCPToolExecutor
executor = MCPToolExecutor()
result = await executor.execute('search_web_tool', query="...")
```

**Features:**
- MCP-compatible tool definitions
- Parameter validation
- Error handling
- Result formatting

### 🔄 Workflow Mode (AutoAgent + DAG)

Autonomous task execution with DAG orchestration.

```python
from Jotty.core.agents import AutoAgent

agent = AutoAgent()
result = await agent.execute(
    "Research topic X, create a report, and send via email"
)
# Automatically breaks down, plans, and executes
```

**Features:**
- Task breakdown into DAG
- Parallel execution where possible
- Dependency management
- Progress tracking

---

## 6. How Everything Links Together

### Complete Flow

```
User Request
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│                         MODE SELECTION                           │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                  │
│  │   Chat   │    │   API    │    │ Workflow │                  │
│  └────┬─────┘    └────┬─────┘    └────┬─────┘                  │
└───────┼───────────────┼───────────────┼─────────────────────────┘
        │               │               │
        └───────────────┼───────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                      UNIFIED REGISTRY                            │
│                                                                  │
│  registry.discover_for_task("create chart with data")           │
│      → skills: [data-analysis, calculator]                      │
│      → ui: [chart, data-table]                                  │
└─────────────────────────────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
   ┌─────────┐    ┌─────────┐    ┌─────────────┐
   │  HANDS  │    │  EYES   │    │   MEMORY    │
   │ Skills  │    │   UI    │    │  Learning   │
   └────┬────┘    └────┬────┘    └──────┬──────┘
        │              │                │
        └──────────────┼────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                          BRAIN                                   │
│                                                                  │
│  BaseSwarm.execute(task)                                        │
│      │                                                          │
│      ├── _pre_execute_learning()                                │
│      │       └── Load learned context from memory               │
│      │       └── Compute MorphAgent scores                      │
│      │       └── Get tool recommendations                       │
│      │                                                          │
│      ├── Agent Team Execution                                   │
│      │       └── DomainSwarm → AgentTeam → Agents               │
│      │       └── Skills (Hands) execute tools                   │
│      │       └── UI (Eyes) format output                        │
│      │                                                          │
│      ├── _post_execute_learning()                               │
│      │       └── Record to HierarchicalMemory                   │
│      │       └── Update TD-Lambda values                        │
│      │       └── Send executor feedback                         │
│      │       └── Persist learning to disk                       │
│      │                                                          │
│      └── Self-Improvement Loop (if enabled)                     │
│              └── Expert → Reviewer → Planner → Actor            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                       │
                       ▼
               Response to User
               (with A2UI rendering)
```

### Data Flow Example

```
Task: "Analyze sales data and create a chart"
                    │
                    ▼
┌──────────────────────────────────────────┐
│ 1. DISCOVERY (UnifiedRegistry)           │
│    Skills: data-analysis, calculator     │
│    UI: chart, data-table                 │
└──────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────┐
│ 2. PRE-LEARNING (Memory)                 │
│    Retrieved: "sales queries work best   │
│    with GROUP BY month"                  │
│    Tool advice: "calculator 95% reliable"│
└──────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────┐
│ 3. EXECUTION (Swarm)                     │
│    DataAnalysisSwarm.execute()           │
│    ├── DataLoadAgent → loads CSV         │
│    ├── AnalystAgent → runs analysis      │
│    └── ReportAgent → generates output    │
└──────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────┐
│ 4. OUTPUT (UI Registry)                  │
│    registry.ui.convert_to_a2ui('chart',  │
│        analysis_data)                    │
│    → A2UI blocks for rendering           │
└──────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────┐
│ 5. POST-LEARNING (Memory)                │
│    Store: "GROUP BY month worked"        │
│    TD-Lambda: update access values       │
│    Profile: trust 0.5 → 0.6              │
│    Persist: ~/jotty/intelligence/*.json  │
└──────────────────────────────────────────┘
```

---

## 7. Persistence Locations

| Data | Location | Auto-Loaded |
|------|----------|-------------|
| **SwarmIntelligence** | `~/jotty/intelligence/{swarm}_{domain}.json` | ✅ Yes |
| **Skills** | `~/jotty/skills/{skill_name}/` | ✅ Yes |
| **Agent Profiles** | In SwarmIntelligence JSON | ✅ Yes |
| **Collective Memory** | In SwarmIntelligence JSON | ✅ Yes |
| **Tool Success Rates** | In SwarmIntelligence JSON | ✅ Yes |
| **MorphAgent Scores** | In SwarmIntelligence JSON | ✅ Yes |

---

## 8. Quick Reference

### Get Started

```python
from Jotty.core.registry import get_unified_registry
from Jotty.core.agents import AutoAgent, ChatAssistant
from Jotty.core.swarms import CodingSwarm

# Registry (Hands + Eyes)
registry = get_unified_registry()

# Chat mode
chat = ChatAssistant()
response = await chat.run(goal="Hello")

# Workflow mode
agent = AutoAgent()
result = await agent.execute("Build a web scraper")

# Swarm mode
swarm = CodingSwarm()
result = await swarm.execute("Implement feature X")
```

### Key Imports

```python
# Registry
from Jotty.core.registry import (
    get_unified_registry,
    SkillsRegistry,
    UIRegistry,
    ToolMetadata,
    UIComponent,
)

# Agents
from Jotty.core.agents import (
    BaseAgent,
    DomainAgent,
    MetaAgent,
    AutoAgent,
    ChatAssistant,
)

# Swarms
from Jotty.core.swarms import (
    BaseSwarm,
    DomainSwarm,
    CodingSwarm,
    DataAnalysisSwarm,
)

# Learning
from Jotty.core.learning import TDLambdaLearner
from Jotty.core.memory import HierarchicalMemory
from Jotty.core.orchestration.v2 import SwarmIntelligence
```

---

## 9. 🌐 Interface Layer (External Connections)

The Interface Layer sits on top of the execution modes, providing multiple entry points.

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          INTERFACE LAYER                                     │
│                     (External Entry Points)                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐│
│  │ 📱 Telegram│ │ 💬 Slack   │ │ 🎮 Discord │ │ 📲 WhatsApp│ │ 🌐 Web     ││
│  │  Webhook   │ │ Events API │ │  Webhook   │ │  Webhook   │ │ PWA/WS     ││
│  └─────┬──────┘ └─────┬──────┘ └─────┬──────┘ └─────┬──────┘ └─────┬──────┘│
│        │              │              │              │              │        │
│        └──────────────┴──────────────┴──────────────┴──────────────┘        │
│                                      │                                       │
│                                      ▼                                       │
│                       ┌────────────────────────────────┐                    │
│                       │     🔀 UnifiedGateway          │                    │
│                       │     (FastAPI + WebSocket)      │                    │
│                       │                                │                    │
│                       │  • HTTP webhooks per channel   │                    │
│                       │  • WebSocket for real-time     │                    │
│                       │  • Health & stats endpoints    │                    │
│                       │  • PWA static files            │                    │
│                       └──────────────┬─────────────────┘                    │
│                                      │                                       │
│                                      ▼                                       │
│                       ┌────────────────────────────────┐                    │
│                       │     📡 ChannelRouter           │                    │
│                       │                                │                    │
│                       │  • Session per user/channel    │                    │
│                       │  • Trust management            │                    │
│                       │  • Context (last 10 msgs)      │                    │
│                       │  • Async message queue         │                    │
│                       └──────────────┬─────────────────┘                    │
│                                      │                                       │
│        ┌─────────────────────────────┼─────────────────────────────┐        │
│        │                             │                             │        │
│        ▼                             ▼                             ▼        │
│  ┌────────────┐              ┌────────────────┐            ┌────────────┐  │
│  │ 💻 CLI     │              │ 🔌 SDK         │            │ 📦 HTTP    │  │
│  │ JottyCLI   │              │ Python/TS/Go   │            │ REST API   │  │
│  │            │              │                │            │            │  │
│  │ • REPL     │              │ • Client libs  │            │ • /message │  │
│  │ • Commands │              │ • Type-safe    │            │ • /health  │  │
│  │ • History  │              │ • Async        │            │ • /stats   │  │
│  └─────┬──────┘              └───────┬────────┘            └─────┬──────┘  │
│        │                             │                           │          │
│        └─────────────────────────────┼───────────────────────────┘          │
│                                      │                                       │
└──────────────────────────────────────┼───────────────────────────────────────┘
                                       ▼
                              ┌─────────────────┐
                              │   JottyCLI      │
                              │   (Core)        │
                              └────────┬────────┘
                                       │
                              ┌────────┴────────┐
                              │                 │
                              ▼                 ▼
                    ┌──────────────┐   ┌────────────────┐
                    │ SwarmManager │   │ SkillsRegistry │
                    │              │   │                │
                    │ Brain        │   │ Hands          │
                    └──────────────┘   └────────────────┘
```

### Channel Types

```python
class ChannelType(Enum):
    TELEGRAM = "telegram"    # Telegram Bot webhooks
    SLACK = "slack"          # Slack Events API
    DISCORD = "discord"      # Discord interactions
    WHATSAPP = "whatsapp"    # WhatsApp Business API
    WEBSOCKET = "websocket"  # Real-time WebSocket
    HTTP = "http"            # Generic HTTP POST
```

### UnifiedGateway Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Redirect to PWA |
| `/app` | GET | PWA chat interface |
| `/health` | GET | Health check with stats |
| `/stats` | GET | Gateway statistics |
| `/webhook/telegram` | POST | Telegram webhook |
| `/webhook/slack` | POST | Slack Events API |
| `/webhook/discord` | POST | Discord interactions |
| `/webhook/whatsapp` | POST/GET | WhatsApp webhook |
| `/ws` | WS | WebSocket real-time |
| `/message` | POST | Generic HTTP message |
| `/docs` | GET | OpenAPI documentation |

### ChannelRouter Features

| Feature | Description |
|---------|-------------|
| **Session Management** | Per-user session with context history |
| **Trust Management** | Authorization and pairing codes |
| **Message Queue** | Async processing with priorities |
| **Responders** | Channel-specific response handlers |
| **Context Window** | Last 10 messages per session |

### CLI Architecture

```
JottyCLI
├── config/
│   ├── loader.py      # Configuration loading
│   └── schema.py      # CLIConfig dataclass
├── ui/
│   └── renderer.py    # RichRenderer (terminal UI)
├── repl/
│   ├── engine.py      # REPLEngine (prompt_toolkit)
│   ├── session.py     # SessionManager
│   ├── history.py     # HistoryManager
│   └── completer.py   # Auto-completion
├── commands/
│   ├── base.py        # CommandRegistry
│   ├── run.py         # /run command
│   ├── agents.py      # /agent command
│   ├── skills.py      # /skills command
│   ├── swarm.py       # /swarm command
│   ├── learn.py       # /learn command
│   ├── memory.py      # /memory command
│   ├── plan.py        # /plan command
│   └── help_cmd.py    # /help, /quit, /clear
├── gateway/
│   ├── server.py      # UnifiedGateway
│   ├── channels.py    # ChannelRouter
│   └── trust.py       # TrustManager
└── plugins/
    └── loader.py      # PluginLoader
```

### SDK Support

Generated client libraries for multiple languages:

```
Jotty/sdk/generated/
├── python/
│   └── jotty_api_client/
│       ├── client.py           # Client, AuthenticatedClient
│       ├── models/
│       │   ├── chat_message.py
│       │   ├── chat_execute_request.py
│       │   ├── chat_execute_response.py
│       │   ├── chat_stream_request.py
│       │   ├── workflow_execute_request.py
│       │   └── workflow_execute_response.py
│       └── types.py
├── typescript/                  # TypeScript client
└── go/                          # Go client
```

### Usage Examples

**1. CLI (Interactive)**
```bash
python -m Jotty.cli
# jotty> Search for AI news and create a summary
```

**2. CLI (Single Command)**
```bash
python -m Jotty.cli -c "Analyze data.csv and create a chart"
```

**3. Web Gateway**
```bash
python Jotty/web.py --port 8766
# Starts: http://localhost:8766 (PWA + API + WebSockets)
```

**4. Telegram Integration**
```bash
# Set TELEGRAM_TOKEN, register webhook to /webhook/telegram
python -m Jotty.cli.gateway --port 8766
```

**5. SDK (Python)**
```python
from jotty_api_client import Client

client = Client(base_url="http://localhost:8766")
response = client.chat_execute(
    ChatExecuteRequest(messages=[
        ChatMessage(role="user", content="Hello")
    ])
)
```

**6. WebSocket**
```javascript
const ws = new WebSocket('ws://localhost:8766/ws');
ws.send(JSON.stringify({
    content: "Research AI trends",
    user_id: "user123"
}));
```

---

## 10. Complete System Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            COMPLETE JOTTY FLOW                               │
└─────────────────────────────────────────────────────────────────────────────┘

   📱 Telegram  💬 Slack  🎮 Discord  📲 WhatsApp  🌐 Web  💻 CLI  📦 SDK
        │          │          │           │          │       │       │
        └──────────┴──────────┴───────────┴──────────┴───────┴───────┘
                                     │
                                     ▼
                    ┌────────────────────────────────┐
                    │      INTERFACE LAYER           │
                    │                                │
                    │  UnifiedGateway ─► ChannelRouter ─► JottyCLI
                    └────────────────────────────────┘
                                     │
                                     ▼
                    ┌────────────────────────────────┐
                    │        EXECUTION MODES         │
                    │                                │
                    │  💬 Chat    🔌 API    🔄 Workflow
                    │  ChatAssist  MCP     AutoAgent │
                    └────────────────────────────────┘
                                     │
                                     ▼
                    ┌────────────────────────────────┐
                    │       UNIFIED REGISTRY         │
                    │                                │
                    │  discover_for_task(text)       │
                    │    → skills + ui components    │
                    └────────────────────────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
              ▼                      ▼                      ▼
     ┌────────────────┐    ┌────────────────┐    ┌────────────────┐
     │   🤚 HANDS     │    │    👁️ EYES     │    │   🧠 MEMORY    │
     │   Skills       │    │      UI        │    │   Learning     │
     │   Registry     │    │    Registry    │    │    System      │
     │                │    │                │    │                │
     │  126 skills    │    │  16 components │    │  5 levels      │
     │  Tools + MCP   │    │  A2UI + AGUI   │    │  TD-Lambda     │
     └────────────────┘    └────────────────┘    └────────────────┘
              │                      │                      │
              └──────────────────────┼──────────────────────┘
                                     │
                                     ▼
                    ┌────────────────────────────────┐
                    │          🧠 BRAIN              │
                    │                                │
                    │  SwarmManager ─► DomainSwarm   │
                    │       │              │         │
                    │  SwarmIntelligence  AgentTeam  │
                    │       │              │         │
                    │  TD-Lambda      DomainAgents   │
                    └────────────────────────────────┘
                                     │
                                     ▼
                    ┌────────────────────────────────┐
                    │        PERSISTENCE             │
                    │                                │
                    │  ~/jotty/intelligence/*.json   │
                    │  ~/jotty/skills/               │
                    │  ~/jotty/sessions/             │
                    └────────────────────────────────┘
```

---

## 11. Core Foundation (Cross-Cutting Concerns)

These components support all layers and are used throughout the system.

### Error Handling (`core/foundation/exceptions.py`)

```
JottyError (base)
├── ConfigurationError
│   ├── InvalidConfigError
│   └── MissingConfigError
├── ExecutionError
│   ├── AgentExecutionError
│   ├── ToolExecutionError
│   ├── TimeoutError
│   └── CircuitBreakerError
├── ContextError
│   ├── ContextOverflowError
│   ├── CompressionError
│   └── ChunkingError
├── MemoryError
│   ├── MemoryRetrievalError
│   ├── MemoryStorageError
│   └── ConsolidationError
├── LearningError
│   ├── RewardCalculationError
│   ├── CreditAssignmentError
│   └── PolicyUpdateError
├── CommunicationError
│   ├── MessageDeliveryError
│   └── FeedbackRoutingError
├── ValidationError
│   ├── InputValidationError
│   └── OutputValidationError
├── PersistenceError
│   ├── StorageError
│   └── RetrievalError
└── IntegrationError
    ├── LLMError
    ├── DSPyError
    └── ExternalToolError
```

**Usage:**
```python
from Jotty.core.foundation.exceptions import (
    AgentExecutionError,
    ContextOverflowError,
    wrap_exception
)

try:
    result = agent.execute(task)
except AgentExecutionError as e:
    logger.error(f"Agent failed: {e.message}")
    # e.context has additional info
    # e.original_error has root cause
```

### API Layer (`core/api/`)

| Component | Purpose |
|-----------|---------|
| `JottyAPI` | Unified entry point (chat + workflow) |
| `ChatAPI` | Chat-specific operations |
| `WorkflowAPI` | Workflow execution |
| `generate_openapi_spec()` | OpenAPI 3.0 spec for SDK generation |

**Usage:**
```python
from Jotty.core.api import JottyAPI, generate_openapi_spec

# Programmatic API
api = JottyAPI(agents=[...])
result = await api.chat_execute(message="Hello")
result = await api.workflow.run(goal="Research X")

# Generate OpenAPI spec for SDKs
spec = generate_openapi_spec(
    title="Jotty API",
    version="1.0.0",
    base_url="http://localhost:8766"
)
```

### Configuration (`cli/config/schema.py`)

```python
@dataclass
class CLIConfig:
    provider: ProviderConfig    # LLM provider settings
    swarm: SwarmConfig          # Swarm behavior
    learning: LearningConfig    # TD-Lambda settings
    ui: UIConfig                # Terminal UI
    features: FeaturesConfig    # Feature flags
    session: SessionConfig      # Session management
    telegram: TelegramConfig    # Telegram integration
    web: WebConfig              # Web server settings
```

**Config file:** `~/.jotty/config.yaml`

### Registry API (`core/registry/api.py`)

HTTP endpoints for tool/widget discovery:

| Endpoint | Method | Returns |
|----------|--------|---------|
| `/api/jotty/registry` | GET | All tools + widgets |
| `/api/jotty/registry/tools` | GET | All tools |
| `/api/jotty/registry/widgets` | GET | All widgets |
| `/api/jotty/registry/skills` | GET | All skills |
| `/api/jotty/registry/tools/{name}` | GET | Specific tool |
| `/api/jotty/registry/widgets/{type}` | GET | Specific widget |

---

## Summary

| Layer | Component | Count | Purpose |
|-------|-----------|-------|---------|
| **Interface** | Channels | 6 | External entry points |
| **Interface** | UnifiedGateway | 1 | HTTP/WS server |
| **Interface** | CLI | 1 | Interactive terminal |
| **Interface** | SDK | 3 | Client libraries |
| **Mode** | Chat | 1 | Interactive conversation |
| **Mode** | API | 1 | Programmatic access |
| **Mode** | Workflow | 1 | Autonomous execution |
| **Brain** | Swarms | 8+ | Coordination |
| **Brain** | Agents | 11+ | Execution |
| **Brain** | SwarmIntelligence | 1 | Learning orchestration |
| **Hands** | Skills | 126 | What we DO |
| **Eyes** | UI Components | 16 | What we SEE |
| **Memory** | Levels | 5 | What we REMEMBER |
| **Foundation** | Exceptions | 30+ | Error handling |
| **Foundation** | API Layer | 4 | Programmatic access |
| **Foundation** | Config | 8 | Configuration schemas |

**The Jotty system is a fully integrated, self-improving multi-agent architecture where:**
- **Interface Layer** provides multiple entry points (Telegram, Slack, Discord, WhatsApp, Web, CLI, SDK)
- **Execution Modes** route requests (Chat, API, Workflow)
- **Brain** coordinates through swarms and agents
- **Hands** execute through skills and tools
- **Eyes** render through UI components
- **Memory** learns and persists across sessions
- **Foundation** provides error handling, API layer, and configuration

**Everything is connected and discoverable through the UnifiedRegistry.**
