# JOTTY Framework

**Self-organizing Neural Agent Protocol for Swarm Execution**

JOTTY is a generic, config-driven multi-agent orchestration framework built on DSPy. It enables autonomous agent swarms to coordinate, validate, learn, and execute complex multi-step tasks with no domain-specific hardcoding.

## 🎯 Key Features

- **Generic Architecture**: Works with ANY agent type (SQL, Code, Marketing, Analytics, etc.)
- **No Hardcoding**: All mappings, validations, and transformations are agentic (LLM-powered)
- **Config-Driven**: All parameters, tools, and behaviors controlled via YAML configs
- **Multi-Agent RL**: DQN-based learning with context updates (no weight updates needed)
- **Brain-Inspired Memory**: Hippocampal consolidation, sharp-wave ripple, synaptic pruning
- **Game Theory Cooperation**: Nash equilibrium communication, Shapley credit assignment

## 📦 Installation

```bash
# Install dependencies
pip install dspy-ai pyyaml

# Add Jotty to your Python path
cd /path/to/Jotty
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or install as a package
pip install -e .
```

## 🚀 Quick Start

```python
import dspy
from Jotty import Conductor, AgentConfig, JottyConfig

# 1. Configure DSPy
dspy.configure(lm=dspy.LM("openai/gpt-4", api_key="your-key"))

# 2. Define your agents (any DSPy module)
class MyAgent(dspy.Module):
    def __init__(self):
        self.predictor = dspy.ChainOfThought("query -> answer")
    
    def forward(self, query: str) -> str:
        return self.predictor(query=query).answer

# 3. Create agent configs
agents = [
    AgentConfig(
        name="MyAgent",
        agent=MyAgent(),
        architect_prompts=["prompts/my_agent_architect.md"],
        auditor_prompts=["prompts/my_agent_auditor.md"],
        parameter_mappings={
            "query": "context.query",
        },
    ),
]

# 4. Initialize Jotty
config = JottyConfig(
    max_actor_iters=100,
    enable_validation=True,
    consolidation_interval=3,
)

conductor = Conductor(
    actors=agents,
    metadata_provider=your_metadata,  # Optional: provides tools/context
    config=config,
)

# 5. Run
result = await conductor.run(goal="Answer the user's question", query="What is X?")
print(result.final_output)
```

## ⚙️ Configuration Reference

### JottyConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_base_dir` | str | "./outputs" | Base directory for all outputs |
| `create_run_folder` | bool | True | Create timestamped folder per run |
| `enable_beautified_logs` | bool | True | Generate human-readable logs |
| `log_level` | str | "INFO" | Logging level |
| `max_actor_iters` | int | 100 | Max iterations per actor (ReAct) |
| `max_episode_iterations` | int | 12 | Max full pipeline iterations |
| `max_eval_iters` | int | 10 | Max evaluation iterations |
| `llm_timeout_seconds` | float | 180.0 | Timeout for LLM calls |
| `actor_timeout` | float | 900.0 | Timeout for actor execution |
| `enable_dead_letter_queue` | bool | True | Enable DLQ for failed operations |
| `dlq_max_retries` | int | 3 | Max retries from DLQ |
| `enable_validation` | bool | True | Enable Architect/Auditor validation |
| `validation_mode` | str | "full" | Validation mode: "full", "light", "none" |
| `max_validation_rounds` | int | 3 | Max validation retry rounds |
| `episodic_capacity` | int | 10000 | Episodic memory capacity |
| `semantic_capacity` | int | 5000 | Semantic memory capacity |
| `consolidation_interval` | int | 3 | Episodes between brain consolidation |
| `learning_rate` | float | 0.1 | Q-learning rate |
| `discount_factor` | float | 0.95 | Future reward discount (γ) |
| `lambda_decay` | float | 0.8 | TD(λ) eligibility trace decay |
| `initial_epsilon` | float | 0.3 | Initial exploration rate |
| `min_epsilon` | float | 0.05 | Minimum exploration rate |
| `epsilon_decay` | float | 0.995 | Epsilon decay per episode |
| `max_context_tokens` | int | 28000 | Max context tokens |
| `chunk_size` | int | 5000 | Chunk size for large inputs |
| `compression_threshold` | float | 0.7 | Threshold to trigger compression |

---

## 🎓 Understanding the Modes (Critical for Usage)

### LearningMode: How the System Learns

| Mode | Meaning | When to Use |
|------|---------|-------------|
| `DISABLED` | **No learning at all** - Just execute, no memory updates, no Q-learning | Production inference with fixed behavior |
| `CONTEXTUAL` | **Session-only learning** - Updates memory and Q-values during the run, but forgets after session ends | Testing, debugging, single-run scenarios |
| `PERSISTENT` | **Durable learning** - Saves Q-tables, memories, brain state to disk. Auto-loads on next run | Training over multiple runs, continuous improvement |

**Why these names?**
- `CONTEXTUAL` = Learning happens IN CONTEXT (within the session's context window)
- `PERSISTENT` = Learning PERSISTS across sessions (written to disk)

**How does JOTTY learn WITHOUT explicit self-play?**

Traditional RL requires millions of simulations. JOTTY uses **"Context as Gradient"**:

```
Traditional RL:  weights -= learning_rate * gradient  (needs simulations)
JOTTY:         context += lessons_learned           (context IS learning)
```

**Example learning flow:**
1. **Run 1**: Agent tries query, fails (used wrong column `transaction_category`)
2. **Memory stores**: `"For P2P transactions, use category IN ('VPA2ACCOUNT'), NOT transaction_category"`
3. **Run 2**: Same/similar query → Memory retrieved → Injected into agent's prompt
4. **Agent succeeds!** The context update WAS the learning.

This is **implicit self-play**:
- **Auditor feedback** = opponent critique in self-play
- **Retry with feedback** = next game iteration
- **Memory consolidation** = learning from past games

### ValidationMode: When to Validate

| Mode | Meaning | When to Use |
|------|---------|-------------|
| `NONE` | **Skip all validation** - Agent runs raw, no pre-check or post-check | Testing agent logic in isolation |
| `ARCHITECT` | **Plan-first only** - Pre-execution planning validates inputs, no post-check | When you trust agent outputs but need input prep |
| `AUDITOR` | **Check-after only** - No pre-planning, but validates outputs after execution | When inputs are known-good but outputs need QA |
| `BOTH` | **Full validation** - Both Architect (pre) and Auditor (post) run | Production: comprehensive quality control |

**Why "Architect" and "Auditor"?**
- **Architect** = Plans before building (pre-validation, input preparation)
- **Auditor** = Reviews after completion (post-validation, output quality assurance)

Think of it like construction:
- Architect reviews blueprints BEFORE building starts
- Auditor inspects the building AFTER it's built

### CooperationMode: How Agents Work Together

| Mode | Meaning | When to Use |
|------|---------|-------------|
| `INDEPENDENT` | **Solo execution** - Agents don't share rewards | Simple pipelines, testing |
| `SHARED_REWARD` | **Team reward** - All agents get same reward (success/failure) | Tightly coupled pipelines |
| `NASH` | **Game-theoretic** - Nash equilibrium communication, Shapley credit assignment | Complex swarms needing optimal cooperation |

---

### AgentConfig

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | str | ✅ | Unique agent identifier |
| `agent` | DSPy Module | ✅ | The DSPy agent module |
| `architect_prompts` | List[str] | ❌ | Paths to Architect (planning) prompts |
| `auditor_prompts` | List[str] | ❌ | Paths to Auditor (validation) prompts |
| `dependencies` | List[str] | ❌ | Names of agents that must run first |
| `parameter_mappings` | Dict | ❌ | Explicit parameter resolution mappings |
| `is_executor` | bool | ❌ | Marks agent as final executor |

### Parameter Mapping Syntax

```yaml
parameter_mappings:
  # From context (shared state)
  query: "context.query"
  current_date: "context.current_date"
  
  # From metadata provider (tool calls)
  tables_metadata: "metadata.get_all_table_metadata()"
  business_terms: "metadata.get_business_terms()"
  
  # From previous agent output
  resolved_terms: "BusinessTermResolver.resolved_terms"
  tables: "BusinessTermResolver.relevant_tables"
```

## 🧠 Core Components

### Conductor
Main orchestrator that coordinates agent execution, parameter resolution, and learning.

### JottyCore
Wraps each agent with Architect (pre-planning) and Auditor (post-validation) capabilities.

### Architect
Pre-execution planner that validates inputs and prepares context before an agent runs.

### Auditor
Post-execution validator that checks outputs and provides feedback for retry if invalid.

### Axon (Agent Communication)
Peer-to-peer agent communication channel with:
- Message routing
- Format transformation
- Nash equilibrium-based communication decisions
- Cooperation tracking

### Cortex (Memory System)
Hierarchical memory with:
- Episodic memory (recent experiences)
- Semantic memory (consolidated patterns)
- Brain-inspired consolidation (sharp-wave ripple)

### Roadmap (Task Planning)
Markovian TODO system for:
- Task hierarchy
- Dependency tracking
- Progress estimation
- Dynamic replanning

### ToolShed
Dynamic tool discovery and access:
- Capability indexing
- Schema-based matching
- Caching wrapper for redundant calls

## 📁 Output Structure

```
outputs/
└── run_20260106_114212/
    ├── jotty_state/
    │   ├── brain_state.json      # Consolidated memories
    │   ├── q_tables/             # Q-learning state
    │   ├── memories/             # Agent memories
    │   └── roadmap.json          # Task progress
    ├── beautified/
    │   └── execution_log.md      # Human-readable log
    ├── session_todo.md           # Session tasks
    └── query_result.txt          # Execution result
```

## 📊 Example: SQL Pipeline

```python
from jotty.pipelines import JottyEnhancedPipeline

pipeline = JottyEnhancedPipeline(
    metadata_dir='data/sample_data/upi_olap'
)

result = await pipeline.arun(
    query="What was the total count of P2P transactions yesterday?",
    model_id="default"
)

print(f"SQL: {result.sql_query}")
print(f"Result: {result.query_result}")
```

## 🔧 Creating Custom Agents

```python
import dspy
from Jotty import AgentConfig

class MyCustomAgent(dspy.Module):
    """Custom agent with DSPy signature."""
    
    class Signature(dspy.Signature):
        """Analyze data and produce insights."""
        data: str = dspy.InputField(desc="Input data to analyze")
        context: str = dspy.InputField(desc="Additional context")
        
        insights: str = dspy.OutputField(desc="Key insights from the data")
        recommendations: list = dspy.OutputField(desc="Action recommendations")
    
    def __init__(self):
        super().__init__()
        self.predictor = dspy.ChainOfThought(self.Signature)
    
    def forward(self, data: str, context: str = "") -> dspy.Prediction:
        return self.predictor(data=data, context=context)

# Create config
config = AgentConfig(
    name="InsightGenerator",
    agent=MyCustomAgent(),
    architect_prompts=["prompts/insight_architect.md"],
    auditor_prompts=["prompts/insight_auditor.md"],
    dependencies=["DataLoader"],
    parameter_mappings={
        "data": "DataLoader.processed_data",
        "context": "context.business_context",
    },
)
```

## 🎮 Multi-Agent Cooperation

JOTTY uses game theory for agent cooperation:

1. **Nash Equilibrium Communication**: Agents decide when to communicate based on information value vs. cost
2. **Shapley Value Credit Assignment**: Fair credit distribution based on marginal contribution
3. **Predictive MARL**: Each agent learns to predict what other agents will do
4. **Context as Gradient**: Context updates serve as learning signals (no weight updates needed)

## 📚 Documentation

- [Architecture Overview](docs/JOTTY_ARCHITECTURE.md)
- [API Reference](docs/API_REFERENCE.md)
- [A-Team Roster](docs/A_TEAM_ROSTER.md)
- [Configuration Guide](docs/CONFIG_GUIDE.md)

## 🏗️ Design Principles

1. **No Domain Hardcoding**: JOTTY knows nothing about SQL or any specific domain
2. **Config-Driven**: All behavior controlled via YAML, not code
3. **Agentic Everything**: No regex, fuzzy matching, or rule-based fallbacks
4. **Self-Organizing**: Agents coordinate, negotiate, and learn autonomously
5. **Observable**: Rich logging, state persistence, and debugging support

## 📝 License

MIT License - See LICENSE file for details.

## 🤝 Contributing

Contributions welcome! Please read CONTRIBUTING.md for guidelines.

---

*Built with ❤️ using DSPy and Multi-Agent Reinforcement Learning*
