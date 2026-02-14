# Getting Started with Jotty

**Welcome to Jotty** - the most advanced multi-agent AI framework with brain-inspired memory, reinforcement learning, and swarm intelligence.

This guide will get you from zero to running your first intelligent agent swarm in **5 minutes**.

---

## Table of Contents

1. [Quick Start (5 minutes)](#quick-start-5-minutes)
2. [Installation](#installation)
3. [Your First Agent](#your-first-agent)
4. [Your First Swarm](#your-first-swarm)
5. [Common Patterns](#common-patterns)
6. [Architecture Overview](#architecture-overview)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start (5 minutes)

### 1. Install Jotty

```bash
cd /path/to/Jotty
pip install -r requirements.txt
```

### 2. Set up API keys

```bash
export ANTHROPIC_API_KEY="your-key-here"  # For Claude
export OPENAI_API_KEY="your-key-here"     # For GPT (optional)
```

### 3. Run your first swarm

```python
from Jotty.core.orchestration import Orchestrator

# Create a swarm from natural language
swarm = Orchestrator(agents="Research AI trends and create a report")

# Execute the task
result = await swarm.run(goal="Find the top 5 AI trends in 2026")

print(f"Success: {result.success}")
print(f"Output: {result.output}")
```

That's it! You just ran a multi-agent swarm that:
- Automatically created specialized agents
- Coordinated their actions using RL-based routing
- Learned from execution to improve future performance

---

## Installation

### Prerequisites

- Python 3.10+
- pip or conda
- API keys for LLM providers (Claude, OpenAI, or Groq)

### Full Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Jotty.git
cd Jotty

# Install dependencies
pip install -r requirements.txt

# Optional: Install development dependencies
pip install -r requirements-dev.txt

# Verify installation
python -m Jotty.cli --help
```

### Environment Variables

Create a `.env` file in the project root:

```bash
# Required: At least one LLM provider
ANTHROPIC_API_KEY=your-anthropic-key
OPENAI_API_KEY=your-openai-key      # Optional
GROQ_API_KEY=your-groq-key          # Optional (free tier available)

# Optional: Communication channels
TELEGRAM_TOKEN=your-telegram-token
SLACK_SIGNING_SECRET=your-slack-secret
```

---

## Your First Agent

Agents are the building blocks of Jotty. Here's how to create and use one:

### Simple Agent

```python
from Jotty.core.agents import ChatAssistant

# Create a chat agent
agent = ChatAssistant()

# Ask a question
response = await agent.chat("What is machine learning?")
print(response)
```

### Agent with Skills

```python
from Jotty.core.agents import AutoAgent
from Jotty.core.registry import get_unified_registry

# Get the skills registry
registry = get_unified_registry()

# Discover skills for a task
skills = registry.discover_for_task("analyze data and create charts")
print(f"Found {len(skills['skills'])} skills")

# Create an agent with these skills
agent = AutoAgent(
    name="DataAnalyst",
    goal="Analyze sales data and visualize trends",
    skills=skills['skills'][:5]  # Use top 5 skills
)

# Execute a workflow
result = await agent.execute("Load sales.csv, analyze trends, create chart")
```

### Agent with Memory

```python
from Jotty.core.agents import AutoAgent
from Jotty.core.memory import get_memory_system

# Get the memory system
memory = get_memory_system()

# Store some context
memory.store(
    content="Our product launch is scheduled for March 15, 2026",
    level="episodic",
    goal="planning",
    metadata={"priority": "high"}
)

# Create agent (it will automatically access memory)
agent = AutoAgent(name="Planner", goal="Plan product launch")

# Agent can now recall the stored memory
result = await agent.execute("What's our product launch date?")
# Will answer: "March 15, 2026"
```

---

## Your First Swarm

Swarms coordinate multiple agents to solve complex tasks. They're more powerful than single agents because they:
- Divide tasks among specialists
- Learn which agents are best for each task type
- Self-organize using swarm intelligence

### Basic Swarm

```python
from Jotty.core.orchestration import Orchestrator

# Zero-config: Just describe what you need
swarm = Orchestrator(
    agents="Researcher + Writer + Reviewer"
)

result = await swarm.run(
    goal="Research quantum computing and write a 500-word article"
)

print(result.output)
```

### Specialized Swarm

```python
from Jotty.core.swarms import CodingSwarm

# Use a pre-built domain swarm
coding_swarm = CodingSwarm()

result = await coding_swarm.execute(
    "Create a Python script that scrapes HackerNews top stories"
)

print(f"Code generated:\n{result.output}")
```

### Swarm with Learning

```python
from Jotty.core.orchestration import Orchestrator

# Create swarm
swarm = Orchestrator(agents="Research team")

# Run multiple tasks - swarm learns which agents work best
tasks = [
    "Research AI safety",
    "Research quantum computing",
    "Research blockchain",
]

for task in tasks:
    result = await swarm.run(goal=task)
    print(f"{task}: {result.success}")

# Learning data is automatically persisted to:
# ~/jotty/intelligence/research_team.json

# Next run will be smarter - agents are routed based on past performance!
```

---

## Common Patterns

### Pattern 1: Research and Summarize

```python
from Jotty.core.swarms import ResearchSwarm

swarm = ResearchSwarm()
result = await swarm.execute(
    "Research the top 10 AI startups that raised funding in 2025"
)
```

### Pattern 2: Generate Educational Content

```python
from Jotty.core.swarms.olympiad_learning_swarm import learn_topic

# Generate comprehensive learning material
result = await learn_topic(
    subject="mathematics",
    topic="Calculus for 10th Grade",
    student_name="Student",
    depth="standard",
    target="foundation"
)
# Generates: PDF + HTML with concepts, examples, problems
```

### Pattern 3: Code Generation with Tests

```python
from Jotty.core.swarms import CodingSwarm, TestingSwarm

# Generate code
coding = CodingSwarm()
code_result = await coding.execute(
    "Create a binary search tree implementation in Python"
)

# Generate tests
testing = TestingSwarm()
test_result = await testing.execute(
    f"Write comprehensive tests for:\n{code_result.output}"
)
```

### Pattern 4: Use Memory for Context

```python
from Jotty.core.memory import get_memory_system
from Jotty.core.agents import AutoAgent

memory = get_memory_system()

# Store important facts
memory.store("Budget limit is $50,000", level="critical", goal="planning")
memory.store("Deadline is June 1, 2026", level="critical", goal="planning")

# Agent automatically accesses relevant memories
agent = AutoAgent(name="ProjectManager", goal="Plan project")
result = await agent.execute("Create a timeline within budget")
# Agent will use the stored budget and deadline constraints
```

### Pattern 5: Custom Skill Creation

```python
# Create skills/my-skill/skill.yaml
name: my-skill
description: "Custom skill for domain-specific task"
tools:
  - my_custom_tool

# Create skills/my-skill/tools.py
def my_custom_tool(params: dict) -> dict:
    """Your custom logic here."""
    return {"result": "success", "data": params}

# Use your skill
from Jotty.core.registry import get_unified_registry

registry = get_unified_registry()
tools = registry.get_claude_tools(['my-skill'])
```

---

## Architecture Overview

Understanding Jotty's architecture helps you use it effectively.

### The Five Layers

```
1. INTERFACE    Telegram | Slack | Discord | Web | CLI | SDK
       ↓
2. MODES        Chat | Workflow | Swarm
       ↓
3. REGISTRY     Skills (164) + UI Components (16) + Memory
       ↓
4. BRAIN        Swarms → Agents → Intelligence → Learning
       ↓
5. PERSISTENCE  ~/jotty/intelligence/*.json
```

### Key Subsystems

**1. Memory System (5-Level Brain-Inspired)**
- **Episodic**: Recent experiences (fast decay, 3 days)
- **Semantic**: General knowledge (medium decay, 7 days)
- **Procedural**: How-to knowledge (medium decay, 7 days)
- **Meta**: Learning about learning (no decay)
- **Causal**: Deep understanding (no decay)

**2. Learning System (Reinforcement Learning)**
- **TD-Lambda**: Temporal difference learning with eligibility traces
- **Shapley Values**: Game theory-based credit assignment
- **Adaptive Learning Rate**: Dynamic α adjustment based on TD error

**3. Swarm Intelligence**
- **Smart Routing**: RL-based agent selection
- **Coalition Formation**: Multi-agent collaboration
- **Trust Tracking**: Agent performance history

**4. Context Management**
- **Token Budget Allocation**: Priority-based context fitting
- **Compression**: LLM-based summarization
- **Preservation**: Critical info never lost

**5. Skills Registry (164 Skills)**
- **Discovery**: Find skills for any task
- **Plugin System**: Extensible via entry points
- **Tool Conversion**: Auto-convert to Claude/OpenAI format

### Entry Points

| Entry Point | Use Case |
|-------------|----------|
| `Jotty.core.memory.get_memory_system()` | Access 5-level memory |
| `Jotty.core.learning.get_td_lambda()` | TD-Lambda learning |
| `Jotty.core.skills.get_registry()` | 164 skills |
| `Jotty.core.orchestration.Orchestrator` | Multi-agent swarms |
| `Jotty.core.agents.AutoAgent` | Workflow automation |
| `Jotty.core.agents.ChatAssistant` | Conversational AI |

---

## Troubleshooting

### Common Issues

**Issue: `ImportError: No module named 'dspy'`**
```bash
pip install dspy-ai
```

**Issue: `APIError: Invalid API key`**
```bash
# Check your .env file or environment variables
export ANTHROPIC_API_KEY="your-actual-key"
```

**Issue: `MemoryError: Out of memory`**
```python
# Reduce context window size
from Jotty.core.context import get_context_manager
ctx = get_context_manager(max_tokens=16000)  # Default is 28000
```

**Issue: `Swarm runs slowly`**
```python
# Use cheaper model tier for simple tasks
from Jotty.core.execution import ExecutionConfig, ExecutionTier

config = ExecutionConfig(tier=ExecutionTier.DIRECT)  # Faster, cheaper
result = await swarm.run(goal="Simple task", config=config)
```

**Issue: `Agent forgets previous context`**
```python
# Use memory system to preserve important info
from Jotty.core.memory import get_memory_system

memory = get_memory_system()
memory.store(
    content="Important context",
    level="semantic",  # Persists longer than episodic
    goal="your-goal"
)
```

### Getting Help

- **Documentation**: `docs/JOTTY_ARCHITECTURE.md`
- **Issues**: https://github.com/yourusername/Jotty/issues
- **CLI Help**: `python -m Jotty.cli --help`
- **Discovery API**:
  ```python
  from Jotty import capabilities
  print(capabilities())  # See everything Jotty can do
  ```

---

## Next Steps

Now that you've got the basics, explore:

1. **📚 Full Architecture**: Read `docs/JOTTY_ARCHITECTURE.md`
2. **🔧 API Reference**: Check subsystem facades documentation
3. **📖 Examples**: See `examples/` directory (coming soon)
4. **🎓 Advanced Topics**:
   - Custom swarm creation
   - RL-based agent routing
   - Memory consolidation strategies
   - Coalition formation algorithms

---

## Quick Reference Card

```python
# Discovery
from Jotty import capabilities
caps = capabilities()

# Memory
from Jotty.core.memory import get_memory_system
mem = get_memory_system()
mem.store(content, level="episodic", goal="research")
results = mem.retrieve(query, goal="research", top_k=5)

# Learning
from Jotty.core.learning import get_td_lambda
td = get_td_lambda()
td.update(state, action, reward, next_state)

# Skills
from Jotty.core.skills import get_registry
registry = get_registry()
skills = registry.list_skills()  # 164 skills

# Swarms
from Jotty.core.orchestration import Orchestrator
swarm = Orchestrator(agents="describe your team")
result = await swarm.run(goal="your goal")

# Agents
from Jotty.core.agents import AutoAgent
agent = AutoAgent(name="Agent", goal="goal")
result = await agent.execute("task")
```

---

**Welcome to the future of multi-agent AI!** 🚀

For questions or contributions, see `CONTRIBUTING.md` or open an issue on GitHub.
