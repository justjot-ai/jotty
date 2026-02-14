# Jotty Examples

Runnable examples demonstrating all Jotty subsystems.

## Directory Structure

```
examples/
├── memory/          - 5-level memory system examples
├── learning/        - TD-Lambda and RL examples
├── context/         - Token budget management examples
├── skills/          - Custom skill creation
├── orchestration/   - Swarm composition examples
└── README.md        - This file
```

## Quick Start

Each example is standalone and can be run directly:

```bash
# Memory example
python examples/memory/01_basic_storage.py

# Learning example
python examples/learning/01_td_lambda_training.py

# Context example
python examples/context/01_budget_allocation.py

# Skills example
python examples/skills/01_custom_skill.py

# Orchestration example
python examples/orchestration/01_basic_swarm.py
```

## Examples by Subsystem

### Memory System
- `01_basic_storage.py` - Store and retrieve memories
- `02_multi_level_memory.py` - Use all 5 memory levels
- `03_context_aware_retrieval.py` - Context-based memory retrieval

### Learning System
- `01_td_lambda_training.py` - Basic TD-Lambda learning
- `02_credit_assignment.py` - Shapley value credit assignment
- `03_adaptive_learning_rate.py` - Dynamic α adjustment

### Context Management
- `01_budget_allocation.py` - Priority-based token budgeting
- `02_compression.py` - LLM-based context compression
- `03_critical_preservation.py` - Preserve critical info

### Skills
- `01_custom_skill.py` - Create a custom skill
- `02_skill_discovery.py` - Discover skills for tasks
- `03_tool_conversion.py` - Convert skills to Claude/OpenAI format

### Orchestration
- `01_basic_swarm.py` - Simple swarm creation
- `02_learning_swarm.py` - Swarm with RL-based routing
- `03_coalition_formation.py` - Multi-agent collaboration

## Requirements

All examples require:
- Python 3.10+
- Jotty installed (see main README)
- API keys set (ANTHROPIC_API_KEY or OPENAI_API_KEY)

## Contributing

To add an example:
1. Follow the naming convention: `NN_descriptive_name.py`
2. Include docstring explaining what the example demonstrates
3. Make it runnable standalone
4. Add to this README
