# Benchmark Setup Guide

**Date**: January 27, 2026

---

## Quick Start

### Option 1: Simple Mock Agent (Fast, No Setup)

```bash
cd /var/www/sites/personal/stock_market/Jotty
python examples/quick_benchmark_test.py
```

This uses a simple mock agent that calls LLM directly. Good for quick testing.

### Option 2: Full Jotty Agent (Requires Setup)

For full Jotty agent benchmarking, you need to configure orchestrators properly.

---

## Setting Up Full Jotty Agents

### Single Agent Setup

```python
import dspy
from core.orchestration.single_agent_orchestrator import SingleAgentOrchestrator
from core.foundation.data_structures import SwarmConfig

# Configure DSPy
dspy.configure(lm=dspy.LM("anthropic/claude-sonnet-4", api_key="..."))

# Create agent
class SimpleAgent(dspy.Module):
    def forward(self, question):
        return dspy.Predict("question -> answer")(question=question)

agent = SimpleAgent()

# Create orchestrator
orchestrator = SingleAgentOrchestrator(
    agent=agent,
    architect_prompts=["prompts/planning.md"],  # You need these
    auditor_prompts=["prompts/validation.md"],   # You need these
    config=SwarmConfig(random_seed=42)
)

# Use in benchmark
from examples.benchmark_test import JottyBenchmarkWrapper

wrapper = JottyBenchmarkWrapper(
    orchestrator=orchestrator,
    use_multi_agent=False
)
```

### Multi-Agent Setup

```python
from core.orchestration.conductor import Conductor
from core.foundation.data_structures import SwarmConfig
from examples.benchmark_test import JottyBenchmarkWrapper

# Create config
config = SwarmConfig(random_seed=42)

# Create conductor (requires proper setup)
conductor = Conductor(config=config)

# Register agents
# conductor.register_actor("Agent1", agent1)
# conductor.register_actor("Agent2", agent2)

# Use in benchmark
wrapper = JottyBenchmarkWrapper(
    orchestrator=conductor,
    use_multi_agent=True
)
```

---

## Current Status

### ✅ Working: Mock Agent

The `quick_benchmark_test.py` uses a simple mock agent that:
- Calls LLM directly via UnifiedLLM
- Works without full orchestrator setup
- Good for quick testing

### ⚠️ Requires Setup: Full Agents

For full Jotty agent benchmarking:
1. Configure DSPy
2. Create agents
3. Set up architect/auditor prompts
4. Initialize orchestrators
5. Pass to benchmark wrapper

---

## Recommended Approach

### For Quick Testing

Use the mock agent approach:

```python
from examples.benchmark_test import JottyBenchmarkWrapper
from core.foundation.data_structures import SwarmConfig

config = SwarmConfig(random_seed=42)
wrapper = JottyBenchmarkWrapper(config=config)  # Uses mock by default
```

### For Real Evaluation

Set up proper orchestrators and pass them:

```python
# Create orchestrator with proper setup
orchestrator = SingleAgentOrchestrator(...)

# Pass to wrapper
wrapper = JottyBenchmarkWrapper(orchestrator=orchestrator)
```

---

## Next Steps

1. ✅ **Use mock agent** for quick testing
2. ⚠️ **Set up proper agents** for real evaluation
3. ✅ **Run benchmarks** with evaluation framework
4. ✅ **Track results** with cost tracking

---

**Last Updated**: January 27, 2026
