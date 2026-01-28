# Using Jotty's RL and Memory Properly

**Date**: January 27, 2026  
**Status**: 📋 **GUIDE** - How to Use RL and Memory

---

## The Issue

We tested multi-agent quality but **didn't use Jotty's RL and Memory systems**!

We were calling `UnifiedLLM` directly instead of using `Conductor` with RL/Memory enabled.

---

## Correct Way to Use Jotty

### 1. Use Conductor (Not Direct LLM Calls)

**❌ Wrong** (what we did):
```python
llm = UnifiedLLM(default_provider="anthropic")
response = llm.generate(prompt)
```

**✅ Correct** (use Conductor):
```python
from core.orchestration import create_conductor
from core.foundation.data_structures import SwarmConfig, AgentConfig

config = SwarmConfig(
    enable_rl=True,  # Enable RL learning
    persist_memories=True,  # Enable memory
)

actors = [
    AgentConfig(name="planner", goal="..."),
    AgentConfig(name="executor", goal="..."),
    AgentConfig(name="reviewer", goal="..."),
]

conductor = create_conductor(actors, config)
result = await conductor.run(task)
```

---

## SwarmConfig Parameters

### RL Settings

```python
config = SwarmConfig(
    enable_rl=True,  # Master switch for RL
    gamma=0.99,  # Discount factor
    lambda_trace=0.95,  # TD(λ) trace
    alpha=0.01,  # Learning rate
    enable_adaptive_alpha=True,  # Adaptive learning rate
    enable_intermediate_rewards=True,  # Reward for intermediate steps
)
```

### Memory Settings

```python
config = SwarmConfig(
    persist_memories=True,  # Enable memory persistence
    episodic_capacity=1000,  # Episodic memory size
    semantic_capacity=500,  # Semantic memory size
    procedural_capacity=200,  # Procedural memory size
    meta_capacity=100,  # Meta memory size
    causal_capacity=150,  # Causal memory size
)
```

---

## Why We Got 4.00/5 Instead of 5/5

### What We Did Wrong

1. **❌ Didn't use Conductor**
   - Called `UnifiedLLM` directly
   - No RL learning
   - No memory

2. **❌ No feedback loop**
   - Evaluation scores not fed back to RL
   - No learning from gaps

3. **❌ No memory of previous runs**
   - Each run started fresh
   - No accumulated knowledge

### What We Should Do

1. **✅ Use Conductor with RL enabled**
   ```python
   config = SwarmConfig(enable_rl=True)
   conductor = create_conductor(actors, config)
   ```

2. **✅ Feed evaluation scores to RL**
   ```python
   score = evaluate(result)
   # Conductor's RL learns from this automatically
   ```

3. **✅ Use memory to remember gaps**
   ```python
   # Conductor's memory stores episodes automatically
   # Next run retrieves similar experiences
   ```

---

## How RL and Memory Help

### RL Learning

**What it does**:
- Learns which agent sequences work best
- Learns which prompts get higher scores
- Learns optimal coordination patterns

**How to use**:
```python
config = SwarmConfig(
    enable_rl=True,  # Enable RL
    enable_intermediate_rewards=True,  # Reward intermediate steps
)

# Conductor automatically:
# 1. Tracks agent contributions
# 2. Updates Q-values based on outcomes
# 3. Learns optimal sequences
```

### Memory System

**What it does**:
- Stores successful patterns
- Remembers previous gaps
- Retrieves similar experiences

**How to use**:
```python
config = SwarmConfig(
    persist_memories=True,  # Enable memory
    episodic_capacity=1000,  # Store episodes
)

# Conductor automatically:
# 1. Stores episodes in memory
# 2. Retrieves similar experiences
# 3. Uses memory in prompts
```

---

## Expected Improvement

### Without RL/Memory (Current)

- Score: 4.00/5
- No learning
- No memory
- Repeats same patterns

### With RL/Memory (Expected)

- Score: 4.5-5.0/5
- Learns from scores
- Remembers gaps
- Improves over time

---

## Next Steps

1. **✅ Use Conductor properly**
   - Create Conductor with RL/Memory enabled
   - Use proper AgentConfig

2. **✅ Feed evaluation feedback**
   - Use evaluation scores as rewards
   - Let RL learn from outcomes

3. **✅ Enable memory**
   - Store episodes automatically
   - Retrieve similar experiences
   - Use in prompts

---

**Last Updated**: January 27, 2026  
**Status**: 📋 **GUIDE** - Ready to Test with Proper Configuration
